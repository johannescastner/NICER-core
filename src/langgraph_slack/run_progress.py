"""Invocation-local activity with negotiated, bounded execution details."""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from typing import Any, Callable

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.errors import GraphBubbleUp
from fastapi.encoders import jsonable_encoder
from pro.superset.lumi_copilot.shared_civic import run_activity as activity_contract


PHASES = frozenset({"preparing", "catalogue", "data", "sources", "visual",
                    "step", "response", "canonical_grounding"})
TOOL_PHASES = {
    "list_datasets_and_tables": "catalogue",
    "sql_database_query": "data", "query_profile": "data",
    "civic_evidence": "sources",
    "superset_tool": "visual",
    "get_node_context": "canonical_grounding",
    "wikidata_describe": "canonical_grounding",
}
MAX_ACTIVE = 32
MAX_QUEUED = 128
MAX_ANCESTRY = 8192
# Explicit chart input fields. Never retain injected clients, credentials,
# returned data/sample rows or arbitrary arguments from other tools.
CHART_FIELDS = frozenset('chart_title schema_type chart_type chart_config_json x y y_columns color_by orientation stacked log_x log_y column group_by distribution_dim nbins labels values parent metric_column metric_aggregate time_column compare_column max_val table_columns metric_columns heatmap_value region entity value lat lon end_lat end_lon path_column size language depth source target flow_value date open high low close volume y2 z group_column ci_lower ci_upper template dashboard_title dataset_name_hint x_axis_label y_axis_label series_labels_json expected_date_range_json max_series unit_code unit_qid indicator_qid entity_qid time_unit_qid width height scale'.split())


def canonical_uuid(value: Any) -> str:
    if not isinstance(value, (str, uuid.UUID)):
        raise ValueError("Expected a run identity")
    return str(uuid.UUID(str(value)))


class RunProgress(BaseCallbackHandler):
    """Observe actual callbacks, never returned/checkpointed message history.

    LangChain may invoke the sync handler from several tool executor threads.
    The handler and its bounded active map belong to one request only. Prompts, hidden reasoning and binary artifacts never enter progress. A
    migrated Native consumer may opt in to bounded model-visible call/results.
    """

    def __init__(self, thread_id: str, client_message_id: str,
                 emit: Callable[[dict], None], *, protocol: int = 1, actor_id: int | None = None,
                 native_run_id: str | None = None, include_parent_ids: bool = False,
                 include_sql_queries: bool = False, include_graph_scopes: bool = False, include_step_details: bool = False):
        self.thread_id = canonical_uuid(thread_id)
        self.client_message_id = canonical_uuid(client_message_id)
        self._emit = emit
        self._lock = threading.RLock()
        self._active: dict[str, tuple[str, str, float, str, str, dict]] = {}
        self._run_id: str | None = None
        self._sequence = 0
        self._closed = False
        self._include_parent_ids = protocol == 2 and include_parent_ids
        self._include_sql_queries = protocol == 2 and include_sql_queries
        self._include_graph_scopes = protocol == 2 and include_parent_ids and include_graph_scopes
        self._include_step_details = protocol == 2 and include_parent_ids and include_step_details
        self._parents: dict[str, str | None] = {}
        self._callback_tasks: dict[str, str] = {}
        self._tool_callbacks: set[str] = set()
        self._tool_origins: dict[tuple[str | None, str, str], str | None] = {}
        self._tool_arguments: dict[tuple[str | None, str, str], dict] = {}
        self._task_parents: dict[str, str] = {}
        self.protocol = protocol
        self._trace_bytes = 1024
        self._trace_stopped = False
        self._trace = (activity_contract.empty(actor_id, self.thread_id, self.client_message_id,
            canonical_uuid(native_run_id)) if protocol == 2 else None)

    def bind_run(self, run_id: str) -> None:
        with self._lock:
            if self._run_id is not None:
                raise ValueError("Run activity is already bound")
            self._run_id = canonical_uuid(run_id)
            if self._trace is not None:self._trace['agent_run_id'] = self._run_id

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._active.clear()
            self._parents.clear()
            self._callback_tasks.clear()
            self._tool_callbacks.clear()
            self._tool_origins.clear()
            self._tool_arguments.clear()
            self._task_parents.clear()

    def _event(self, task_id: str, phase: str, state: str,
               elapsed_ms: int | None = None, *, action: str = 'model_response', kind: str = 'model', query: dict | None = None) -> None:
        if self._sequence >= 1000000:
            if self._trace is not None:self._trace['truncated']=True
            return
        self._sequence += 1
        event = {"type": "activity", "thread_id": self.thread_id,
                 "client_message_id": self.client_message_id,
                 "run_id": self._run_id, "sequence": self._sequence,
                 "task_id": task_id, "phase": phase, "state": state}
        if elapsed_ms is not None:
            event["elapsed_ms"] = max(0, elapsed_ms)
        if self.protocol == 2:
            event.update(protocol=2, action=action, kind=kind)
            if query:event.update(query)
            if task_id in self._task_parents:
                event['parent_task_id'] = self._task_parents[task_id]
            compact = {k:event[k] for k in activity_contract.EVENT_FIELDS if k in event}
            self._trace['observed_count'] = self._sequence
            if not self._trace_stopped:
                size = len(activity_contract.canonical(compact).encode()) + 1
                if len(self._trace['events']) >= activity_contract.MAX_EVENTS or self._trace_bytes + size > activity_contract.MAX_BYTES:
                    self._trace['truncated'] = True
                    self._trace_stopped = True
                else:
                    self._trace['events'].append(compact);self._trace_bytes += size
        self._emit(event)

    def _remember_parent(self, callback_id: Any, parent_id: Any) -> None:
        if not self._include_parent_ids or self._closed:return
        key = canonical_uuid(callback_id)
        parent = canonical_uuid(parent_id) if parent_id is not None else None
        if key in self._parents:
            if self._parents[key] != parent:self._trace['truncated'] = True
            return
        if len(self._parents) >= MAX_ANCESTRY:
            self._trace['truncated'] = True
            return
        self._parents[key] = parent

    def on_chain_start(self, serialized: dict | None, inputs: Any, *,
                       run_id: uuid.UUID, parent_run_id: uuid.UUID | None = None,
                       **kwargs: Any) -> None:
        # Native callback ancestry crosses Pregel/ToolNode scopes. Retain only
        # their identities, never graph inputs, prompts or private reasoning.
        with self._lock:
            self._remember_parent(run_id, parent_run_id)
            name, tags = kwargs.get('name'), kwargs.get('tags') or []
            # LangGraph emits both the parent node wrapper (graph:step) and
            # the invoked compiled graph (seq:step). Only the latter is a scope.
            wrapper = any(str(tag).startswith('graph:step:') for tag in tags) and not any(str(tag).startswith('seq:step:') for tag in tags)
            if self._include_graph_scopes and name in activity_contract.GRAPH_ACTIONS and not wrapper:
                self._start(run_id, 'step', action=name, kind='graph', parent_run_id=parent_run_id)

    def on_chain_end(self, outputs: Any, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        self._end(run_id)

    def on_chain_error(self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        if isinstance(error, GraphBubbleUp):
            with self._lock:self._active.pop(canonical_uuid(run_id), None)
            return
        self._end(run_id, True)

    def _enclosing_tool(self, callback_id: str) -> str | None:
        parent = self._parents.get(callback_id)
        seen = {callback_id}
        while parent is not None and parent not in seen:
            if parent in self._tool_callbacks:
                return self._callback_tasks.get(parent)
            seen.add(parent)
            parent = self._parents.get(parent)
        return None

    def _start(self, callback_id: Any, phase: str, *, action: str = 'model_response', kind: str = 'model', parent_run_id: Any = None, query: dict | None = None, triggered_by: str | None = None) -> None:
        key = canonical_uuid(callback_id)
        with self._lock:
            if (self._closed or self._run_id is None or key in self._active
                    or len(self._active) >= MAX_ACTIVE):
                if self._trace is not None and len(self._active) >= MAX_ACTIVE:self._trace['truncated'] = True
                return
            task_id = str(uuid.uuid4())
            if self._include_parent_ids:
                self._remember_parent(key, parent_run_id)
                parent = self._parents.get(key)
                seen = {key}
                while parent is not None and parent not in seen:
                    if parent in self._callback_tasks:
                        if len(self._task_parents) < activity_contract.MAX_EVENTS:
                            self._task_parents[task_id] = self._callback_tasks[parent]
                        else:self._trace['truncated'] = True
                        break
                    seen.add(parent)
                    parent = self._parents.get(parent)
                if parent in seen:self._trace['truncated'] = True
                # ToolNode is a sibling of the model node in LangGraph's runtime
                # tree. Its actual tool-call ID still proves which model turn
                # dispatched it. Keep that real causal hierarchy in the UI.
                if triggered_by is not None:self._task_parents[task_id] = triggered_by
                if len(self._callback_tasks) < activity_contract.MAX_EVENTS:
                    self._callback_tasks[key] = task_id
                    if kind == 'tool':self._tool_callbacks.add(key)
                else:self._trace['truncated'] = True
            self._active[key] = (task_id, phase, time.monotonic(), action, kind, query or {})
            self._event(task_id, phase, "started", action=action, kind=kind, query=query)

    @staticmethod
    def _inspection_value(name: str, value: Any) -> dict:
        # Only model-visible JSON/text. Never stringify a client, checkpoint,
        # exception, binary artifact or private reasoning channel.
        try:
            if isinstance(value,list) and any(isinstance(item,dict) and 'type' in item for item in value):
                value='\n'.join(item['text'] for item in value if isinstance(item,dict) and item.get('type') in ('text','output_text') and isinstance(item.get('text'),str))
            if not isinstance(value,(str,dict,list,int,float,bool)) and value is not None:return {}
            text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True,indent=2)
            if not text.strip() or '\x00' in text:return {}
            raw=text.encode('utf-8');preview=raw[:activity_contract.MAX_SQL_BYTES].decode('utf-8',errors='ignore')
            return {name:preview,name+'_truncated':len(raw)>activity_contract.MAX_SQL_BYTES} if preview.strip() else {}
        except (ValueError,TypeError,UnicodeError):return {}

    def _end(self, callback_id: Any, failed: bool = False, *, returned: dict | None = None) -> None:
        key = canonical_uuid(callback_id)
        with self._lock:
            if self._closed:
                return
            active = self._active.pop(key, None)
            if active is None:
                return
            task_id, phase, started, action, kind, query = active
            self._event(task_id, phase, "failed" if failed else "finished",
                        int((time.monotonic() - started) * 1000), action=action, kind=kind, query={**query,**(returned or {})})

    def on_tool_start(self, serialized: dict | None, input_str: str, *,
                      run_id: uuid.UUID, **kwargs: Any) -> None:
        name = serialized.get("name") if isinstance(serialized, dict) else None
        # This identifier comes from the registered BaseTool's maintained
        # callback, never tool-call text, arguments or returned content.
        action = name if isinstance(name,str) and activity_contract.ACTION.fullmatch(name) else 'unidentified_tool'
        if self._trace is not None and action == 'unidentified_tool':
            self._trace['truncated'] = True
            return
        query = {}
        if self._include_sql_queries and action in activity_contract.QUERY_ACTIONS:
            inputs = kwargs.get('inputs')
            if inputs is None and isinstance(input_str,str) and len(input_str)<=200000:
                try:inputs=json.loads(input_str)
                except (ValueError,TypeError):
                    # BaseTool.invoke(str) passes the exact sole argument here
                    # and inputs=None. It is not a JSON-encoded argument dict.
                    # This fallback belongs only to the registered query tool;
                    # other raw tool input must not become SQL or configuration.
                    if action == 'sql_database_query':inputs={'query':input_str}
            sql = inputs.get('query' if action == 'sql_database_query' else 'sql') if isinstance(inputs,dict) else None
            if isinstance(sql,str) and sql.strip() and '\x00' not in sql:
                try:
                    raw=sql.encode('utf-8')
                    preview=raw[:activity_contract.MAX_SQL_BYTES].decode('utf-8',errors='ignore')
                    if preview.strip():query={'sql_query':preview,'sql_query_truncated':len(raw)>activity_contract.MAX_SQL_BYTES}
                except UnicodeError:pass
            if action in activity_contract.CHART_ACTIONS and isinstance(inputs,dict):
                configuration={key:inputs[key] for key in sorted(CHART_FIELDS & inputs.keys())}
                if configuration:
                    try:
                        raw=json.dumps(configuration,ensure_ascii=False,sort_keys=True,allow_nan=False,indent=2).encode('utf-8')
                        if b'\x00' not in raw:
                            query.update(chart_configuration=raw[:activity_contract.MAX_SQL_BYTES].decode('utf-8',errors='ignore'),
                                chart_configuration_truncated=len(raw)>activity_contract.MAX_SQL_BYTES)
                    except (UnicodeError,ValueError,TypeError):pass
        with self._lock:
            triggered_by = None
            if self._include_parent_ids:
                key = canonical_uuid(run_id)
                self._remember_parent(key, kwargs.get('parent_run_id'))
                call_id = kwargs.get('tool_call_id')
                if isinstance(call_id, str):
                    origin=(self._enclosing_tool(key), call_id, action)
                    triggered_by = self._tool_origins.get(origin)
                    if self._include_step_details and triggered_by is not None:query.update(self._tool_arguments.get(origin,{}))
            self._start(run_id, TOOL_PHASES.get(name, "step") if isinstance(name, str) else "step", action=action, kind='tool', parent_run_id=kwargs.get('parent_run_id'),query=query,triggered_by=triggered_by)

    def on_tool_end(self, output: Any, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        # A handled tool error is a ToolMessage, not necessarily an exception.
        # Its fixed status is sufficient; its content is deliberately ignored.
        failed=getattr(output, "status", None) == "error"
        returned={}
        if self._include_step_details:
            content=getattr(output,'content',output)
            returned=self._inspection_value('returned_value',content)
        self._end(run_id, failed, returned=returned)

    def on_tool_error(self, error: BaseException, *, run_id: uuid.UUID,
                      **kwargs: Any) -> None:
        if isinstance(error, GraphBubbleUp):
            # Human interrupts/handoffs are maintained graph control flow,
            # not failed data checks. The terminal interrupted response owns
            # the clarification UI; never expose its private payload here.
            with self._lock:
                self._active.pop(canonical_uuid(run_id), None)
            return
        self._end(run_id, True)

    def on_chat_model_start(self, serialized: dict, messages: list, *,
                            run_id: uuid.UUID, **kwargs: Any) -> None:
        # This registered wrapper performs the existing chart-image review.
        # LangChain supplies its class identity; never infer the stage from a
        # prompt, model response, tool argument or provider-supplied name.
        review = isinstance(serialized, dict) and serialized.get('id') == [
            'pro', 'ml_inference', 'qwen_vl_chat', 'ChatQwen25VL']
        self._start(run_id, "visual" if review else "preparing",
                    parent_run_id=kwargs.get('parent_run_id'))

    def on_llm_start(self, serialized: dict, prompts: list, *,
                     run_id: uuid.UUID, **kwargs: Any) -> None:
        self._start(run_id, "preparing", parent_run_id=kwargs.get('parent_run_id'))

    def on_llm_end(self, response: Any, *, run_id: uuid.UUID,
                   **kwargs: Any) -> None:
        with self._lock:
            key = canonical_uuid(run_id)
            task_id = self._callback_tasks.get(key)
            generations = getattr(response, 'generations', None)
            message = getattr(generations[0][0], 'message', None) if generations and generations[0] else None
            if self._include_parent_ids and task_id:
                scope = self._enclosing_tool(key)
                for call in getattr(message, 'tool_calls', []) or []:
                    if not isinstance(call, dict):continue
                    call_id, name = call.get('id'), call.get('name')
                    if not isinstance(call_id, str) or not 0 < len(call_id) <= 256 or not isinstance(name, str) or not activity_contract.ACTION.fullmatch(name):continue
                    origin = (scope, call_id, name)
                    if origin in self._tool_origins:
                        # Reused IDs are ambiguous. Preserve actual runtime
                        # ancestry instead of assigning the wrong model parent.
                        self._tool_origins[origin] = None
                        self._tool_arguments.pop(origin,None)
                    elif len(self._tool_origins) < MAX_ANCESTRY:
                        self._tool_origins[origin] = task_id
                        if self._include_step_details:self._tool_arguments[origin]=self._inspection_value('call_arguments',call.get('args',{}))
                    else:self._trace['truncated'] = True
            returned=self._inspection_value('returned_value',message.content) if self._include_step_details and message is not None else {}
            active = self._active.get(key)
            if self._include_step_details and active and active[1] == 'visual' and message is not None:
                # The structured reviewer returns its decision as a tool call,
                # with empty message content. Retain that actual bounded result
                # under the same callback, not a fictional executed tool step.
                reviews = [call['args'] for call in getattr(message, 'tool_calls', []) or []
                           if isinstance(call, dict) and call.get('name') == 'ChartValidation'
                           and isinstance(call.get('args'), dict)]
                if len(reviews) == 1:
                    returned = self._inspection_value('returned_value', reviews[0])
            self._end(run_id,returned=returned)

    def on_llm_error(self, error: BaseException, *, run_id: uuid.UUID,
                     **kwargs: Any) -> None:
        self._end(run_id, True)


    def snapshot(self, outcome: str):
        with self._lock:
            if self._trace is None:return None
            value = activity_contract.validate(self._trace)
            value.update(state='sealed', outcome=outcome,
                complete=not value['truncated'] and value['observed_count']==len(value['events']))
            return activity_contract.validate(value)


async def stream_run(invoke: Callable, thread_id: str, client_message_id: str, *, protocol=1, actor_id=None, native_run_id=None, include_parent_ids=False, include_sql_queries=False, include_graph_scopes=False, include_step_details=False):
    """Stream safe activity and one private service-to-service result.

    The graph task has one owner and is never reinvoked. A disconnected consumer
    does not cancel that task: its existing run deadline and checkpoint recovery
    still govern completion. No cross-request registry or durable store is used.
    """
    loop = asyncio.get_running_loop()
    output: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUED)
    accepting = True

    def enqueue(event):
        if accepting and output.qsize() < MAX_QUEUED - 1:
            output.put_nowait(event)

    def emit(event):
        loop.call_soon_threadsafe(enqueue, event)

    progress = RunProgress(thread_id, client_message_id, emit, protocol=protocol, actor_id=actor_id, native_run_id=native_run_id, include_parent_ids=include_parent_ids, include_sql_queries=include_sql_queries, include_graph_scopes=include_graph_scopes, include_step_details=include_step_details)

    async def produce():
        try:
            result = await invoke(progress)
            if protocol == 2 and isinstance(result,dict):
                result = {**result, 'activity_trace': progress.snapshot(result.get('status','success'))}
            event = {"type": "result", "result": result}
        except Exception as error:
            # The existing handler already logs failures. Never relay its raw
            # detail/exception (which may contain private data) as activity.
            code = getattr(error, "status_code", 500)
            event = {"type": "run_error", "status_code": code if type(code) is int else 500}
            if protocol == 2 and progress._run_id is not None:event['activity_trace'] = progress.snapshot('failed')
        finally:
            progress.close()
        if accepting:
            # Progress reserves one slot so terminal delivery cannot deadlock
            # behind a slow or disconnected activity consumer.
            output.put_nowait(event)

    task = asyncio.create_task(produce())
    try:
        while True:
            try:
                event = await asyncio.wait_for(output.get(), 10)
            except asyncio.TimeoutError:
                yield ": waiting\n\n"
                continue
            # Match FastAPI's ordinary JSON return path for actual LangChain
            # BaseMessages and other supported result objects. This terminal
            # frame stays private between services; Native commits/discloses it.
            yield "data: " + json.dumps(jsonable_encoder(event), separators=(",", ":")) + "\n\n"
            if event["type"] != "activity":
                break
    finally:
        accepting = False
        # Shield the single existing invocation through transport teardown;
        # cleanup is bounded by the existing GRAPH_TIMEOUT, not a new retry.
        await asyncio.shield(task)
