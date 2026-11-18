"""Invocation-local, content-free activity for the existing MIT run transport."""
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
MAX_ACTIVE = 16
MAX_QUEUED = 128


def canonical_uuid(value: Any) -> str:
    if not isinstance(value, (str, uuid.UUID)):
        raise ValueError("Expected a run identity")
    return str(uuid.UUID(str(value)))


class RunProgress(BaseCallbackHandler):
    """Observe actual callbacks, never returned/checkpointed message history.

    LangChain may invoke the sync handler from several tool executor threads.
    The handler and its bounded active map belong to one request only. Neither
    tool data nor model messages are retained, formatted or sent to the sink.
    """

    def __init__(self, thread_id: str, client_message_id: str,
                 emit: Callable[[dict], None]):
        self.thread_id = canonical_uuid(thread_id)
        self.client_message_id = canonical_uuid(client_message_id)
        self._emit = emit
        self._lock = threading.RLock()
        self._active: dict[str, tuple[str, str, float]] = {}
        self._run_id: str | None = None
        self._sequence = 0
        self._closed = False

    def bind_run(self, run_id: str) -> None:
        with self._lock:
            if self._run_id is not None:
                raise ValueError("Run activity is already bound")
            self._run_id = canonical_uuid(run_id)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._active.clear()

    def _event(self, task_id: str, phase: str, state: str,
               elapsed_ms: int | None = None) -> None:
        self._sequence += 1
        event = {"type": "activity", "thread_id": self.thread_id,
                 "client_message_id": self.client_message_id,
                 "run_id": self._run_id, "sequence": self._sequence,
                 "task_id": task_id, "phase": phase, "state": state}
        if elapsed_ms is not None:
            event["elapsed_ms"] = max(0, elapsed_ms)
        self._emit(event)

    def _start(self, callback_id: Any, phase: str) -> None:
        key = canonical_uuid(callback_id)
        with self._lock:
            if (self._closed or self._run_id is None or key in self._active
                    or len(self._active) >= MAX_ACTIVE):
                return
            task_id = str(uuid.uuid4())
            self._active[key] = (task_id, phase, time.monotonic())
            self._event(task_id, phase, "started")

    def _end(self, callback_id: Any, failed: bool = False) -> None:
        key = canonical_uuid(callback_id)
        with self._lock:
            if self._closed:
                return
            active = self._active.pop(key, None)
            if active is None:
                return
            task_id, phase, started = active
            self._event(task_id, phase, "failed" if failed else "finished",
                        int((time.monotonic() - started) * 1000))

    def on_tool_start(self, serialized: dict | None, input_str: str, *,
                      run_id: uuid.UUID, **kwargs: Any) -> None:
        name = serialized.get("name") if isinstance(serialized, dict) else None
        self._start(run_id, TOOL_PHASES.get(name, "step") if isinstance(name, str) else "step")

    def on_tool_end(self, output: Any, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        # A handled tool error is a ToolMessage, not necessarily an exception.
        # Its fixed status is sufficient; its content is deliberately ignored.
        self._end(run_id, getattr(output, "status", None) == "error")

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
        self._start(run_id, "preparing")

    def on_llm_start(self, serialized: dict, prompts: list, *,
                     run_id: uuid.UUID, **kwargs: Any) -> None:
        self._start(run_id, "preparing")

    def on_llm_end(self, response: Any, *, run_id: uuid.UUID,
                   **kwargs: Any) -> None:
        self._end(run_id)

    def on_llm_error(self, error: BaseException, *, run_id: uuid.UUID,
                     **kwargs: Any) -> None:
        self._end(run_id, True)


async def stream_run(invoke: Callable, thread_id: str, client_message_id: str):
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

    progress = RunProgress(thread_id, client_message_id, emit)

    async def produce():
        try:
            result = await invoke(progress)
            event = {"type": "result", "result": result}
        except Exception as error:
            # The existing handler already logs failures. Never relay its raw
            # detail/exception (which may contain private data) as activity.
            code = getattr(error, "status_code", 500)
            event = {"type": "run_error", "status_code": code if type(code) is int else 500}
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
