"""Persist completed LangGraph tool calls to the existing BigQuery audit table."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from pro.persistence import get_persistence_manager


def _field(value: Any, name: str, default: Any = None) -> Any:
    return (
        value.get(name, default)
        if isinstance(value, Mapping)
        else getattr(value, name, default)
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value:
            header, encoded = value.split(",", 1)
            return {
                "redacted_data_url": True,
                "media_type": header[5:].split(";", 1)[0],
                "encoded_length": len(encoded),
            }
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_json_safe(item) for item in value]
    dump = getattr(value, "model_dump", None)
    return _json_safe(dump(mode="json")) if callable(dump) else str(value)


def tool_usage_rows(
    messages: Sequence[Any], *, run_id: str, deployment_id: str | None = None
) -> list[dict[str, Any]]:
    calls: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    observations: dict[str, Any] = {}
    for message in messages:
        agent_name = _field(message, "name")
        for call in _field(message, "tool_calls", []) or []:
            call_id = str(_field(call, "id", "")).strip()
            tool_name = str(_field(call, "name", "")).strip()
            if not call_id or not tool_name:
                continue
            if call_id not in calls:
                order.append(call_id)
            calls[call_id] = {
                "tool_name": tool_name,
                "tool_input": _json_safe(_field(call, "args", {}) or {}),
                "agent_name": str(agent_name).strip() if agent_name else None,
            }
        tool_call_id = str(_field(message, "tool_call_id", "")).strip()
        if tool_call_id:
            observations[tool_call_id] = message
    timestamp = datetime.now(UTC).isoformat()
    rows = []
    for call_id in order:
        call = calls[call_id]
        observation = observations.get(call_id)
        content = _field(observation, "content") if observation is not None else None
        success = (
            observation is not None
            and str(_field(observation, "status", "success")).lower() != "error"
        )
        rows.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:{call_id}")),
                "timestamp": timestamp,
                "deployment_id": deployment_id,
                "agent_name": call["agent_name"],
                "run_id": run_id,
                "tool_name": call["tool_name"],
                "tool_success_bool": success,
                "execution_time_ms": None,
                "error_message": None
                if success
                else (
                    str(content)
                    if content is not None
                    else "missing tool observation"
                ),
                "tool_input": call["tool_input"],
                "tool_output": _json_safe(content),
            }
        )
    return rows


async def persist_run_tool_usage(
    messages: Sequence[Any], *, run_id: str, persistence: Any = None
) -> int:
    manager = persistence or get_persistence_manager()
    rows = tool_usage_rows(
        messages, run_id=run_id, deployment_id=getattr(manager, "deployment_id", None)
    )
    if not rows:
        return 0
    result = manager.insert_rows(
        manager.central_tool_usage_table, rows, [row["id"] for row in rows]
    )
    if inspect.isawaitable(result):
        await result
    elif hasattr(result, "result"):
        await asyncio.wrap_future(result)
    return len(rows)
