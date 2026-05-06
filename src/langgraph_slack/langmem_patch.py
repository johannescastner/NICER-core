"""Monkey-patch for langmem 0.0.30's ``_preprocess_messages``.

Mirrors the fix in https://github.com/langchain-ai/langmem/pull/141
(currently OPEN, not merged). When the summarization cutoff lands on a
``ToolMessage`` mid-sequence — instead of on an ``AIMessage`` — langmem's
original logic only checks whether the *last* message in the slice is an
``AIMessage(tool_calls)`` and appends matching ``ToolMessage``s for that
case. It misses the (more common, in tool-heavy traces) case where one or
more ``AIMessage(tool_calls)`` earlier in the slice are missing matching
``ToolMessage``s.

Symptom in production: DeepSeek's ``deepseek-v4-flash`` (and other strict
providers) reject the resulting summarization-input message list with
HTTP 400::

    An assistant message with 'tool_calls' must be followed by tool
    messages responding to each 'tool_call_id'.

This patch wraps the original ``_preprocess_messages`` and post-processes
its result: for every ``AIMessage(tool_calls)`` in
``messages_to_summarize``, append any matching ``ToolMessage`` from the
source conversation that's missing — exactly mirroring PR #141.

Delete this module (and the ``import`` at the bottom of patch_typing.py
that activates it) when langmem releases a version that includes PR #141
and we upgrade.

Tracking issues:
  * https://github.com/langchain-ai/langmem/pull/141 (the fix)
  * https://github.com/langchain-ai/langmem/issues/126 (parallel tool calls)
  * https://github.com/langchain-ai/langmem/issues/112 (truncation breaks pairing)
"""
from __future__ import annotations

from dataclasses import replace as _dataclass_replace
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langmem.short_term import summarization as _ls


# Preserve a reference to the original so we can delegate to it.
_original_preprocess = _ls._preprocess_messages


def _patched_preprocess_messages(
    messages: list,
    running_summary: Any,
    max_tokens: int,
    max_tokens_before_summary: int,
    max_summary_tokens: int,
    token_counter,
):
    """Wraps the original ``_preprocess_messages`` and repairs orphan
    tool-call pairings in the returned ``messages_to_summarize``.

    Algorithm (mirrors PR #141):
      1. Delegate to the original to get the slice + token bookkeeping.
      2. Build a ``tool_call_id → ToolMessage`` map from the FULL source
         ``messages`` list (so we can find missing tool results).
      3. For every ``AIMessage(tool_calls)`` in the slice, append any
         matching ``ToolMessage`` that isn't already present.
      4. Update ``n_tokens_to_summarize`` to account for appended messages.
    """
    result = _original_preprocess(
        messages=messages,
        running_summary=running_summary,
        max_tokens=max_tokens,
        max_tokens_before_summary=max_tokens_before_summary,
        max_summary_tokens=max_summary_tokens,
        token_counter=token_counter,
    )

    if not result.messages_to_summarize:
        return result

    # Build the lookup from the FULL source conversation.
    tool_call_id_to_tool_message: dict[str, ToolMessage] = {
        m.tool_call_id: m
        for m in messages
        if isinstance(m, ToolMessage) and m.tool_call_id
    }

    # Track which tool_call_ids are already represented in the slice.
    existing_tool_result_ids: set[str] = {
        m.tool_call_id
        for m in result.messages_to_summarize
        if isinstance(m, ToolMessage) and m.tool_call_id
    }

    added_tokens = 0
    # Iterate over a snapshot — we may append to the underlying list.
    for m in list(result.messages_to_summarize):
        if not (isinstance(m, AIMessage) and m.tool_calls):
            continue
        for tool_call in m.tool_calls:
            tc_id = tool_call.get("id") if isinstance(tool_call, dict) else None
            if tc_id is None:
                continue
            if tc_id in existing_tool_result_ids:
                continue
            tool_msg = tool_call_id_to_tool_message.get(tc_id)
            if tool_msg is None:
                continue
            added_tokens += token_counter([tool_msg])
            result.messages_to_summarize.append(tool_msg)
            existing_tool_result_ids.add(tc_id)

    if added_tokens:
        return _dataclass_replace(
            result,
            n_tokens_to_summarize=result.n_tokens_to_summarize + added_tokens,
        )
    return result


# Apply the patch at import time.
_ls._preprocess_messages = _patched_preprocess_messages
