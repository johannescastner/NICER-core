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

VERSION-DRIFT GUARD (PR-A0-3g, Latent-A):
The patch targets the private symbol ``_preprocess_messages``. If langmem
is upgraded past 0.0.30 in a way that renames or removes that symbol —
or if a venv has the wrong version installed (e.g., 0.0.17, 13 minor
versions behind the ``langmem>=0.0.30,<0.1.0`` pin in pyproject.toml) —
the bare attribute access at line 44 would crash at IMPORT time with a
cryptic ``AttributeError`` deep inside ``patch_typing.py``'s import chain.

The defensive guard below uses ``getattr`` (a structural attribute-presence
check, NOT regex / string-version-matching) and raises an explicit
``ImportError`` that names: (a) the upstream PR this patch mirrors,
(b) the pin version, (c) the diagnostic action. Per the lessons-learned
"Bugs Should Scream" principle: fail loudly at startup with actionable
context, not silently mid-execution.

Verified at PR-A0-3g time:
  * .venv/lib (0.0.30): _preprocess_messages EXISTS — patch applies cleanly
  * env/lib   (0.0.17): _preprocess_messages MISSING — guard fires
  * Production deploys use the pinned 0.0.30 — patch always applies in prod
"""
from __future__ import annotations

import uuid
from dataclasses import replace as _dataclass_replace
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langmem.short_term import summarization as _ls


# Preserve a reference to the original so we can delegate to it.
# Defensive guard (PR-A0-3g, Latent-A): if installed langmem version
# lacks this private symbol — typically because a venv drifted off the
# ``langmem>=0.0.30,<0.1.0`` pin or because upstream eventually renamed
# or removed the helper after merging PR #141 — fail LOUDLY at import
# with actionable context rather than crash mid-execution with a cryptic
# AttributeError. Pure attribute presence check via ``getattr`` (no
# regex, no string-version matching).
_original_preprocess = getattr(_ls, "_preprocess_messages", None)
if _original_preprocess is None:
    import langmem as _langmem
    raise ImportError(
        "src/langgraph_slack/langmem_patch.py: vendored fix for upstream "
        "langmem PR #141 cannot apply — the private symbol "
        "``langmem.short_term.summarization._preprocess_messages`` is not "
        f"present in the installed langmem version "
        f"{getattr(_langmem, '__version__', '<unknown>')!r}. "
        "Expected version range per pyproject.toml: ``langmem>=0.0.30,<0.1.0``. "
        "If upstream PR #141 (https://github.com/langchain-ai/langmem/pull/141) "
        "has merged into a newer langmem release, DELETE this entire module "
        "and the import at the bottom of src/langgraph_slack/patch_typing.py. "
        "If the venv has drifted off the pin, run ``pip install -e .`` or "
        "the equivalent to restore the pinned version. "
        "Failing loudly here per the 'Bugs Should Scream' lessons-learned "
        "discipline — silent fallback would let the patched behaviour "
        "(orphan tool-call repair) silently regress in production."
    )


def _sanitize_message_ids(messages: list) -> list:
    """Return a NEW list where every message has a non-None ``id``.

    langmem's ``_preprocess_messages`` (line 171) raises
    ``ValueError("Messages are required to have ID field.")`` when ANY
    message has ``id is None``. v22's eval surfaced this in a code
    path the wrapper at ``pro/ensemble/summarization_service.py:560``
    catches and silently degrades to "summarization skipped" — losing
    summarization on that turn AND corrupting the DSPy "did we
    summarize?" metric.

    Mirrors the defensive UUID-assignment pattern from ryoma 0.8.1's
    ``_format_messages`` fix. Does NOT mutate input messages in place
    (callers may hold references); returns a new list with new
    instances for any message that needed an id.
    """
    sanitized: list = []
    for m in messages:
        if getattr(m, "id", None) is None:
            sanitized.append(m.model_copy(update={"id": str(uuid.uuid4())}))
        else:
            sanitized.append(m)
    return sanitized


def _patched_preprocess_messages(
    messages: list,
    running_summary: Any,
    max_tokens: int,
    max_tokens_before_summary: int,
    max_summary_tokens: int,
    token_counter,
):
    """Wraps the original ``_preprocess_messages`` with two repairs:

      A. ID sanitization (this patch's v22 extension): assign UUIDs
         to any id-less message before delegating to the strict
         original — prevents the ValueError raise + silent
         degradation surfaced in IntellAgent v22.
      B. Orphan tool-call pairing (mirrors PR #141): post-process the
         original's slice to append any matching ``ToolMessage`` from
         the source conversation that's missing for an
         ``AIMessage(tool_calls)``.

    Algorithm:
      1. Sanitize ids on the input ``messages`` list (return new list).
      2. Delegate to the original to get the slice + token bookkeeping.
      3. Build a ``tool_call_id → ToolMessage`` map from the FULL
         (sanitized) source list.
      4. For every ``AIMessage(tool_calls)`` in the slice, append any
         matching ``ToolMessage`` that isn't already present.
      5. Update ``n_tokens_to_summarize`` to account for appended msgs.
    """
    messages = _sanitize_message_ids(messages)

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
