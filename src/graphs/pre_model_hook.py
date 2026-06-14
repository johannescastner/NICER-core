# src/graphs/pre_model_hook.py
"""Automatic memory-at-intent ``pre_model_hook`` for the baby-NICER LangGraph agent.

A ``pre_model_hook`` runs before EVERY model step in ``create_react_agent``; this
one retrieves the agent's semantic / episodic / procedural memories for the user's
fresh intent and injects them as a ``SystemMessage`` so the model CONDITIONS on
them (heed-by-construction, not a post-hoc gate). It is model-agnostic (baby-NICER
runs on DeepSeek).

WHY this lives in its OWN module and NOT in ``memory.py``: ``memory.py:72`` binds
``NamespaceTemplate = Tuple[str, ...]`` — a non-callable type alias that would
SHADOW langmem's callable resolver in that module's namespace and silently produce
zero-retrieval. Here we import the langmem resolver under an unambiguous alias and
never reference the tuple alias.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping

from langmem.errors import ConfigurationError
from langmem.utils import NamespaceTemplate as LangmemNamespaceTemplate
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


def _content_to_text(content) -> str:
    """The LangChain message ``content`` is a plain string OR a list of content
    blocks (multimodal: ``[{"type":"text","text":...}, {"type":"image_url",...}]``).
    ``asearch`` needs a TEXT query — a raw list silently no-ops retrieval (a real
    embedding store errors on a non-string query). Flatten list content to its
    text parts; a bare string passes through unchanged."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return str(content)

# The three agent-scoped namespace templates (mirroring agent.py's intent). Each
# carries the runtime ``{langgraph_auth_user_id}`` placeholder, resolved per turn
# against the running config (langmem's get_config() fallback, or an explicit
# config threaded by the graph).
_SEMANTIC_NS = ("semantic", "general", "{langgraph_auth_user_id}")
_EPISODIC_NS = ("episodic", "general", "{langgraph_auth_user_id}")
_PROCEDURAL_NS = ("procedural", "general", "{langgraph_auth_user_id}")

# Per-kind section header + leading instruction shown to the model.
_SECTIONS = (
    ("semantic", _SEMANTIC_NS, "## FACTS (semantic)"),
    ("episodic", _EPISODIC_NS, "## EVENTS (episodic)"),
    ("procedural", _PROCEDURAL_NS, "## PROCEDURES — when to act (procedural)"),
)

_PREAMBLE = (
    "You have the following memories retrieved for the user's current request. "
    "Treat them as authoritative context and follow them where they apply."
)


def _format_items(items) -> str:
    """Render a store's search results as plain text lines, or the explicit
    queried-but-empty marker. ``(none retrieved)`` is DISTINCT from a kind being
    absent — every kind's header is always emitted (ROOT 3)."""
    if not items:
        return "(none retrieved)"
    lines = []
    for it in items:
        value = getattr(it, "value", None)
        if isinstance(value, dict):
            content = value.get("content")
            text = content if isinstance(content, str) else _stringify(value)
        else:
            text = _stringify(value if value is not None else it)
        lines.append(f"- {text}")
    return "\n".join(lines)


def _stringify(value) -> str:
    if isinstance(value, dict):
        # Stable, readable rendering of a structured memory value.
        return "; ".join(f"{k}={v}" for k, v in value.items())
    return str(value)


def make_pre_model_hook(
    semantic_store,
    episodic_store,
    procedural_store,
    *,
    limit: int = 3,
):
    """Build the async ``pre_model_hook`` closure over the 3 distinct stores.

    The closure is forced: ``create_react_agent(store=...)`` carries ONE
    ``BaseStore``, but there are 3 physically-distinct stores (semantic /
    episodic / procedural), so they are captured here instead.
    """
    _stores = {
        "semantic": semantic_store,
        "episodic": episodic_store,
        "procedural": procedural_store,
    }
    # Architecture-enforced, not hoped-for (canonical-9 fidelity round-2, AIMA):
    # a None/missing store would otherwise fail SILENTLY on first call inside the
    # BaseException fail-soft (zero retrieval, no operator signal). The invariant
    # "all 3 stores present" is decidable at construction time — enforce it here,
    # loudly, before the closure is ever handed to create_react_agent.
    _missing = [kind for kind, store in _stores.items() if store is None]
    if _missing:
        raise ValueError(
            f"make_pre_model_hook: store(s) {_missing} are None; memory-at-intent "
            "requires all three (semantic, episodic, procedural) stores."
        )

    async def pre_model_hook(state, config=None):
        # Extract messages by the INVARIANT (a state that carries `messages`),
        # not by the concrete `dict` type (canonical-9 fidelity round-2 fix:
        # brittle_enumeration_vs_invariant). create_react_agent may pass a
        # Mapping OR a typed (Pydantic/AgentState) object depending on
        # state_schema; isinstance(state, dict) silently skipped retrieval for
        # the typed-state case. Cover Mapping and attribute access.
        if isinstance(state, Mapping):
            msgs = state.get("messages")
        else:
            msgs = getattr(state, "messages", None)
        if not msgs:
            # Empty / missing messages — pass through, no IndexError, no asearch.
            return {"llm_input_messages": msgs or []}
        if not isinstance(msgs[-1], HumanMessage):
            # ReAct re-entry (last message AI / Tool): return the LIVE messages
            # (NOT {}), so no stale first-turn injected block lingers in the slot.
            return {"llm_input_messages": msgs}

        query = _content_to_text(msgs[-1].content)
        try:
            blocks = [_PREAMBLE]
            for kind, ns_template, header in _SECTIONS:
                resolved_ns = LangmemNamespaceTemplate(ns_template)(config)
                items = await _stores[kind].asearch(
                    resolved_ns, query=query, limit=limit
                )
                blocks.append(f"{header}\n{_format_items(items)}")
            block = "\n\n".join(blocks)
            return {"llm_input_messages": [SystemMessage(content=block), *msgs]}
        except ConfigurationError as exc:
            # No langgraph_auth_user_id in config → the namespace placeholder
            # cannot resolve, so there is no user to scope memory to. This is an
            # EXPECTED benign case (config not threaded), DISTINCT from a real
            # retrieval failure — log quietly at debug and pass through, never a
            # logger.exception stack trace that would be noise (and would drown a
            # genuine failure). ConfigurationError subclasses BaseException, so it
            # MUST be caught before the BaseException fail-soft below.
            logger.debug(
                "pre_model_hook: no auth user id in config; "
                "skipping memory retrieval (%s)", exc,
            )
            return {"llm_input_messages": msgs}
        except BaseException as exc:  # noqa: BLE001 — see below
            # Fail-soft is STRUCTURAL: namespace resolution may raise
            # langmem.errors.ConfigurationError, which subclasses BaseException
            # (NOT Exception) — an `except Exception` would let it ESCAPE and
            # break the model call. So we catch BaseException, but RE-RAISE the
            # control-flow / cooperative-cancellation exceptions that must never
            # be swallowed (a Slack-webhook timeout / client disconnect / uvicorn
            # shutdown that cancels the graph task must propagate).
            if isinstance(
                exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)
            ):
                raise
            logger.exception("pre_model_hook retrieval failed; passing through")
            return {"llm_input_messages": msgs}

    return pre_model_hook
