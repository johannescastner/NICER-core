# src/middleware/conversation_middleware.py
"""
LangGraph Conversation Logging Middleware

This middleware mechanically logs ALL conversation turns from ANY agent
with embeddings and tone analysis. Simple and reliable.

Design Philosophy:
- Turn-by-turn logging at LangGraph level (not agent level)
- Agent-agnostic: works with any agent without modification
- Mechanical: just log every turn, no optimization logic
- Clean separation: agents focus on domain, middleware handles conversation
- Ensemble-wide: like SummarizationNode, works across all agents
"""
from __future__ import annotations
import logging
import time
from typing import Dict, Any, Optional
from functools import lru_cache
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig

from src.logging.conversation_logger import ConversationLogger

_logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_conversation_logger() -> ConversationLogger:
    """
    Get a process-wide ConversationLogger instance.

    lru_cache gives us lazy, thread-safe singleton semantics
    without using a module-level mutable global + global statement.
    """
    return ConversationLogger()

def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                # common LC content blocks have "text" or "content"
                v = c.get("text") or c.get("content") or c.get("value") or ""
                parts.append(v if isinstance(v, str) else str(v))
            else:
                parts.append(str(c))
        return " ".join(p for p in parts if p)
    return str(content)

async def _log_turn_middleware(
    state: Dict[str, Any],
    config: RunnableConfig,
    turn_type: str,  # 'human' | 'agent'
) -> Dict[str, Any]:
    """
    Universal conversation logging middleware for all LangGraph agents.

    Logs either a human or agent turn with full metadata.
    - Never raises: logging failures are reported to logs but do not break the graph.
    - Turn numbering:
        * Increments once per human turn (including ask_human resumes).
        * Agent/tool turns reuse the last human turn_number.
    """
    try:
        # ---- Basic extraction -------------------------------------------------
        conversation_id = state.get("conversation_id", "unknown")
        turn_number = int(state.get("turn_number", 0))
        messages = state.get("messages", [])

        if not messages:
            return state

        # ---- Validate turn_type early ----------------------------------------
        if turn_type not in ("human", "agent"):
            _logger.warning("Unexpected turn_type=%r; skipping turn log", turn_type)
            return state

        # ---- Resolve config/configurable once --------------------------------
        if isinstance(config, dict):
            configurable = config.get("configurable", {}) or {}
        elif config is not None:
            configurable = getattr(config, "configurable", {}) or {}
        else:
            configurable = {}

        message_to_log: Any = None
        speaker: str
        agent_name: Optional[str] = None

        # ---- Human turn: increment counter + last_human_ts -------------------
        if turn_type == "human":
            # Bump once per *human* turn (includes ask_human resumes).
            turn_number += 1
            state["turn_number"] = turn_number

            # Ensure we have a dict context for idle/engagement tracking.
            raw_ctx = state.get("context") or {}
            if not isinstance(raw_ctx, dict):
                _logger.warning(
                    "State context is not a dict (got %r); resetting to empty.",
                    type(raw_ctx),
                )
                ctx: Dict[str, Any] = {}
            else:
                # Shallow copy to avoid aliasing surprises.
                ctx = dict(raw_ctx)

            ctx["last_human_ts"] = time.time()
            state["context"] = ctx

            # Last human message in the transcript (works for chat + ask_human resumes).
            message_to_log = next(
                (m for m in reversed(messages) if isinstance(m, HumanMessage)),
                None,
            )
            speaker = "human"
            agent_name = None  # No agent for human turns

        # ---- Agent/tool turn: reuse last turn_number -------------------------
        else:  # turn_type == "agent"
            # Prefer AIMessage; fall back to ToolMessage; then any ai/tool-typed message.
            message_to_log = next(
                (m for m in reversed(messages) if isinstance(m, AIMessage)),
                None,
            ) or next(
                (m for m in reversed(messages) if isinstance(m, ToolMessage)),
                None,
            ) or next(
                (m for m in reversed(messages)
                 if getattr(m, "type", None) in ("ai", "tool")),
                None,
            )

            speaker = "assistant"
            # Prefer the message's .name; fall back to state/config; then default.
            agent_name = (
                (getattr(message_to_log, "name", None) if message_to_log else None)
                or state.get("active_agent")
                or configurable.get("active_agent")
                or "unknown_agent"
            )

        if not message_to_log:
            msg_types = [
                getattr(m, "type", type(m).__name__)
                for m in messages
            ]
            # For human turns, this is suspicious → keep as warning.
            # For agent/tool turns, it's often legitimate (internal nodes) → debug.
            level = logging.WARNING if turn_type == "human" else logging.DEBUG
            _logger.log(
                level,
                "Could not find a %s message to log in the current state. "
                "message_types=%s",
                turn_type,
                msg_types,
            )
            return state

        # ---- User/context metadata -------------------------------------------
        user_id = configurable.get("langgraph_auth_user_id", "unknown")

        metadata = {
            "user_id": user_id,
            "agent_name": agent_name,
            "conversation_context": state.get("context", {}),
            "message_count": len(messages),
            "config_context": configurable,
        }

        # ---- Log the turn (non-blocking to convo path) -----------------------
        conv_logger = get_conversation_logger()
        await conv_logger.log_turn(
            conversation_id=conversation_id,
            speaker=speaker,
            content=_as_text(getattr(message_to_log, "content", "")),
            message_type=(
                "tool_call"
                if isinstance(message_to_log, ToolMessage)
                else ("response" if speaker == "assistant" else "question")
            ),
            metadata=metadata,
            agent_name=agent_name,
            # Lightweight context for observability
            memory_token_length=len(str(messages)),
            full_context_content=str(messages[-3:]) if messages else None,
            turn_number=turn_number,
        )

        _logger.debug(
            "Logged %s turn: %s / turn %s / agent=%s",
            turn_type,
            conversation_id,
            turn_number,
            agent_name,
        )

    except Exception as e:  # noqa: BLE001
        # Last-resort guardrail: logging must never break the graph.
        _logger.error(
            "Conversation logging failed for %s turn in conversation %s: %s",
            turn_type,
            state.get("conversation_id", "unknown"),
            e,
            exc_info=True,
        )

    return state

async def log_human_turn_middleware(
        state: Dict[str, Any],
        config: RunnableConfig
) -> Dict[str, Any]:
    """Logs the human's turn in the conversation."""
    return await _log_turn_middleware(state, config, 'human')


async def log_agent_turn_middleware(
        state: Dict[str, Any],
        config: RunnableConfig
) -> Dict[str, Any]:
    """Logs the agent's turn in the conversation."""
    return await _log_turn_middleware(state, config, 'agent')


def create_human_turn_logging_node():
    """
    Create a node that logs the human's turn.
    
    Returns:
        Node function for LangGraph StateGraph.
    """
    return log_human_turn_middleware


def create_agent_turn_logging_node():
    """
    Create a node that logs the agent's turn.
    
    Returns:
        Node function for LangGraph StateGraph.
    """
    return log_agent_turn_middleware
