"""
Comprehensive Conversation Logging System (agent-agnostic)

This module is now persistence-agnostic. It focuses on:
1) building per-turn artifacts (tone analysis, embeddings),
2) assembling structured payloads,
3) delegating all storage/search/summary ops to the Persistence Manager (PM).

✅ MODAL MICROSERVICES: Tone analysis and embeddings via Modal HTTP API
✅ ZERO LOCAL MODELS: No PyTorch/transformers loaded in Cloud Run
✅ FAST STARTUP: 5-10s instead of 60-180s
✅ LOW MEMORY: 1Gi instead of 4Gi
"""
import atexit
import concurrent.futures
import functools
import json
import asyncio
import logging
import uuid
import os
from datetime import datetime, timezone
from typing import (
    Dict,
    List,
    Any,
    NamedTuple,
    Optional,
)
from dataclasses import dataclass, asdict, fields
from collections.abc import Mapping, Sequence

from pro.persistence import get_persistence_manager
from pro.monitoring.langsmith_integration import get_langsmith_integration
from src.langgraph_slack.config import (
    PROJECT_ID,
    LANGSMITH_PROJECT,
    CONVERSATION_MAX_CONTENT_CHARS,
)

# ============================================================================
# ML INFERENCE CLIENT (replaces local HuggingFace models)
# ============================================================================
from pro.ml_inference.client import (
    get_inference_client,
    analyze_tone as _analyze_tone_api,
    generate_embedding as _generate_embedding_api,
)
# Reuse the ONE shared embedding cache (keyed (model, text)) that
# ModalEmbeddings uses for memory asearch/aput — consolidating the two
# previously-independent embed caches into one. src→src import (allowed);
# memory.py does not import this module, so no cycle.
from src.graphs.memory import _embed_cache_get, _embed_cache_put

# Set up logging
logger = logging.getLogger(__name__)


# Bounded executor for conversation-logger telemetry writes (BQ inserts +
# error logging). Mirrors PR-A0-3a / PR-A0-3e / PR-A0-3f pattern with
# atexit shutdown(wait=False, cancel_futures=True).
#
# Bug #79 root cause: ``asyncio.to_thread`` lands work on the default
# executor, whose 300s ``shutdown_default_executor`` timeout closes the
# event loop while callbacks are still queued. Routing through a
# dedicated executor keeps the default empty.
#
# Defined LOCALLY in src/ to preserve the src/→pro/ dependency
# direction: src/ is the open-source surface, pro/ is proprietary,
# pro/ may depend on src/ but NEVER the reverse. Sized to match
# pro/persistence.py::_TELEMETRY_EXECUTOR (max_workers=3) since the
# write profiles are identical (post-hoc BQ inserts, per-turn,
# idempotent on re-run).
_CONVERSATION_LOGGER_EXECUTOR: concurrent.futures.ThreadPoolExecutor = (
    concurrent.futures.ThreadPoolExecutor(
        max_workers=3,
        thread_name_prefix="conversation-logger",
    )
)
atexit.register(
    _CONVERSATION_LOGGER_EXECUTOR.shutdown,
    wait=False,
    cancel_futures=True,
)

# Tone analysis token cap
_TONE_MAX_LENGTH_ENV = "CONVERSATION_TONE_MAX_TOKENS"
_DEFAULT_TONE_MAX_LENGTH = 256
_MIN_TONE_MAX_LENGTH = 32
_MAX_TONE_MAX_LENGTH = 512


def _get_tone_max_length_from_env() -> int:
    raw = os.getenv(_TONE_MAX_LENGTH_ENV, str(_DEFAULT_TONE_MAX_LENGTH))
    try:
        val = int(raw)
    except (ValueError, TypeError):
        val = _DEFAULT_TONE_MAX_LENGTH
    if val < _MIN_TONE_MAX_LENGTH:
        return _MIN_TONE_MAX_LENGTH
    if val > _MAX_TONE_MAX_LENGTH:
        return _MAX_TONE_MAX_LENGTH
    return val


def _sanitize_for_json(value: Any) -> Any:
    """
    Best-effort sanitizer to make arbitrary nested structures JSON-safe.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(k): _sanitize_for_json(v)
            for k, v in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize_for_json(v) for v in value]
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError, OverflowError):
        return repr(value)


class EmbeddingAttempt(NamedTuple):
    """Outcome of a single ``generate_embedding`` call.

    PR-A0-13b (2026-05-14, Bug #77b fix): replaces the prior
    silent ``return []`` pattern. Explicit two-channel result:

    * ``embedding`` is the vector on success, ``None`` on failure.
    * ``status`` is ``None`` on success, otherwise the underlying
      exception class name (``"HTTPStatusError"``,
      ``"TimeoutException"``, ``"NetworkError"``, ``"RuntimeError"``,
      etc.) — Modal client's typed exceptions ARE the failure
      vocabulary; we don't need a separate envelope class.

    Stored in the conversation_turns row's ``embedding_generation_status``
    column so post-hoc queries can identify rows that need re-embedding
    AND distinguish failure modes by class name.
    """
    embedding: Optional[List[float]]
    status: Optional[str]


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation with comprehensive tone analysis."""
    conversation_id: str
    turn_id: str
    turn_number: int
    timestamp: str
    speaker: str  # 'human' or 'assistant' (normalized)
    content: str
    message_type: str  # 'question', 'response', 'tool_call', 'error'
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    # PR-A0-13b: None on success; exception class name on failure
    # (e.g. "HTTPStatusError", "TimeoutException"). Lets downstream
    # cosine-search queries filter ``WHERE embedding_generation_status
    # IS NULL`` to skip rows where Modal embedding failed.
    embedding_generation_status: Optional[str] = None
    langsmith_trace_id: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    # Context tracking
    memory_token_length: Optional[int] = None
    full_context_content: Optional[str] = None
    # 🎯 UNIVERSAL TONE ANALYSIS: Applied to every turn regardless of agent
    agent_name: Optional[str] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    sentiment_confidence: Optional[float] = None
    formality_score: Optional[float] = None
    formality_label: Optional[str] = None
    formality_confidence: Optional[float] = None
    professional_score: Optional[float] = None
    professional_label: Optional[str] = None
    nuanced_emotions: Optional[Dict[str, Any]] = None

    thumbs_up: Optional[bool] = None
    thumbs_down: Optional[bool] = None
    # LangSmith Feedback Scores for DSPy Metrics
    langsmith_feedback_score: Optional[float] = None
    feedback_type: Optional[str] = None
    evaluation_details: Optional[str] = None

    def to_row(self) -> Dict[str, Any]:
        """Build a BigQuery-ready row dict."""
        data: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in ("metadata", "error_details", "nuanced_emotions"):
                data[f.name] = _sanitize_for_json(value)
            else:
                data[f.name] = value
        return data


@dataclass
class ConversationMetadata:
    """High-level conversation metadata."""
    conversation_id: str
    start_time: str
    end_time: Optional[str]
    participants: List[str]
    conversation_type: str
    focus_area: str
    total_turns: int
    status: str
    langsmith_session_id: Optional[str] = None


# ============================================================================
# CONVERSATION LOGGER CLASS (Refactored for Modal)
# ============================================================================

class ConversationLogger:
    """
    Comprehensive conversation logging with Modal-based ML inference.
    
    No local model loading required - all ML inference delegated to Modal.
    This reduces memory from 4Gi to ~1Gi and eliminates 60-180s cold starts.
    """
    
    def __init__(
            self,
            project_id: str = PROJECT_ID or "",
            dataset_id: str | None = None,
            langsmith_project: str = LANGSMITH_PROJECT,
            pm=None
    ):
        """Initialize the conversation logger (lightweight, no model loading)."""
        self.project_id = project_id
        self.langsmith_project = langsmith_project

        # Persistence Manager
        self.pm = pm or get_persistence_manager()

        # Dataset name — derive from persistence manager if not explicitly set
        if dataset_id is not None:
            self.dataset_id = dataset_id
        else:
            self.dataset_id = self.pm.conversation_logs_dataset

        # Tone analysis token cap
        self._tone_max_length: int = _get_tone_max_length_from_env()
        
        # ML Inference client (HTTP client to Modal, no local models)
        self._inference_client = get_inference_client()

        logger.info(
            "ConversationLogger initialized for project %s (Modal-based inference)",
            self.project_id
        )

    async def analyze_turn_tone(
            self,
            content: str,
            speaker: str,
            agent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        🎯 Unified tone analysis for ANY turn (human or any agent).
        
        Uses Modal-hosted models via HTTP API. No local model loading required.
        Single API call returns sentiment + formality + emotion analysis.
        """
        spk = (speaker or "").strip().lower()
        tone = {
            "agent_name": agent_name if spk == "assistant" else None,
            "sentiment_score": None,
            "sentiment_label": None,
            "sentiment_confidence": None,
            "formality_score": None,
            "formality_label": None,
            "formality_confidence": None,
            "professional_score": None,
            "professional_label": None,
            "nuanced_emotions": None,
        }

        if not content or not content.strip():
            return tone

        try:
            # Single Modal API call for all tone analysis
            result = await _analyze_tone_api(content, max_length=self._tone_max_length)
            
            # Merge results into tone dict
            tone.update({
                "sentiment_score": result.get("sentiment_score"),
                "sentiment_label": result.get("sentiment_label"),
                "sentiment_confidence": result.get("sentiment_confidence"),
                "formality_score": result.get("formality_score"),
                "formality_label": result.get("formality_label"),
                "formality_confidence": result.get("formality_confidence"),
                "professional_score": result.get("professional_score"),
                "professional_label": result.get("professional_label"),
                "nuanced_emotions": result.get("nuanced_emotions"),
            })
            
            # Keep agent_name from our calculation
            tone["agent_name"] = agent_name if spk == "assistant" else None
            
        except Exception as exc:
            logger.warning("⚠️ Tone analysis failed: %s", exc, exc_info=True)

        return tone

    async def generate_embedding(self, text: str) -> EmbeddingAttempt:
        """
        Generate semantic embedding for text via Modal bge-base-en-v1.5
        (768 dimensions).

        PR-A0-13b: returns an explicit ``EmbeddingAttempt`` (embedding,
        status). On success: ``(vector, None)``. On failure: ``(None,
        type(exc).__name__)``. The prior silent ``return []`` pattern
        wrote rows with empty embedding arrays that downstream cosine
        search treated as zero-distance — silent semantic-memory
        corruption (43 rows in v38's Modal storm window).

        Modal client's typed exceptions ARE the failure vocabulary:
        ``HTTPStatusError`` (4xx/5xx after retry), ``TimeoutException``,
        ``NetworkError``, ``RemoteProtocolError``, ``RuntimeError``
        (Modal retry exhaustion). PR-A0-13a (commit a05fa31) made the
        Modal client honour ``Retry-After`` up to 180s in both sync
        and async paths, so this failure should be rare in v39+.
        """
        try:
            # Shared-cache hit → no Modal call at all (de-dup: a text already
            # embedded by memory search/write is reused here, and vice versa).
            cached = _embed_cache_get("bge-base", text)
            if cached is not None:
                return EmbeddingAttempt(embedding=cached, status=None)
            # Miss → circuit-breaker-protected _call_modal path (unchanged),
            # then populate the shared cache on success only.
            embedding = await _generate_embedding_api(text, model="bge-base")
            _embed_cache_put("bge-base", text, embedding)
            return EmbeddingAttempt(embedding=embedding, status=None)
        except Exception as e:
            status = type(e).__name__
            logger.error(
                "Failed to generate embedding (status=%s): %s",
                status, e,
            exc_info=True)
            return EmbeddingAttempt(embedding=None, status=status)

    async def log_turn(
            self,
            conversation_id: str,
            speaker: str,
            content: str,
            message_type: str = "message",
            metadata: Optional[Dict[str, Any]] = None,
            langsmith_trace_id: Optional[str] = None,
            error_details: Optional[Dict[str, Any]] = None,
            memory_token_length: Optional[int] = None,
            full_context_content: Optional[str] = None,
            agent_name: Optional[str] = None,
            *,
            turn_number: int,
    ) -> str:
        """Log a conversation turn with tone analysis."""
        turn_id = str(uuid.uuid4())

        # Content safety cap
        content = content or ""
        try:
            max_chars = int(
                CONVERSATION_MAX_CONTENT_CHARS
            ) if CONVERSATION_MAX_CONTENT_CHARS else None
        except (ValueError, TypeError):
            max_chars = None

        original_chars = len(content)
        safe_content = (
            content[:max_chars]
            if (isinstance(content, str) and max_chars and original_chars > max_chars)
            else content
        )

        # Annotate truncation
        meta = dict(metadata or {})
        if original_chars != len(safe_content):
            try:
                trunc = meta.setdefault("truncation", {})
                trunc.update({
                    "field": "content",
                    "original_chars": original_chars,
                    "stored_chars": len(safe_content),
                    "max_chars": max_chars,
                })
            except (KeyError, TypeError):
                meta["content_original_chars"] = original_chars
                meta["content_stored_chars"] = len(safe_content)
                meta["content_max_chars"] = max_chars
            logger.debug(
                "Turn content truncated from %s to %s chars (cap=%s)",
                original_chars,
                len(safe_content),
                max_chars
            )

        # Apply tone analysis via Modal API (non-blocking HTTP call)
        tone_analysis = await self.analyze_turn_tone(
            safe_content,
            speaker,
            agent_name
        )

        # Generate embedding via Modal API (non-blocking HTTP call).
        # PR-A0-13b: result is an explicit ``EmbeddingAttempt`` —
        # on failure, embedding is None (not []) and status carries
        # the exception class name for the audit trail.
        embed_result = await self.generate_embedding(safe_content)

        # Validate turn_number
        if not isinstance(turn_number, int) or turn_number < 1:
            logger.warning("Invalid turn_number %r; defaulting to 1", turn_number)
            turn_number = 1

        # Create turn object
        turn = ConversationTurn(
            conversation_id=conversation_id,
            turn_id=turn_id,
            turn_number=turn_number,
            timestamp=datetime.now(timezone.utc).isoformat(),
            speaker=speaker,
            content=safe_content,
            message_type=message_type,
            metadata=meta or {},
            embedding=embed_result.embedding,
            embedding_generation_status=embed_result.status,
            langsmith_trace_id=langsmith_trace_id,
            error_details=error_details or {},
            memory_token_length=memory_token_length,
            full_context_content=full_context_content,
            agent_name=tone_analysis.get("agent_name"),
            sentiment_score=tone_analysis.get("sentiment_score"),
            sentiment_label=tone_analysis.get("sentiment_label"),
            sentiment_confidence=tone_analysis.get("sentiment_confidence"),
            formality_score=tone_analysis.get("formality_score"),
            formality_label=tone_analysis.get("formality_label"),
            formality_confidence=tone_analysis.get("formality_confidence"),
            professional_score=tone_analysis.get("professional_score"),
            professional_label=tone_analysis.get("professional_label"),
            nuanced_emotions=tone_analysis.get("nuanced_emotions"),
            thumbs_up=None,
            thumbs_down=None,
            langsmith_feedback_score=0.5,
            feedback_type="neutral",
            evaluation_details="{}"
        )

        # Store turn
        await self._store_turn(turn)

        # Store LangSmith trace
        if langsmith_trace_id:
            try:
                md = meta or {}
                await self.pm.store_langsmith_trace(
                    langsmith_trace_id,
                    conversation_id=conversation_id,
                    thread_id=md.get("thread_id"),
                    user_id=md.get("user_id") or md.get("slack_user_id"),
                    channel_id=md.get("channel_id"),
                    turn_id=turn_id,
                    turn_number=turn_number,
                    source="conversation_logger",
                )
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning("⚠️ Failed to store LangSmith trace for turn %s: %s", turn_id, e)

        logger.debug("Logged turn %s for conversation %s", turn_id, conversation_id)
        return turn_id

    async def log_error(
        self,
        conversation_id: str,
        error_type: str,
        error_message: str,
        turn_id: Optional[str] = None,
        stack_trace: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "error",
    ) -> str:
        """Log an error that occurred during conversation."""
        error_id = str(uuid.uuid4())

        error_data = {
            "error_id": error_id,
            "conversation_id": conversation_id,
            "turn_id": turn_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "stack_trace": stack_trace,
            "context": json.dumps(context) if context else None,
            "severity": severity,
        }

        try:
            ls = get_langsmith_integration()
            run_id = ls.get_last_trace_id() or conversation_id
        except (RuntimeError, ValueError, KeyError):
            run_id = conversation_id

        try:
            # PR-A0-13c: bounded local executor (Bug #79).
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                _CONVERSATION_LOGGER_EXECUTOR,
                functools.partial(
                    self.pm.log_cognitive_error,
                    run_id=run_id,
                    module="conversation",
                    error_type=error_type,
                    error_message=error_message,
                    operation_context=error_data,
                ),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Failed to persist cognitive error for conversation %s: %s",
                conversation_id,
                exc,
                exc_info=True,
            )

        logger.error(
            "Logged error %s for conversation %s: %s",
            error_id,
            conversation_id,
            error_message,
        )
        return error_id

    async def log_metric(
            self,
            conversation_id: str,
            metric_type: str,
            metric_name: str,
            metric_value: float,
            metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a performance or learning metric."""
        metric_id = str(uuid.uuid4())

        metric_data = {
            "metric_id": metric_id,
            "conversation_id": conversation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metric_type": metric_type,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "metadata": json.dumps(metadata) if metadata else None
        }

        table_id = (
            f"{self.project_id}.{self.dataset_id}.conversation_metrics"
            if self.project_id and self.dataset_id
            else "conversation_metrics"
        )
        # PR-A0-13c: bounded local executor (Bug #79).
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _CONVERSATION_LOGGER_EXECUTOR,
            self.pm.insert_rows,
            table_id, [metric_data], [metric_id],
        )

        logger.debug(
            "Logged metric %s=%s for conversation %s", 
            metric_name,
            metric_value,
            conversation_id
        )
        return metric_id

    async def end_conversation(
            self,
            conversation_id: str,
            status: str = "completed"
    ):
        """Conversation lifecycle hook (currently a no-op)."""
        return

    async def search_conversations(
            self,
            query: str,
            limit: int = 10,
            conversation_type: Optional[str] = None,
            focus_area: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Delegate semantic search to PM."""
        if hasattr(self.pm, "search_conversations"):
            return await self.pm.search_conversations(
                query=query,
                limit=limit,
                conversation_type=conversation_type,
                focus_area=focus_area,
            )
        raise NotImplementedError(
            "search_conversations is handled by the Persistence Manager."
        )

    async def get_conversation_summary(
            self,
            conversation_id: str
    ) -> Dict[str, Any]:
        """Delegate summary construction to PM."""
        return await self.pm.get_conversation_summary(conversation_id)

    async def _store_conversation_metadata(
            self, metadata: ConversationMetadata,
            extra_metadata: Optional[Dict[str, Any]] = None
    ):
        """Store conversation metadata via PM."""
        data = asdict(metadata)
        if extra_metadata:
            data["metadata"] = json.dumps(extra_metadata)
        else:
            data["metadata"] = None

        table_id = (
            f"{self.project_id}.{self.dataset_id}.conversations"
            if self.project_id and self.dataset_id
            else "conversations"
        )
        # PR-A0-13c: bounded local executor (Bug #79).
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _CONVERSATION_LOGGER_EXECUTOR,
            self.pm.insert_rows,
            table_id, [data], [data["conversation_id"]],
        )

    async def _store_turn(
            self,
            turn: ConversationTurn
    ):
        """Store a conversation turn via PM."""
        data = turn.to_row()

        # JSON-encode metadata and error_details
        if data.get("metadata"):
            data["metadata"] = json.dumps(data["metadata"])
        else:
            data["metadata"] = None

        if data.get("error_details"):
            data["error_details"] = json.dumps(data["error_details"])
        else:
            data["error_details"] = None

        # Handle nuanced_emotions JSON column
        ne = data.get("nuanced_emotions")
        try:
            if ne is None:
                data["nuanced_emotions"] = None
                logger.debug(
                    "ConversationLogger._store_turn nuanced_emotions is None for turn_id=%s",
                    data.get("turn_id"),
                )
            elif isinstance(ne, str):
                json.loads(ne)  # Validate
                data["nuanced_emotions"] = ne
                logger.debug(
                    "ConversationLogger._store_turn nuanced_emotions (str) sample=%s",
                    ne[:500],
                )
            else:
                json_str = json.dumps(ne, default=_sanitize_for_json)
                data["nuanced_emotions"] = json_str
                logger.debug(
                    "ConversationLogger._store_turn nuanced_emotions (obj) type=%s sample=%s",
                    type(ne),
                    json_str[:500],
                )
        except (ValueError, TypeError, OverflowError) as exc:
            logger.warning(
                "ConversationLogger._store_turn: failed to serialise nuanced_emotions; "
                "dropping field for turn_id=%s: %s",
                data.get("turn_id"),
                exc,
                exc_info=True,
            )
            data["nuanced_emotions"] = None

        # Store via PM
        table_id = (
            f"{self.project_id}.{self.dataset_id}.conversation_turns"
            if self.project_id and self.dataset_id
            else "conversation_turns"
        )
        # PR-A0-13c: bounded local executor (Bug #79).
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _CONVERSATION_LOGGER_EXECUTOR,
            self.pm.insert_rows,
            table_id, [data], [data["turn_id"]],
        )