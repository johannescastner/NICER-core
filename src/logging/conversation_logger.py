# /src/logging/conversation_logger.py
"""
Comprehensive Conversation Logging System (agent-agnostic)

This module is now persistence-agnostic. It focuses on:
1) building per-turn artifacts (tone analysis, embeddings),
2) assembling structured payloads,
3) delegating all storage/search/summary ops to the Persistence Manager (PM).
"""
import json
import asyncio
import logging
import uuid
import os
import threading
from datetime import datetime, timezone
from typing import (
    Dict,
    List,
    Any,
    Optional
)
from dataclasses import dataclass, asdict, fields
from collections.abc import Mapping, Sequence
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from pro.persistence import get_persistence_manager
from pro.monitoring.langsmith_integration import get_langsmith_integration
from src.langgraph_slack.config import (
    PROJECT_ID,
    LANGSMITH_PROJECT,
    HF_CACHE_DIR,
    CONVERSATION_MAX_CONTENT_CHARS
)

# Set up logging
logger = logging.getLogger(__name__)
 
# ---------------- Tone analysis token caps ----------------
# HF text-classification/sentiment pipelines accept `truncation=True`
# and `max_length` to control tokenizer truncation. RoBERTa-family
# checkpoints typically support up to ~512 tokens. We default lower
# because tone signal usually saturates early, and shorter caps reduce
# latency and memory pressure.
_TONE_MAX_LENGTH_ENV = "CONVERSATION_TONE_MAX_TOKENS"
_DEFAULT_TONE_MAX_LENGTH = 256
_MIN_TONE_MAX_LENGTH = 32
_MAX_TONE_MAX_LENGTH = 512

def _get_tone_max_length_from_env() -> int:
    raw = os.getenv(_TONE_MAX_LENGTH_ENV, str(_DEFAULT_TONE_MAX_LENGTH))
    try:
        val = int(raw)
    except Exception:
        val = _DEFAULT_TONE_MAX_LENGTH
    if val < _MIN_TONE_MAX_LENGTH:
        return _MIN_TONE_MAX_LENGTH
    if val > _MAX_TONE_MAX_LENGTH:
        return _MAX_TONE_MAX_LENGTH
    return val

def _sanitize_for_json(value: Any) -> Any:
    """
    Best-effort sanitizer to make arbitrary nested structures JSON-safe.

    - Leaves primitives as-is
    - Converts datetimes to ISO strings
    - Recurses into mappings/sequences
    - Falls back to repr(...) for objects that can't be json-dumped
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
    # Last resort: if it's already JSON-serializable, keep it; otherwise repr(...)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)

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
    langsmith_trace_id: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    # Context tracking
    memory_token_length: Optional[int] = None
    full_context_content: Optional[str] = None
    # 🎯 UNIVERSAL TONE ANALYSIS: Applied to every turn regardless of agent
    agent_name: Optional[str] = None  # Which agent spoke (if assistant)
    sentiment_score: Optional[float] = None  # 0.0-1.0 (1.0 = very positive)
    sentiment_label: Optional[str] = None  # POSITIVE, NEGATIVE, NEUTRAL
    sentiment_confidence: Optional[float] = None  # Model confidence
    formality_score: Optional[float] = None  # 0.0-1.0 (1.0 = very formal)
    formality_label: Optional[str] = None  # FORMAL, INFORMAL
    formality_confidence: Optional[float] = None  # Model confidence
    professional_score: Optional[float] = None  # 0.0-1.0 (1.0 = very professional)
    professional_label: Optional[str] = None  # PROFESSIONAL, CASUAL, INAPPROPRIATE
    nuanced_emotions: Optional[Dict[str, Any]] = None  # Structured emotion analysis

    thumbs_up: Optional[bool] = None  # User reaction tracking
    thumbs_down: Optional[bool] = None  # User reaction tracking
    # LangSmith Feedback Scores for DSPy Metrics
    langsmith_feedback_score: Optional[float] = None  # 0.0-1.0 from LangSmith evaluators
    feedback_type: Optional[str] = None  # "positive", "negative", "neutral"
    evaluation_details: Optional[str] = None  # JSON with detailed evaluation results

    def to_row(self) -> Dict[str, Any]:
        """
        Build a BigQuery-ready row dict without deep-copying problematic
        objects (e.g. uvloop.Loop, PGStore instances) that might live in
        metadata / error_details / nuanced_emotions.
        """
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
    conversation_type: str  # 'metadata_learning', 'testing', 'evaluation'
    focus_area: str  # 'semantic_memory', 'ego_networks', etc.
    total_turns: int
    status: str  # 'active', 'completed', 'error'
    langsmith_session_id: Optional[str] = None


class ConversationLogger:
    """
    Comprehensive conversation logging (persistence-agnostic).
    """
    def __init__(
            self,
            project_id: str = PROJECT_ID or "",
            dataset_id: str = "conversation_logs_eu",
            langsmith_project: str = LANGSMITH_PROJECT,
            pm=None
    ):
        """
        Initialize the conversation logger.
        NOTE: project_id/dataset_id/langsmith_project are no-ops here and kept only for
        backwards compatibility with call-sites; PM owns all persistence config.
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.langsmith_project = langsmith_project

        # Persistence Manager (injected or global factory)
        self.pm = pm or get_persistence_manager()
 
        # Tone analysis token cap (configurable)
        self._tone_max_length: int = _get_tone_max_length_from_env()
        logger.info("ConversationLogger tone max_length=%s (env=%s)",
                    self._tone_max_length, _TONE_MAX_LENGTH_ENV)

        # --- HF cache root (managed via config) ---
        # In Cloud, path is ephemeral; in self-hosted, point to a mounted volume.
        self._hf_cache_dir = HF_CACHE_DIR
        if self._hf_cache_dir:
            try:
                os.makedirs(self._hf_cache_dir, exist_ok=True)
            except Exception as e:
                logger.warning("HF cache dir %s not creatable: %s", self._hf_cache_dir, e)

        # Lazy, process-lifetime caches (instantiate on first use)
        self._embedding_model: Optional[SentenceTransformer] = None
        self._sentiment_classifier = None
        self._formality_classifier = None
        self._emotion_classifier = None
        # Thread-safety for lazy initialisation
        self._embed_lock = threading.Lock()
        self._pipeline_lock = threading.Lock()

        # ✅ BLOCKING WARMUP: Load all models during initialization
        # This prevents the first turn from timing out while downloading models
        logger.info("🔄 Loading HuggingFace models (this may take a few minutes on first run)...")
        self._warmup_models_sync()
        logger.info("✅ All models loaded and ready")

        logger.info("ConversationLogger initialized for project %s", self.project_id)

    # -------- Lazy constructors (first-use load, then reuse) --------
    def _ensure_embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            with self._embed_lock:
                if self._embedding_model is None:
                    try:
                        if self._hf_cache_dir:
                            self._embedding_model = SentenceTransformer(
                                "sentence-transformers/all-MiniLM-L6-v2",
                                cache_folder=self._hf_cache_dir
                            )
                        else:
                            self._embedding_model = SentenceTransformer(
                                "sentence-transformers/all-MiniLM-L6-v2"
                            )
                        logger.info("✅ SBERT embedding model ready")
                    except Exception as e:
                        logger.warning("⚠️ Failed to init embedding model: %s", e, exc_info=True)
                        raise
        return self._embedding_model

    def _ensure_tone_pipelines(self) -> None:
        if self._sentiment_classifier and self._formality_classifier and self._emotion_classifier:
            return
        with self._pipeline_lock:
            pipe_kwargs = {}
            if self._hf_cache_dir:
                pipe_kwargs = {
                    "model_kwargs": {"cache_dir": self._hf_cache_dir},
                    "tokenizer_kwargs": {"cache_dir": self._hf_cache_dir},
                }
            if self._sentiment_classifier is None:
                try:
                    self._sentiment_classifier = pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        return_all_scores=True,
                        **pipe_kwargs,
                    )
                    logger.info("✅ Sentiment pipeline ready")
                except Exception as e:
                    logger.warning("⚠️ Sentiment pipeline load failed: %s", e, exc_info=True)
            if self._formality_classifier is None:
                try:
                    self._formality_classifier = pipeline(
                        "text-classification",
                        model="s-nlp/roberta-base-formality-ranker",
                        return_all_scores=True,
                        **pipe_kwargs,
                    )
                    logger.info("✅ Formality pipeline ready")
                except Exception as e:
                    logger.warning("⚠️ Formality pipeline load failed: %s", e, exc_info=True)
            if self._emotion_classifier is None:
                try:
                    self._emotion_classifier = pipeline(
                        "text-classification",
                        model="SamLowe/roberta-base-go_emotions",  # multi-label
                        return_all_scores=True,
                        **pipe_kwargs,
                    )
                    logger.info("✅ Emotion pipeline ready")
                except Exception as e:
                    logger.warning("⚠️ Emotion pipeline load failed: %s", e, exc_info=True)

    def _warmup_models_sync(self) -> None:
        """
        ✅ SYNCHRONOUS MODEL WARMUP
        
        Loads all HuggingFace models during initialization.
        This blocks __init__ but ensures the first turn never times out.
        
        Models loaded (total ~1.5GB on first run):
        1. sentence-transformers/all-MiniLM-L6-v2 (~90MB)
        2. cardiffnlp/twitter-roberta-base-sentiment-latest (~500MB)
        3. s-nlp/roberta-base-formality-ranker (~500MB)
        4. SamLowe/roberta-base-go_emotions (~500MB)
        """
        try:
            # Load all 3 tone analysis pipelines
            logger.info("📥 Loading tone analysis pipelines...")
            self._ensure_tone_pipelines()
            
            # Load embedding model
            logger.info("📥 Loading embedding model...")
            self._ensure_embedding_model()
            
            # Dry-run to ensure model weights are fully loaded into memory
            logger.info("🔄 Running warmup inference...")
            _ = self.generate_embedding("warmup test sentence for model initialization")
            
            logger.info("🔋 ConversationLogger warmup complete - all models ready")
        except Exception as e:
            logger.error("❌ Model warmup failed: %s", e, exc_info=True)
            raise

    async def _warmup_models(self) -> None:
        """
        Async variant (kept for compatibility but now just wraps sync version).
        """
        await asyncio.to_thread(self._warmup_models_sync)

    def analyze_turn_tone(
            self,
            content: str,
            speaker: str,
            agent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        🎯 Unified tone analysis for ANY turn (human or any agent).

        - Works regardless of which agent produced the message.
        - Records `agent_name` only when `speaker == "assistant"` (aligns with BQ schema).
        - Truncates long content defensively.
        - Supports both return shapes from HF pipelines
        (single dict OR list of dicts via return_all_scores=True).

        Returns a dict shaped for BigQuery:
        {
            "agent_name": str|None,
            "sentiment_score": float|None,
            "sentiment_label": str|None,
            "sentiment_confidence": float|None,
            "formality_score": float|None,
            "formality_label": str|None,
            "formality_confidence": float|None,
            "professional_score": float|None,
            "professional_label": str|None,
            "nuanced_emotions": {
                "primary_emotion": str|None, "primary_confidence": float|None,
                "secondary_emotion": str|None, "secondary_confidence": float|None,
                "tertiary_emotion": str|None, "tertiary_confidence": float|None,
                "all_emotions": str|None  # JSON string of {label: score}
            } | None
        }
        """
        # Ensure tone pipelines are available; if any fail, analysis gracefully no-ops.
        self._ensure_tone_pipelines()
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

        # 1) Truncate for model stability (char cap as a first-pass guard).
        # Token-level safety is enforced by passing truncation/max_length to pipelines.
        text = content[:2000] if len(content) > 2000 else content
        # NOTE: classifiers might be None; we treat that as "no-op" and keep defaults.

        try:
            # ---------------------------
            # Sentiment (defensive parse)
            # ---------------------------
            if self._sentiment_classifier:
                s_res = self._sentiment_classifier(
                    text,
                    truncation=True,
                    max_length=self._tone_max_length,
                )
                # HF shapes: [ {label, score} ]  OR  [ [ {label, score}, ... ] ]
                candidates = []
                if isinstance(s_res, list) and s_res:
                    first = s_res[0]
                    if isinstance(first, list):          # return_all_scores=True
                        candidates = first
                    elif isinstance(first, dict):        # default pipeline output
                        candidates = [first]
                if candidates:
                    best = max(candidates, key=lambda x: x.get("score", 0.0))
                    lbl = str(best.get("label", "")).upper()
                    conf = float(best.get("score", 0.0))
                    tone["sentiment_label"] = lbl
                    tone["sentiment_confidence"] = conf
                    if lbl == "POSITIVE":
                        tone["sentiment_score"] = conf
                    elif lbl == "NEGATIVE":
                        tone["sentiment_score"] = 1.0 - conf
                    else:  # NEUTRAL/OTHER
                        tone["sentiment_score"] = 0.5

            # ---------------------------
            # Formality (defensive parse)
            # ---------------------------
            if self._formality_classifier:
                f_res = self._formality_classifier(
                    text,
                    truncation=True,
                    max_length=self._tone_max_length,
                )

                candidates = []
                if isinstance(f_res, list) and f_res:
                    first = f_res[0]
                    if isinstance(first, list):
                        candidates = first
                    elif isinstance(first, dict):
                        candidates = [first]
                if candidates:
                    best = max(candidates, key=lambda x: x.get("score", 0.0))
                    lbl_raw = str(best.get("label", ""))
                    lbl = lbl_raw.upper()
                    conf = float(best.get("score", 0.0))
                    tone["formality_label"] = lbl
                    tone["formality_confidence"] = conf
                    tone["formality_score"] = conf if lbl == "FORMAL" else (1.0 - conf)

            # ---------------------------------------
            # Nuanced emotions (scores map + thresholded labels)
            # ---------------------------------------
            if self._emotion_classifier:
                e_res = self._emotion_classifier(
                    text,
                    truncation=True,
                    max_length=self._tone_max_length,
                )

                # Expect: [ [ {label, score}, ... ] ] OR [ {label, score}, ... ]
                emotions_all = []
                if isinstance(e_res, list) and e_res:
                    first = e_res[0]
                    if isinstance(first, list):
                        emotions_all = first
                    elif isinstance(first, dict):
                        emotions_all = [first]
                    elif isinstance(first, (tuple, set)):
                        emotions_all = list(first)

                if emotions_all:
                    emotions_all_sorted = sorted(
                        emotions_all,
                        key=lambda x: x.get("score", 0.0),
                        reverse=True
                    )
                    # GoEmotions is multi-label; keep thresholded labels + full score map.
                    threshold = 0.3
                    scores_map = {
                        str(e.get("label")): float(e.get("score", 0.0))
                        for e in emotions_all_sorted
                    }
                    labels_above = [
                        lbl for lbl, sc in scores_map.items() if sc >= threshold
                    ]
                    tone["nuanced_emotions"] = {
                        "labels_above_threshold": labels_above,
                        "threshold": threshold,
                        "scores": scores_map  # BigQuery JSON column
                    }

            # -------------------------------------------------------
            # Professionalism heuristic (requires both signals)
            # -------------------------------------------------------
            if tone["formality_score"] is not None and tone["sentiment_score"] is not None:
                prof = (tone["formality_score"] * 0.7) + (tone["sentiment_score"] * 0.3)
                tone["professional_score"] = prof
                tone["professional_label"] = (
                    "PROFESSIONAL" if prof >= 0.7 else
                    "CASUAL" if prof >= 0.4 else
                    "INAPPROPRIATE"
                )

        except Exception as exc:
            logger.warning("⚠️ Tone analysis failed: %s", exc)

        return tone

    def generate_embedding(self, text: str) -> List[float]:
        """Generate semantic embedding for text using same model as memory system."""
        try:
            model = self._ensure_embedding_model()
            embedding = model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(
                "Failed to generate embedding: %s",
                e
            )
            return []

    async def _warmup_async(self) -> None:
        """Pre-initialize embedding model and HF pipelines (async wrapper)."""
        await asyncio.to_thread(self._warmup_models_sync)

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
        """
        🎯 AGENT-INDEPENDENT TURN LOGGING

        Logs any turn from any agent with consistent tone analysis.
        Automatically normalizes speaker values and extracts agent identity.
        """
        turn_id = str(uuid.uuid4())

        # ── Global safety cap for stored content & embeddings ────────────────
        content = content or ""
        try:
            max_chars = int(
                CONVERSATION_MAX_CONTENT_CHARS
            ) if CONVERSATION_MAX_CONTENT_CHARS else None
        except Exception:
            max_chars = None
        original_chars = len(content)
        safe_content = (
            content[:max_chars]
            if (isinstance(content, str) and max_chars and original_chars > max_chars)
            else content
        )
        # annotate truncation in metadata (without schema changes)
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
            except Exception:
                meta["content_original_chars"] = original_chars
                meta["content_stored_chars"] = len(safe_content)
                meta["content_max_chars"] = max_chars
            logger.debug(
                "Turn content truncated from %s to %s chars (cap=%s)",
                original_chars,
                len(safe_content),
                max_chars
            )
        # 🎯 APPLY TONE ANALYSIS (off the event loop)
        # Models are already loaded during __init__, so this is fast
        tone_analysis = await asyncio.to_thread(
            self.analyze_turn_tone,
            safe_content,
            speaker,
            agent_name
        )

        # Generate embedding for semantic search (off the event loop)
        embedding = await asyncio.to_thread(
            self.generate_embedding,
            safe_content,
        )

        # Use graph-provided turn_number (LangGraph thread state = source of truth)
        if not isinstance(turn_number, int) or turn_number < 1:
            logger.warning("Invalid turn_number %r; defaulting to 1", turn_number)
            turn_number = 1

        # Create turn object with tone analysis
        turn = ConversationTurn(
            conversation_id=conversation_id,
            turn_id=turn_id,
            turn_number=turn_number,
            timestamp=datetime.now(timezone.utc).isoformat(),
            speaker=speaker,
            content=safe_content,
            message_type=message_type,
            metadata=meta or {},
            embedding=embedding,
            langsmith_trace_id=langsmith_trace_id,
            error_details=error_details or {},  # Never NULL - empty dict for successful turns
            # Context tracking
            memory_token_length=memory_token_length,
            full_context_content=full_context_content,
            # 🎯 TONE ANALYSIS: Applied to every turn regardless of agent
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

            thumbs_up=None,  # Will be set by user reactions
            thumbs_down=None,  # Will be set by user reactions
            # LangSmith feedback scoring - provide defaults to prevent NULL validation errors
            langsmith_feedback_score=0.5,  # Default neutral score
            feedback_type="neutral",  # Default feedback type
            evaluation_details="{}"  # Empty JSON object instead of NULL
        )

        # Store via PM (dedup on turn_id)
        await self._store_turn(turn)

        # Optionally store associated LangSmith trace (when present)
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
            except Exception as e:
                logger.warning("⚠️ Failed to store LangSmith trace for turn %s: %s", turn_id, e)


        logger.debug(
            "Logged turn %s for conversation %s",
            turn_id,
            conversation_id,
            )
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
        except Exception:
            run_id = conversation_id

        # Offload blocking PM call from the event loop.
        # Best-effort only: failures here must not break callers.
        try:
            await asyncio.to_thread(
                self.pm.log_cognitive_error,
                run_id=run_id,
                module="conversation",
                error_type=error_type,
                error_message=error_message,
                operation_context=error_data,
            )
        except Exception as exc:  # noqa: BLE001
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

        # Store via PM (off the event loop)
        table_id = (
            f"{self.project_id}.{self.dataset_id}.conversation_metrics"
            if self.project_id and self.dataset_id
            else "conversation_metrics"
        )
        await asyncio.to_thread(
            self.pm.insert_rows,
            table_id,
            [metric_data],
            [metric_id],
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
        """
        Conversation lifecycle hook (currently a no-op).

        TODO: When we formalise "conversation sessions" at the swarm level,
        this is where we should:
          • Persist a lifecycle row (conversation_id, start_time, end_time, status).
          • Apply the 5-minute idle handoff rule:
                - When control is handed from chat_pro → ReflectiveSqlAgent
                  (or any future specialist), mark the old conversation as
                  ended and start a new conversation_id for the new agent.
                - Likewise when control is handed back to chat_pro.
          • Allow explicit "end" signals from UX (e.g. user clicks "End session").
        
        For now, logging remains purely turn-based and status is inferred
        from turns/metrics.
        """
        return

    async def search_conversations(
            self,
            query: str,
            limit: int = 10,
            conversation_type: Optional[str] = None,
            focus_area: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Delegate semantic search to the PM. (PM can implement vector search
        against stored turn embeddings and any additional filters.)
        """
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
        """
        Delegate "summary view" construction to PM (which already implements it).
        This keeps the logger focused on building artifacts, not querying stores.
        """
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
        await asyncio.to_thread(
            self.pm.insert_rows,
            table_id,
            [data],
            [data["conversation_id"]],
        )

    async def _store_turn(
            self,
            turn: ConversationTurn
    ):
        """Store a conversation turn via PM."""
        # Avoid dataclasses.asdict here – it deep-copies nested structures and
        # blows up on objects without __reduce__ (e.g. uvloop.loop.Loop).
        data = turn.to_row()

        # metadata / error_details are stored as STRING columns; JSON-encode them.
        if data.get("metadata"):
            data["metadata"] = json.dumps(data["metadata"])
        else:
            data["metadata"] = None

        if data.get("error_details"):
            data["error_details"] = json.dumps(data["error_details"])
        else:
            data["error_details"] = None

        # nuanced_emotions: BigQuery JSON column.
        # The streaming API expects a JSON *literal* (usually a string),
        # not a RECORD/STRUCT. So we:
        #   - keep None as None
        #   - validate string values as JSON
        #   - json.dumps() any dict/list into a JSON string
        ne = data.get("nuanced_emotions")
        try:
            if ne is None:
                data["nuanced_emotions"] = None
                logger.debug(
                    "ConversationLogger._store_turn nuanced_emotions is None "
                    "for turn_id=%s",
                    data.get("turn_id"),
                )
            elif isinstance(ne, str):
                # ensure it's valid JSON; if not, we'll drop it below
                json.loads(ne)
                data["nuanced_emotions"] = ne
                logger.debug(
                    "ConversationLogger._store_turn nuanced_emotions (str) sample=%s",
                    ne[:500],
                )
            else:
                # dict / list / other JSON-serialisable type
                json_str = json.dumps(ne, default=_sanitize_for_json)  # reuse sanitizer
                data["nuanced_emotions"] = json_str
                logger.debug(
                    "ConversationLogger._store_turn nuanced_emotions (obj) type=%s sample=%s",
                    type(ne),
                    json_str[:500],
                )
        except Exception as exc:
            # If anything goes wrong, don't block the graph – just drop the field.
            logger.warning(
                "ConversationLogger._store_turn: failed to serialise nuanced_emotions; "
                "dropping field for turn_id=%s: %s",
                data.get("turn_id"),
                exc,
                exc_info=True,
            )
            data["nuanced_emotions"] = None

        # Store via PM (off the event loop)
        table_id = (
            f"{self.project_id}.{self.dataset_id}.conversation_turns"
            if self.project_id and self.dataset_id
            else "conversation_turns"
        )
        await asyncio.to_thread(
            self.pm.insert_rows,
            table_id,
            [data],
            [data["turn_id"]],
        )
