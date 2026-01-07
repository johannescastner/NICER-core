# /src/logging/conversation_logger.py
"""
Comprehensive Conversation Logging System (agent-agnostic)

This module is now persistence-agnostic. It focuses on:
1) building per-turn artifacts (tone analysis, embeddings),
2) assembling structured payloads,
3) delegating all storage/search/summary ops to the Persistence Manager (PM).

✅ CLOUD RUN JOB INTEGRATION: Models download in separate job with dedicated resources
✅ LAZY IMPORTS: PyTorch/transformers only imported when first needed (not at module load)
✅ DEPRECATION FIX: Uses top_k=None instead of deprecated return_all_scores=True
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

# ============================================================================
# CRITICAL FIX: REMOVED TOP-LEVEL IMPORT
# ============================================================================
# The following import was causing 180+ second startup times:
#
#   from transformers import pipeline  # DON'T DO THIS!
#
# Even though pipeline() was only called lazily inside methods, the import
# statement itself pulls in PyTorch at module load time.
#
# Now `from transformers import pipeline` is done inside _ensure_tone_pipelines()
# and other methods that actually need it.
# ============================================================================

# SentenceTransformer is lighter but still pulls some deps - consider lazy-loading too
# if startup is still slow. For now, it's less problematic than transformers.
from sentence_transformers import SentenceTransformer

from pro.persistence import get_persistence_manager
from pro.monitoring.langsmith_integration import get_langsmith_integration
from pro.utils.gcs_model_cache import get_model_cache
from src.langgraph_slack.config import (
    PROJECT_ID,
    LANGSMITH_PROJECT,
    HF_CACHE_DIR,
    CONVERSATION_MAX_CONTENT_CHARS,
    SERVICE_ACCOUNT_EMAIL,
    GCP_SERVICE_ACCOUNT_BASE64,
    GCP_REGION,
    CREDENTIALS
)

# Set up logging
logger = logging.getLogger(__name__)

# ---------------- Tone analysis token caps ----------------
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
    # Last resort: if it's already JSON-serializable, keep it; otherwise repr(...)
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError, OverflowError):
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
# CLOUD RUN JOB INTEGRATION
# ============================================================================

def trigger_model_download_job() -> bool:
    """
    Trigger Cloud Run Job to download models to GCS.
    
    ✅ Uses CREDENTIALS from config (already extracted)
    ✅ Self-healing: creates job if missing
    ✅ Simple and correct
    """
    try:
        project_id = os.getenv("GCP_PROJECT_ID")
        region = GCP_REGION
        job_name = "model-downloader"
        
        if not project_id:
            logger.warning("⚠️  GCP_PROJECT_ID not set")
            return False
        
        logger.info(f"🚀 Triggering Cloud Run Job: {job_name}")
        
        from google.cloud import run_v2
        from google.api_core import exceptions as gcp_exceptions
        
        # ✅ Use extracted CREDENTIALS from config
        client = run_v2.JobsClient(credentials=CREDENTIALS)
        job_path = f"projects/{project_id}/locations/{region}/jobs/{job_name}"
        
        # Check if job exists
        job_exists = False
        try:
            client.get_job(name=job_path)
            job_exists = True
            logger.info(f"✅ Job exists: {job_name}")
            
        except (gcp_exceptions.NotFound, gcp_exceptions.PermissionDenied):
            logger.info(f"📦 Job not found, creating...")
            job_exists = _create_job_automatically(
                client=client,
                project_id=project_id,
                region=region,
                job_name=job_name
            )
            
            if not job_exists:
                logger.warning("⚠️  Failed to create job - models will load on-demand")
                return False
        
        if not job_exists:
            return False
        
        # Trigger job execution
        request = run_v2.RunJobRequest(name=job_path)
        client.run_job(request=request)
        
        logger.info(f"✅ Job triggered successfully")
        logger.info(f"   Console: https://console.cloud.google.com/run/jobs/{region}/{job_name}?project={project_id}")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️  Failed to trigger job: {e}")
        return False


def _create_job_automatically(
    client,
    project_id: str,
    region: str,
    job_name: str
) -> bool:
    """
    Create Cloud Run Job.
    
    ✅ Job runs as SERVICE_ACCOUNT_EMAIL - inherits identity via ADC
    ✅ No explicit credentials needed in env vars
    ✅ Only passes necessary config (project_id, region)
    """
    try:
        from google.cloud import run_v2
        
        if not SERVICE_ACCOUNT_EMAIL:
            logger.error("   ❌ SERVICE_ACCOUNT_EMAIL not available")
            return False
        
        logger.info(f"   Job will run as: {SERVICE_ACCOUNT_EMAIL}")
        
        image_name = f"{region}-docker.pkg.dev/{project_id}/jobs/{job_name}"
        logger.info(f"   Using image: {image_name}")
        
        # ✅ Job inherits SA identity - only needs config, not credentials!
        env_vars = [
            run_v2.EnvVar(name="GCP_PROJECT_ID", value=project_id),
            run_v2.EnvVar(name="GCP_REGION", value=region),
        ]
        
        job = run_v2.Job(
            template=run_v2.ExecutionTemplate(
                template=run_v2.TaskTemplate(
                    containers=[
                        run_v2.Container(
                            image=image_name,
                            env=env_vars,
                            resources=run_v2.ResourceRequirements(
                                limits={"cpu": "4", "memory": "8Gi"}
                            )
                        )
                    ],
                    max_retries=0,
                    timeout="1800s",
                    service_account=SERVICE_ACCOUNT_EMAIL  # ✅ Gives job its identity
                )
            )
        )
        
        parent = f"projects/{project_id}/locations/{region}"
        request = run_v2.CreateJobRequest(
            parent=parent,
            job=job,
            job_id=job_name
        )
        
        logger.info(f"   Creating job...")
        operation = client.create_job(request=request)
        
        # ✅ Wait for creation to complete
        try:
            operation.result(timeout=60)
            logger.info(f"   ✅ Job created successfully")
            return True
        except Exception as e:
            logger.error(f"   ❌ Creation failed: {e}")
            return False
        
    except Exception as e:
        logger.error(f"   ❌ Failed: {e}", exc_info=True)
        return False


def check_models_in_gcs() -> tuple[int, int]:
    """
    Check how many models are already in GCS.
    
    Returns:
        Tuple of (models_in_gcs, total_models)
    """
    try:
        cache = get_model_cache()
        if not cache:
            logger.info("⚠️  GCS cache not available")
            return (0, 0)
        
        models = [
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "SamLowe/roberta-base-go_emotions",
            "s-nlp/roberta-base-formality-ranker",
            "sentence-transformers/all-MiniLM-L6-v2",
        ]
        
        # Check each model
        models_in_gcs = 0
        for model in models:
            if cache.check_model_exists_in_gcs(model):
                models_in_gcs += 1
        
        return (models_in_gcs, len(models))
        
    except Exception as e:
        logger.warning(f"⚠️  Failed to check models in GCS: {e}")
        return (0, 0)


def download_models_from_gcs_background(models: List[str], cache):
    """
    Download models from GCS in background thread.
    
    This is fast (30 seconds total) and won't block deployment.
    """
    import time
    
    def _download():
        for model in models:
            try:
                logger.info(f"📥 Downloading {model} from GCS...")
                start = time.time()
                
                if cache.download_from_gcs(model):
                    elapsed = time.time() - start
                    logger.info(f"✅ Downloaded {model} in {elapsed:.1f}s")
                else:
                    logger.warning(f"⚠️  Failed to download {model} from GCS")
                    
            except Exception as e:
                logger.warning(f"⚠️  Error downloading {model}: {e}")
    
    # Start download in daemon thread (won't block shutdown)
    thread = threading.Thread(target=_download, daemon=True, name="GCSModelDownloader")
    thread.start()
    logger.info("🚀 Started GCS model download in background")


def download_missing_models_immediately(models: List[str], cache):
    """
    Download missing models from HuggingFace immediately in background thread.
    
    ✅ SELF-HEALING: Downloads missing models in current container
    ✅ Uploads to GCS for next deployment
    ✅ No manual intervention needed
    ✅ Works for thousands of clients automatically
    
    This ensures the current deployment has models ready, not just future deployments.
    Models are also uploaded to GCS for next deployment.
    """
    import time
    from pathlib import Path
    
    def _download():
        for model in models:
            try:
                # Check if already in local cache
                safe_model = model.replace("/", "--")
                model_dir = Path(cache.cache_dir) / "hub" / f"models--{safe_model}"
                
                if model_dir.exists():
                    logger.info(f"✅ {model} already in local cache")
                    continue
                
                # Check if in GCS (validated) - download from GCS first (faster)
                if cache.check_model_exists_in_gcs(model):
                    logger.info(f"📥 Downloading {model} from GCS...")
                    start = time.time()
                    if cache.download_from_gcs(model):
                        elapsed = time.time() - start
                        logger.info(f"✅ Downloaded {model} from GCS in {elapsed:.1f}s")
                        continue
                
                # Not in GCS or incomplete - download from HuggingFace
                logger.info(f"📥 Downloading {model} from HuggingFace...")
                start = time.time()
                
                # ✅ DETECT MODEL TYPE AND USE CORRECT LIBRARY
                if model.startswith("sentence-transformers/"):
                    from sentence_transformers import SentenceTransformer
                    SentenceTransformer(model, cache_folder=cache.cache_dir)
                else:
                    # ✅ LAZY IMPORT: Only import transformers when actually needed
                    from transformers import AutoModel, AutoTokenizer
                    AutoModel.from_pretrained(model, cache_dir=cache.cache_dir)
                    AutoTokenizer.from_pretrained(model, cache_dir=cache.cache_dir)
                
                elapsed = time.time() - start
                logger.info(f"✅ Downloaded {model} from HuggingFace in {elapsed:.1f}s")
                
                # Upload to GCS for next deployment
                logger.info(f"📤 Uploading {model} to GCS for future deployments...")
                cache.upload_to_gcs(model)
                
            except Exception as e:
                logger.error(f"❌ Failed to download {model}: {e}", exc_info=True)
    
    # Start download in background thread (won't block startup)
    thread = threading.Thread(target=_download, daemon=True, name="HFModelDownloader")
    thread.start()
    logger.info("🚀 Started HuggingFace model download in background")


def initialize_model_cache_strategy():
    """
    Initialize model caching strategy with IMMEDIATE download on corruption.
    
    ✅ SELF-HEALING Strategy:
    1. Check if models are in GCS
    2. If all present: Download from GCS (fast, 30s total)
    3. If missing/corrupt: Download immediately from HuggingFace (this container)
    4. Also trigger Cloud Run Job for redundancy
    
    This ensures:
    - Deployment NEVER times out (job runs separately)
    - Models are ready IMMEDIATELY (no 1-deployment lag)
    - Works for thousands of clients automatically
    """
    try:
        cache = get_model_cache()
        if not cache:
            logger.info("⚠️  GCS cache not available, models will load on-demand")
            return
        
        # Check which models are in GCS
        models_in_gcs, total_models = check_models_in_gcs()
        
        logger.info(f"📦 Model cache status: {models_in_gcs}/{total_models} models in GCS")
        
        # Define models list
        models = [
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "SamLowe/roberta-base-go_emotions",
            "s-nlp/roberta-base-formality-ranker",
            "sentence-transformers/all-MiniLM-L6-v2",
        ]
        
        if models_in_gcs == total_models:
            # ✅ All models validated and present in GCS - download them (fast!)
            logger.info("✅ All models found in GCS cache")
            logger.info("📥 Downloading models from GCS in background...")
            download_models_from_gcs_background(models, cache)
            
        elif models_in_gcs > 0:
            # ⚠️ Some models missing/incomplete - IMMEDIATE DOWNLOAD
            logger.info(f"⚠️  Partial cache: {models_in_gcs}/{total_models} models in GCS")
            logger.info("📥 Downloading missing models from HuggingFace immediately...")
            
            # ✅ Download missing models NOW in current container
            download_missing_models_immediately(models, cache)
            
            # Also trigger job for redundancy/next time
            logger.info("🚀 Triggering Cloud Run Job for redundancy...")
            trigger_model_download_job()
            
        else:
            # ❌ No models cached at all (first deployment)
            logger.info("📥 No models in GCS cache (first deployment)")
            logger.info("📥 Downloading all models from HuggingFace immediately...")
            
            # ✅ Download all models NOW in current container
            download_missing_models_immediately(models, cache)
            
            # Trigger job for next deployment
            logger.info("🚀 Triggering Cloud Run Job for next deployment...")
            trigger_model_download_job()
        
    except Exception as e:
        logger.error(f"❌ Error in initialize_model_cache_strategy: {e}", exc_info=True)
        logger.info("⚠️  Continuing startup - models will load on-demand")


# ============================================================================
# CONVERSATION LOGGER CLASS
# ============================================================================

class ConversationLogger:
    """
    Comprehensive conversation logging with Cloud Run Job integration.
    
    Models are downloaded asynchronously in a separate Cloud Run Job,
    ensuring fast deployment and no timeout issues.
    """
    def __init__(
            self,
            project_id: str = PROJECT_ID or "",
            dataset_id: str = "conversation_logs_eu",
            langsmith_project: str = LANGSMITH_PROJECT,
            pm=None
    ):
        """Initialize the conversation logger with Cloud Run Job integration."""
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.langsmith_project = langsmith_project

        # Persistence Manager
        self.pm = pm or get_persistence_manager()

        # Tone analysis token cap
        self._tone_max_length: int = _get_tone_max_length_from_env()
        logger.info("ConversationLogger tone max_length=%s (env=%s)",
                    self._tone_max_length, _TONE_MAX_LENGTH_ENV)

        # HF cache root
        self._hf_cache_dir = HF_CACHE_DIR
        if self._hf_cache_dir:
            try:
                os.makedirs(self._hf_cache_dir, exist_ok=True)
            except OSError as e:
                logger.warning("HF cache dir %s not creatable: %s", self._hf_cache_dir, e)

        # Lazy model caches
        self._embedding_model: Optional[SentenceTransformer] = None
        self._sentiment_classifier = None
        self._formality_classifier = None
        self._emotion_classifier = None

        # Thread-safety
        self._embed_lock = threading.Lock()
        self._pipeline_lock = threading.Lock()

        # ✅ CLOUD RUN JOB INTEGRATION
        logger.info("🚀 Initializing model cache strategy...")
        initialize_model_cache_strategy()

        logger.info("ConversationLogger initialized for project %s", self.project_id)

    def _ensure_embedding_model(self) -> SentenceTransformer:
        """
        Lazy-load embedding model.
        
        If model is in GCS, it will be downloaded quickly.
        Otherwise, it will be downloaded from HuggingFace on first use.
        """
        if self._embedding_model is None:
            with self._embed_lock:
                if self._embedding_model is None:
                    try:
                        # Check GCS cache first
                        cache = get_model_cache()
                        if cache:
                            cache.ensure_model_cached("sentence-transformers/all-MiniLM-L6-v2")
                        
                        # Load model (uses local cache if available)
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
                        
                        # Cache to GCS for next deployment
                        if cache:
                            cache.cache_model_after_download("sentence-transformers/all-MiniLM-L6-v2")
                            
                    except (OSError, RuntimeError, ImportError) as e:
                        logger.warning("⚠️ Failed to init embedding model: %s", e, exc_info=True)
                        raise
        return self._embedding_model

    def _ensure_tone_pipelines(self) -> None:
        """
        Lazy-load tone analysis pipelines.
        
        ✅ CRITICAL FIX #1: `from transformers import pipeline` is now done HERE,
        not at module load time. This prevents 180+ second startup delays.
        
        ✅ CRITICAL FIX #2: Uses `top_k=None` instead of deprecated `return_all_scores=True`
        to eliminate the deprecation warning.
        
        If models are in GCS cache (from previous deployment),
        they load quickly. Otherwise, they download from HuggingFace on first use.
        """
        if self._sentiment_classifier and self._formality_classifier and self._emotion_classifier:
            return

        with self._pipeline_lock:
            pipe_kwargs = {}
            if self._hf_cache_dir:
                pipe_kwargs = {
                    "model_kwargs": {"cache_dir": self._hf_cache_dir},
                    "tokenizer_kwargs": {"cache_dir": self._hf_cache_dir},
                }
            
            cache = get_model_cache()
            
            if self._sentiment_classifier is None:
                try:
                    # ✅ LAZY IMPORT: Only import when first needed
                    from transformers import pipeline
                    
                    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
                    if cache:
                        cache.ensure_model_cached(model_name)
                    
                    # ✅ FIX: Use top_k=None instead of deprecated return_all_scores=True
                    self._sentiment_classifier = pipeline(
                        "sentiment-analysis",
                        model=model_name,
                        top_k=None,  # Returns all scores (replaces return_all_scores=True)
                        **pipe_kwargs,
                    )
                    logger.info("✅ Sentiment pipeline ready")
                    
                    if cache:
                        cache.cache_model_after_download(model_name)
                        
                except (OSError, RuntimeError, ImportError, ValueError) as e:
                    logger.warning("⚠️ Sentiment pipeline load failed: %s", e, exc_info=True)
            
            if self._formality_classifier is None:
                try:
                    # ✅ LAZY IMPORT: Only import when first needed
                    from transformers import pipeline
                    
                    model_name = "s-nlp/roberta-base-formality-ranker"
                    if cache:
                        cache.ensure_model_cached(model_name)
                    
                    # ✅ FIX: Use top_k=None instead of deprecated return_all_scores=True
                    self._formality_classifier = pipeline(
                        "text-classification",
                        model=model_name,
                        top_k=None,  # Returns all scores (replaces return_all_scores=True)
                        **pipe_kwargs,
                    )
                    logger.info("✅ Formality pipeline ready")
                    
                    if cache:
                        cache.cache_model_after_download(model_name)
                        
                except (OSError, RuntimeError, ImportError, ValueError) as e:
                    logger.warning("⚠️ Formality pipeline load failed: %s", e, exc_info=True)
            
            if self._emotion_classifier is None:
                try:
                    # ✅ LAZY IMPORT: Only import when first needed
                    from transformers import pipeline
                    
                    model_name = "SamLowe/roberta-base-go_emotions"
                    if cache:
                        cache.ensure_model_cached(model_name)
                    
                    # ✅ FIX: Use top_k=None instead of deprecated return_all_scores=True
                    self._emotion_classifier = pipeline(
                        "text-classification",
                        model=model_name,
                        top_k=None,  # Returns all scores (replaces return_all_scores=True)
                        **pipe_kwargs,
                    )
                    logger.info("✅ Emotion pipeline ready")
                    
                    if cache:
                        cache.cache_model_after_download(model_name)
                        
                except (OSError, RuntimeError, ImportError, ValueError) as e:
                    logger.warning("⚠️ Emotion pipeline load failed: %s", e, exc_info=True)

    def analyze_turn_tone(
            self,
            content: str,
            speaker: str,
            agent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        🎯 Unified tone analysis for ANY turn (human or any agent).
        
        Models load on-demand. If models are in GCS cache (from previous deployment),
        they load quickly. Otherwise, they download from HuggingFace on first use.
        """
        # Ensure tone pipelines are available
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

        # Truncate for model stability
        text = content[:2000] if len(content) > 2000 else content

        try:
            # Sentiment analysis
            if self._sentiment_classifier:
                s_res = self._sentiment_classifier(
                    text,
                    truncation=True,
                    max_length=self._tone_max_length,
                )
                candidates = []
                if isinstance(s_res, list) and s_res:
                    first = s_res[0]
                    if isinstance(first, list):
                        candidates = first
                    elif isinstance(first, dict):
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
                    else:
                        tone["sentiment_score"] = 0.5

            # Formality analysis
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
                    lbl = str(best.get("label", "")).upper()
                    conf = float(best.get("score", 0.0))
                    tone["formality_label"] = lbl
                    tone["formality_confidence"] = conf
                    tone["formality_score"] = conf if lbl == "FORMAL" else (1.0 - conf)

            # Emotion analysis
            if self._emotion_classifier:
                e_res = self._emotion_classifier(
                    text,
                    truncation=True,
                    max_length=self._tone_max_length,
                )
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
                        "scores": scores_map
                    }

            # Professionalism heuristic
            if tone["formality_score"] is not None and tone["sentiment_score"] is not None:
                prof = (tone["formality_score"] * 0.7) + (tone["sentiment_score"] * 0.3)
                tone["professional_score"] = prof
                tone["professional_label"] = (
                    "PROFESSIONAL" if prof >= 0.7 else
                    "CASUAL" if prof >= 0.4 else
                    "INAPPROPRIATE"
                )

        except (RuntimeError, ValueError, TypeError) as exc:
            logger.warning("⚠️ Tone analysis failed: %s", exc)

        return tone

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate semantic embedding for text.
        
        Model loads on-demand. If model is in GCS cache, it loads quickly.
        """
        try:
            model = self._ensure_embedding_model()
            embedding = model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except (RuntimeError, ValueError, OSError) as e:
            logger.error("Failed to generate embedding: %s", e)
            return []

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

        # Apply tone analysis (off event loop)
        tone_analysis = await asyncio.to_thread(
            self.analyze_turn_tone,
            safe_content,
            speaker,
            agent_name
        )

        # Generate embedding
        embedding = await asyncio.to_thread(
            self.generate_embedding,
            safe_content,
        )

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
            embedding=embedding,
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
            await asyncio.to_thread(
                self.pm.log_cognitive_error,
                run_id=run_id,
                module="conversation",
                error_type=error_type,
                error_message=error_message,
                operation_context=error_data,
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
        await asyncio.to_thread(
            self.pm.insert_rows,
            table_id,
            [data],
            [data["turn_id"]],
        )