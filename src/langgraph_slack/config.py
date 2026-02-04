# src/langgraph_slack/config.py
"""
This is where globals are defined and configured.
"""
import logging
import os
import base64
from typing import Optional
import json
import importlib
import re
from os import environ
from google.oauth2 import service_account

from .deepseek_utils import (
    wrap_llm_with_deepseek_backoff,
    make_balance_checker,
    BackoffConfig,
)

LOGGER = logging.getLogger(__name__)

ENVIRONMENT = (environ.get("ENVIRONMENT", "PROD") or "PROD").upper()
LOGGER.info("🔍 MPLCONFIGDIR env var: %s", os.environ.get("MPLCONFIGDIR"))
LOGGER.info("🔍 MPLBACKEND env var: %s", os.environ.get("MPLBACKEND"))
if DEPLOY_MODAL := environ.get("DEPLOY_MODAL"):
    DEPLOY_MODAL = DEPLOY_MODAL.lower() == "true"

BOT_USER_ID = environ.get("SLACK_BOT_USER_ID")
BOT_TOKEN = environ.get("SLACK_BOT_TOKEN")

def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "tenant"
# Company configuration
COMPANY = environ.get("COMPANY", "Towards People")
# Tenant + region
# IMPORTANT:
#  - GCP_LOCATION here is your BigQuery location (EU/US multi-region label).
#  - Cloud Run/Cloud SQL/Redis require a *region* like europe-west1.
#
# If you don't supply SUPERSET_REGION explicitly, we fall back to:
#   GCP_REGION -> SUPERSET_REGION -> europe-west1
#
GCP_REGION = environ.get("GCP_REGION", "europe-west1")  # optional, if you start standardizing this
SUPERSET_TENANT = environ.get("SUPERSET_TENANT") or _slugify(COMPANY)
SUPERSET_REGION = (
    environ.get("SUPERSET_REGION")
    or GCP_REGION
    or "europe-west1"
)
LOCATION = environ.get("GCP_LOCATION", GCP_REGION)

if DEPLOY_MODAL:
    if not environ.get("SLACK_BOT_TOKEN"):
        environ["SLACK_BOT_TOKEN"] = "fake-token"
    BOT_USER_ID = BOT_USER_ID or "fake-user-id"
else:
    if BOT_TOKEN is None:
        LOGGER.warning("BOT_TOKEN not set - will be provided dynamically by router")

LANGGRAPH_URL = environ.get("LANGGRAPH_URL")
ASSISTANT_ID = environ.get("LANGGRAPH_ASSISTANT_ID", "default_assistant_id")
CONFIG = environ.get("CONFIG") or "{}"
DEPLOYMENT_URL = environ.get("DEPLOYMENT_URL", "")
SLACK_CHANNEL_ID = environ.get("SLACK_CHANNEL_ID")

# Google Cloud project details
PROJECT_ID = environ.get("GCP_PROJECT_ID", "default_project_id")
DATASET_ID = environ.get("GCP_DATASET_ID", "agent_system_memory")

# Table names for different memory types
SEMANTIC_TABLE = environ.get("GCP_SEMANTIC_TABLE", "semantic_memory")
EPISODIC_TABLE = environ.get("GCP_EPISODIC_TABLE", "episodic_memory")
PROCEDURAL_TABLE = environ.get("GCP_PROCEDURAL_TABLE", "procedural_memory")
# Raw base64 service account (for passing to Cloud Run Jobs)
GCP_SERVICE_ACCOUNT_BASE64 = environ.get("GCP_SERVICE_ACCOUNT_BASE64", "")
# Service account credentials
try:
    service_account_json = base64.b64decode(
        GCP_SERVICE_ACCOUNT_BASE64
    ).decode("utf-8")
    SERVICE_ACCOUNT_INFO: dict = json.loads(service_account_json)
    CREDENTIALS = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO)
    LOGGER.info("Service account credentials loaded successfully.")
except Exception as e:
    LOGGER.error(
        "Failed to load service account credentials: %s",
        e,
        exc_info=True
    )
    SERVICE_ACCOUNT_INFO = {}
    CREDENTIALS = None

# ───────────────────── CollectiWise Central Service Account ─────────────────────
CW_SERVICE_ACCOUNT_BASE64 = environ.get("CW_SERVICE_ACCOUNT_BASE64", "")

# Fallback logic with appropriate warnings
if not CW_SERVICE_ACCOUNT_BASE64:
    CW_SERVICE_ACCOUNT_BASE64 = GCP_SERVICE_ACCOUNT_BASE64
    if GCP_SERVICE_ACCOUNT_BASE64:
        LOGGER.warning(
            "⚠️  CW_SERVICE_ACCOUNT_BASE64 not set - falling back to GCP_SERVICE_ACCOUNT_BASE64. "
            "This is only valid for canary or dev deployments."
        )

# Parse CollectiWise credentials
try:
    if CW_SERVICE_ACCOUNT_BASE64:
        cw_service_account_json = base64.b64decode(CW_SERVICE_ACCOUNT_BASE64).decode("utf-8")
        CW_SERVICE_ACCOUNT_INFO: dict = json.loads(cw_service_account_json)
        CW_CREDENTIALS = service_account.Credentials.from_service_account_info(CW_SERVICE_ACCOUNT_INFO)
        CW_PROJECT_ID = CW_SERVICE_ACCOUNT_INFO.get("project_id", "collectiwise-nicer")
    else:
        CW_SERVICE_ACCOUNT_INFO = {}
        CW_CREDENTIALS = None
        CW_PROJECT_ID = "collectiwise-nicer"
except Exception as e:
    LOGGER.error("Failed to load CollectiWise central SA: %s", e)
    CW_CREDENTIALS = None
    CW_PROJECT_ID = "collectiwise-nicer"

# Location-aware dataset naming
def get_regional_dataset_suffix(location: str) -> str:
    return location.lower().replace("-", "_")

CENTRAL_DATASET_SUFFIX = get_regional_dataset_suffix(LOCATION)
CENTRAL_DATASET_ID = f"agent_cognitive_processes_{CENTRAL_DATASET_SUFFIX}"
# Best-effort derive the SA email from embedded credentials.
# This is useful for provisioning scripts that need to grant Secret Manager access
# to the runtime identity without forcing clients to repeat themselves.
SERVICE_ACCOUNT_EMAIL = (
    SERVICE_ACCOUNT_INFO.get("client_email")
    if isinstance(SERVICE_ACCOUNT_INFO, dict)
    else None
)

if SERVICE_ACCOUNT_EMAIL:
    LOGGER.info("Derived service account email from GCP_SERVICE_ACCOUNT_BASE64: %s",
                SERVICE_ACCOUNT_EMAIL)
else:
    LOGGER.warning(
        "SERVICE_ACCOUNT_EMAIL could not be derived from GCP_SERVICE_ACCOUNT_BASE64; "
        "Superset deployment will fail if this is required."
    )

# ───────────────────── Superset Configuration ─────────────────────
# Service account used by the Superset Cloud Run service at runtime.
# If not explicitly set, fall back to the embedded SA email.
SUPERSET_RUN_SERVICE_ACCOUNT = (
    environ.get("SUPERSET_RUN_SERVICE_ACCOUNT")
    or SERVICE_ACCOUNT_EMAIL
    or ""
)

# ───────────────────── Superset Configuration ─────────────────────
#
# We treat Superset as a sibling service to the LangGraph deployment.
# The agent will use a Superset tool to create/update virtual datasets
# and charts, returning URLs when a visual answer is best.
#
# Superset architecture (high level):
#   - Metadata DB (Postgres/MySQL)
#   - Cache/Results backend (often Redis)
#   - SQL engines (BigQuery, etc.) accessed via SQLAlchemy
#
# This module only reads env; provisioning is handled by an external script.


# Enable/disable Superset integration at runtime
SUPERSET_ENABLED = (environ.get("SUPERSET_ENABLED", "true").lower() == "true")


# URL of the Superset instance (set after deploy)
SUPERSET_URL = environ.get("SUPERSET_URL", "")

# Secret Manager secret IDs (used by provisioning script).
# These are names/IDs, not secret values.
# Google OAuth secret IDs (used by provisioning script)
# These map to container env vars GOOGLE_KEY/GOOGLE_SECRET per Superset docs.
SUPERSET_GOOGLE_KEY_SECRET_NAME = environ.get(
    "SUPERSET_GOOGLE_KEY_SECRET_NAME",
    f"superset-google-key--{SUPERSET_TENANT}",
)
SUPERSET_GOOGLE_SECRET_SECRET_NAME = environ.get(
    "SUPERSET_GOOGLE_SECRET_SECRET_NAME",
    f"superset-google-secret--{SUPERSET_TENANT}",
)

# Optional hosted-domain restriction for Google OAuth
SUPERSET_OAUTH_AUTH_DOMAIN = environ.get("SUPERSET_OAUTH_AUTH_DOMAIN", "")

# Auth mode runtime toggle (db|google)
SUPERSET_AUTH_TYPE = (environ.get("SUPERSET_AUTH_TYPE", "db") or "db").lower().strip()

SUPERSET_SECRET_KEY_SECRET_NAME = environ.get(
    "SUPERSET_SECRET_KEY_SECRET_NAME",
    f"superset-secret-key--{SUPERSET_TENANT}",
)
SUPERSET_SQL_PASSWORD_SECRET_NAME = environ.get(
    "SUPERSET_SQL_PASSWORD_SECRET_NAME",
    f"superset-sql-password--{SUPERSET_TENANT}",
)
SUPERSET_ADMIN_PASSWORD_SECRET_NAME = environ.get(
    "SUPERSET_ADMIN_PASSWORD_SECRET_NAME",
    f"superset-admin-password--{SUPERSET_TENANT}",
)

# Core Superset secret key (required by Superset)
# In production, generate a strong random key and store in Secret Manager.
SUPERSET_SECRET_KEY = environ.get("SUPERSET_SECRET_KEY", "")

# Admin bootstrap creds for first-time init
SUPERSET_ADMIN_USERNAME = environ.get("SUPERSET_ADMIN_USERNAME", "admin")
SUPERSET_ADMIN_PASSWORD = environ.get("SUPERSET_ADMIN_PASSWORD", "")
SUPERSET_ADMIN_EMAIL = environ.get("SUPERSET_ADMIN_EMAIL", "")

# ---------- Cloud SQL (metadata) ----------
SUPERSET_SQL_INSTANCE_NAME = environ.get(
    "SUPERSET_SQL_INSTANCE_NAME",
    f"superset-{SUPERSET_TENANT}",
)
SUPERSET_SQL_DB = environ.get("SUPERSET_SQL_DB", "superset")
SUPERSET_SQL_USER = environ.get("SUPERSET_SQL_USER", "superset")
SUPERSET_SQL_PASSWORD = environ.get("SUPERSET_SQL_PASSWORD", "")

# ---------- Redis ----------
# Name is for provisioning; host/port are for runtime wiring.
SUPERSET_REDIS_NAME = environ.get(
    "SUPERSET_REDIS_NAME",
    f"superset-redis-{SUPERSET_TENANT}",
)
SUPERSET_REDIS_HOST = environ.get("SUPERSET_REDIS_HOST", "")
SUPERSET_REDIS_PORT = int(environ.get("SUPERSET_REDIS_PORT", "6379"))

# ---------- Image / Service naming ----------
SUPERSET_IMAGE_NAME = environ.get(
    "SUPERSET_IMAGE_NAME",
    f"collectiwise-{SUPERSET_TENANT}",
)
SUPERSET_CLOUDRUN_SERVICE_NAME = environ.get(
    "SUPERSET_CLOUDRUN_SERVICE_NAME",
    f"collectiwise-{SUPERSET_TENANT}",
)

# Optional: BigQuery database id to use inside Superset for tool defaults
# This is a Superset-internal id that your tool can override per call.
SUPERSET_BQ_DATABASE_ID = (
    int(environ["SUPERSET_BQ_DATABASE_ID"])
    if environ.get("SUPERSET_BQ_DATABASE_ID")
    else None
)

# Minimal validation helpers (non-fatal)
def superset_config_ok() -> bool:
    """
    SECRET_KEY is mandatory for Superset to boot correctly.
    """
    if not SUPERSET_ENABLED:
        return False
    if not SUPERSET_SECRET_KEY:
        LOGGER.warning("SUPERSET_SECRET_KEY not set.")
    if not PROJECT_ID or PROJECT_ID == "default_project_id":
        LOGGER.warning("GCP_PROJECT_ID not set for Superset provisioning.")
    if SUPERSET_AUTH_TYPE == "google":
        if not SUPERSET_GOOGLE_KEY_SECRET_NAME:
            LOGGER.warning("SUPERSET_GOOGLE_KEY_SECRET_NAME not set.")
        if not SUPERSET_GOOGLE_SECRET_SECRET_NAME:
            LOGGER.warning("SUPERSET_GOOGLE_SECRET_SECRET_NAME not set.")
    return True

# ───── Model Provider Configuration ─────
# Easy switching between OpenAI and DeepSeek models
MODEL_PROVIDERS = {
    "openai": {
        "chat": "gpt-4o",
        "reasoning": "o1-preview",
        "cheap": "gpt-4o-mini",
        "import_class": "ChatOpenAI",
        "import_module": "langchain_openai",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None
    },
    "deepseek": {
        "chat": "deepseek-chat",
        "reasoning": "deepseek-reasoner",
        "cheap": "deepseek-chat",
        "import_class": "ChatDeepSeek",
        "import_module": "langchain_deepseek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1"
    }
}

# Current provider selection (normalize to lowercase for robustness)
CURRENT_PROVIDER = (environ.get("MODEL_PROVIDER", "deepseek") or "deepseek").lower()
CURRENT_MODEL_TYPE = (environ.get("MODEL_TYPE", "chat") or "chat").lower()

# Validate provider
if CURRENT_PROVIDER not in MODEL_PROVIDERS:
    raise ValueError(
        f"Invalid MODEL_PROVIDER: {CURRENT_PROVIDER}."
        f"Must be one of: {list(MODEL_PROVIDERS.keys())}"
    )

# Get current provider config
PROVIDER_CONFIG = MODEL_PROVIDERS[CURRENT_PROVIDER]

# ───── Language-model defaults ─────
DEFAULT_MODEL = f"{CURRENT_PROVIDER}:{PROVIDER_CONFIG[CURRENT_MODEL_TYPE]}"
DEFAULT_TEMPERATURE = float(environ.get("NICER_TEMPERATURE", "0.1"))
DEFAULT_DATASET = "linkedin_raw"

# API configuration for current provider
API_KEY = environ.get(PROVIDER_CONFIG["api_key_env"])
BASE_URL = PROVIDER_CONFIG["base_url"]

# Legacy DeepSeek configuration (for backward compatibility)
DEEPSEEK_API_KEY = environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# ───── Dynamic Model Creation Helper ─────
def create_llm(
    model_type="chat",
    provider=None,
    with_backoff=True,
    **kwargs
):
    """
    Create an LLM instance based on the current provider configuration.
    
    Args:
        model_type: Type of model ("chat", "reasoning", "cheap")
        provider: Override the current provider (optional)
        with_backoff: Apply DeepSeek exponential backoff for rate limiting (default: True)
        **kwargs: Additional parameters to pass to the model constructor
    
    Returns:
        LLM instance configured for the specified provider
    
    Example:
        # Use current provider (from MODEL_PROVIDER env var)
        llm = create_llm("chat", temperature=0.1)
        
        # Override provider
        llm = create_llm("reasoning", provider="openai", temperature=0.0)
        
        # Disable backoff for testing
        llm = create_llm("chat", with_backoff=False)
    """
    # Normalize inputs
    provider = (provider or CURRENT_PROVIDER).lower()
    model_type = (model_type or CURRENT_MODEL_TYPE).lower()
    
    if provider not in MODEL_PROVIDERS:
        raise ValueError(
            f"Invalid provider: {provider}. Must be one of: {list(MODEL_PROVIDERS.keys())}"
        )
    
    config = MODEL_PROVIDERS[provider]
    
    # Import the appropriate class
    module = importlib.import_module(config["import_module"])
    model_class = getattr(module, config["import_class"])
    
    # Validate model_type for this provider
    if model_type not in config:
        valid_types = sorted(k for k in config if k in ("chat", "reasoning", "cheap"))
        raise ValueError(
            f"Invalid model_type `{model_type}` for provider `{provider}`. "
            f"Expected one of: {', '.join(valid_types)}."
        )
    
    # Set up model parameters
    model_params = {
        "model": config[model_type],
        **kwargs
    }
    
    # Add provider-specific configuration (generic)
    if config.get("base_url"):
        model_params["base_url"] = config["base_url"]
    
    api_env = config.get("api_key_env")
    if api_env and environ.get(api_env):
        model_params["api_key"] = environ.get(api_env)
    
    # Create the LLM instance
    llm = model_class(**model_params)
    
    # Apply DeepSeek-specific rate limiting if enabled
    if with_backoff and provider == "deepseek":
        try:
            balance_checker = make_balance_checker(
                api_key=environ.get(config["api_key_env"]),
                base_url=config["base_url"],
            )
            llm = wrap_llm_with_deepseek_backoff(
                llm,
                enable_backoff=True,
                provider=provider,
                cfg=BackoffConfig(),
                balance_checker=balance_checker,
            )
            LOGGER.info(
                "Applied DeepSeek exponential backoff to %s LLM",
                model_class.__name__
            )
        except ImportError:
            LOGGER.warning(
                "DeepSeek utils not available, using %s without rate limiting",
                model_class.__name__
            )
    
    return llm

# ───── Convenience Functions ─────
def get_current_provider_info():
    """Get information about the current model provider."""
    return {
        "provider": CURRENT_PROVIDER,
        "model_type": CURRENT_MODEL_TYPE,
        "model": PROVIDER_CONFIG[CURRENT_MODEL_TYPE],
        "full_model": DEFAULT_MODEL,
        "api_key_set": bool(environ.get(PROVIDER_CONFIG["api_key_env"])),
        "cost_efficiency": "14x cheaper" if CURRENT_PROVIDER == "deepseek" else "standard"
    }

def switch_provider(provider, model_type="chat"):
    """
    Switch to a different provider (for runtime switching).
    Note: This only affects new create_llm() calls, not existing instances.
    """
    global CURRENT_PROVIDER, CURRENT_MODEL_TYPE, PROVIDER_CONFIG, DEFAULT_MODEL
    
    if provider not in MODEL_PROVIDERS:
        raise ValueError(
            f"Invalid provider: {provider}. Must be one of: {list(MODEL_PROVIDERS.keys())}"
        )
    
    CURRENT_PROVIDER = (provider or "").lower()
    CURRENT_MODEL_TYPE = (model_type or "chat").lower()
    PROVIDER_CONFIG = MODEL_PROVIDERS[CURRENT_PROVIDER]
    DEFAULT_MODEL = f"{CURRENT_PROVIDER}:{PROVIDER_CONFIG[CURRENT_MODEL_TYPE]}"
    
    LOGGER.info(
        "Switched to provider: %s with model: %s",
        provider,
        DEFAULT_MODEL
    )

# LangSmith configuration ------------------------------------------------------
#
# DEV: you likely run against your personal LangSmith account -> key required.
# PROD (LangGraph Cloud / managed): do NOT override; let the SDK/platform resolve.
#
# Note: langsmith-sdk commonly uses env-driven auth; avoid forcing a specific env
# name in production when the platform injects its own.
if ENVIRONMENT == "DEV":
    LANGSMITH_API_KEY: Optional[str] = (
        environ.get("LANGSMITH_API_KEY") or environ.get("LANGCHAIN_API_KEY")
    )
else:
    LANGSMITH_API_KEY = None

LANGSMITH_PROJECT = environ.get("LANGSMITH_PROJECT", "baby-NICER-61")
WRITE_METADATA = True

# ───── Hugging Face / Transformers cache configuration ─────
# Prefer explicit per-library cache envs when present. Otherwise, derive from HF_HOME.
# In LangGraph Cloud (managed), this will be an ephemeral path. In self-hosted, point these
# to a mounted volume to persist caches across restarts.
HF_HOME = environ.get("HF_HOME")
HF_HUB_CACHE = environ.get("HF_HUB_CACHE")        # Shared hub cache (recommended)
TRANSFORMERS_CACHE = environ.get("TRANSFORMERS_CACHE")  # Transformers-specific override

# Export a single canonical cache dir for app code to consume.
# Order: TRANSFORMERS_CACHE > HF_HUB_CACHE > (HF_HOME + "/hub") > None
if HF_HOME:
    # Hub convention uses "{HF_HOME}/hub" as the hub cache directory
    _DERIVED_HUB_CACHE = os.path.join(HF_HOME, "hub")
else:
    _DERIVED_HUB_CACHE = None

HF_CACHE_DIR = TRANSFORMERS_CACHE or HF_HUB_CACHE or _DERIVED_HUB_CACHE

if HF_CACHE_DIR:
    LOGGER.info("HF cache directory configured: %s", HF_CACHE_DIR)
    # Best-effort: ensure it exists; warn if the platform disallows writes.
    try:
        os.makedirs(HF_CACHE_DIR, exist_ok=True)
    except Exception as e:
        LOGGER.warning("Could not create HF cache directory %s: %s", HF_CACHE_DIR, e)
else:
    LOGGER.info("HF cache directory not set; falling back to library defaults.")

# ───── Conversation payload safety limits ─────
# Cap the number of characters from a single turn's `content` that we will
# persist and embed. Prevents oversized rows and tool dumps from bloating storage.
# Override via env var: CONVERSATION_MAX_CONTENT_CHARS.
_cap_str = environ.get("CONVERSATION_MAX_CONTENT_CHARS", "12000")
try:
    _cap_val = int(_cap_str)
except Exception:
    _cap_val = 12000

# Treat non-positive as "no cap"
CONVERSATION_MAX_CONTENT_CHARS = _cap_val if _cap_val > 0 else None
LOGGER.info("Conversation content cap: %s", CONVERSATION_MAX_CONTENT_CHARS)

# Superset logging snapshot for debugging
LOGGER.info(
    "Superset enabled=%s tenant=%s region=%s url_set=%s redis_host_set=%s",
    SUPERSET_ENABLED,
    SUPERSET_TENANT,
    SUPERSET_REGION,
    bool(SUPERSET_URL),
    bool(SUPERSET_REDIS_HOST),
)