"""
this is where globals are defined and configured.
"""
# src/langgraph_slack/config.py
import logging
import os
import base64
import json
import importlib
from os import environ
from google.oauth2 import service_account
from .deepseek_utils import (
    wrap_llm_with_deepseek_backoff,
    make_balance_checker,
    BackoffConfig,
)
LOGGER = logging.getLogger(__name__)

if DEPLOY_MODAL := environ.get("DEPLOY_MODAL"):
    DEPLOY_MODAL = DEPLOY_MODAL.lower() == "true"
BOT_USER_ID = environ.get("SLACK_BOT_USER_ID")
BOT_TOKEN = environ.get("SLACK_BOT_TOKEN")
if DEPLOY_MODAL:
    if not environ.get("SLACK_BOT_TOKEN"):
        environ["SLACK_BOT_TOKEN"] = "fake-token"
    BOT_USER_ID = BOT_USER_ID or "fake-user-id"
else:
    assert isinstance(BOT_TOKEN, str)


    # APP_TOKEN = environ["SLACK_APP_TOKEN"]


LANGGRAPH_URL = environ.get("LANGGRAPH_URL")
ASSISTANT_ID = environ.get("LANGGRAPH_ASSISTANT_ID", "default_assistant_id")
CONFIG = environ.get("CONFIG") or "{}"
DEPLOYMENT_URL = environ.get("DEPLOYMENT_URL", "")
SLACK_CHANNEL_ID = environ.get("SLACK_CHANNEL_ID")

# Company configuration
COMPANY = environ.get("COMPANY", "Towards People")

# Google Cloud project details
PROJECT_ID = environ.get("GCP_PROJECT_ID", "default_project_id")
DATASET_ID = environ.get("GCP_DATASET_ID", "agent_system_memory")
LOCATION = environ.get("GCP_LOCATION", "EU")  # Default to 'US' if not set

# Table names for different memory types
SEMANTIC_TABLE = environ.get("GCP_SEMANTIC_TABLE", "semantic_memory")
EPISODIC_TABLE = environ.get("GCP_EPISODIC_TABLE", "episodic_memory")
PROCEDURAL_TABLE = environ.get("GCP_PROCEDURAL_TABLE", "procedural_memory")

# Service account credentials
try:
    service_account_json = base64.b64decode(
        environ.get("GCP_SERVICE_ACCOUNT_BASE64", "")
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
LANGSMITH_API_KEY = environ.get("LANGSMITH_API_KEY")
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
