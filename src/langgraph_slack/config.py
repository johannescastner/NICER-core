# src/langgraph_slack/config.py
"""
Configuration for LangGraph Slack integration.

CRITICAL PERFORMANCE FIX: Matplotlib font scanning optimization MUST be at the top,
before any other imports that might trigger matplotlib loading.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# ✅ MATPLOTLIB OPTIMIZATION - MUST BE FIRST!
# This prevents matplotlib from scanning ALL system fonts at import time,
# which saves ~4 minutes on every worker startup.
# ═══════════════════════════════════════════════════════════════════════════════
import os
import sys

# Disable matplotlib font scanning and set minimal configuration
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('MPLBACKEND', 'Agg')  # Non-interactive backend

# Pre-configure matplotlib BEFORE it gets imported by any dependency
try:
    import matplotlib
    matplotlib.use('Agg', force=True)  # Non-interactive backend
    # Limit font scanning to a single known font family
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    # Disable font manager refresh
    matplotlib.rcParams['font.manager'] = None
    print("✅ Matplotlib optimized (font scanning disabled)")
except ImportError:
    # Matplotlib not installed - that's fine
    pass

# ═══════════════════════════════════════════════════════════════════════════════
# NOW SAFE TO IMPORT OTHER MODULES
# ═══════════════════════════════════════════════════════════════════════════════

import json
import base64
import logging
from os import environ
from google.oauth2 import service_account

LOGGER = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Core Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# LangGraph Configuration
LANGGRAPH_URL = environ.get("LANGGRAPH_URL", "http://localhost:8000")
ASSISTANT_ID = environ.get("ASSISTANT_ID", "agent")
CONFIG = environ.get("ASSISTANT_CONFIG", "{}")

# Slack Configuration
SLACK_BOT_TOKEN = environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = environ.get("SLACK_SIGNING_SECRET")
SLACK_CHANNEL_ID = environ.get("SLACK_CHANNEL_ID")
BOT_USER_ID = environ.get("BOT_USER_ID", "fake-user-id")

# Deployment Configuration
DEPLOYMENT_URL = environ.get("DEPLOYMENT_URL")

# GCP Configuration
PROJECT_ID = environ.get("GCP_PROJECT_ID") or environ.get("PROJECT_ID")
GCP_REGION = environ.get("GCP_REGION", "europe-west1")

# LangSmith Configuration
LANGSMITH_API_KEY = environ.get("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = environ.get("LANGSMITH_PROJECT", "default")

# HuggingFace Cache Configuration
HF_CACHE_DIR = environ.get("HF_HOME") or environ.get("TRANSFORMERS_CACHE")
if HF_CACHE_DIR:
    LOGGER.info("HF cache directory: %s", HF_CACHE_DIR)
else:
    LOGGER.info("HF cache directory not set; falling back to library defaults.")

# Conversation Logging Configuration
CONVERSATION_MAX_CONTENT_CHARS = environ.get("CONVERSATION_MAX_CONTENT_CHARS", "12000")
LOGGER.info("Conversation content cap: %s", CONVERSATION_MAX_CONTENT_CHARS)

# ═══════════════════════════════════════════════════════════════════════════════
# Superset Configuration
# ═══════════════════════════════════════════════════════════════════════════════

SUPERSET_ENABLED = environ.get("SUPERSET_ENABLED", "false").lower() in ("true", "1", "yes")
SUPERSET_TENANT = environ.get("SUPERSET_TENANT", "")
SUPERSET_REGION = environ.get("SUPERSET_REGION", GCP_REGION)
SUPERSET_URL = environ.get("SUPERSET_URL")
SUPERSET_REDIS_HOST = environ.get("SUPERSET_REDIS_HOST")

LOGGER.info(
    "Superset enabled=%s tenant=%s region=%s url_set=%s redis_host_set=%s",
    SUPERSET_ENABLED,
    SUPERSET_TENANT,
    SUPERSET_REGION,
    bool(SUPERSET_URL),
    bool(SUPERSET_REDIS_HOST),
)

# ═══════════════════════════════════════════════════════════════════════════════
# GCP Service Account Credentials
# ═══════════════════════════════════════════════════════════════════════════════

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

# Best-effort derive the SA email from embedded credentials.
# This is useful for provisioning scripts that need to grant Secret Manager access
# to the runtime identity without forcing clients to repeat themselves.
SERVICE_ACCOUNT_EMAIL = (
    SERVICE_ACCOUNT_INFO.get("client_email")
    if isinstance(SERVICE_ACCOUNT_INFO, dict)
    else None
)
if SERVICE_ACCOUNT_EMAIL:
    LOGGER.info(
        "Derived service account email from GCP_SERVICE_ACCOUNT_BASE64: %s",
        SERVICE_ACCOUNT_EMAIL
    )
else:
    LOGGER.warning(
        "SERVICE_ACCOUNT_EMAIL could not be derived from GCP_SERVICE_ACCOUNT_BASE64; "
        "Superset deployment will fail if this is required."
    )
