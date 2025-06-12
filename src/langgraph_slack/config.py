import logging
import base64
import json
from os import environ
from google.oauth2 import service_account

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
    service_account_json = base64.b64decode(environ.get("GCP_SERVICE_ACCOUNT_BASE64", "")).decode("utf-8")
    SERVICE_ACCOUNT_INFO: dict = json.loads(service_account_json)
    CREDENTIALS = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO)
    LOGGER.info("Service account credentials loaded successfully.")
except Exception as e:
    LOGGER.error(f"Failed to load service account credentials: {e}")

# ───── Language-model defaults ─────
DEFAULT_MODEL    = environ.get("NICER_MODEL", "openai:o3-mini")
DEFAULT_TEMPERATURE = float(environ.get("NICER_TEMPERATURE", "0.0"))
DEFAULT_DATASET     ="linkedin_raw"

