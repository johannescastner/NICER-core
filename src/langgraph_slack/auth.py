"""
This is where authentication lives
"""
import logging
import os
from langgraph_sdk import Auth

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the environment from OS variables (default to "PROD" if not set)
environment = os.getenv("ENVIRONMENT", "PROD")
print(f"Current environment: {environment}")

auth = Auth()

STUDIO_ORIGINS = {
    b"https://smith.langchain.com",
    "https://smith.langchain.com",
}

@auth.authenticate
async def authenticate(request, path, headers, method):
    """
    authentication function for langgraph
    """
    logger.info("authenticate function called")
    # Always allow CORS preflight
    if method in (b"OPTIONS", "OPTIONS"):
        return {"identity": "cors-preflight", "permissions": ["read", "write"]}

    user_agent = headers.get(b"user-agent") or headers.get("user-agent")
    origin = headers.get(b"origin") or headers.get("origin") or headers.get(b"Origin") or headers.get("Origin")

    logger.info("User-Agent: %s", user_agent)

    # Allow LangSmith Studio (origin-based)
    if origin in STUDIO_ORIGINS:
        logger.info("LangSmith Studio origin identified; authentication successful")
        return {"identity": "studio-user", "permissions": ["read", "write"]}

    if (user_agent and isinstance(user_agent, (bytes, bytearray)) and user_agent.startswith(b"Slackbot")) or environment == "DEV":

        logger.info("Slackbot identified; authentication successful")
        return {"identity": "default-user", "permissions": ["read", "write"]}
    logger.warning("Authentication failed; User-Agent not recognized")

    return None
