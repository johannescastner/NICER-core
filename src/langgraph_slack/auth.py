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

@auth.authenticate
async def authenticate(request, path, headers, method):
    """
    authentication function for langgraph
    """
    logger.info("authenticate function called")
    user_agent = headers.get(b"user-agent")
    logger.info("User-Agent: %s", user_agent)
    if (user_agent and user_agent.startswith(b"Slackbot")) or environment == "DEV":
        logger.info("Slackbot identified; authentication successful")
        return {"identity": "default-user", "permissions": ["read", "write"]}
    logger.warning("Authentication failed; User-Agent not recognized")

    return None
