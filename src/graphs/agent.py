from langgraph.prebuilt import create_react_agent
from src.graphs.memory import MEMORY_TOOLS  # Import the memory tools
import logging

logger = logging.getLogger(__name__)
# Log what tools are being passed to the agent **before creation**
logger.debug(f"Agent tools before creation: {[tool.name for tool in MEMORY_TOOLS]}")

# Create the agent using memory tools and keep the variable name `my_agent`
my_agent = create_react_agent(
    "openai:o3-mini",
    tools=MEMORY_TOOLS,  #Use memory tools
   )
# Log what tools were actually registered
logger.debug("Agent successfully created!")