import src.langgraph_slack.patch_typing  # must run before any Pydantic model loading
from langgraph.prebuilt import create_react_agent
import logging
from src.graphs.memory import get_memory_tools
import asyncio
logging.basicConfig(level=logging.DEBUG)
from langgraph_slack.config import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from src.prompts import BABY_NICER_PROMPT  # OSS system prompt
logger = logging.getLogger(__name__)

async def retrieve_memory_tools():
    return await get_memory_tools()

MEMORY_TOOLS = asyncio.run(retrieve_memory_tools())
# Log what tools are being passed to the agent **before creation**
logger.debug(f"Agent tools before creation: {[tool.name for tool in MEMORY_TOOLS]}")



# Create the agent using memory tools and keep the variable name `my_agent`
my_agent = create_react_agent(
    model=DEFAULT_MODEL,
    tools=MEMORY_TOOLS,
    debug=True, 
    prompt=BABY_NICER_PROMPT
   )
# Log what tools were actually registered
logger.debug("Agent successfully created!")