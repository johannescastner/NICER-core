import src.langgraph_slack.patch_typing  # must run before any Pydantic model loading
from langgraph.prebuilt import create_react_agent
import logging
from src.graphs.memory import get_memory_tools
import asyncio
logging.basicConfig(level=logging.DEBUG)
from langgraph_slack.config import DEFAULT_MODEL, DEFAULT_TEMPERATURE
logger = logging.getLogger(__name__)

async def retrieve_memory_tools():
    return await get_memory_tools()

MEMORY_TOOLS = asyncio.run(retrieve_memory_tools())
# Log what tools are being passed to the agent **before creation**
logger.debug(f"Agent tools before creation: {[tool.name for tool in MEMORY_TOOLS]}")

system_prompt = "You are baby-NICER, an evolving, modular agentic system under active development by Johannes Castner at Towards People. You have three integrated memory stores—semantic (factual knowledge), episodic (event/interaction history), and procedural (skills/processes)—enabling you to recall, learn, and improve over time. Unlike single-user chat assistants, you can converse with multiple people on Slack, maintaining continuity across their shared discussions. Your ultimate purpose is to help the Towards People team (including David Cuff’s psychological and consulting expertise) build the fuller NICER system. NICER will include specialized coding agents, web-search agents, data-warehouse and BI agents, and a “Habermas machine” for facilitating fair, consensus-driven communication. You can be configured to use various language models (e.g., DeepSeek, ChatGPT). Above all, your mission is to assist with the team’s project work—ranging from memory optimization and data-warehouse integration to broader business and innovation tasks—so that people can collaborate more effectively, reach shared understanding, and make progress toward building NICER."

# Create the agent using memory tools and keep the variable name `my_agent`
my_agent = create_react_agent(
    model=DEFAULT_MODEL,
    tools=MEMORY_TOOLS,
    debug=True, 
    prompt=system_prompt,
    model_parameters={"temperature": DEFAULT_TEMPERATURE}
   )
# Log what tools were actually registered
logger.debug("Agent successfully created!")