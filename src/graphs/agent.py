# src/graphs/agent.py

import src.langgraph_slack.patch_typing  # must run before any Pydantic model loading
from langgraph.prebuilt import create_react_agent
import logging
from src.graphs.memory import get_memory_tools, NamespaceTemplate
import asyncio
logging.basicConfig(level=logging.DEBUG)
from langgraph_slack.config import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from src.prompts import BABY_NICER_PROMPT  # OSS system prompt
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

async def retrieve_memory_tools() -> list:
    """
    Build a dictionary of namespace templates that this core agent wants.
    We want three namespaces:
      1. 'semantic'   -> ("semantic", "general", "{langgraph_auth_user_id}")
      2. 'episodic'   -> ("episodic", "general", "{langgraph_auth_user_id}")
      3. 'procedural' -> ("procedural", "general", "{langgraph_auth_user_id}")
    """
    from typing import Dict

    namespace_templates: Dict[str, NamespaceTemplate] = {
        # "semantic" store: e.g. namespace="semantic.general.user-123"
        "semantic": ("semantic", "general", "{langgraph_auth_user_id}"),

        # "episodic" store: e.g. namespace="episodic.general.user-123"
        "episodic": ("episodic", "general", "{langgraph_auth_user_id}"),

        # "procedural" store: e.g. namespace="procedural.general.user-123"
        "procedural": ("procedural", "general", "{langgraph_auth_user_id}"),
    }

    return await get_memory_tools(namespace_templates=namespace_templates)

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