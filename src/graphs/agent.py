from langgraph.prebuilt import create_react_agent
from src.graphs.memory import MEMORY_TOOLS  # Import the memory tools

# Create the agent using memory tools and keep the variable name `my_agent`
my_agent = create_react_agent(
    "openai:o3-mini",
    tools=MEMORY_TOOLS,  # Use memory-powered tools
)