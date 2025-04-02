import typing
import typing_extensions
typing.TypedDict = typing_extensions.TypedDict
import uvicorn

uvicorn.run("langgraph_slack.server:app", host="0.0.0.0", port=8080)
