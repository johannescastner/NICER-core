import sys
import typing
import typing_extensions

# For Python < 3.12, patch typing.TypedDict to avoid Pydantic crash
if sys.version_info < (3, 12):
    if not hasattr(typing, "TypedDict") or typing.TypedDict is not typing_extensions.TypedDict:
        typing.TypedDict = typing_extensions.TypedDict


import pydantic
import logging
import importlib.metadata


import dotenv

dotenv.load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Log versions of relevant libraries
logging.info(f"typing.TypedDict = {typing.TypedDict}")
logging.info(f"typing_extensions.TypedDict = {typing_extensions.TypedDict}")
logging.info(f"Is typing.TypedDict patched? {typing.TypedDict is typing_extensions.TypedDict}")
try:
    version = importlib.metadata.version("typing_extensions")
except importlib.metadata.PackageNotFoundError:
    version = "unknown"

logging.info(f"typing-extensions version: {version}")
logging.info(f"pydantic version: {pydantic.VERSION}")