import sys

# Patch typing.TypedDict to typing_extensions.TypedDict for Python < 3.12
if sys.version_info < (3, 12):
    import typing_extensions
    import typing
    typing.TypedDict = typing_extensions.TypedDict
import typing
import typing_extensions
import pydantic
import logging


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
logging.info(f"typing-extensions version: {typing_extensions.__version__}")
logging.info(f"pydantic version: {pydantic.VERSION}")