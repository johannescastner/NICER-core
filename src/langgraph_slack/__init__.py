import sys

# Patch typing.TypedDict to typing_extensions.TypedDict for Python < 3.12
if sys.version_info < (3, 12):
    import typing_extensions
    import typing
    typing.TypedDict = typing_extensions.TypedDict

import logging
import dotenv

dotenv.load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
