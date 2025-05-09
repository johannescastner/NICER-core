import sys
import typing
import typing_extensions

# Monkey-patch TypedDict for Pydantic compatibility
if sys.version_info < (3, 12):
    if not hasattr(typing, "TypedDict") or typing.TypedDict is not typing_extensions.TypedDict:
        typing.TypedDict = typing_extensions.TypedDict