"""Lazy-initialize ConversationLogger to avoid blocking async event loop.

CRITICAL FIX: The previous eager initialization caused the swarm graph to hang
because ConversationLogger.__init__() calls initialize_model_cache_strategy()
which makes synchronous GCS API calls. When this runs during import in an
async context (like server_mit.py), it blocks the event loop indefinitely.

Now we use lazy initialization - the logger is only created on first use,
by which point the async event loop is properly running and can handle
the initialization in a non-blocking way.
"""
from typing import TYPE_CHECKING, Optional
import threading

if TYPE_CHECKING:
    from src.nicer_logging.conversation_logger import ConversationLogger

_global_logger: Optional["ConversationLogger"] = None
_init_lock = threading.Lock()


def get_conversation_logger() -> "ConversationLogger":
    """Return the global logger instance, creating it lazily on first use.
    
    Thread-safe lazy initialization ensures:
    1. No blocking during module import
    2. Logger created only when actually needed
    3. Single instance shared across all callers
    """
    global _global_logger
    if _global_logger is None:
        with _init_lock:
            # Double-check pattern for thread safety
            if _global_logger is None:
                from src.nicer_logging.conversation_logger import ConversationLogger
                _global_logger = ConversationLogger()
    return _global_logger