"""Pre-initialize ConversationLogger at module import time."""
from src.nicer_logging.conversation_logger import ConversationLogger

# Force instantiation during import (before any graph is created)
# This ensures models are loaded BEFORE the queue worker starts
_global_logger = ConversationLogger()

def get_conversation_logger() -> ConversationLogger:
    """Return the pre-warmed global logger instance."""
    return _global_logger