# src/langgraph_slack/server_mit.py
"""
MIT-Licensed LangGraph Server for baby-NICER

This replaces the proprietary langchain/langgraph-api Docker image with a 
self-hosted FastAPI server using only MIT-licensed components:
- langgraph (MIT) - graph execution
- langgraph-checkpoint-postgres (MIT) - persistence
- fastapi (MIT) - HTTP server

Key changes from server.py:
1. Replaces langgraph_sdk.get_client() with direct graph compilation
2. Uses AsyncPostgresSaver for thread persistence
3. No dependency on LangGraph Platform licensing

MULTI-TENANT SUPPORT preserved:
- /slack/event endpoint for CollectiWise Router
- Bot token flows through task queue → graph config → callback
"""
import src.langgraph_slack.patch_typing  # must run before any Pydantic model loading
import asyncio
import logging
import os
import re
import json
import uuid
from urllib.parse import urlparse, quote_plus
from typing import Awaitable, Callable, Optional, Dict, Any
from typing_extensions import TypedDict
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from slack_sdk.web.async_client import AsyncWebClient
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.async_app import AsyncApp

# MIT-licensed LangGraph imports
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# PostgreSQL connection pool (required for AsyncPostgresSaver)
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

from langgraph_slack import config
from langgraph_slack.auth_fastapi import verify_request  # New auth module

# Ambient HTTP endpoints (closed-source)
from pro.http.ambient import router as ambient_router
from pro.http.cron_lifecycle import ensure_ambient_cron_exists
from pro.persistence import close_persistence_manager
from pro.utils.blocking_detector import install_blocking_detector

# Set MIT mode BEFORE importing swarm_graph to prevent auto-creation
os.environ["LANGGRAPH_MIT_MODE"] = "true"

# Import graph builders/factories
# NOTE: We import FACTORIES (not pre-compiled graphs) so we can inject checkpointer
from pro.graphs.chat_pro import chat_pro  # Pre-compiled, will use as-is
from pro.graphs.sql_graph import get_sql_graph_standalone  # MIT mode: accepts checkpointer
from pro.graphs.ambient_sql_graph import create_ambient_sql_graph_standalone  # MIT mode: accepts checkpointer

# Import swarm_graph module to access both the factory AND any existing instance
import pro.graphs.swarm_graph as swarm_module

# LangSmith tracing (MIT-licensed, continues to work!)
# Note: LangSmith observability is SEPARATE from LangGraph Platform licensing.
# The LANGSMITH_API_KEY is for tracing, not for the runtime server.
from pro.monitoring.langsmith_integration import get_langsmith_integration

LOGGER = logging.getLogger(__name__)

# Initialize LangSmith integration (for tracing)
# This is optional but recommended for observability
try:
    _LSI = get_langsmith_integration()
    LOGGER.info("✅ LangSmith tracing initialized (project: %s)", _LSI.project_name)
except Exception as e:
    _LSI = None
    LOGGER.warning("⚠️ LangSmith tracing not available: %s", e)

# ═══════════════════════════════════════════════════════════════════════════════
# Graph Registry & Checkpointer Setup
# ═══════════════════════════════════════════════════════════════════════════════

# Global checkpointer - initialized in lifespan
_checkpointer: Optional[AsyncPostgresSaver] = None

# Global connection pool - initialized in lifespan, cleaned up on shutdown
_connection_pool: Optional[AsyncConnectionPool] = None

# Compiled graphs - initialized in lifespan
_compiled_graphs: Dict[str, CompiledStateGraph] = {}

# Graph configuration
# Each entry is either:
#   - ("factory", func) → call func(checkpointer=checkpointer) 
#   - ("builder", func) → call func().compile(checkpointer=checkpointer)
#   - ("compiled", graph) → use graph as-is (no persistence)
#   - ("module_factory", (module, func_name)) → getattr(module, func_name)(checkpointer=...)
GRAPH_CONFIG = {
    "chat": ("compiled", chat_pro),  # Already compiled, no checkpointer support
    "sql_agent": ("factory", get_sql_graph_standalone),  # MIT mode: accepts checkpointer
    "swarm": ("module_factory", (swarm_module, "create_sql_swarm")),  # Factory in module
    "ambient_sql": ("factory", create_ambient_sql_graph_standalone),  # MIT mode: accepts checkpointer
}

# Default graph for Slack messages
DEFAULT_GRAPH = os.getenv("DEFAULT_GRAPH", "swarm")


def _build_database_uri() -> str:
    """
    Build PostgreSQL connection URI from environment variables.
    
    Connection modes:
    1. Cloud SQL via Proxy (TCP on localhost) - CLOUD_SQL_INSTANCE set
       The entrypoint starts cloud-sql-proxy on localhost:5432
    2. Direct TCP connection - DB_HOST set explicitly
    
    Note: We always use TCP, not Unix sockets, because the Cloud SQL Proxy
    is configured in TCP mode by entrypoint.sh for better compatibility.
    """
    db_user = os.getenv("DB_USER", "langgraph")
    db_password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "langgraph")
    db_host = os.getenv("DB_HOST", "127.0.0.1")  # Default to localhost (proxy)
    db_port = os.getenv("DB_PORT", "5432")
    
    # URL-encode password in case it contains special characters
    encoded_password = quote_plus(db_password) if db_password else ""
    
    uri = f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
    
    # Log connection info (redacted password)
    LOGGER.info(
        "Database URI: postgresql://%s:***@%s:%s/%s",
        db_user, db_host, db_port, db_name
    )
    
    return uri


async def _init_checkpointer() -> AsyncPostgresSaver:
    """
    Initialize the PostgreSQL checkpointer with connection pool.
    
    IMPORTANT: AsyncPostgresSaver.from_conn_string() returns an async context
    manager, not a checkpointer directly. For long-running servers, we need
    to manage the connection pool ourselves.
    
    Required psycopg settings:
    - autocommit=True: Required for setup() to commit table creation
    - row_factory=dict_row: Required because AsyncPostgresSaver uses dict-style row access
    """
    global _checkpointer, _connection_pool
    
    database_uri = _build_database_uri()
    
    # Create connection pool (stays open for app lifetime)
    # IMPORTANT: autocommit=True and row_factory=dict_row are REQUIRED
    _connection_pool = AsyncConnectionPool(
        conninfo=database_uri,
        max_size=20,
        open=False,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
        }
    )
    await _connection_pool.open(wait=True, timeout=30)
    LOGGER.info("✅ Connection pool opened (max_size=20)")
    
    # Create checkpointer with the pool
    _checkpointer = AsyncPostgresSaver(conn=_connection_pool)
    
    # Setup tables (idempotent - safe to call every startup)
    # Creates: checkpoint_migrations, checkpoints, checkpoint_blobs, checkpoint_writes
    await _checkpointer.setup()
    LOGGER.info("✅ Checkpointer initialized and tables created")
    
    return _checkpointer


async def _cleanup_checkpointer():
    """Cleanup connection pool on shutdown."""
    global _connection_pool, _checkpointer
    
    if _connection_pool:
        try:
            await _connection_pool.close()
            LOGGER.info("✅ Connection pool closed")
        except Exception as e:
            LOGGER.warning("⚠️ Error closing connection pool: %s", e)
        _connection_pool = None
    
    _checkpointer = None


async def _compile_graphs(checkpointer: AsyncPostgresSaver) -> Dict[str, CompiledStateGraph]:
    """
    Compile all graphs with the checkpointer.
    
    Handles four patterns:
    - "factory": Functions that accept checkpointer as parameter
    - "builder": Functions that return StateGraph builders (need .compile())
    - "compiled": Pre-compiled graphs (used as-is, no persistence)
    - "module_factory": Factory function in a module (handles singleton pattern)
    """
    global _compiled_graphs
    
    for name, config in GRAPH_CONFIG.items():
        pattern = config[0]
        source = config[1]
        
        try:
            if pattern == "factory":
                # Factory function accepts checkpointer directly
                _compiled_graphs[name] = source(checkpointer=checkpointer)
                LOGGER.info("✅ Compiled graph via factory: %s", name)
                
            elif pattern == "builder":
                # Function returns a StateGraph builder
                builder = source()
                _compiled_graphs[name] = builder.compile(checkpointer=checkpointer)
                LOGGER.info("✅ Compiled graph from builder: %s", name)
                
            elif pattern == "compiled":
                # Already compiled graph (no checkpointer support)
                _compiled_graphs[name] = source
                LOGGER.warning("⚠️ Using pre-compiled graph (no persistence): %s", name)
                
            elif pattern == "module_factory":
                # Factory function in a module - handles singleton pattern
                module, func_name = source
                
                # Check if swarm already exists (singleton was created at import)
                existing = getattr(module, "_SWARM_INSTANCE", None)
                if existing is not None:
                    LOGGER.warning(
                        "⚠️ Using existing %s (created at import, no checkpointer). "
                        "To enable persistence, apply swarm_graph_patch.md",
                        name
                    )
                    _compiled_graphs[name] = existing
                else:
                    # Create with checkpointer
                    factory_func = getattr(module, func_name)
                    _compiled_graphs[name] = factory_func(checkpointer=checkpointer)
                    LOGGER.info("✅ Compiled graph via module factory: %s", name)
                
            else:
                raise ValueError(f"Unknown pattern: {pattern}")
                
        except Exception as e:
            LOGGER.error("❌ Failed to compile graph %s: %s", name, e, exc_info=True)
            raise
    
    return _compiled_graphs


def get_graph(name: str) -> CompiledStateGraph:
    """Get a compiled graph by name."""
    if name not in _compiled_graphs:
        raise ValueError(f"Unknown graph: {name}. Available: {list(_compiled_graphs.keys())}")
    return _compiled_graphs[name]


# ═══════════════════════════════════════════════════════════════════════════════
# Slack Message Handling (preserved from server.py)
# ═══════════════════════════════════════════════════════════════════════════════

class SlackMessageData(TypedDict):
    user: str
    type: str
    subtype: str | None
    ts: str
    thread_ts: str | None
    client_msg_id: str
    text: str
    team: str
    parent_user_id: str
    blocks: list[dict]
    channel: str
    event_ts: str
    channel_type: str


USER_NAME_CACHE: dict[str, str] = {}
TASK_QUEUE: asyncio.Queue = asyncio.Queue()
MENTION_REGEX = re.compile(r"<@([A-Z0-9]+)>")


async def worker():
    """Background worker for processing Slack messages."""
    LOGGER.info("Background worker started.")
    while True:
        task = None
        try:
            task = await TASK_QUEUE.get()
            if task is None:
                LOGGER.info("Worker received sentinel, exiting.")
                break
            LOGGER.info("Worker got a new task: %s", task.get("type"))
            await _process_task(task)
        except asyncio.CancelledError:
            LOGGER.info("Worker task was cancelled.")
            break
        except Exception as exc:
            LOGGER.exception("Error in worker: %s", exc)
        finally:
            if task is not None:
                TASK_QUEUE.task_done()


async def _process_task(task: dict):
    """Process a task from the queue."""
    event = task["event"]
    event_type = task["type"]
    bot_token: Optional[str] = task.get("bot_token")
    
    if event_type == "slack_message":
        await _handle_slack_message(event, bot_token)
    elif event_type == "callback":
        await _handle_callback(event)
    else:
        raise ValueError(f"Unknown event type: {event_type}")


async def _handle_slack_message(event: SlackMessageData, bot_token: Optional[str]):
    """
    Handle a Slack message by invoking the graph directly.
    
    This replaces the langgraph_sdk.runs.create() call with direct graph invocation.
    LangSmith tracing is added for observability (optional but recommended).
    """
    thread_id = _get_thread_id(
        event.get("thread_ts") or event["ts"], event["channel"]
    )
    channel_id = event["channel"]
    user_id = event["user"]
    
    # Check if it's a mention or DM
    if bot_token:
        is_mention = await _is_mention_with_token(event, bot_token)
    else:
        is_mention = await _is_mention(event)
    
    if not (is_mention or _is_dm(event)):
        LOGGER.info("Skipping non-mention message")
        return
    
    # Build contextual message with thread history
    text_with_names = await _build_contextual_message(event, bot_token=bot_token)
    
    # Generate conversation_id for tracking
    conversation_id = f"slack_{thread_id}"
    
    # Calculate turn number (simplified - you may want to track this in state)
    turn_number = 1  # TODO: Track actual turn number via checkpointer state
    
    LOGGER.info(
        "[%s].[%s] invoking graph with message: %s...",
        channel_id,
        thread_id,
        text_with_names[:100],
    )
    
    # Build graph config
    graph_config = {
        "configurable": {
            "thread_id": thread_id,
            "langgraph_auth_user_id": user_id,
        }
    }
    
    # Build input for the graph
    graph_input = {
        "messages": [
            {
                "role": "user",
                "content": text_with_names,
            }
        ],
        "context": {
            "slack_user_id": user_id,
            "channel_id": channel_id,
            "thread_id": thread_id,
            "thread_ts": event.get("thread_ts") or event["ts"],
        },
        "conversation_id": conversation_id,
    }
    
    # Store metadata for callback (we'll process response inline)
    metadata = {
        "event": "slack",
        "slack_event_type": "message",
        "bot_user_id": config.BOT_USER_ID,
        "slack_user_id": user_id,
        "channel_id": channel_id,
        "channel": channel_id,
        "thread_ts": event.get("thread_ts") or event["ts"],
        "event_ts": event["ts"],
        "channel_type": event.get("channel_type"),
        "conversation_id": conversation_id,
    }
    
    if bot_token:
        metadata["bot_token"] = bot_token
    
    try:
        # Get the default graph and invoke it directly
        graph = get_graph(DEFAULT_GRAPH)
        
        # ════════════════════════════════════════════════════════════════════
        # Invoke graph with optional LangSmith tracing
        # ════════════════════════════════════════════════════════════════════
        if _LSI is not None:
            # Wrap execution in LangSmith turn tracing for observability
            # Note: start_turn is a sync context manager (uses threading internally)
            with _LSI.start_turn(
                conversation_id=conversation_id,
                thread_id=thread_id,
                user_id=user_id,
                channel_id=channel_id,
                turn_number=turn_number,
                name="swarm_turn",
                tags=["slack", "mit-server"],
            ) as turn:
                turn.set_inputs({"messages": graph_input["messages"]})
                
                result = await graph.ainvoke(graph_input, config=graph_config)
                
                # Record outputs for LangSmith
                messages = result.get("messages", [])
                if messages:
                    turn.set_outputs({"response": _get_text(messages[-1].get("content", ""))})
        else:
            # No LangSmith - just invoke directly
            result = await graph.ainvoke(graph_input, config=graph_config)
        
        LOGGER.info("[%s].[%s] Graph execution complete", channel_id, thread_id)
        
        # Extract response and send to Slack
        messages = result.get("messages", [])
        if messages:
            response_message = messages[-1]
            response_text = _get_text(response_message.get("content", ""))
            
            # Send response to Slack
            await _send_slack_response(
                channel_id=channel_id,
                thread_ts=metadata["thread_ts"],
                text=response_text,
                bot_token=bot_token,
            )
        else:
            LOGGER.warning("[%s].[%s] No messages in graph result", channel_id, thread_id)
            
    except Exception as e:
        LOGGER.exception("[%s].[%s] Graph execution failed: %s", channel_id, thread_id, e)
        # Optionally send error message to Slack
        await _send_slack_response(
            channel_id=channel_id,
            thread_ts=metadata["thread_ts"],
            text=f"Sorry, I encountered an error processing your request.",
            bot_token=bot_token,
        )


async def _handle_callback(event: dict):
    """Handle webhook callbacks (for backward compatibility if needed)."""
    LOGGER.info("Processing callback: %s", event.get('thread_id'))
    
    state_values = event.get("values", {})
    response_message = state_values.get("messages", [{}])[-1]
    thread_ts = event.get("metadata", {}).get("thread_ts") or event.get("metadata", {}).get("event_ts")
    channel_id = event.get("metadata", {}).get("channel") or config.SLACK_CHANNEL_ID
    bot_token = event.get("metadata", {}).get("bot_token")
    
    if not channel_id:
        raise ValueError("Channel ID not found in event metadata")
    
    response_text = _get_text(response_message.get("content", ""))
    
    await _send_slack_response(
        channel_id=channel_id,
        thread_ts=thread_ts,
        text=response_text,
        bot_token=bot_token,
    )


async def _send_slack_response(
    channel_id: str,
    thread_ts: str,
    text: str,
    bot_token: Optional[str] = None,
):
    """Send a response to Slack."""
    cleaned_text = _clean_markdown(text)
    
    if bot_token:
        # Multi-tenant: use workspace-specific token
        LOGGER.info(
            "[%s].[%s] Using router-provided bot_token for response",
            channel_id,
            thread_ts,
        )
        client = AsyncWebClient(token=bot_token)
    else:
        # Single-tenant fallback
        client = APP_HANDLER.app.client
    
    await client.chat_postMessage(
        channel=channel_id,
        thread_ts=thread_ts,
        text=cleaned_text,
    )
    
    LOGGER.info("[%s].[%s] Sent message to Slack", channel_id, thread_ts)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions (preserved from server.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_text(content: str | list[dict]) -> str:
    if isinstance(content, str):
        return content
    else:
        return "".join([block["text"] for block in content if block.get("type") == "text"])


def _clean_markdown(text: str) -> str:
    text = re.sub(r"^```[^\n]*\n", "```\n", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"*\1*", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"_\1_", text)
    text = re.sub(r"_([^_]+)_", r"_\1_", text)
    text = re.sub(r"^\s*[-*]\s", "• ", text, flags=re.MULTILINE)
    return text


def _get_thread_id(thread_ts: str, channel: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"SLACK:{thread_ts}-{channel}"))


def _is_dm(event: SlackMessageData) -> bool:
    return event.get("channel_type") == "im"


async def _is_mention(event: SlackMessageData) -> bool:
    """Check if event mentions the bot using global client."""
    if not config.BOT_USER_ID or config.BOT_USER_ID == "fake-user-id":
        try:
            auth_result = await APP_HANDLER.app.client.auth_test()
            config.BOT_USER_ID = auth_result["user_id"]
        except Exception:
            return False
    
    pattern = re.compile(rf"<@{config.BOT_USER_ID}>")
    return bool(re.search(pattern, event.get("text", "")))


async def _is_mention_with_token(event: SlackMessageData, bot_token: str) -> bool:
    """Check if event mentions the bot using a specific bot token."""
    try:
        client = AsyncWebClient(token=bot_token)
        auth_result = await client.auth_test()
        bot_user_id = auth_result["user_id"]
        pattern = re.compile(rf"<@{bot_user_id}>")
        return bool(re.search(pattern, event.get("text", "")))
    except Exception as exc:
        LOGGER.warning("Failed to check mention with bot_token: %s", exc)
        return await _is_mention(event)


async def _fetch_thread_history(
    channel_id: str, thread_ts: str, *, bot_token: Optional[str] = None
) -> list[SlackMessageData]:
    """Fetch all messages in a Slack thread."""
    if bot_token:
        client = AsyncWebClient(token=bot_token)
    else:
        client = APP_HANDLER.app.client
    
    all_messages = []
    cursor = None
    
    while True:
        try:
            kwargs = {
                "channel": channel_id,
                "ts": thread_ts,
                "inclusive": True,
                "limit": 150,
            }
            if cursor:
                kwargs["cursor"] = cursor
            
            response = await client.conversations_replies(**kwargs)
            all_messages.extend(response["messages"])
            
            if not response.get("has_more"):
                break
            cursor = response["response_metadata"]["next_cursor"]
        except Exception as exc:
            LOGGER.exception("Error fetching thread messages: %s", exc)
            break
    
    return all_messages


async def _fetch_user_names(
    user_ids: set[str], *, bot_token: Optional[str] = None
) -> dict[str, str]:
    """Fetch and cache Slack display names for user IDs."""
    if bot_token:
        client = AsyncWebClient(token=bot_token)
    else:
        client = APP_HANDLER.app.client
    
    uncached_ids = [uid for uid in user_ids if uid not in USER_NAME_CACHE]
    if uncached_ids:
        tasks = [client.users_info(user=uid) for uid in uncached_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for uid, result in zip(uncached_ids, results):
            if isinstance(result, Exception):
                continue
            user_obj = result.get("user", {})
            profile = user_obj.get("profile", {})
            display_name = profile.get("display_name") or profile.get("real_name") or uid
            USER_NAME_CACHE[uid] = display_name
    
    return {uid: USER_NAME_CACHE[uid] for uid in user_ids if uid in USER_NAME_CACHE}


async def _build_contextual_message(
    event: SlackMessageData, *, bot_token: Optional[str] = None
) -> str:
    """Build a message with thread context, using display names."""
    thread_ts = event.get("thread_ts") or event["ts"]
    channel_id = event["channel"]
    
    history = await _fetch_thread_history(channel_id, thread_ts, bot_token=bot_token)
    included = []
    for msg in reversed(history):
        if msg.get("bot_id") == config.BOT_USER_ID:
            break
        included.append(msg)
    
    all_user_ids = set()
    for msg in included:
        all_user_ids.add(msg.get("user", "unknown"))
        all_user_ids.update(MENTION_REGEX.findall(msg.get("text", "")))
    
    all_user_ids.add(event["user"])
    all_user_ids.update(MENTION_REGEX.findall(event.get("text", "")))
    
    user_names = await _fetch_user_names(all_user_ids, bot_token=bot_token)
    
    def format_message(msg: SlackMessageData) -> str:
        text = msg.get("text", "")
        user_id = msg.get("user", "unknown")
        
        def repl(match: re.Match) -> str:
            uid = match.group(1)
            return user_names.get(uid, uid)
        
        replaced_text = MENTION_REGEX.sub(repl, text)
        speaker_name = user_names.get(user_id, user_id)
        return f'<slackMessage user="{speaker_name}">{replaced_text}</slackMessage>'
    
    context_parts = [format_message(msg) for msg in reversed(included)]
    new_message = context_parts[-1] if context_parts else ""
    preceding_context = "\n".join(context_parts[:-1]) if len(context_parts) > 1 else ""
    
    contextual_message = (
        (("Preceding context:\n" + preceding_context) if preceding_context else "")
        + "\n\nNew message:\n"
        + new_message
    )
    return contextual_message


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI App Setup
# ═══════════════════════════════════════════════════════════════════════════════

# Slack App Handler (for backward compatibility with direct Slack events)
APP_HANDLER = AsyncSlackRequestHandler(AsyncApp(
    token="xoxb-placeholder-for-multi-tenant-router",
    logger=LOGGER
))


def _log_task_result(task: asyncio.Task) -> None:
    """Ensure background task exceptions don't get swallowed."""
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        LOGGER.exception("Background worker crashed", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize checkpointer and compile graphs."""
    worker_task: asyncio.Task | None = None
    LOGGER.info("🚀 MIT-Licensed LangGraph Server starting up...")
    
    try:
        # Install blocking detector
        install_blocking_detector()
        LOGGER.info("✅ Blocking detector installed")
    except Exception:
        LOGGER.exception("⚠️ Failed to install blocking detector")
    
    try:
        # Initialize PostgreSQL checkpointer
        checkpointer = await _init_checkpointer()
        
        # Compile all graphs with checkpointer
        await _compile_graphs(checkpointer)
        LOGGER.info("✅ All graphs compiled successfully")
        
        # Start background worker
        worker_task = asyncio.create_task(worker(), name="slack_background_worker")
        worker_task.add_done_callback(_log_task_result)
        
        # Post-startup setup (non-blocking)
        async def _post_startup_setup():
            try:
                if os.getenv("AMBIENT_CRON_ENABLED", "true").lower() in {"1", "true", "yes", "y"}:
                    await ensure_ambient_cron_exists()
                    LOGGER.info("✅ ensure_ambient_cron_exists finished")
            except Exception:
                LOGGER.exception("❌ Post-startup setup failed")
        
        asyncio.create_task(_post_startup_setup(), name="post_startup_setup")
        
        yield
        
    except Exception:
        LOGGER.exception("❌ Lifespan failed during startup")
        raise
    finally:
        LOGGER.info("App is shutting down...")
        
        # Stop worker
        try:
            await TASK_QUEUE.put(None)
        except Exception:
            pass
        
        if worker_task is not None:
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
        
        # Close persistence
        try:
            close_persistence_manager()
        except Exception:
            pass
        
        # Close checkpointer connection pool
        await _cleanup_checkpointer()


APP = FastAPI(lifespan=lifespan)

# Mount ambient endpoints
APP.include_router(ambient_router)


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@APP.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "graphs": list(_compiled_graphs.keys()),
        "checkpointer": "connected" if _checkpointer else "disconnected",
    }


@APP.get("/info")
async def info():
    """Server info endpoint (replaces LangGraph Platform /info)."""
    return {
        "version": "1.0.0-mit",
        "license": "MIT",
        "graphs": list(_compiled_graphs.keys()),
        "default_graph": DEFAULT_GRAPH,
    }


@APP.post("/")
async def verify_slack(req: Request):
    """Handle Slack's URL verification challenge."""
    data = await req.json()
    if "challenge" in data:
        return {"challenge": data["challenge"]}
    return JSONResponse({"detail": "Unauthorized"}, status_code=401)


@APP.post("/events/slack")
async def slack_endpoint(req: Request):
    """
    EXISTING ENDPOINT: Direct Slack → Agent communication.
    Preserved for backward compatibility.
    """
    body = await req.json()
    if body.get("type") == "url_verification" and "challenge" in body:
        return {"challenge": body["challenge"]}
    return await APP_HANDLER.handle(req)


@APP.post("/slack/event")
async def slack_event_from_router(req: Request, _: None = Depends(verify_request)):
    """
    Multi-Tenant Router Endpoint.
    
    The centralized router forwards Slack events from multiple workspaces.
    Uses FastAPI dependency injection for auth (verify_request).
    """
    bot_token = req.headers.get("x-slack-bot-token")
    team_id = req.headers.get("x-slack-team-id", "unknown")
    
    LOGGER.info(
        "🔀 Router-forwarded event from team=%s (bot_token present: %s)",
        team_id,
        bool(bot_token),
    )
    
    body = await req.json()
    
    # Handle URL verification
    if body.get("type") == "url_verification" and "challenge" in body:
        return {"challenge": body["challenge"]}
    
    event = body.get("event", {})
    event_type = event.get("type")
    
    if not event_type:
        return {"ok": True}
    
    # Filter events
    if event_type not in ("message", "app_mention"):
        return {"ok": True}
    
    if event.get("bot_id") or not event.get("user"):
        return {"ok": True}
    
    # For messages, check if it's a mention or DM
    if event_type == "message":
        if bot_token:
            is_mention = await _is_mention_with_token(event, bot_token)
        else:
            is_mention = await _is_mention(event)
        
        if not (is_mention or _is_dm(event)):
            return {"ok": True}
    
    # Enqueue for processing
    LOGGER.info(
        "🔀 Enqueuing router event: type=%s, user=%s, channel=%s",
        event_type,
        event.get("user"),
        event.get("channel"),
    )
    
    TASK_QUEUE.put_nowait({
        "type": "slack_message",
        "event": event,
        "bot_token": bot_token,
    })
    
    return {"ok": True}


@APP.post("/callbacks/{thread_id}")
async def webhook_callback(req: Request):
    """Handle webhook callbacks (backward compatibility)."""
    body = await req.json()
    LOGGER.info("Received webhook callback for %s", req.path_params.get('thread_id'))
    TASK_QUEUE.put_nowait({"type": "callback", "event": body})
    return {"status": "success"}


# ═══════════════════════════════════════════════════════════════════════════════
# Direct Graph Invocation Endpoints (new for MIT version)
# ═══════════════════════════════════════════════════════════════════════════════

@APP.post("/runs")
async def create_run(req: Request, _: None = Depends(verify_request)):
    """
    Create a new run (replaces LangGraph Platform runs.create).
    
    This is a simplified version for programmatic access.
    Most Slack traffic goes through /slack/event.
    """
    body = await req.json()
    
    graph_name = body.get("assistant_id", DEFAULT_GRAPH)
    thread_id = body.get("thread_id", str(uuid.uuid4()))
    input_data = body.get("input", {})
    config_data = body.get("config", {})
    
    # Ensure thread_id is in config
    if "configurable" not in config_data:
        config_data["configurable"] = {}
    config_data["configurable"]["thread_id"] = thread_id
    
    try:
        graph = get_graph(graph_name)
        result = await graph.ainvoke(input_data, config=config_data)
        
        return {
            "thread_id": thread_id,
            "status": "success",
            "values": result,
        }
    except Exception as e:
        LOGGER.exception("Run failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@APP.get("/threads/{thread_id}/state")
async def get_thread_state(thread_id: str, _: None = Depends(verify_request)):
    """Get the current state of a thread."""
    if not _checkpointer:
        raise HTTPException(status_code=503, detail="Checkpointer not initialized")
    
    try:
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = await _checkpointer.aget_tuple(config)
        
        if checkpoint is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        return {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint.checkpoint.get("id"),
            "values": checkpoint.checkpoint.get("channel_values", {}),
        }
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception("Failed to get thread state: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("langgraph_slack.server_mit:APP", host="0.0.0.0", port=8080)