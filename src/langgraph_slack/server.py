# src/langgraph_slack/server.py
"""This is the slack server interface"""
import src.langgraph_slack.patch_typing  # must run before any Pydantic model loading
import asyncio
import contextlib
import logging
import os
import re
import json
import uuid
from urllib.parse import urlparse
from typing import Awaitable, Callable
from typing_extensions import TypedDict
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from langgraph_sdk import get_client
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.async_app import AsyncApp
from langgraph_slack import config
# Ambient HTTP endpoints (closed-source)
from pro.http.ambient import router as ambient_router
from pro.http.cron_lifecycle import ensure_ambient_cron_exists
from pro.persistence import close_persistence_manager
from pro.utils.blocking_detector import install_blocking_detector

LOGGER = logging.getLogger(__name__)
LANGGRAPH_CLIENT = get_client(url=config.LANGGRAPH_URL)
GRAPH_CONFIG = (
    json.loads(config.CONFIG) if isinstance(config.CONFIG, str) else config.CONFIG
)

def _origin_from_url(url: str | None) -> str | None:
    """Return 'scheme://host[:port]' from a URL, or None if invalid/empty."""
    if not url:
        return None
    try:
        u = urlparse(url)
        if not u.scheme or not u.netloc:
            return None
        return f"{u.scheme}://{u.netloc}"
    except Exception:
        return None

ALLOWED_ORIGINS: list[str] = [
    "https://smith.langchain.com",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Optional: auto-allow whatever DEPLOYMENT_URL is (useful in Cloud Run / dev)
_deployment_origin = _origin_from_url(getattr(config, "DEPLOYMENT_URL", None))
if _deployment_origin and _deployment_origin not in ALLOWED_ORIGINS:
    ALLOWED_ORIGINS.append(_deployment_origin)

# Allow dynamic Cloudflare tunnel origins for `langgraph dev --tunnel`.
# You can override this via env if you ever want to tighten/expand it.
ALLOW_ORIGIN_REGEX = os.getenv(
    "ALLOW_ORIGIN_REGEX",
    r"^https://[a-z0-9-]+\.trycloudflare\.com$",
)

USER_NAME_CACHE: dict[str, str] = {}
TASK_QUEUE: asyncio.Queue = asyncio.Queue()


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


async def worker():
    """
    The worker function for the background task.
    """
    LOGGER.info("Background worker started.")
    while True:
        task = None
        try:
            task = await TASK_QUEUE.get()
            if task is None:
                LOGGER.info("Worker received sentinel, exiting.")
                break
            LOGGER.info(
                "Worker got a new task: %s",
                task
            )
            await _process_task(task)
        except asyncio.CancelledError:
            LOGGER.info("Worker task was cancelled.")
            break
        except Exception as exc:
            LOGGER.exception(
                "Error in worker: %s",
                exc
            )
        finally:
            if task is not None:
                TASK_QUEUE.task_done()


async def _process_task(task: dict):
    event = task["event"]
    event_type = task["type"]
    if event_type == "slack_message":
        thread_id = _get_thread_id(
            event.get("thread_ts") or event["ts"], event["channel"]
        )
        channel_id = event["channel"]
        # This will connect to the loopback endpoint if not provided.
        webhook = f"{config.DEPLOYMENT_URL}/callbacks/{thread_id}"

        if (await _is_mention(event)) or _is_dm(event):
            text_with_names = await _build_contextual_message(event)
        else:
            LOGGER.info("Skipping non-mention message")
            return

        # Add the langgraph_auth_user_id to the GRAPH_CONFIG in the configurable field
        user_id = event["user"]
        updated_graph_config = {**GRAPH_CONFIG}
        if "configurable" not in updated_graph_config:
            updated_graph_config["configurable"] = {}
        updated_graph_config["configurable"]["langgraph_auth_user_id"] = user_id

        # Log the message content being sent to LangGraph
        LOGGER.debug(
            "Processed message for LangGraph: %s",
            text_with_names
        )

        # Log the event and user info
        LOGGER.debug(
            "Event info: %s",
            event
        )
        LOGGER.debug(
            "User info: %s",
            event['user']
        )

        LOGGER.info(
            """
            [%s].[%s] sending message to LangGraph: ",
            with webhook %s: %s
            """,
            channel_id,
            thread_id,
            webhook,
            text_with_names
        )

        # 🚨 CRITICAL FIX: Add conversation context for SummarizationNode and cost tracking
        # Generate conversation_id from thread_id for consistency
        conversation_id = f"slack_{thread_id}"

        # 🎯 CONVERSATION LOGGING IS NOW HANDLED BY THE GLOBAL SWARM GRAPH
        # The server is only responsible for initiating the graph execution.
        # All logging, including human and agent turns, is managed within the
        # swarm_graph's orchestrated workflow to ensure consistency and
        # capture of rich metadata like embeddings and sentiment.

        result = await LANGGRAPH_CLIENT.runs.create(
            thread_id=thread_id,
            assistant_id=config.ASSISTANT_ID,
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": text_with_names,
                    }
                ],
                # 🚨 Add conversation context required by SQL graph
                "context": {
                    "slack_user_id": event["user"],
                    "channel_id": channel_id,
                    "thread_id": thread_id,
                    "thread_ts": event.get("thread_ts") or event["ts"],
                },  # Required for LangMem SummarizationNode
                "conversation_id": conversation_id,  # For cost tracking correlation
            },
            config=updated_graph_config,
            metadata={
                "event": "slack",
                "slack_event_type": "message",
                "bot_user_id": config.BOT_USER_ID,
                "slack_user_id": event["user"],
                "channel_id": channel_id,
                "channel": channel_id,
                "thread_ts": event.get("thread_ts"),
                "event_ts": event["ts"],
                "channel_type": event.get("channel_type"),
                # 🚨 CRITICAL: Add conversation context to metadata for correlation
                "conversation_id": conversation_id
            },
            multitask_strategy="interrupt",
            if_not_exists="create",
            webhook=webhook,
        )
        LOGGER.debug(
            "LangGraph run: %s",
            result
        )

    elif event_type == "callback":
        LOGGER.info(
            "Processing LangGraph callback: %s",
            event['thread_id']
        )
        state_values = event["values"]
        response_message = state_values["messages"][-1]
        thread_ts = event["metadata"].get("thread_ts") or event["metadata"].get(
            "event_ts"
        )
        channel_id = event["metadata"].get("channel") or config.SLACK_CHANNEL_ID
        if not channel_id:
            raise ValueError(
                "Channel ID not found in event metadata and not set in environment"
            )

        # 🎯 CONVERSATION LOGGING IS NOW HANDLED BY THE GLOBAL SWARM GRAPH
        # The server is only responsible for delivering the final message to Slack.
        # The agent's response has already been logged by the `log_agent_turn`
        # node within the global graph.

        await APP_HANDLER.app.client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=_clean_markdown(_get_text(response_message["content"])),
            metadata={
                "event_type": "webhook",
                "event_payload": {"thread_id": event["thread_id"]},
            },
        )
        LOGGER.info(
            "[%s].[%s] sent message to Slack for callback %s", 
            channel_id,
            thread_ts,
            event['thread_id']
        )
    else:
        raise ValueError(f"Unknown event type: {event_type}")


async def handle_message(
        event: SlackMessageData,
        say: Callable,
        ack: Callable
):
    """
    Handle incoming Slack messages.
    """
    LOGGER.info("Enqueuing handle_message task...")
    nouser = not event.get("user")
    ismention = await _is_mention(event)
    userisbot = event.get("bot_id") == config.BOT_USER_ID
    isdm = _is_dm(event)
    if nouser or userisbot or not (ismention or isdm):
        LOGGER.info(
            "Ignoring message not directed at the bot: %s",
            event
        )
        return

    TASK_QUEUE.put_nowait({"type": "slack_message", "event": event})
    await ack()


async def just_ack(ack: Callable[..., Awaitable], event):
    """
    simple helper
    """
    LOGGER.info(
        "Acknowledging %s event", 
        event.get('type') or event.get('subtype')
    )
    await ack()


APP_HANDLER = AsyncSlackRequestHandler(AsyncApp(logger=LOGGER))
MENTION_REGEX = re.compile(r"<@([A-Z0-9]+)>")
USER_ID_PATTERN = re.compile(rf"<@{config.BOT_USER_ID}>")
APP_HANDLER.app.event("message")(ack=just_ack, lazy=[handle_message])
APP_HANDLER.app.event("app_mention")(
    ack=just_ack,
    lazy=[],
)

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
    """
    defines the lifespan for the app
    """
    worker_task: asyncio.Task | None = None
    LOGGER.info("App is starting up. Creating background worker...")

    # Install blocking detector early to catch event loop blocks
    try:
        install_blocking_detector()
        LOGGER.info("✅ Blocking detector installed")
    except Exception:
        LOGGER.exception("⚠️ Failed to install blocking detector", exc_info=True)

    try:
        # Everything before the first `yield` is "startup".
        # If anything fails here, asynccontextmanager otherwise surfaces only
        # "RuntimeError: generator didn't yield" without the root cause.
        worker_task = asyncio.create_task(worker(), name="slack_background_worker")
        worker_task.add_done_callback(_log_task_result)

        # Idempotent ambient SQL cron setup (runs on every deploy/restart)
        await ensure_ambient_cron_exists()

        yield
    except Exception:
        # This is the money line: you'll now see the real startup exception.
        LOGGER.exception("❌ Lifespan failed (startup or runtime) before/around yield", exc_info=True)
        raise
    finally:
        LOGGER.info("App is shutting down. Stopping background worker...")
        # Stop worker
        try:
            await TASK_QUEUE.put(None)  # sentinel
        except Exception:
            LOGGER.exception("Failed to enqueue worker sentinel", exc_info=True)

        if worker_task is not None:
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
            except Exception:
                LOGGER.exception("Worker raised during shutdown", exc_info=True)

        # Close persistence (moved off atexit; safe to no-op if never created)
        try:
            close_persistence_manager()
        except Exception:
            LOGGER.exception("Failed to close persistence manager on shutdown", exc_info=True)

APP = FastAPI(lifespan=lifespan)

@APP.middleware("http")
async def _log_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    if origin:
        LOGGER.info("HTTP Origin=%s %s %s", origin, request.method, request.url.path)
    return await call_next(request)

# NOTE: Studio/Agent-Server CORS is controlled by langgraph.json -> http.cors.
# This middleware only affects routes served by this FastAPI app.
@APP.post("/")
async def verify_slack(req: Request):
    """
    Handle Slack's URL verification challenge.
    """
    data = await req.json()

    # Respond to Slack verification challenge
    if "challenge" in data:
        return {"challenge": data["challenge"]}

    return {"detail": "Unauthorized"}, 401

# Mount ambient endpoints (for cron / webhooks driving background SQL coverage)
APP.include_router(ambient_router)

@APP.post("/events/slack")
async def slack_endpoint(req: Request):
    body = await req.json()
    if body.get("type") == "url_verification" and "challenge" in body:
        return {"challenge": body["challenge"]}
    return await APP_HANDLER.handle(req)


def _get_text(content: str | list[dict]):
    if isinstance(content, str):
        return content
    else:
        return "".join([block["text"] for block in content if block["type"] == "text"])


def _clean_markdown(text: str) -> str:
    text = re.sub(r"^```[^\n]*\n", "```\n", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"*\1*", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"_\1_", text)
    text = re.sub(r"_([^_]+)_", r"_\1_", text)
    text = re.sub(r"^\s*[-*]\s", "• ", text, flags=re.MULTILINE)
    return text


@APP.post("/callbacks/{thread_id}")
async def webhook_callback(req: Request):
    """
    Handle LangGraph webhook callbacks.
    """
    body = await req.json()
    LOGGER.info(
        "Received webhook callback for %s/%s",
        req.path_params['thread_id'],
        body['thread_id']
    )
    TASK_QUEUE.put_nowait({"type": "callback", "event": body})
    return {"status": "success"}


async def _is_mention(event: SlackMessageData):
    global USER_ID_PATTERN
    if not config.BOT_USER_ID or config.BOT_USER_ID == "fake-user-id":
        config.BOT_USER_ID = (await APP_HANDLER.app.client.auth_test())["user_id"]
        USER_ID_PATTERN = re.compile(rf"<@{config.BOT_USER_ID}>")
    matches = re.search(USER_ID_PATTERN, event["text"])
    return bool(matches)


def _get_thread_id(thread_ts: str, channel: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"SLACK:{thread_ts}-{channel}"))


def _is_dm(event: SlackMessageData):
    if channel_type := event.get("channel_type"):
        return channel_type == "im"
    return False


async def _fetch_thread_history(
    channel_id: str, thread_ts: str
) -> list[SlackMessageData]:
    """
    Fetch all messages in a Slack thread, following pagination if needed.
    """
    LOGGER.info(
        "Fetching thread history for channel=%s, thread_ts=%s", 
        channel_id,
        thread_ts
    )
    all_messages = []
    cursor = None

    while True:
        try:
            if cursor:
                response = await APP_HANDLER.app.client.conversations_replies(
                    channel=channel_id,
                    ts=thread_ts,
                    inclusive=True,
                    limit=150,
                    cursor=cursor,
                )
            else:
                response = await APP_HANDLER.app.client.conversations_replies(
                    channel=channel_id,
                    ts=thread_ts,
                    inclusive=True,
                    limit=150,
                )
            all_messages.extend(response["messages"])
            if not response.get("has_more"):
                break
            cursor = response["response_metadata"]["next_cursor"]
        except Exception as exc:
            LOGGER.exception(
                "Error fetching thread messages: %s",
                exc
            )
            break

    return all_messages


async def _fetch_user_names(user_ids: set[str]) -> dict[str, str]:
    """Fetch and cache Slack display names for user IDs."""
    uncached_ids = [uid for uid in user_ids if uid not in USER_NAME_CACHE]
    if uncached_ids:
        tasks = [APP_HANDLER.app.client.users_info(user=uid) for uid in uncached_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for uid, result in zip(uncached_ids, results):
            if isinstance(result, Exception):
                LOGGER.warning(
                    "Failed to fetch user info for %s: %s",
                    uid,
                    result
                )
                continue
            user_obj = result.get("user", {})
            profile = user_obj.get("profile", {})
            display_name = (
                profile.get("display_name") or profile.get("real_name") or uid
            )
            USER_NAME_CACHE[uid] = display_name
    return {uid: USER_NAME_CACHE[uid] for uid in user_ids if uid in USER_NAME_CACHE}


async def _build_contextual_message(event: SlackMessageData) -> str:
    """Build a message with thread context, using display names for all users."""
    thread_ts = event.get("thread_ts") or event["ts"]
    channel_id = event["channel"]

    history = await _fetch_thread_history(channel_id, thread_ts)
    included = []
    for msg in reversed(history):
        if msg.get("bot_id") == config.BOT_USER_ID:
            break
        included.append(msg)

    all_user_ids = set()
    for msg in included:
        all_user_ids.add(msg.get("user", "unknown"))
        all_user_ids.update(MENTION_REGEX.findall(msg["text"]))

    all_user_ids.add(event["user"])
    all_user_ids.update(MENTION_REGEX.findall(event["text"]))

    user_names = await _fetch_user_names(all_user_ids)

    def format_message(msg: SlackMessageData) -> str:
        text = msg["text"]
        user_id = msg.get("user", "unknown")

        def repl(match: re.Match) -> str:
            uid = match.group(1)
            return user_names.get(uid, uid)

        replaced_text = MENTION_REGEX.sub(repl, text)
        speaker_name = user_names.get(user_id, user_id)

        return (
            f'<slackMessage user="{speaker_name}">' f"{replaced_text}" "</slackMessage>"
        )

    context_parts = [format_message(msg) for msg in reversed(included)]
    new_message = context_parts[-1]
    preceding_context = "\n".join(context_parts[:-1])

    contextual_message = (
        (("Preceding context:\n" + preceding_context) if preceding_context else "")
        + "\n\nNew message:\n"
        + new_message
    )
    return contextual_message


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("langgraph_slack.server:APP", host="0.0.0.0", port=8080)
