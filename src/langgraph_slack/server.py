# src/langgraph_slack/server.py
"""
This is the slack server interface

MULTI-TENANT SUPPORT (2026-01-21):
Added /slack/event endpoint for CollectiWise Router.
- Router forwards events with X-Slack-Bot-Token header
- Token flows through task queue → LangGraph metadata → callback
- Callback uses dynamic token for workspace-specific responses
- Existing /events/slack endpoint preserved for backward compatibility
"""
import src.langgraph_slack.patch_typing  # must run before any Pydantic model loading
import asyncio
import contextlib
import logging
import os
import re
import json
import uuid
from urllib.parse import urlparse
from typing import Any, Awaitable, Callable, Optional
from typing_extensions import TypedDict
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from langgraph_sdk import get_client
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient  # NEW: for dynamic bot tokens
from langgraph_slack import config
# Ambient HTTP endpoints (closed-source)
from pro.http.ambient import router as ambient_router
from pro.http.cron_lifecycle import ensure_ambient_cron_exists
from pro.persistence import close_persistence_manager
from pro.utils.blocking_detector import install_blocking_detector
# Impact-report feature surface (kept at PARITY with server_mit — the deployed
# self-hosted module — so the two receivers do not drift):
from pro.slack_app.agent_launch import register_agent_launcher
from pro.slack_app.file_upload import claim_file_for_processing
from pro.slack_app.impact_report_commands import build_impact_report_task
from pro.slack_app.assumption_commands import build_assumptions_task

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

TASK_QUEUE: asyncio.Queue = asyncio.Queue()

# Shared with src/langgraph_slack/server_mit.py via the canonical module.
# Approach C (May 2026 fix for the VIEBEG bug) lives there; see that
# module's docstring.
from src.langgraph_slack.contextual_message import (  # noqa: E402
    MENTION_REGEX,
    USER_NAME_CACHE,
    build_contextual_message,
    fetch_user_names,
    resolve_user_mentions,
)

# Interrupt/resume: REAL Slack message ts → {thread_id, channel_id, bot_token, ...}
# (keyed on the real posted ts, never a fabricated anchor — see _valid_slack_ts).
_INTERRUPT_THREAD_MAP: dict[str, dict[str, Any]] = {}
# DM fallback: channel_id → the pending interrupt's real posted ts. A 1:1 DM
# reply is top-level (no thread_ts) so it can't match the ts-keyed map; resume by
# channel (DM-only). Mirrors server_mit (the two server modules are deliberate
# deployment variants that each carry their own copies of these helpers).
_CHANNEL_PENDING_INTERRUPT: dict[str, str] = {}


# Single source of truth for "is this a real Slack ts?" — shared by every Slack
# post sink (server, server_mit, reflective ack) so the rule cannot drift.
from pro.slack_app.slack_ts import valid_slack_ts as _valid_slack_ts  # noqa: E402


async def _post_resume_ack(channel_id, thread_ts, bot_token, user_reply, question=None):
    """Immediately acknowledge the human's reply on resume — as the SAME agent,
    continuing — BEFORE the long work (UX). FIXED: always thank them for the
    answer that unblocks the work + warn it's now underway and will take a bit.
    FLEXIBLE: the wording, in the USER'S language, grounded on the actual question
    asked (never assuming a task type). AGENT-rendered (no fixed strings — the
    prompt is LLM INPUT; the post is the model output); thread_ts guarded; best-
    effort. Cloud variant of server_mit._post_resume_ack. (The cloud thread state
    is remote, so the working-context enrichment server_mit does is a follow-up
    here; the question + reply still ground it.)"""
    try:
        from langchain_core.messages import (
            HumanMessage as _HM, SystemMessage as _SM,
        )
        _prompt = [_SM(content=(
            "You are continuing a conversation as the same assistant: a human just "
            "answered a clarifying question that was BLOCKING your work, so you can "
            "now resume. In ONE or TWO short sentences, in the SAME LANGUAGE as "
            "their reply: warmly thank them for the answer that unblocks you, and "
            "let them know you're now working on it and it will take a little time. "
            "Be specific to what was actually being clarified; do NOT assume a task "
            "type, and do not promise specific numbers or results you don't have."
        ))]
        if question:
            _prompt.append(_SM(content=f"The clarifying question you had asked: {question}"))
        _prompt.append(_HM(content=(user_reply or "")[:500]))
        _resp = await config.create_llm().ainvoke(_prompt)
        _ack = (_resp.content or "").strip()[:400] if hasattr(_resp, "content") else ""
        if _ack:
            _client = AsyncWebClient(token=bot_token) if bot_token else APP_HANDLER.app.client
            await _client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts if _valid_slack_ts(thread_ts) else None,
                text=_ack,
            )
    except Exception:
        LOGGER.warning("[%s] resume ack failed (non-fatal)", channel_id, exc_info=True)

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
    
    # NEW: Extract bot_token if present (from router-forwarded events)
    bot_token: Optional[str] = task.get("bot_token")
    
    if event_type == "slack_message":
        thread_id = _get_thread_id(
            event.get("thread_ts") or event["ts"], event["channel"]
        )
        channel_id = event["channel"]

        # ═══ Check if this is a reply to an interrupted (paused) graph ═══
        parent_ts = event.get("thread_ts")
        resume_key = None
        if parent_ts and parent_ts in _INTERRUPT_THREAD_MAP:
            resume_key = parent_ts  # threaded reply matches the real posted ts
        elif not parent_ts and _is_dm(event):
            # 1:1 DM top-level reply resumes the channel's pending interrupt
            # (DM-only; a shared channel relies on the thread-ts match above).
            resume_key = _CHANNEL_PENDING_INTERRUPT.get(event["channel"])
        if resume_key and resume_key in _INTERRUPT_THREAD_MAP:
            mapping = _INTERRUPT_THREAD_MAP[resume_key]
            resume_thread_id = mapping["thread_id"]
            user_reply = (event.get("text") or "").strip()
            effective_token = mapping.get("bot_token") or bot_token
            webhook = f"{config.DEPLOYMENT_URL}/callbacks/{resume_thread_id}"

            LOGGER.info("[%s] Resuming interrupted run thread_id=%s", channel_id, resume_thread_id)

            # UX: thank the user + say we're continuing, BEFORE the long resumed run.
            await _post_resume_ack(
                channel_id, resume_key, effective_token, user_reply,
                question=mapping.get("question"),
            )

            await LANGGRAPH_CLIENT.runs.create(
                thread_id=resume_thread_id,
                assistant_id=config.ASSISTANT_ID,
                command={"resume": {
                    "answers": [user_reply],
                    "status": "answered",
                    "responders": [{"id": event.get("user", "")}],
                    "thread_ts": resume_key,
                }},
                config={**GRAPH_CONFIG},
                metadata={
                    "event": "slack",
                    "channel_id": channel_id,
                    "channel": channel_id,
                    "thread_ts": resume_key,
                    "event_ts": event["ts"],
                    "bot_token": effective_token,
                    "resume": True,
                },
                webhook=webhook,
                multitask_strategy="interrupt",
            )
            return

        # This will connect to the loopback endpoint if not provided.
        webhook = f"{config.DEPLOYMENT_URL}/callbacks/{thread_id}"

        # NEW: Use bot_token-specific client for mention check if available
        if bot_token:
            is_mention = await _is_mention_with_token(event, bot_token)
        else:
            is_mention = await _is_mention(event)
            
        if is_mention or _is_dm(event):
            # NEW: Pass bot_token for user name resolution
            text_with_names = await build_contextual_message(event, bot_token=bot_token)
        else:
            LOGGER.info("Skipping non-mention message")
            return

        # Add the langgraph_auth_user_id to the GRAPH_CONFIG in the configurable field
        user_id = event["user"]
        updated_graph_config = {**GRAPH_CONFIG}
        if "configurable" not in updated_graph_config:
            updated_graph_config["configurable"] = {}
        updated_graph_config["configurable"]["langgraph_auth_user_id"] = user_id

        # ── Slack context for instant acknowledgment ─────────────────
        # These flow through LangGraph configurable → sql_graph bridge
        # → reflective.py ack logic. State.context may get dropped by
        # the swarm subgraph handoff, but configurable always propagates.
        updated_graph_config["configurable"]["bot_token"] = (
            bot_token or os.environ.get("SLACK_BOT_TOKEN", "")
        )
        updated_graph_config["configurable"]["channel_id"] = channel_id
        updated_graph_config["configurable"]["thread_ts"] = (
            event.get("thread_ts") or event["ts"]
        )
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

        # Build metadata - include bot_token if from router (for callback to use)
        run_metadata = {
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
            "conversation_id": conversation_id,
        }
        
        # NEW: Include bot_token in metadata for callback to use
        # This enables multi-tenant responses via router
        if bot_token:
            run_metadata["bot_token"] = bot_token
            LOGGER.info(
                "[%s].[%s] Using router-provided bot_token (redacted: %s...%s)",
                channel_id,
                thread_id,
                bot_token[:10] if len(bot_token) > 15 else "***",
                bot_token[-4:] if len(bot_token) > 15 else "***",
            )

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
                    "bot_token": bot_token or os.environ.get("SLACK_BOT_TOKEN", ""),
                },  # Required for LangMem SummarizationNode
                "conversation_id": conversation_id,  # For cost tracking correlation
            },
            config=updated_graph_config,
            metadata=run_metadata,
            multitask_strategy="interrupt",
            if_not_exists="create",
            webhook=webhook,
        )
        LOGGER.debug(
            "LangGraph run: %s",
            result
        )

    elif event_type == "file_upload":
        # Phase A.0 Stage 3 (PR-A0-4): Slack ``file_shared`` event for
        # expert YAML assumption uploads. All real work — files_info,
        # HTTP download, parse, BQ write, Block Kit response — lives in
        # ``pro.slack_app.file_upload.process_file_upload``. The
        # hot-path lazy handler ``handle_file_shared_dispatch`` only
        # dedups by file_id and enqueues here.
        from pro.slack_app.file_upload import process_file_upload
        await process_file_upload(event, bot_token)

    elif event_type == "assumptions_command":
        # Phase A.0 Stage 4 (PR-A0-5): ``/assumptions`` slash command for
        # inspecting the impact_assumptions BQ store. Subcommands:
        # ``list``, ``show <scenario_id>``, ``help``. All real work — BQ
        # queries, Block Kit rendering, response_url POST — lives in
        # ``pro.slack_app.assumption_commands.process_assumptions_command``.
        # The hot-path lazy handler parses the subcommand structurally
        # and enqueues here.
        from pro.slack_app.assumption_commands import (
            process_assumptions_command,
        )
        await process_assumptions_command(event, bot_token)

    elif event_type == "impact_report_command":
        # Phase A.0 Stage 5 (PR-A0-6d): ``/make-impact-report
        # <scenario_id>``. Background processor renders the impact-
        # report prompt template (the load-bearing artifact at
        # ``pro/prompts/impact_report.py``) and feeds it to the agent
        # via ``LANGGRAPH_CLIENT.runs.create`` — same path Slack
        # messages take. The agent's reply flows back through the
        # existing webhook / chat path.
        from pro.slack_app.impact_report_commands import (
            process_impact_report_command,
        )
        await process_impact_report_command(event, bot_token)

    elif event_type == "callback":
        LOGGER.info(
            "Processing LangGraph callback: %s",
            event['thread_id']
        )

        # ═══ Check if this callback is for an interrupted (paused) run ═══
        run_status = event.get("status")
        if run_status == "interrupted":
            thread_ts = event["metadata"].get("thread_ts") or event["metadata"].get("event_ts")
            cb_channel = event["metadata"].get("channel") or config.SLACK_CHANNEL_ID
            cb_token = event["metadata"].get("bot_token")
            cb_thread_id = event.get("thread_id")

            # Extract the agent's question from the interrupt payload
            state_vals = event.get("values", {})
            interrupts = state_vals.get("__interrupt__", [])
            question = None
            if interrupts:
                payload = interrupts[0] if isinstance(interrupts[0], dict) else getattr(interrupts[0], "value", {})
                if isinstance(payload, dict):
                    question = payload.get("message") or payload.get("title")
                if not question:
                    question = str(payload)

            if question and cb_channel:
                # Forward the agent's question directly — no wrapper template
                if cb_token:
                    client = AsyncWebClient(token=cb_token)
                else:
                    client = APP_HANDLER.app.client

                # SINK GUARD: never send a fabricated/synthetic ts to Slack
                # (Slack rejects a non-numeric thread_ts). Post un-threaded if the
                # metadata thread_ts is our synthetic anchor; capture the REAL
                # posted ts and key the resume map on THAT, never the anchor.
                resp = await client.chat_postMessage(
                    channel=cb_channel,
                    thread_ts=thread_ts if _valid_slack_ts(thread_ts) else None,
                    text=question,
                )
                posted_thread_ts = (resp.get("ts") if resp else None) or (
                    thread_ts if _valid_slack_ts(thread_ts) else None
                )
                if posted_thread_ts:
                    _INTERRUPT_THREAD_MAP[posted_thread_ts] = {
                        "thread_id": cb_thread_id,
                        "channel_id": cb_channel,
                        "bot_token": cb_token,
                        "question": question,  # ack context on resume
                    }
                    # DM fallback: a 1:1 top-level reply resumes by channel.
                    _CHANNEL_PENDING_INTERRUPT[cb_channel] = posted_thread_ts
                LOGGER.info("[%s] Interrupt stored: %s → %s", cb_channel, posted_thread_ts, cb_thread_id)
            return

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

        # NEW: Use dynamic bot_token if present in metadata (multi-tenant support)
        # This allows responding to the correct Slack workspace
        callback_bot_token = event["metadata"].get("bot_token")
        
        if callback_bot_token:
            # Multi-tenant: use workspace-specific token from router
            LOGGER.info(
                "[%s].[%s] Using router-provided bot_token for response (redacted: %s...%s)",
                channel_id,
                thread_ts,
                callback_bot_token[:10] if len(callback_bot_token) > 15 else "***",
                callback_bot_token[-4:] if len(callback_bot_token) > 15 else "***",
            )
            slack_client = AsyncWebClient(token=callback_bot_token)
            await slack_client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=_clean_markdown(_get_text(response_message["content"])),
                metadata={
                    "event_type": "webhook",
                    "event_payload": {"thread_id": event["thread_id"]},
                },
            )
        else:
            # Single-tenant fallback: use global APP_HANDLER client
            await APP_HANDLER.app.client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=_clean_markdown(_get_text(response_message["content"])),
                metadata={
                    "event_type": "webhook",
                    "event_payload": {"thread_id": event["thread_id"]},
                },
            )
            
        # Clean up the interrupt mapping for this conversation. The map is now
        # keyed on the REAL posted ts (not the anchor in `thread_ts` here), so
        # clean up via the channel-pending pointer; also pop the anchor key
        # defensively (legacy / no-op).
        real_pending = _CHANNEL_PENDING_INTERRUPT.pop(channel_id, None)
        if real_pending:
            _INTERRUPT_THREAD_MAP.pop(real_pending, None)
        if thread_ts:
            _INTERRUPT_THREAD_MAP.pop(thread_ts, None)

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
    Handle incoming Slack messages (direct from Slack, not via router).
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

    # NOTE: No bot_token here - this path uses global APP_HANDLER client
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


APP_HANDLER = AsyncSlackRequestHandler(AsyncApp(
    token="xoxb-placeholder-for-multi-tenant-router",
    logger=LOGGER
))
USER_ID_PATTERN = re.compile(rf"<@{config.BOT_USER_ID}>")
APP_HANDLER.app.event("message")(ack=just_ack, lazy=[handle_message])
APP_HANDLER.app.event("app_mention")(
    ack=just_ack,
    lazy=[],
)

# Phase A.0 Stage 3 (PR-A0-4): Slack ``file_shared`` event handler for
# expert YAML assumption uploads. The lazy handler is intentionally tiny
# (file_id dedup + enqueue); all real work happens in the background
# task via ``_process_task``'s ``file_upload`` branch.
#
# Prerequisite: the bot must have the Slack ``files:read`` scope granted
# in the workspace app definition (Slack dashboard, not in code). Without
# it, ``client.files_info`` raises and the user sees an error message.
from pro.slack_app.file_upload import handle_file_shared_dispatch as _handle_file_shared_dispatch  # noqa: E402
APP_HANDLER.app.event("file_shared")(
    ack=just_ack,
    lazy=[_handle_file_shared_dispatch],
)

# Phase A.0 Stage 4 (PR-A0-5): ``/assumptions`` slash command for
# inspecting the impact_assumptions BQ store. The lazy handler is
# intentionally tiny (subcommand parse + enqueue); all real work
# (BQ queries, response_url POST) happens in the background task via
# ``_process_task``'s ``assumptions_command`` branch.
#
# Prerequisite: the ``/assumptions`` slash command MUST be registered
# in the Slack app dashboard (workspace admin task — not in code). The
# command's request URL points at this app's /events/slack endpoint;
# Slack delivers slash-command invocations through the same event
# pipeline as message/file_shared events.
from pro.slack_app.assumption_commands import handle_assumptions_command_dispatch as _handle_assumptions_command_dispatch  # noqa: E402
APP_HANDLER.app.command("/assumptions")(
    ack=just_ack,
    lazy=[_handle_assumptions_command_dispatch],
)

# ── /make-impact-report slash command registration (PR-A0-6d) ──────
# Phase A.0 Stage 5 — the final visible feature on the SPARQL-native
# foundation. User types ``/make-impact-report <scenario_id>``; the
# hot-path handler enqueues a ``impact_report_command`` task; the
# background processor renders the impact-report prompt template and
# feeds it to the agent via ``LANGGRAPH_CLIENT.runs.create`` (same
# entry path as Slack messages). The agent's reply flows back through
# the existing webhook/chat path.
#
# Prerequisite: the ``/make-impact-report`` slash command MUST be
# registered in the Slack app dashboard (workspace admin task — not
# in code). The command's request URL points at this app's
# /events/slack endpoint; Slack delivers slash-command invocations
# through the same event pipeline as other commands.
from pro.slack_app.impact_report_commands import handle_impact_report_command_dispatch as _handle_impact_report_command_dispatch  # noqa: E402
APP_HANDLER.app.command("/make-impact-report")(
    ack=just_ack,
    lazy=[_handle_impact_report_command_dispatch],
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

        # ✅ IMPORTANT: Don't block startup on external/network setup.
        # LangGraph Cloud can mark the deployment unhealthy if import/startup is slow.
        async def _post_startup_setup():
            try:
                # Gate with env so you can disable quickly if needed.
                if os.getenv("AMBIENT_CRON_ENABLED", "true").lower() in {"1","true","yes","y"}:
                    await ensure_ambient_cron_exists()
                    LOGGER.info("✅ ensure_ambient_cron_exists finished")
                else:
                    LOGGER.info("AMBIENT_CRON_ENABLED=false; skipping cron setup")
            except Exception:
                LOGGER.exception("❌ Post-startup setup failed", exc_info=True)

        asyncio.create_task(_post_startup_setup(), name="post_startup_setup")
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
    """
    EXISTING ENDPOINT: Direct Slack → Agent communication.
    Preserved for backward compatibility during transition.
    Uses global APP_HANDLER with environment-configured bot token.
    """
    body = await req.json()
    if body.get("type") == "url_verification" and "challenge" in body:
        return {"challenge": body["challenge"]}
    return await APP_HANDLER.handle(req)


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Multi-Tenant Router Endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@APP.post("/slack/event")
async def slack_event_from_router(req: Request):
    """
    NEW ENDPOINT: CollectiWise Router → Agent communication.
    
    The centralized router forwards Slack events from multiple workspaces.
    Each request includes:
    - X-CollectiWise-Router: true (for auth)
    - X-Slack-Bot-Token: xoxb-... (workspace-specific token)
    - X-Slack-Team-Id: T... (for logging)
    
    The bot_token flows through to the callback handler, enabling
    responses to the correct Slack workspace.
    """
    # Extract router-provided headers
    bot_token = req.headers.get("x-slack-bot-token")
    team_id = req.headers.get("x-slack-team-id", "unknown")
    
    LOGGER.info(
        "🔀 Router-forwarded event from team=%s (bot_token present: %s)",
        team_id,
        bool(bot_token),
    )
    
    body = await req.json()
    
    # Handle Slack URL verification (shouldn't happen via router, but just in case)
    if body.get("type") == "url_verification" and "challenge" in body:
        LOGGER.info("Responding to URL verification challenge via router")
        return {"challenge": body["challenge"]}
    
    # Extract the event
    event = body.get("event", {})
    event_type = event.get("type")
    
    if not event_type:
        LOGGER.warning("No event type in router-forwarded body: %s", body)
        return {"ok": True}  # Ack anyway to satisfy Slack

    # ── File uploads (expert-assumption YAML) — PARITY with server_mit ──
    # Two carriers: a ``file_shared`` event, OR a ``message``/file_share with a
    # ``files`` array. Route BOTH to the file_upload task → process_file_upload;
    # claim_file_for_processing dedups by file_id (ingest once).
    if event_type == "file_shared" or (
        event_type == "message" and event.get("files")
    ):
        if event.get("bot_id"):
            return {"ok": True}
        file_id = await claim_file_for_processing(event)
        if file_id:
            LOGGER.info("🔀 Enqueuing router file_upload: file_id=%s", file_id)
            TASK_QUEUE.put_nowait(
                {"type": "file_upload", "event": event, "bot_token": bot_token}
            )
        return {"ok": True}

    # Filter: only process messages and app_mentions
    if event_type not in ("message", "app_mention"):
        LOGGER.info("Ignoring non-message event type: %s", event_type)
        return {"ok": True}
    
    # Skip bot's own messages
    if event.get("bot_id"):
        LOGGER.info("Ignoring bot message from bot_id=%s", event.get("bot_id"))
        return {"ok": True}
    
    # Skip messages without user (system messages, etc.)
    if not event.get("user"):
        LOGGER.info("Ignoring message without user")
        return {"ok": True}
    
    # For message events, check if it's a mention or DM
    if event_type == "message":
        # Check if it's a mention using the bot token
        if bot_token:
            is_mention = await _is_mention_with_token(event, bot_token)
        else:
            is_mention = await _is_mention(event)
        
        is_dm = _is_dm(event)
        
        if not (is_mention or is_dm):
            LOGGER.info("Ignoring non-mention, non-DM message")
            return {"ok": True}
    
    # Enqueue for processing with bot_token
    LOGGER.info(
        "🔀 Enqueuing router event: type=%s, user=%s, channel=%s, team=%s",
        event_type,
        event.get("user"),
        event.get("channel"),
        team_id,
    )
    
    TASK_QUEUE.put_nowait({
        "type": "slack_message",
        "event": event,
        "bot_token": bot_token,  # NEW: flows through to callback
    })
    
    return {"ok": True}


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

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


@APP.post("/slack/command")
async def slack_command_from_router(req: Request):
    """Multi-tenant router → agent slash-command endpoint (PARITY with
    server_mit). The slack-router forwards slash commands here as JSON
    ``{team_id, command, data, bot_token}`` — ``bot_token`` is in the BODY (the
    command-forward does not send the ``X-Slack-Bot-Token`` header). Map the
    command to its task via the shared builders and enqueue it.
    """
    body = await req.json()
    bot_token = body.get("bot_token")
    command = body.get("command")
    data = body.get("data") or {}
    LOGGER.info(
        "🔀 Router-forwarded command: %s, team=%s", command, body.get("team_id")
    )
    if command == "/make-impact-report":
        task = build_impact_report_task(data, bot_token=bot_token)
    elif command == "/assumptions":
        task = build_assumptions_task(data, bot_token=bot_token)
    else:
        LOGGER.info("Ignoring unknown forwarded command: %s", command)
        return {"ok": True}
    if task is not None:
        TASK_QUEUE.put_nowait(task)
    return {"ok": True}


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
    """Check if event mentions the bot using global APP_HANDLER client."""
    global USER_ID_PATTERN
    if not config.BOT_USER_ID or config.BOT_USER_ID == "fake-user-id":
        config.BOT_USER_ID = (await APP_HANDLER.app.client.auth_test())["user_id"]
        USER_ID_PATTERN = re.compile(rf"<@{config.BOT_USER_ID}>")
    matches = re.search(USER_ID_PATTERN, event["text"])
    return bool(matches)


async def _is_mention_with_token(event: SlackMessageData, bot_token: str) -> bool:
    """
    NEW: Check if event mentions the bot using a specific bot token.
    Used for router-forwarded events where we have workspace-specific token.
    """
    try:
        client = AsyncWebClient(token=bot_token)
        auth_result = await client.auth_test()
        bot_user_id = auth_result["user_id"]
        pattern = re.compile(rf"<@{bot_user_id}>")
        matches = re.search(pattern, event.get("text", ""))
        return bool(matches)
    except Exception as exc:
        LOGGER.warning(
            "Failed to check mention with bot_token: %s. Falling back to global check.",
            exc, 
        exc_info=True)
        return await _is_mention(event)


def _get_thread_id(thread_ts: str, channel: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"SLACK:{thread_ts}-{channel}"))


async def _launch_agent_turn_cloud(
    prompt: str,
    *,
    channel_id: Optional[str],
    thread_anchor_ts: str,
    bot_token: Optional[str] = None,
    user_id: Optional[str] = None,
    scenario_id: Optional[str] = None,
) -> None:
    """Cloud (LangGraph Platform) agent-turn launcher: feed the rendered prompt
    as a ``user`` turn via ``LANGGRAPH_CLIENT.runs.create``; the reply flows
    back through the ``/callbacks`` webhook. Registered with the
    deployment-agnostic ``launch_agent_turn`` registry. This is the block that
    used to live inline in ``process_impact_report_command`` — extracted so the
    processor (and the upload tutorial) are deployment-agnostic. The MIT
    counterpart (direct graph) is ``server_mit._launch_agent_turn_mit``.
    """
    thread_id = _get_thread_id(thread_anchor_ts, channel_id or "")
    webhook = f"{config.DEPLOYMENT_URL}/callbacks/{thread_id}"
    conversation_id = f"slack_{thread_id}"
    effective_bot_token = bot_token or os.environ.get("SLACK_BOT_TOKEN", "")

    updated_config = {**GRAPH_CONFIG}
    if "configurable" not in updated_config:
        updated_config["configurable"] = {}
    updated_config["configurable"]["langgraph_auth_user_id"] = user_id or ""
    updated_config["configurable"]["bot_token"] = effective_bot_token
    updated_config["configurable"]["channel_id"] = channel_id or ""
    updated_config["configurable"]["thread_ts"] = thread_anchor_ts
    updated_config["configurable"]["conversation_id"] = conversation_id
    if scenario_id:
        updated_config["configurable"]["scenario_id"] = scenario_id

    run_metadata = {
        "event": "slack",
        "slack_event_type": "agent_turn",
        "bot_user_id": config.BOT_USER_ID,
        "slack_user_id": user_id or "",
        "channel_id": channel_id or "",
        "channel": channel_id or "",
        # Deterministic anchor (NOT raw thread_ts, None outside a thread): the
        # interrupted-callback branch keys ask_human resume on metadata.thread_ts.
        "thread_ts": thread_anchor_ts,
        "conversation_id": conversation_id,
    }
    if scenario_id:
        run_metadata["scenario_id"] = scenario_id
    if effective_bot_token:
        run_metadata["bot_token"] = effective_bot_token

    await LANGGRAPH_CLIENT.runs.create(
        thread_id=thread_id,
        assistant_id=config.ASSISTANT_ID,
        input={
            "messages": [{"role": "user", "content": prompt}],
            "context": {
                "slack_user_id": user_id or "",
                "channel_id": channel_id or "",
                "thread_id": thread_id,
                "thread_ts": thread_anchor_ts,
                "bot_token": effective_bot_token,
            },
            "conversation_id": conversation_id,
        },
        config=updated_config,
        metadata=run_metadata,
        multitask_strategy="interrupt",
        if_not_exists="create",
        webhook=webhook,
    )


# Register the cloud launcher with the deployment-agnostic registry at import.
register_agent_launcher(_launch_agent_turn_cloud)


def _is_dm(event: SlackMessageData):
    if channel_type := event.get("channel_type"):
        return channel_type == "im"
    return False


# Note: _fetch_thread_history / _fetch_user_names / _build_contextual_message
# used to live here. Approach C (May 2026) consolidated them into
# src/langgraph_slack/contextual_message.py and dropped the thread-history
# wrapping. The LangGraph checkpointer keeps prior turns in state.messages
# across messages — see that module's docstring for full context.


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("langgraph_slack.server:APP", host="0.0.0.0", port=8080)