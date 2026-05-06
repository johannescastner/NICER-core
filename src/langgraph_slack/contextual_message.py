"""Shared helpers for turning a Slack event into the text the LLM consumes.

History — why this module exists:

  Production had two near-identical implementations of contextual-message
  building (one in ``server.py`` for LangGraph Cloud, one in
  ``server_mit.py`` for MIT self-hosted), each with its own
  ``MENTION_REGEX``, ``USER_NAME_CACHE``, ``_fetch_user_names``,
  ``_fetch_thread_history``, and ``_build_contextual_message``. Both
  wrapped the user's new event text into a 37 k-char prompt by
  prepending the entire Slack thread history. The wrapping caused the
  VIEBEG bug: ``last_human`` extraction in ``pro/graphs/sql_graph.py``
  picked up the whole wrapped string, and three downstream consumers
  (Ack, schema retrieval, LLM "Current question") used the head — which
  was the *oldest* thread message, not the user's latest question.

  Approach C (this module) drops the thread-history wrapping. The
  ``AsyncPostgresSaver`` checkpointer keyed on a stable
  ``thread_id = uuid5(NAMESPACE_DNS, f"SLACK:{thread_ts}-{channel}")``
  already keeps prior turns in ``state.messages`` across messages, so
  the wrapping was redundant. It existed as defence-in-depth against an
  earlier checkpointer-state-loss bug ("Tobias/Viebeg outage") that was
  fixed in Feb 2026 via connection-pool health checks + TCP keepalives.

  This module replaces ALL the duplicated logic in the two server files
  with a single canonical implementation. The Slack mention grammar
  ``<@USERID>`` is structurally parsed via a documented Slack regex —
  this is grammar parsing of a structured identifier format, not
  heuristic string matching.

Public API
----------

* ``MENTION_REGEX`` — compiled regex for the Slack mention grammar
* ``USER_NAME_CACHE`` — process-level cache of user-id → display-name
* ``resolve_user_mentions(text, user_names)`` — pure helper; no I/O
* ``fetch_user_names(user_ids, *, bot_token)`` — Slack-API helper;
  caches into ``USER_NAME_CACHE``
* ``build_contextual_message(event, *, bot_token)`` — top-level entry
  used by the Slack handlers; resolves mentions in the event text and
  returns a clean string suitable for ``HumanMessage.content``
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Optional

from slack_sdk.web.async_client import AsyncWebClient


_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slack mention grammar — ``<@USERID>``
#
# Slack's documented format. Parsing this is structural — Slack guarantees
# the wrapping characters don't appear inside identifiers.
# ---------------------------------------------------------------------------

MENTION_REGEX = re.compile(r"<@([A-Z0-9]+)>")


# ---------------------------------------------------------------------------
# Process-level cache for user-id → display-name resolution.
#
# Slack display names are stable enough that caching across the process
# lifetime is safe; the alternative is a Slack API call per mention per
# event. The cache is shared across all server entry points that import
# this module.
# ---------------------------------------------------------------------------

USER_NAME_CACHE: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Pure helper — no I/O, no Slack dependencies. Easy to test in isolation.
# ---------------------------------------------------------------------------


def resolve_user_mentions(text: str, user_names: dict[str, str]) -> str:
    """Replace each ``<@USERID>`` mention in ``text`` with the corresponding
    display name from ``user_names``. Unknown ids are left as-is (Slack
    convention: deleted users may still appear in old mentions; we do not
    crash on those).

    Args:
        text: arbitrary message text. May contain zero or more mentions.
        user_names: mapping of Slack user id → display name. Read-only;
            never mutated by this function.

    Returns:
        ``text`` with mentions replaced by display names.
    """
    if not text:
        return text

    def _repl(match: re.Match) -> str:
        uid = match.group(1)
        return user_names.get(uid, match.group(0))

    return MENTION_REGEX.sub(_repl, text)


# ---------------------------------------------------------------------------
# Slack-API helper — fetches display names for user IDs not yet cached.
# ---------------------------------------------------------------------------


async def fetch_user_names(
    user_ids: set[str], *, bot_token: Optional[str] = None
) -> dict[str, str]:
    """Look up Slack display names for the given user ids.

    Returns a mapping covering every id for which a name was successfully
    fetched (or already cached). Ids that fail to fetch are simply absent
    from the return value — callers handle missing ids via
    ``resolve_user_mentions``'s "leave the original token in place"
    behaviour.

    Args:
        user_ids: set of Slack user ids to resolve.
        bot_token: tenant bot token. If ``None`` and there are uncached
            ids to fetch, the function silently returns the cache it
            already has — it does NOT raise — because mention resolution
            is best-effort.
    """
    uncached = [uid for uid in user_ids if uid not in USER_NAME_CACHE]
    if uncached and bot_token:
        client = AsyncWebClient(token=bot_token)
        results = await asyncio.gather(
            *(client.users_info(user=uid) for uid in uncached),
            return_exceptions=True,
        )
        for uid, result in zip(uncached, results):
            if isinstance(result, Exception):
                _LOGGER.debug("users_info failed for %s: %s", uid, result)
                continue
            user_obj = result.get("user", {})
            profile = user_obj.get("profile", {})
            display_name = (
                profile.get("display_name") or profile.get("real_name") or uid
            )
            USER_NAME_CACHE[uid] = display_name

    return {uid: USER_NAME_CACHE[uid] for uid in user_ids if uid in USER_NAME_CACHE}


# ---------------------------------------------------------------------------
# Top-level entry — used by both server entry points (server.py + server_mit.py)
# ---------------------------------------------------------------------------


async def build_contextual_message(
    event: dict[str, Any], *, bot_token: Optional[str] = None
) -> str:
    """Turn a Slack event into the text the agent's LLM will consume.

    Approach C: returns the event's ``text`` with ``<@USERID>`` mentions
    resolved to display names. The thread-history wrapping that used to
    live here is intentionally absent — see module docstring.

    Args:
        event: the Slack event dict. Only ``event["text"]`` is read.
        bot_token: tenant bot token, used for ``users_info`` calls when
            mentions need resolving.

    Returns:
        The clean event text. Empty string if the event has no text.
    """
    text = event.get("text") or ""
    if not text:
        return ""

    # Find all mention ids in the text. If none, skip the Slack API call
    # entirely — saves a round-trip per plain-text message.
    mentioned_ids = set(MENTION_REGEX.findall(text))
    if not mentioned_ids:
        return text

    user_names = await fetch_user_names(mentioned_ids, bot_token=bot_token)
    return resolve_user_mentions(text, user_names)
