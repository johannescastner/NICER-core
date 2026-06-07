"""Shared HTTP Retry-After header parsing.

Consolidates two byte-identical helpers that existed in:

  * ``pro/canonical/wikidata_client.py:135-162`` (``_parse_retry_after``)
  * ``src/graphs/memory.py:221-243``       (``_parse_embed_retry_after``)

Both were introduced separately (Wikidata helper by commit ``9e23a0c``,
2026-05-07; Modal-embed helper by commit ``013b2b7``, 2026-04-14 — Bug 24
fix) and the memory.py docstring explicitly noted they should mirror each
other. The wikidata version was a strict ``str | None`` signature; the
memory version added an ``isinstance`` guard so non-string headers could
flow through unharmed. This consolidation picks the **permissive** shape
so callers don't have to coerce headers themselves.

Used by all three external-service retry layers (Wikidata SPARQL, Modal
embedding, Modal chart-validator via ``pro/ml_inference/client.py::retry_post``)
so that ``Retry-After`` semantics stay identical across services.

RFC 7231 §7.1.3 specifies two valid forms:

  * **delta-seconds:** bare decimal integer (e.g. ``"30"``)
  * **HTTP-date:**     RFC 1123 absolute date
    (e.g. ``"Wed, 14 May 2026 18:25:00 GMT"``)

Both forms are accepted. When the header is absent, malformed, or in the
past, the caller's ``fallback`` value is returned (or ``0`` if the date
is past-relative).
"""
from __future__ import annotations

from typing import Any


def parse_retry_after(hdr: Any, fallback: float) -> float:
    """Parse RFC 7231 ``Retry-After`` (delta-seconds or HTTP-date).

    Args:
        hdr: the header value. Accepts ``None``, ``str``, or any
            object whose ``str()`` representation is parseable.
            None/empty returns ``fallback``.
        fallback: seconds-to-wait to return when the header is
            absent or unparseable. Callers typically pass an
            exponential-backoff value.

    Returns:
        Non-negative float — seconds to wait. ``0.0`` is possible when
        the HTTP-date form is in the past (caller should treat as
        "retry immediately").
    """
    if not hdr:
        return fallback
    hdr_s = hdr.strip() if isinstance(hdr, str) else str(hdr).strip()
    try:
        return float(hdr_s)
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        import datetime as _dt
        target = parsedate_to_datetime(hdr_s)
        now = (
            _dt.datetime.now(target.tzinfo)
            if target.tzinfo is not None
            else _dt.datetime.utcnow()
        )
        return max((target - now).total_seconds(), 0.0)
    except Exception:
        return fallback
