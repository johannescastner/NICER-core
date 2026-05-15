# src/langgraph_slack/deepseek_utils.py
"""
DeepSeek API utilities for rate limiting, balance checking, and error handling.
Provides robust exponential backoff and DeepSeek-specific optimizations.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Optional,
    Tuple,
)
import httpx
import requests


logger = logging.getLogger(__name__)

# ----------------- Exceptions -----------------
class DeepSeekRateLimitError(Exception):
    """Raised when rate-limiting persists after all retries."""


class DeepSeekInsufficientBalanceError(Exception):
    """Raised when balance is known to be insufficient."""


# ----------------- Typed transient detection -----------------
# Bundle E Fix 1 (2026-05-14, PR-A0-9): typed-exception classification.
# Replaces the substring-matching classifier that originally lived inside
# both retry loops (and violated the project's "never use regex / no
# substring matching" rule). Mirrors the typed pattern in
# ``pro/ml_inference/client.py::_is_transient_error`` and the typed
# ``_is_transient_http_exc`` in ``src/graphs/memory.py:250-263``.
#
# Closes Bug #76: ``httpx.RemoteProtocolError`` (peer-closed mid-stream,
# 12 firings in v38) had no status code and no matching substring under
# the prior classifier, so it fell through unretried and crashed the SQL
# agent at sql_graph.py:1238 (1 fatal crash + 1 graceful DSPy fallback in
# v38). Now classified as transient and retried.
_TRANSIENT_HTTPX_EXCEPTIONS: Tuple[type, ...] = (
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.ConnectError,
    httpx.NetworkError,            # parent of ReadError, WriteError, CloseError
    httpx.RemoteProtocolError,     # peer-closed-mid-stream (Bug #76)
)

# HTTP status codes that warrant retry. Aligned with
# ``pro/ml_inference/client.py:RETRY_STATUS_CODES`` so DeepSeek and Modal
# retry the same status families. (DeepSeek's 402 = insufficient balance
# is a HARD failure handled separately below; not retried.)
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


# ----------------- Optional balance checker -----------------
def make_balance_checker(*, api_key: Optional[str], base_url: Optional[str]) -> Callable[[], bool]:
    """
    Return a callable that best-effort checks whether DeepSeek balance is sufficient.
    If api_key/base_url are missing or the call fails, it returns True (don't block).
    """

    if not api_key or not base_url or not requests:
        # No credentials or requests not available → don't block retries
        def _noop() -> bool:
            return True
        return _noop

    base_url = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{base_url}/user/balance"

    def _check() -> bool:
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json() or {}
            if not data.get("is_available", True):
                return False
            for b in (data.get("balance_infos") or []):
                cur = (b.get("currency") or "").upper()
                total = float(b.get("total_balance", 0) or 0)
                if cur == "USD":
                    return total >= 1.0
                if cur == "CNY":
                    # very rough fallback conversion
                    return (total / 7.2) >= 1.0
        except Exception as e:
            logger.debug("DeepSeek balance check failed: %s", e)
            return True  # don't block retries on telemetry errors
        return True

    return _check


# ----------------- Backoff core -----------------
@dataclass(frozen=True)
class BackoffConfig:
    "the configuration class for backoff"
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    factor: float = 2.0
    jitter: bool = True
    # Optional: inspect Retry-After like semantics if your exception carries a response
    respect_retry_after: bool = True

def _status_code(exc: Exception) -> Optional[int]:
    """Best-effort extraction of an HTTP-ish status code from various SDK errors."""
    for attr in ("status_code", "status", "http_status", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(exc, "response", None)
    if resp is not None:
        for attr in ("status", "status_code"):
            try:
                v = getattr(resp, attr)
                if isinstance(v, int):
                    return v
            except Exception:
                pass
    return None


def _classify_deepseek_error(exc: Exception) -> Tuple[bool, bool, Optional[int]]:
    """Typed classification of a DeepSeek/HTTP error.

    Replaces the substring-matching classifier introduced in the file's
    original commit (``8cf7fd8``, 2025-11-29). The substring approach
    violated the project's "never use regex / no substring matching"
    ironclad rule and missed ``httpx.RemoteProtocolError`` (Bug #76, 12
    v38 firings, 1 fatal SQL-agent crash).

    Returns:
        Tuple of ``(is_retryable, is_402_balance, status_code)``:

        - ``is_retryable``: True if the caller should backoff and retry.
          True for typed transient httpx exceptions
          (``_TRANSIENT_HTTPX_EXCEPTIONS``) OR an HTTP status code in
          ``_RETRYABLE_STATUS_CODES`` (429/500/502/503/504).
        - ``is_402_balance``: True if this is DeepSeek's insufficient-
          balance signal (HTTP 402). Callers raise
          ``DeepSeekInsufficientBalanceError`` instead of retrying.
        - ``status_code``: The HTTP status code (or ``None`` when the
          error is purely connection-level, e.g. ``RemoteProtocolError``).

    The function is the single source of truth for "is this DeepSeek
    error retryable" — both sync and async retry loops consume it.
    """
    code = _status_code(exc)
    if isinstance(exc, _TRANSIENT_HTTPX_EXCEPTIONS):
        return (True, False, code)
    if code == 402:
        return (False, True, code)
    if code in _RETRYABLE_STATUS_CODES:
        return (True, False, code)
    return (False, False, code)

def _compute_sleep(
    delay: float,
    cfg: BackoffConfig,
    exc: Exception
) -> float:
    # Try to respect Retry-After style hints *if* the exception exposes them
    # (many SDKs attach HTTP response objects).
    if cfg.respect_retry_after:
        retry_after = None
        # Very defensive introspection to avoid SDK coupling:
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                ra = response.headers.get("Retry-After") if hasattr(response, "headers") else None
                if ra:
                    try:
                        retry_after = float(ra)
                    except Exception:
                        retry_after = None
            except Exception:
                retry_after = None
        if isinstance(retry_after, (int, float)) and retry_after >= 0:
            return min(float(retry_after), cfg.max_delay)

    # Standard exponential backoff with (full) jitter
    sleep = min(delay, cfg.max_delay)
    if cfg.jitter:
        sleep = random.uniform(0, sleep)  # full jitter
    return max(0.0, min(sleep, cfg.max_delay))


def deepseek_exponential_backoff(
    cfg: BackoffConfig = BackoffConfig(),
    *,
    balance_checker: Optional[Callable[[], bool]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for DeepSeek-related calls. Retries on 429/503/500-ish errors,
    optionally checks balance on rate-limit errors.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args, **kwargs) -> Any:
            delay = cfg.base_delay
            for attempt in range(1, cfg.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Bundle E Fix 1 (2026-05-14): typed classification.
                    is_retryable, is_402_balance, code = _classify_deepseek_error(e)

                    if is_402_balance:
                        raise DeepSeekInsufficientBalanceError(str(e)) from e

                    if not is_retryable:
                        # non-retryable — propagate the typed exception
                        # so callers can discriminate downstream
                        raise

                    if attempt == cfg.max_retries:
                        raise DeepSeekRateLimitError(
                            f"Retries exhausted after {cfg.max_retries}: {e}"
                        ) from e

                    # Rate-limit (429) optionally probes balance to
                    # convert sustained 429s into a clearer
                    # InsufficientBalanceError when the upstream cause
                    # is actually an out-of-credit account.
                    if code == 429 and balance_checker and not balance_checker():
                        raise DeepSeekInsufficientBalanceError(
                            "Insufficient balance detected during rate limiting"
                        ) from e

                    sleep = _compute_sleep(delay, cfg, e)
                    logger.warning(
                        "DeepSeek backoff for %s (attempt %d/%d, code=%s, type=%s). Sleeping %.2fs …",
                        getattr(func, "__qualname__", repr(func)),
                        attempt,
                        cfg.max_retries,
                        (code if code is not None else "?"),
                        type(e).__name__,
                        sleep
                    )
                    time.sleep(sleep)
                    delay = min(delay * cfg.factor, cfg.max_delay)
                    continue
        return wrapper
    return decorator


async def async_deepseek_exponential_backoff(
    func: Callable[..., Any],
    *args,
    cfg: BackoffConfig = BackoffConfig(),
    balance_checker: Optional[Callable[[], bool]] = None,
    **kwargs
) -> Any:
    """Async variant of the above. Shares the typed classifier with the
    sync decorator so both paths retry the SAME set of errors — closing
    the prior bug where a retryable error caught by one path was missed
    by the other due to substring-classifier drift."""
    delay = cfg.base_delay
    for attempt in range(1, cfg.max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            # Bundle E Fix 1 (2026-05-14): same typed classifier as sync path.
            is_retryable, is_402_balance, code = _classify_deepseek_error(e)

            if is_402_balance:
                raise DeepSeekInsufficientBalanceError(str(e)) from e
            if not is_retryable:
                raise
            if attempt == cfg.max_retries:
                raise DeepSeekRateLimitError(
                    f"Retries exhausted after {cfg.max_retries}: {e}"
                ) from e
            if code == 429 and balance_checker and not balance_checker():
                raise DeepSeekInsufficientBalanceError(
                    "Insufficient balance during rate limiting"
                ) from e
            sleep = _compute_sleep(delay, cfg, e)
            logger.warning(
                "DeepSeek backoff for %s (attempt %d/%d, code=%s, type=%s). Sleeping %.2fs …",
                getattr(func, "__qualname__", repr(func)),
                attempt,
                cfg.max_retries,
                (code if code is not None else "?"),
                type(e).__name__,
                sleep
            )
            await asyncio.sleep(sleep)
            delay = min(delay * cfg.factor, cfg.max_delay)
            continue


# ----------------- Thin LLM wrapper -----------------
def wrap_llm_with_deepseek_backoff(
    llm,
    *,
    provider: str,
    enable_backoff: bool = True,
    cfg: BackoffConfig = BackoffConfig(),
    balance_checker: Optional[Callable[[], bool]] = None,
):
    """
    Wrap a LangChain LLM-like object so its .invoke() uses DeepSeek backoff.
    No imports from project config; fully decoupled.
    """
    if not enable_backoff or provider.lower() != "deepseek":
        return llm

    class _Wrapper:
        def __init__(self, inner):
            self.llm = inner

        @deepseek_exponential_backoff(cfg, balance_checker=balance_checker)
        def invoke(self, *a, **kw):
            "the invoke method"
            return self.llm.invoke(*a, **kw)

        async def ainvoke(self, *a, **kw):
            """Async variant if the underlying model supports it."""
            if not hasattr(self.llm, "ainvoke"):
                raise AttributeError("Underlying LLM has no 'ainvoke'")
            return await async_deepseek_exponential_backoff(
                self.llm.ainvoke,
                *a,
                cfg=cfg,
                balance_checker=balance_checker,
                **kw
            )

        def __getattr__(self, name):
            return getattr(self.llm, name)

    return _Wrapper(llm)
