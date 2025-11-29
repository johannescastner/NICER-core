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
    Optional
)
import requests


logger = logging.getLogger(__name__)

# ----------------- Exceptions -----------------
class DeepSeekRateLimitError(Exception):
    """Raised when rate-limiting persists after all retries."""


class DeepSeekInsufficientBalanceError(Exception):
    """Raised when balance is known to be insufficient."""


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
                    msg = str(e).lower()
                    code = _status_code(e)
                    is_rate = (code == 429) or ("rate limit" in msg)
                    is_402  = (code == 402) or ("insufficient balance" in msg)
                    is_503  = (code == 503) or (
                        "server overloaded" in msg
                    ) or ("temporarily unavailable" in msg)
                    is_500  = (code == 500) or ("internal server error" in msg)


                    if is_402:
                        raise DeepSeekInsufficientBalanceError(str(e)) from e


                    if not (is_rate or is_503 or is_500):
                        # non-retryable
                        raise

                    if attempt == cfg.max_retries:
                        raise DeepSeekRateLimitError(
                            f"Retries exhausted after {cfg.max_retries}: {e}"
                        ) from e


                    if is_rate and balance_checker and not balance_checker():
                        raise DeepSeekInsufficientBalanceError(
                            "Insufficient balance detected during rate limiting"
                        ) from e

                    sleep = _compute_sleep(delay, cfg, e)
                    logger.warning(
                        "DeepSeek backoff for %s (attempt %d/%d, code=%s). Sleeping %.2fs …",
                        getattr(func, "__qualname__", repr(func)),
                        attempt,
                        cfg.max_retries,
                        (code if code is not None else "?"),
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
    """Async variant of the above."""
    delay = cfg.base_delay
    for attempt in range(1, cfg.max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            code = _status_code(e)
            is_rate = (code == 429) or ("rate limit" in msg)
            is_402  = (code == 402) or ("insufficient balance" in msg)
            is_503  = (code == 503) or (
                "server overloaded" in msg
            ) or ("temporarily unavailable" in msg)
            is_500  = (code == 500) or ("internal server error" in msg)

            if is_402:
                raise DeepSeekInsufficientBalanceError(str(e)) from e
            if not (is_rate or is_503 or is_500):
                raise
            if attempt == cfg.max_retries:
                raise DeepSeekRateLimitError(
                    f"Retries exhausted after {cfg.max_retries}: {e}"
                ) from e
            if is_rate and balance_checker and not balance_checker():
                raise DeepSeekInsufficientBalanceError(
                    "Insufficient balance during rate limiting"
                ) from e
            sleep = _compute_sleep(delay, cfg, e)
            logger.warning(
                "DeepSeek backoff for %s (attempt %d/%d, code=%s). Sleeping %.2fs …",
                getattr(func, "__qualname__", repr(func)),
                attempt,
                cfg.max_retries,
                (code if code is not None else "?"),
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
