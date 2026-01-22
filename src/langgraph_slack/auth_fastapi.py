# src/langgraph_slack/auth_fastapi.py
"""
FastAPI Authentication Middleware

This replaces langgraph_sdk.Auth with a pure FastAPI dependency.
The logic is identical to auth.py, but uses Depends() instead of @auth.authenticate.

Usage:
    from langgraph_slack.auth_fastapi import verify_request
    
    @app.post("/protected")
    async def protected(request: Request, _: None = Depends(verify_request)):
        ...
"""
import logging
import os
import ipaddress
from typing import Any, Optional

from fastapi import Request, HTTPException

logger = logging.getLogger(__name__)

environment = os.getenv("ENVIRONMENT", "PROD")

STUDIO_ORIGINS = {
    "https://smith.langchain.com",
    b"https://smith.langchain.com",
}


def _hget(headers: Any, key: str) -> Optional[Any]:
    """Read a header value from a headers mapping that might use str or bytes keys."""
    if headers is None:
        return None
    try:
        return headers.get(key) or headers.get(key.lower())
    except Exception:
        pass
    try:
        bkey = key.encode("utf-8")
        return headers.get(bkey)
    except Exception:
        return None


def _to_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", errors="replace")
        except Exception:
            return repr(x)
    return str(x)


def _redact_header_val(k: str, v: Any) -> Any:
    lk = (k or "").lower()
    if lk in {"authorization", "cookie", "set-cookie", "x-api-key", "x_auth_token", "x-slack-bot-token"}:
        s = _to_str(v)
        return (s[:6] + "…" + s[-4:]) if len(s) > 12 else "…"
    return v


def _is_private_or_loopback_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
        return bool(addr.is_private or addr.is_loopback)
    except Exception:
        return False


def _request_client_ip(request: Request) -> str:
    try:
        client = getattr(request, "client", None)
        host = getattr(client, "host", None)
        return _to_str(host)
    except Exception:
        return ""


def _collect_debug(request: Request) -> dict:
    headers = request.headers
    keys = [
        "user-agent", "origin", "referer", "host",
        "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto",
        "x-request-id", "x-cloud-trace-context",
        "authorization", "x-api-key", "x-collectiwise-router",
    ]
    subset = {}
    for k in keys:
        v = headers.get(k)
        if v is not None:
            subset[k] = _redact_header_val(k, v)
    
    return {
        "env": environment,
        "path": str(request.url.path),
        "method": request.method,
        "client_ip": _request_client_ip(request),
        "headers_subset": subset,
    }


async def verify_request(request: Request) -> dict:
    """
    FastAPI dependency for authenticating requests.
    
    This replaces the langgraph_sdk Auth decorator pattern.
    
    Returns:
        dict with identity and permissions
    
    Raises:
        HTTPException(401) if unauthorized
    """
    headers = request.headers
    method = request.method
    path = request.url.path
    
    dbg = _collect_debug(request)
    logger.info("auth.verify_request called", extra=dbg)
    
    # ─────────────────────────────────────────────────────────────
    # 0) ALWAYS allow CORS preflight first
    # ─────────────────────────────────────────────────────────────
    if method == "OPTIONS":
        logger.info("auth: CORS preflight allowed", extra=dbg)
        return {"identity": "cors-preflight", "permissions": ["read", "write"]}
    
    # ─────────────────────────────────────────────────────────────
    # 0.5) DEV MODE: Allow everything
    # ─────────────────────────────────────────────────────────────
    if environment == "DEV":
        logger.info("auth: DEV mode - allowing all requests", extra=dbg)
        return {"identity": "dev-user", "permissions": ["read", "write"]}
    
    # Normalize common fields
    user_agent = headers.get("user-agent", "")
    origin = headers.get("origin", "")
    ua_s = _to_str(user_agent)
    origin_s = _to_str(origin)
    
    # ─────────────────────────────────────────────────────────────
    # 1) Allow internal/queue calls (loopback/private IP)
    # ─────────────────────────────────────────────────────────────
    client_ip = _request_client_ip(request)
    xff = headers.get("x-forwarded-for", "")
    xff_first = xff.split(",")[0].strip() if xff else ""
    
    if _is_private_or_loopback_ip(client_ip) or _is_private_or_loopback_ip(xff_first):
        logger.info("auth: internal request allowed (loopback/private)", extra=dbg)
        return {"identity": "internal", "permissions": ["read", "write"]}
    
    # ─────────────────────────────────────────────────────────────
    # 2) Allow LangSmith Studio by Origin
    # ─────────────────────────────────────────────────────────────
    if origin in STUDIO_ORIGINS or origin_s == "https://smith.langchain.com":
        logger.info("auth: Studio origin allowed", extra=dbg)
        return {"identity": "studio-user", "permissions": ["read", "write"]}
    
    # ─────────────────────────────────────────────────────────────
    # 2.5) Allow CollectiWise Slack Router
    # ─────────────────────────────────────────────────────────────
    router_header = headers.get("x-collectiwise-router", "")
    if router_header.lower() == "true":
        logger.info("auth: CollectiWise Router allowed", extra=dbg)
        return {"identity": "collectiwise-router", "permissions": ["read", "write"]}
    
    # ─────────────────────────────────────────────────────────────
    # 3) Slackbot UA or API key
    # ─────────────────────────────────────────────────────────────
    if ua_s.startswith("Slackbot"):
        logger.info("auth: Slackbot UA allowed", extra=dbg)
        return {"identity": "slackbot", "permissions": ["read", "write"]}
    
    expected = os.getenv("LANGGRAPH_API_KEY", "").strip()
    presented = headers.get("x-api-key", "").strip()
    if expected and presented and presented == expected:
        logger.info("auth: x-api-key allowed", extra=dbg)
        return {"identity": "api-key", "permissions": ["read", "write"]}
    
    # ─────────────────────────────────────────────────────────────
    # 4) Public endpoints (no auth required)
    # ─────────────────────────────────────────────────────────────
    public_paths = {"/", "/health", "/info", "/docs", "/openapi.json"}
    if path in public_paths:
        logger.info("auth: public path allowed", extra=dbg)
        return {"identity": "anonymous", "permissions": ["read"]}
    
    # ─────────────────────────────────────────────────────────────
    # 5) Deny
    # ─────────────────────────────────────────────────────────────
    logger.error(
        f"auth: DENY - origin={origin_s!r} ua={ua_s[:100]!r} client_ip={client_ip!r}",
        extra=dbg
    )
    raise HTTPException(status_code=401, detail="Unauthorized")


# Convenience: optional auth (doesn't raise, returns None on failure)
async def optional_auth(request: Request) -> Optional[dict]:
    """Non-raising auth check. Returns None if auth fails."""
    try:
        return await verify_request(request)
    except HTTPException:
        return None