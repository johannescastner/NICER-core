"""
This is where authentication lives
"""
import logging
import os
import ipaddress
from typing import Any, Mapping, Optional
from langgraph_sdk import Auth

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the environment from OS variables (default to "PROD" if not set)
environment = os.getenv("ENVIRONMENT", "PROD")
print(f"Current environment: {environment}")

auth = Auth()

STUDIO_ORIGINS = {
    b"https://smith.langchain.com",
    "https://smith.langchain.com",
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
    if lk in {"authorization", "cookie", "set-cookie", "x-api-key", "x_auth_token"}:
        s = _to_str(v)
        return (s[:6] + "…" + s[-4:]) if len(s) > 12 else "…"
    return v

def _is_private_or_loopback_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
        return bool(addr.is_private or addr.is_loopback)
    except Exception:
        return False

def _request_client_ip(request: Any) -> str:
    """
    Best-effort: Starlette/FastAPI request has request.client.host.
    In some runtimes, request may be None or a lightweight shim.
    """
    try:
        client = getattr(request, "client", None)
        host = getattr(client, "host", None)
        return _to_str(host)
    except Exception:
        return ""

def _collect_debug(headers: Any, request: Any, *, path: Any, method: Any) -> dict:
    # Small, high-signal header subset for diagnosing denials.
    keys = [
        "user-agent",
        "origin",
        "referer",
        "host",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
        "x-request-id",
        "x-cloud-trace-context",
        "authorization",
        "x-api-key",
    ]
    subset = {}
    for k in keys:
        v = _hget(headers, k)
        if v is not None:
            subset[k] = _redact_header_val(k, v)

    client_ip = _request_client_ip(request)
    return {
        "env": environment,
        "path": _to_str(path),
        "method": _to_str(method),
        "client_ip": client_ip,
        "headers_subset": {k: _to_str(v) for k, v in subset.items()},
        # Helpful for catching bytes-vs-str key mismatches:
        "header_key_types": (
            sorted({type(k).__name__ for k in headers.keys()}) if hasattr(headers, "keys") else []
        ),
    }

@auth.authenticate
async def authenticate(request, path, headers, method):
    """
    authentication function for langgraph
    """
    dbg = _collect_debug(headers, request, path=path, method=method)
    logger.info("auth.authenticate called", extra=dbg)

    # Always allow CORS preflight
    if method in (b"OPTIONS", "OPTIONS"):
        return {"identity": "cors-preflight", "permissions": ["read", "write"]}

    # Normalize common fields
    user_agent = _hget(headers, "user-agent")
    origin = (
        _hget(headers, "origin")
        or _hget(headers, "Origin")
    )
    ua_s = _to_str(user_agent)
    origin_s = origin if origin in STUDIO_ORIGINS else _to_str(origin)

    # ─────────────────────────────────────────────────────────────
    # 1) PROD FIX: allow internal/queue calls (loopback/private IP)
    # Many deployments have internal worker/queue components calling the
    # API over 127.0.0.1 / private network and they won't have Slackbot UA
    # or Studio Origin headers.
    # ─────────────────────────────────────────────────────────────
    client_ip = dbg.get("client_ip") or ""
    xff = _to_str(_hget(headers, "x-forwarded-for"))
    # x-forwarded-for may contain a chain: "client, proxy1, proxy2"
    xff_first = (xff.split(",")[0].strip() if xff else "")

    if _is_private_or_loopback_ip(client_ip) or _is_private_or_loopback_ip(xff_first):
        logger.info("auth: internal request allowed (loopback/private)", extra=dbg)
        return {"identity": "internal", "permissions": ["read", "write"]}

    # ─────────────────────────────────────────────────────────────
    # 2) Allow LangSmith Studio by Origin (you already do this)
    # Note: LangGraph supports Studio access even with custom auth; you can
    # disable that via disable_studio_auth if you want.  [oai_citation:1‡LangChain Docs](https://docs.langchain.com/langsmith/set-up-custom-auth)
    # ─────────────────────────────────────────────────────────────
    if origin in STUDIO_ORIGINS:
        logger.info("auth: Studio origin allowed", extra=dbg)
        return {"identity": "studio-user", "permissions": ["read", "write"]}

    # ─────────────────────────────────────────────────────────────
    # 3) DEV convenience: allow everything (but log it explicitly)
    # ─────────────────────────────────────────────────────────────
    if environment == "DEV":
        logger.info("auth: DEV mode allow", extra=dbg)
        return {"identity": "dev-user", "permissions": ["read", "write"]}

    # ─────────────────────────────────────────────────────────────
    # 4) PROD external policy:
    #    - Slackbot UA path OR (optionally) shared API key header.
    # ─────────────────────────────────────────────────────────────
    if ua_s.startswith("Slackbot"):
        logger.info("auth: Slackbot UA allowed", extra=dbg)
        return {"identity": "slackbot", "permissions": ["read", "write"]}

    expected = os.getenv("LANGGRAPH_API_KEY", "").strip()
    presented = _to_str(_hget(headers, "x-api-key") or _hget(headers, "X-Api-Key")).strip()
    if expected and presented and presented == expected:
        logger.info("auth: x-api-key allowed", extra=dbg)
        return {"identity": "api-key", "permissions": ["read", "write"]}

    # ─────────────────────────────────────────────────────────────
    # 5) Deny with a *stack trace* + a clean 401.
    # LangGraph custom auth supports raising an HTTPException for denial.  [oai_citation:2‡LangChain Docs](https://docs.langchain.com/langsmith/set-up-custom-auth)
    # ─────────────────────────────────────────────────────────────
    try:
        raise PermissionError("Auth denied")
    except PermissionError:
        logger.exception("auth: DENY", extra=dbg)

    raise auth.exceptions.HTTPException(status_code=401, detail="Unauthorized")

