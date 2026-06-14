# src/graphs/memory.py
"""
This is where we define the BigQuery memory store,
which connects the langmem longrun memory system to BigQuery.
"""
from __future__ import annotations
from functools import lru_cache
import asyncio
import atexit
import concurrent.futures
import os
import time
from collections.abc import Iterable as IterableABC
import uuid
import logging
import json
from datetime import datetime, timezone
from typing import (
    Any,
    Type,
    Optional,
    Union,
    Literal,
    Tuple,
    List,
    Dict,
    Iterable,
    Sequence,
    AsyncIterator,
    Iterator,
    NamedTuple,
    overload
)
from pydantic import BaseModel, model_validator
import google
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_community.bq_storage_vectorstores.bigquery import BigQueryVectorStore
from langchain_google_community.bq_storage_vectorstores.utils import validate_column_in_bq_schema
from langgraph.store.base import (
    BaseStore,
    Item,
    SearchItem,
    NamespacePath,
    NOT_PROVIDED,
    NotProvided,
    _validate_namespace,
)
from langgraph.store.base.batch import (
    AsyncBatchedBaseStore,
    Op,
    PutOp,
    GetOp,
    SearchOp,
    ListNamespacesOp,
    Result
)
from langmem import create_manage_memory_tool, create_search_memory_tool

from src.canonical.bq_batch import merge_upsert_json_rows


import src.langgraph_slack.patch_typing  # must run before any Pydantic model loading

from src.langgraph_slack.config import (
    PROJECT_ID, DATASET_ID, LOCATION,
    SEMANTIC_TABLE, EPISODIC_TABLE, PROCEDURAL_TABLE,
    CREDENTIALS
)

NamespaceTemplate = Tuple[str, ...]


# ═════════════════════════════════════════════════════════════════════════════
# MEMORY STORE EXECUTOR — module-scope, dedicated to BQ vector-store I/O.
#
# Bug 6 / Bug 7 high-leverage closure (PR-A0-3f). Bug 6 fires from
# ``/usr/lib/python3.12/asyncio/base_events.py:607`` inside
# ``BaseEventLoop.shutdown_default_executor(THREAD_JOIN_TIMEOUT=300)`` —
# triggered ONLY for ``loop._default_executor`` (the executor used by
# ``loop.run_in_executor(None, ...)``).
#
# v35 traceback context (line 4086 of build_logv35_full.txt) — the 30 lines
# BEFORE the warning fire show 4× ``[asearch]`` + 4× ``[abatch]`` calls,
# i.e. memory.py was hammering BQ via the default executor at the moment
# ``asyncio.run()`` exited a per-scenario boundary. By routing every
# ``run_in_executor`` call in this file through a dedicated bounded
# executor, the default executor stays empty for memory I/O and
# ``shutdown_default_executor`` has nothing to drain — eliminating the
# 300s join warning and the downstream ``StreamMessagesHandler.
# on_llm_new_token`` callback flood (Bug 7, 1281 firings in v35).
#
# Sizing rationale:
#   * ``max_workers=6`` matches the empirical peak observed in v34/v35
#     (6 concurrent ``[asearch]`` ops in a single 5-second window during
#     scenario fanout).
#   * Right-sized for actual demand under Cloud Run regional-quota
#     pressure — generous enough to cover observed fanout, conservative
#     enough that 6 threads × ~8MB stack = ~48MB doesn't bloat per-instance
#     memory budget. (Per the user's "unrestrict, not restrict" principle:
#     covers actual demand without preemptively bounding hypothetical
#     futures.)
#   * Mirrors the canonical sizing rationale of
#     ``_HIERARCHICAL_SUMMARIZER_EXECUTOR(max_workers=8)`` and
#     ``_SUMMARIZATION_EXECUTOR(max_workers=4)``.
#
# Lifecycle:
#   * Module-scope singleton — created once at import; shared across all
#     ``BigQueryMemoryStore`` instances and across all 12 executor-using
#     sites in this file (9 ``run_in_executor(None, ...)`` calls + 3
#     per-call ``with ThreadPoolExecutor(max_workers=1)`` blocks lifted
#     in PR-A0-3f).
#   * ``atexit.register(... wait=False, cancel_futures=True)``: at process
#     exit, drop pending memory writes rather than blocking on them. Memory
#     writes are structurally idempotent (atomic MERGE-via-staging path
#     established in commit ``a2fd219``, 2026-05-07) — re-running an aborted
#     write reproduces the same row.
#   * No ``.result()`` timeout on lifted call sites — preserves the
#     ``e079fd7`` principle ("no arbitrary cutoffs on legitimate
#     intelligent work"). The previous per-call ``with ... as ex:``
#     pattern's blocking ``__exit__`` shutdown is what the lift replaces.
#
# Mirror of ``pro/agents/reflective.py::_SUMMARIZATION_EXECUTOR`` and
# ``pro/persistence.py::_TELEMETRY_EXECUTOR`` (added in PR-A0-3e).
# ═════════════════════════════════════════════════════════════════════════════
_MEMORY_STORE_EXECUTOR: concurrent.futures.ThreadPoolExecutor = (
    concurrent.futures.ThreadPoolExecutor(
        max_workers=6,
        thread_name_prefix="memory-store",
    )
)
atexit.register(
    _MEMORY_STORE_EXECUTOR.shutdown,
    wait=False,
    cancel_futures=True,
)


# =============================================================================
# FIX: Table schema pre-creation to prevent "No schema specified" errors
# =============================================================================

def ensure_table_with_schema(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    table_name: str,
    schema: List[bigquery.SchemaField],
) -> bigquery.Table:
    """
    Ensure a BigQuery table exists with the specified schema.
    
    MUST be called BEFORE BigQueryVectorStore is instantiated.
    BigQueryVectorStore._initialize_bq_table() creates empty tables.
    
    Handles:
    1. Table doesn't exist -> Create with schema
    2. Table exists with no schema -> Delete and recreate  
    3. Table exists with schema -> Return as-is
    """
    table_id = f"{project_id}.{dataset_id}.{table_name}"
    
    try:
        existing = client.get_table(table_id)
        
        if len(existing.schema) == 0:
            logger.warning("Table %s has no schema. Recreating.", table_id)
            client.delete_table(table_id)
            raise NotFound(f"Deleted empty table {table_id}")
        
        logger.info("Table %s exists with %d columns", table_id, len(existing.schema))
        return existing
            
    except NotFound:
        logger.info("Creating table %s with %d columns", table_id, len(schema))
        table = bigquery.Table(table_id, schema=schema)
        return client.create_table(table)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ════════════════════════════════════════════════════════════════════════════
# Bug 24 fix (2026-04-14): retry + soft-dedupe for Modal embedding calls.
#
# Root cause: three concurrent vectorstore constructions (semantic/episodic/
# procedural) each called ``embed_query("test")`` as a Pydantic validator
# probe.  They hit Modal through a single shared ``httpx.Client`` connection
# pool, serialising HEAD-of-line on one HTTP/1.1 connection.  Under Modal
# cold-start, the 60s read timeout fired and the unhandled ``ReadTimeout``
# bubbled up through the store builder → asyncio.to_thread → gather →
# LangGraph task, killing the whole scenario.
#
# Fix shape (mirrors ``pro/canonical/wikidata_client`` Bug 22(a) pattern):
#   * bounded exponential-backoff retry for transient network errors
#     (ReadTimeout, ConnectTimeout, NetworkError, PoolTimeout) and HTTP
#     429 / 5xx.  Retry-After header honoured (delta-seconds or HTTP-date)
#     when present, capped at _RETRY_AFTER_MAX_EMBED seconds per attempt.
#   * per-``(model, text)`` result cache: embeddings are deterministic, so
#     caching the Pydantic "test" probe means second + third probes never
#     touch Modal at all — eliminating the connection-pool amplification
#     that turned one slow call into three.  Bounded FIFO (size 512) so
#     the cache can't grow unbounded on normal traffic.
# ════════════════════════════════════════════════════════════════════════════

_EMBED_CACHE_MAX = 512
_embed_cache: "dict[tuple[str, str], list[float]]" = {}


def _embed_cache_get(model: str, text: str):
    return _embed_cache.get((model, text))


def _embed_cache_put(model: str, text: str, embedding):
    if len(_embed_cache) >= _EMBED_CACHE_MAX:
        # FIFO eviction — drop an arbitrary oldest item
        _embed_cache.pop(next(iter(_embed_cache)), None)
    _embed_cache[(model, text)] = embedding


# Retry-After parsing consolidated into shared helper 2026-05-14 (Bundle E
# Fix 2B). The local ``_parse_embed_retry_after`` here was byte-identical
# (modulo isinstance guard) with ``_parse_retry_after`` in
# pro/canonical/wikidata_client.py — the docstring explicitly noted they
# should mirror each other. Both now import from pro.http.retry_after.
# Module-level alias preserves the existing
# ``_parse_embed_retry_after`` name for any external references.
from src.http.retry_after import parse_retry_after as _parse_embed_retry_after


# Upper bound on a single Retry-After wait for Modal embedding.
#
# Modal returns 429 with multi-minute ``Retry-After`` during embedding
# capacity bursts (v38: 870× firings, Bug #77 storm). Embedding is BATCH
# (called from validator probes + async embed_documents); user isn't
# waiting inline, so latency-tolerant. Previous 60.0 cap defeated the
# server's hint and exhausted retries before Modal recovered.
#
# PR-A0-13a (2026-05-14): renamed from _MAX_EMBED_RETRY_WAIT, raised
# default 60→180s, added env-var override. Distinct from exponential-
# backoff fallback (still capped at the same value via min(2^n, cap)
# below — both share this ceiling because the embedding loop doesn't
# separately track an exponential ceiling).
_RETRY_AFTER_MAX_EMBED = float(os.getenv("MODAL_EMBED_RETRY_AFTER_MAX", "180.0"))
_EMBED_MAX_RETRIES = 4


def _is_transient_http_exc(exc) -> bool:
    """True for httpx exceptions that represent transient network failures
    safe to retry (timeouts, connection errors, pool saturation).
    """
    import httpx
    return isinstance(exc, (
        httpx.ReadTimeout,
        httpx.ConnectTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
        httpx.ConnectError,
        httpx.NetworkError,
        httpx.RemoteProtocolError,
    ))


def _sync_modal_post_with_retry(client, url: str, payload: dict):
    """POST to Modal with retry on 429/5xx + transient network errors.

    Raises the last exception / HTTPStatusError when all retries exhausted.
    """
    import httpx
    last_exc: Optional[BaseException] = None
    for attempt in range(_EMBED_MAX_RETRIES):
        try:
            response = client.post(url, json=payload)
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt >= _EMBED_MAX_RETRIES - 1:
                    response.raise_for_status()
                wait = min(
                    _parse_embed_retry_after(
                        response.headers.get("Retry-After"),
                        fallback=2 ** (attempt + 1),
                    ),
                    _RETRY_AFTER_MAX_EMBED,
                )
                # Bundle E Fix 3 (2026-05-14, PR-A0-9): demoted from
                # WARNING to DEBUG. The per-attempt message fired 870×
                # in v38's Modal embed storm (Bug #77) — pure log spam.
                # The give-up signal ("Sync embed_documents failed
                # after retries") stays at ERROR, preserving storm
                # visibility at the right level (one event per failure,
                # not one per attempt).
                logger.debug(
                    "Modal embed HTTP %d (attempt %d/%d), retrying in %.1fs",
                    response.status_code, attempt + 1, _EMBED_MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response
        except httpx.HTTPError as e:
            if not _is_transient_http_exc(e) or attempt >= _EMBED_MAX_RETRIES - 1:
                raise
            last_exc = e
            wait = min(2 ** (attempt + 1), _RETRY_AFTER_MAX_EMBED)
            # Same Bundle E Fix 3 reasoning: transient-network retries
            # don't deserve WARNING-per-attempt either.
            logger.debug(
                "Modal embed transient %s (attempt %d/%d), retrying in %.1fs: %s",
                type(e).__name__, attempt + 1, _EMBED_MAX_RETRIES, wait, e,
            )
            time.sleep(wait)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Modal embed retry loop exhausted without exception")


async def _async_modal_post_with_retry(client, url: str, payload: dict):
    """Async variant of ``_sync_modal_post_with_retry``."""
    import httpx
    last_exc: Optional[BaseException] = None
    for attempt in range(_EMBED_MAX_RETRIES):
        try:
            response = await client.post(url, json=payload)
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt >= _EMBED_MAX_RETRIES - 1:
                    response.raise_for_status()
                wait = min(
                    _parse_embed_retry_after(
                        response.headers.get("Retry-After"),
                        fallback=2 ** (attempt + 1),
                    ),
                    _RETRY_AFTER_MAX_EMBED,
                )
                # Bundle E Fix 3 (2026-05-14, PR-A0-9): see sync sibling.
                logger.debug(
                    "Modal embed HTTP %d (attempt %d/%d), retrying in %.1fs",
                    response.status_code, attempt + 1, _EMBED_MAX_RETRIES, wait,
                )
                await asyncio.sleep(wait)
                continue
            response.raise_for_status()
            return response
        except httpx.HTTPError as e:
            if not _is_transient_http_exc(e) or attempt >= _EMBED_MAX_RETRIES - 1:
                raise
            last_exc = e
            wait = min(2 ** (attempt + 1), _RETRY_AFTER_MAX_EMBED)
            logger.debug(
                "Modal embed transient %s (attempt %d/%d), retrying in %.1fs: %s",
                type(e).__name__, attempt + 1, _EMBED_MAX_RETRIES, wait, e,
            )
            await asyncio.sleep(wait)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Modal embed retry loop exhausted without exception")


class ModalEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings wrapper that calls Modal endpoints.

    Uses DUAL httpx clients to avoid event loop conflicts:
    - Sync httpx.Client for embed_query/embed_documents (called from sync contexts)
    - Async httpx.AsyncClient for aembed_query/aembed_documents (called from async contexts)

    This pattern is identical to how langchain-openai handles the same problem.
    NEVER use asyncio.run() to bridge sync→async - it creates/destroys event loops
    which corrupts httpx connection pools and causes "Event loop is closed" errors.

    Uses bge-base-en-v1.5 (768 dims) for best retrieval quality.
    """
    
    def __init__(self, model: str = "bge-base"):
        """
        Args:
            model: "bge-base" (768 dims, best quality) or "minilm" (384 dims, legacy)
        """
        self.model = model
        self._sync_client = None  # httpx.Client for sync calls
        self._async_client = None  # httpx.AsyncClient for async calls
        self._inference_client = None  # Our Modal inference client (for URL building)
    
    def _get_inference_client(self):
        """Lazy-load the inference client (used for URL building)."""
        if self._inference_client is None:
            from pro.ml_inference.client import get_inference_client
            self._inference_client = get_inference_client()
        return self._inference_client
    
    def _get_sync_client(self):
        """Get or create sync httpx client."""
        if self._sync_client is None:
            import httpx
            self._sync_client = httpx.Client(timeout=60.0)
        return self._sync_client
    
    def _get_async_client(self):
        """Get or create async httpx client."""
        if self._async_client is None:
            import httpx
            self._async_client = httpx.AsyncClient(timeout=60.0)
        return self._async_client
    
    def _get_url(self, endpoint: str) -> str:
        """Get the Modal endpoint URL from the inference client."""
        client = self._get_inference_client()
        
        # Map our endpoint names to the client's URL attributes
        if endpoint == "embed":
            return client.embedding_url
        elif endpoint == "embed_batch":
            return client.embedding_batch_url
        else:
            raise ValueError(f"Unknown endpoint: {endpoint}")
     
    # =========================================================================
    # SYNC METHODS - Use sync httpx.Client, NO asyncio.run()
    # =========================================================================
 
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents (sync, for LangChain compatibility).

        Uses sync httpx.Client - safe to call from any context including
        inside run_in_executor or ThreadPoolExecutor.

        Bug 24 fix: retries transient failures (429/5xx/timeouts) via
        ``_sync_modal_post_with_retry``.  Per-text result cache means
        repeat calls with identical text skip Modal entirely.
        """
        # Serve cache hits first; only POST the uncached subset.
        cached_out: "list[Optional[list[float]]]" = [
            _embed_cache_get(self.model, t) for t in texts
        ]
        missing_idx = [i for i, v in enumerate(cached_out) if v is None]
        if not missing_idx:
            return [v for v in cached_out if v is not None]
        missing_texts = [texts[i] for i in missing_idx]

        sync_http = self._get_sync_client()
        url = self._get_url("embed_batch")
        payload = {"texts": missing_texts, "model": self.model}
        try:
            response = _sync_modal_post_with_retry(sync_http, url, payload)
            data = response.json()
            fetched = self._parse_batch_response(data)
        except Exception as e:
            logger.error("Sync embed_documents failed after retries: %s", e, exc_info=True)
            raise

        for local_i, global_i in enumerate(missing_idx):
            emb = fetched[local_i]
            _embed_cache_put(self.model, texts[global_i], emb)
            cached_out[global_i] = emb
        return [v for v in cached_out if v is not None]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query (sync, for LangChain compatibility).

        Uses sync httpx.Client - safe to call from any context.
        Bug 24 fix: retries + result cache (eliminates the 3-simultaneous-
        probe connection-pool amplification during vectorstore construction).
        """
        cached = _embed_cache_get(self.model, text)
        if cached is not None:
            return cached

        sync_http = self._get_sync_client()
        url = self._get_url("embed")
        payload = {"text": text, "model": self.model}
        try:
            response = _sync_modal_post_with_retry(sync_http, url, payload)
            data = response.json()
            embedding = self._parse_single_response(data)
        except Exception as e:
            logger.error("Sync embed_query failed after retries: %s", e, exc_info=True)
            raise
        _embed_cache_put(self.model, text, embedding)
        return embedding
    
    # =========================================================================
    # ASYNC METHODS - Use async httpx.AsyncClient  
    # =========================================================================
    
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async embed documents - uses native async client.

        Bug 24 fix: retry + cache (same shape as the sync path).
        """
        cached_out: "list[Optional[list[float]]]" = [
            _embed_cache_get(self.model, t) for t in texts
        ]
        missing_idx = [i for i, v in enumerate(cached_out) if v is None]
        if not missing_idx:
            return [v for v in cached_out if v is not None]
        missing_texts = [texts[i] for i in missing_idx]

        async_http = self._get_async_client()
        url = self._get_url("embed_batch")
        payload = {"texts": missing_texts, "model": self.model}
        try:
            response = await _async_modal_post_with_retry(async_http, url, payload)
            data = response.json()
            fetched = self._parse_batch_response(data)
        except Exception as e:
            logger.error("Async aembed_documents failed after retries: %s", e, exc_info=True)
            raise

        for local_i, global_i in enumerate(missing_idx):
            emb = fetched[local_i]
            _embed_cache_put(self.model, texts[global_i], emb)
            cached_out[global_i] = emb
        return [v for v in cached_out if v is not None]

    async def aembed_query(self, text: str) -> list[float]:
        """Async embed query - uses native async client.

        Bug 24 fix: retry + cache.
        """
        cached = _embed_cache_get(self.model, text)
        if cached is not None:
            return cached

        async_http = self._get_async_client()
        url = self._get_url("embed")
        payload = {"text": text, "model": self.model}
        try:
            response = await _async_modal_post_with_retry(async_http, url, payload)
            data = response.json()
            embedding = self._parse_single_response(data)
        except Exception as e:
            logger.error("Async aembed_query failed after retries: %s", e, exc_info=True)
            raise
        _embed_cache_put(self.model, text, embedding)
        return embedding
    
    # =========================================================================
    # RESPONSE PARSING (shared by sync/async)
    # =========================================================================
    
    def _parse_batch_response(self, data) -> list[list[float]]:
        """Parse batch embedding response from Modal."""
        if isinstance(data, list):
            return [
                item.get("embedding", item) if isinstance(item, dict) else item 
                for item in data
            ]
        elif isinstance(data, dict):
            if "embeddings" in data:
                return data["embeddings"]
            elif "results" in data:
                return [r.get("embedding", r) for r in data["results"]]
        logger.warning("Unexpected batch response format: %s", type(data))
        return data
    
    def _parse_single_response(self, data) -> list[float]:
        """Parse single embedding response from Modal."""
        if isinstance(data, dict) and "embedding" in data:
            return data["embedding"]
        elif isinstance(data, list):
            return data
        logger.warning("Unexpected single response format: %s", type(data))
        return data
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    
    def close(self):
        """Close sync client."""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
    
    async def aclose(self):
        """Close async client."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
    
    def __del__(self):
        """Cleanup on garbage collection."""
        if self._sync_client:
            try:
                self._sync_client.close()
            except Exception:
                pass

@lru_cache(maxsize=1)
def get_embedding():
    """
    Lazy embedding factory.
    
    Returns a Modal-based embeddings object that's compatible with LangChain
    but doesn't load any local models.
    
    Uses bge-base-en-v1.5 (768 dims) for 22% better retrieval than MiniLM.
    """
    return ModalEmbeddings(model="bge-base")

@lru_cache(maxsize=1)
def get_bq_client() -> bigquery.Client:
    """
    Lazy BigQuery client factory.
    Avoid doing API/client setup at import time.
    """
    try:
        client = bigquery.Client(credentials=CREDENTIALS, project=PROJECT_ID, location=LOCATION)
        logger.info("BigQuery client initialized successfully.")
        return client
    except Exception as e:
        logger.error("Failed to initialize BigQuery client: %s", e, exc_info=True)
        raise

CONTENT_FIELDS = {
    SEMANTIC_TABLE: "fact",
    EPISODIC_TABLE: "episode",
    PROCEDURAL_TABLE: "procedure",
}

def parse_fqn(fqn: str, default_project: str | None = None) -> Tuple[str, ...]:
    """
    Accepts 'dataset.table' or 'project.dataset.table'.
    Returns (project, dataset, table).
    """
    parts = fqn.split(".")
    if len(parts) == 2:
        if not default_project:
            raise ValueError("Default project required for <dataset>.<table> form")
        project, dataset, table = default_project, parts[0], parts[1]
    elif len(parts) == 3:
        project, dataset, table = parts
    else:
        raise ValueError("FQN must be <dataset>.<table> or <project>.<dataset>.<table>")
    return (project, dataset, table)

def ns_sem_dataset(project: str, dataset: str) -> Tuple[str, ...]:
    return ("metadata", project, dataset)

def ns_sem_table(project: str, dataset: str, table: str) -> Tuple[str, ...]:
    return ("metadata", project, dataset, table)

def ns_sem_column(project: str, dataset: str, table: str, column: str) -> Tuple[str, ...]:
    return ("metadata", project, dataset, table, column)

def ns_join(parts: Tuple[str, ...]) -> str:
    """Serialize for storage (STRING namespace column in BQ)."""
    return ".".join(parts)

def ns_with_bucket(ns: Tuple[str, ...], bucket: str, anchor: str = "episodic") -> Tuple[str, ...]:
    """
    Insert a priority bucket *right after* the 'anchor' segment (e.g., 'episodic').
    Returns a tuple, regardless of input.
    """
    try:
        i = ns.index(anchor)
        return (*ns[:i+1], bucket, *ns[i+1:])
    except ValueError:
        # If anchor isn't present, prepend it (keeps behavior predictable)
        return (anchor, bucket, *ns)

def ns_sem_for_target(target: str, default_project: str | None = PROJECT_ID) -> tuple[str, ...]:
    """
    Accepts: "dataset", "dataset.table", "dataset.table.column",
             or "project.dataset.table[.column]".
    Returns the correct semantic namespace tuple.
    """
    parts = [p for p in (target or "").split(".") if p]
    if not parts:
        raise ValueError("empty target")

    if len(parts) == 1:                    # dataset
        return ns_sem_dataset(default_project, parts[0])
    if len(parts) == 2:                    # dataset.table
        return ns_sem_table(default_project, parts[0], parts[1])
    if len(parts) == 3:                    # dataset.table.column
        return ns_sem_column(default_project, parts[0], parts[1], parts[2])
    if len(parts) == 4:                    # project.dataset.table.column
        return ns_sem_column(parts[0], parts[1], parts[2], parts[3])

    raise ValueError(f"Unrecognized target format: {target!r}")

class Fact(BaseModel):
    """
    satisfying the fact type
    """
    @model_validator(mode="before")
    @classmethod
    def _coerce_string(cls, v):
        if isinstance(v, str):
            return {"content": v}
        return v

    content: str
    # Phase-3 (C2): ``importance`` and ``category`` dropped.
    # They were declared but never populated — only 5 of 731 production
    # rows in Viebeg's semantic memory had either set, and no code
    # consumed them. Per the Phase-3 diagnosis (Item 6) and user
    # decision, these are removed rather than formalized.

class Episode(BaseModel):
    """
    satisfying the episode type
    """
    @model_validator(mode="before")
    @classmethod
    def _coerce_string(cls, v):
        if isinstance(v, str):
            return {"observation": v}
        return v

    observation: Optional[str] = None
    thoughts: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None

class Procedure(BaseModel):
    """
    satisfying the procedure type
    """
    @model_validator(mode="before")
    @classmethod
    def _coerce_string(cls, v):
        if isinstance(v, str):
            return {"name": v}
        return v

    name: Optional[str] = None
    conditions: Optional[str] = None
    steps: Optional[str] = None
    notes: Optional[str] = None

PYDANTIC_MODELS = {
    "fact": Fact,
    "episode": Episode,
    "procedure": Procedure,
}


class PatchedBigQueryVectorStore(BigQueryVectorStore):
    """
    The original BigQueryVectorStore has to be patched
    so that it allows for struct types
    """
    def _normalize_page_content(self, content: Any) -> str:
        if isinstance(content, dict):
            return json.dumps(content)
        return str(content)

    def _create_langchain_documents(
        self,
        search_results: List[List[Any]],
        k: int,
        num_queries: int,
        with_embeddings: bool = False,
    ) -> List[List[List[Any]]]:
        if len(search_results) == 0:
            return [[]]

        result_fields = list(search_results[0].keys())  # type: ignore[attr-defined]
        metadata_fields = [
            x
            for x in result_fields
            if x not in [self.embedding_field, self.content_field, "row_num"]
        ]
        documents = []
        for result in search_results:
            metadata = {
                metadata_field: result[metadata_field]
                for metadata_field in metadata_fields
            }
            document = Document(
                page_content=self._normalize_page_content(
                    result[self.content_field]
                ),
                metadata=metadata,
            )
            if with_embeddings:
                document_record = [
                    document,
                    metadata["score"],
                    result[self.embedding_field],  # type: ignore
                ]
            else:
                document_record = [document, metadata["score"]]
            documents.append(document_record)

        results_docs = [documents[i * k : (i + 1) * k] for i in range(num_queries)]
        return results_docs

    def get_documents(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Union[Dict[str, Any], str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search documents by their ids or metadata values.

        Args:
            ids: List of ids of documents to retrieve from the vectorstore.
            filter: (Optional) A dictionary or a string specifying filter criteria.
                - If a dictionary is provided, it should map column names to their
                corresponding values. The method will generate SQL expressions based
                on the data types defined in `self.table_schema`. The value is enclosed
                in single quotes unless the column is of type "INTEGER" or "FLOAT", in
                which case the value is used directly. E.g., `{"str_property": "foo",
                "int_property": 123}`.
                - If a string is provided, it is assumed to be a valid SQL WHERE clause.
        Returns:
            List of ids from adding the texts into the vectorstore.
        """

        if ids and len(ids) > 0:
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("ids", "STRING", ids),
                ]
            )
            id_expr = f"{self.doc_id_field} IN UNNEST(@ids)"
        else:
            job_config = None
            id_expr = "TRUE"

        where_filter_expr = self._create_filters(filter)

        job = self._bq_client.query(  # type: ignore[union-attr]
            f"""
                    SELECT * FROM `{self.full_table_id}`
                    WHERE {id_expr} AND {where_filter_expr}
                    """,
            job_config=job_config,
        )
        docs: List[Document] = []
        for row in job:
            metadata = {}
            for field in row.keys():
                if field not in [
                    self.embedding_field,
                    self.content_field,
                ]:
                    metadata[field] = row[field]
            metadata["__id"] = row[self.doc_id_field]
            doc = Document(
                page_content=self._normalize_page_content(row[self.content_field]),
                metadata=metadata
            )
            docs.append(doc)
        return docs

    def add_texts_with_embeddings(
        self,
        texts: List[Union[str, dict]],
        embs: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        # Generate IDs if not provided
        if ids is None:
            ids = [uuid.uuid4().hex for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]

        values_dict: List[Dict[str, Any]] = []
        for idx, text, emb, metadata_dict in zip(ids, texts, embs, metadatas):
            record = {
                self.doc_id_field: idx,
                self.embedding_field: emb,
            }

            if self._is_record_column():
                # Use the structured data from metadata_dict for RECORD fields
                if isinstance(metadata_dict.get(self.content_field), dict):
                    record[self.content_field] = metadata_dict[self.content_field]
                else:
                    raise ValueError(
                        "Expected dict for RECORD column %s, got: %s" % (
                            self.content_field,
                            type(metadata_dict.get(self.content_field))
                        )
                    )
            else:
                # For non-RECORD fields, handle as before
                if isinstance(text, dict):
                    record[self.content_field] = json.dumps(text)
                elif isinstance(text, str):
                    record[self.content_field] = text
                else:
                    raise ValueError(
                        f"Expected str or dict for `{self.content_field}`, got: {type(text)}"
                    )

            record.update(metadata_dict)
            values_dict.append(record)

        logger.debug("Type of 'fact' field: %s", type(record[self.content_field]))
        logger.debug("Content of 'fact' field: %s", record[self.content_field])
        table = self._bq_client.get_table(self.full_table_id)
        try:

            # Schema enforcement happens at table definition time; load just sends rows.
            job = self._bq_client.load_table_from_json(values_dict, table)
            logger.debug("Loaded %d row(s) into %s", len(values_dict), self.full_table_id)
            job.result()  # Wait for the job to complete
        except google.api_core.exceptions.GoogleAPIError as e:
            # Handle errors returned by the Google API
            print("Google API error occurred: %s", e)
            logger.debug("Google API error occurred: %s", e)
            raise
        except ValueError as e:
            # Handle value errors, such as schema mismatches
            print("Value error: %s", e)
            logger.debug("Value error: %s", e)
            raise
        except Exception as e:
            # Handle other unexpected exceptions
            print("An unexpected error occurred: %s", e)
            logger.debug("An unexpected error occurred: %s", e)
            raise
        self._validate_bq_table()
        self._logger.debug("Stored %s records in BigQuery.", len(ids))
        self.sync_data()
        return ids

    def _is_record_column(self) -> bool:
        """Check whether the content_field is a RECORD in the BigQuery schema."""
        table = self._bq_client.get_table(self.full_table_id)
        for field in table.schema:
            if field.name == self.content_field:
                return field.field_type.upper() == "RECORD"
        return False

    def _validate_bq_table(self) -> Any:

        table_ref = bigquery.TableReference.from_string(self.full_table_id)

        try:
            # Attempt to retrieve the table information
            table = self._bq_client.get_table(table_ref)
        except NotFound:
            self._logger.debug("Couldn't find table %s", self.full_table_id)
            return

        schema = table.schema.copy()
        if schema:  # Check if table has a schema
            self.table_schema = {field.name: field.field_type for field in schema}
            columns = {c.name: c for c in schema}

            # Validate doc_id_field
            validate_column_in_bq_schema(
                column_name=self.doc_id_field,
                columns=columns,
                expected_types=["STRING"],
                expected_modes=["NULLABLE", "REQUIRED"],
            )

            # Validate content_field based on actual schema
            if self.content_field in columns:
                content_field_type = columns[self.content_field].field_type.upper()
                expected_types = [content_field_type]  # Accept the actual type
                validate_column_in_bq_schema(
                    column_name=self.content_field,
                    columns=columns,
                    expected_types=expected_types,
                    expected_modes=["NULLABLE", "REQUIRED"],
                )
            else:
                raise ValueError(f"Column '{self.content_field}' not found in the table schema.")

            # Validate embedding_field
            validate_column_in_bq_schema(
                column_name=self.embedding_field,
                columns=columns,
                expected_types=["FLOAT", "FLOAT64"],
                expected_modes=["REPEATED"],
            )

            # Validate extra_fields if provided
            if self.extra_fields is None:
                extra_fields = {}
                for column in schema:
                    if column.name not in [
                        self.doc_id_field,
                        self.content_field,
                        self.embedding_field,
                    ]:
                        # Check for unsupported REPEATED mode
                        if column.mode == "REPEATED":
                            raise ValueError(
                                f"Column '{column.name}' is REPEATED. "
                                f"REPEATED fields are not supported in this context."
                            )
                        extra_fields[column.name] = column.field_type
                self.extra_fields = extra_fields
            else:
                for field, type in self.extra_fields.items():
                    validate_column_in_bq_schema(
                        column_name=field,
                        columns=columns,
                        expected_types=[type],
                        expected_modes=["NULLABLE", "REQUIRED"],
                    )

            self._logger.debug("Table %s validated", self.full_table_id)
        return table_ref


# =============================================================================
# Helpers for asearch namespace_prefix → SQL filter translation (B0 fix)
# =============================================================================
#
# Background: ``BaseStore.asearch``'s ``namespace_prefix`` argument is documented
# as scoping the search to items whose stored namespace tuple starts with that
# prefix (``langgraph.store.base.BaseStore.asearch`` docstring; reference
# ``InMemoryStore._filter_items``). Our store persists the namespace as a
# ``.``-joined STRING column (see ``aput``: ``data["namespace"] = ".".join(namespace)``,
# matched by ``ns_join`` at module top), so tuple-prefix matching becomes a
# string-prefix match. ``BigQueryVectorStore._create_filters`` accepts
# ``filter`` either as a dict (column = value equality only) OR a literal SQL
# WHERE-clause string. We exploit the latter to combine namespace scoping +
# caller's optional Mongo-style filter into a single SQL string.
#
# Uses ``STARTS_WITH`` (GoogleSQL) for prefix matching — no LIKE wildcards to
# worry about. Strings are escaped against SQL injection (single-quote doubling
# + backslash doubling per BQ string-literal grammar).
# =============================================================================

import sqlglot.expressions as _sqlglot_exp


def _column_expression(name: str) -> _sqlglot_exp.Column:
    """Build a sqlglot ``Column`` expression for ``name``, backtick-quoted at
    render time. Rejects only the one character (backtick) that breaks
    backtick-quoted identifiers in BQ's lexical grammar; everything else
    — unicode, spaces, hyphens, digits, underscores — flows through cleanly.

    Replaces the old regex-based ``[A-Za-z_][A-Za-z0-9_]*`` validation that
    was over-restrictive (rejected unicode column names like ``列``).
    """
    if not isinstance(name, str):
        raise ValueError(
            f"BQ identifier must be a string, got {type(name).__name__}"
        )
    if not name:
        raise ValueError("BQ identifier must be non-empty")
    if "`" in name:
        raise ValueError(
            f"BQ identifier {name!r} contains a backtick, the one character "
            "BQ's grammar disallows inside backtick-quoted identifiers"
        )
    return _sqlglot_exp.column(name, quoted=True)


def _namespace_prefix_to_sql(
    namespace_prefix: tuple[str, ...],
) -> _sqlglot_exp.Expression | None:
    """Build a sqlglot ``Expression`` matching items whose stored ``namespace``
    column starts with ``namespace_prefix``.

    Uses ``ns_join`` to serialize the prefix the same way ``aput`` serializes
    stored namespaces — single source of truth. Combines an exact-equals
    branch with ``STARTS_WITH`` so the root of the subtree (e.g. ``"a.b"``
    when prefix is ``("a", "b")``) is included alongside descendants.

    Returns ``None`` for empty prefix (matches everything).

    The Expression is rendered by ``_combine_sql_clauses`` (single rendering
    site) — sqlglot's BQ dialect handles all string-literal escaping
    (backslash form, per BQ's lexical grammar) and identifier quoting
    (backtick form) automatically. No manual escape, no regex, no string
    concatenation.
    """
    if not namespace_prefix:
        return None
    prefix_str = ns_join(namespace_prefix)
    col = _column_expression("namespace")
    eq = _sqlglot_exp.EQ(
        this=col, expression=_sqlglot_exp.Literal.string(prefix_str)
    )
    sw = _sqlglot_exp.func(
        "STARTS_WITH",
        col,
        _sqlglot_exp.Literal.string(prefix_str + "."),
    )
    return _sqlglot_exp.Paren(
        this=_sqlglot_exp.Or(this=eq, expression=sw)
    )


def _filter_dict_to_sql(
    filter: dict[str, Any],
) -> _sqlglot_exp.Expression | None:
    """Build a sqlglot ``Expression`` from a LangMem-style filter dict.

    Supported per the LangMem reference (``langmem/graph_rag.py`` examples
    and ``InMemoryStore._compare_values``):

      * bare value     ``{"k": "v"}``                → ``k = 'v'``
      * ``$eq``        ``{"k": {"$eq": "v"}}``        → ``k = 'v'``
      * ``$ne``        ``{"k": {"$ne": "v"}}``        → ``k != 'v'``
      * ``$in``        ``{"k": {"$in": [a, b, c]}}``  → ``k IN ('a','b','c')``
      * ``$prefix``    ``{"k": {"$prefix": "p"}}``    → ``STARTS_WITH(k, 'p')``

    Multiple keys combine with AND. Empty ``$in`` list yields the FALSE
    literal (matches LangMem reference semantics: empty set in ⇒ no rows).
    Unknown operators raise ``ValueError`` to surface typos at first call
    rather than silently dropping filter intent.

    All identifier quoting and string-literal escaping is delegated to
    sqlglot's BQ-dialect rendering (called by ``_combine_sql_clauses``).
    Identifiers can be unicode (``列``), contain spaces (``KPI IMPACT
    METRICS``), or hyphens (``viebeg-data-vault``) — anything BQ's grammar
    accepts inside backtick quotes.

    Returns ``None`` for empty filter.
    """
    if not filter:
        return None
    clauses: list[_sqlglot_exp.Expression] = []
    for key, val in filter.items():
        col = _column_expression(key)  # raises on backtick / non-string / empty
        if isinstance(val, dict):
            for op, op_val in val.items():
                clauses.append(_filter_op_expression(col, op, op_val, key))
        else:
            clauses.append(
                _sqlglot_exp.EQ(
                    this=col, expression=_sqlglot_exp.Literal.string(str(val))
                )
            )
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return _sqlglot_exp.and_(*clauses)


def _filter_op_expression(
    col: _sqlglot_exp.Column,
    op: str,
    op_val: Any,
    key: str,
) -> _sqlglot_exp.Expression:
    """Build the sqlglot Expression for one (col, op, op_val) triple."""
    if op == "$eq":
        return _sqlglot_exp.EQ(
            this=col, expression=_sqlglot_exp.Literal.string(str(op_val))
        )
    if op == "$ne":
        return _sqlglot_exp.NEQ(
            this=col, expression=_sqlglot_exp.Literal.string(str(op_val))
        )
    if op == "$in":
        if not isinstance(op_val, (list, tuple)):
            raise ValueError(
                f"filter $in expects list, got {type(op_val).__name__}"
            )
        if not op_val:
            return _sqlglot_exp.false()
        items = [_sqlglot_exp.Literal.string(str(v)) for v in op_val]
        return _sqlglot_exp.In(this=col, expressions=items)
    if op == "$prefix":
        if not isinstance(op_val, str):
            raise ValueError(
                f"filter $prefix expects string, got {type(op_val).__name__}"
            )
        return _sqlglot_exp.func(
            "STARTS_WITH", col, _sqlglot_exp.Literal.string(op_val)
        )
    raise ValueError(
        f"unsupported filter operator {op!r} for key {key!r}; "
        "supported: $eq $ne $in $prefix"
    )


def _combine_sql_clauses(
    *clauses: _sqlglot_exp.Expression | None,
) -> str:
    """AND-combine sqlglot ``Expression`` objects and render to BQ SQL.

    Returns ``'TRUE'`` when all clauses are ``None`` —
    ``BigQueryVectorStore._create_filters`` interpolates the result into
    ``WHERE {id_expr} AND {where_filter_expr}``, so an empty filter must
    evaluate to ``TRUE`` to leave the upstream conjunct intact.

    Single rendering site: sqlglot's BQ dialect emits backslash-escaped
    string literals and backtick-quoted identifiers per BQ's lexical
    grammar. Verified empirically (``test_real_bq_*`` coverage).
    """
    valid = [c for c in clauses if c is not None]
    if not valid:
        return "TRUE"
    if len(valid) == 1:
        return valid[0].sql(dialect="bigquery")
    return _sqlglot_exp.and_(*valid).sql(dialect="bigquery")


class BigQueryMemoryStore(AsyncBatchedBaseStore):
    """
    This gives persistence of memory directly into Bigquery.
    """
    # We don't currently implement TTL semantics end-to-end in BigQuery.
    # Keep this False so BaseStore.put enforces correctness.
    supports_ttl: bool = False

    # Typed ontological discriminator for the three distinct memory kinds.
    # content_field is the BQ RECORD column name ("fact"/"episode"/"procedure");
    # memory_kind is the ONTOLOGICAL kind that affordance-branching keys on.
    _CONTENT_FIELD_TO_KIND = {
        "fact": "semantic",
        "episode": "episodic",
        "procedure": "procedural",
    }
    _VALID_MEMORY_KINDS = frozenset({"semantic", "episodic", "procedural"})

    def __init__(
        self,
        vectorstore: PatchedBigQueryVectorStore,
        content_field: str,
        content_model: Optional[Type[BaseModel]] = None,
        schema: Optional[List[bigquery.SchemaField]] = None,
        memory_kind: Optional[str] = None,
    ):
        # AsyncBatchedBaseStore expects to be constructed inside a running
        # event loop. When we build this store in a worker thread (via
        # asyncio.to_thread) there is no running loop, so `super().__init__()`
        # would raise `RuntimeError: no running event loop`.
        #
        # We catch that here and provide the attributes that the base class'
        # destructor expects so shutdown is clean. Our implementation of
        # `abatch` does not rely on the base-class batching machinery.
        try:
            super().__init__()
        except RuntimeError:
            # We intentionally do not rely on AsyncBatchedBaseStore's
            # background queue here. This store may be constructed in a
            # worker thread with no running loop.
            self._loop = None  # type: ignore[assignment]
            self._task = None  # type: ignore[assignment]
            self._aqueue = None  # type: ignore[assignment]

        self.vectorstore = vectorstore
        self.content_field = content_field
        self.content_model = content_model
        self.schema = schema
        # Typed ontological discriminator (canonical-9 fidelity round-2 fix:
        # bug_kind=brittle_enumeration_vs_invariant). Derived from content_field
        # when not supplied, so existing NICER-core / agent-q callers are
        # unaffected. An EXPLICIT out-of-vocabulary kind fails loudly (no silent
        # default); an un-derivable content_field with no explicit kind leaves
        # None (backward-compatible — the episodic affordance simply does not
        # fire, exactly as the old content_field!="episode" branch behaved).
        kind = memory_kind if memory_kind is not None else self._CONTENT_FIELD_TO_KIND.get(content_field)
        if kind is not None and kind not in self._VALID_MEMORY_KINDS:
            raise ValueError(
                f"memory_kind must be one of {sorted(self._VALID_MEMORY_KINDS)}; "
                f"got {kind!r} (memory_kind={memory_kind!r}, content_field={content_field!r})"
            )
        self.memory_kind = kind

    @classmethod
    def from_client(
        cls,
        dataset_name: str,
        table_name: str,
        content_field: str,
        content_model: Optional[Type[BaseModel]] = None,
        schema: Optional[List[bigquery.SchemaField]] = None,
        bq_client: Optional[bigquery.Client] = None,
        **kwargs,
    ) -> "BigQueryMemoryStore":
        # b65ed6bf (consolidation, not a third copy): accept an INJECTED client
        # (e.g. a CW-SA-credentialed, location-matched client) instead of always
        # using the module-default tenant client; callers needing the central
        # store reuse this whole builder rather than reimplementing its body.
        bq_client = bq_client if bq_client is not None else get_bq_client()

        # =====================================================================
        # FIX: Detect the dataset's ACTUAL location and use a location-matched
        # client.  BigQuery load jobs MUST run in the same region as the
        # dataset — a client created with location=europe-west1 cannot write
        # to a dataset in EU (or US), even though reads are location-agnostic.
        # =====================================================================
        dataset_ref = f"{bq_client.project}.{dataset_name}"
        dataset_location = bq_client.location  # default from config

        if schema:
            # Ensure dataset exists
            try:
                existing_ds = bq_client.get_dataset(dataset_ref)
                dataset_location = existing_ds.location
                logger.info(
                    "Dataset %s exists in location=%s (client default=%s)",
                    dataset_ref, dataset_location, bq_client.location,
                )
            except NotFound:
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = bq_client.location
                created_ds = bq_client.create_dataset(dataset)
                dataset_location = created_ds.location
                logger.info("Created dataset %s in location=%s", dataset_ref, dataset_location)
        else:
            # Even without schema pre-creation, detect the dataset location
            try:
                existing_ds = bq_client.get_dataset(dataset_ref)
                dataset_location = existing_ds.location
            except NotFound:
                pass  # will be created later if needed

        # If the dataset lives in a different location than the client,
        # create a location-matched client for all subsequent operations.
        if dataset_location != bq_client.location:
            logger.warning(
                "Location mismatch: client=%s, dataset=%s. "
                "Creating location-matched client for %s.",
                bq_client.location, dataset_location, dataset_ref,
            )
            bq_client = bigquery.Client(
                credentials=bq_client._credentials,
                project=bq_client.project,
                location=dataset_location,
            )

        if schema:
            # Ensure table exists with proper schema
            ensure_table_with_schema(
                client=bq_client,
                project_id=bq_client.project,
                dataset_id=dataset_name,
                table_name=table_name,
                schema=schema,
            )
        # =====================================================================

        vectorstore = PatchedBigQueryVectorStore(
            embedding=get_embedding(),
            project_id=bq_client.project,
            doc_id_field="doc_id",
            dataset_name=dataset_name,
            table_name=table_name,
            content_field=content_field,
            credentials=bq_client._credentials,
            location=dataset_location,
            **kwargs,
        )
        object.__setattr__(vectorstore, "_bq_client", bq_client)
        return cls(
            vectorstore=vectorstore,
            content_field=content_field,
            content_model=content_model,
            schema=schema,
        )

    def _normalize_structured_field(self, raw: Any) -> dict:
        if isinstance(raw, dict) and self.content_model:
            # Wrap into the content model (e.g., Fact)
            return raw  # Return as a dictionary
        elif isinstance(raw, str) and self.content_model:
            # If it's a string, wrap it into the content model
            return {"content":raw}
        elif self.content_model is not None and isinstance(raw, self.content_model):
            return raw.model_dump()  # Already in the correct format
        else:
            raise ValueError(
                "%s must be a dict, str, or %s — got: %s"
                % (
                    self.content_field,
                    self.content_model.__name__ if self.content_model else "BaseModel",
                    type(raw),
                )
            )

    async def aput(
        self,
        namespace: Tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Optional[Union[Literal[False], list[str]]] = None,
        *,
        ttl: float | None | NotProvided = NOT_PROVIDED,
    ) -> None:
        """REPLACE-semantics put — delegates to ``aput_batch`` for atomic
        MERGE-by-doc_id (LOAD-staging + MERGE in one DML).

        Why delegate rather than do per-row DELETE + INSERT here:

        * ``aput_batch``'s MERGE satisfies ``BaseStore.put``'s REPLACE
          contract atomically — no DELETE-INSERT race window where
          concurrent writers see ``Could not serialize access`` (BQ DML
          serialization conflict on the same row).
        * Single source of truth: row construction lives in
          ``_build_row_for_batch``; BQ DML lives in
          ``merge_upsert_json_rows``. This method is just the
          single-key entry point of the BaseStore contract.
        * Halves DML rate-limit pressure even for single writes:
          1 MERGE vs. 1 DELETE + 1 LOAD.

        See ``aput_batch`` for the implementation; row schemas plus
        episodic-timestamp handling are centralised in
        ``_build_row_for_batch``.
        """
        # Validate eagerly so misuse fails fast rather than at the BQ
        # boundary; ``_build_row_for_batch`` re-validates each item, but
        # the duplicate call is cheap and useful for stack-trace clarity.
        _validate_namespace(namespace)
        logger.info(
            "[aput] Inserting doc_id=%s into namespace=%s",
            key,
            ".".join(namespace),
        )
        await self.aput_batch([(namespace, key, value)])

    def _build_row_for_batch(
        self,
        namespace: Tuple[str, ...],
        key: str,
        value: dict,
    ) -> Optional[Tuple[dict, str]]:
        """Build a row dict + embedding text for one item, mirroring ``aput``'s
        per-item logic without the BQ writes. Returns ``None`` when the value
        is malformed (already logged via the same diagnostic as aput).

        The returned row dict includes everything except the embedding —
        ``aput_batch`` adds that after a single batched embed call.
        """
        _validate_namespace(namespace)
        data: dict = {"namespace": ".".join(namespace), "doc_id": key}

        raw_content = value.get("content")
        if raw_content is None:
            value_keys = sorted(value.keys()) if isinstance(value, dict) else None
            logger.error(
                "Content for %s is None. This is not expected. "
                "namespace=%s doc_id=%s value_keys=%s value_repr=%s",
                self.content_field,
                ".".join(namespace) if namespace else None,
                key,
                value_keys,
                repr(value)[:500],
            )
            return None

        actual_content = raw_content

        # Episodic-specific: extract or generate timestamp. Keyed on the TYPED
        # ontological kind (canonical-9 fidelity round-2 fix), not the
        # content_field string — the affordance belongs to the episodic KIND.
        if self.memory_kind == "episodic":
            if isinstance(raw_content, dict) and "episode" in raw_content:
                actual_content = raw_content["episode"]
                if "timestamp" in raw_content:
                    ts = raw_content["timestamp"]
                    if isinstance(ts, datetime):
                        data["timestamp"] = ts.isoformat()
                    elif isinstance(ts, str):
                        data["timestamp"] = ts
            if "timestamp" not in data and "timestamp" in value:
                ts = value["timestamp"]
                if isinstance(ts, datetime):
                    data["timestamp"] = ts.isoformat()
                elif isinstance(ts, str):
                    data["timestamp"] = ts
            if "timestamp" not in data:
                data["timestamp"] = datetime.now(timezone.utc).isoformat()

        text = self._normalize_structured_field(actual_content)
        embedding_text = json.dumps(text)
        data[self.content_field] = text

        # ``stored_at`` is part of every memory table's schema (see
        # ``SCHEMAS`` below). Setting it here means ``merge_upsert_json_rows``
        # will include it in the MERGE's UPDATE clause, refreshing it on
        # every put — preserving the semantics that the legacy DELETE +
        # INSERT path produced via the column's ``CURRENT_TIMESTAMP()``
        # default. Activity-rate views that aggregate ``DATE(stored_at)``
        # see fresh-on-update behaviour. (For first-stored / discovery-rate
        # semantics, add a separate ``updated_at`` column rather than
        # changing ``stored_at``'s meaning — see master plan's ambient-
        # agent observability migration task.)
        data["stored_at"] = datetime.now(timezone.utc).isoformat()

        return data, embedding_text

    async def aput_batch(
        self,
        items: List[Tuple[Tuple[str, ...], str, dict]],
    ) -> None:
        """Batched upsert for ``aput`` operations. Replaces the per-row
        DELETE+INSERT pattern (1 DML per row) with a single LOAD-staging +
        MERGE (1 DML for any batch size).

        Each ``items`` entry is ``(namespace, key, value)`` — same shape as
        ``aput``'s arguments. Items with malformed values are skipped (logged
        via the same diagnostic as ``aput``); the remaining items are written
        in one batched MERGE.

        Duplicate keys within the batch are deduplicated last-wins, mirroring
        ``aput``'s REPLACE semantics; this also avoids BigQuery's MERGE
        "multiple source rows match a single target row" error.
        """
        if not items:
            return

        # Build rows + embedding texts; drop malformed items.
        rows_with_texts: List[Tuple[dict, str]] = []
        for namespace, key, value in items:
            built = self._build_row_for_batch(namespace, key, value)
            if built is not None:
                rows_with_texts.append(built)

        if not rows_with_texts:
            return

        # Last-wins dedup on doc_id within this batch.
        deduped: dict[str, Tuple[dict, str]] = {}
        for data, text in rows_with_texts:
            deduped[data["doc_id"]] = (data, text)
        ordered = list(deduped.values())

        rows = [d for d, _ in ordered]
        texts = [t for _, t in ordered]

        # One batched embed call → list of embedding vectors aligned with rows.
        embeddings = self.vectorstore.embedding.embed_documents(texts)
        for row, emb in zip(rows, embeddings):
            row[self.vectorstore.embedding_field] = emb

        # Single LOAD-staging + MERGE — one DML regardless of batch size.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _MEMORY_STORE_EXECUTOR,
            lambda: merge_upsert_json_rows(
                self.vectorstore._bq_client,
                self.vectorstore.full_table_id,
                rows,
                merge_keys=[self.vectorstore.doc_id_field],
            ),
        )

    async def aget(
            self,
            namespace: Tuple[str, ...],
            key: str,
            *,
            refresh_ttl: Optional[bool] = None
    ) -> Optional[Item]:
        logger.debug("[aget] Retrieving doc_id=%s from namespace=%s", key, namespace)

        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(
            _MEMORY_STORE_EXECUTOR,
            self.vectorstore.get_documents,
            [key],
        )
        if not docs:
            return None
        doc = docs[0]
        metadata = doc.metadata

        # Per BaseStore.aget contract, the (namespace, key) pair identifies one
        # item: returning a doc whose stored namespace doesn't match the caller's
        # namespace would violate the contract and could leak across scopes.
        # Reuse ``ns_join`` so the comparison is symmetric with ``aput``'s write.
        stored_ns_str = metadata.get("namespace", "")
        expected_ns_str = ns_join(namespace)
        if stored_ns_str != expected_ns_str:
            logger.debug(
                "[aget] doc_id=%s found but namespace mismatch (stored=%r, expected=%r)",
                key, stored_ns_str, expected_ns_str,
            )
            return None
        ns = tuple(stored_ns_str.split(".")) if stored_ns_str else ()

        content_val = doc.page_content
        if isinstance(content_val, str) and self.content_model:
            try:
                metadata[self.content_field] = self.content_model(**json.loads(content_val))
            except Exception:
                pass

        raw_created = doc.metadata.get("created_at")
        raw_updated = doc.metadata.get("updated_at")

        def to_dt(x):
            if isinstance(x, datetime):
                return x
            if isinstance(x, str):
                return datetime.fromisoformat(x)
            return datetime.now(timezone.utc)

        created_at = to_dt(raw_created)
        updated_at = to_dt(raw_updated)

        return Item(
            namespace=ns,
            key=key,
            value=metadata,
            created_at=created_at,
            updated_at=updated_at
        )
    # --- Overloads to satisfy both older & newer langgraph versions ---
    @overload
    async def asearch(  # current shape (with refresh_ttl)
        self,
        namespace_prefix: tuple[str, ...], /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]: ...
    @overload
    async def asearch(  # older shape (without refresh_ttl)
        self,
        namespace_prefix: tuple[str, ...], /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[SearchItem]: ...

    # --- Implementation matches the *newer* signature; extra kw is optional ---
    async def asearch(
        self,
        namespace_prefix: tuple[str, ...], /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,   # accepted; not used here
    ) -> list[SearchItem]:
        logger.info(
            "[asearch] ns_prefix=%s, query=%r, filter=%r, limit=%d, offset=%d",
            namespace_prefix, query, filter, limit, offset
        )

        # ── Build the combined SQL WHERE clause ────────────────────────────
        # Per BaseStore.asearch contract, namespace_prefix scopes the search
        # to items whose stored namespace tuple starts with that prefix.
        # ``BigQueryVectorStore._create_filters`` accepts ``filter`` either as
        # a dict (column = value equality only) OR a literal SQL string. We
        # take the latter path so we can express prefix matching + Mongo-style
        # operators in one combined clause.
        ns_clause = _namespace_prefix_to_sql(namespace_prefix)
        user_clause = _filter_dict_to_sql(filter) if filter else None
        combined_sql = _combine_sql_clauses(ns_clause, user_clause)

        # If caller gave a dict (or you have a content_model), preserve its shape
        if isinstance(query, dict) or self.content_model:
            try:
                model_cls = self.content_model
                if model_cls and isinstance(query, dict):
                    query = json.dumps(model_cls(**query).model_dump())
                elif isinstance(query, dict):
                    query = json.dumps(query)
            except Exception:
                query = str(query)

        loop = asyncio.get_running_loop()

        # ── Filter-only path (query=None) ──────────────────────────────────
        # Per BaseStore.asearch contract, ``query=None`` is a valid filter-only
        # search. ``similarity_search_with_score`` requires a query string, so
        # we bypass it and run a direct SELECT against the vectorstore's table.
        if not query:
            return await loop.run_in_executor(
                _MEMORY_STORE_EXECUTOR,
                self._search_filter_only,
                combined_sql,
                limit,
                offset,
            )

        # ── Standard similarity-search path ────────────────────────────────
        # Pass combined_sql as the ``filter`` arg — ``BigQueryVectorStore``'s
        # ``_create_filters`` treats non-dict filter values as a literal SQL
        # WHERE clause, which is exactly what we want.
        hits = await loop.run_in_executor(
            _MEMORY_STORE_EXECUTOR,
            lambda: self.vectorstore.similarity_search_with_score(
                query=query, filter=combined_sql, k=limit + offset
            ),
        )
        hits = hits[offset:]

        out: list[SearchItem] = []
        for doc, score in hits:
            ns = tuple((doc.metadata.get("namespace") or "").split("."))
            key = doc.metadata.get("doc_id") or ""

            # Deserialize page_content if it was JSON
            raw_page = doc.page_content
            try:
                content = json.loads(raw_page)
            except Exception:
                content = raw_page

            md = dict(doc.metadata)
            md[self.content_field] = content

            # Normalize timestamps defensively
            def _to_dt(x):
                if isinstance(x, datetime):
                    return x
                if isinstance(x, str):
                    try:
                        return datetime.fromisoformat(x)
                    except Exception:
                        pass
                return datetime.now(timezone.utc)

            created_at = _to_dt(doc.metadata.get("created_at"))
            updated_at = _to_dt(doc.metadata.get("updated_at"))

            out.append(
                SearchItem(
                    namespace=ns,
                    key=key,
                    value=md,
                    score=float(score),
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        return out

    def _search_filter_only(
        self, where_sql: str, limit: int, offset: int
    ) -> list[SearchItem]:
        """Filter-only search path for ``asearch(query=None)``.

        Bypasses ``similarity_search_with_score`` (which requires a query
        string) and runs a direct SELECT against the underlying BQ table.
        Reuses the same ``namespace`` STRING column the embedding path reads,
        and the same ``content_field`` / ``doc_id_field`` the vectorstore knows.

        Returns ``SearchItem`` objects with ``score=0.0`` since there's no
        similarity ranking in filter-only mode (callers can sort by other
        metadata if they want order).
        """
        bq_client = self.vectorstore._bq_client
        full_table = self.vectorstore.full_table_id
        embedding_field = self.vectorstore.embedding_field
        doc_id_field = self.vectorstore.doc_id_field
        sql = (
            f"SELECT * EXCEPT ({embedding_field}) "
            f"FROM `{full_table}` "
            f"WHERE {where_sql} "
            f"LIMIT {int(limit)} OFFSET {int(offset)}"
        )
        logger.debug("[asearch:filter-only] sql=%s", sql)
        job = bq_client.query(sql)
        out: list[SearchItem] = []
        for row in job:
            row_dict = dict(row.items())
            ns = tuple((row_dict.get("namespace") or "").split(".")) \
                if row_dict.get("namespace") else ()
            key = row_dict.get(doc_id_field) or ""
            raw_content = row_dict.get(self.content_field)
            try:
                content = (
                    json.loads(raw_content)
                    if isinstance(raw_content, str)
                    else raw_content
                )
            except Exception:
                content = raw_content
            md = dict(row_dict)
            md[self.content_field] = content

            def _to_dt(x):
                if isinstance(x, datetime):
                    return x
                if isinstance(x, str):
                    try:
                        return datetime.fromisoformat(x)
                    except Exception:
                        pass
                return datetime.now(timezone.utc)

            out.append(SearchItem(
                namespace=ns,
                key=key,
                value=md,
                score=0.0,  # no similarity ranking in filter-only mode
                created_at=_to_dt(md.get("created_at")),
                updated_at=_to_dt(md.get("updated_at")),
            ))
        return out

    def search(
        self,
        namespace_prefix: tuple[str, ...], /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        # accept refresh_ttl if the base class ever passes it through
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        """
        Synchronous search wrapper.

        Some callers (e.g. sync tool paths) expect `search` to be *sync*.
        We resolve the async `asearch` coroutine here so nobody ever ends up
        with an un-awaited `BigQueryMemoryStore.asearch` object.
        """
        async def _runner() -> list[SearchItem]:
            return await self.asearch(
                namespace_prefix,
                query=query,
                filter=filter,
                limit=limit,
                offset=offset,
                refresh_ttl=refresh_ttl,
            )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop: safe to create one and run to completion
            return asyncio.run(_runner())

        # Already in an event loop: hop to a worker thread with its own loop.
        # The shared ``_MEMORY_STORE_EXECUTOR`` (module-scope, atexit-shutdown)
        # replaces the prior per-call ``with ThreadPoolExecutor(max_workers=1)``
        # whose ``__exit__`` blocked on ``shutdown(wait=True)`` and contributed
        # to the Bug 6 cascade (PR-A0-3f lift).
        def _thread_runner() -> list[SearchItem]:
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(_runner())
            finally:
                new_loop.close()

        return _MEMORY_STORE_EXECUTOR.submit(_thread_runner).result()

    async def adelete(self, namespace: Tuple[str, ...], key: str) -> None:
        logger.info("[adelete] Deleting doc_id=%s from namespace=%s", key, namespace)

        adelete = getattr(self.vectorstore, "adelete", None)
        if callable(adelete):
            await adelete(ids=[key])  # type: ignore[misc]
            return

        delete = getattr(self.vectorstore, "delete", None)
        if not callable(delete):
            raise NotImplementedError("Vector store does not support delete/adelete")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_MEMORY_STORE_EXECUTOR, delete, [key])

    def mget(self, keys: Sequence[str]) -> List[Optional[dict]]:
        """
        Synchronous multi-get. BigQueryVectorStore only exposes a synchronous
        get_documents() API, so we call it directly here.
        """
        return self.vectorstore.get_documents(ids=list(keys))

    async def amget(self, keys: Sequence[str]) -> List[Optional[dict]]:
        """
        Async-friendly multi-get.

        The underlying BigQueryVectorStore is synchronous, so we offload the
        blocking call to a worker thread when running inside an event loop.
        This keeps the Store API consistent for callers and avoids any
        un-awaited coroutine warnings.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _MEMORY_STORE_EXECUTOR, self.vectorstore.get_documents, list(keys)
        )

    def mset(self, key_value_pairs: Sequence[Tuple[str, dict]]) -> None:
        documents = [
            Document(page_content=value.get(self.content_field), metadata={**value, "doc_id": key})
            for key, value in key_value_pairs
        ]
        self.vectorstore.add_documents(documents)

    async def amset(
            self,
            key_value_pairs: Sequence[Tuple[str, dict]]
    ) -> None:
        """
        Async-friendly multi-set.

        BigQueryVectorStore.add_documents is synchronous, so we offload it to
        a worker thread to avoid blocking the event loop.
        """
        documents = [
            Document(
                page_content=value.get(self.content_field),
                metadata={**value, "doc_id": key}
            )
            for key, value in key_value_pairs
        ]
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _MEMORY_STORE_EXECUTOR,
            self.vectorstore.add_documents,
            documents
        )


    def mdelete(self, keys: Sequence[str]) -> None:
        """
        Synchronous delete.

        Prefer a synchronous `delete()` method on the underlying vectorstore.
        If only an async `adelete()` is available, run it to completion in a
        one-shot event loop so callers do not get a `coroutine was never awaited`
        warning.
        """
        delete = getattr(self.vectorstore, "delete", None)
        if callable(delete):
            delete(ids=list(keys))
            return

        adelete = getattr(self.vectorstore, "adelete", None)
        if adelete is None:
            raise NotImplementedError("Vector store does not support delete/adelete")

        # If we're in a plain sync context, just run the coroutine to completion.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(adelete(ids=list(keys)))
            return

        # If we're *already* in an event loop, hop to a worker thread that owns
        # its own loop so we don't try to nest event loops and explode. Uses
        # the shared ``_MEMORY_STORE_EXECUTOR`` (module-scope, atexit-shutdown)
        # in place of a per-call ``with ThreadPoolExecutor(max_workers=1)``
        # whose blocking ``__exit__`` contributed to the Bug 6 cascade.
        def _runner() -> None:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(adelete(ids=list(keys)))
            finally:
                loop.close()

        _MEMORY_STORE_EXECUTOR.submit(_runner).result()

    async def amdelete(self, keys: Sequence[str]) -> None:
        """
        Async delete wrapper.

        If the underlying vectorstore exposes `adelete`, await it directly;
        otherwise fall back to running the sync `delete()` in a worker thread.
        """
        adelete = getattr(self.vectorstore, "adelete", None)
        if callable(adelete):
            await adelete(ids=list(keys))  # type: ignore[misc]
            return

        delete = getattr(self.vectorstore, "delete", None)
        if not callable(delete):
            raise NotImplementedError("Vector store does not support delete/adelete")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_MEMORY_STORE_EXECUTOR, delete, list(keys))

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        "yields keys"
        return self.vectorstore.yield_keys(prefix=prefix)

    async def ayield_keys(self, prefix: Optional[str] = None) -> AsyncIterator[str]:
        """
        Async adapter for yield_keys.
        BigQueryVectorStore.yield_keys is synchronous; offload collection.
        """
        loop = asyncio.get_running_loop()
        keys = await loop.run_in_executor(
            _MEMORY_STORE_EXECUTOR, lambda: list(self.vectorstore.yield_keys(prefix=prefix))
        )
        for key in keys:
            yield key

    # ---------------------------------------------------------------------
    # Sync API: override AsyncBatchedBaseStore wrappers.
    #
    # We cannot rely on AsyncBatchedBaseStore._loop because this store may be
    # constructed in a worker thread (no running loop). These overrides ensure
    # callers never hit run_coroutine_threadsafe(..., None).
    # ---------------------------------------------------------------------

    def batch(self, ops: IterableABC[Op]) -> list[Result]:
        async def _runner() -> list[Result]:
            return await self.abatch(ops)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_runner())

        # Lifted to shared ``_MEMORY_STORE_EXECUTOR`` (PR-A0-3f) — same Bug 6
        # rationale as ``search`` / ``mdelete`` sync wrappers above.
        def _thread_runner() -> list[Result]:
            new_loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(new_loop)
                return new_loop.run_until_complete(_runner())
            finally:
                new_loop.close()

        return _MEMORY_STORE_EXECUTOR.submit(_thread_runner).result()

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        return BaseStore.get(self, namespace, key, refresh_ttl=refresh_ttl)

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None | NotProvided = NOT_PROVIDED,
    ) -> None:
        return BaseStore.put(self, namespace, key, value, index=index, ttl=ttl)

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        return BaseStore.delete(self, namespace, key)

    def list_namespaces(
        self,
        *,
        prefix: NamespacePath | None = None,
        suffix: NamespacePath | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        return BaseStore.list_namespaces(
            self,
            prefix=prefix,
            suffix=suffix,
            max_depth=max_depth,
            limit=limit,
            offset=offset,
        )

    async def abatch(self, ops: Iterable[Op]) -> List[Result]:
        """Single dispatch path for write/read ops.

        Every ``PutOp`` — whether 1 or N — flows through ``aput_batch``.
        That keeps a single source of truth for upsert (LOAD-staging +
        MERGE-by-doc_id, atomic, no DELETE-INSERT race) regardless of
        batch size. ``aput_batch`` itself is a no-op on empty input.
        ``GetOp`` / ``SearchOp`` / ``ListNamespacesOp`` go individually
        because they don't share a batched primitive.
        """
        ops_list = list(ops)
        logger.info("[abatch] Executing %s batch operations", len(ops_list))

        put_indices = [i for i, op in enumerate(ops_list) if isinstance(op, PutOp)]
        if put_indices:
            put_items = [
                (ops_list[i].namespace, ops_list[i].key, ops_list[i].value)
                for i in put_indices
            ]
            await self.aput_batch(put_items)

        results: List[Result] = []
        for op in ops_list:
            if isinstance(op, PutOp):
                # Handled by the single ``aput_batch`` call above.
                results.append(None)
            elif isinstance(op, GetOp):
                item = await self.aget(op.namespace, op.key, refresh_ttl=op.refresh_ttl)
                results.append(item)
            elif isinstance(op, SearchOp):
                items = await self.asearch(
                    op.namespace_prefix,
                    query=op.query,
                    filter=op.filter,
                    limit=op.limit,
                    offset=op.offset,
                    refresh_ttl=op.refresh_ttl,
                )
                results.append(items)
            elif isinstance(op, ListNamespacesOp):
                results.append([])
            else:
                raise NotImplementedError(f"Unsupported op: {op}")
        return results

#----------------------SCHEMAS------------------------------------------------
# Define the desired schemas for each table
# Define the desired schemas for each table
SCHEMAS = {
    SEMANTIC_TABLE: [
        bigquery.SchemaField("doc_id", "STRING", mode="REQUIRED", description="Unique identifier for the semantic memory."),
        bigquery.SchemaField("fact", "RECORD", mode="REQUIRED", fields=[
            bigquery.SchemaField("content", "STRING", description="The factual statement or knowledge."),
            bigquery.SchemaField("importance", "STRING", description="Importance level e.g., HIGH, MEDIUM, LOW."),
            bigquery.SchemaField("category", "STRING", description="Type of fact e.g., BACKGROUND, HOBBY, SKILL."),
        ], description="Structured representation of factual knowledge."),
        bigquery.SchemaField("source", "STRING", mode="NULLABLE", description="Where the fact came from (user input, document, etc)."),
        bigquery.SchemaField("namespace", "STRING", mode="NULLABLE", description="Namespace for logical grouping."),
        bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED", description="Embedding for vector similarity."),
        bigquery.SchemaField("stored_at", "TIMESTAMP", mode="NULLABLE", default_value_expression='CURRENT_TIMESTAMP()', description="When the record was stored."),
    ],
    EPISODIC_TABLE: [
        bigquery.SchemaField("doc_id", "STRING", mode="REQUIRED", description="Unique identifier for the episode."),
        bigquery.SchemaField("episode", "RECORD", mode="REQUIRED", fields=[
            bigquery.SchemaField("observation", "STRING", description="Context or setup of the interaction."),
            bigquery.SchemaField("thoughts", "STRING", description="Reasoning or internal monologue."),
            bigquery.SchemaField("action", "STRING", description="What was done in response."),
            bigquery.SchemaField("result", "STRING", description="Outcome and what made it successful."),
        ], description="Structured record of a specific past interaction."),
        bigquery.SchemaField("participants", "STRING", mode="REPEATED", description="Users or agents involved in the episode."),
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED", description="When the episode occurred."),
        bigquery.SchemaField("namespace", "STRING", mode="NULLABLE", description="Namespace for logical grouping."),
        bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED", description="Vector embedding for similarity search."),
        bigquery.SchemaField("stored_at", "TIMESTAMP", mode="NULLABLE", default_value_expression='CURRENT_TIMESTAMP()', description="When the record was stored."),
    ],
    PROCEDURAL_TABLE: [
        bigquery.SchemaField("doc_id", "STRING", mode="REQUIRED", description="Unique identifier for the procedure."),
        bigquery.SchemaField("procedure", "RECORD", mode="REQUIRED", fields=[
            bigquery.SchemaField("name", "STRING", description="Name or title of the procedure."),
            bigquery.SchemaField("conditions", "STRING", description="When or under what conditions to execute."),
            bigquery.SchemaField("steps", "STRING", description="Step-by-step instructions."),
            bigquery.SchemaField("notes", "STRING", description="Additional notes or guidelines."),
        ], description="Structured behavioral rule or policy."),
        bigquery.SchemaField("version", "STRING", mode="NULLABLE", description="Version identifier for procedural logic."),
        bigquery.SchemaField("namespace", "STRING", mode="NULLABLE", description="Namespace for logical grouping."),
        bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED", description="Embedding for vector-based retrieval."),
        bigquery.SchemaField("stored_at", "TIMESTAMP", mode="NULLABLE", default_value_expression='CURRENT_TIMESTAMP()', description="When the record was stored."),
]
}

async def _build_store_in_thread(
    *,
    table_name: str,
    content_field: str,
    model: Type[BaseModel],
    schema: List[bigquery.SchemaField]
) -> "BigQueryMemoryStore":
    """
    Build a BigQueryMemoryStore in a worker thread so that the synchronous
    BigQuery dataset/table validation runs off the main event loop (and
    doesn't trip LangGraph dev's blocking-call detector).
    """

    def _builder() -> "BigQueryMemoryStore":
        return BigQueryMemoryStore.from_client(
            dataset_name=DATASET_ID,
            table_name=table_name,
            content_field=content_field,
            content_model=model,
            schema=schema,
        )

    # PR-A0-13c: route through the bounded _MEMORY_STORE_EXECUTOR
    # (module-scope, already used for all memory-store I/O at lines
    # 1497, 1517, 1636, 1648, 1807, 1822, 1842). Bug #79: default
    # executor floods callbacks at loop close. _MEMORY_STORE_EXECUTOR
    # is the right choice — it's in this module (no cross-module
    # circular-import risk with memory_tools.py which imports FROM
    # this module).
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_MEMORY_STORE_EXECUTOR, _builder)

#---------------------------------get memory tools--------------------------------
class MemoryBundle(NamedTuple):
    """In-process result-container (value-object) carrying the langmem tool list
    together with the 3 distinct ``BigQueryMemoryStore`` instances they were built
    over. NOT a persisted entity — it exists only to surface the 3 stores to the
    ``pre_model_hook`` factory at the call site WITHOUT changing
    ``get_memory_tools``' frozen ``List`` return. Replaces what would otherwise be
    an untyped ``(tools, sem, epi, proc)`` tuple."""
    tools: list
    semantic_store: "BigQueryMemoryStore"
    episodic_store: "BigQueryMemoryStore"
    procedural_store: "BigQueryMemoryStore"


async def _build_three_stores():
    """Construct the 3 physically-distinct memory stores ONCE, in worker threads,
    so the synchronous BigQuery dataset/table validation runs off the main event
    loop (LangGraph dev's blocking-call detector). Returns
    ``(semantic_store, episodic_store, procedural_store)``.

    Lifted out of ``get_memory_tools`` so ``get_memory_tools`` and
    ``get_memory_tools_and_stores`` share ONE construction path (zero duplication).
    """
    return await asyncio.gather(
        _build_store_in_thread(
            table_name=SEMANTIC_TABLE,
            content_field=CONTENT_FIELDS[SEMANTIC_TABLE],
            model=PYDANTIC_MODELS[CONTENT_FIELDS[SEMANTIC_TABLE]],
            schema=SCHEMAS[SEMANTIC_TABLE],
        ),
        _build_store_in_thread(
            table_name=EPISODIC_TABLE,
            content_field=CONTENT_FIELDS[EPISODIC_TABLE],
            model=PYDANTIC_MODELS[CONTENT_FIELDS[EPISODIC_TABLE]],
            schema=SCHEMAS[EPISODIC_TABLE],
        ),
        _build_store_in_thread(
            table_name=PROCEDURAL_TABLE,
            content_field=CONTENT_FIELDS[PROCEDURAL_TABLE],
            model=PYDANTIC_MODELS[CONTENT_FIELDS[PROCEDURAL_TABLE]],
            schema=SCHEMAS[PROCEDURAL_TABLE],
        ),
    )


def _build_tools_for(
    namespace_templates: Dict[str, NamespaceTemplate],
    semantic_memory_store: "BigQueryMemoryStore",
    episodic_memory_store: "BigQueryMemoryStore",
    procedural_memory_store: "BigQueryMemoryStore",
) -> list:
    """The per-namespace (manage, search) tool-pairing loop lifted out of
    ``get_memory_tools``. Pure store-dispatch + create_manage/create_search; no
    I/O (the 3 stores are constructed by ``_build_three_stores`` and passed in)."""
    tools: list = []
    # 2. For each namespace_template, build both a manage‐tool and a search‐tool
    for base_key, ns_template in namespace_templates.items():
        # Determine which physical store to hook up. ``base_key`` may be the
        # bare memory-type name (``semantic`` / ``episodic`` / ``procedural``)
        # or a Phase-3 level-specific extension (``semantic_dataset``,
        # ``semantic_table``, ``semantic_column``, ``semantic_fact``,
        # ``semantic_edge``). All ``semantic_*`` keys share the same physical
        # ``semantic_memory`` BQ table — they differ only in the namespace
        # shape used at write/search time, allowing per-level scoping via
        # ``BigQueryMemoryStore.asearch``'s namespace_prefix matching (B0).
        if base_key == "semantic" or base_key.startswith("semantic_"):
            store = semantic_memory_store
            schema_model = Fact
        elif base_key == "episodic" or base_key.startswith("episodic_"):
            store = episodic_memory_store
            schema_model = PYDANTIC_MODELS[CONTENT_FIELDS[EPISODIC_TABLE]]
        elif base_key == "procedural" or base_key.startswith("procedural_"):
            store = procedural_memory_store
            schema_model = PYDANTIC_MODELS[CONTENT_FIELDS[PROCEDURAL_TABLE]]
        else:
            # Unknown base_key — fall back to semantic_store with Fact schema
            # (preserves pre-Phase-3 behavior for any caller registering
            # extension keys we haven't anticipated).
            store = semantic_memory_store
            schema_model = Fact

        # 2a. Create the “manage” tool
        manage_tool = create_manage_memory_tool(
            namespace=ns_template,
            store=store,
            name=f"manage_{base_key}_memory",
            schema=schema_model
        )
        tools.append(manage_tool)

        # 2b. Create the corresponding “search” tool
        search_tool = create_search_memory_tool(
            namespace=ns_template,
            store=store,
            name=f"search_{base_key}_memory"
        )
        tools.append(search_tool)

    return tools


async def get_memory_tools(
    namespace_templates: Dict[str, NamespaceTemplate]
) -> List:
    """
    Create pairs of (manage, search) memory tools for each namespace template.

    Args:
      namespace_templates: a dict where
         - key = tool‐base‐name (e.g. "semantic", "episodic", "procedural", or anything you choose)
         - value = a NamespaceTemplate tuple, e.g.
             ("metadata", "{object_type}", "{langgraph_auth_user_id}", "{object_name}")

    Returns:
      A list of all created tools (both manage_* and search_*).

    SIGNATURE + List return UNCHANGED — the body now delegates to the shared
    ``_build_three_stores`` + ``_build_tools_for`` helpers (construct-ONCE; the 3
    stores are discarded here, exactly as before). All existing callers are
    byte-for-byte unaffected.
    """
    sem, epi, proc = await _build_three_stores()
    return _build_tools_for(namespace_templates, sem, epi, proc)


async def get_memory_tools_and_stores(
    namespace_templates: Dict[str, NamespaceTemplate]
) -> MemoryBundle:
    """Like ``get_memory_tools`` but ALSO surfaces the 3 underlying stores.

    Shares the SAME ``_build_three_stores`` + ``_build_tools_for`` construction
    path as ``get_memory_tools`` (no second construction, no duplication). The 3
    distinct stores are needed by the ``pre_model_hook`` factory (one closure over
    3 physically-distinct ``BigQueryMemoryStore`` instances), which ``create_react_agent``'s
    single-``store`` slot cannot carry. Returns a typed ``MemoryBundle``.
    """
    sem, epi, proc = await _build_three_stores()
    tools = _build_tools_for(namespace_templates, sem, epi, proc)
    return MemoryBundle(
        tools=tools,
        semantic_store=sem,
        episodic_store=epi,
        procedural_store=proc,
    )

# ── agent-q Piece-4 store-builder seam (NICER-core public seam) ───────────────────────
class EmbeddingDimMismatchError(RuntimeError):
    """Raised at agent-q store construction when the store's embedding dim differs
    from the dim of the existing BQ rows (prevents silent cross-dim vector pollution)."""


def _agentq_cw_client():
    """CW-credentialed, location-matched BigQuery client from ENV
    (GCP_SERVICE_ACCOUNT_BASE64 -> SA creds; AGENTQ_MEMORY_PROJECT/DATASET overrides).
    Realizes the b65ed6bf credentials injection for the public agent-q deployment."""
    import base64 as _b64, json as _json, os as _os
    from google.oauth2 import service_account as _sa
    sa_b64 = _os.environ.get("GCP_SERVICE_ACCOUNT_BASE64")
    if not sa_b64:
        raise RuntimeError("GCP_SERVICE_ACCOUNT_BASE64 not set (agent-q CW memory store)")
    info = _json.loads(_b64.b64decode(sa_b64))
    creds = _sa.Credentials.from_service_account_info(info)
    project = _os.environ.get("AGENTQ_MEMORY_PROJECT") or info.get("project_id")
    dataset = _os.environ.get("AGENTQ_MEMORY_DATASET", "agent_system_memory")
    client = bigquery.Client(credentials=creds, project=project)
    ds = client.get_dataset("%s.%s" % (project, dataset))
    if client.location != ds.location:
        client = bigquery.Client(credentials=creds, project=project, location=ds.location)
    return client, dataset


def _agentq_dim_guard(store, client, project: str, dataset: str, table_name: str) -> None:
    """Probe ONE existing row's embedding length; raise EmbeddingDimMismatchError if
    it differs from this store's embedding dim. No-op on empty table / probe error."""
    try:
        sql = ("SELECT ARRAY_LENGTH(embedding) AS d FROM `%s.%s.%s` "
               "WHERE embedding IS NOT NULL LIMIT 1" % (project, dataset, table_name))
        rows = list(client.query(sql).result())
    except Exception:
        return
    if not rows or rows[0]["d"] is None:
        return
    probe_dim = len(store.vectorstore.embedding.embed_query("__dim_probe__"))
    if rows[0]["d"] != probe_dim:
        raise EmbeddingDimMismatchError(
            "%s: existing embedding dim %s != store dim %s"
            % (table_name, rows[0]["d"], probe_dim)
        )


def _build_agentq_store(table_name: str):
    client, dataset = _agentq_cw_client()
    cf = CONTENT_FIELDS[table_name]
    store = BigQueryMemoryStore.from_client(
        dataset_name=dataset, table_name=table_name, content_field=cf,
        content_model=PYDANTIC_MODELS[cf], schema=SCHEMAS[table_name], bq_client=client,
    )
    _agentq_dim_guard(store, client, client.project, dataset, table_name)
    return store


def build_agentq_semantic_store():
    """Public NICER-core seam: the CW semantic store agent-q reads preferences from."""
    return _build_agentq_store(SEMANTIC_TABLE)


def build_agentq_episodic_store():
    """Public NICER-core seam: the CW episodic store agent-q writes completed-command
    episodes to."""
    return _build_agentq_store(EPISODIC_TABLE)
