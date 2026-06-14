# src/graphs/agent.py

import src.langgraph_slack.patch_typing  # must run before any Pydantic model loading
from langgraph.prebuilt import create_react_agent
import logging
import concurrent.futures
from src.graphs.memory import get_memory_tools_and_stores, MemoryBundle
from src.graphs.pre_model_hook import (
    make_pre_model_hook,
    _SEMANTIC_NS,
    _EPISODIC_NS,
    _PROCEDURAL_NS,
)
import asyncio
logging.basicConfig(level=logging.DEBUG)


def _run_sync(make_coro):
    """Run an async builder to completion and return its result, whether or
    not an event loop is already running on the importing thread.

    ``asyncio.run`` raises ``RuntimeError`` when called from within a running
    loop, so the module-level bundle construction below would crash whenever
    this module is first imported from an async context (an ``async def`` test,
    a uvicorn/FastAPI lifespan hook, an async request handler). When a loop is
    already running we offload the one-shot build to a dedicated worker thread
    that owns its own loop; otherwise we run it directly. ``make_coro`` is a
    zero-arg callable returning a FRESH coroutine, so the coroutine is created
    on the thread that will await it.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make_coro())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(make_coro())).result()
from langgraph_slack.config import DEFAULT_MODEL, DEFAULT_TEMPERATURE
from src.prompts import NICER_PROMPT as BABY_NICER_PROMPT  # OSS system prompt (module exports NICER_PROMPT)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

async def retrieve_memory_tools() -> MemoryBundle:
    """
    Build a dictionary of namespace templates that this core agent wants and
    return BOTH the langmem tools AND the 3 underlying stores (a ``MemoryBundle``).
    We want three namespaces:
      1. 'semantic'   -> ("semantic", "general", "{langgraph_auth_user_id}")
      2. 'episodic'   -> ("episodic", "general", "{langgraph_auth_user_id}")
      3. 'procedural' -> ("procedural", "general", "{langgraph_auth_user_id}")
    """
    # Single source of truth for the 3 namespace tuples lives in
    # pre_model_hook.py (the module that also RESOLVES them per turn), imported
    # here so the tool-build namespaces and the hook-retrieval namespaces can
    # never silently diverge (canonical-9 fidelity round-2 fix: Reuse + AIMA
    # flagged the prior inline duplication).
    namespace_templates = {
        "semantic": _SEMANTIC_NS,    # e.g. namespace="semantic.general.user-123"
        "episodic": _EPISODIC_NS,    # e.g. namespace="episodic.general.user-123"
        "procedural": _PROCEDURAL_NS,  # e.g. namespace="procedural.general.user-123"
    }

    return await get_memory_tools_and_stores(namespace_templates=namespace_templates)

# Lazy construction (mirrors pro/graphs/chat_pro.py's documented rule: "NO
# asyncio.run() at import time"). Importing this module does NO async work and
# opens NO BigQuery connection — the memory bundle + agent are built ONCE on
# first access to any of the public names below, via the loop-safe ``_run_sync``
# (so first access works whether or not an event loop is already running). This
# closes code-review finding H at the right altitude: the previous module-level
# ``asyncio.run(...)`` crashed under a running loop AND did import-time network
# I/O; deferring to first access removes both, matching the sibling agent's
# pattern instead of hand-rolling an import-time bandaid.
_LAZY: dict = {}
_LAZY_NAMES = frozenset({"_MEMORY_BUNDLE", "MEMORY_TOOLS", "pre_model_hook", "my_agent"})


def _build_agent_singletons() -> dict:
    """Build the memory bundle, the 3-store pre_model_hook, and the compiled
    ``my_agent`` exactly once. Returns the public-name → value mapping."""
    bundle = _run_sync(retrieve_memory_tools)
    # Automatic memory-at-intent: a pre_model_hook retrieves the 3 memory kinds
    # and injects them before every model step (heed-by-construction). Closes
    # over the 3 distinct stores (create_react_agent's single `store` slot
    # cannot carry 3).
    hook = make_pre_model_hook(
        bundle.semantic_store,
        bundle.episodic_store,
        bundle.procedural_store,
    )
    logger.debug(
        "Agent tools before creation: %s", [t.name for t in bundle.tools]
    )
    agent = create_react_agent(
        model=DEFAULT_MODEL,
        tools=bundle.tools,
        debug=True,
        prompt=BABY_NICER_PROMPT,
        pre_model_hook=hook,
    )
    logger.debug("Agent successfully created!")
    return {
        "_MEMORY_BUNDLE": bundle,
        "MEMORY_TOOLS": bundle.tools,
        "pre_model_hook": hook,
        "my_agent": agent,
    }


def __getattr__(name: str):
    """PEP 562 module-level lazy attribute access. The heavy build (live BQ
    stores + agent compile) fires on first access to a public name, never at
    import — so importing this module from any context (incl. a running event
    loop) is side-effect-free."""
    if name in _LAZY_NAMES:
        if not _LAZY:
            _LAZY.update(_build_agent_singletons())
        return _LAZY[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")