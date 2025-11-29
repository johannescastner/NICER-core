#!/usr/bin/env python3
"""
LLM Cost Tracking Wrapper

This module wraps LLM calls to automatically stream cost and account data to BigQuery.
It integrates with:
1. LangSmith tracing for run correlation
2. BigQuery conversation logs for turn correlation  
3. Live account streaming for real-time balance monitoring
4. Provider-specific cost calculation

Usage:
    from src.monitoring.llm_cost_wrapper import wrap_llm_with_cost_tracking
    
    llm = create_llm("chat")
    tracked_llm = wrap_llm_with_cost_tracking(llm, conversation_id="conv_123", turn_number=1)
    
    # All calls now automatically stream cost data to BigQuery
    response = tracked_llm.invoke("What is 2+2?")
"""

import asyncio
import logging
import threading
import concurrent.futures
import queue
from typing import Any, Dict, Optional, Union, List
from functools import wraps
import inspect
from datetime import datetime

# LangChain imports for _generate method override
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import LLMResult
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import Runnable
from langchain_core.runnables.base import RunnableBinding
from typing import Union
from pydantic import Field

from src.monitoring.live_account_streaming import get_live_streamer, record_api_call_cost
from src.langgraph_slack.config import get_current_provider_info
from src.langgraph_slack.deepseek_utils import DeepSeekLLMWrapper

_logger = logging.getLogger(__name__)


class AsyncWorkerSingleton:
    """Singleton async worker to handle all balance streaming without event loop conflicts."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._task_queue = queue.Queue()
            self._worker_thread = None
            self._loop = None
            self._shutdown = False
            self._start_worker()
            self._initialized = True

    def _start_worker(self):
        """Start the dedicated async worker thread."""
        def worker():
            """Worker thread that runs the async event loop."""
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            try:
                self._loop.run_until_complete(self._async_worker())
            except Exception as e:
                _logger.error(f"Async worker thread failed: {e}")
            finally:
                self._loop.close()

        self._worker_thread = threading.Thread(target=worker, daemon=False)
        self._worker_thread.start()
        _logger.debug("🚀 Started dedicated async worker thread for balance streaming")

    async def _async_worker(self):
        """Async worker that processes balance streaming tasks."""
        while not self._shutdown:
            try:
                # Check for new tasks every 100ms
                await asyncio.sleep(0.1)

                # Process all queued tasks
                while not self._task_queue.empty():
                    try:
                        task_func = self._task_queue.get_nowait()
                        await task_func()
                        self._task_queue.task_done()
                    except queue.Empty:
                        break
                    except Exception as e:
                        _logger.error(f"Failed to process balance streaming task: {e}")

            except Exception as e:
                _logger.error(f"Async worker loop error: {e}")

    def submit_balance_streaming(self, callback_instance):
        """Submit a balance streaming task to the worker."""
        async def stream_task():
            """The actual streaming task."""
            try:
                await callback_instance._stream_balance_if_enabled()
                _logger.debug("✅ Balance streaming completed via async worker")
            except Exception as e:
                _logger.error(f"Balance streaming task failed: {e}")

        if not self._shutdown:
            self._task_queue.put(stream_task)
            _logger.debug("📤 Submitted balance streaming task to async worker")

    def shutdown(self):
        """Shutdown the async worker."""
        self._shutdown = True
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)


# Global async worker instance
_async_worker = AsyncWorkerSingleton()


class CostTrackingCallback(BaseCallbackHandler):
    """Callback handler that tracks LLM costs and streams to BigQuery."""
    
    def __init__(self,
                 conversation_id: Optional[str] = None,
                 turn_number: Optional[int] = None,
                 auto_stream_balance: bool = True):
        """Initialize cost tracking callback."""
        super().__init__()
        self.conversation_id = conversation_id
        self.turn_number = turn_number
        self.call_start_time = None
        self.langsmith_run_id = None
        self.auto_stream_balance = auto_stream_balance
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts running."""
        self.call_start_time = datetime.now()
        # Try to get LangSmith run ID from kwargs
        self.langsmith_run_id = kwargs.get("run_id")
        _logger.debug(f"LLM call started for conversation {self.conversation_id}")

        # Stream balance before LLM call
        if self.auto_stream_balance:
            self._stream_balance_threadsafe()
        
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM ends successfully."""
        try:
            input_tokens = 0
            output_tokens = 0

            # Method 1: Try to extract from llm_output (traditional LangChain way)
            if hasattr(response, 'llm_output') and response.llm_output:
                token_usage = response.llm_output.get('token_usage', {})
                input_tokens = token_usage.get('prompt_tokens', 0)
                output_tokens = token_usage.get('completion_tokens', 0)
                _logger.debug(f"Extracted tokens from llm_output: {input_tokens} input, "
                             f"{output_tokens} output")

            # Method 2: Try to extract from generations[0][0].message.response_metadata (modern way)
            if (input_tokens == 0 and output_tokens == 0 and
                hasattr(response, 'generations') and response.generations and
                len(response.generations) > 0 and len(response.generations[0]) > 0):

                generation = response.generations[0][0]
                if (hasattr(generation, 'message') and
                    hasattr(generation.message, 'response_metadata')):
                    metadata = generation.message.response_metadata
                    if 'token_usage' in metadata:
                        token_usage = metadata['token_usage']
                        input_tokens = token_usage.get('prompt_tokens', 0)
                        output_tokens = token_usage.get('completion_tokens', 0)
                        _logger.debug(f"Extracted tokens from response_metadata: "
                                     f"{input_tokens} input, {output_tokens} output")

            if input_tokens > 0 or output_tokens > 0:
                # Get current provider info
                provider_info = get_current_provider_info()
                model = provider_info["model"]

                _logger.info(f"Recording cost for {input_tokens} input + "
                            f"{output_tokens} output tokens")

                # Update usage tokens in the live streamer
                try:
                    from src.monitoring.live_account_streaming import get_live_streamer
                    streamer = get_live_streamer()
                    streamer.update_usage_from_api_response(input_tokens, output_tokens)
                except Exception as e:
                    _logger.warning(f"Failed to update usage tokens in streamer: {e}")

                # Record cost using thread-safe approach
                self._record_cost_threadsafe(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )
            else:
                _logger.warning("No token usage found in LLM response")

            # Stream balance after LLM call
            if self.auto_stream_balance:
                self._stream_balance_threadsafe()

        except Exception as e:
            _logger.error(f"Failed to track LLM cost: {e}")
            import traceback
            _logger.error(f"Cost tracking traceback: {traceback.format_exc()}")

    def _record_cost_threadsafe(self, model: str, input_tokens: int, output_tokens: int,
                                balance_before: Optional[float] = None,
                                balance_after: Optional[float] = None,
                                actual_cost: Optional[float] = None):
        """Record cost metrics in a thread-safe way that works in both sync and async contexts."""
        def run_async_in_thread():
            """Run the async cost recording in a separate thread with its own event loop."""
            try:
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self._record_cost_async(
                        model, input_tokens, output_tokens,
                        balance_before, balance_after, actual_cost
                    ))
                finally:
                    loop.close()
            except Exception as e:
                _logger.error(f"Failed to record cost in background thread: {e}")

        # Run in background thread with daemon=False to ensure completion
        thread = threading.Thread(target=run_async_in_thread, daemon=False)
        thread.start()
        _logger.debug(f"Started background cost recording thread for {input_tokens + output_tokens} tokens")

    def _stream_balance_threadsafe(self):
        """Stream account balance using the dedicated async worker to prevent event loop conflicts."""
        global _async_worker
        _async_worker.submit_balance_streaming(self)

    async def _stream_balance_if_enabled(self):
        """Stream account balance if auto-streaming is enabled."""
        if self.auto_stream_balance:
            try:
                _logger.debug("Starting account balance streaming...")
                streamer = get_live_streamer()
                snapshot = await streamer.stream_account_snapshot()
                _logger.info(f"✅ Account snapshot streamed: {snapshot.provider} - ${snapshot.balance_usd} - {snapshot.account_status}")
            except Exception as e:
                _logger.error(f"Failed to stream account balance: {e}")
                import traceback
                _logger.error(f"Account streaming traceback: {traceback.format_exc()}")
    
    async def _record_cost_async(self, model: str, input_tokens: int, output_tokens: int,
                                balance_before: Optional[float] = None,
                                balance_after: Optional[float] = None,
                                actual_cost: Optional[float] = None):
        """Record cost metrics asynchronously."""
        try:
            await record_api_call_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                conversation_id=self.conversation_id,
                turn_number=self.turn_number,
                langsmith_run_id=None,  # LangSmith run ID not available in this context
                balance_before=balance_before,
                balance_after=balance_after,
                actual_cost=actual_cost
            )
        except Exception as e:
            _logger.error(f"Failed to record API call cost: {e}")


class CostTrackingLLMWrapper(BaseChatModel):
    """Wrapper that adds cost tracking to any LLM and implements proper LangChain Runnable interface."""

    llm: Union[BaseLanguageModel, Runnable, RunnableBinding, DeepSeekLLMWrapper] = Field(
        ...,
        description="The underlying LLM to wrap with cost tracking"
    )
    conversation_id: Optional[str] = None
    turn_number: Optional[int] = None
    auto_stream_balance: bool = True

    def __init__(self,
                 llm: Union[BaseLanguageModel, Runnable, RunnableBinding],
                 conversation_id: Optional[str] = None,
                 turn_number: Optional[int] = None,
                 auto_stream_balance: bool = True,
                 **kwargs):
        """Initialize the cost tracking wrapper."""
        super().__init__(
            llm=llm,
            conversation_id=conversation_id,
            turn_number=turn_number,
            auto_stream_balance=auto_stream_balance,
            **kwargs
        )

        _logger.info(f"Cost tracking enabled for {type(self.llm).__name__}")

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        """Override _generate to capture balance before and after LLM call - this is where actual LLM calls happen."""
        _logger.info("🔄 _GENERATE METHOD CALLED - Starting balance difference calculation")

        # Get balance before the call
        _logger.info("🔍 About to call _get_balance_sync() for balance_before")
        try:
            balance_before = self._get_balance_sync()
            _logger.info(f"💰 Balance before LLM call: ${balance_before}")
        except Exception as e:
            _logger.error(f"🚨 Exception in balance_before call: {e}")
            balance_before = None

        # Call the original _generate method - handle both chat models and regular LLMs
        if hasattr(self.llm, '_generate'):
            result = self.llm._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        else:
            # For regular LLMs, convert messages to prompt and use generate
            prompt = "\n".join([msg.content for msg in messages if hasattr(msg, 'content')])
            llm_result = self.llm.generate([prompt], stop=stop, **kwargs)
            # Convert LLMResult to ChatResult
            if llm_result.generations and len(llm_result.generations) > 0:
                generation = llm_result.generations[0][0]
                chat_generation = ChatGeneration(
                    message=AIMessage(content=generation.text),
                    generation_info=generation.generation_info
                )
                result = ChatResult(generations=[chat_generation], llm_output=llm_result.llm_output)
            else:
                result = ChatResult(generations=[], llm_output=llm_result.llm_output)

        # Get balance after the call
        _logger.info("🔍 About to call _get_balance_sync() for balance_after")
        try:
            balance_after = self._get_balance_sync()
            _logger.info(f"💰 Balance after LLM call: ${balance_after}")
        except Exception as e:
            _logger.error(f"🚨 Exception in balance_after call: {e}")
            balance_after = None

        # Calculate actual cost
        actual_cost = None
        if balance_before is not None and balance_after is not None:
            actual_cost = balance_before - balance_after
            _logger.info(f"💵 Calculated actual cost: ${actual_cost}")
        else:
            _logger.warning("⚠️ Could not calculate actual cost - balance values are None")

        # Get model name from provider info
        provider_info = get_current_provider_info()
        model_name = provider_info.get("model", "unknown")

        # Extract token usage from result
        input_tokens = 0
        output_tokens = 0
        if result.llm_output and 'token_usage' in result.llm_output:
            token_usage = result.llm_output['token_usage']
            input_tokens = token_usage.get('prompt_tokens', 0)
            output_tokens = token_usage.get('completion_tokens', 0)
        elif len(result.generations) > 0 and hasattr(result.generations[0], 'message'):
            # Try to get usage from the message
            message = result.generations[0].message
            if hasattr(message, 'usage_metadata') and message.usage_metadata:
                input_tokens = message.usage_metadata.get('input_tokens', 0)
                output_tokens = message.usage_metadata.get('output_tokens', 0)

        # Record cost with balance information
        if input_tokens > 0 or output_tokens > 0:
            self._record_cost_threadsafe(
                model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                balance_before=balance_before,
                balance_after=balance_after,
                actual_cost=actual_cost
            )

        return result

    def bind_tools(self, tools, **kwargs):
        """Override bind_tools to maintain cost tracking on the bound LLM."""
        _logger.debug(f"🔧 bind_tools called with {len(tools)} tools - creating new cost tracking wrapper")

        # Get the bound LLM from the wrapped LLM
        bound_llm = self.llm.bind_tools(tools, **kwargs)

        # Create a new CostTrackingLLMWrapper around the bound LLM to preserve cost tracking
        return CostTrackingLLMWrapper(
            llm=bound_llm,
            conversation_id=self.conversation_id,
            turn_number=self.turn_number,
            auto_stream_balance=self.auto_stream_balance
        )

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return f"cost-tracking-{getattr(self.llm, '_llm_type', type(self.llm).__name__.lower())}"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        base_params = {}
        if hasattr(self.llm, '_identifying_params'):
            base_params = self.llm._identifying_params

        return {
            **base_params,
            "cost_tracking_enabled": True,
            "conversation_id": self.conversation_id,
            "turn_number": self.turn_number,
            "auto_stream_balance": self.auto_stream_balance,
        }

    def _record_cost_threadsafe(self, model: str, input_tokens: int, output_tokens: int,
                                balance_before: Optional[float] = None,
                                balance_after: Optional[float] = None,
                                actual_cost: Optional[float] = None):
        """Record cost metrics in a thread-safe way that works in both sync and async contexts."""
        def run_async_in_thread():
            """Run the async cost recording in a separate thread with its own event loop."""
            try:
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self._record_cost_async(
                        model, input_tokens, output_tokens,
                        balance_before, balance_after, actual_cost
                    ))
                finally:
                    loop.close()
            except Exception as e:
                _logger.error(f"Failed to record cost in background thread: {e}")

        # Run in background thread with daemon=False to ensure completion
        thread = threading.Thread(target=run_async_in_thread, daemon=False)
        thread.start()
        _logger.debug(f"Started background cost recording thread for {input_tokens + output_tokens} tokens")

    async def _record_cost_async(self, model: str, input_tokens: int, output_tokens: int,
                                balance_before: Optional[float] = None,
                                balance_after: Optional[float] = None,
                                actual_cost: Optional[float] = None):
        """Record cost metrics asynchronously."""
        try:
            await record_api_call_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                conversation_id=self.conversation_id,
                turn_number=self.turn_number,
                langsmith_run_id=None,  # We don't have run_id in this context
                balance_before=balance_before,
                balance_after=balance_after,
                actual_cost=actual_cost
            )
        except Exception as e:
            _logger.error(f"Failed to record API call cost: {e}")

    @property
    def model(self):
        """Expose model attribute - maps to model_name for compatibility."""
        return getattr(self.llm, 'model_name', None)

    @property
    def model_name(self):
        """Expose model_name attribute without bypassing cost tracking."""
        return getattr(self.llm, 'model_name', None)

    @property
    def temperature(self):
        """Expose temperature attribute without bypassing cost tracking."""
        return getattr(self.llm, 'temperature', None)

    @property
    def max_tokens(self):
        """Expose max_tokens attribute without bypassing cost tracking."""
        return getattr(self.llm, 'max_tokens', None)

    @property
    def __name__(self):
        """Expose __name__ attribute without bypassing cost tracking."""
        return getattr(self.llm, '__name__', type(self.llm).__name__)

    def __getattr__(self, name):
        """Delegate other attributes to the wrapped LLM with CRITICAL ERROR logging."""
        # 🚨 CRITICAL: ANY attribute access through __getattr__ means we're missing
        # an attribute on the wrapper and potentially bypassing cost tracking!
        _logger.error(f"🚨 CRITICAL BYPASS: '{name}' attribute missing from CostTrackingLLMWrapper - this bypasses cost tracking!")
        _logger.error(f"🚨 Add @property {name} to CostTrackingLLMWrapper to fix this bypass!")

        attr = getattr(self.llm, name)
        return attr
    
    async def _stream_balance_if_enabled(self):
        """Stream account balance if auto-streaming is enabled."""
        if self.auto_stream_balance:
            try:
                _logger.debug("Starting account balance streaming...")
                streamer = get_live_streamer()
                snapshot = await streamer.stream_account_snapshot()
                _logger.info(f"✅ Account snapshot streamed: {snapshot.provider} - ${snapshot.balance_usd} - {snapshot.account_status}")
            except Exception as e:
                _logger.error(f"Failed to stream account balance: {e}")
                import traceback
                _logger.error(f"Account streaming traceback: {traceback.format_exc()}")
    


    def _get_balance_sync(self) -> Optional[float]:
        """Get current account balance synchronously."""
        try:
            _logger.info("🔍 _get_balance_sync: Starting balance retrieval")
            from src.monitoring.live_account_streaming import get_live_streamer
            streamer = get_live_streamer()
            _logger.info("🔍 _get_balance_sync: Got live streamer")

            # Use the synchronous method that exists in LiveAccountStreamer
            balance = streamer.get_current_balance_sync()
            _logger.info(f"🔍 _get_balance_sync: Retrieved balance: {balance}")
            return balance
        except Exception as e:
            _logger.error(f"🚨 Failed to get balance in _get_balance_sync: {e}")
            import traceback
            _logger.error(f"🚨 Balance retrieval traceback: {traceback.format_exc()}")
            return None

    def _record_cost_with_balance(self, model: str, input_tokens: int, output_tokens: int,
                                 balance_before: Optional[float], balance_after: Optional[float],
                                 actual_cost: Optional[float]):
        """Record cost with actual balance verification - synchronous version."""
        try:
            # Use synchronous recording to avoid event loop issues
            import asyncio

            def run_recording():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(
                            record_api_call_cost(
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                model=model,
                                conversation_id=self.conversation_id,
                                turn_number=self.turn_number,
                                langsmith_run_id=str(self.cost_callback.langsmith_run_id) if self.cost_callback.langsmith_run_id else None,
                                balance_before=balance_before,
                                balance_after=balance_after,
                                actual_cost=actual_cost
                            )
                        )
                        _logger.info(f"✅ Cost recorded successfully: ${actual_cost:.6f}" if actual_cost else "✅ Cost recorded (estimated)")
                    finally:
                        loop.close()
                except Exception as e:
                    _logger.error(f"Failed to record cost in new event loop: {e}")

            # Run in background thread
            import threading
            thread = threading.Thread(target=run_recording, daemon=False)
            thread.start()

        except Exception as e:
            _logger.error(f"Failed to record cost with balance: {e}")



    def _stream_balance_threadsafe(self):
        """Stream account balance in a thread-safe way."""
        def run_async_in_thread():
            """Run the async balance streaming in a separate thread with its own event loop."""
            try:
                _logger.debug("Creating new event loop for account balance streaming...")
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self._stream_balance_if_enabled())
                    _logger.debug("Account balance streaming completed successfully")
                finally:
                    loop.close()
            except Exception as e:
                _logger.error(f"Failed to stream balance in background thread: {e}")
                import traceback
                _logger.error(f"Balance streaming thread traceback: {traceback.format_exc()}")

        # Run in background thread with daemon=False to ensure completion
        thread = threading.Thread(target=run_async_in_thread, daemon=False)
        thread.start()
        _logger.info("🚀 Started background balance streaming thread")
    



def wrap_llm_with_cost_tracking(llm: BaseLanguageModel,
                               conversation_id: Optional[str] = None,
                               turn_number: Optional[int] = None,
                               auto_stream_balance: bool = True) -> CostTrackingLLMWrapper:
    """
    Wrap an LLM with cost tracking capabilities.
    
    Args:
        llm: The LLM to wrap
        conversation_id: Optional conversation ID for correlation
        turn_number: Optional turn number for correlation
        auto_stream_balance: Whether to automatically stream account balance
        
    Returns:
        Wrapped LLM with cost tracking
    """
    return CostTrackingLLMWrapper(
        llm=llm,
        conversation_id=conversation_id,
        turn_number=turn_number,
        auto_stream_balance=auto_stream_balance
    )


def enable_global_cost_tracking():
    """
    Enable global cost tracking by monkey-patching the create_llm function.
    This makes ALL LLM calls automatically track costs.
    """
    from src.langgraph_slack.config import create_llm as original_create_llm
    
    def create_llm_with_tracking(*args, **kwargs):
        """Create LLM with automatic cost tracking."""
        llm = original_create_llm(*args, **kwargs)
        return wrap_llm_with_cost_tracking(llm)
    
    # Replace the original function
    import src.langgraph_slack.config
    src.langgraph_slack.config.create_llm = create_llm_with_tracking
    
    _logger.info("Global cost tracking enabled - all LLM calls will now stream cost data")


async def get_recent_cost_summary(hours: int = 1) -> Dict[str, Any]:
    """Get a summary of recent costs."""
    streamer = get_live_streamer()
    return await streamer.get_cost_analytics(hours=hours)


async def demo_cost_tracking():
    """Demo the cost tracking functionality."""
    print("🔍 COST TRACKING DEMO")
    print("=" * 50)
    
    # Create a tracked LLM
    from src.langgraph_slack.config import create_llm
    llm = create_llm("cheap")
    tracked_llm = wrap_llm_with_cost_tracking(
        llm, 
        conversation_id="demo_conv_123",
        turn_number=1
    )
    
    print(f"✅ Created tracked LLM: {type(tracked_llm.llm).__name__}")
    
    # Make a test call
    print("🔄 Making test API call...")
    response = tracked_llm.invoke("What is 2+2? Be brief.")
    print(f"📝 Response: {response.content}")
    
    # Wait a moment for async operations
    await asyncio.sleep(2)
    
    # Get cost summary
    print("💰 Getting cost summary...")
    summary = await get_recent_cost_summary(hours=1)
    print(f"📊 Recent costs: ${summary['total_cost']:.4f}")
    print(f"📞 Total calls: {summary['total_calls']}")
    print(f"🔤 Total tokens: {summary['total_tokens']}")
    
    return summary


if __name__ == "__main__":
    asyncio.run(demo_cost_tracking())
