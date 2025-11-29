#!/usr/bin/env python3
"""
Live Account Balance & Usage Streaming to BigQuery

This module streams real-time account information alongside LangSmith traces:
1. DeepSeek account balance and usage
2. OpenAI account balance and usage  
3. Cost calculations per API call
4. Provider-specific metrics
5. Integration with existing BigQuery conversation logs

Features:
- Real-time balance monitoring
- Cost tracking per conversation turn
- Provider usage analytics
- Integration with MODEL_PROVIDER toggle
- Streaming to BigQuery with LangSmith trace correlation
"""

import os
import json
import asyncio
import logging
import concurrent.futures
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import httpx
from google.cloud import bigquery

from src.langgraph_slack.config import get_current_provider_info, CURRENT_PROVIDER

_logger = logging.getLogger(__name__)


@dataclass
class AccountSnapshot:
    """Snapshot of account balance and usage at a point in time."""
    provider: str
    timestamp: datetime
    balance_usd: Optional[float]
    usage_tokens_input: Optional[int]
    usage_tokens_output: Optional[int]
    usage_requests: Optional[int]
    cost_per_1k_input: Optional[float]
    cost_per_1k_output: Optional[float]
    rate_limit_remaining: Optional[int]
    rate_limit_reset: Optional[datetime]
    account_status: str
    error_message: Optional[str] = None


@dataclass
class CallCostMetrics:
    """Cost metrics for a specific API call."""
    provider: str
    model: str
    timestamp: datetime
    conversation_id: Optional[str]
    turn_number: Optional[int]
    langsmith_run_id: Optional[str]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    actual_cost_usd: Optional[float]
    balance_before: Optional[float]
    balance_after: Optional[float]


class LiveAccountStreamer:
    """Streams live account data to BigQuery alongside LangSmith traces."""
    
    def __init__(self):
        """Initialize the live account streamer."""
        # BigQuery setup (reuse existing credentials)
        from src.langgraph_slack.config import PROJECT_ID, CREDENTIALS
        self.project_id = PROJECT_ID
        self.bq_client = bigquery.Client(project=PROJECT_ID, credentials=CREDENTIALS)
        self.dataset_id = "conversation_logs"
        
        # Initialize tables
        self._ensure_tables_exist()
        
        # Provider-specific clients
        self.deepseek_client = httpx.AsyncClient(
            base_url="https://api.deepseek.com/v1",
            headers={"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"}
        )
        
        self.openai_client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
        )

        # Track cumulative usage tokens from API responses
        self._session_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "requests": 0
        }

        _logger.info("LiveAccountStreamer initialized")

    def update_usage_from_api_response(self, input_tokens: int, output_tokens: int):
        """Update cumulative usage tokens from API response."""
        self._session_usage["input_tokens"] += input_tokens
        self._session_usage["output_tokens"] += output_tokens
        self._session_usage["requests"] += 1
        _logger.debug(f"Updated usage: {self._session_usage}")

    def _ensure_tables_exist(self):
        """Ensure BigQuery tables exist for account streaming."""
        
        # Account snapshots table
        account_snapshots_schema = [
            bigquery.SchemaField("provider", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("balance_usd", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("usage_tokens_input", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("usage_tokens_output", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("usage_requests", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("cost_per_1k_input", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("cost_per_1k_output", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("rate_limit_remaining", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("rate_limit_reset", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("account_status", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("error_message", "STRING", mode="NULLABLE"),
        ]
        
        # Call cost metrics table
        call_costs_schema = [
            bigquery.SchemaField("provider", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("model", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("conversation_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("turn_number", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("langsmith_run_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("input_tokens", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("output_tokens", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("total_tokens", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("estimated_cost_usd", "NUMERIC", mode="REQUIRED"),
            bigquery.SchemaField("actual_cost_usd", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("balance_before", "NUMERIC", mode="NULLABLE"),
            bigquery.SchemaField("balance_after", "NUMERIC", mode="NULLABLE"),
        ]
        
        # Create tables if they don't exist
        self._create_table_if_not_exists("account_snapshots", account_snapshots_schema)
        self._create_table_if_not_exists("call_cost_metrics", call_costs_schema)
    
    def _create_table_if_not_exists(self, table_name: str, schema: List[bigquery.SchemaField]):
        """Create a BigQuery table if it doesn't exist."""
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        
        try:
            self.bq_client.get_table(table_id)
            _logger.info(f"Table {table_name} already exists")
        except Exception:
            table = bigquery.Table(table_id, schema=schema)
            
            # Partition by timestamp for performance
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="timestamp"
            )
            
            # Cluster for better query performance
            if table_name == "account_snapshots":
                table.clustering_fields = ["provider", "account_status"]
            elif table_name == "call_cost_metrics":
                table.clustering_fields = ["provider", "model", "conversation_id"]
            
            table = self.bq_client.create_table(table)
            _logger.info(f"Created table {table_name}")
    
    async def get_deepseek_account_info(self) -> AccountSnapshot:
        """Get current DeepSeek account information."""
        try:
            # Create fresh client for this event loop to avoid "Event loop is closed" errors
            import os
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY not found in environment")

            _logger.info("🔍 Creating fresh DeepSeek client for current event loop")
            async with httpx.AsyncClient(
                base_url=base_url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0
            ) as fresh_client:
                # DeepSeek API endpoint for account info (correct endpoint is /user/balance)
                response = await fresh_client.get("/user/balance")

                if response.status_code == 200:
                    data = response.json()
                    _logger.info(f"🔍 DeepSeek balance API response: {data}")

                    # Parse DeepSeek balance response format
                    is_available = data.get("is_available", False)
                    balance_infos = data.get("balance_infos", [])
                    _logger.info(f"🔍 DeepSeek is_available: {is_available}, balance_infos: {balance_infos}")

                    # Find USD balance, fallback to CNY if available
                    balance_usd = None
                    balance_cny = None

                    for balance_info in balance_infos:
                        currency = balance_info.get("currency")
                        total_balance = float(balance_info.get("total_balance", 0))
                        _logger.info(f"🔍 Processing balance_info: currency={currency}, total_balance={total_balance}")

                        if currency == "USD":
                            balance_usd = total_balance
                        elif currency == "CNY":
                            balance_cny = total_balance

                    # Convert CNY to USD if needed (approximate rate: 1 USD = 7.2 CNY)
                    if balance_usd is None and balance_cny is not None:
                        balance_usd = balance_cny / 7.2
                        _logger.info(f"🔍 Converted CNY to USD: {balance_cny} CNY -> {balance_usd} USD")

                    _logger.info(f"🔍 Final balance_usd: {balance_usd}")

                    # Parse rate limit reset from headers if available
                    rate_limit_reset = None
                    if "x-ratelimit-reset" in response.headers:
                        try:
                            reset_timestamp = int(response.headers["x-ratelimit-reset"])
                            rate_limit_reset = datetime.fromtimestamp(reset_timestamp, timezone.utc)
                        except (ValueError, TypeError) as e:
                            _logger.debug(f"Could not parse rate limit reset: {e}")

                    return AccountSnapshot(
                        provider="deepseek",
                        timestamp=datetime.now(timezone.utc),
                        balance_usd=balance_usd,
                        usage_tokens_input=self._session_usage["input_tokens"],  # From API call responses
                        usage_tokens_output=self._session_usage["output_tokens"],  # From API call responses
                        usage_requests=self._session_usage["requests"],  # From API call responses
                        cost_per_1k_input=0.14,  # DeepSeek pricing
                        cost_per_1k_output=0.28,
                        rate_limit_remaining=int(response.headers.get("x-ratelimit-remaining", 0)),
                        rate_limit_reset=rate_limit_reset,
                        account_status="active" if is_available else "insufficient_balance",
                        error_message=None  # Explicitly set to None for successful calls
                    )
                else:
                    _logger.error(f"🚨 DeepSeek balance API error: {response.status_code} - {response.text}")
                    _logger.error(f"🚨 DeepSeek API URL: {base_url}/user/balance")
                    return AccountSnapshot(
                        provider="deepseek",
                        timestamp=datetime.now(timezone.utc),
                        balance_usd=None,
                        usage_tokens_input=None,
                        usage_tokens_output=None,
                        usage_requests=None,
                        cost_per_1k_input=0.14,
                        cost_per_1k_output=0.28,
                        rate_limit_remaining=None,
                        rate_limit_reset=None,
                        account_status="error",
                        error_message=f"API error: {response.status_code}"
                    )
                
        except Exception as e:
            _logger.error(f"🚨 Failed to get DeepSeek account info: {e}")
            import traceback
            _logger.error(f"🚨 DeepSeek balance API traceback: {traceback.format_exc()}")
            return AccountSnapshot(
                provider="deepseek",
                timestamp=datetime.now(timezone.utc),
                balance_usd=None,
                usage_tokens_input=None,
                usage_tokens_output=None,
                usage_requests=None,
                cost_per_1k_input=0.14,
                cost_per_1k_output=0.28,
                rate_limit_remaining=None,
                rate_limit_reset=None,
                account_status="error",
                error_message=str(e)
            )
    
    async def get_openai_account_info(self) -> AccountSnapshot:
        """Get current OpenAI account information.

        Note: OpenAI has NO official billing/usage API endpoints. All endpoints like
        /v1/usage were deprecated in 2023 and require browser session tokens only.
        We return a basic snapshot with balance_usd=None since real-time balance
        tracking is impossible with OpenAI's API.
        """
        try:
            # OpenAI doesn't provide programmatic access to usage/billing data
            # Return a basic snapshot indicating the provider is available
            return AccountSnapshot(
                provider="openai",
                timestamp=datetime.now(timezone.utc),
                balance_usd=None,  # OpenAI doesn't provide balance API
                usage_tokens_input=None,  # No usage API available
                usage_tokens_output=None,
                usage_requests=None,
                cost_per_1k_input=15.0,  # GPT-4o pricing for cost calculation
                cost_per_1k_output=30.0,
                rate_limit_remaining=None,  # Would need to track from API response headers
                rate_limit_reset=None,
                account_status="active",  # Assume active since we can't check
                error_message=None
            )
                
        except Exception as e:
            _logger.error(f"Failed to get OpenAI account info: {e}")
            return AccountSnapshot(
                provider="openai",
                timestamp=datetime.now(timezone.utc),
                balance_usd=None,
                usage_tokens_input=None,
                usage_tokens_output=None,
                usage_requests=None,
                cost_per_1k_input=15.0,
                cost_per_1k_output=30.0,
                rate_limit_remaining=None,
                rate_limit_reset=None,
                account_status="error",
                error_message=str(e)
            )

    async def stream_account_snapshot(self, provider: Optional[str] = None) -> AccountSnapshot:
        """Stream current account snapshot to BigQuery."""
        if provider is None:
            provider = CURRENT_PROVIDER

        if provider == "deepseek":
            snapshot = await self.get_deepseek_account_info()
        elif provider == "openai":
            snapshot = await self.get_openai_account_info()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Stream to BigQuery
        await self._insert_account_snapshot(snapshot)
        return snapshot

    def get_current_balance_sync(self) -> Optional[float]:
        """Get current account balance synchronously - handles existing event loops."""
        try:
            import asyncio
            _logger.info("🔍 get_current_balance_sync: Starting balance retrieval")

            # Check if we're already in an event loop
            try:
                current_loop = asyncio.get_running_loop()
                # We're in an async context - use asyncio.create_task or run in thread
                _logger.info("🔍 get_current_balance_sync: Already in event loop - using thread-based approach")

                def get_balance_in_thread():
                    """Run balance check in separate thread with its own event loop."""
                    try:
                        _logger.info("🔍 Thread: Creating new event loop")
                        # Create new event loop in this thread
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            _logger.info("🔍 Thread: Creating fresh LiveAccountStreamer")
                            # Create a fresh LiveAccountStreamer instance for this thread
                            fresh_streamer = LiveAccountStreamer()
                            _logger.info("🔍 Thread: Calling stream_account_snapshot")
                            # Get account snapshot using the existing async method
                            snapshot = new_loop.run_until_complete(fresh_streamer.stream_account_snapshot())
                            _logger.info(f"🔍 Thread: Got snapshot: {snapshot}")
                            if snapshot:
                                _logger.info(f"🔍 Thread: Balance USD: {snapshot.balance_usd}")
                                return snapshot.balance_usd
                            else:
                                _logger.error("🚨 Thread: Snapshot is None")
                                return None
                        finally:
                            _logger.info("🔍 Thread: Closing event loop")
                            new_loop.close()
                    except (asyncio.TimeoutError, httpx.RequestError, Exception) as e:
                        _logger.error(f"🚨 Thread: Failed to get balance: {e}")
                        import traceback
                        _logger.error(f"🚨 Thread: Balance traceback: {traceback.format_exc()}")
                        return None

                # Run in thread to avoid event loop conflicts
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(get_balance_in_thread)
                    return future.result(timeout=30)  # 30 second timeout

            except RuntimeError:
                # No event loop running - safe to create one
                _logger.info("🔍 get_current_balance_sync: No event loop running - creating new one")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    _logger.info("🔍 get_current_balance_sync: Calling stream_account_snapshot")
                    # Get account snapshot
                    snapshot = loop.run_until_complete(self.stream_account_snapshot())
                    _logger.info(f"🔍 get_current_balance_sync: Got snapshot: {snapshot}")
                    return snapshot.balance_usd if snapshot else None
                finally:
                    loop.close()

        except (RuntimeError, asyncio.TimeoutError, concurrent.futures.TimeoutError, httpx.RequestError, Exception) as e:
            _logger.error(f"Failed to get balance synchronously: {e}")
            import traceback
            _logger.error(f"Balance sync traceback: {traceback.format_exc()}")
            return None

    async def record_call_cost(self,
                              model: str,
                              input_tokens: int,
                              output_tokens: int,
                              conversation_id: Optional[str] = None,
                              turn_number: Optional[int] = None,
                              langsmith_run_id: Optional[str] = None,
                              balance_before: Optional[float] = None,
                              balance_after: Optional[float] = None,
                              actual_cost: Optional[float] = None) -> CallCostMetrics:
        """Record cost metrics for an API call."""
        provider_info = get_current_provider_info()
        provider = provider_info["provider"]

        # Calculate estimated cost
        if provider == "deepseek":
            cost_input = (input_tokens / 1000) * 0.14
            cost_output = (output_tokens / 1000) * 0.28
        elif provider == "openai":
            cost_input = (input_tokens / 1000) * 15.0
            cost_output = (output_tokens / 1000) * 30.0
        else:
            cost_input = cost_output = 0.0

        estimated_cost = cost_input + cost_output

        # Determine actual cost based on provider capabilities
        if actual_cost is not None:
            # DeepSeek: Use real balance difference (even if $0.0 - that's correct!)
            final_actual_cost = actual_cost
            _logger.info(f"🎯 ACTUAL cost recorded: ${actual_cost} "
                       f"(vs estimated: ${estimated_cost})")
        elif provider == "openai":
            # OpenAI: Use estimated cost as actual cost (no balance API available)
            final_actual_cost = estimated_cost
            _logger.info(f"📊 OpenAI estimated cost used as actual: ${estimated_cost} "
                       f"(balance tracking not available)")
        else:
            # Other providers: No actual cost available
            final_actual_cost = None
            _logger.info(f"📊 Estimated cost recorded: ${estimated_cost}")

        call_metrics = CallCostMetrics(
            provider=provider,
            model=model,
            timestamp=datetime.now(timezone.utc),
            conversation_id=conversation_id,
            turn_number=turn_number,
            langsmith_run_id=langsmith_run_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost_usd=estimated_cost,
            actual_cost_usd=final_actual_cost,  # Balance difference for DeepSeek, estimated for OpenAI
            balance_before=balance_before,
            balance_after=balance_after
        )

        # Stream to BigQuery
        await self._insert_call_cost_metrics(call_metrics)
        return call_metrics

    async def _insert_account_snapshot(self, snapshot: AccountSnapshot):
        """Insert account snapshot into BigQuery."""
        table_id = f"{self.project_id}.{self.dataset_id}.account_snapshots"

        # Convert to BigQuery row
        row = {
            "provider": snapshot.provider,
            "timestamp": snapshot.timestamp.isoformat(),
            "balance_usd": snapshot.balance_usd,
            "usage_tokens_input": snapshot.usage_tokens_input,
            "usage_tokens_output": snapshot.usage_tokens_output,
            "usage_requests": snapshot.usage_requests,
            "cost_per_1k_input": snapshot.cost_per_1k_input,
            "cost_per_1k_output": snapshot.cost_per_1k_output,
            "rate_limit_remaining": snapshot.rate_limit_remaining,
            "rate_limit_reset": snapshot.rate_limit_reset.isoformat() if snapshot.rate_limit_reset else None,
            "account_status": snapshot.account_status,
            "error_message": snapshot.error_message
        }

        errors = self.bq_client.insert_rows_json(table_id, [row])
        if errors:
            _logger.error(f"Failed to insert account snapshot: {errors}")
        else:
            _logger.info(f"Account snapshot streamed for {snapshot.provider}")

    async def _insert_call_cost_metrics(self, metrics: CallCostMetrics):
        """Insert call cost metrics into BigQuery."""
        table_id = f"{self.project_id}.{self.dataset_id}.call_cost_metrics"

        # Convert to BigQuery row
        row = {
            "provider": metrics.provider,
            "model": metrics.model,
            "timestamp": metrics.timestamp.isoformat(),
            "conversation_id": metrics.conversation_id,
            "turn_number": metrics.turn_number,
            "langsmith_run_id": metrics.langsmith_run_id,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "total_tokens": metrics.total_tokens,
            "estimated_cost_usd": metrics.estimated_cost_usd,
            "actual_cost_usd": metrics.actual_cost_usd,
            "balance_before": metrics.balance_before,
            "balance_after": metrics.balance_after
        }

        errors = self.bq_client.insert_rows_json(table_id, [row])
        if errors:
            _logger.error(f"Failed to insert call cost metrics: {errors}")
        else:
            _logger.info(f"Call cost metrics streamed: ${metrics.estimated_cost_usd:.4f}")

    async def get_cost_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get cost analytics for the last N hours."""
        query = f"""
        SELECT
            provider,
            model,
            COUNT(*) as total_calls,
            SUM(input_tokens) as total_input_tokens,
            SUM(output_tokens) as total_output_tokens,
            SUM(total_tokens) as total_tokens,
            SUM(estimated_cost_usd) as total_estimated_cost,
            AVG(estimated_cost_usd) as avg_cost_per_call,
            MIN(timestamp) as first_call,
            MAX(timestamp) as last_call
        FROM `{self.project_id}.{self.dataset_id}.call_cost_metrics`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
        GROUP BY provider, model
        ORDER BY total_estimated_cost DESC
        """

        job = self.bq_client.query(query)
        results = [dict(row) for row in job]

        return {
            "analytics_period_hours": hours,
            "total_providers": len(set(r["provider"] for r in results)),
            "cost_breakdown": results,
            "total_cost": sum(r["total_estimated_cost"] for r in results),
            "total_calls": sum(r["total_calls"] for r in results),
            "total_tokens": sum(r["total_tokens"] for r in results)
        }


# Global instance for easy access
_live_streamer: Optional[LiveAccountStreamer] = None


def get_live_streamer() -> LiveAccountStreamer:
    """Get the global live account streamer instance."""
    global _live_streamer
    if _live_streamer is None:
        _logger.info("🔍 get_live_streamer: Creating new LiveAccountStreamer instance")
        try:
            _live_streamer = LiveAccountStreamer()
            _logger.info("🔍 get_live_streamer: LiveAccountStreamer created successfully")
        except Exception as e:
            _logger.error(f"🚨 get_live_streamer: Failed to create LiveAccountStreamer: {e}")
            import traceback
            _logger.error(f"🚨 get_live_streamer: Creation traceback: {traceback.format_exc()}")
            raise
    return _live_streamer


async def stream_current_account() -> AccountSnapshot:
    """Convenience function to stream current provider's account info."""
    streamer = get_live_streamer()
    return await streamer.stream_account_snapshot()


async def record_api_call_cost(input_tokens: int,
                              output_tokens: int,
                              model: str,
                              conversation_id: Optional[str] = None,
                              turn_number: Optional[int] = None,
                              langsmith_run_id: Optional[str] = None,
                              balance_before: Optional[float] = None,
                              balance_after: Optional[float] = None,
                              actual_cost: Optional[float] = None) -> CallCostMetrics:
    """Convenience function to record API call cost with optional balance verification."""
    streamer = get_live_streamer()
    return await streamer.record_call_cost(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        conversation_id=conversation_id,
        turn_number=turn_number,
        langsmith_run_id=langsmith_run_id,
        balance_before=balance_before,
        balance_after=balance_after,
        actual_cost=actual_cost
    )
