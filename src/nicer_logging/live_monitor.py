#!/usr/bin/env python3
"""
Live Conversation Monitoring System

This provides real-time monitoring of conversations stored in BigQuery:
1. Watch conversations as they progress turn-by-turn
2. Monitor errors and performance metrics in real-time
3. Display semantic search results for ongoing conversations
4. Alert on conversation stalls or errors
5. Show LangSmith integration status

Usage:
    python src/logging/live_monitor.py --conversation-id <id>
    python src/logging/live_monitor.py --focus-area semantic_memory
    python src/logging/live_monitor.py --live-all
"""

import asyncio
import argparse
import base64
import json
import logging
import time
from datetime import datetime, timedelta
from os import environ
from typing import Dict, List, Any, Optional
from pathlib import Path

from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConversationMonitor:
    """Real-time conversation monitoring system."""
    
    def __init__(self, 
                 project_id: str = "datawarehouse-447422",
                 dataset_id: str = "conversation_logs"):
        """Initialize the conversation monitor."""
        self.project_id = project_id
        self.dataset_id = dataset_id
        
        # Initialize BigQuery client
        self._init_bigquery_client()
        
        # Track last seen data for live monitoring
        self.last_seen_turns = {}
        self.last_seen_errors = {}
        self.last_seen_metrics = {}
        
        logger.info(f"ConversationMonitor initialized for project {project_id}")
    
    def _init_bigquery_client(self):
        """Initialize BigQuery client with service account credentials."""
        try:
            # Load service account credentials using the same pattern as langgraph_slack
            service_account_json = base64.b64decode(environ.get("GCP_SERVICE_ACCOUNT_BASE64", "")).decode("utf-8")
            service_account_info = json.loads(service_account_json)
            credentials = service_account.Credentials.from_service_account_info(service_account_info)

            self.bq_client = bigquery.Client(
                project=self.project_id,
                credentials=credentials
            )
            logger.info("BigQuery client initialized successfully with service account credentials")

        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {e}")
            logger.error("Make sure GCP_SERVICE_ACCOUNT_BASE64 environment variable is set")
            raise
    
    async def monitor_conversation(self, conversation_id: str, refresh_interval: int = 5):
        """Monitor a specific conversation in real-time."""
        logger.info(f"🔍 Monitoring conversation {conversation_id}")
        logger.info(f"📊 Refresh interval: {refresh_interval} seconds")
        logger.info("=" * 60)
        
        last_turn_count = 0
        
        while True:
            try:
                # Get conversation summary
                summary = await self._get_conversation_summary(conversation_id)
                
                if not summary:
                    logger.warning(f"Conversation {conversation_id} not found")
                    break
                
                # Check for new turns
                current_turn_count = summary.get('total_turns', 0)
                if current_turn_count > last_turn_count:
                    # Display new turns
                    new_turns = await self._get_recent_turns(conversation_id, last_turn_count)
                    for turn in new_turns:
                        self._display_turn(turn)
                    
                    last_turn_count = current_turn_count
                
                # Check for new errors
                await self._check_new_errors(conversation_id)
                
                # Check for new metrics
                await self._check_new_metrics(conversation_id)
                
                # Display status
                status = summary.get('status', 'unknown')
                if status == 'completed':
                    logger.info(f"✅ Conversation {conversation_id} completed")
                    break
                elif status == 'error':
                    logger.error(f"❌ Conversation {conversation_id} ended with error")
                    break
                
                # Wait before next check
                await asyncio.sleep(refresh_interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error during monitoring: {e}")
                await asyncio.sleep(refresh_interval)
    
    async def monitor_all_active(self, refresh_interval: int = 10):
        """Monitor all active conversations."""
        logger.info("🔍 Monitoring all active conversations")
        logger.info(f"📊 Refresh interval: {refresh_interval} seconds")
        logger.info("=" * 60)
        
        while True:
            try:
                # Get all active conversations
                active_conversations = await self._get_active_conversations()
                
                if not active_conversations:
                    logger.info("No active conversations found")
                else:
                    logger.info(f"📈 Found {len(active_conversations)} active conversations")
                    
                    for conv in active_conversations:
                        conv_id = conv['conversation_id']
                        focus_area = conv['focus_area']
                        turn_count = conv['total_turns']
                        duration = self._calculate_duration(conv['start_time'])
                        
                        logger.info(f"  🗣️  {conv_id[:8]}... | {focus_area} | {turn_count} turns | {duration}")
                        
                        # Check for stalled conversations (no activity in 5 minutes)
                        if duration > 300 and turn_count == self.last_seen_turns.get(conv_id, 0):
                            logger.warning(f"  ⚠️  Conversation {conv_id[:8]}... may be stalled")
                        
                        self.last_seen_turns[conv_id] = turn_count
                
                # Check for recent errors across all conversations
                await self._check_recent_errors()
                
                # Display comprehensive metrics
                await self._display_recent_metrics()
                await self._display_comprehensive_metrics()
                
                # Wait before next check
                await asyncio.sleep(refresh_interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error during monitoring: {e}")
                await asyncio.sleep(refresh_interval)
    
    async def search_conversations_live(self, query: str, focus_area: Optional[str] = None):
        """Search conversations semantically and display results."""
        logger.info(f"🔍 Searching conversations for: '{query}'")
        if focus_area:
            logger.info(f"📍 Focus area: {focus_area}")
        logger.info("=" * 60)
        
        # This would use the ConversationLogger's search functionality
        # For now, we'll do a simple text search
        results = await self._search_conversations_text(query, focus_area)
        
        if not results:
            logger.info("No matching conversations found")
        else:
            logger.info(f"Found {len(results)} matching conversations:")
            for result in results:
                conv_id = result['conversation_id']
                content = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                speaker = result['speaker']
                timestamp = result['timestamp']
                
                logger.info(f"  📝 {conv_id[:8]}... | {speaker} | {timestamp}")
                logger.info(f"     {content}")
                logger.info("")
    
    def _display_turn(self, turn: Dict[str, Any]):
        """Display a conversation turn in a formatted way."""
        turn_num = turn.get('turn_number', '?')
        speaker = turn.get('speaker', 'unknown')
        content = turn.get('content', '')
        message_type = turn.get('message_type', 'message')
        timestamp = turn.get('timestamp', '')
        
        # Truncate long content
        if len(content) > 200:
            content = content[:200] + "..."
        
        # Format based on speaker and message type
        if speaker == 'human':
            icon = "👤"
        elif speaker == 'sql_agent':
            icon = "🤖"
        elif speaker == 'intellagent':
            icon = "🧠"
        else:
            icon = "❓"
        
        if message_type == 'error':
            icon = "❌"
        elif message_type == 'tool_call':
            icon = "🔧"
        
        logger.info(f"{icon} Turn {turn_num} | {speaker} | {timestamp}")
        logger.info(f"   {content}")
        logger.info("")
    
    async def _get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation summary from BigQuery."""
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.conversations`
        WHERE conversation_id = @conversation_id
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("conversation_id", "STRING", conversation_id)
            ]
        )
        
        job = self.bq_client.query(query, job_config=job_config)
        results = list(job)
        
        if results:
            return dict(results[0])
        else:
            return {}
    
    async def _get_recent_turns(self, conversation_id: str, since_turn: int) -> List[Dict[str, Any]]:
        """Get recent turns for a conversation."""
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.conversation_turns`
        WHERE conversation_id = @conversation_id
        AND turn_number > @since_turn
        ORDER BY turn_number ASC
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("conversation_id", "STRING", conversation_id),
                bigquery.ScalarQueryParameter("since_turn", "INT64", since_turn)
            ]
        )
        
        job = self.bq_client.query(query, job_config=job_config)
        return [dict(row) for row in job]
    
    async def _get_active_conversations(self) -> List[Dict[str, Any]]:
        """Get all active conversations."""
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.conversations`
        WHERE status = 'active'
        ORDER BY start_time DESC
        """
        
        job = self.bq_client.query(query)
        return [dict(row) for row in job]
    
    async def _check_new_errors(self, conversation_id: str):
        """Check for new errors in a conversation."""
        last_error_time = self.last_seen_errors.get(conversation_id, datetime.min)
        
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.conversation_errors`
        WHERE conversation_id = @conversation_id
        AND timestamp > @last_error_time
        ORDER BY timestamp ASC
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("conversation_id", "STRING", conversation_id),
                bigquery.ScalarQueryParameter("last_error_time", "TIMESTAMP", last_error_time)
            ]
        )
        
        job = self.bq_client.query(query, job_config=job_config)
        errors = list(job)
        
        for error in errors:
            logger.error(f"❌ ERROR: {error['error_type']} - {error['error_message']}")
            self.last_seen_errors[conversation_id] = error['timestamp']
    
    async def _check_new_metrics(self, conversation_id: str):
        """Check for new metrics in a conversation."""
        last_metric_time = self.last_seen_metrics.get(conversation_id, datetime.min)
        
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.conversation_metrics`
        WHERE conversation_id = @conversation_id
        AND timestamp > @last_metric_time
        ORDER BY timestamp ASC
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("conversation_id", "STRING", conversation_id),
                bigquery.ScalarQueryParameter("last_metric_time", "TIMESTAMP", last_metric_time)
            ]
        )
        
        job = self.bq_client.query(query, job_config=job_config)
        metrics = list(job)
        
        for metric in metrics:
            logger.info(f"📊 METRIC: {metric['metric_name']} = {metric['metric_value']}")
            self.last_seen_metrics[conversation_id] = metric['timestamp']

    async def _check_recent_errors(self):
        """Check for recent errors across all conversations."""
        five_minutes_ago = datetime.now() - timedelta(minutes=5)

        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.conversation_errors`
        WHERE timestamp > @five_minutes_ago
        ORDER BY timestamp DESC
        LIMIT 5
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("five_minutes_ago", "TIMESTAMP", five_minutes_ago)
            ]
        )

        job = self.bq_client.query(query, job_config=job_config)
        errors = list(job)

        if errors:
            logger.warning(f"⚠️  {len(errors)} recent errors found:")
            for error in errors:
                conv_id = error['conversation_id'][:8]
                logger.warning(f"   {conv_id}... | {error['error_type']} | {error['error_message'][:50]}...")

    async def _display_recent_metrics(self):
        """Display recent metrics across all conversations."""
        five_minutes_ago = datetime.now() - timedelta(minutes=5)

        query = f"""
        SELECT
          conversation_id,
          metric_name,
          AVG(metric_value) as avg_value,
          COUNT(*) as count
        FROM `{self.project_id}.{self.dataset_id}.conversation_metrics`
        WHERE timestamp > @five_minutes_ago
        GROUP BY conversation_id, metric_name
        ORDER BY conversation_id, metric_name
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("five_minutes_ago", "TIMESTAMP", five_minutes_ago)
            ]
        )

        job = self.bq_client.query(query, job_config=job_config)
        metrics = list(job)

        if metrics:
            logger.info(f"📊 Recent metrics summary ({len(metrics)} metric types):")
            for metric in metrics:
                conv_id = metric['conversation_id'][:8]
                logger.info(f"   {conv_id}... | {metric['metric_name']} | avg: {metric['avg_value']:.2f} | count: {metric['count']}")

    async def _display_comprehensive_metrics(self):
        """Display ALL metrics from the comprehensive metrics ecosystem."""
        try:
            # Get all recent metrics (last 10 minutes for comprehensive view)
            query = f"""
            SELECT
                conversation_id,
                metric_type,
                metric_name,
                metric_value,
                timestamp,
                metadata
            FROM `{self.project_id}.{self.dataset_id}.conversation_metrics`
            WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 10 MINUTE)
            ORDER BY timestamp DESC
            LIMIT 100
            """

            query_job = self.bq_client.query(query)
            results = query_job.result()

            metrics = [dict(row) for row in results]

            if not metrics:
                logger.info("📊 No recent metrics found")
                return

            # Comprehensive categorization of ALL metrics types
            self_discovery_metrics = []
            learning_system_metrics = []
            reflection_metrics = []
            memory_metrics = []
            performance_metrics = []
            metadata_learning_metrics = []
            hierarchical_metrics = []
            tree_search_metrics = []
            dspy_metrics = []
            sql_agent_metrics = []
            error_metrics = []
            timing_metrics = []
            efficiency_metrics = []
            coverage_metrics = []
            question_quality_metrics = []

            for metric in metrics:
                metric_type = metric['metric_type'].lower()
                metric_name = metric['metric_name'].lower()

                # Self-Discovery Orchestrator Metrics
                if 'self_discovery' in metric_type or 'orchestrator' in metric_type or 'optimization_cycles' in metric_name:
                    self_discovery_metrics.append(metric)
                # Learning System Metrics
                elif 'learning' in metric_type or 'knowledge' in metric_name or 'retention' in metric_name:
                    learning_system_metrics.append(metric)
                # Reflection Quality Metrics
                elif 'reflection' in metric_type or 'insight' in metric_name or 'self_assessment' in metric_name:
                    reflection_metrics.append(metric)
                # Memory Integration Metrics
                elif 'memory' in metric_type or 'retrieval' in metric_name or 'consolidation' in metric_name:
                    memory_metrics.append(metric)
                # Performance & Timing Metrics
                elif ('performance' in metric_type or 'timing' in metric_type or 'latency' in metric_name or
                      'response_time' in metric_name or 'processing_time' in metric_name or 'duration' in metric_name):
                    if 'time' in metric_name or 'duration' in metric_name or 'latency' in metric_name:
                        timing_metrics.append(metric)
                    else:
                        performance_metrics.append(metric)
                # Metadata Learning Metrics (7 areas)
                elif ('metadata' in metric_type or 'semantic_memory' in metric_name or 'ego_network' in metric_name or
                      'human_interaction' in metric_name or 'data_limitation' in metric_name):
                    metadata_learning_metrics.append(metric)
                # Hierarchical Learning Metrics
                elif 'hierarchical' in metric_type or 'component_coordination' in metric_name:
                    hierarchical_metrics.append(metric)
                # Tree Search Planning Metrics
                elif 'tree_search' in metric_type or 'planning' in metric_name or 'branching' in metric_name:
                    tree_search_metrics.append(metric)
                # DSPy Optimization Metrics
                elif 'dspy' in metric_type or 'optimization' in metric_name or 'compilation' in metric_name:
                    dspy_metrics.append(metric)
                # SQL Agent Discovery Metrics
                elif ('sql_agent' in metric_type or 'facts_discovered' in metric_name or 'llm_calls' in metric_name or
                      'discovery' in metric_name):
                    sql_agent_metrics.append(metric)
                # Efficiency Metrics
                elif 'efficiency' in metric_name or 'facts_per' in metric_name or 'waste' in metric_name:
                    efficiency_metrics.append(metric)
                # Coverage Metrics
                elif 'coverage' in metric_name or 'completeness' in metric_name:
                    coverage_metrics.append(metric)
                # Question Quality Metrics
                elif 'question' in metric_name or 'excellent' in metric_name or 'poor' in metric_name:
                    question_quality_metrics.append(metric)
                # Error Metrics
                elif 'error' in metric_type or 'failure' in metric_name:
                    error_metrics.append(metric)
                else:
                    # Default to performance for uncategorized
                    performance_metrics.append(metric)

            # Display ALL metric categories
            logger.info("🔬 COMPREHENSIVE METRICS ECOSYSTEM:")

            # Self-Discovery Orchestrator
            if self_discovery_metrics:
                logger.info("  🎯 SELF-DISCOVERY ORCHESTRATOR:")
                for metric in self_discovery_metrics[:2]:
                    conv_id = metric['conversation_id'][:8]
                    timestamp = self._format_timestamp(metric['timestamp'])
                    logger.info(f"    🎯 {conv_id}... | {metric['metric_name']} = {metric['metric_value']:.3f} @ {timestamp}")

            # Learning System Performance
            if learning_system_metrics:
                logger.info("  🎓 LEARNING SYSTEM:")
                for metric in learning_system_metrics[:2]:
                    conv_id = metric['conversation_id'][:8]
                    timestamp = self._format_timestamp(metric['timestamp'])
                    logger.info(f"    🎓 {conv_id}... | {metric['metric_name']} = {metric['metric_value']:.3f} @ {timestamp}")

            # Reflection Quality
            if reflection_metrics:
                logger.info("  🪞 REFLECTION QUALITY:")
                for metric in reflection_metrics[:2]:
                    conv_id = metric['conversation_id'][:8]
                    timestamp = self._format_timestamp(metric['timestamp'])
                    logger.info(f"    🪞 {conv_id}... | {metric['metric_name']} = {metric['metric_value']:.3f} @ {timestamp}")

            # Memory Integration
            if memory_metrics:
                logger.info("  🧠 MEMORY INTEGRATION:")
                for metric in memory_metrics[:2]:
                    conv_id = metric['conversation_id'][:8]
                    timestamp = self._format_timestamp(metric['timestamp'])
                    logger.info(f"    🧠 {conv_id}... | {metric['metric_name']} = {metric['metric_value']:.3f} @ {timestamp}")

            # Performance & Timing
            if timing_metrics:
                logger.info("  ⚡ TIMING & PERFORMANCE:")
                for metric in timing_metrics[:3]:
                    conv_id = metric['conversation_id'][:8]
                    timestamp = self._format_timestamp(metric['timestamp'])
                    unit = "s" if "time" in metric['metric_name'] or "duration" in metric['metric_name'] else ""
                    logger.info(f"    ⚡ {conv_id}... | {metric['metric_name']} = {metric['metric_value']:.2f}{unit} @ {timestamp}")

            # Metadata Learning (7 Areas)
            if metadata_learning_metrics:
                logger.info("  🔍 METADATA LEARNING (7 AREAS):")
                for metric in metadata_learning_metrics[:3]:
                    conv_id = metric['conversation_id'][:8]
                    timestamp = self._format_timestamp(metric['timestamp'])
                    logger.info(f"    🔍 {conv_id}... | {metric['metric_name']} = {metric['metric_value']:.3f} @ {timestamp}")

            # SQL Agent Discovery
            if sql_agent_metrics:
                logger.info("  🗄️ SQL AGENT DISCOVERY:")
                for metric in sql_agent_metrics[:2]:
                    conv_id = metric['conversation_id'][:8]
                    timestamp = self._format_timestamp(metric['timestamp'])
                    logger.info(f"    🗄️ {conv_id}... | {metric['metric_name']} = {metric['metric_value']:.3f} @ {timestamp}")

            # Efficiency Metrics
            if efficiency_metrics:
                logger.info("  ⚙️ EFFICIENCY:")
                for metric in efficiency_metrics[:2]:
                    conv_id = metric['conversation_id'][:8]
                    timestamp = self._format_timestamp(metric['timestamp'])
                    logger.info(f"    ⚙️ {conv_id}... | {metric['metric_name']} = {metric['metric_value']:.3f} @ {timestamp}")

            # Question Quality
            if question_quality_metrics:
                logger.info("  ❓ QUESTION QUALITY:")
                for metric in question_quality_metrics[:2]:
                    conv_id = metric['conversation_id'][:8]
                    timestamp = self._format_timestamp(metric['timestamp'])
                    logger.info(f"    ❓ {conv_id}... | {metric['metric_name']} = {metric['metric_value']:.3f} @ {timestamp}")

            # Tree Search Planning
            if tree_search_metrics:
                logger.info("  🌳 TREE SEARCH PLANNING:")
                for metric in tree_search_metrics[:1]:
                    conv_id = metric['conversation_id'][:8]
                    timestamp = self._format_timestamp(metric['timestamp'])
                    logger.info(f"    🌳 {conv_id}... | {metric['metric_name']} = {metric['metric_value']:.3f} @ {timestamp}")

            # DSPy Optimization
            if dspy_metrics:
                logger.info("  🔧 DSPY OPTIMIZATION:")
                for metric in dspy_metrics[:1]:
                    conv_id = metric['conversation_id'][:8]
                    timestamp = self._format_timestamp(metric['timestamp'])
                    logger.info(f"    🔧 {conv_id}... | {metric['metric_name']} = {metric['metric_value']:.3f} @ {timestamp}")

            # Hierarchical Learning
            if hierarchical_metrics:
                logger.info("  🏗️ HIERARCHICAL LEARNING:")
                for metric in hierarchical_metrics[:1]:
                    conv_id = metric['conversation_id'][:8]
                    timestamp = self._format_timestamp(metric['timestamp'])
                    logger.info(f"    🏗️ {conv_id}... | {metric['metric_name']} = {metric['metric_value']:.3f} @ {timestamp}")

            # Errors
            if error_metrics:
                logger.info("  ❌ ERRORS:")
                for metric in error_metrics[:2]:
                    conv_id = metric['conversation_id'][:8]
                    timestamp = self._format_timestamp(metric['timestamp'])
                    logger.info(f"    ❌ {conv_id}... | {metric['metric_name']} = {metric['metric_value']:.3f} @ {timestamp}")

            # Summary statistics
            total_categories = sum([
                len(self_discovery_metrics), len(learning_system_metrics), len(reflection_metrics),
                len(memory_metrics), len(timing_metrics), len(metadata_learning_metrics),
                len(hierarchical_metrics), len(tree_search_metrics), len(dspy_metrics),
                len(sql_agent_metrics), len(efficiency_metrics), len(question_quality_metrics),
                len(error_metrics)
            ])

            logger.info(f"  📈 ECOSYSTEM STATS: {len(metrics)} total metrics across {total_categories} categorized metrics from {len(set(m['conversation_id'] for m in metrics))} conversations")

        except Exception as e:
            logger.error(f"Error displaying comprehensive metrics: {e}")

    def _format_timestamp(self, timestamp):
        """Format timestamp for display."""
        try:
            if hasattr(timestamp, 'strftime'):
                return timestamp.strftime('%H:%M:%S')
            elif isinstance(timestamp, str):
                return timestamp[:8]
            else:
                return str(timestamp)[:8]
        except:
            return "??:??:??"

    async def _search_conversations_text(self, query: str, focus_area: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search conversations using text matching (simplified version)."""
        where_clauses = ["LOWER(t.content) LIKE LOWER(@query)"]
        parameters = [bigquery.ScalarQueryParameter("query", "STRING", f"%{query}%")]

        if focus_area:
            where_clauses.append("c.focus_area = @focus_area")
            parameters.append(bigquery.ScalarQueryParameter("focus_area", "STRING", focus_area))

        where_clause = " AND ".join(where_clauses)

        sql_query = f"""
        SELECT
          c.conversation_id,
          c.focus_area,
          t.speaker,
          t.content,
          t.timestamp,
          t.turn_number
        FROM `{self.project_id}.{self.dataset_id}.conversations` c
        JOIN `{self.project_id}.{self.dataset_id}.conversation_turns` t
          ON c.conversation_id = t.conversation_id
        WHERE {where_clause}
        ORDER BY t.timestamp DESC
        LIMIT 10
        """

        job_config = bigquery.QueryJobConfig(query_parameters=parameters)
        job = self.bq_client.query(sql_query, job_config=job_config)

        return [dict(row) for row in job]

    def _calculate_duration(self, start_time) -> int:
        """Calculate duration in seconds from start time."""
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))

        duration = datetime.now(start_time.tzinfo) - start_time
        return int(duration.total_seconds())


async def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Live Conversation Monitoring")
    parser.add_argument("--conversation-id", help="Monitor specific conversation")
    parser.add_argument("--focus-area", help="Monitor conversations with specific focus area")
    parser.add_argument("--live-all", action="store_true", help="Monitor all active conversations")
    parser.add_argument("--search", help="Search conversations for specific text")
    parser.add_argument("--refresh-interval", type=int, default=5, help="Refresh interval in seconds")

    args = parser.parse_args()

    monitor = ConversationMonitor()

    try:
        if args.conversation_id:
            await monitor.monitor_conversation(args.conversation_id, args.refresh_interval)
        elif args.live_all:
            await monitor.monitor_all_active(args.refresh_interval)
        elif args.search:
            await monitor.search_conversations_live(args.search, args.focus_area)
        else:
            logger.info("Please specify --conversation-id, --live-all, or --search")
            parser.print_help()

    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
