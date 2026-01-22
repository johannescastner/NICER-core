#!/usr/bin/env python3
"""
Real Metrics Integration System

This module connects all the real metrics calculation systems to the BigQuery conversation logging.
Instead of injecting fake metrics, this system collects REAL metrics from:

1. Knowledge Discovery Metrics (src/metrics/knowledge_discovery_metrics.py)
2. Self-Discovery Meta-Learning (pro/agents/self_discovery_meta_learning.py)  
3. Real Evaluation Metrics (pro/agents/real_evaluation_metrics.py)
4. Robust Metrics Framework (pro/measurement/robust_metrics.py)

And automatically logs them to BigQuery through ConversationLogger.log_metric()
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from src.nicer_logging.conversation_logger import ConversationLogger
from src.metrics.knowledge_discovery_metrics import (
    SQLAgentPerformanceEvaluator,
    DiscoveredFact,
    SQLAgentQuestion,
    evaluate_sql_agent_metadata_learning,
    evaluate_sql_agent_session
)

logger = logging.getLogger(__name__)

class RealMetricsIntegrator:
    """
    Integrates real metrics from all systems into BigQuery conversation logging.
    
    This class acts as a bridge between the various metrics calculation systems
    and the conversation logging system, ensuring all real metrics are captured.
    """
    
    def __init__(self, conversation_logger: ConversationLogger):
        """Initialize the real metrics integrator."""
        self.conversation_logger = conversation_logger
        self.sql_evaluator = SQLAgentPerformanceEvaluator()
        
        logger.info("Real metrics integrator initialized")
    
    async def log_knowledge_discovery_metrics(self, 
                                            conversation_id: str,
                                            discovered_facts: List[DiscoveredFact],
                                            questions: List[SQLAgentQuestion],
                                            focus_area: str = "general") -> None:
        """Log real knowledge discovery metrics from SQL agent interactions."""
        
        try:
            # Calculate real metrics using the existing evaluator
            performance_metrics = await evaluate_sql_agent_session(
                discovered_facts, questions
            )
            
            # Calculate metadata learning metrics
            metadata_metrics = await evaluate_sql_agent_metadata_learning(
                discovered_facts, questions
            )
            
            # Log SQL Agent Discovery Metrics
            await self.conversation_logger.log_metric(
                conversation_id, "sql_agent", "total_facts_discovered", 
                float(len(discovered_facts)), {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "sql_agent", "facts_per_llm_call",
                performance_metrics.facts_per_llm_call, {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "sql_agent", "knowledge_type_accuracy",
                performance_metrics.knowledge_type_accuracy, {"focus_area": focus_area}
            )
            
            # Log Question Quality Metrics
            await self.conversation_logger.log_metric(
                conversation_id, "question_quality", "excellent_questions",
                float(performance_metrics.excellent_questions), {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "question_quality", "poor_questions",
                float(performance_metrics.poor_questions), {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "question_quality", "question_quality_score",
                performance_metrics.question_quality_score, {"focus_area": focus_area}
            )
            
            # Log Efficiency Metrics (from performance_metrics)
            await self.conversation_logger.log_metric(
                conversation_id, "efficiency", "llm_calls_saved",
                float(performance_metrics.llm_calls_saved_by_good_questions), {"focus_area": focus_area}
            )

            await self.conversation_logger.log_metric(
                conversation_id, "efficiency", "llm_calls_wasted",
                float(performance_metrics.llm_calls_wasted_by_bad_questions), {"focus_area": focus_area}
            )

            await self.conversation_logger.log_metric(
                conversation_id, "efficiency", "net_efficiency_gain",
                float(performance_metrics.net_efficiency_gain), {"focus_area": focus_area}
            )

            await self.conversation_logger.log_metric(
                conversation_id, "efficiency", "efficiency_improvement_ratio",
                performance_metrics.efficiency_improvement_ratio, {"focus_area": focus_area}
            )
            
            # Log Metadata Learning Metrics (7 Areas)
            await self.conversation_logger.log_metric(
                conversation_id, "metadata", "semantic_memory_completion",
                metadata_metrics.semantic_memory_completion, {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "metadata", "human_interaction_optimization",
                metadata_metrics.human_interaction_optimization, {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "metadata", "data_limitation_recognition",
                metadata_metrics.data_limitation_recognition, {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "metadata", "contextual_threshold_learning",
                metadata_metrics.contextual_threshold_learning, {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "metadata", "self_discovery_meta_learning",
                metadata_metrics.self_discovery_meta_learning, {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "metadata", "ego_network_pattern_learning",
                metadata_metrics.ego_network_pattern_learning, {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "metadata", "overall_metadata_coverage",
                metadata_metrics.overall_metadata_coverage, {"focus_area": focus_area}
            )
            
            # Log Coverage Metrics
            await self.conversation_logger.log_metric(
                conversation_id, "coverage", "comprehensive_coverage",
                metadata_metrics.comprehensive_coverage, {"focus_area": focus_area}
            )
            
            logger.info(f"✅ Logged {len(discovered_facts)} facts and {len(questions)} questions metrics to BigQuery")
            
        except Exception as e:
            logger.error(f"Failed to log knowledge discovery metrics: {e}")
            await self.conversation_logger.log_error(
                conversation_id, "metrics_integration_error", str(e),
                {"system": "knowledge_discovery", "focus_area": focus_area}
            )
    
    async def log_self_discovery_metrics(self,
                                       conversation_id: str,
                                       optimization_cycles: int,
                                       tool_selections: Dict[str, int],
                                       strategy_adaptations: int,
                                       focus_area: str = "general") -> None:
        """Log real self-discovery orchestrator metrics."""
        
        try:
            # Calculate real self-discovery metrics
            total_selections = sum(tool_selections.values())
            correct_selections = tool_selections.get("correct", 0)
            tool_accuracy = correct_selections / max(total_selections, 1)
            
            # Log Self-Discovery Orchestrator Metrics
            await self.conversation_logger.log_metric(
                conversation_id, "self_discovery", "optimization_cycles_triggered",
                float(optimization_cycles), {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "self_discovery", "tool_selection_accuracy",
                tool_accuracy, {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "self_discovery", "strategy_adaptations",
                float(strategy_adaptations), {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "self_discovery", "correct_tool_selections",
                float(correct_selections), {"focus_area": focus_area}
            )
            
            logger.info(f"✅ Logged self-discovery metrics: {optimization_cycles} cycles, {tool_accuracy:.3f} accuracy")
            
        except Exception as e:
            logger.error(f"Failed to log self-discovery metrics: {e}")
            await self.conversation_logger.log_error(
                conversation_id, "metrics_integration_error", str(e),
                {"system": "self_discovery", "focus_area": focus_area}
            )
    
    async def log_learning_system_metrics(self,
                                        conversation_id: str,
                                        facts_learned: int,
                                        knowledge_applications: int,
                                        successful_applications: int,
                                        retention_rate: float,
                                        focus_area: str = "general") -> None:
        """Log real learning system performance metrics."""
        
        try:
            # Calculate real learning metrics
            application_success_rate = successful_applications / max(knowledge_applications, 1)
            
            # Log Learning System Metrics
            await self.conversation_logger.log_metric(
                conversation_id, "learning", "facts_learned",
                float(facts_learned), {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "learning", "knowledge_application_attempts",
                float(knowledge_applications), {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "learning", "successful_applications",
                float(successful_applications), {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "learning", "application_success_rate",
                application_success_rate, {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "learning", "retention_rate",
                retention_rate, {"focus_area": focus_area}
            )
            
            logger.info(f"✅ Logged learning metrics: {facts_learned} facts, {application_success_rate:.3f} success rate")
            
        except Exception as e:
            logger.error(f"Failed to log learning system metrics: {e}")
            await self.conversation_logger.log_error(
                conversation_id, "metrics_integration_error", str(e),
                {"system": "learning_system", "focus_area": focus_area}
            )
    
    async def log_memory_integration_metrics(self,
                                           conversation_id: str,
                                           memory_queries: int,
                                           relevant_retrievals: int,
                                           episodic_memories: int,
                                           consolidated_memories: int,
                                           focus_area: str = "general") -> None:
        """Log real memory integration metrics."""
        
        try:
            # Calculate real memory metrics
            retrieval_accuracy = relevant_retrievals / max(memory_queries, 1)
            consolidation_rate = consolidated_memories / max(episodic_memories, 1)
            
            # Log Memory Integration Metrics
            await self.conversation_logger.log_metric(
                conversation_id, "memory", "memory_queries",
                float(memory_queries), {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "memory", "relevant_retrievals",
                float(relevant_retrievals), {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "memory", "retrieval_accuracy",
                retrieval_accuracy, {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "memory", "episodic_memories_created",
                float(episodic_memories), {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "memory", "memories_consolidated_to_semantic",
                float(consolidated_memories), {"focus_area": focus_area}
            )
            
            await self.conversation_logger.log_metric(
                conversation_id, "memory", "consolidation_rate",
                consolidation_rate, {"focus_area": focus_area}
            )
            
            logger.info(f"✅ Logged memory metrics: {retrieval_accuracy:.3f} retrieval accuracy, {consolidation_rate:.3f} consolidation rate")
            
        except Exception as e:
            logger.error(f"Failed to log memory integration metrics: {e}")
            await self.conversation_logger.log_error(
                conversation_id, "metrics_integration_error", str(e),
                {"system": "memory_integration", "focus_area": focus_area}
            )

    async def log_custom_metric(self,
                              conversation_id: str,
                              metric_name: str,
                              metric_value: float,
                              metadata: Optional[Dict[str, Any]] = None,
                              metric_type: str = "custom") -> None:
        """Log a custom metric to BigQuery."""
        try:
            await self.conversation_logger.log_metric(
                conversation_id, metric_type, metric_name, metric_value, metadata or {}
            )
            logger.debug(f"✅ Logged custom metric: {metric_name} = {metric_value}")

        except Exception as e:
            logger.error(f"Failed to log custom metric {metric_name}: {e}")
            await self.conversation_logger.log_error(
                conversation_id, "custom_metric_error", str(e),
                {"metric_name": metric_name, "metric_value": metric_value}
            )

    async def log_completeness_metrics(self,
                                     conversation_id: str,
                                     completeness_metrics: Dict[str, float],
                                     focus_area: str = "general") -> None:
        """Log knowledge completeness metrics to BigQuery."""
        try:
            # Log discoverable knowledge completeness
            await self.conversation_logger.log_metric(
                conversation_id, "completeness", "discoverable_knowledge_completeness",
                completeness_metrics.get("discoverable_completeness", 0.0),
                {"focus_area": focus_area, "target": "100_percent_discoverable"}
            )

            # Log job-relevant undiscoverable knowledge completeness
            await self.conversation_logger.log_metric(
                conversation_id, "completeness", "job_relevant_undiscoverable_completeness",
                completeness_metrics.get("undiscoverable_completeness", 0.0),
                {"focus_area": focus_area, "target": "100_percent_job_relevant_undiscoverable"}
            )

            # Log total knowledge completeness
            await self.conversation_logger.log_metric(
                conversation_id, "completeness", "total_knowledge_completeness",
                completeness_metrics.get("total_completeness", 0.0),
                {"focus_area": focus_area, "target": "100_percent_all_knowledge"}
            )

            # Log detailed counts
            await self.conversation_logger.log_metric(
                conversation_id, "completeness", "discoverable_facts_found",
                float(completeness_metrics.get("discoverable_found", 0)),
                {"focus_area": focus_area, "total_available": completeness_metrics.get("total_discoverable", 0)}
            )

            await self.conversation_logger.log_metric(
                conversation_id, "completeness", "undiscoverable_facts_found",
                float(completeness_metrics.get("undiscoverable_found", 0)),
                {"focus_area": focus_area, "total_available": completeness_metrics.get("total_undiscoverable", 0)}
            )

            logger.info(f"✅ Logged completeness metrics: {completeness_metrics.get('total_completeness', 0.0):.3f} total completeness")

        except Exception as e:
            logger.error(f"Failed to log completeness metrics: {e}")
            await self.conversation_logger.log_error(
                conversation_id, "completeness_metrics_error", str(e),
                {"system": "knowledge_completeness", "focus_area": focus_area}
            )
