"""
Custom Metrics for SQL Agent Knowledge Discovery Performance

Judges the SQL agent's performance in:
1. Discovering knowledge efficiently (facts per LLM call)
2. Asking good questions to humans (targeting undiscoverable + essential knowledge)
3. Avoiding bad questions (asking about discoverable things)
4. Achieving coverage across 7 metadata understanding areas

Uses IntellAgent's verified knowledge base as ground truth.
"""

import asyncio
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class KnowledgeType(Enum):
    """Classification of knowledge discoverability."""
    DISCOVERABLE = "discoverable"      # Can be found through data exploration
    UNDISCOVERABLE = "undiscoverable"  # Requires human domain expertise
    UNKNOWN = "unknown"                # Not yet classified


class QuestionQuality(Enum):
    """Quality rating for SQL agent's questions to humans."""
    EXCELLENT = "excellent"    # Targets undiscoverable + essential knowledge
    GOOD = "good"             # Targets undiscoverable OR saves many LLM calls
    POOR = "poor"             # Asks about discoverable things
    TERRIBLE = "terrible"     # Asks about easily discoverable + wastes time


@dataclass
class SQLAgentQuestion:
    """A question the SQL agent asked to humans via Slack."""
    question_text: str
    human_response: str
    knowledge_gained: List[str]
    estimated_llm_calls_saved: int
    targets_undiscoverable: bool
    is_essential_for_job: bool
    could_be_discovered_normally: bool
    quality_rating: QuestionQuality


@dataclass
class DiscoverySession:
    """A complete knowledge discovery session by the SQL agent."""
    session_id: str
    total_llm_calls: int
    facts_discovered: List[Dict]
    questions_to_humans: List[SQLAgentQuestion]
    coverage_achieved: Dict[str, float]  # Coverage per metadata area
    session_duration_minutes: int


@dataclass
class SQLAgentPerformanceMetrics:
    """Comprehensive performance metrics for the SQL agent."""
    
    # Core Discovery Efficiency
    total_facts_discovered: int
    verified_facts_matched: int        # Facts matching IntellAgent's knowledge
    new_valid_facts: int              # New facts that are verified as correct
    total_llm_calls: int
    facts_per_llm_call: float
    verified_facts_per_llm_call: float
    
    # Knowledge Discovery Quality
    discoverable_facts_found: int      # Should be high (good exploration)
    undiscoverable_facts_found: int    # Should require human help
    knowledge_type_accuracy: float     # How well agent distinguishes types
    
    # Human Interaction Performance
    total_questions_to_humans: int
    excellent_questions: int           # Target undiscoverable + essential
    good_questions: int               # Target undiscoverable OR save LLM calls
    poor_questions: int               # Ask about discoverable things
    terrible_questions: int           # Waste time on obvious things
    question_quality_score: float     # Weighted average
    
    # LLM Call Efficiency
    llm_calls_saved_by_good_questions: int
    llm_calls_wasted_by_bad_questions: int
    net_efficiency_gain: int
    efficiency_improvement_ratio: float
    
    # Coverage Metrics (7 Areas)
    semantic_memory_coverage: float
    human_interaction_coverage: float
    data_limitation_coverage: float
    contextual_threshold_coverage: float
    meta_learning_coverage: float
    ego_network_coverage: float
    comprehensive_coverage: float
    overall_coverage_score: float


class SQLAgentPerformanceEvaluator:
    """Evaluates SQL agent performance against IntellAgent's knowledge base."""
    
    def __init__(self):
        self.ground_truth_knowledge: List[Dict] = []
        self.discoverable_patterns: List[str] = []
        self.undiscoverable_patterns: List[str] = []
        self.essential_knowledge_areas: List[str] = []
        
    async def initialize_ground_truth(self) -> None:
        """Load verified knowledge from IntellAgent as ground truth."""
        try:
            from src.graphs.memory import get_memory_tools
            
            namespace_templates = {
                'intellagent': {
                    'langgraph_auth_user_id': 'intellagent',
                    'thread_id': 'domain_expertise.business_context'
                }
            }
            
            memory_tools = await get_memory_tools(namespace_templates)
            search_tool = None
            for tool in memory_tools:
                if tool.name == 'search_intellagent_memory':
                    search_tool = tool
                    break
            
            if not search_tool:
                logger.error("IntellAgent search tool not found")
                return
            
            # Load all verified knowledge
            all_results = search_tool.invoke({'query': '', 'limit': 100})
            if all_results:
                for result in all_results:
                    if isinstance(result, dict) and 'value' in result:
                        value = result['value']
                        if isinstance(value, dict) and 'fact' in value:
                            fact_content = value['fact']
                            if isinstance(fact_content, dict):
                                self.ground_truth_knowledge.append(fact_content)
            
            # Define discoverable vs undiscoverable patterns
            self._define_knowledge_patterns()
            
            logger.info(f"Loaded {len(self.ground_truth_knowledge)} ground truth facts")
            
        except Exception as e:
            logger.error(f"Failed to load ground truth: {e}")
    
    def _define_knowledge_patterns(self) -> None:
        """Define patterns for discoverable vs undiscoverable knowledge."""
        
        # DISCOVERABLE: Can be found through data exploration
        self.discoverable_patterns = [
            'table has columns', 'column data type', 'row count', 'schema structure',
            'primary key', 'foreign key', 'null values', 'data samples',
            'table exists', 'column exists', 'index information'
        ]
        
        # UNDISCOVERABLE: Requires human domain expertise  
        self.undiscoverable_patterns = [
            'business context', 'entity meaning', 'semantic relationship',
            'company director', 'professional network', 'privacy constraint',
            'data quality pattern', 'plugin usage', 'email privacy limitation',
            'entity resolution strategy', 'cross-dataset linkage'
        ]
        
        # ESSENTIAL: Critical for metadata understanding job
        self.essential_knowledge_areas = [
            'entity resolution', 'business context', 'data structure',
            'cross-dataset relationships', 'data limitations'
        ]
    
    def classify_knowledge_discoverability(self, fact_content: str) -> KnowledgeType:
        """Classify if knowledge is discoverable or undiscoverable."""
        content_lower = fact_content.lower()
        
        discoverable_score = sum(1 for pattern in self.discoverable_patterns 
                               if pattern in content_lower)
        undiscoverable_score = sum(1 for pattern in self.undiscoverable_patterns 
                                 if pattern in content_lower)
        
        if undiscoverable_score > discoverable_score:
            return KnowledgeType.UNDISCOVERABLE
        elif discoverable_score > 0:
            return KnowledgeType.DISCOVERABLE
        else:
            return KnowledgeType.UNKNOWN
    
    def evaluate_question_quality(self, question: SQLAgentQuestion) -> QuestionQuality:
        """Judge the quality of a question the SQL agent asked to humans."""
        
        # EXCELLENT: Targets undiscoverable + essential knowledge
        if (question.targets_undiscoverable and 
            question.is_essential_for_job and 
            not question.could_be_discovered_normally):
            return QuestionQuality.EXCELLENT
        
        # GOOD: Targets undiscoverable OR saves significant LLM calls
        if (question.targets_undiscoverable or 
            question.estimated_llm_calls_saved >= 5):
            return QuestionQuality.GOOD
        
        # TERRIBLE: Asks about easily discoverable things
        if (question.could_be_discovered_normally and 
            question.estimated_llm_calls_saved < 2):
            return QuestionQuality.TERRIBLE
        
        # POOR: Everything else
        return QuestionQuality.POOR
    
    def calculate_efficiency_metrics(self, session: DiscoverySession) -> Dict[str, float]:
        """Calculate LLM call efficiency metrics."""
        
        # Base efficiency
        facts_per_call = len(session.facts_discovered) / max(session.total_llm_calls, 1)
        
        # Question quality impact
        llm_calls_saved = sum(q.estimated_llm_calls_saved 
                             for q in session.questions_to_humans 
                             if q.quality_rating in [QuestionQuality.EXCELLENT, QuestionQuality.GOOD])
        
        llm_calls_wasted = sum(2 for q in session.questions_to_humans 
                              if q.quality_rating == QuestionQuality.TERRIBLE)
        
        net_efficiency_gain = llm_calls_saved - llm_calls_wasted
        
        # Adjusted efficiency accounting for human interaction quality
        effective_llm_calls = session.total_llm_calls - net_efficiency_gain
        adjusted_efficiency = len(session.facts_discovered) / max(effective_llm_calls, 1)
        
        return {
            'base_facts_per_llm_call': facts_per_call,
            'llm_calls_saved': llm_calls_saved,
            'llm_calls_wasted': llm_calls_wasted,
            'net_efficiency_gain': net_efficiency_gain,
            'adjusted_facts_per_llm_call': adjusted_efficiency,
            'efficiency_improvement_ratio': adjusted_efficiency / facts_per_call if facts_per_call > 0 else 1.0
        }
    
    def evaluate_coverage_completeness(self, session: DiscoverySession) -> Dict[str, float]:
        """Evaluate coverage across the 7 metadata understanding areas."""
        
        # Count facts discovered in each area
        area_counts = {
            'semantic_memory': 0,
            'human_interaction': 0, 
            'data_limitation': 0,
            'contextual_threshold': 0,
            'meta_learning': 0,
            'ego_network': 0,
            'comprehensive': 0
        }
        
        # Count ground truth facts in each area
        ground_truth_counts = {area: 0 for area in area_counts.keys()}
        
        for fact in self.ground_truth_knowledge:
            category = fact.get('category', '').lower()
            content = fact.get('content', '').lower()
            
            if any(term in category or term in content 
                   for term in ['data_structure', 'semantic', 'meaning']):
                ground_truth_counts['semantic_memory'] += 1
            elif any(term in category or term in content 
                     for term in ['entity_resolution', 'business_context']):
                ground_truth_counts['human_interaction'] += 1
            elif any(term in category or term in content 
                     for term in ['data_limitation', 'privacy', 'constraint']):
                ground_truth_counts['data_limitation'] += 1
            # ... continue for other areas
        
        # Calculate coverage percentages
        coverage = {}
        for area in area_counts.keys():
            if ground_truth_counts[area] > 0:
                coverage[f'{area}_coverage'] = area_counts[area] / ground_truth_counts[area]
            else:
                coverage[f'{area}_coverage'] = 1.0  # No ground truth = perfect coverage
        
        # Overall coverage score
        coverage['overall_coverage'] = sum(coverage.values()) / len(coverage)
        
        return coverage
