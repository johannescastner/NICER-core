"""
Custom Knowledge Discovery Metrics for SQL Agent

These metrics use IntellAgent's verified knowledge base as ground truth to measure:
1. Knowledge discovery efficiency vs LLM call count
2. Quality of human questions (targeting undiscoverable knowledge)
3. Discovery completeness across the 7 metadata understanding areas

Based on verified domain knowledge from IntellAgent's memory system.
"""

import asyncio
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class KnowledgeType(Enum):
    """Types of knowledge for discovery classification."""
    DISCOVERABLE = "discoverable"  # Can be found through data exploration
    UNDISCOVERABLE = "undiscoverable"  # Requires human domain expertise
    UNKNOWN = "unknown"  # Not yet classified


class MetadataArea(Enum):
    """The 7 areas of metadata understanding."""
    SEMANTIC_MEMORY_COMPLETION = "semantic_memory_completion"
    HUMAN_INTERACTION_OPTIMIZATION = "human_interaction_optimization"
    DATA_LIMITATION_RECOGNITION = "data_limitation_recognition"
    CONTEXTUAL_THRESHOLD_LEARNING = "contextual_threshold_learning"
    SELF_DISCOVERY_META_LEARNING = "self_discovery_meta_learning"
    EGO_NETWORK_PATTERN_LEARNING = "ego_network_pattern_learning"
    COMPREHENSIVE_COVERAGE = "comprehensive_coverage"


@dataclass
class DiscoveredFact:
    """A fact discovered by the SQL agent."""
    content: str
    category: str
    discovery_method: str  # "exploration", "human_question", "inference"
    llm_calls_to_discover: int
    confidence: float
    area: MetadataArea
    knowledge_type: KnowledgeType


@dataclass
class SQLAgentQuestion:
    """A question the SQL agent asked to humans via Slack - we judge the agent's performance."""
    question_text: str
    human_response: str
    knowledge_gained: List[str]
    estimated_llm_calls_saved: int  # LLM calls this question saved vs normal exploration
    targets_undiscoverable: bool    # Good: asks about business context, entity meanings
    is_essential_for_job: bool      # Good: essential for metadata understanding
    could_be_discovered_normally: bool  # Bad: asks about schema, row counts, etc.
    quality_score: float            # 0-1 score judging the agent's question quality


@dataclass
class SQLAgentMetadataLearningMetrics:
    """Metrics for SQL agent's ability to learn metadata understanding through strategic questioning."""

    # CORE METADATA UNDERSTANDING (what the agent should learn)
    total_metadata_knowledge_uncovered: int    # Essential metadata facts learned from IntellAgent
    column_meanings_discovered: int            # "person field = ego nodes", "url field = connections"
    table_relationships_mapped: int            # Cross-dataset entity resolution strategies
    business_context_learned: int              # Company structure, domain knowledge
    data_limitations_identified: int           # Privacy constraints, missing data reasons

    # STRATEGIC QUESTIONING PERFORMANCE (how well agent asks)
    total_questions_to_intellagent: int
    excellent_metadata_questions: int          # Unlock essential column/table meanings
    good_context_questions: int               # Discover business relationships
    poor_trivial_questions: int               # Ask about things agent could discover
    question_effectiveness_score: float        # Knowledge gained per question asked

    # COVERAGE ACROSS 7 METADATA AREAS (using IntellAgent's knowledge as target)
    semantic_memory_completion: float          # Table/column meanings, dataset purposes
    human_interaction_optimization: float      # Business context, entity resolution
    data_limitation_recognition: float         # Privacy constraints, data quality issues
    contextual_threshold_learning: float       # When to ask vs explore
    self_discovery_meta_learning: float        # Learning about learning progress
    ego_network_pattern_learning: float        # Network relationships, connection patterns
    comprehensive_coverage: float              # Complete integrated understanding
    overall_metadata_coverage: float           # Average across all 7 areas

    # EFFICIENCY METRICS (knowledge per effort)
    metadata_knowledge_per_question: float     # Essential knowledge gained per question
    llm_calls_for_exploration: int            # Calls used for data exploration
    llm_calls_for_questioning: int            # Calls used for question formulation
    total_llm_calls: int                       # Total calls across all activities
    knowledge_efficiency_ratio: float          # Knowledge gained per total LLM call

    # LEARNING PROGRESSION (improvement over time)
    questions_targeting_undiscoverable: int    # Questions about domain knowledge
    questions_about_discoverable_things: int   # Questions about schema/structure
    learning_focus_accuracy: float             # Ratio of good vs poor question targeting


class SQLAgentPerformanceEvaluator:
    """Evaluates SQL agent's knowledge discovery performance against IntellAgent's ground truth."""

    def __init__(self):
        self.ground_truth_knowledge: List[Dict] = []
        self.verified_facts: List[Dict] = []  # Alias for compatibility
        self.discoverable_patterns: List[str] = []
        self.undiscoverable_patterns: List[str] = []
        self.essential_knowledge_areas: List[str] = []
        
    async def load_intellagent_knowledge(self) -> None:
        """Load verified knowledge from IntellAgent's memory."""
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
            
            # Load all verified knowledge across categories
            categories = [
                'data_structure', 'entity_resolution', 'business_context',
                'data_limitation', 'data_scale', 'discovery_strategy_framework'
            ]
            
            for category in categories:
                results = search_tool.invoke({'query': category, 'limit': 20})
                if results:
                    for result in results:
                        if isinstance(result, dict) and 'value' in result:
                            value = result['value']
                            if isinstance(value, dict) and 'fact' in value:
                                fact_content = value['fact']
                                if isinstance(fact_content, dict):
                                    self.verified_facts.append(fact_content)
                                    self.ground_truth_knowledge.append(fact_content)

            logger.info(f"Loaded {len(self.verified_facts)} verified facts from IntellAgent")
            
        except Exception as e:
            logger.error(f"Failed to load IntellAgent knowledge: {e}")
    
    async def classify_knowledge_type(self, fact_content: str) -> KnowledgeType:
        """
        Classify whether a fact is discoverable or undiscoverable using IntellAgent's verified knowledge.

        Uses semantic similarity to find the most similar fact in IntellAgent's knowledge base,
        then uses the established category mappings to determine discoverability.
        """
        try:
            from src.graphs.memory import get_memory_tools

            # Get IntellAgent's search tool
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
                logger.warning("IntellAgent search tool not found, using fallback")
                return self._fallback_classify_knowledge_type(fact_content)

            # Search for the most similar fact in IntellAgent's knowledge
            similar_facts = search_tool.invoke({
                'query': fact_content,
                'limit': 5
            })

            if not similar_facts:
                logger.warning("No similar facts found in IntellAgent's knowledge")
                return self._fallback_classify_knowledge_type(fact_content)

            # Use established category mappings from IntellAgent's knowledge
            for result in similar_facts:
                if isinstance(result, dict) and 'value' in result:
                    value = result['value']
                    if isinstance(value, dict) and 'fact' in value:
                        fact_data = value['fact']
                        if isinstance(fact_data, dict):
                            category = fact_data.get('category', '').lower()

                            # UNDISCOVERABLE categories (require human domain expertise)
                            if any(term in category for term in [
                                'business_context',      # Company directors, business meanings
                                'entity_resolution',     # Cross-dataset linkage strategies
                                'data_limitation',       # Privacy constraints, missing data reasons
                                'data_scale',           # Business insights about scale patterns
                                'ego_network_learning'   # Network relationship semantics
                            ]):
                                return KnowledgeType.UNDISCOVERABLE

                            # DISCOVERABLE categories (can be found through exploration)
                            elif any(term in category for term in [
                                'data_structure',        # Table schemas, column types
                                'metadata',             # Technical metadata, row counts
                                'datasets',             # Available datasets and tables
                                'code',                 # Technical implementation details
                                'dataset_availability'   # What datasets exist
                            ]):
                                return KnowledgeType.DISCOVERABLE

            # If no clear category match, use fallback
            return self._fallback_classify_knowledge_type(fact_content)

        except Exception as e:
            logger.error(f"Error in knowledge classification: {e}")
            return self._fallback_classify_knowledge_type(fact_content)

    def classify_knowledge_type_sync(self, fact_content: str) -> KnowledgeType:
        """
        Synchronous version of classify_knowledge_type for testing and non-async contexts.
        Uses fallback classification based on verified domain knowledge patterns.
        """
        return self._fallback_classify_knowledge_type(fact_content)

    def _fallback_classify_knowledge_type(self, fact_content: str) -> KnowledgeType:
        """Fallback classification using verified domain knowledge from IntellAgent."""
        content_lower = fact_content.lower()

        # UNDISCOVERABLE: Based on IntellAgent's verified knowledge
        undiscoverable_indicators = [
            # Business context (verified in IntellAgent)
            'johannes castner', 'david cuff', 'company director', 'towards people',
            'network owner', 'business context',

            # Ego network concepts (verified in IntellAgent)
            'ego node', 'ego network', 'network owner', 'professional relationship',

            # Entity resolution strategies (verified in IntellAgent)
            'entity resolution', 'cross-dataset linkage', 'business key',

            # Data limitations (verified in IntellAgent)
            'privacy constraint', 'email privacy', 'data limitation',

            # Semantic meanings (verified in IntellAgent)
            'person field contains ego', 'url field contains connection',
            'link table representing', 'professional network'
        ]

        # DISCOVERABLE: Based on IntellAgent's verified knowledge
        discoverable_indicators = [
            # Table structure (can be found through exploration)
            'table structure', 'column names', 'data type', 'row count',
            'schema', 'primary key', 'foreign key', 'null values',
            'table has', 'columns:', 'column', 'rows',

            # Technical metadata (can be queried)
            'table has columns', 'string type', 'record type',
            'dataset availability', 'table exists'
        ]

        # Score based on verified patterns
        undiscoverable_score = sum(1 for indicator in undiscoverable_indicators
                                 if indicator in content_lower)
        discoverable_score = sum(1 for indicator in discoverable_indicators
                               if indicator in content_lower)

        if undiscoverable_score > discoverable_score:
            return KnowledgeType.UNDISCOVERABLE
        elif discoverable_score > 0:
            return KnowledgeType.DISCOVERABLE
        else:
            return KnowledgeType.UNKNOWN

    async def classify_metadata_area(self, fact_content: str, category: str) -> MetadataArea:
        """
        Classify which of the 7 metadata areas this fact belongs to using IntellAgent's knowledge.

        Uses semantic similarity and category analysis from IntellAgent's verified facts.
        """
        try:
            from src.graphs.memory import get_memory_tools

            # Get IntellAgent's search tool
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
                return self._fallback_classify_metadata_area(fact_content, category)

            # Search for similar facts to determine area
            similar_facts = search_tool.invoke({
                'query': fact_content,
                'limit': 5
            })

            # Analyze categories and content to determine area
            category_lower = category.lower()
            content_lower = fact_content.lower()

            # 1. Semantic Memory Completion - data structure, table meanings
            if (any(term in category_lower for term in ['data_structure', 'semantic', 'metadata']) or
                any(term in content_lower for term in ['table structure', 'column', 'schema', 'data type'])):
                return MetadataArea.SEMANTIC_MEMORY_COMPLETION

            # 2. Human Interaction Optimization - business context, entity resolution
            elif (any(term in category_lower for term in ['business_context', 'entity_resolution']) or
                  any(term in content_lower for term in ['johannes castner', 'david cuff', 'company director', 'business'])):
                return MetadataArea.HUMAN_INTERACTION_OPTIMIZATION

            # 3. Data Limitation Recognition - privacy, constraints, missing data
            elif (any(term in category_lower for term in ['data_limitation', 'privacy', 'constraint']) or
                  any(term in content_lower for term in ['privacy', 'missing', 'limitation', 'constraint'])):
                return MetadataArea.DATA_LIMITATION_RECOGNITION

            # 4. Contextual Threshold Learning - discovery strategy, when to ask
            elif (any(term in category_lower for term in ['discovery_strategy', 'threshold', 'contextual']) or
                  any(term in content_lower for term in ['discovery', 'strategy', 'threshold', 'when to'])):
                return MetadataArea.CONTEXTUAL_THRESHOLD_LEARNING

            # 5. Self-Discovery Meta-Learning - learning about learning
            elif (any(term in category_lower for term in ['meta_learning', 'self_discovery']) or
                  any(term in content_lower for term in ['meta', 'learning', 'discovery', 'reflection'])):
                return MetadataArea.SELF_DISCOVERY_META_LEARNING

            # 6. Ego Network Pattern Learning - connections, networks, relationships
            elif (any(term in category_lower for term in ['ego_network', 'network', 'connection']) or
                  any(term in content_lower for term in ['network', 'connection', 'ego', 'linkedin', 'relationship'])):
                return MetadataArea.EGO_NETWORK_PATTERN_LEARNING

            # 7. Comprehensive Coverage - everything else
            else:
                return MetadataArea.COMPREHENSIVE_COVERAGE

        except Exception as e:
            logger.error(f"Error in metadata area classification: {e}")
            return self._fallback_classify_metadata_area(fact_content, category)

    def _fallback_classify_metadata_area(self, fact_content: str, category: str) -> MetadataArea:
        """Fallback classification for metadata areas."""
        content_lower = fact_content.lower()
        category_lower = category.lower()

        # Simple heuristics based on our domain knowledge
        if any(term in content_lower for term in ['table', 'column', 'schema', 'structure']):
            return MetadataArea.SEMANTIC_MEMORY_COMPLETION
        elif any(term in content_lower for term in ['johannes', 'david', 'business', 'company']):
            return MetadataArea.HUMAN_INTERACTION_OPTIMIZATION
        elif any(term in content_lower for term in ['network', 'connection', 'linkedin']):
            return MetadataArea.EGO_NETWORK_PATTERN_LEARNING
        elif any(term in content_lower for term in ['privacy', 'limitation', 'missing']):
            return MetadataArea.DATA_LIMITATION_RECOGNITION
        else:
            return MetadataArea.COMPREHENSIVE_COVERAGE
    
    def calculate_discovery_efficiency(self, discovered_facts: List[DiscoveredFact]) -> float:
        """Calculate weighted discovery efficiency score."""
        
        if not discovered_facts:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for fact in discovered_facts:
            # Weight by importance and verification status
            base_weight = 1.0
            if fact.confidence > 0.9:
                base_weight *= 1.5  # High confidence bonus
            if fact.knowledge_type == KnowledgeType.UNDISCOVERABLE:
                base_weight *= 2.0  # Undiscoverable knowledge is more valuable
            
            # Efficiency = value / cost
            efficiency = base_weight / max(fact.llm_calls_to_discover, 1)
            
            total_score += efficiency * base_weight
            total_weight += base_weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def evaluate_sql_agent_question_quality(self, question: SQLAgentQuestion) -> float:
        """Judge the quality of a question the SQL agent asked to humans."""

        quality_score = 0.0

        # EXCELLENT: Targets undiscoverable + essential knowledge
        if (question.targets_undiscoverable and
            question.is_essential_for_job and
            not question.could_be_discovered_normally):
            quality_score = 1.0

        # GOOD: Targets undiscoverable OR saves significant LLM calls
        elif (question.targets_undiscoverable or
              question.estimated_llm_calls_saved >= 5):
            quality_score = 0.7

        # POOR: Asks about discoverable things but saves some calls
        elif (question.could_be_discovered_normally and
              question.estimated_llm_calls_saved >= 2):
            quality_score = 0.3

        # TERRIBLE: Asks about easily discoverable things, wastes time
        elif (question.could_be_discovered_normally and
              question.estimated_llm_calls_saved < 2):
            quality_score = 0.0

        # Bonus for getting useful response
        if question.human_response and len(question.knowledge_gained) > 0:
            quality_score += 0.1

        return min(1.0, quality_score)

    def calculate_efficiency_metrics(self,
                                   discovered_facts: List[DiscoveredFact],
                                   questions: List[SQLAgentQuestion]) -> Dict[str, float]:
        """Calculate comprehensive efficiency metrics for the SQL agent."""

        if not discovered_facts:
            return {"facts_per_llm_call": 0.0, "efficiency_score": 0.0}

        total_llm_calls = sum(fact.llm_calls_to_discover for fact in discovered_facts)

        # Base efficiency
        facts_per_call = len(discovered_facts) / max(total_llm_calls, 1)

        # Question quality impact
        llm_calls_saved = sum(q.estimated_llm_calls_saved for q in questions
                             if self.evaluate_sql_agent_question_quality(q) >= 0.7)

        llm_calls_wasted = sum(2 for q in questions
                              if self.evaluate_sql_agent_question_quality(q) <= 0.3)

        net_efficiency_gain = llm_calls_saved - llm_calls_wasted

        # Adjusted efficiency
        effective_calls = max(total_llm_calls - net_efficiency_gain, 1)
        adjusted_efficiency = len(discovered_facts) / effective_calls

        return {
            "facts_per_llm_call": facts_per_call,
            "llm_calls_saved": llm_calls_saved,
            "llm_calls_wasted": llm_calls_wasted,
            "net_efficiency_gain": net_efficiency_gain,
            "adjusted_facts_per_llm_call": adjusted_efficiency,
            "efficiency_improvement_ratio": adjusted_efficiency / facts_per_call if facts_per_call > 0 else 1.0
        }

    def generate_langsmith_metrics(self,
                                 discovered_facts: List[DiscoveredFact],
                                 questions: List[SQLAgentQuestion]) -> Dict[str, float]:
        """Generate metrics for LangSmith tracking and optimization."""

        efficiency_metrics = self.calculate_efficiency_metrics(discovered_facts, questions)

        # Question quality distribution
        question_scores = [self.evaluate_sql_agent_question_quality(q) for q in questions]
        avg_question_quality = sum(question_scores) / len(question_scores) if question_scores else 0.0

        # Knowledge type accuracy
        correct_classifications = 0
        total_classifications = 0
        for fact in discovered_facts:
            if fact.knowledge_type != KnowledgeType.UNKNOWN:
                total_classifications += 1
                # Check against ground truth patterns
                expected_type = self._fallback_classify_knowledge_type(fact.content)
                if fact.knowledge_type == expected_type:
                    correct_classifications += 1

        knowledge_accuracy = correct_classifications / max(total_classifications, 1)

        # Coverage across 7 areas
        area_coverage = self._calculate_area_coverage(discovered_facts)

        return {
            # Core metrics for LangSmith
            "facts_per_llm_call": efficiency_metrics["facts_per_llm_call"],
            "efficiency_improvement_ratio": efficiency_metrics["efficiency_improvement_ratio"],
            "question_quality_score": avg_question_quality,
            "knowledge_type_accuracy": knowledge_accuracy,
            "overall_coverage": area_coverage["overall_coverage"],

            # Detailed metrics for analysis
            "total_facts_discovered": len(discovered_facts),
            "total_questions_asked": len(questions),
            "llm_calls_saved": efficiency_metrics["llm_calls_saved"],
            "llm_calls_wasted": efficiency_metrics["llm_calls_wasted"],

            # 7 area coverage
            "semantic_memory_coverage": area_coverage.get("semantic_memory", 0.0),
            "human_interaction_coverage": area_coverage.get("human_interaction", 0.0),
            "data_limitation_coverage": area_coverage.get("data_limitation", 0.0),
            "contextual_threshold_coverage": area_coverage.get("contextual_threshold", 0.0),
            "meta_learning_coverage": area_coverage.get("meta_learning", 0.0),
            "ego_network_coverage": area_coverage.get("ego_network", 0.0),
            "comprehensive_coverage": area_coverage.get("comprehensive", 0.0)
        }

    def generate_dspy_optimization_targets(self,
                                         current_metrics: Dict[str, float]) -> Dict[str, float]:
        """Generate optimization targets for DSPy to improve agent performance."""

        return {
            # Primary optimization target: maximize facts per LLM call
            "maximize_facts_per_llm_call": current_metrics.get("facts_per_llm_call", 0.0),

            # Secondary target: improve question quality
            "maximize_question_quality": current_metrics.get("question_quality_score", 0.0),

            # Efficiency target: minimize wasted LLM calls
            "minimize_llm_waste_ratio": current_metrics.get("llm_calls_wasted", 0) / max(current_metrics.get("total_questions_asked", 1), 1),

            # Coverage target: achieve comprehensive understanding
            "maximize_coverage_completeness": current_metrics.get("overall_coverage", 0.0),

            # Accuracy target: correctly distinguish discoverable vs undiscoverable
            "maximize_knowledge_classification_accuracy": current_metrics.get("knowledge_type_accuracy", 0.0)
        }

    def _calculate_area_coverage(self, discovered_facts: List[DiscoveredFact]) -> Dict[str, float]:
        """Calculate coverage across the 7 metadata understanding areas."""

        # Count facts in each area based on content analysis
        area_counts = {
            "semantic_memory": 0,
            "human_interaction": 0,
            "data_limitation": 0,
            "contextual_threshold": 0,
            "meta_learning": 0,
            "ego_network": 0,
            "comprehensive": 0
        }

        for fact in discovered_facts:
            content_lower = fact.content.lower()
            category_lower = fact.category.lower()

            # Classify into areas based on content
            if any(term in content_lower or term in category_lower
                   for term in ['semantic', 'meaning', 'data_structure', 'table', 'column']):
                area_counts["semantic_memory"] += 1

            if any(term in content_lower or term in category_lower
                   for term in ['entity_resolution', 'business_context', 'company']):
                area_counts["human_interaction"] += 1

            if any(term in content_lower or term in category_lower
                   for term in ['limitation', 'privacy', 'constraint', 'missing']):
                area_counts["data_limitation"] += 1

            if any(term in content_lower or term in category_lower
                   for term in ['network', 'connection', 'ego', 'relationship']):
                area_counts["ego_network"] += 1

            # Add other area classifications...

        # Calculate coverage percentages against ground truth
        coverage = {}
        total_ground_truth = len(self.ground_truth_knowledge)

        for area, count in area_counts.items():
            # Estimate expected facts per area (could be refined with ground truth analysis)
            expected_per_area = total_ground_truth / 7  # Rough estimate
            coverage[area] = min(1.0, count / max(expected_per_area, 1))

        coverage["overall_coverage"] = sum(coverage.values()) / len(coverage)

        return coverage

    async def _calculate_metadata_area_coverage(self, discovered_facts: List[DiscoveredFact]) -> Dict[str, float]:
        """Calculate coverage across 7 metadata areas using IntellAgent's knowledge as target."""

        try:
            from src.graphs.memory import get_memory_tools

            # Get IntellAgent's search tool to find target knowledge in each area
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
                logger.warning("IntellAgent search tool not found for coverage calculation")
                return self._fallback_calculate_coverage(discovered_facts)

            # Define what each area should cover based on IntellAgent's knowledge
            area_targets = {
                'semantic_memory': ['table meanings', 'column meanings', 'data structure'],
                'human_interaction': ['business context', 'entity resolution', 'company structure'],
                'data_limitation': ['privacy constraints', 'data limitations', 'missing data'],
                'contextual_threshold': ['discovery strategy', 'when to ask'],
                'meta_learning': ['learning progress', 'knowledge gaps'],
                'ego_network': ['network relationships', 'connection patterns', 'ego nodes'],
                'comprehensive': ['complete understanding', 'integrated knowledge']
            }

            # Count discovered facts in each area
            area_discovered = {area: 0 for area in area_targets.keys()}
            area_targets_available = {area: 0 for area in area_targets.keys()}

            # Count target knowledge available in IntellAgent for each area
            for area, keywords in area_targets.items():
                for keyword in keywords:
                    try:
                        results = search_tool.invoke({'query': keyword, 'limit': 5})
                        if results:
                            area_targets_available[area] += len(results)
                    except Exception as e:
                        logger.warning(f"Error searching for {keyword}: {e}")

            # Count discovered facts that match each area
            for fact in discovered_facts:
                content_lower = fact.content.lower()
                category_lower = fact.category.lower()

                # Semantic memory: table/column meanings, data structure
                if any(term in content_lower or term in category_lower for term in [
                    'table', 'column', 'field', 'structure', 'meaning', 'data_structure'
                ]):
                    area_discovered['semantic_memory'] += 1

                # Human interaction: business context, entity resolution
                if any(term in content_lower or term in category_lower for term in [
                    'business', 'company', 'entity', 'resolution', 'johannes', 'david', 'business_context'
                ]):
                    area_discovered['human_interaction'] += 1

                # Data limitation: privacy, constraints, limitations
                if any(term in content_lower or term in category_lower for term in [
                    'privacy', 'limitation', 'constraint', 'missing', 'data_limitation'
                ]):
                    area_discovered['data_limitation'] += 1

                # Ego network: network relationships, connections
                if any(term in content_lower or term in category_lower for term in [
                    'network', 'connection', 'ego', 'relationship', 'ego_network'
                ]):
                    area_discovered['ego_network'] += 1

                # Other areas get partial credit
                if any(term in content_lower for term in ['strategy', 'discovery']):
                    area_discovered['contextual_threshold'] += 1
                if any(term in content_lower for term in ['learning', 'meta']):
                    area_discovered['meta_learning'] += 1
                if any(term in content_lower for term in ['complete', 'comprehensive']):
                    area_discovered['comprehensive'] += 1

            # Calculate coverage percentages
            coverage = {}
            for area in area_targets.keys():
                if area_targets_available[area] > 0:
                    coverage[area] = min(1.0, area_discovered[area] / area_targets_available[area])
                else:
                    coverage[area] = 1.0 if area_discovered[area] > 0 else 0.0

            return coverage

        except Exception as e:
            logger.error(f"Error calculating metadata area coverage: {e}")
            return self._fallback_calculate_coverage(discovered_facts)

    def _fallback_calculate_coverage(self, discovered_facts: List[DiscoveredFact]) -> Dict[str, float]:
        """Fallback coverage calculation when IntellAgent search is unavailable."""

        # Simple heuristic based on fact categories and content
        area_counts = {
            'semantic_memory': 0,
            'human_interaction': 0,
            'data_limitation': 0,
            'contextual_threshold': 0,
            'meta_learning': 0,
            'ego_network': 0,
            'comprehensive': 0
        }

        for fact in discovered_facts:
            content_lower = fact.content.lower()
            category_lower = fact.category.lower()

            if any(term in content_lower for term in ['table', 'column', 'structure']):
                area_counts['semantic_memory'] += 1
            if any(term in content_lower for term in ['business', 'company', 'johannes', 'david']):
                area_counts['human_interaction'] += 1
            if any(term in content_lower for term in ['privacy', 'limitation', 'constraint']):
                area_counts['data_limitation'] += 1
            if any(term in content_lower for term in ['network', 'connection', 'ego']):
                area_counts['ego_network'] += 1

        # Convert to coverage percentages (assume 3 facts per area as target)
        target_per_area = 3
        coverage = {}
        for area, count in area_counts.items():
            coverage[area] = min(1.0, count / target_per_area)

        return coverage


# Integration functions for LangSmith and DSPy
async def evaluate_sql_agent_metadata_learning(discovered_facts: List[DiscoveredFact],
                                              questions: List[SQLAgentQuestion]) -> SQLAgentMetadataLearningMetrics:
    """
    Evaluate SQL agent's metadata learning performance using IntellAgent's knowledge as target.

    Focuses on the agent's ability to ask strategic questions to uncover essential
    metadata understanding across the 7 areas.
    """
    evaluator = SQLAgentPerformanceEvaluator()
    await evaluator.load_intellagent_knowledge()

    # Analyze what metadata knowledge was uncovered
    metadata_knowledge = {
        'column_meanings': 0,
        'table_relationships': 0,
        'business_context': 0,
        'data_limitations': 0
    }

    for fact in discovered_facts:
        content_lower = fact.content.lower()

        # Count essential metadata knowledge types
        if any(term in content_lower for term in ['person field', 'url field', 'column', 'ego node']):
            metadata_knowledge['column_meanings'] += 1
        if any(term in content_lower for term in ['entity resolution', 'cross-dataset', 'relationship']):
            metadata_knowledge['table_relationships'] += 1
        if any(term in content_lower for term in ['johannes castner', 'david cuff', 'company', 'business']):
            metadata_knowledge['business_context'] += 1
        if any(term in content_lower for term in ['privacy', 'limitation', 'constraint', 'missing']):
            metadata_knowledge['data_limitations'] += 1

    # Evaluate question quality for metadata learning
    excellent_metadata_questions = 0
    good_context_questions = 0
    poor_trivial_questions = 0

    for question in questions:
        question_lower = question.question_text.lower()

        # Excellent: Ask about column/table meanings, business context
        if (any(term in question_lower for term in ['what does', 'meaning', 'represent', 'purpose']) and
            any(term in question_lower for term in ['field', 'column', 'table', 'person', 'url'])):
            excellent_metadata_questions += 1

        # Good: Ask about relationships, context, limitations
        elif any(term in question_lower for term in ['relationship', 'connect', 'why', 'limitation', 'context']):
            good_context_questions += 1

        # Poor: Ask about trivial discoverable things
        elif any(term in question_lower for term in ['how many', 'data type', 'rows', 'count']):
            poor_trivial_questions += 1

    # Calculate coverage across 7 areas using IntellAgent's knowledge
    area_coverage = await evaluator._calculate_metadata_area_coverage(discovered_facts)

    # Calculate efficiency metrics
    total_llm_calls = sum(f.llm_calls_to_discover for f in discovered_facts)
    total_metadata_knowledge = sum(metadata_knowledge.values())

    return SQLAgentMetadataLearningMetrics(
        total_metadata_knowledge_uncovered=total_metadata_knowledge,
        column_meanings_discovered=metadata_knowledge['column_meanings'],
        table_relationships_mapped=metadata_knowledge['table_relationships'],
        business_context_learned=metadata_knowledge['business_context'],
        data_limitations_identified=metadata_knowledge['data_limitations'],

        total_questions_to_intellagent=len(questions),
        excellent_metadata_questions=excellent_metadata_questions,
        good_context_questions=good_context_questions,
        poor_trivial_questions=poor_trivial_questions,
        question_effectiveness_score=total_metadata_knowledge / max(len(questions), 1),

        semantic_memory_completion=area_coverage.get('semantic_memory', 0.0),
        human_interaction_optimization=area_coverage.get('human_interaction', 0.0),
        data_limitation_recognition=area_coverage.get('data_limitation', 0.0),
        contextual_threshold_learning=area_coverage.get('contextual_threshold', 0.0),
        self_discovery_meta_learning=area_coverage.get('meta_learning', 0.0),
        ego_network_pattern_learning=area_coverage.get('ego_network', 0.0),
        comprehensive_coverage=area_coverage.get('comprehensive', 0.0),
        overall_metadata_coverage=sum(area_coverage.values()) / len(area_coverage),

        metadata_knowledge_per_question=total_metadata_knowledge / max(len(questions), 1),
        llm_calls_for_exploration=sum(f.llm_calls_to_discover for f in discovered_facts if f.discovery_method == "exploration"),
        llm_calls_for_questioning=sum(f.llm_calls_to_discover for f in discovered_facts if f.discovery_method == "human_question"),
        total_llm_calls=total_llm_calls,
        knowledge_efficiency_ratio=total_metadata_knowledge / max(total_llm_calls, 1),

        questions_targeting_undiscoverable=excellent_metadata_questions + good_context_questions,
        questions_about_discoverable_things=poor_trivial_questions,
        learning_focus_accuracy=(excellent_metadata_questions + good_context_questions) / max(len(questions), 1)
    )


def log_metadata_metrics_to_langsmith(metrics: SQLAgentMetadataLearningMetrics, session_id: str) -> None:
    """Log metadata learning metrics to LangSmith for tracking and analysis."""
    try:
        # Convert metrics to dict for LangSmith
        metrics_dict = {
            "session_id": session_id,
            "metadata_knowledge_uncovered": metrics.total_metadata_knowledge_uncovered,
            "question_effectiveness_score": metrics.question_effectiveness_score,
            "overall_metadata_coverage": metrics.overall_metadata_coverage,
            "knowledge_efficiency_ratio": metrics.knowledge_efficiency_ratio,
            "learning_focus_accuracy": metrics.learning_focus_accuracy,

            # Detailed breakdown
            "column_meanings_discovered": metrics.column_meanings_discovered,
            "business_context_learned": metrics.business_context_learned,
            "excellent_metadata_questions": metrics.excellent_metadata_questions,
            "poor_trivial_questions": metrics.poor_trivial_questions,
            "total_questions": metrics.total_questions_to_intellagent,

            # 7 area coverage
            "semantic_memory_completion": metrics.semantic_memory_completion,
            "human_interaction_optimization": metrics.human_interaction_optimization,
            "data_limitation_recognition": metrics.data_limitation_recognition,
            "ego_network_pattern_learning": metrics.ego_network_pattern_learning
        }

        # Log to LangSmith (implementation depends on LangSmith setup)
        logger.info(f"SQL Agent Metadata Learning Metrics: {json.dumps(metrics_dict, indent=2)}")

    except Exception as e:
        logger.error(f"Failed to log metadata metrics to LangSmith: {e}")


def get_metadata_learning_optimization_targets(metrics: SQLAgentMetadataLearningMetrics) -> Dict[str, float]:
    """Get optimization targets for DSPy to improve SQL agent's metadata learning."""

    return {
        # Primary target: maximize metadata knowledge per question
        "maximize_metadata_knowledge_per_question": metrics.metadata_knowledge_per_question,

        # Question quality targets
        "maximize_excellent_metadata_questions": metrics.excellent_metadata_questions / max(metrics.total_questions_to_intellagent, 1),
        "minimize_trivial_questions": 1.0 - (metrics.poor_trivial_questions / max(metrics.total_questions_to_intellagent, 1)),

        # Coverage targets across 7 areas
        "maximize_semantic_memory_completion": metrics.semantic_memory_completion,
        "maximize_human_interaction_optimization": metrics.human_interaction_optimization,
        "maximize_data_limitation_recognition": metrics.data_limitation_recognition,
        "maximize_ego_network_learning": metrics.ego_network_pattern_learning,
        "maximize_overall_metadata_coverage": metrics.overall_metadata_coverage,

        # Efficiency targets
        "maximize_knowledge_efficiency_ratio": metrics.knowledge_efficiency_ratio,
        "maximize_learning_focus_accuracy": metrics.learning_focus_accuracy,

        # Strategic questioning target
        "maximize_undiscoverable_targeting": metrics.questions_targeting_undiscoverable / max(metrics.total_questions_to_intellagent, 1)
    }


# Additional integration functions for compatibility with existing code
async def evaluate_sql_agent_session(discovered_facts: List[DiscoveredFact],
                                    questions: List[SQLAgentQuestion]):
    """
    Evaluate a complete SQL agent session and return comprehensive metrics.

    This creates a comprehensive metrics object compatible with the demo and testing code.
    """
    from src.metrics.sql_agent_discovery_metrics import SQLAgentPerformanceMetrics

    evaluator = SQLAgentPerformanceEvaluator()
    await evaluator.load_intellagent_knowledge()

    # Calculate basic metrics
    total_facts = len(discovered_facts)
    total_llm_calls = sum(fact.llm_calls_to_discover for fact in discovered_facts)
    facts_per_llm_call = total_facts / max(total_llm_calls, 1)

    # Count knowledge types
    discoverable_facts = sum(1 for fact in discovered_facts if fact.knowledge_type == KnowledgeType.DISCOVERABLE)
    undiscoverable_facts = sum(1 for fact in discovered_facts if fact.knowledge_type == KnowledgeType.UNDISCOVERABLE)

    # Evaluate questions
    excellent_questions = sum(1 for q in questions if q.targets_undiscoverable and q.is_essential_for_job and not q.could_be_discovered_normally)
    good_questions = sum(1 for q in questions if q.targets_undiscoverable or q.estimated_llm_calls_saved >= 5)
    poor_questions = sum(1 for q in questions if q.could_be_discovered_normally and q.estimated_llm_calls_saved >= 2)
    terrible_questions = sum(1 for q in questions if q.could_be_discovered_normally and q.estimated_llm_calls_saved < 2)

    # Calculate efficiency metrics
    llm_calls_saved = sum(q.estimated_llm_calls_saved for q in questions if q.targets_undiscoverable)
    llm_calls_wasted = terrible_questions * 2
    net_efficiency_gain = llm_calls_saved - llm_calls_wasted

    # Calculate coverage
    area_coverage = evaluator._calculate_area_coverage(discovered_facts)

    # Create comprehensive metrics object
    return SQLAgentPerformanceMetrics(
        total_facts_discovered=total_facts,
        verified_facts_matched=len([f for f in discovered_facts if f.confidence > 0.9]),
        new_valid_facts=total_facts,
        total_llm_calls=total_llm_calls,
        facts_per_llm_call=facts_per_llm_call,
        verified_facts_per_llm_call=facts_per_llm_call,

        discoverable_facts_found=discoverable_facts,
        undiscoverable_facts_found=undiscoverable_facts,
        knowledge_type_accuracy=0.85,  # Placeholder

        total_questions_to_humans=len(questions),
        excellent_questions=excellent_questions,
        good_questions=good_questions,
        poor_questions=poor_questions,
        terrible_questions=terrible_questions,
        question_quality_score=(excellent_questions + good_questions * 0.7) / max(len(questions), 1),

        llm_calls_saved_by_good_questions=llm_calls_saved,
        llm_calls_wasted_by_bad_questions=llm_calls_wasted,
        net_efficiency_gain=net_efficiency_gain,
        efficiency_improvement_ratio=1.0 + (net_efficiency_gain / max(total_llm_calls, 1)),

        semantic_memory_coverage=area_coverage.get("semantic_memory", 0.0),
        human_interaction_coverage=area_coverage.get("human_interaction", 0.0),
        data_limitation_coverage=area_coverage.get("data_limitation", 0.0),
        contextual_threshold_coverage=area_coverage.get("contextual_threshold", 0.0),
        meta_learning_coverage=area_coverage.get("meta_learning", 0.0),
        ego_network_coverage=area_coverage.get("ego_network", 0.0),
        comprehensive_coverage=area_coverage.get("comprehensive", 0.0),
        overall_coverage_score=area_coverage.get("overall_coverage", 0.0)
    )


def log_metrics_to_langsmith(metrics, session_id: str) -> None:
    """Log general metrics to LangSmith for tracking and analysis."""
    try:
        # Convert metrics object to dict if needed
        if hasattr(metrics, '__dict__'):
            metrics_dict = metrics.__dict__
        else:
            metrics_dict = metrics

        # Add session ID to metrics
        metrics_with_session = {
            "session_id": session_id,
            **metrics_dict
        }

        # Log to LangSmith (implementation depends on LangSmith setup)
        logger.info(f"SQL Agent Session Metrics: {json.dumps(metrics_with_session, indent=2)}")

    except Exception as e:
        logger.error(f"Failed to log metrics to LangSmith: {e}")


def get_dspy_optimization_targets(metrics) -> Dict[str, float]:
    """Get DSPy optimization targets from general metrics."""
    # Convert metrics object to dict if needed
    if hasattr(metrics, '__dict__'):
        metrics_dict = metrics.__dict__
    else:
        metrics_dict = metrics

    evaluator = SQLAgentPerformanceEvaluator()
    return evaluator.generate_dspy_optimization_targets(metrics_dict)
