"""
Difficulty Assessment Engine for NeuronMap
==========================================

This module implements a robust difficulty assessment system for automatic
question complexity analysis with linguistic, cognitive, and semantic evaluation.
"""

import spacy
import torch
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re
import json
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to import transformers for BERT analysis
try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    """Difficulty levels on a 10-point scale."""
    VERY_EASY = 1      # Basic recall, simple facts
    EASY = 2           # Straightforward questions
    BASIC = 3          # Basic application
    MODERATE_LOW = 4   # Two-step reasoning
    MODERATE = 5       # Analysis and synthesis
    MODERATE_HIGH = 6  # Multi-step reasoning
    CHALLENGING = 7    # Critical evaluation
    HARD = 8           # Complex integration
    VERY_HARD = 9      # Creative problem-solving
    EXPERT = 10        # Research-level understanding


class ReasoningType(Enum):
    """Types of reasoning patterns."""
    FACTUAL_RECALL = "factual_recall"
    DEFINITIONAL = "definitional"
    DESCRIPTIVE = "descriptive"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    CAUSAL = "causal"
    EVALUATIVE = "evaluative"
    SYNTHETIC = "synthetic"
    CREATIVE = "creative"
    HYPOTHETICAL = "hypothetical"


@dataclass
class LinguisticFeatures:
    """Linguistic complexity features."""
    sentence_length: float
    dependency_depth: float
    subordinate_clauses: int
    type_token_ratio: float
    word_frequency_score: float
    technical_term_density: float
    syntactic_complexity: float
    lexical_diversity: float


@dataclass
class CognitiveLoadMetrics:
    """Cognitive load assessment metrics."""
    information_units: int
    working_memory_load: float
    reasoning_steps: int
    cross_references: int
    temporal_sequencing: float
    abstraction_level: float
    domain_knowledge_required: float


@dataclass
class SemanticComplexity:
    """Semantic complexity analysis."""
    concept_abstractness: float
    semantic_density: float
    polysemy_score: float
    context_dependency: float
    metaphor_usage: float
    domain_specificity: float


@dataclass
class DifficultyMetrics:
    """Complete difficulty assessment results."""
    difficulty_level: DifficultyLevel
    difficulty_score: float  # 1.0 - 10.0
    confidence: float  # 0.0 - 1.0
    linguistic_features: LinguisticFeatures
    cognitive_load: CognitiveLoadMetrics
    semantic_complexity: SemanticComplexity
    reasoning_type: ReasoningType
    domain_indicators: List[str]
    complexity_factors: Dict[str, float]
    recommendations: List[str]


class VocabularyFrequencyAnalyzer:
    """Analyzes vocabulary frequency for lexical complexity assessment."""

    def __init__(self):
        self.frequency_dict = self._load_default_frequencies()
        self.technical_terms = self._load_technical_terms()

    def _load_default_frequencies(self) -> Dict[str, float]:
        """Load default word frequency data (simplified implementation)."""
        # In a full implementation, this would load from Brown Corpus or similar
        common_words = {
            'the': 1.0, 'be': 0.9, 'to': 0.85, 'of': 0.8, 'and': 0.75,
            'a': 0.7, 'in': 0.65, 'that': 0.6, 'have': 0.55, 'it': 0.5,
            'for': 0.45, 'not': 0.4, 'on': 0.35, 'with': 0.3, 'he': 0.25,
            'as': 0.2, 'you': 0.15, 'do': 0.1, 'at': 0.08, 'this': 0.06,
            'but': 0.05, 'his': 0.04, 'by': 0.03, 'from': 0.02
        }
        return common_words

    def _load_technical_terms(self) -> Dict[str, List[str]]:
        """Load technical terms by domain."""
        return {
            'science': ['hypothesis', 'methodology', 'empirical', 'correlation', 'variable',
                       'paradigm', 'systematic', 'theoretical', 'experimental', 'analysis'],
            'mathematics': ['theorem', 'proof', 'axiom', 'derivative', 'integral',
                           'algorithm', 'optimization', 'computational', 'mathematical'],
            'philosophy': ['epistemology', 'ontology', 'metaphysics', 'dialectic',
                          'phenomenology', 'consciousness', 'existential', 'philosophical'],
            'medicine': ['pathology', 'etiology', 'prognosis', 'syndrome', 'diagnosis',
                        'clinical', 'therapeutic', 'medical'],
            'technology': ['algorithm', 'optimization', 'implementation', 'architecture',
                          'computational', 'systematic', 'technological'],
            'psychology': ['cognitive', 'behavioral', 'psychological', 'mental', 'consciousness'],
            'general_academic': ['comprehensive', 'systematic', 'analytical', 'theoretical',
                                'methodological', 'empirical', 'conceptual', 'implications']
        }

    def get_word_frequency(self, word: str) -> float:
        """Get frequency score for a word (0.0-1.0, higher = more common)."""
        word_lower = word.lower()
        return self.frequency_dict.get(word_lower, 0.01)  # Default low frequency

    def calculate_technical_density(self, text: str) -> float:
        """Calculate density of technical terms in text."""
        words = text.lower().split()
        if not words:
            return 0.0

        technical_count = 0
        for domain_terms in self.technical_terms.values():
            for term in domain_terms:
                technical_count += text.lower().count(term)

        return technical_count / len(words)


class BERTSemanticAnalyzer:
    """BERT-based semantic complexity analysis."""

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.enabled = HAS_TRANSFORMERS

        if self.enabled:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.eval()
            except Exception as e:
                logger.warning(f"Failed to load BERT model: {e}")
                self.enabled = False

    def analyze_semantic_complexity(self, text: str) -> SemanticComplexity:
        """Analyze semantic complexity using BERT embeddings."""
        if not self.enabled:
            # Fallback to simple heuristics
            return self._fallback_semantic_analysis(text)

        try:
            # Tokenize and get embeddings
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.squeeze(0)

            # Calculate semantic metrics
            embedding_variance = torch.var(embeddings, dim=0).mean().item()
            semantic_density = self._calculate_semantic_density(embeddings)
            context_dependency = self._calculate_context_dependency(embeddings)

            return SemanticComplexity(
                concept_abstractness=min(embedding_variance * 10, 1.0),
                semantic_density=semantic_density,
                polysemy_score=self._estimate_polysemy(text),
                context_dependency=context_dependency,
                metaphor_usage=self._detect_metaphors(text),
                domain_specificity=self._calculate_domain_specificity(text)
            )

        except Exception as e:
            logger.warning(f"BERT analysis failed: {e}, using fallback")
            return self._fallback_semantic_analysis(text)

    def _fallback_semantic_analysis(self, text: str) -> SemanticComplexity:
        """Fallback semantic analysis without BERT."""
        words = text.split()
        word_count = len(words)
        unique_words = len(set(words))

        # Simple heuristics
        concept_abstractness = min(len([w for w in words if len(w) > 8]) / max(word_count, 1), 1.0)
        semantic_density = unique_words / max(word_count, 1)

        return SemanticComplexity(
            concept_abstractness=concept_abstractness,
            semantic_density=semantic_density,
            polysemy_score=0.5,  # Default
            context_dependency=0.5,  # Default
            metaphor_usage=self._detect_metaphors(text),
            domain_specificity=self._calculate_domain_specificity(text)
        )

    def _calculate_semantic_density(self, embeddings: torch.Tensor) -> float:
        """Calculate semantic density from embeddings."""
        # Simplified: use average cosine similarity between token embeddings
        similarities = cosine_similarity(embeddings.numpy())
        return float(np.mean(similarities))

    def _calculate_context_dependency(self, embeddings: torch.Tensor) -> float:
        """Calculate how much meaning depends on context."""
        # Simplified: variance in token similarities
        similarities = cosine_similarity(embeddings.numpy())
        return float(np.var(similarities))

    def _estimate_polysemy(self, text: str) -> float:
        """Estimate polysemy (multiple meanings) in text."""
        # Simple heuristic: presence of words with multiple meanings
        polysemous_words = ['bank', 'bark', 'bat', 'bow', 'fair', 'left', 'right', 'bank']
        words = text.lower().split()
        polysemy_count = sum(1 for word in words if word in polysemous_words)
        return min(polysemy_count / max(len(words), 1), 1.0)

    def _detect_metaphors(self, text: str) -> float:
        """Detect metaphorical language usage."""
        metaphor_indicators = ['like', 'as if', 'metaphorically', 'symbolically', 'represents']
        score = sum(1 for indicator in metaphor_indicators if indicator in text.lower())
        return min(score / 10.0, 1.0)

    def _calculate_domain_specificity(self, text: str) -> float:
        """Calculate how domain-specific the text is."""
        # Simple heuristic based on technical term density
        analyzer = VocabularyFrequencyAnalyzer()
        return analyzer.calculate_technical_density(text)


class DifficultyAssessmentEngine:
    """Main difficulty assessment engine."""

    def __init__(self, bert_model: str = "bert-base-uncased", enable_bert: bool = False, enable_spacy: bool = True):
        """
        Initialize the difficulty assessment engine.

        Args:
            bert_model: BERT model name to use for semantic analysis
            enable_bert: Whether to enable BERT-based semantic analysis (slower but more accurate)
            enable_spacy: Whether to enable spaCy for linguistic analysis (faster than BERT)
        """
        # Performance optimization: make components optional
        self.enable_bert = enable_bert
        self.enable_spacy = enable_spacy

        # Initialize spaCy parser (faster than BERT)
        self.spacy_parser = None
        if self.enable_spacy:
            try:
                self.spacy_parser = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, using fallback parsing")
                self.spacy_parser = None

        # Initialize BERT analyzer (slower, optional)
        if self.enable_bert:
            self.bert_analyzer = BERTSemanticAnalyzer(bert_model)
        else:
            # Use lightweight analyzer without BERT initialization
            self.bert_analyzer = BERTSemanticAnalyzer.__new__(BERTSemanticAnalyzer)
            self.bert_analyzer.enabled = False
            self.bert_analyzer.model_name = ""
            self.bert_analyzer.tokenizer = None
            self.bert_analyzer.model = None

        self.vocab_analyzer = VocabularyFrequencyAnalyzer()

        # Question pattern classifiers
        self.question_patterns = self._initialize_question_patterns()
        self.reasoning_indicators = self._initialize_reasoning_indicators()

    def _initialize_question_patterns(self) -> Dict[str, List[str]]:
        """Initialize question pattern recognition."""
        return {
            'factual': [r'\bwhat is\b', r'\bwho is\b', r'\bwhen did\b', r'\bwhere is\b'],
            'analytical': [r'\bwhy does\b', r'\bhow does\b', r'\banalyze\b', r'\bexplain\b'],
            'evaluative': [r'\bevaluate\b', r'\bassess\b', r'\bjudge\b', r'\bcritique\b'],
            'comparative': [r'\bcompare\b', r'\bcontrast\b', r'\bdifference\b', r'\bsimilar\b'],
            'creative': [r'\bdesign\b', r'\bcreate\b', r'\bimagine\b', r'\binvent\b'],
            'hypothetical': [r'\bif\b.*\bthen\b', r'\bwhat if\b', r'\bsuppose\b', r'\bassume\b']
        }

    def _initialize_reasoning_indicators(self) -> Dict[ReasoningType, List[str]]:
        """Initialize reasoning type indicators."""
        return {
            ReasoningType.FACTUAL_RECALL: ['what is', 'who is', 'when was', 'where is', 'which is'],
            ReasoningType.DEFINITIONAL: ['define', 'definition', 'meaning of', 'what does', 'term'],
            ReasoningType.DESCRIPTIVE: ['describe', 'how does', 'process of', 'steps to'],
            ReasoningType.ANALYTICAL: ['analyze', 'examination of', 'investigate', 'why does'],
            ReasoningType.COMPARATIVE: ['compare', 'contrast', 'difference between', 'similarities'],
            ReasoningType.CAUSAL: ['what causes', 'why did', 'because of', 'results from', 'leads to'],
            ReasoningType.EVALUATIVE: ['evaluate', 'assess', 'judge', 'critique', 'effectiveness of'],
            ReasoningType.SYNTHETIC: ['synthesize', 'combine', 'integrate', 'merge'],
            ReasoningType.CREATIVE: ['create', 'design', 'invent', 'imagine', 'develop'],
            ReasoningType.HYPOTHETICAL: ['what if', 'suppose that', 'assume that', 'hypothetical']
        }

    def assess_difficulty(self, question: str) -> DifficultyMetrics:
        """Main method to assess question difficulty."""
        # Clean and prepare text
        cleaned_question = self._clean_text(question)

        # Analyze linguistic features
        linguistic_features = self._analyze_linguistic_features(cleaned_question)

        # Assess cognitive load
        cognitive_load = self._assess_cognitive_load(cleaned_question)

        # Analyze semantic complexity
        semantic_complexity = self.bert_analyzer.analyze_semantic_complexity(cleaned_question)

        # Determine reasoning type
        reasoning_type = self._classify_reasoning_type(cleaned_question)

        # Identify domain indicators
        domain_indicators = self._identify_domain_indicators(cleaned_question)

        # Calculate overall difficulty
        difficulty_score, complexity_factors = self._calculate_difficulty_score(
            linguistic_features, cognitive_load, semantic_complexity, reasoning_type
        )

        # Convert to difficulty level
        difficulty_level = self._score_to_level(difficulty_score)

        # Calculate confidence
        confidence = self._calculate_confidence(complexity_factors)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            difficulty_score, linguistic_features, cognitive_load
        )

        return DifficultyMetrics(
            difficulty_level=difficulty_level,
            difficulty_score=difficulty_score,
            confidence=confidence,
            linguistic_features=linguistic_features,
            cognitive_load=cognitive_load,
            semantic_complexity=semantic_complexity,
            reasoning_type=reasoning_type,
            domain_indicators=domain_indicators,
            complexity_factors=complexity_factors,
            recommendations=recommendations
        )

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Handle common contractions
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "shouldn't": "should not", "wouldn't": "would not"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)

        return text

    def _analyze_linguistic_features(self, text: str) -> LinguisticFeatures:
        """Analyze linguistic complexity features."""
        if self.spacy_parser:
            doc = self.spacy_parser(text)
            return self._spacy_linguistic_analysis(doc)
        else:
            return self._fallback_linguistic_analysis(text)

    def _spacy_linguistic_analysis(self, doc) -> LinguisticFeatures:
        """Linguistic analysis using spaCy."""
        # Sentence-level features
        sentences = list(doc.sents)
        avg_sentence_length = np.mean([len(sent) for sent in sentences]) if sentences else 0

        # Dependency depth analysis
        max_depth = 0
        for token in doc:
            depth = self._calculate_dependency_depth(token)
            max_depth = max(max_depth, depth)

        # Subordinate clauses
        subordinate_clauses = len([token for token in doc if token.dep_ in ['advcl', 'ccomp', 'xcomp']])

        # Lexical diversity
        words = [token.text.lower() for token in doc if token.is_alpha]
        type_token_ratio = len(set(words)) / len(words) if words else 0

        # Word frequency analysis
        word_frequency_score = np.mean([
            self.vocab_analyzer.get_word_frequency(word) for word in words
        ]) if words else 0

        # Technical term density
        technical_term_density = self.vocab_analyzer.calculate_technical_density(doc.text)

        return LinguisticFeatures(
            sentence_length=avg_sentence_length,
            dependency_depth=max_depth,
            subordinate_clauses=subordinate_clauses,
            type_token_ratio=type_token_ratio,
            word_frequency_score=word_frequency_score,
            technical_term_density=technical_term_density,
            syntactic_complexity=max_depth / 10.0,  # Normalized
            lexical_diversity=type_token_ratio
        )

    def _fallback_linguistic_analysis(self, text: str) -> LinguisticFeatures:
        """Fallback linguistic analysis without spaCy."""
        sentences = text.split('.')
        words = text.split()

        avg_sentence_length = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
        type_token_ratio = len(set(words)) / len(words) if words else 0
        word_frequency_score = np.mean([
            self.vocab_analyzer.get_word_frequency(word) for word in words
        ]) if words else 0
        technical_term_density = self.vocab_analyzer.calculate_technical_density(text)

        return LinguisticFeatures(
            sentence_length=avg_sentence_length,
            dependency_depth=3.0,  # Default estimate
            subordinate_clauses=text.count(','),  # Rough estimate
            type_token_ratio=type_token_ratio,
            word_frequency_score=word_frequency_score,
            technical_term_density=technical_term_density,
            syntactic_complexity=0.3,  # Default
            lexical_diversity=type_token_ratio
        )

    def _calculate_dependency_depth(self, token) -> int:
        """Calculate dependency tree depth for a token."""
        depth = 0
        current = token
        while current.head != current:
            depth += 1
            current = current.head
            if depth > 20:  # Prevent infinite loops
                break
        return depth

    def _assess_cognitive_load(self, text: str) -> CognitiveLoadMetrics:
        """Assess cognitive load requirements."""
        # Information units (rough estimate based on content words)
        content_words = len([word for word in text.split() if len(word) > 3])
        information_units = min(content_words // 3, 10)  # Cluster into units

        # Working memory load indicators
        conjunctions = len(re.findall(r'\b(and|but|however|therefore|moreover)\b', text.lower()))
        comparisons = len(re.findall(r'\b(more|less|better|worse|compare)\b', text.lower()))
        working_memory_load = min((conjunctions + comparisons) / 5.0, 1.0)

        # Reasoning steps estimation
        reasoning_steps = self._estimate_reasoning_steps(text)

        # Cross-references
        cross_references = len(re.findall(r'\b(this|that|these|those|it|they)\b', text.lower()))

        # Temporal sequencing
        temporal_words = ['first', 'then', 'next', 'finally', 'before', 'after']
        temporal_sequencing = len([word for word in temporal_words if word in text.lower()]) / 10.0

        # Abstraction level
        abstraction_level = self._calculate_abstraction_level(text)

        # Domain knowledge requirements
        domain_knowledge_required = self.vocab_analyzer.calculate_technical_density(text)

        return CognitiveLoadMetrics(
            information_units=information_units,
            working_memory_load=working_memory_load,
            reasoning_steps=reasoning_steps,
            cross_references=cross_references,
            temporal_sequencing=temporal_sequencing,
            abstraction_level=abstraction_level,
            domain_knowledge_required=domain_knowledge_required
        )

    def _estimate_reasoning_steps(self, text: str) -> int:
        """Estimate number of reasoning steps required."""
        # Look for logical connectors and step indicators
        step_indicators = ['because', 'therefore', 'thus', 'hence', 'so', 'since', 'given']
        steps = sum(1 for indicator in step_indicators if indicator in text.lower())

        # Question complexity indicators - enhanced scoring
        if any(word in text.lower() for word in ['why', 'how', 'explain']):
            steps += 2
        if any(word in text.lower() for word in ['analyze', 'evaluate', 'compare', 'examine']):
            steps += 3
        if any(word in text.lower() for word in ['synthesize', 'design', 'create', 'develop']):
            steps += 4
        if any(word in text.lower() for word in ['critically', 'comprehensively', 'systematically']):
            steps += 2

        # Complex reasoning indicators
        if any(phrase in text.lower() for phrase in ['implications of', 'relationship between', 'factors contributing']):
            steps += 2
        if any(phrase in text.lower() for phrase in ['philosophical', 'theoretical', 'epistemological']):
            steps += 3
        if any(phrase in text.lower() for phrase in ['framework', 'methodology', 'comprehensive']):
            steps += 2

        # Multi-part questions
        if ' and ' in text:
            steps += 1
        if text.count(',') >= 2:
            steps += 1

        return max(1, min(steps, 10))  # Cap at 10 steps

    def _calculate_abstraction_level(self, text: str) -> float:
        """Calculate level of abstraction in the text."""
        # Enhanced abstract indicators
        abstract_indicators = [
            'concept', 'theory', 'principle', 'idea', 'notion', 'framework',
            'paradigm', 'model', 'hypothesis', 'assumption', 'perspective',
            'philosophical', 'theoretical', 'conceptual', 'epistemological',
            'ontological', 'metaphysical', 'phenomenological', 'systematic',
            'comprehensive', 'implications', 'foundations', 'nature of',
            'consciousness', 'reality', 'existence', 'objective', 'subjective'
        ]

        # High-level reasoning indicators
        reasoning_indicators = [
            'analyze', 'synthesize', 'evaluate', 'critically', 'systematically',
            'comprehensively', 'implications', 'methodology', 'theoretical framework'
        ]

        text_lower = text.lower()

        # Count abstract concepts
        abstract_count = sum(1 for indicator in abstract_indicators if indicator in text_lower)
        reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in text_lower)

        total_words = len(text.split())

        # Calculate base abstraction
        base_abstraction = (abstract_count + reasoning_count) / max(total_words, 1)

        # Boost for philosophical/theoretical content
        if any(term in text_lower for term in ['philosophical', 'epistemological', 'ontological', 'consciousness', 'reality']):
            base_abstraction *= 2

        # Boost for complex analytical questions
        if any(term in text_lower for term in ['critically analyze', 'comprehensively evaluate', 'synthesize']):
            base_abstraction *= 1.5

        return min(base_abstraction * 5, 1.0)  # Scale and cap at 1.0

    def _classify_reasoning_type(self, text: str) -> ReasoningType:
        """Classify the type of reasoning required."""
        text_lower = text.lower()

        # Score each reasoning type
        scores = {}
        for reasoning_type, indicators in self.reasoning_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            scores[reasoning_type] = score

        # Return the type with highest score, default to factual
        if not scores or max(scores.values()) == 0:
            return ReasoningType.FACTUAL_RECALL

        return max(scores.items(), key=lambda x: x[1])[0]

    def _identify_domain_indicators(self, text: str) -> List[str]:
        """Identify domain-specific indicators in the text."""
        domains = []
        text_lower = text.lower()

        domain_keywords = {
            'science': ['experiment', 'hypothesis', 'data', 'research', 'study'],
            'mathematics': ['equation', 'formula', 'calculate', 'proof', 'theorem'],
            'history': ['century', 'period', 'historical', 'era', 'ancient'],
            'literature': ['author', 'novel', 'poem', 'literary', 'character'],
            'philosophy': ['ethics', 'moral', 'existence', 'reality', 'truth'],
            'technology': ['system', 'algorithm', 'computer', 'software', 'digital'],
            'medicine': ['patient', 'diagnosis', 'treatment', 'medical', 'disease'],
            'psychology': ['behavior', 'cognitive', 'mental', 'psychological', 'mind']
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                domains.append(domain)

        return domains

    def _calculate_difficulty_score(self, linguistic: LinguisticFeatures,
                                  cognitive: CognitiveLoadMetrics,
                                  semantic: SemanticComplexity,
                                  reasoning: ReasoningType) -> Tuple[float, Dict[str, float]]:
        """Calculate overall difficulty score and component factors."""

        # Adaptive component weights based on reasoning type
        if reasoning in [ReasoningType.SYNTHETIC, ReasoningType.CREATIVE, ReasoningType.EVALUATIVE]:
            # For high-level reasoning, increase reasoning weight
            weights = {
                'linguistic': 0.20,
                'cognitive': 0.30,
                'semantic': 0.25,
                'reasoning': 0.25
            }
        elif reasoning in [ReasoningType.ANALYTICAL, ReasoningType.COMPARATIVE, ReasoningType.CAUSAL]:
            # For analytical reasoning, balance cognitive and reasoning
            weights = {
                'linguistic': 0.20,
                'cognitive': 0.35,
                'semantic': 0.20,
                'reasoning': 0.25
            }
        else:
            # Default weights for basic reasoning
            weights = {
                'linguistic': 0.25,
                'cognitive': 0.35,
                'semantic': 0.25,
                'reasoning': 0.15
            }

        # Calculate component scores (0-10 scale)
        linguistic_score = self._score_linguistic_complexity(linguistic)
        cognitive_score = self._score_cognitive_load(cognitive)
        semantic_score = self._score_semantic_complexity(semantic)
        reasoning_score = self._score_reasoning_complexity(reasoning)

        # Weighted combination
        base_score = (
            weights['linguistic'] * linguistic_score +
            weights['cognitive'] * cognitive_score +
            weights['semantic'] * semantic_score +
            weights['reasoning'] * reasoning_score
        )

        # Apply complexity boosters for questions with multiple high-complexity indicators
        complexity_boost = 0.0

        # Boost for high technical density + advanced reasoning
        if linguistic.technical_term_density > 0.15 and reasoning in [ReasoningType.ANALYTICAL, ReasoningType.EVALUATIVE, ReasoningType.SYNTHETIC]:
            complexity_boost += 1.0

        # Boost for high abstraction + complex reasoning
        if cognitive.abstraction_level > 0.7 and reasoning in [ReasoningType.EVALUATIVE, ReasoningType.SYNTHETIC, ReasoningType.CREATIVE]:
            complexity_boost += 1.5

        # Boost for multiple reasoning steps + advanced reasoning type
        if cognitive.reasoning_steps >= 4 and reasoning in [ReasoningType.ANALYTICAL, ReasoningType.EVALUATIVE, ReasoningType.SYNTHETIC]:
            complexity_boost += 1.0

        # Boost for questions with philosophical/theoretical content
        if semantic.concept_abstractness > 0.6 and reasoning in [ReasoningType.EVALUATIVE, ReasoningType.SYNTHETIC]:
            complexity_boost += 0.5

        total_score = base_score + complexity_boost

        complexity_factors = {
            'linguistic_complexity': linguistic_score,
            'cognitive_load': cognitive_score,
            'semantic_complexity': semantic_score,
            'reasoning_complexity': reasoning_score,
            'technical_density': linguistic.technical_term_density * 10,
            'abstraction_level': cognitive.abstraction_level * 10,
            'complexity_boost': complexity_boost
        }

        return min(max(total_score, 1.0), 10.0), complexity_factors

    def _score_linguistic_complexity(self, linguistic: LinguisticFeatures) -> float:
        """Score linguistic complexity (0-10)."""
        # Sentence length complexity (0-3 points)
        sentence_score = min(linguistic.sentence_length / 12.0 * 3, 3)

        # Syntax complexity (0-2.5 points)
        syntax_score = min(linguistic.dependency_depth / 6.0 * 2.5, 2.5)

        # Lexical complexity (0-2.5 points)
        lexical_score = min((1 - linguistic.word_frequency_score) * 3, 2.5)

        # Technical terminology (0-2 points) - boosted for complex terms
        technical_score = min(linguistic.technical_term_density * 15, 2)

        total = sentence_score + syntax_score + lexical_score + technical_score
        return min(total, 10.0)

    def _score_cognitive_load(self, cognitive: CognitiveLoadMetrics) -> float:
        """Score cognitive load (0-10)."""
        # Working memory load (0-2.5 points)
        memory_score = min(cognitive.working_memory_load * 2.5, 2.5)

        # Reasoning steps (0-3.5 points) - boosted for multi-step reasoning
        reasoning_score = min(cognitive.reasoning_steps / 6.0 * 3.5, 3.5)

        # Abstraction level (0-2.5 points) - boosted for abstract concepts
        abstraction_score = min(cognitive.abstraction_level * 3, 2.5)

        # Domain knowledge requirements (0-1.5 points)
        domain_score = min(cognitive.domain_knowledge_required * 1.5, 1.5)

        total = memory_score + reasoning_score + abstraction_score + domain_score
        return min(total, 10.0)

    def _score_semantic_complexity(self, semantic: SemanticComplexity) -> float:
        """Score semantic complexity (0-10)."""
        abstractness_score = min(semantic.concept_abstractness * 3, 3)    # Max 3 points
        density_score = min(semantic.semantic_density * 2, 2)             # Max 2 points
        polysemy_score = min(semantic.polysemy_score * 2, 2)              # Max 2 points
        context_score = min(semantic.context_dependency * 2, 2)           # Max 2 points
        metaphor_score = min(semantic.metaphor_usage * 1, 1)              # Max 1 point

        return abstractness_score + density_score + polysemy_score + context_score + metaphor_score

    def _score_reasoning_complexity(self, reasoning: ReasoningType) -> float:
        """Score reasoning complexity based on type (0-10)."""
        reasoning_scores = {
            ReasoningType.FACTUAL_RECALL: 1.0,
            ReasoningType.DEFINITIONAL: 2.0,
            ReasoningType.DESCRIPTIVE: 3.0,
            ReasoningType.ANALYTICAL: 5.0,
            ReasoningType.COMPARATIVE: 6.0,
            ReasoningType.CAUSAL: 7.0,
            ReasoningType.EVALUATIVE: 8.0,
            ReasoningType.SYNTHETIC: 9.0,
            ReasoningType.CREATIVE: 9.5,
            ReasoningType.HYPOTHETICAL: 8.5
        }

        return reasoning_scores.get(reasoning, 5.0)

    def _score_to_level(self, score: float) -> DifficultyLevel:
        """Convert numeric score to difficulty level."""
        if score <= 1.5:
            return DifficultyLevel.VERY_EASY
        elif score <= 2.5:
            return DifficultyLevel.EASY
        elif score <= 3.5:
            return DifficultyLevel.BASIC
        elif score <= 4.5:
            return DifficultyLevel.MODERATE_LOW
        elif score <= 5.5:
            return DifficultyLevel.MODERATE
        elif score <= 6.5:
            return DifficultyLevel.MODERATE_HIGH
        elif score <= 7.5:
            return DifficultyLevel.CHALLENGING
        elif score <= 8.5:
            return DifficultyLevel.HARD
        elif score <= 9.5:
            return DifficultyLevel.VERY_HARD
        else:
            return DifficultyLevel.EXPERT

    def _calculate_confidence(self, complexity_factors: Dict[str, float]) -> float:
        """Calculate confidence in the difficulty assessment."""
        # Higher confidence when factors are consistent
        factor_values = list(complexity_factors.values())
        if not factor_values:
            return 0.5

        # Calculate coefficient of variation
        mean_val = np.mean(factor_values)
        std_val = np.std(factor_values)

        if mean_val == 0:
            return 0.5

        cv = std_val / mean_val
        confidence = max(0.0, min(1.0, 1.0 - cv / 2.0))  # Lower CV = higher confidence

        return confidence

    def _generate_recommendations(self, difficulty_score: float,
                                linguistic: LinguisticFeatures,
                                cognitive: CognitiveLoadMetrics) -> List[str]:
        """Generate recommendations for difficulty adjustment."""
        recommendations = []

        if difficulty_score < 3.0:
            recommendations.append("Consider adding more complex vocabulary or concepts")
            recommendations.append("Increase reasoning steps or add analytical components")

        if difficulty_score > 8.0:
            recommendations.append("Simplify sentence structure or vocabulary")
            recommendations.append("Break down into smaller, more manageable parts")

        if linguistic.technical_term_density > 0.3:
            recommendations.append("High technical density detected - ensure target audience appropriateness")

        if cognitive.reasoning_steps > 6:
            recommendations.append("Complex reasoning required - consider providing scaffolding")

        if linguistic.sentence_length > 20:
            recommendations.append("Long sentences detected - consider breaking into shorter segments")

        return recommendations


# Convenience functions
def assess_question_difficulty(question: str, bert_model: str = "bert-base-uncased", fast_mode: bool = True) -> DifficultyMetrics:
    """
    Convenience function to assess a single question's difficulty.

    Args:
        question: The question to assess
        bert_model: BERT model to use (if fast_mode=False)
        fast_mode: If True, use faster algorithms without BERT (recommended for <150ms performance)
    """
    engine = DifficultyAssessmentEngine(bert_model, enable_bert=not fast_mode, enable_spacy=True)
    return engine.assess_difficulty(question)


def assess_question_difficulty_fast(question: str) -> DifficultyMetrics:
    """Fast assessment without BERT for <150ms performance."""
    engine = get_fast_engine()
    return engine.assess_difficulty(question)
def batch_assess_difficulty(questions: List[str],
                          bert_model: str = "bert-base-uncased",
                          fast_mode: bool = True) -> List[DifficultyMetrics]:
    """Assess difficulty for a batch of questions."""
    engine = DifficultyAssessmentEngine(bert_model, enable_bert=not fast_mode, enable_spacy=True)
    return [engine.assess_difficulty(q) for q in questions]


def get_difficulty_summary(metrics: DifficultyMetrics) -> Dict[str, Any]:
    """Get a summary of difficulty metrics for reporting."""
    return {
        "difficulty_level": metrics.difficulty_level.name,
        "difficulty_score": round(metrics.difficulty_score, 2),
        "confidence": round(metrics.confidence, 2),
        "reasoning_type": metrics.reasoning_type.value,
        "key_factors": {
            factor: round(score, 2)
            for factor, score in metrics.complexity_factors.items()
        },
        "domain_indicators": metrics.domain_indicators,
        "recommendations": metrics.recommendations
    }


# Global singleton for performance optimization
_fast_engine = None
_full_engine = None

def get_fast_engine() -> DifficultyAssessmentEngine:
    """Get a singleton fast engine for performance."""
    global _fast_engine
    if _fast_engine is None:
        _fast_engine = DifficultyAssessmentEngine(enable_bert=False, enable_spacy=True)
    return _fast_engine

def get_full_engine(bert_model: str = "bert-base-uncased") -> DifficultyAssessmentEngine:
    """Get a singleton full engine with BERT."""
    global _full_engine
    if _full_engine is None:
        _full_engine = DifficultyAssessmentEngine(bert_model, enable_bert=True, enable_spacy=True)
    return _full_engine
