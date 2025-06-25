"""
Difficulty Assessment Engine for NeuronMap
==========================================

This module provides a comprehensive framework for assessing the difficulty
of neural network questions and content across different domains and complexity levels.
"""

import logging
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    """Enumeration of difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class DifficultyMetrics:
    """Metrics for difficulty assessment."""
    complexity_score: float
    vocabulary_density: float
    concept_depth: int
    reasoning_level: int
    overall_difficulty: float
    confidence: float
    level: DifficultyLevel


class DifficultyAssessmentEngine:
    """Engine for assessing question and content difficulty."""

    def __init__(self):
        """Initialize the difficulty assessment engine."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Technical vocabulary indicators
        self.technical_terms = {
            'neural', 'network', 'activation', 'gradient', 'backpropagation',
            'tensor', 'matrix', 'algorithm', 'optimization', 'regularization',
            'convergence', 'epoch', 'batch', 'learning_rate', 'loss_function',
            'supervised', 'unsupervised', 'reinforcement', 'deep_learning',
            'convolutional', 'recurrent', 'transformer', 'attention'
        }

        # Complexity indicators
        self.complexity_indicators = {
            'mathematical': ['equation', 'formula', 'derivative', 'integral', 'calculus'],
            'conceptual': ['theory', 'principle', 'concept', 'framework', 'paradigm'],
            'implementation': ['code', 'implement', 'algorithm', 'function', 'class'],
            'analysis': ['analyze', 'evaluate', 'compare', 'contrast', 'critique']
        }

        # Reasoning level keywords
        self.reasoning_keywords = {
            1: ['what', 'define', 'list', 'identify'],
            2: ['explain', 'describe', 'summarize', 'outline'],
            3: ['analyze', 'compare', 'contrast', 'examine'],
            4: ['evaluate', 'critique', 'justify', 'synthesize']
        }

    def assess_difficulty(self, text: str, context: Optional[str] = None) -> DifficultyMetrics:
        """
        Assess the difficulty of a given text.

        Args:
            text: The text to assess
            context: Optional context for domain-specific assessment

        Returns:
            DifficultyMetrics object with assessment results
        """
        try:
            # Calculate various metrics
            complexity_score = self._calculate_complexity_score(text)
            vocabulary_density = self._calculate_vocabulary_density(text)
            concept_depth = self._calculate_concept_depth(text)
            reasoning_level = self._calculate_reasoning_level(text)

            # Calculate overall difficulty
            overall_difficulty = self._calculate_overall_difficulty(
                complexity_score, vocabulary_density, concept_depth, reasoning_level
            )

            # Determine confidence level
            confidence = self._calculate_confidence(text)

            # Determine difficulty level
            level = self._determine_level(overall_difficulty)

            return DifficultyMetrics(
                complexity_score=complexity_score,
                vocabulary_density=vocabulary_density,
                concept_depth=concept_depth,
                reasoning_level=reasoning_level,
                overall_difficulty=overall_difficulty,
                confidence=confidence,
                level=level
            )

        except Exception as e:
            self.logger.error(f"Error assessing difficulty: {e}")
            # Return default metrics on error
            return DifficultyMetrics(
                complexity_score=0.5,
                vocabulary_density=0.5,
                concept_depth=2,
                reasoning_level=2,
                overall_difficulty=0.5,
                confidence=0.1,
                level=DifficultyLevel.INTERMEDIATE
            )

    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate complexity score based on various indicators."""
        text_lower = text.lower()
        words = text_lower.split()

        if not words:
            return 0.0

        # Technical term density
        technical_count = sum(1 for word in words if any(term in word for term in self.technical_terms))
        technical_density = technical_count / len(words)

        # Sentence complexity (average length)
        sentences = re.split(r'[.!?]+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)

        # Normalize sentence length score (assuming complex sentences are 15+ words)
        sentence_complexity = min(avg_sentence_length / 15.0, 1.0)

        # Combine scores
        complexity_score = (technical_density * 0.6 + sentence_complexity * 0.4)
        return min(complexity_score, 1.0)

    def _calculate_vocabulary_density(self, text: str) -> float:
        """Calculate the density of technical vocabulary."""
        text_lower = text.lower()
        words = text_lower.split()

        if not words:
            return 0.0

        # Count technical terms
        technical_count = sum(1 for word in words if any(term in word for term in self.technical_terms))

        # Count complexity indicators
        complexity_count = 0
        for category, terms in self.complexity_indicators.items():
            complexity_count += sum(1 for word in words if any(term in word for term in terms))

        total_technical = technical_count + complexity_count
        density = total_technical / len(words)

        return min(density, 1.0)

    def _calculate_concept_depth(self, text: str) -> int:
        """Calculate the conceptual depth (1-5 scale)."""
        text_lower = text.lower()

        # Check for different types of complexity indicators
        depth_score = 1

        for category, terms in self.complexity_indicators.items():
            if any(term in text_lower for term in terms):
                if category == 'mathematical':
                    depth_score = max(depth_score, 4)
                elif category == 'conceptual':
                    depth_score = max(depth_score, 3)
                elif category == 'implementation':
                    depth_score = max(depth_score, 3)
                elif category == 'analysis':
                    depth_score = max(depth_score, 4)

        # Check for advanced neural network concepts
        advanced_concepts = ['transformer', 'attention', 'gan', 'vae', 'reinforcement']
        if any(concept in text_lower for concept in advanced_concepts):
            depth_score = max(depth_score, 5)

        return min(depth_score, 5)

    def _calculate_reasoning_level(self, text: str) -> int:
        """Calculate the reasoning level required (1-4 scale)."""
        text_lower = text.lower()

        reasoning_level = 1

        for level, keywords in self.reasoning_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                reasoning_level = max(reasoning_level, level)

        return reasoning_level

    def _calculate_overall_difficulty(self, complexity: float, vocabulary: float,
                                    depth: int, reasoning: int) -> float:
        """Calculate overall difficulty score (0.0-1.0)."""
        # Normalize depth and reasoning to 0-1 scale
        normalized_depth = (depth - 1) / 4.0
        normalized_reasoning = (reasoning - 1) / 3.0

        # Weighted combination
        overall = (
            complexity * 0.3 +
            vocabulary * 0.3 +
            normalized_depth * 0.2 +
            normalized_reasoning * 0.2
        )

        return min(overall, 1.0)

    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence in the assessment."""
        # Base confidence on text length and content richness
        words = text.split()

        if len(words) < 5:
            return 0.3
        elif len(words) < 15:
            return 0.6
        elif len(words) < 30:
            return 0.8
        else:
            return 0.9

    def _determine_level(self, overall_difficulty: float) -> DifficultyLevel:
        """Determine difficulty level from overall score."""
        if overall_difficulty < 0.25:
            return DifficultyLevel.BEGINNER
        elif overall_difficulty < 0.5:
            return DifficultyLevel.INTERMEDIATE
        elif overall_difficulty < 0.75:
            return DifficultyLevel.ADVANCED
        else:
            return DifficultyLevel.EXPERT

    def assess_question_difficulty(self, question: str, domain: Optional[str] = None) -> DifficultyMetrics:
        """
        Assess the difficulty of a specific question.

        Args:
            question: The question to assess
            domain: Optional domain context

        Returns:
            DifficultyMetrics object
        """
        return self.assess_difficulty(question, domain)

    def assess_question_quality(self, question: str) -> Dict[str, Any]:
        """Assess question quality - compatibility method for tests."""
        metrics = self.assess_question_difficulty(question)

        # Convert DifficultyMetrics to dict format expected by tests
        return {
            'overall_score': metrics.overall_difficulty,
            'complexity': metrics.complexity_score,
            'vocabulary_density': metrics.vocabulary_density,
            'clarity': metrics.confidence,
            'level': metrics.level.value,
            'metrics': metrics
        }

    def batch_assess(self, texts: List[str], context: Optional[str] = None) -> List[DifficultyMetrics]:
        """
        Assess difficulty for multiple texts.

        Args:
            texts: List of texts to assess
            context: Optional context for assessment

        Returns:
            List of DifficultyMetrics objects
        """
        return [self.assess_difficulty(text, context) for text in texts]

    def get_difficulty_summary(self, metrics_list: List[DifficultyMetrics]) -> Dict[str, Any]:
        """
        Generate a summary of difficulty metrics for a list of assessments.

        Args:
            metrics_list: List of DifficultyMetrics objects

        Returns:
            Dictionary with summary statistics
        """
        if not metrics_list:
            return {}

        # Calculate averages
        avg_complexity = sum(m.complexity_score for m in metrics_list) / len(metrics_list)
        avg_vocabulary = sum(m.vocabulary_density for m in metrics_list) / len(metrics_list)
        avg_depth = sum(m.concept_depth for m in metrics_list) / len(metrics_list)
        avg_reasoning = sum(m.reasoning_level for m in metrics_list) / len(metrics_list)
        avg_overall = sum(m.overall_difficulty for m in metrics_list) / len(metrics_list)
        avg_confidence = sum(m.confidence for m in metrics_list) / len(metrics_list)

        # Level distribution
        level_counts = {}
        for level in DifficultyLevel:
            level_counts[level.value] = sum(1 for m in metrics_list if m.level == level)

        return {
            'total_assessments': len(metrics_list),
            'averages': {
                'complexity_score': avg_complexity,
                'vocabulary_density': avg_vocabulary,
                'concept_depth': avg_depth,
                'reasoning_level': avg_reasoning,
                'overall_difficulty': avg_overall,
                'confidence': avg_confidence
            },
            'level_distribution': level_counts,
            'difficulty_range': {
                'min': min(m.overall_difficulty for m in metrics_list),
                'max': max(m.overall_difficulty for m in metrics_list)
            }
        }


# Factory function for backwards compatibility
def create_difficulty_assessment_engine() -> DifficultyAssessmentEngine:
    """Create and return a DifficultyAssessmentEngine instance."""
    return DifficultyAssessmentEngine()
