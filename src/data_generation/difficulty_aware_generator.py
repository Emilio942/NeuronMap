"""
Enhanced Question Generator with Difficulty Assessment Integration
=================================================================

This module extends the existing question generation system with automatic
difficulty assessment and intelligent question optimization.
"""

import logging
import time
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import json

# Import difficulty assessment components
from .difficulty_analyzer import (
    DifficultyAssessmentEngine,
    DifficultyMetrics,
    DifficultyLevel as AssessedDifficultyLevel,
    ReasoningType,
    assess_question_difficulty,
    get_difficulty_summary)

# Import existing enhanced generator components
try:
    from .enhanced_question_generator import (
        EnhancedQuestionGenerator, QuestionMetadata, DifficultyLevel,
        QuestionCategory, QualityValidator
    )
except ImportError:
    # Fallback imports or definitions
    class DifficultyLevel(Enum):
        BEGINNER = "beginner"
        INTERMEDIATE = "intermediate"
        ADVANCED = "advanced"
        EXPERT = "expert"

    class QuestionCategory(Enum):
        FACTUAL = "factual"
        REASONING = "reasoning"
        ANALYTICAL = "analytical"
        CREATIVE = "creative"
        ETHICAL = "ethical"

logger = logging.getLogger(__name__)


@dataclass
class DifficultyAssessment:
    """Extended difficulty assessment with detailed metrics."""
    assessed_level: AssessedDifficultyLevel
    difficulty_score: float  # 1.0-10.0
    confidence: float        # 0.0-1.0
    reasoning_type: ReasoningType
    linguistic_complexity: float
    cognitive_load: float
    semantic_complexity: float
    domain_indicators: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'assessed_level': self.assessed_level.name,
            'difficulty_score': self.difficulty_score,
            'confidence': self.confidence,
            'reasoning_type': self.reasoning_type.value,
            'linguistic_complexity': self.linguistic_complexity,
            'cognitive_load': self.cognitive_load,
            'semantic_complexity': self.semantic_complexity,
            'domain_indicators': self.domain_indicators,
            'recommendations': self.recommendations
        }


@dataclass
class EnhancedQuestionMetadata:
    """Enhanced question metadata with difficulty assessment."""
    question_id: str
    category: QuestionCategory
    difficulty_level: DifficultyLevel  # Original target difficulty
    difficulty_assessment: DifficultyAssessment  # Assessed difficulty
    source_text: str
    generation_model: str
    generation_timestamp: str
    quality_score: float
    validation_passed: bool
    processing_time_ms: float
    topic_tags: List[str]
    complexity_factors: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            **asdict(self),
            'difficulty_assessment': self.difficulty_assessment.to_dict()
        }


class DifficultyTargetManager:
    """Manages difficulty targeting and validation."""

    def __init__(self):
        # Mapping from target difficulty to assessment range
        self.difficulty_mappings = {
            DifficultyLevel.BEGINNER: (1.0, 3.5),      # Very easy to basic
            DifficultyLevel.INTERMEDIATE: (3.5, 6.5),  # Basic to moderate high
            DifficultyLevel.ADVANCED: (6.5, 8.5),      # Challenging to hard
            DifficultyLevel.EXPERT: (8.5, 10.0)        # Very hard to expert
        }

        # Tolerance for difficulty matching
        self.tolerance = 1.0

    def validate_difficulty_match(self, target: DifficultyLevel,
                                  assessed: DifficultyAssessment) -> bool:
        """Check if assessed difficulty matches target."""
        target_range = self.difficulty_mappings[target]
        target_min, target_max = target_range

        # Check if score falls within range (with tolerance)
        return (target_min - self.tolerance <= assessed.difficulty_score <=
                target_max + self.tolerance)

    def get_difficulty_gap(self, target: DifficultyLevel,
                           assessed: DifficultyAssessment) -> float:
        """Calculate gap between target and assessed difficulty."""
        target_range = self.difficulty_mappings[target]
        target_center = (target_range[0] + target_range[1]) / 2

        return assessed.difficulty_score - target_center

    def suggest_adjustments(self, target: DifficultyLevel,
                            assessed: DifficultyAssessment) -> List[str]:
        """Suggest adjustments to reach target difficulty."""
        gap = self.get_difficulty_gap(target, assessed)
        suggestions = []

        if gap > 1.0:  # Too difficult
            suggestions.extend([
                "Simplify vocabulary and sentence structure",
                "Reduce reasoning steps or complexity",
                "Add more context or scaffolding",
                "Break complex concepts into smaller parts"
            ])
        elif gap < -1.0:  # Too easy
            suggestions.extend([
                "Add more sophisticated vocabulary",
                "Increase reasoning requirements",
                "Introduce abstract concepts",
                "Add analytical or evaluative components"
            ])

        # Add specific recommendations from assessment
        suggestions.extend(assessed.recommendations)

        return suggestions


class DifficultyAwareQuestionGenerator:
    """Question generator with integrated difficulty assessment and optimization."""

    def __init__(self, config_path: Optional[str] = None):
        # Initialize difficulty assessment engine
        self.difficulty_engine = DifficultyAssessmentEngine()
        self.difficulty_manager = DifficultyTargetManager()

        # Initialize base generator if available
        self.base_generator = None
        try:
            self.base_generator = EnhancedQuestionGenerator(config_path)
        except Exception as e:
            logger.warning(f"Base generator not available: {e}")

        # Configuration
        self.max_refinement_attempts = 3
        self.quality_threshold = 0.7

        # Statistics
        self.generation_stats = {
            'total_generated': 0,
            'difficulty_matches': 0,
            'refinement_attempts': 0,
            'quality_failures': 0
        }

    def generate_question_with_difficulty_control(self,
                                                  source_text: str,
                                                  target_difficulty: DifficultyLevel,
                                                  category: QuestionCategory,
                                                  context: Optional[str] = None) -> Optional[Tuple[str,
                                                                                                   EnhancedQuestionMetadata]]:
        """Generate a question with specific difficulty targeting."""

        start_time = time.time()

        # Initial question generation
        if self.base_generator:
            # Use enhanced generator if available
            initial_result = self.base_generator.generate_questions(
                source_text, count=1, difficulty=target_difficulty, category=category
            )
            if not initial_result:
                return None

            initial_question = initial_result[0]['question']
            quality_score = initial_result[0].get('quality_score', 0.5)
        else:
            # Fallback to simple generation
            initial_question = self._simple_question_generation(
                source_text, target_difficulty, category)
            quality_score = 0.5

        if not initial_question:
            return None

        # Assess difficulty
        difficulty_assessment = self._assess_question_difficulty(initial_question)

        # Check if difficulty matches target
        current_question = initial_question
        refinement_attempts = 0

        while (refinement_attempts < self.max_refinement_attempts and
               not self.difficulty_manager.validate_difficulty_match(target_difficulty, difficulty_assessment)):

            # Attempt to refine question
            refined_question = self._refine_question_difficulty(
                current_question, target_difficulty, difficulty_assessment
            )

            if refined_question and refined_question != current_question:
                current_question = refined_question
                difficulty_assessment = self._assess_question_difficulty(
                    current_question)
                refinement_attempts += 1
                self.generation_stats['refinement_attempts'] += 1
            else:
                break

        # Create enhanced metadata
        processing_time = (time.time() - start_time) * 1000

        metadata = EnhancedQuestionMetadata(
            question_id=self._generate_question_id(current_question),
            category=category,
            difficulty_level=target_difficulty,
            difficulty_assessment=difficulty_assessment,
            source_text=source_text[:500],  # Truncate for storage
            generation_model="difficulty-aware-generator",
            generation_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            quality_score=quality_score,
            validation_passed=self.difficulty_manager.validate_difficulty_match(
                target_difficulty, difficulty_assessment),
            processing_time_ms=processing_time,
            topic_tags=self._extract_topic_tags(source_text),
            complexity_factors=self._extract_complexity_factors(difficulty_assessment)
        )

        # Update statistics
        self.generation_stats['total_generated'] += 1
        if metadata.validation_passed:
            self.generation_stats['difficulty_matches'] += 1

        return current_question, metadata

    def _assess_question_difficulty(self, question: str) -> DifficultyAssessment:
        """Assess question difficulty using the difficulty engine."""
        try:
            metrics = self.difficulty_engine.assess_difficulty(question)

            return DifficultyAssessment(
                assessed_level=metrics.difficulty_level,
                difficulty_score=metrics.difficulty_score,
                confidence=metrics.confidence,
                reasoning_type=metrics.reasoning_type,
                linguistic_complexity=metrics.complexity_factors.get(
                    'linguistic_complexity',
                    0),
                cognitive_load=metrics.complexity_factors.get(
                    'cognitive_load',
                    0),
                semantic_complexity=metrics.complexity_factors.get(
                    'semantic_complexity',
                    0),
                domain_indicators=metrics.domain_indicators,
                recommendations=metrics.recommendations)
        except Exception as e:
            logger.error(f"Difficulty assessment failed: {e}")
            # Return default assessment
            return DifficultyAssessment(
                assessed_level=AssessedDifficultyLevel.MODERATE,
                difficulty_score=5.0,
                confidence=0.5,
                reasoning_type=ReasoningType.FACTUAL_RECALL,
                linguistic_complexity=5.0,
                cognitive_load=5.0,
                semantic_complexity=5.0,
                domain_indicators=[],
                recommendations=["Unable to assess difficulty"]
            )

    def _refine_question_difficulty(
            self,
            question: str,
            target_difficulty: DifficultyLevel,
            current_assessment: DifficultyAssessment) -> Optional[str]:
        """Attempt to refine question to match target difficulty."""

        gap = self.difficulty_manager.get_difficulty_gap(
            target_difficulty, current_assessment)

        if abs(gap) < 0.5:  # Close enough
            return question

        # Apply difficulty-based modifications
        if gap > 1.0:  # Too difficult - simplify
            refined = self._simplify_question(question, current_assessment)
        elif gap < -1.0:  # Too easy - complexify
            refined = self._complexify_question(question, current_assessment)
        else:
            return question

        return refined if refined else question

    def _simplify_question(
            self,
            question: str,
            assessment: DifficultyAssessment) -> Optional[str]:
        """Simplify a question to reduce difficulty."""
        modifications = []

        # Simplify vocabulary
        if assessment.linguistic_complexity > 6:
            # Replace complex words with simpler alternatives
            complex_replacements = {
                'analyze': 'look at',
                'evaluate': 'judge',
                'synthesize': 'combine',
                'demonstrate': 'show',
                'elaborate': 'explain more',
                'predominantly': 'mostly',
                'consequently': 'so',
                'furthermore': 'also'
            }

            modified = question
            for complex_word, simple_word in complex_replacements.items():
                modified = modified.replace(complex_word, simple_word)
            modifications.append(modified)

        # Simplify sentence structure
        if assessment.linguistic_complexity > 7:
            # Break complex sentences
            if ' and ' in question and len(question.split()) > 15:
                parts = question.split(' and ')
                if len(parts) == 2:
                    modifications.append(f"{parts[0].strip()}?")

        # Reduce reasoning requirements
        if assessment.cognitive_load > 6:
            # Add scaffolding
            if question.startswith('Why'):
                modifications.append(f"What causes {question[4:].lower()}")
            elif question.startswith('How'):
                modifications.append(f"What are the steps in {question[4:].lower()}")

        return modifications[0] if modifications else None

    def _complexify_question(
            self,
            question: str,
            assessment: DifficultyAssessment) -> Optional[str]:
        """Increase question complexity to raise difficulty."""
        modifications = []

        # Add analytical components
        if assessment.reasoning_type in [
                ReasoningType.FACTUAL_RECALL,
                ReasoningType.DEFINITIONAL]:
            if question.startswith('What'):
                modifications.append(f"Analyze why {question[5:].lower()}")
            elif question.startswith('Who'):
                modifications.append(f"Evaluate the impact of {question[4:].lower()}")

        # Add complexity through comparison
        if assessment.cognitive_load < 4:
            modifications.append(
                f"{question.rstrip('?')} and compare this with alternative approaches?")

        # Add theoretical framework requirement
        if 'theory' not in question.lower() and assessment.semantic_complexity < 5:
            modifications.append(
                f"{question.rstrip('?')} using relevant theoretical frameworks?")

        # Add evaluation component
        if assessment.reasoning_type not in [
                ReasoningType.EVALUATIVE,
                ReasoningType.CREATIVE]:
            modifications.append(
                f"{question.rstrip('?')} and critically evaluate the implications?")

        return modifications[0] if modifications else None

    def _simple_question_generation(self, source_text: str, difficulty: DifficultyLevel,
                                    category: QuestionCategory) -> Optional[str]:
        """Fallback simple question generation."""
        # Basic template-based generation
        templates = {
            QuestionCategory.FACTUAL: [
                "What is {concept}?",
                "Who discovered {concept}?",
                "When did {event} occur?"],
            QuestionCategory.REASONING: [
                "Why does {concept} occur?",
                "How does {concept} work?"],
            QuestionCategory.ANALYTICAL: ["Analyze the relationship between {concept1} and {concept2}"],
            QuestionCategory.CREATIVE: ["Design a solution for {problem}"]}

        category_templates = templates.get(
            category, templates[QuestionCategory.FACTUAL])

        # Extract key concepts (simplified)
        words = source_text.split()
        concepts = [word for word in words if len(word) > 5 and word.isalpha()]

        if concepts:
            import random
            template = random.choice(category_templates)
            concept = random.choice(concepts[:5])  # Use first 5 concepts
            return template.replace(
                '{concept}',
                concept).replace(
                '{concept1}',
                concept).replace(
                '{concept2}',
                concepts[1] if len(concepts) > 1 else concept)

        return "What is the main idea in this text?"

    def _generate_question_id(self, question: str) -> str:
        """Generate unique ID for question."""
        import hashlib
        return hashlib.md5(question.encode()).hexdigest()[:16]

    def _extract_topic_tags(self, text: str) -> List[str]:
        """Extract topic tags from source text."""
        # Simple keyword extraction
        words = text.lower().split()
        common_words = {
            'the',
            'a',
            'an',
            'and',
            'or',
            'but',
            'in',
            'on',
            'at',
            'to',
            'for',
            'of',
            'with',
            'by'}
        keywords = [word for word in words if len(
            word) > 4 and word not in common_words]

        # Return top 5 most frequent keywords
        from collections import Counter
        return [word for word, count in Counter(keywords).most_common(5)]

    def _extract_complexity_factors(
            self, assessment: DifficultyAssessment) -> Dict[str, float]:
        """Extract complexity factors from assessment."""
        return {
            'linguistic_complexity': assessment.linguistic_complexity,
            'cognitive_load': assessment.cognitive_load,
            'semantic_complexity': assessment.semantic_complexity,
            'difficulty_score': assessment.difficulty_score,
            'confidence': assessment.confidence
        }

    def batch_generate_with_difficulty_control(self,
                                               source_texts: List[str],
                                               target_difficulty: DifficultyLevel,
                                               category: QuestionCategory,
                                               count_per_text: int = 1) -> List[Tuple[str,
                                                                                      EnhancedQuestionMetadata]]:
        """Generate multiple questions with difficulty control."""
        results = []

        for source_text in source_texts:
            for _ in range(count_per_text):
                result = self.generate_question_with_difficulty_control(
                    source_text, target_difficulty, category
                )
                if result:
                    results.append(result)

        return results

    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get generation statistics and performance metrics."""
        stats = self.generation_stats.copy()

        if stats['total_generated'] > 0:
            stats['difficulty_match_rate'] = stats['difficulty_matches'] / \
                stats['total_generated']
            stats['average_refinements'] = stats['refinement_attempts'] / \
                stats['total_generated']
        else:
            stats['difficulty_match_rate'] = 0.0
            stats['average_refinements'] = 0.0

        return stats

    def export_questions_with_difficulty_analysis(
            self, questions_with_metadata: List[Tuple[str, EnhancedQuestionMetadata]], output_file: str) -> None:
        """Export questions with comprehensive difficulty analysis."""

        export_data = []
        for question, metadata in questions_with_metadata:
            export_data.append({
                'question': question,
                'metadata': metadata.to_dict(),
                'difficulty_summary': get_difficulty_summary(metadata.difficulty_assessment)
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Exported {
                len(export_data)} questions with difficulty analysis to {output_file}")


# Convenience functions
def generate_question_with_target_difficulty(source_text: str,
                                             target_difficulty: DifficultyLevel,
                                             category: QuestionCategory = QuestionCategory.FACTUAL) -> Optional[Tuple[str,
                                                                                                                      Dict[str,
                                                                                                                           Any]]]:
    """Convenience function to generate a single question with target difficulty."""
    generator = DifficultyAwareQuestionGenerator()
    result = generator.generate_question_with_difficulty_control(
        source_text, target_difficulty, category)

    if result:
        question, metadata = result
        return question, metadata.to_dict()

    return None


def assess_existing_questions(questions: List[str]) -> List[Dict[str, Any]]:
    """Assess difficulty of existing questions."""
    assessments = []

    for question in questions:
        try:
            metrics = assess_question_difficulty(question)
            summary = get_difficulty_summary(metrics)
            assessments.append({
                'question': question,
                'assessment': summary
            })
        except Exception as e:
            logger.error(f"Failed to assess question '{question}': {e}")
            assessments.append({
                'question': question,
                'assessment': {'error': str(e)}
            })

    return assessments
