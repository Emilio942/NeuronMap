"""
Domain Specialization Framework for NeuronMap
===========================================

This module implements a modular domain-specific question generation system
with specialized knowledge bases, terminology databases, and expert validation.
"""

import logging
import json
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from collections import defaultdict, Counter

# Import difficulty assessment
from .difficulty_analyzer import (
    DifficultyAssessmentEngine, DifficultyMetrics, DifficultyLevel,
    assess_question_difficulty_fast
)

logger = logging.getLogger(__name__)


class DomainType(Enum):
    """Types of knowledge domains."""
    STEM = "stem"
    HUMANITIES = "humanities"
    SOCIAL_SCIENCES = "social_sciences"
    APPLIED_FIELDS = "applied_fields"
    INTERDISCIPLINARY = "interdisciplinary"


class QuestionType(Enum):
    """Types of domain-specific questions."""
    CONCEPTUAL = "conceptual"
    QUANTITATIVE = "quantitative"
    EXPERIMENTAL = "experimental"
    APPLICATION = "application"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    EVALUATIVE = "evaluative"
    CREATIVE = "creative"


@dataclass
class DomainVocabulary:
    """Domain-specific vocabulary and terminology."""
    core_terms: List[str] = field(default_factory=list)
    advanced_terms: List[str] = field(default_factory=list)
    methodology_terms: List[str] = field(default_factory=list)
    concept_hierarchies: Dict[str, List[str]] = field(default_factory=dict)
    synonyms: Dict[str, List[str]] = field(default_factory=dict)
    frequency_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class DomainComplexityScore:
    """Domain-specific complexity assessment."""
    terminology_density: float
    concept_depth: float
    methodological_rigor: float
    prerequisite_knowledge: float
    domain_specificity: float
    overall_score: float
    confidence: float


@dataclass
class DomainValidationResult:
    """Validation results for domain-specific questions."""
    is_domain_appropriate: bool
    terminology_score: float
    concept_alignment: float
    methodological_soundness: float
    expert_validation_score: Optional[float] = None
    suggestions: List[str] = field(default_factory=list)


@dataclass
class DomainQuestion:
    """A domain-specific question with metadata."""
    question: str
    domain: str
    subdomain: Optional[str]
    question_type: QuestionType
    difficulty_assessment: DifficultyMetrics
    domain_complexity: DomainComplexityScore
    validation_result: DomainValidationResult
    terminology_used: List[str]
    concepts_required: List[str]
    generated_timestamp: str
    source_context: Optional[str] = None


class DomainSpecialist(ABC):
    """Abstract base class for domain-specific question generators."""

    def __init__(self, domain_name: str, domain_type: DomainType):
        self.domain_name = domain_name
        self.domain_type = domain_type
        self.vocabulary = self._initialize_vocabulary()
        self.question_patterns = self._initialize_question_patterns()
        self.difficulty_engine = DifficultyAssessmentEngine(enable_bert=False)

        # Statistics
        self.generation_stats = {
            'total_generated': 0,
            'validation_passed': 0,
            'terminology_density_avg': 0.0,
            'complexity_scores': []
        }

    @abstractmethod
    def _initialize_vocabulary(self) -> DomainVocabulary:
        """Initialize domain-specific vocabulary."""
        pass

    @abstractmethod
    def _initialize_question_patterns(self) -> Dict[QuestionType, List[str]]:
        """Initialize question pattern templates for this domain."""
        pass

    @abstractmethod
    def generate_domain_questions(
            self,
            topic: str,
            count: int = 5,
            question_type: Optional[QuestionType] = None) -> List[DomainQuestion]:
        """Generate domain-specific questions for a given topic."""
        pass

    def validate_domain_specificity(self, question: str) -> DomainValidationResult:
        """Validate if a question is appropriate for this domain."""
        # Calculate terminology density
        words = question.lower().split()
        all_terms = (self.vocabulary.core_terms +
                     self.vocabulary.advanced_terms +
                     self.vocabulary.methodology_terms)

        terminology_count = sum(
            1 for word in words if word in [
                term.lower() for term in all_terms])
        terminology_density = terminology_count / len(words) if words else 0

        # Assess concept alignment
        concept_alignment = self._assess_concept_alignment(question)

        # Check methodological soundness
        methodological_soundness = self._assess_methodological_soundness(question)

        # Determine if domain appropriate
        is_appropriate = (terminology_density >= 0.15 and
                          concept_alignment >= 0.6 and
                          methodological_soundness >= 0.5)

        suggestions = []
        if terminology_density < 0.15:
            suggestions.append(
                f"Increase domain terminology (current: {
                    terminology_density:.2f}, target: >0.15)")
        if concept_alignment < 0.6:
            suggestions.append("Improve alignment with domain concepts")
        if methodological_soundness < 0.5:
            suggestions.append("Enhance methodological rigor")

        return DomainValidationResult(
            is_domain_appropriate=is_appropriate,
            terminology_score=terminology_density,
            concept_alignment=concept_alignment,
            methodological_soundness=methodological_soundness,
            suggestions=suggestions
        )

    def get_domain_vocabulary(self) -> DomainVocabulary:
        """Get the domain vocabulary."""
        return self.vocabulary

    def assess_domain_complexity(self, question: str) -> DomainComplexityScore:
        """Assess domain-specific complexity of a question."""
        # Terminology density
        words = question.lower().split()
        all_terms = (self.vocabulary.core_terms +
                     self.vocabulary.advanced_terms +
                     self.vocabulary.methodology_terms)

        terminology_density = sum(
            1 for word in words if word in [
                term.lower() for term in all_terms]) / len(words) if words else 0

        # Concept depth (based on advanced terminology usage)
        advanced_term_count = sum(
            1 for word in words if word in [
                term.lower() for term in self.vocabulary.advanced_terms])
        concept_depth = advanced_term_count / len(words) if words else 0

        # Methodological rigor
        method_terms = [
            "method",
            "methodology",
            "approach",
            "technique",
            "procedure",
            "protocol"]
        methodological_rigor = sum(
            1 for term in method_terms if term in question.lower()) / 10.0

        # Prerequisite knowledge (based on concept hierarchies)
        prerequisite_score = self._calculate_prerequisite_knowledge(question)

        # Domain specificity
        domain_specificity = min(terminology_density * 2, 1.0)

        # Overall score
        overall_score = np.mean([terminology_density,
                                 concept_depth,
                                 methodological_rigor,
                                 prerequisite_score,
                                 domain_specificity])

        return DomainComplexityScore(
            terminology_density=terminology_density,
            concept_depth=concept_depth,
            methodological_rigor=methodological_rigor,
            prerequisite_knowledge=prerequisite_score,
            domain_specificity=domain_specificity,
            overall_score=overall_score,
            confidence=0.8  # Default confidence
        )

    def _assess_concept_alignment(self, question: str) -> float:
        """Assess how well the question aligns with domain concepts."""
        question_lower = question.lower()

        # Check for core concepts
        core_alignment = sum(1 for term in self.vocabulary.core_terms if term.lower(
        ) in question_lower) / max(len(self.vocabulary.core_terms), 1)

        # Check for concept hierarchies
        hierarchy_alignment = 0
        for concept, subconcepts in self.vocabulary.concept_hierarchies.items():
            if concept.lower() in question_lower:
                hierarchy_alignment += 0.5
                hierarchy_alignment += sum(
                    0.3 for sub in subconcepts if sub.lower() in question_lower)

        return min((core_alignment + hierarchy_alignment) / 2, 1.0)

    def _assess_methodological_soundness(self, question: str) -> float:
        """Assess methodological soundness of the question."""
        question_lower = question.lower()

        # Check for methodology terms
        method_score = sum(1 for term in self.vocabulary.methodology_terms if term.lower(
        ) in question_lower) / max(len(self.vocabulary.methodology_terms), 1)

        # Check for reasoning indicators
        reasoning_indicators = [
            "analyze",
            "evaluate",
            "compare",
            "synthesize",
            "apply",
            "explain"]
        reasoning_score = sum(1 for indicator in reasoning_indicators
                              if indicator in question_lower) / len(reasoning_indicators)

        return min((method_score + reasoning_score) / 2, 1.0)

    def _calculate_prerequisite_knowledge(self, question: str) -> float:
        """Calculate the prerequisite knowledge score."""
        # Simplified calculation based on term complexity
        question_lower = question.lower()

        advanced_count = sum(1 for term in self.vocabulary.advanced_terms
                             if term.lower() in question_lower)
        total_terms = sum(
            1 for term in (
                self.vocabulary.core_terms +
                self.vocabulary.advanced_terms) if term.lower() in question_lower)

        if total_terms == 0:
            return 0.0

        return advanced_count / total_terms

    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get generation statistics for this domain specialist."""
        stats = self.generation_stats.copy()

        if stats['total_generated'] > 0:
            stats['validation_pass_rate'] = stats['validation_passed'] / \
                stats['total_generated']
        else:
            stats['validation_pass_rate'] = 0.0

        if stats['complexity_scores']:
            stats['avg_complexity'] = np.mean(stats['complexity_scores'])
            stats['complexity_std'] = np.std(stats['complexity_scores'])
        else:
            stats['avg_complexity'] = 0.0
            stats['complexity_std'] = 0.0

        return stats


# Convenience functions for loading domain specialists
def get_available_domains() -> List[str]:
    """Get list of available domain specialists."""
    return [
        'physics',
        'chemistry',
        'biology',
        'mathematics',
        'computer_science',
        'engineering',
        'history',
        'philosophy',
        'literature',
        'linguistics',
        'art_history',
        'psychology',
        'sociology',
        'political_science',
        'economics',
        'anthropology',
        'medicine',
        'law',
        'education',
        'business']


def create_domain_specialist(domain_name: str) -> Optional[DomainSpecialist]:
    """Factory function to create domain specialists."""
    # Import specific specialists here to avoid circular imports
    from .domain_specialists import (
        PhysicsSpecialist,
        ChemistrySpecialist,
        BiologySpecialist,
        MathematicsSpecialist,
        ComputerScienceSpecialist,
        EngineeringSpecialist,
        HistorySpecialist,
        PhilosophySpecialist,
        LiteratureSpecialist,
        LinguisticsSpecialist,
        ArtHistorySpecialist,
        PsychologySpecialist,
        SociologySpecialist,
        PoliticalScienceSpecialist,
        EconomicsSpecialist,
        AnthropologySpecialist,
        MedicineSpecialist,
        LawSpecialist,
        EducationSpecialist,
        BusinessSpecialist)

    specialist_map = {
        'physics': PhysicsSpecialist,
        'chemistry': ChemistrySpecialist,
        'biology': BiologySpecialist,
        'mathematics': MathematicsSpecialist,
        'computer_science': ComputerScienceSpecialist,
        'engineering': EngineeringSpecialist,
        'history': HistorySpecialist,
        'philosophy': PhilosophySpecialist,
        'literature': LiteratureSpecialist,
        'linguistics': LinguisticsSpecialist,
        'art_history': ArtHistorySpecialist,
        'psychology': PsychologySpecialist,
        'sociology': SociologySpecialist,
        'political_science': PoliticalScienceSpecialist,
        'economics': EconomicsSpecialist,
        'anthropology': AnthropologySpecialist,
        'medicine': MedicineSpecialist,
        'law': LawSpecialist,
        'education': EducationSpecialist,
        'business': BusinessSpecialist
    }

    specialist_class = specialist_map.get(domain_name.lower())
    if specialist_class:
        return specialist_class()
    else:
        logger.warning(f"Unknown domain: {domain_name}")
        return None
