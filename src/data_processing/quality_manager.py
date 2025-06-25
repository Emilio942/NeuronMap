"""Data quality management for NeuronMap."""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import hashlib
import time
import uuid

try:
    from ..utils.config import get_config
except ImportError:
    # Fallback for development
    get_config = lambda: type('Config', (), {'get_experiment_config': lambda x: {}})()


logger = logging.getLogger(__name__)


@dataclass
class QuestionMetadata:
    """Metadata for a generated question."""
    id: str
    text: str
    category: str
    difficulty: int
    language: str
    generation_time: float
    model_used: str
    prompt_template: str
    hash: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuestionMetadata':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ActivationMetadata:
    """Metadata for extracted activations."""
    question_id: str
    model_name: str
    layer_name: str
    extraction_time: float
    vector_size: int
    sparsity: float
    mean_activation: float
    std_activation: float
    max_activation: float
    min_activation: float
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActivationMetadata':
        """Create from dictionary."""
        return cls(**data)


class DataQualityManager:
    """Manage data quality and validation."""

    def __init__(self, config_name: str = "default"):
        """Initialize data quality manager.

        Args:
            config_name: Name of experiment configuration.
        """
        try:
            self.config = get_config()
            self.experiment_config = self.config.get_experiment_config(config_name)
        except:
            self.config = None
            self.experiment_config = {}

        self.quality_rules = self._load_quality_rules()

    def _load_quality_rules(self) -> Dict[str, Any]:
        """Load data quality rules from configuration."""
        return {
            'question_min_length': 10,
            'question_max_length': 1000,
            'allowed_languages': ['en', 'de', 'fr', 'es'],
            'min_difficulty': 1,
            'max_difficulty': 10,
            'max_duplicate_threshold': 0.9,
            'min_activation_variance': 1e-6,
            'max_sparsity': 0.99,
            'required_categories': ['factual', 'reasoning', 'creative', 'ethical']
        }

    def validate_question(self, question: str, metadata: Optional[QuestionMetadata] = None) -> Dict[str, Any]:
        """Validate a single question.

        Args:
            question: Question text to validate.
            metadata: Optional question metadata.

        Returns:
            Validation result with success status and issues.
        """
        issues = []

        # Length validation
        if len(question) < self.quality_rules['question_min_length']:
            issues.append(f"Question too short: {len(question)} < {self.quality_rules['question_min_length']}")

        if len(question) > self.quality_rules['question_max_length']:
            issues.append(f"Question too long: {len(question)} > {self.quality_rules['question_max_length']}")

        # Content validation
        if not question.strip():
            issues.append("Question is empty or only whitespace")

        if not question.endswith('?'):
            issues.append("Question doesn't end with question mark")

        # Metadata validation
        if metadata:
            if metadata.language not in self.quality_rules['allowed_languages']:
                issues.append(f"Unsupported language: {metadata.language}")

            if not (self.quality_rules['min_difficulty'] <= metadata.difficulty <= self.quality_rules['max_difficulty']):
                issues.append(f"Invalid difficulty: {metadata.difficulty}")

            if metadata.category not in self.quality_rules['required_categories']:
                issues.append(f"Unknown category: {metadata.category}")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'score': max(0, 1.0 - len(issues) * 0.1)
        }

    def detect_duplicates(self, questions: List[str], threshold: float = 0.9) -> List[Tuple[int, int, float]]:
        """Detect duplicate questions using text similarity.

        Args:
            questions: List of question texts.
            threshold: Similarity threshold for duplicates.

        Returns:
            List of (index1, index2, similarity) tuples for duplicates.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            logger.warning("scikit-learn not available, using simple duplicate detection")
            return self._simple_duplicate_detection(questions, threshold)

        if len(questions) < 2:
            return []

        # Vectorize questions
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        try:
            tfidf_matrix = vectorizer.fit_transform(questions)
        except ValueError:
            # Not enough features (very short texts)
            return []

        # Calculate similarities
        similarities = cosine_similarity(tfidf_matrix)

        # Find duplicates
        duplicates = []
        for i in range(len(questions)):
            for j in range(i + 1, len(questions)):
                similarity = similarities[i, j]
                if similarity >= threshold:
                    duplicates.append((i, j, similarity))

        return duplicates

    def _simple_duplicate_detection(self, questions: List[str], threshold: float) -> List[Tuple[int, int, float]]:
        """Simple duplicate detection using exact matches."""
        duplicates = []
        for i in range(len(questions)):
            for j in range(i + 1, len(questions)):
                if questions[i].lower().strip() == questions[j].lower().strip():
                    duplicates.append((i, j, 1.0))
        return duplicates

    def validate_activations(self, activations: np.ndarray,
                           metadata: Optional[ActivationMetadata] = None) -> Dict[str, Any]:
        """Validate activation data.

        Args:
            activations: Activation vector/matrix.
            metadata: Optional activation metadata.

        Returns:
            Validation result with success status and issues.
        """
        issues = []

        # Basic checks
        if activations.size == 0:
            issues.append("Empty activation array")
            return {'valid': False, 'issues': issues, 'score': 0.0}

        # Check for NaN/Inf values
        if np.any(np.isnan(activations)):
            issues.append("Contains NaN values")

        if np.any(np.isinf(activations)):
            issues.append("Contains infinite values")

        # Variance check
        variance = np.var(activations)
        if variance < self.quality_rules['min_activation_variance']:
            issues.append(f"Very low variance: {variance}")

        # Sparsity check
        sparsity = np.mean(activations == 0)
        if sparsity > self.quality_rules['max_sparsity']:
            issues.append(f"Very sparse activations: {sparsity:.3f}")

        # Range checks
        if np.all(activations == activations[0]):
            issues.append("All activation values are identical")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'score': max(0, 1.0 - len(issues) * 0.15),
            'stats': {
                'mean': float(np.mean(activations)),
                'std': float(np.std(activations)),
                'min': float(np.min(activations)),
                'max': float(np.max(activations)),
                'sparsity': float(sparsity),
                'variance': float(variance)
            }
        }
