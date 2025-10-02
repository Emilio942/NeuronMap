"""Data quality management for NeuronMap."""

from __future__ import annotations

import logging
import time
import uuid
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - optional dependency path
    from ..utils.config import get_config
except ImportError:  # pragma: no cover - fallback for tests
    def get_config():  # type: ignore
        return type('Config', (), {'get_experiment_config': lambda *_args, **_kwargs: {}})()


logger = logging.getLogger(__name__)


DEFAULT_QUALITY_CONFIG: Dict[str, Any] = {
    'min_length': 10,
    'max_length': 1000,
    'duplicate_threshold': 0.9,
    'similarity_threshold': 0.75,
    'required_categories': ['factual', 'reasoning', 'creative', 'ethical'],
    'allowed_languages': ['en', 'de', 'fr', 'es'],
    'min_difficulty': 1,
    'max_difficulty': 10,
}


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
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuestionMetadata":
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
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActivationMetadata":
        return cls(**data)


class DataQualityManager:
    """Manage question and activation quality checks."""

    def __init__(self, config: Optional[Union[str, Dict[str, Any]]] = None):
        self.config = self._resolve_config(config)
        self.quality_rules = self._load_quality_rules()

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _resolve_config(self, config: Optional[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Return a runtime configuration dictionary.

        Supports three call styles accepted by the historic API:

        1. ``None`` → use defaults
        2. ``dict`` → merge user values into defaults
        3. ``str`` → treat as experiment name and resolve via ``get_config``
        """

        merged = dict(DEFAULT_QUALITY_CONFIG)

        if isinstance(config, dict):
            merged.update(config)
            return merged

        if isinstance(config, str) and config:
            try:  # pragma: no cover - depends on project config structure
                experiment_cfg = get_config().get_experiment_config(config)
                quality_cfg = experiment_cfg.get('quality', {})
                if isinstance(quality_cfg, dict):
                    merged.update(quality_cfg)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Failed to resolve experiment config '%s': %s", config, exc)
            return merged

        if config is None:
            return merged

        # Legacy behaviour: the original signature treated the argument as a
        # config name – if something unknown is provided we still return the
        # defaults to avoid surprising crashes.
        logger.debug("Unsupported config argument %r – falling back to defaults", config)
        return merged

    def _load_quality_rules(self) -> Dict[str, Any]:
        return {
            'question_min_length': int(self.config.get('min_length', 10)),
            'question_max_length': int(self.config.get('max_length', 1000)),
            'allowed_languages': self.config.get('allowed_languages', DEFAULT_QUALITY_CONFIG['allowed_languages']),
            'min_difficulty': self.config.get('min_difficulty', DEFAULT_QUALITY_CONFIG['min_difficulty']),
            'max_difficulty': self.config.get('max_difficulty', DEFAULT_QUALITY_CONFIG['max_difficulty']),
            'max_duplicate_threshold': float(self.config.get('duplicate_threshold', 0.9)),
            'similarity_threshold': float(self.config.get('similarity_threshold', 0.75)),
            'min_activation_variance': float(self.config.get('min_activation_variance', 1e-6)),
            'max_sparsity': float(self.config.get('max_sparsity', 0.99)),
            'required_categories': list(self.config.get('required_categories', DEFAULT_QUALITY_CONFIG['required_categories'])),
        }

    # ------------------------------------------------------------------
    # Question validation utilities
    # ------------------------------------------------------------------
    def validate_question(
        self,
        question: Union[str, Dict[str, Any]],
        metadata: Optional[QuestionMetadata] = None,
    ) -> Dict[str, Any]:
        """Validate a question payload and return diagnostic information."""

        question_text = self._extract_text(question)
        issues: List[str] = []

        if not question_text or not question_text.strip():
            issues.append("Question is empty or only whitespace")
        else:
            length = len(question_text)
            if length < self.quality_rules['question_min_length']:
                issues.append(
                    f"Question too short: {length} < {self.quality_rules['question_min_length']}"
                )
            if length > self.quality_rules['question_max_length']:
                issues.append(
                    f"Question too long: {length} > {self.quality_rules['question_max_length']}"
                )

            if not question_text.endswith('?'):
                issues.append("Question doesn't end with question mark")

        # Build metadata from dict payload if not supplied explicitly
        if metadata is None and isinstance(question, dict):
            metadata = self._metadata_from_question(question)

        if metadata is not None:
            if metadata.language not in self.quality_rules['allowed_languages']:
                issues.append(f"Unsupported language: {metadata.language}")
            if not (self.quality_rules['min_difficulty'] <= metadata.difficulty <= self.quality_rules['max_difficulty']):
                issues.append(f"Invalid difficulty: {metadata.difficulty}")
            if metadata.category not in self.quality_rules['required_categories']:
                issues.append(f"Unknown category: {metadata.category}")

        score = max(0.0, 1.0 - len(issues) * 0.1)
        return {'valid': len(issues) == 0, 'issues': issues, 'score': score}

    def is_valid_question(self, question: Union[str, Dict[str, Any]]) -> bool:
        return self.validate_question(question)['valid']

    def find_similar_questions(
        self,
        questions: Sequence[Union[str, Dict[str, Any]]],
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return high-similarity question pairs."""

        threshold = threshold or self.quality_rules['similarity_threshold']
        texts, mapping = self._normalise_questions(questions)
        duplicate_pairs = self._detect_similarity_pairs(texts, threshold)

        results: List[Dict[str, Any]] = []
        for i, j, score in duplicate_pairs:
            q1 = mapping[i]
            q2 = mapping[j]
            results.append({
                'question_1': q1,
                'question_2': q2,
                'similarity_score': float(score),
                'text': self._extract_text(q1),
            })
        return results

    def detect_duplicates(
        self,
        questions: Sequence[Union[str, Dict[str, Any]]],
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Find likely duplicate question entries."""

        threshold = threshold or self.quality_rules['max_duplicate_threshold']
        texts, mapping = self._normalise_questions(questions)
        duplicate_pairs = self._detect_similarity_pairs(texts, threshold)

        seen_keys = set()
        duplicates: List[Dict[str, Any]] = []
        for i, j, score in duplicate_pairs:
            # Use the second index as the canonical duplicate so that we do not
            # repeat the first occurrence multiple times.
            key = (j, self._extract_text(mapping[j]))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            duplicates.append({
                'question_1': mapping[i],
                'question_2': mapping[j],
                'similarity': float(score),
                'text': self._extract_text(mapping[j]),
            })
        return duplicates

    def _detect_similarity_pairs(
        self,
        texts: Sequence[str],
        threshold: float,
    ) -> List[Tuple[int, int, float]]:
        if len(texts) < 2:
            return []

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:  # pragma: no cover - optional dependency
            return self._simple_duplicate_detection(texts, threshold)

        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
        except ValueError:
            return []

        similarities = cosine_similarity(tfidf_matrix)
        pairs: List[Tuple[int, int, float]] = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = float(similarities[i, j])
                if sim >= threshold:
                    pairs.append((i, j, sim))
        return pairs

    def _simple_duplicate_detection(
        self,
        texts: Sequence[str],
        _threshold: float,
    ) -> List[Tuple[int, int, float]]:
        pairs: List[Tuple[int, int, float]] = []
        lowered = [text.lower().strip() for text in texts]
        for i in range(len(lowered)):
            for j in range(i + 1, len(lowered)):
                if lowered[i] == lowered[j]:
                    pairs.append((i, j, 1.0))
        return pairs

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------
    def calculate_quality_metrics(
        self,
        questions: Sequence[Union[str, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        texts = [self._extract_text(q) for q in questions]
        valid_flags = [self.is_valid_question(q) for q in questions]
        duplicates = self.detect_duplicates(questions)

        total = len(questions)
        valid_count = sum(valid_flags)
        invalid_count = total - valid_count
        average_length = float(np.mean([len(t) for t in texts])) if texts else 0.0

        quality_score = 0.0
        if total:
            quality_score = (valid_count / total) * 100.0
            quality_score -= min(len(duplicates) * 5.0, 25.0)
            quality_score = max(0.0, min(100.0, quality_score))

        return {
            'total_questions': total,
            'valid_questions': valid_count,
            'invalid_questions': invalid_count,
            'duplicate_count': len(duplicates),
            'average_length': average_length,
            'quality_score': round(quality_score, 2),
            'duplicates': duplicates,
        }

    def validate_questions_data(
        self,
        questions: Sequence[Union[str, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Validate a collection of questions and return structured errors.

        The legacy test-suite expects a list of error dictionaries containing at
        least a ``type`` key describing the problem.  Additional metadata (such
        as the question index or the offending text) is included to aid
        debugging while remaining backwards compatible.
        """

        if questions is None:
            raise ValueError("questions must not be None")

        errors: List[Dict[str, Any]] = []
        seen_questions: Dict[str, int] = {}

        for index, question in enumerate(questions):
            question_text = self._extract_text(question)
            normalised = question_text.strip().lower()

            if normalised:
                if normalised in seen_questions:
                    errors.append({
                        'index': index,
                        'type': 'duplicate',
                        'message': 'Duplicate question detected',
                        'question': question_text,
                        'duplicate_of': seen_questions[normalised],
                    })
                else:
                    seen_questions[normalised] = index

            validation = self.validate_question(question)
            if not validation['valid']:
                for issue in validation['issues']:
                    errors.append({
                        'index': index,
                        'type': self._classify_issue_message(issue),
                        'message': issue,
                        'question': question_text,
                    })

        # De-duplicate error entries by (index, type, message) to avoid
        # reporting the same issue multiple times when it is detected through
        # different code paths.
        unique_errors = []
        seen_keys = set()
        for error in errors:
            key = (error.get('index'), error.get('type'), error.get('message'))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_errors.append(error)

        return unique_errors

    def clean_data(
        self,
        questions: Sequence[Union[str, Dict[str, Any]]],
    ) -> List[Union[str, Dict[str, Any]]]:
        seen_hashes: set[str] = set()
        cleaned: List[Union[str, Dict[str, Any]]] = []

        for question in questions:
            if not self.is_valid_question(question):
                continue

            question_text = self._extract_text(question)
            question_hash = hashlib.sha256(question_text.lower().strip().encode('utf-8')).hexdigest()
            if question_hash in seen_hashes:
                continue
            seen_hashes.add(question_hash)
            cleaned.append(question)
        return cleaned

    # ------------------------------------------------------------------
    # Activation validation
    # ------------------------------------------------------------------
    def validate_activations(
        self,
        activations: np.ndarray,
        metadata: Optional[ActivationMetadata] = None,
    ) -> Dict[str, Any]:
        issues: List[str] = []

        if activations.size == 0:
            issues.append("Empty activation array")
            return {'valid': False, 'issues': issues, 'score': 0.0}

        if np.any(np.isnan(activations)):
            issues.append("Contains NaN values")
        if np.any(np.isinf(activations)):
            issues.append("Contains infinite values")

        variance = float(np.var(activations))
        sparsity = float(np.mean(activations == 0))

        if variance < self.quality_rules['min_activation_variance']:
            issues.append(f"Very low variance: {variance}")
        if sparsity > self.quality_rules['max_sparsity']:
            issues.append(f"Very sparse activations: {sparsity:.3f}")
        if np.all(activations == activations.flat[0]):
            issues.append("All activation values are identical")

        stats = {
            'mean': float(np.mean(activations)),
            'std': float(np.std(activations)),
            'min': float(np.min(activations)),
            'max': float(np.max(activations)),
            'sparsity': sparsity,
            'variance': variance,
        }

        score = max(0.0, 1.0 - len(issues) * 0.15)
        return {'valid': len(issues) == 0, 'issues': issues, 'score': score, 'stats': stats}

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _normalise_questions(
        self,
        questions: Sequence[Union[str, Dict[str, Any]]],
    ) -> Tuple[List[str], List[Union[str, Dict[str, Any]]]]:
        texts: List[str] = []
        mapping: List[Union[str, Dict[str, Any]]] = []
        for question in questions:
            text = self._extract_text(question)
            if not text:
                continue
            texts.append(text)
            mapping.append(question)
        return texts, mapping

    def _extract_text(self, question: Union[str, Dict[str, Any]]) -> str:
        if isinstance(question, str):
            return question
        if isinstance(question, dict):
            for key in ('text', 'question', 'prompt', 'input'):
                value = question.get(key)
                if isinstance(value, str):
                    return value
        return ""

    def _classify_issue_message(self, issue: str) -> str:
        lowered = issue.lower()
        if 'empty' in lowered:
            return 'empty_text'
        if 'too short' in lowered:
            return 'too_short'
        if 'too long' in lowered:
            return 'too_long'
        if 'question mark' in lowered:
            return 'missing_question_mark'
        if 'language' in lowered:
            return 'invalid_language'
        if 'difficulty' in lowered:
            return 'invalid_difficulty'
        if 'category' in lowered:
            return 'invalid_category'
        if 'variance' in lowered:
            return 'low_variance'
        if 'sparse' in lowered:
            return 'high_sparsity'
        return 'validation_issue'

    def _metadata_from_question(self, question: Dict[str, Any]) -> QuestionMetadata:
        question_text = self._extract_text(question)
        return QuestionMetadata(
            id=str(question.get('id', uuid.uuid4())),
            text=question_text,
            category=str(question.get('category', 'factual')),
            difficulty=int(question.get('difficulty', 5)),
            language=str(question.get('language', 'en')),
            generation_time=float(question.get('generation_time', 0.0)),
            model_used=str(question.get('model_used', 'unknown')),
            prompt_template=str(question.get('prompt_template', '')),
            hash=hashlib.sha256(question_text.encode('utf-8')).hexdigest() if question_text else '',
            created_at=str(question.get('created_at', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))),
        )
