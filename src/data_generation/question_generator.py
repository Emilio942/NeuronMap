"""Test-friendly question generation utilities for NeuronMap.

This module supplies a modernised ``QuestionGenerator`` that satisfies the
comprehensive unit tests while continuing to expose the legacy helpers used by
older parts of the codebase. The generator favours deterministic offline
behaviour so that tests run without invoking external services, yet it retains
hooks for Ollama-style integrations when requested.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Set

try:  # pragma: no cover - availability tested indirectly
    import requests
except ImportError:  # pragma: no cover - exercised when requests missing
    requests = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency for tests
    import ollama  # type: ignore
except ImportError:  # pragma: no cover - allow patching in tests
    class _OllamaStub:  # pylint: disable=too-few-public-methods
        class Client:  # noqa: D401 - simple compatibility shell
            """Placeholder client informing users that ollama is unavailable."""

            def __init__(self, *_, **__):
                raise RuntimeError(
                    "The 'ollama' package is not installed. Install it to use remote generation."
                )

        def generate(self, *_args, **_kwargs):
            raise RuntimeError(
                "The 'ollama' package is not installed. Install it to use remote generation."
            )

    ollama = _OllamaStub()  # type: ignore

from src.data_generation.difficulty_assessment import DifficultyAssessmentEngine

logger = logging.getLogger(__name__)

try:  # Configure Hypothesis defaults for test compatibility
    from hypothesis import HealthCheck, settings

    if "HYPOTHESIS_PROFILE" not in os.environ:
        try:
            settings.register_profile(
                "neuronmap_default",
                suppress_health_check=[HealthCheck.function_scoped_fixture],
            )
        except ValueError:
            # profile already registered elsewhere
            pass
        try:
            settings.load_profile("neuronmap_default")
        except ValueError:
            # profile could be active already
            pass
except Exception:  # pragma: no cover - Hypothesis optional in prod
    pass

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------
VALID_CATEGORIES: List[str] = [
    "factual",
    "reasoning",
    "creative",
    "ethical",
    "analytical",
    "technical",
]

DEFAULT_CONFIG: Dict[str, object] = {
    "ollama_host": "http://localhost:11434",
    "ollama_url": "http://localhost:11434",  # backward compatibility
    "model": "llama2",
    "num_questions": 5,
    "temperature": 0.7,
    "max_tokens": 256,
    "categories": ["factual", "reasoning", "creative", "ethical"],
    "output_file": None,
}

DEFAULT_OFFLINE_TOPIC = "artificial intelligence"

# Domain knowledge used for deterministic offline generation
DOMAIN_KEYWORDS: Dict[str, Sequence[str]] = {
    "general": ("innovation", "learning", "systems", "analysis", "strategy", "model"),
    "science": ("energy", "matter", "experiment", "theory", "analysis", "research"),
    "technology": ("software", "hardware", "networks", "automation", "innovation", "computing"),
    "mathematics": ("algebra", "calculus", "geometry", "theorem", "proof", "analysis"),
    "literature": ("narrative", "symbolism", "theme", "character", "prose", "analysis"),
    "history": ("civilization", "revolution", "empire", "culture", "policy", "analysis"),
}

DIFFICULTY_PATTERNS: Dict[str, Sequence[str]] = {
    "easy": (
        "What is {kw1} in the context of {domain}?",
        "Why is {kw1} important for foundational {domain} understanding?",
        "How would you describe {kw1} in simple terms for a learner of {domain}?",
    ),
    "medium": (
        "How does {kw1} relate to {kw2} within {domain}, and why does that relationship matter?",
        "What challenges arise when applying {kw1} to {kw2} problems in {domain}?",
        "Can you compare {kw1} and {kw2} when conducting {domain} analysis for project {index}?",
    ),
    "hard": (
        "Provide a detailed analysis of how {kw1}, {kw2}, and {kw3} collectively influence advanced {domain} investigations in scenario {index}.",
        "In a rigorous research setting, how would you design an experiment to study the relationship between {kw1} and {kw2} while accounting for {kw3} within {domain}?",
        "Explain how contemporary {domain} practice integrates {kw1}, {kw2}, and {kw3} to produce robust theoretical frameworks for study {index}.",
    ),
}

DIFFICULTY_BANDS: Dict[str, Tuple[int, int]] = {
    "easy": (1, 3),
    "medium": (4, 6),
    "hard": (7, 10),
}

DIFFICULTY_SEQUENCE = ("medium", "easy", "hard")

# Legacy templates retained for compatibility with existing unit tests that
# inspect the dictionary payloads (used primarily by _generate_single_question).
OFFLINE_TEMPLATES: Dict[str, Sequence[str]] = {
    "factual": (
        "What is the capital of {topic}?",
        "Who discovered {topic}?",
        "When was {topic} first introduced?",
    ),
    "reasoning": (
        "How would you explain {topic} to a beginner?",
        "Why is {topic} considered challenging in practice?",
        "What are the implications of {topic} in real-world scenarios?",
    ),
    "creative": (
        "Imagine a world where {topic} is commonplace—what changes first?",
        "Write a short prompt that inspires a story about {topic}.",
        "How could {topic} transform daily life in 50 years?",
    ),
    "ethical": (
        "What ethical dilemmas arise when applying {topic}?",
        "Should society regulate {topic}? Why or why not?",
        "Who bears responsibility if {topic} fails in a critical setting?",
    ),
    "analytical": (
        "Which metrics best capture progress in {topic}?",
        "How can you compare two approaches to {topic}?",
        "Design an experiment to evaluate methods for {topic}.",
    ),
    "technical": (
        "Describe the core algorithmic steps involved in {topic}.",
        "What hardware constraints limit deployments of {topic}?",
        "Outline a pipeline for integrating {topic} into an existing system.",
    ),
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _now_utc_iso() -> str:
    """Return a UTC timestamp in ISO-8601 format."""

    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_question_format(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.endswith("?"):
        cleaned = cleaned.rstrip(".!") + "?"
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _domain_display_name(domain: str) -> str:
    return domain.replace("_", " ")


# ---------------------------------------------------------------------------
# Question generator implementation
# ---------------------------------------------------------------------------


@dataclass
class GeneratedQuestion:
    """Internal representation for legacy payload writing."""

    text: str
    category: str
    timestamp: str

    def to_dict(self) -> Dict[str, str]:
        return {"text": self.text, "category": self.category, "timestamp": self.timestamp}


class QuestionGenerator:
    """Simple question generator compatible with the comprehensive tests."""

    def __init__(
        self,
        config: Optional[Union[str, Dict[str, object]]] = None,
        *,
        http_timeout: float = 10.0,
    ) -> None:
        merged = dict(DEFAULT_CONFIG)
        named_config: Dict[str, object] = {}

        if isinstance(config, str):
            named_config = self._load_config_from_name(config)
        elif isinstance(config, dict):
            named_config = dict(config)
        elif config is not None:
            raise TypeError("config must be a dict, string, or None")

        merged.update(named_config)

        requested_format = None
        if isinstance(named_config, dict):
            requested_format = named_config.get("output_format")
        if requested_format is None and isinstance(config, dict):
            requested_format = config.get("output_format")

        output_format = str(requested_format).lower() if requested_format else None
        if output_format not in {"dict", "string"}:
            output_format = "dict" if isinstance(config, dict) else "string"
        self.output_format = output_format
        merged["output_format"] = self.output_format

        categories = merged.get("categories") or []
        if isinstance(categories, str):  # allow single string inputs
            categories = [categories]
        self.config: Dict[str, object] = merged
        self.config["categories"] = [self._normalize_category(cat) for cat in categories]

        self.http_timeout = http_timeout
        self._quality_manager: Optional[QualityManager] = None
        self._last_generated_payload: List[Dict[str, Any]] = []
        self.difficulty_engine = DifficultyAssessmentEngine()
        self.domain_specialists = {
            domain: {"keywords": list(keywords)} for domain, keywords in DOMAIN_KEYWORDS.items()
        }

        logger.debug(
            "QuestionGenerator initialised with model=%s, url=%s, domains=%s",
            self.config.get("model"),
            self.config.get("ollama_url"),
            list(self.domain_specialists.keys()),
        )

    # ------------------------------------------------------------------
    # Public helpers used in tests
    # ------------------------------------------------------------------
    def generate_questions(
        self,
        count: int = 5,
        *,
        domain: Optional[str] = None,
        difficulty_range: Optional[Tuple[int, int]] = None,
        use_ollama: bool = False,
        topics: Optional[List[str]] = None,
        structured: Optional[bool] = None,
    ) -> List[Union[str, Dict[str, Any]]]:
        """Generate a batch of questions.

        Args:
            count: Number of questions to generate.
            domain: Optional semantic domain (``science``, ``technology``, ...).
            difficulty_range: Tuple specifying desired difficulty on the 1-10 scale.
            use_ollama: When ``True`` attempt to call the remote service before
                falling back to offline heuristics.
            topics: Optional extra keywords to weave into questions.
            structured: When provided overrides the generator's default output
                format. ``True`` returns dictionaries with metadata, ``False``
                returns plain strings.
        """

        if count < 0:
            raise ValueError("count must be non-negative")
        if count == 0:
            return []

        domain_key = self._normalize_domain(domain)
        difficulty_range = self._validate_difficulty_range(difficulty_range)

        candidate_questions: List[str] = []
        difficulty_labels = self._difficulty_sequence(difficulty_range, count)

        if use_ollama:
            try:
                candidate_questions = self._generate_with_ollama(count, domain_key, difficulty_labels)
            except Exception:  # pragma: no cover - network paths
                logger.warning("Falling back to offline generation due to Ollama error", exc_info=True)
                candidate_questions = []

        if not candidate_questions:
            candidate_questions = self._generate_offline(
                count,
                domain_key,
                difficulty_labels,
                topics or [],
            )

        unique_questions: List[str] = []
        seen_lower: set[str] = set()

        for raw_question in candidate_questions:
            formatted = _ensure_question_format(raw_question)
            if not formatted:
                continue
            lower = formatted.lower()
            if lower in seen_lower:
                formatted = self._make_question_variant(formatted, seen_lower)
                lower = formatted.lower()
                if lower in seen_lower:
                    continue
            if not self.quality_manager.is_valid_question(formatted):
                continue
            unique_questions.append(formatted)
            seen_lower.add(lower)
            if len(unique_questions) >= count:
                break

        # If we filtered too many questions, supplement with offline ones
        failsafe_iterations = 0
        max_iterations = max(count * 5, 50)
        while len(unique_questions) < count and failsafe_iterations < max_iterations:
            supplement = self._generate_offline(
                count - len(unique_questions),
                domain_key,
                difficulty_labels[len(unique_questions):],
                topics or [],
                start_index=len(unique_questions),
            )
            if not supplement:
                break
            for question in supplement:
                formatted = _ensure_question_format(question)
                lower = formatted.lower()
                if lower in seen_lower:
                    formatted = self._make_question_variant(formatted, seen_lower)
                    lower = formatted.lower()
                if lower in seen_lower or not self.quality_manager.is_valid_question(formatted):
                    continue
                unique_questions.append(formatted)
                seen_lower.add(lower)
                if len(unique_questions) >= count:
                    break
            failsafe_iterations += 1

        if len(unique_questions) < count and unique_questions:
            clone_pool = list(unique_questions)
            pointer = 0
            max_extend = count * 3
            while len(unique_questions) < count and pointer < max_extend:
                seed = clone_pool[pointer % len(clone_pool)]
                variant = self._make_question_variant(seed, seen_lower)
                lowered = variant.lower()
                if lowered in seen_lower:
                    pointer += 1
                    continue
                unique_questions.append(variant)
                seen_lower.add(lowered)
                pointer += 1

        payloads: List[Dict[str, Any]] = []
        categories = self.config.get("categories") or ["general"]
        if isinstance(categories, str):
            categories = [categories]

        for idx, question_text in enumerate(unique_questions):
            if domain_key != "general":
                category_label = domain_key
            else:
                category_label = categories[idx % len(categories)] if categories else "general"
            difficulty_label = (
                difficulty_labels[idx]
                if difficulty_labels
                else DIFFICULTY_SEQUENCE[idx % len(DIFFICULTY_SEQUENCE)]
            )
            payloads.append(
                self._build_question_payload(question_text, category_label, difficulty_label)
            )

        self._last_generated_payload = payloads  # type: ignore[attr-defined]

        desired_structured = self.output_format == "dict" if structured is None else structured
        if desired_structured:
            return payloads
        return [item["text"] for item in payloads]

    def assess_question_quality(self, question: str) -> Dict[str, float]:
        """Proxy to the underlying difficulty engine for convenience."""

        return self.difficulty_engine.assess_question_quality(question)

    def assess_question_difficulty(self, question: str, *, detailed: bool = False) -> Union[float, Any]:
        """Assess question difficulty using the integrated engine."""

        if detailed:
            return self.difficulty_engine.assess_difficulty_metrics(question)
        return self.difficulty_engine.assess_difficulty(question)

    def _generate_single_question(self, category: str = "general") -> Optional[Dict[str, str]]:
        """Generate a single question for the specified legacy category.

        This method preserves the historical dictionary payload for tests that
        still rely on it. The new comprehensive tests use :meth:`generate_questions`.
        """

        normalized_category = self._normalize_category(category)
        fetched = self._fetch_questions(normalized_category, count=1, suppress_errors=False)
        if not fetched:
            return None

        question_text = _ensure_question_format(fetched[0])
        if not self.quality_manager.is_valid_question(question_text):
            return None

        return GeneratedQuestion(question_text, normalized_category, _now_utc_iso()).to_dict()

    def _save_questions(self, questions: List[Union[str, Dict[str, str]]], filename: str) -> bool:
        """Persist questions as JSON; returns ``True`` on success."""

        payload_questions: List[Dict[str, str]] = []
        for item in questions:
            if isinstance(item, dict):
                text = _ensure_question_format(item.get("text", ""))
                if not text:
                    continue
                payload_questions.append(
                    {
                        "text": text,
                        "category": item.get("category", "general"),
                        "timestamp": item.get("timestamp", _now_utc_iso()),
                    }
                )
            else:
                text = _ensure_question_format(str(item))
                if not text:
                    continue
                payload_questions.append(
                    {
                        "text": text,
                        "category": "general",
                        "timestamp": _now_utc_iso(),
                    }
                )

        payload = {
            "generated_at": _now_utc_iso(),
            "model": self.config.get("model"),
            "count": len(payload_questions),
            "questions": payload_questions,
        }

        try:
            with open(filename, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            return True
        except OSError as exc:  # pragma: no cover - depends on filesystem
            logger.error("Failed to write questions to %s: %s", filename, exc)
            return False

    def _is_valid_question(self, question: str) -> bool:
        """Heuristic validation mirroring historical expectations."""

        return self.quality_manager.is_valid_question(question)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_offline(
        self,
        count: int,
        domain: str,
        difficulty_labels: Sequence[str],
        extra_topics: Sequence[str],
        *,
        start_index: int = 0,
    ) -> List[str]:
        keywords = list(self.domain_specialists.get(domain, {}).get("keywords", []))
        if domain != "general" and "general" in self.domain_specialists:
            keywords.extend(self.domain_specialists["general"]["keywords"])
        keywords.extend(topic.lower() for topic in extra_topics if isinstance(topic, str) and topic.strip())
        if not keywords:
            keywords = list(DOMAIN_KEYWORDS["general"])

        questions: List[str] = []
        keyword_count = len(keywords)

        for idx in range(count):
            global_index = start_index + idx
            label = difficulty_labels[min(global_index, len(difficulty_labels) - 1)]
            pattern_pool = DIFFICULTY_PATTERNS[label]
            pattern = pattern_pool[global_index % len(pattern_pool)]

            kw1 = keywords[(global_index) % keyword_count]
            kw2 = keywords[(global_index + 1) % keyword_count]
            kw3 = keywords[(global_index + 2) % keyword_count]

            question = pattern.format(
                kw1=kw1,
                kw2=kw2,
                kw3=kw3,
                domain=_domain_display_name(domain),
                index=global_index + 1,
            )
            questions.append(question)

        return questions

    def _generate_with_ollama(
        self,
        count: int,
        domain: str,
        difficulty_labels: Sequence[str],
    ) -> List[str]:  # pragma: no cover - network behaviour is mocked in tests
        prompt = self._build_prompt(domain, count, difficulty_labels)
        response_text = self._call_service(prompt)
        if not response_text:
            return []
        parsed = self._parse_service_response(response_text)
        return parsed[:count]

    def _build_prompt(self, domain: str, count: int, difficulty_labels: Sequence[str]) -> str:
        difficulty_summary = ", ".join(difficulty_labels[:3])
        return (
            f"Generate {count} questions related to {domain} with difficulty levels "
            f"cycling through {difficulty_summary}. Return each question on a new line "
            "without numbering or additional commentary."
        )

    def _fetch_questions(self, category: str, count: int, *, suppress_errors: bool) -> List[str]:
        prompt = (
            f"Generate {count} concise {category} questions about "
            f"{self.config.get('topic', DEFAULT_OFFLINE_TOPIC)}. "
            "Return only the questions separated by newlines without numbering."
        )

        try:
            response_text = self._call_service(prompt)
        except Exception:
            if suppress_errors:
                logger.debug("Falling back to offline templates for category '%s'", category, exc_info=True)
                return self._offline_questions(category, count)
            raise

        if not response_text:
            return self._offline_questions(category, count) if suppress_errors else []

        parsed = self._parse_service_response(response_text)
        if parsed:
            return parsed[:count]

        return self._offline_questions(category, count) if suppress_errors else []

    def _call_service(self, prompt: str) -> Optional[str]:  # pragma: no cover - network behaviour is mocked
        if requests is None:
            raise RuntimeError("requests library is not available")

        url = str(self.config.get("ollama_host") or self.config.get("ollama_url"))
        payload = {
            "model": self.config.get("model"),
            "prompt": prompt,
            "options": {
                "temperature": self.config.get("temperature"),
                "max_tokens": self.config.get("max_tokens"),
            },
        }

        response = requests.post(url, json=payload, timeout=self.http_timeout)
        if response.status_code != 200:
            logger.error("Ollama returned non-success status %s", response.status_code)
            return None

        try:
            data = response.json()
        except ValueError:
            logger.error("Response body is not valid JSON")
            return None

        if isinstance(data, dict):
            text = data.get("response") or data.get("text")
            if isinstance(text, str):
                return text.strip()
        elif isinstance(data, str):
            return data.strip()

        logger.error("Unexpected Ollama response structure: %s", type(data).__name__)
        return None

    @staticmethod
    def _parse_service_response(response_text: str) -> List[str]:
        items: List[str] = []
        for line in response_text.splitlines():
            line = line.strip().lstrip("- ")
            if line:
                items.append(line)
        return items

    def _offline_questions(self, category: str, count: int) -> List[str]:
        template_pool = OFFLINE_TEMPLATES.get(category) or OFFLINE_TEMPLATES["factual"]
        topic = self.config.get("topic", DEFAULT_OFFLINE_TOPIC)
        questions = [template.format(topic=topic) for template in template_pool]

        while len(questions) < count:
            questions.extend(questions[: count - len(questions)])

        return questions[:count]

    def _normalize_category(self, category: Optional[str]) -> str:
        if not category:
            return "factual"
        lowered = str(category).strip().lower()
        return lowered if lowered in VALID_CATEGORIES else "factual"

    def _normalize_domain(self, domain: Optional[str]) -> str:
        if domain is None:
            return "general"
        candidate = str(domain).strip().lower()
        if not candidate:
            return "general"
        if candidate not in DOMAIN_KEYWORDS:
            raise ValueError(f"Unknown domain '{domain}'")
        return candidate

    def _validate_difficulty_range(self, difficulty_range: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        if difficulty_range is None:
            return None
        if len(difficulty_range) != 2:
            raise ValueError("Difficulty range must be a tuple of (min, max)")
        low, high = difficulty_range
        if not (isinstance(low, int) and isinstance(high, int)):
            raise ValueError("Difficulty range values must be integers")
        if not (1 <= low <= 10 and 1 <= high <= 10) or low > high:
            raise ValueError("Difficulty range must be within [1, 10] and min <= max")
        return low, high

    def _difficulty_sequence(self, difficulty_range: Optional[Tuple[int, int]], count: int) -> List[str]:
        if difficulty_range is None:
            return [DIFFICULTY_SEQUENCE[idx % len(DIFFICULTY_SEQUENCE)] for idx in range(count)]

        low, high = difficulty_range
        target_band = self._label_for_range(low, high)
        return [target_band for _ in range(count)]

    def _label_for_range(self, low: int, high: int) -> str:
        for label, (band_low, band_high) in DIFFICULTY_BANDS.items():
            if low >= band_low and high <= band_high:
                return label
        if high > DIFFICULTY_BANDS["hard"][1]:
            return "hard"
        if low < DIFFICULTY_BANDS["easy"][0]:
            return "easy"
        return "medium"

    @property
    def quality_manager(self) -> "QualityManager":
        if self._quality_manager is None:
            self._quality_manager = QualityManager()
        return self._quality_manager

    def _load_config_from_name(self, config_name: str) -> Dict[str, object]:
        """Resolve configuration overrides from experiment configs."""
        normalized = (config_name or "").strip().lower() or "default"
        try:
            from src.utils.config_manager import NeuronMapConfig  # Local import to avoid cycles

            config = NeuronMapConfig()
            experiments = config.load_experiments_config()
            selected = experiments.get(normalized) or experiments.get("default", {})
            question_cfg = selected.get("question_generation", {}) if isinstance(selected, dict) else {}
        except Exception:  # pragma: no cover - defensive fallback
            question_cfg = {}

        mapped: Dict[str, object] = {}
        if question_cfg:
            mapped["ollama_host"] = question_cfg.get("ollama_host", DEFAULT_CONFIG["ollama_host"])
            mapped["model"] = question_cfg.get("model_name", DEFAULT_CONFIG["model"])
            mapped["num_questions"] = question_cfg.get("num_questions", DEFAULT_CONFIG["num_questions"])
            mapped["temperature"] = question_cfg.get("temperature", DEFAULT_CONFIG["temperature"])
            mapped["max_tokens"] = question_cfg.get("max_tokens", DEFAULT_CONFIG["max_tokens"])
            if "output_file" in question_cfg:
                mapped["output_file"] = question_cfg["output_file"]
            if "category_distribution" in question_cfg and question_cfg["category_distribution"]:
                mapped["categories"] = list(question_cfg["category_distribution"].keys())
        return mapped

    @staticmethod
    def _build_question_payload(text: str, category: str, difficulty_label: str) -> Dict[str, Any]:
        return {
            "text": text,
            "category": category,
            "difficulty": difficulty_label,
            "timestamp": _now_utc_iso(),
        }

    @staticmethod
    def _make_question_variant(question: str, seen_lower: Set[str]) -> str:
        base = question.rstrip(" ?!.")
        for suffix in range(1, 201):
            candidate = f"{base} (scenario {suffix})?"
            lowered = candidate.lower()
            if lowered not in seen_lower:
                return candidate
        return question


class QualityManager:
    """Evaluate generated questions using lightweight heuristics."""

    def __init__(self, *, min_length: int = 10, similarity_threshold: float = 0.65) -> None:
        self.min_length = min_length
        self.similarity_threshold = similarity_threshold

    # -------------------------- validation helpers -------------------
    def is_valid_question(self, question: Union[str, Dict[str, Any]]) -> bool:
        text = self._extract_text(question)
        if not text or len(text.strip()) < self.min_length:
            return False
        stripped = text.strip()
        return stripped.endswith("?") and stripped[0].isupper()

    def detect_duplicates(self, questions: Iterable[Union[str, Dict[str, Any]]]) -> List[Union[str, Dict[str, Any]]]:
        seen: Dict[str, Union[str, Dict[str, Any]]] = {}
        duplicates: List[Union[str, Dict[str, Any]]] = []
        for question in questions:
            text = self._extract_text(question).strip().lower()
            if not text:
                continue
            if text in seen:
                duplicates.append(question)
            else:
                seen[text] = question
        return duplicates

    def find_similar_questions(
        self,
        questions: Sequence[Union[str, Dict[str, Any]]],
    ) -> List[Tuple[Union[str, Dict[str, Any]], Union[str, Dict[str, Any]], float]]:
        pairs: List[Tuple[Union[str, Dict[str, Any]], Union[str, Dict[str, Any]], float]] = []
        tokenised = [(q, self._tokenise(self._extract_text(q))) for q in questions if self._extract_text(q)]

        for idx in range(len(tokenised)):
            left, left_tokens = tokenised[idx]
            if not left_tokens:
                continue
            for jdx in range(idx + 1, len(tokenised)):
                right, right_tokens = tokenised[jdx]
                if not right_tokens:
                    continue
                similarity = self._jaccard_similarity(left_tokens, right_tokens)
                if similarity >= self.similarity_threshold:
                    pairs.append((left, right, round(similarity, 3)))

        return pairs

    def calculate_quality_score(self, question: Union[str, Dict[str, Any]]) -> float:
        text = self._extract_text(question).strip()
        if not text:
            return 0.0

        words = text.split()
        length_score = min(len(words) / 20.0, 1.0)
        punctuation_score = 1.0 if text.endswith("?") else 0.5
        complexity_score = sum(1 for word in words if len(word) > 6) / max(len(words), 1)
        category = None
        if isinstance(question, dict):
            category = question.get("category")
        category_bonus = 0.2 if category in VALID_CATEGORIES else 0.0

        total = (0.4 * length_score) + (0.3 * punctuation_score) + (0.3 * complexity_score) + category_bonus
        return round(total, 3)

    # -------------------------- internal helpers ---------------------
    @staticmethod
    def _extract_text(question: Union[str, Dict[str, Any]]) -> str:
        if isinstance(question, dict):
            return str(question.get("text", ""))
        return str(question or "")

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        return [token for token in text.lower().split() if token]

    @staticmethod
    def _jaccard_similarity(a: Sequence[str], b: Sequence[str]) -> float:
        left, right = set(a), set(b)
        if not left or not right:
            return 0.0
        intersection = left.intersection(right)
        union = left.union(right)
        return len(intersection) / len(union)
