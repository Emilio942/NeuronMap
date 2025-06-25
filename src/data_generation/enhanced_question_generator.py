"""Enhanced question generator with difficulty control and rich metadata."""

import ollama
import json
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from tqdm import tqdm
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import re
from datetime import datetime
import numpy as np

from ..utils.config import get_config
from ..utils.error_handling import (
    with_retry, RetryConfig, NetworkError, ConfigurationError,
    ErrorHandler, BatchProcessor, log_error
)
from ..utils.monitoring import HealthChecker
from ollama import ResponseError, RequestError


logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    """Enumeration of question difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class QuestionCategory(Enum):
    """Enumeration of question categories."""
    FACTUAL = "factual"
    REASONING = "reasoning"
    CREATIVE = "creative"
    ETHICAL = "ethical"
    MATHEMATICAL = "mathematical"
    TECHNICAL = "technical"
    ANALYTICAL = "analytical"
    CONCEPTUAL = "conceptual"


@dataclass
class QuestionMetadata:
    """Metadata for generated questions."""
    id: str
    text: str
    difficulty: DifficultyLevel
    category: QuestionCategory
    subcategory: Optional[str] = None
    keywords: List[str] = None
    estimated_complexity_score: float = 0.0
    word_count: int = 0
    character_count: int = 0
    has_specific_knowledge: bool = False
    requires_reasoning_steps: int = 0
    domain: Optional[str] = None
    language: str = "en"
    generated_at: datetime = None
    generation_model: str = ""
    quality_score: float = 0.0
    validation_passed: bool = False
    tags: List[str] = None

    def __post_init__(self):
        """Post-initialization processing."""
        if self.keywords is None:
            self.keywords = []
        if self.tags is None:
            self.tags = []
        if self.generated_at is None:
            self.generated_at = datetime.now()

        # Auto-calculate basic metrics
        self.word_count = len(self.text.split())
        self.character_count = len(self.text)

        # Generate unique ID based on content
        if not self.id:
            content_hash = hashlib.md5(self.text.encode()).hexdigest()[:8]
            self.id = f"q_{content_hash}_{int(time.time())}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with serializable values."""
        data = asdict(self)
        data['difficulty'] = self.difficulty.value
        data['category'] = self.category.value
        data['generated_at'] = self.generated_at.isoformat()
        return data


class DifficultyAnalyzer:
    """Analyze and estimate question difficulty."""

    # Complexity indicators
    SIMPLE_WORDS = {
        'what', 'who', 'when', 'where', 'is', 'are', 'was', 'were',
        'do', 'does', 'did', 'can', 'will', 'would', 'should'
    }

    COMPLEX_INDICATORS = {
        'analyze', 'synthesize', 'evaluate', 'compare', 'contrast',
        'critique', 'justify', 'hypothesize', 'deduce', 'infer',
        'implications', 'consequences', 'relationships', 'patterns'
    }

    REASONING_INDICATORS = {
        'why', 'how', 'explain', 'because', 'therefore', 'thus',
        'if', 'then', 'cause', 'effect', 'reason', 'logic'
    }

    def estimate_difficulty(self, question: str) -> Tuple[DifficultyLevel, float]:
        """Estimate difficulty level and complexity score.

        Args:
            question: The question text to analyze.

        Returns:
            Tuple of (difficulty_level, complexity_score)
        """
        text_lower = question.lower()
        words = text_lower.split()

        complexity_score = 0.0

        # Length factor
        if len(words) > 20:
            complexity_score += 1.0
        elif len(words) > 10:
            complexity_score += 0.5

        # Simple word ratio (inverse complexity)
        simple_count = sum(1 for word in words if word in self.SIMPLE_WORDS)
        simple_ratio = simple_count / len(words) if words else 0
        complexity_score -= simple_ratio * 0.5

        # Complex indicators
        complex_count = sum(1 for word in words if word in self.COMPLEX_INDICATORS)
        complexity_score += complex_count * 0.8

        # Reasoning indicators
        reasoning_count = sum(1 for word in words if word in self.REASONING_INDICATORS)
        complexity_score += reasoning_count * 0.6

        # Multiple clauses (commas, semicolons)
        clause_markers = text_lower.count(',') + text_lower.count(';') + text_lower.count(' and ')
        complexity_score += clause_markers * 0.3

        # Question complexity patterns
        if 'what would happen if' in text_lower or 'how would' in text_lower:
            complexity_score += 1.0
        if re.search(r'\b(implications|consequences|relationships)\b', text_lower):
            complexity_score += 0.8
        if re.search(r'\b(compare|contrast|analyze|evaluate)\b', text_lower):
            complexity_score += 0.7

        # Normalize complexity score
        complexity_score = max(0.0, min(4.0, complexity_score))

        # Map to difficulty levels
        if complexity_score < 0.8:
            difficulty = DifficultyLevel.BEGINNER
        elif complexity_score < 1.8:
            difficulty = DifficultyLevel.INTERMEDIATE
        elif complexity_score < 2.8:
            difficulty = DifficultyLevel.ADVANCED
        else:
            difficulty = DifficultyLevel.EXPERT

        return difficulty, complexity_score

    def categorize_question(self, question: str) -> QuestionCategory:
        """Categorize a question based on its content.

        Args:
            question: The question text to categorize.

        Returns:
            The determined question category.
        """
        text_lower = question.lower()

        # Mathematical patterns
        if re.search(r'\b(calculate|solve|equation|formula|number|mathematical)\b', text_lower):
            return QuestionCategory.MATHEMATICAL

        # Technical patterns
        if re.search(r'\b(algorithm|programming|code|technical|system|software)\b', text_lower):
            return QuestionCategory.TECHNICAL

        # Reasoning patterns
        if re.search(r'\b(why|how|explain|because|analyze|reason|logic)\b', text_lower):
            return QuestionCategory.REASONING

        # Creative patterns
        if re.search(r'\b(imagine|creative|design|invent|story|poem)\b', text_lower):
            return QuestionCategory.CREATIVE

        # Ethical patterns
        if re.search(r'\b(ethical|moral|right|wrong|should|ought|justice)\b', text_lower):
            return QuestionCategory.ETHICAL

        # Analytical patterns
        if re.search(r'\b(compare|contrast|evaluate|assess|analyze|critique)\b', text_lower):
            return QuestionCategory.ANALYTICAL

        # Conceptual patterns
        if re.search(r'\b(concept|theory|principle|framework|understanding)\b', text_lower):
            return QuestionCategory.CONCEPTUAL

        # Default to factual
        return QuestionCategory.FACTUAL

    def extract_keywords(self, question: str, max_keywords: int = 5) -> List[str]:
        """Extract key terms from the question.

        Args:
            question: The question text.
            max_keywords: Maximum number of keywords to extract.

        Returns:
            List of extracted keywords.
        """
        # Simple keyword extraction based on content words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'what', 'when', 'where', 'who', 'why', 'how'
        }

        # Extract words, filter stop words, and get unique terms
        words = re.findall(r'\b[a-zA-Z]+\b', question.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        # Remove duplicates while preserving order
        unique_keywords = []
        seen = set()
        for word in keywords:
            if word not in seen:
                unique_keywords.append(word)
                seen.add(word)

        return unique_keywords[:max_keywords]


class QualityValidator:
    """Validate question quality and assign quality scores."""

    def __init__(self):
        self.difficulty_analyzer = DifficultyAnalyzer()

    def validate_question(self, question: str) -> Tuple[bool, float, List[str]]:
        """Validate a question and return quality metrics.

        Args:
            question: The question text to validate.

        Returns:
            Tuple of (is_valid, quality_score, issues)
        """
        issues = []
        quality_score = 1.0

        # Basic structure checks
        if not question.strip():
            issues.append("Empty question")
            return False, 0.0, issues

        if len(question.strip()) < 10:
            issues.append("Question too short")
            quality_score -= 0.3

        if len(question.strip()) > 500:
            issues.append("Question too long")
            quality_score -= 0.2

        # Grammar and structure checks
        if not question.strip().endswith('?'):
            issues.append("Missing question mark")
            quality_score -= 0.2

        if question.count('?') > 1:
            issues.append("Multiple question marks")
            quality_score -= 0.1

        # Content quality checks
        words = question.split()
        if len(words) < 3:
            issues.append("Too few words")
            quality_score -= 0.3

        # Repetitive content check
        word_set = set(words)
        if len(word_set) < len(words) * 0.6:  # Less than 60% unique words
            issues.append("High word repetition")
            quality_score -= 0.2

        # Meaningfulness check
        meaningful_words = [w for w in words if len(w) > 2 and w.isalpha()]
        if len(meaningful_words) < len(words) * 0.4:
            issues.append("Low meaningful content ratio")
            quality_score -= 0.2

        # Bonus points for good structure
        if any(word in question.lower() for word in ['explain', 'describe', 'analyze', 'compare']):
            quality_score += 0.1

        if re.search(r'\b(specific|detailed|example|instance)\b', question.lower()):
            quality_score += 0.1

        # Ensure score is within bounds
        quality_score = max(0.0, min(1.0, quality_score))

        # A question is valid if it has a quality score above threshold and no critical issues
        is_valid = quality_score >= 0.6 and not any("Empty" in issue or "too short" in issue.lower() for issue in issues)

        return is_valid, quality_score, issues


class EnhancedQuestionGenerator:
    """Enhanced question generator with difficulty control and metadata."""

    def __init__(self, config_name: str = "default"):
        """Initialize enhanced question generator.

        Args:
            config_name: Name of experiment configuration to use.
        """
        self.config = get_config()
        self.experiment_config = self.config.get_experiment_config(config_name)
        self.gen_config = self.experiment_config["question_generation"]

        # Initialize components
        self.error_handler = ErrorHandler(f"enhanced_question_generator_{config_name}_errors.jsonl")
        self.health_checker = HealthChecker()
        self.difficulty_analyzer = DifficultyAnalyzer()
        self.quality_validator = QualityValidator()

        # Create Ollama client
        self.client = ollama.Client(host=self.gen_config["ollama_host"])

        # Setup retry configuration
        self.retry_config = RetryConfig(
            max_retries=self.gen_config.get("max_retries", 5),
            base_delay=self.gen_config.get("retry_delay", 10),
            exponential_backoff=True
        )

        # Enhanced configuration
        self.target_difficulty_distribution = self.gen_config.get("difficulty_distribution", {
            "beginner": 0.3,
            "intermediate": 0.4,
            "advanced": 0.2,
            "expert": 0.1
        })

        self.target_category_distribution = self.gen_config.get("category_distribution", {
            "factual": 0.3,
            "reasoning": 0.25,
            "analytical": 0.15,
            "creative": 0.1,
            "technical": 0.1,
            "ethical": 0.05,
            "mathematical": 0.05
        })

        self.min_quality_score = self.gen_config.get("min_quality_score", 0.6)
        self.enable_metadata_enrichment = self.gen_config.get("enable_metadata_enrichment", True)

    def generate_questions_with_difficulty(
        self,
        target_difficulty: DifficultyLevel,
        target_category: QuestionCategory,
        num_questions: int = 5
    ) -> List[QuestionMetadata]:
        """Generate questions for specific difficulty and category.

        Args:
            target_difficulty: Desired difficulty level.
            target_category: Desired question category.
            num_questions: Number of questions to generate.

        Returns:
            List of generated questions with metadata.
        """
        prompt = self._create_targeted_prompt(target_difficulty, target_category, num_questions)

        try:
            response = self._generate_with_ollama(prompt)
            if not response:
                return []

            raw_questions = self._parse_questions_from_response(response)
            questions_with_metadata = []

            for question_text in raw_questions:
                # Validate question quality
                is_valid, quality_score, issues = self.quality_validator.validate_question(question_text)

                if not is_valid or quality_score < self.min_quality_score:
                    logger.debug(f"Skipping low-quality question: {question_text[:50]}... (Score: {quality_score:.2f})")
                    continue

                # Analyze difficulty and category
                actual_difficulty, complexity_score = self.difficulty_analyzer.estimate_difficulty(question_text)
                actual_category = self.difficulty_analyzer.categorize_question(question_text)
                keywords = self.difficulty_analyzer.extract_keywords(question_text)

                # Create metadata
                metadata = QuestionMetadata(
                    id="",  # Will be auto-generated
                    text=question_text,
                    difficulty=actual_difficulty,
                    category=actual_category,
                    keywords=keywords,
                    estimated_complexity_score=complexity_score,
                    generation_model=self.gen_config["model_name"],
                    quality_score=quality_score,
                    validation_passed=True,
                    domain=self._infer_domain(question_text),
                    requires_reasoning_steps=self._estimate_reasoning_steps(question_text),
                    has_specific_knowledge=self._requires_specific_knowledge(question_text)
                )

                # Add targeting information as tags
                metadata.tags.extend([
                    f"target_difficulty:{target_difficulty.value}",
                    f"target_category:{target_category.value}",
                    f"actual_difficulty:{actual_difficulty.value}",
                    f"actual_category:{actual_category.value}"
                ])

                questions_with_metadata.append(metadata)

            return questions_with_metadata

        except Exception as e:
            logger.error(f"Error generating questions with difficulty {target_difficulty}: {e}")
            return []

    def generate_balanced_question_set(self, total_questions: int) -> List[QuestionMetadata]:
        """Generate a balanced set of questions across difficulties and categories.

        Args:
            total_questions: Total number of questions to generate.

        Returns:
            List of generated questions with metadata.
        """
        all_questions = []

        # Calculate target counts for each difficulty level
        difficulty_targets = {}
        for difficulty, ratio in self.target_difficulty_distribution.items():
            difficulty_targets[DifficultyLevel(difficulty)] = int(total_questions * ratio)

        # Calculate target counts for each category
        category_targets = {}
        for category, ratio in self.target_category_distribution.items():
            category_targets[QuestionCategory(category)] = int(total_questions * ratio)

        logger.info(f"Generating balanced question set: {total_questions} total questions")
        logger.info(f"Difficulty targets: {[(d.value, count) for d, count in difficulty_targets.items()]}")
        logger.info(f"Category targets: {[(c.value, count) for c, count in category_targets.items()]}")

        # Generate questions for each combination
        total_combinations = len(difficulty_targets) * len(category_targets)
        questions_per_combination = max(1, total_questions // total_combinations)

        with tqdm(total=total_combinations, desc="Generating question combinations") as pbar:
            for difficulty in difficulty_targets.keys():
                for category in category_targets.keys():
                    questions = self.generate_questions_with_difficulty(
                        difficulty, category, questions_per_combination
                    )
                    all_questions.extend(questions)
                    pbar.update(1)

        # Shuffle and trim to exact target
        np.random.shuffle(all_questions)
        all_questions = all_questions[:total_questions]

        # Generate summary statistics
        self._log_generation_summary(all_questions)

        return all_questions

    def save_questions_with_metadata(self, questions: List[QuestionMetadata], output_file: str) -> bool:
        """Save questions with rich metadata to JSON file.

        Args:
            questions: List of questions with metadata.
            output_file: Path to output file.

        Returns:
            True if successful, False otherwise.
        """
        try:
            filepath = Path(output_file)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data for JSON serialization
            output_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "total_questions": len(questions),
                    "generator_version": "enhanced_v1.0",
                    "model_used": self.gen_config["model_name"],
                    "configuration": self.gen_config
                },
                "questions": [q.to_dict() for q in questions],
                "statistics": self._generate_statistics(questions)
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(questions)} questions with metadata to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error saving questions to {output_file}: {e}")
            return False

    def _create_targeted_prompt(
        self,
        difficulty: DifficultyLevel,
        category: QuestionCategory,
        num_questions: int
    ) -> str:
        """Create a targeted prompt for specific difficulty and category."""

        difficulty_instructions = {
            DifficultyLevel.BEGINNER: "simple, straightforward questions that require basic knowledge",
            DifficultyLevel.INTERMEDIATE: "moderately complex questions that require some analysis or application",
            DifficultyLevel.ADVANCED: "challenging questions that require deep thinking, analysis, or synthesis",
            DifficultyLevel.EXPERT: "highly sophisticated questions that require expert-level knowledge and complex reasoning"
        }

        category_instructions = {
            QuestionCategory.FACTUAL: "asking for specific facts, definitions, or concrete information",
            QuestionCategory.REASONING: "requiring logical thinking, cause-and-effect analysis, or step-by-step reasoning",
            QuestionCategory.CREATIVE: "encouraging imagination, creative thinking, or innovative solutions",
            QuestionCategory.ETHICAL: "exploring moral dilemmas, ethical considerations, or value judgments",
            QuestionCategory.MATHEMATICAL: "involving mathematical concepts, calculations, or quantitative reasoning",
            QuestionCategory.TECHNICAL: "related to technology, engineering, programming, or technical systems",
            QuestionCategory.ANALYTICAL: "requiring analysis, comparison, evaluation, or critical thinking",
            QuestionCategory.CONCEPTUAL: "exploring abstract concepts, theories, or fundamental principles"
        }

        prompt = f"""Generate {num_questions} {difficulty_instructions[difficulty]} {category_instructions[category]}.

Requirements:
- Each question should be appropriate for {difficulty.value} level
- Questions should be {category.value} in nature
- Ensure proper grammar and clear wording
- End each question with a question mark
- Make questions specific and meaningful
- Avoid overly broad or vague questions

Format your response with one question per line, no numbering or bullet points.

Examples of {difficulty.value} {category.value} questions:"""

        # Add category-specific examples
        if category == QuestionCategory.FACTUAL:
            if difficulty == DifficultyLevel.BEGINNER:
                prompt += "\n- What is the capital of France?\n- Who wrote Romeo and Juliet?"
            elif difficulty == DifficultyLevel.EXPERT:
                prompt += "\n- What are the specific biochemical pathways involved in cellular autophagy?\n- Which quantum mechanical principles govern the behavior of electrons in superconducting materials?"

        elif category == QuestionCategory.REASONING:
            if difficulty == DifficultyLevel.BEGINNER:
                prompt += "\n- Why do leaves change color in autumn?\n- How does rain form in clouds?"
            elif difficulty == DifficultyLevel.EXPERT:
                prompt += "\n- How would the implementation of universal basic income affect macroeconomic stability across different socioeconomic strata?\n- What logical framework would you use to resolve the apparent paradox between free will and deterministic physical laws?"

        prompt += "\n\nGenerate the questions now:"

        return prompt

    @with_retry()
    def _generate_with_ollama(self, prompt: str) -> Optional[str]:
        """Generate response using Ollama with retry logic."""
        try:
            response = self.client.generate(
                model=self.gen_config["model_name"],
                prompt=prompt,
                stream=False
            )
            return response.get("response", "").strip()

        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise

    def _parse_questions_from_response(self, response_text: str) -> List[str]:
        """Parse questions from LLM response."""
        questions = []
        if not response_text:
            return questions

        lines = response_text.split('\n')
        for line in lines:
            cleaned_line = line.strip()
            if cleaned_line and not cleaned_line.startswith('#') and '?' in cleaned_line:
                # Remove numbering if present
                if re.match(r'^\d+\.?\s+', cleaned_line):
                    cleaned_line = re.sub(r'^\d+\.?\s+', '', cleaned_line)

                # Remove bullet points
                if cleaned_line.startswith(('- ', '* ', '• ')):
                    cleaned_line = cleaned_line[2:]

                if cleaned_line.strip():
                    questions.append(cleaned_line.strip())

        return questions

    def _infer_domain(self, question: str) -> Optional[str]:
        """Infer the domain/subject area of a question."""
        text_lower = question.lower()

        domain_keywords = {
            "science": ["science", "scientific", "experiment", "hypothesis", "theory", "physics", "chemistry", "biology"],
            "technology": ["technology", "computer", "software", "algorithm", "programming", "digital", "internet"],
            "history": ["history", "historical", "ancient", "century", "civilization", "war", "empire"],
            "literature": ["literature", "book", "novel", "poem", "author", "writer", "story"],
            "mathematics": ["mathematics", "mathematical", "equation", "formula", "calculate", "number"],
            "philosophy": ["philosophy", "philosophical", "ethics", "moral", "existence", "consciousness"],
            "art": ["art", "artistic", "painting", "sculpture", "music", "creative", "aesthetic"],
            "politics": ["politics", "political", "government", "democracy", "election", "policy"],
            "economics": ["economics", "economic", "market", "trade", "finance", "business"],
            "psychology": ["psychology", "psychological", "behavior", "mind", "emotion", "cognitive"]
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return domain

        return None

    def _estimate_reasoning_steps(self, question: str) -> int:
        """Estimate the number of reasoning steps required."""
        text_lower = question.lower()

        # Simple heuristic based on question complexity
        reasoning_indicators = text_lower.count("why") + text_lower.count("how") + text_lower.count("explain")
        conditional_indicators = text_lower.count("if") + text_lower.count("when") + text_lower.count("suppose")
        comparison_indicators = text_lower.count("compare") + text_lower.count("contrast") + text_lower.count("versus")

        total_indicators = reasoning_indicators + conditional_indicators + comparison_indicators

        if total_indicators == 0:
            return 1  # Simple factual question
        elif total_indicators <= 2:
            return 2  # Moderate reasoning
        else:
            return min(5, total_indicators)  # Complex reasoning, capped at 5

    def _requires_specific_knowledge(self, question: str) -> bool:
        """Determine if question requires specific/specialized knowledge."""
        text_lower = question.lower()

        # Indicators of specific knowledge requirement
        specific_indicators = [
            "specific", "particular", "exact", "precise", "detailed",
            "named", "called", "termed", "defined as", "known as"
        ]

        technical_terms_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+ology\b',   # Fields ending in -ology
            r'\b\w+ism\b',     # Concepts ending in -ism
            r'\b\w+tion\b'     # Technical terms ending in -tion
        ]

        # Check for specific indicators
        if any(indicator in text_lower for indicator in specific_indicators):
            return True

        # Check for technical patterns
        if any(re.search(pattern, question) for pattern in technical_terms_patterns):
            return True

        return False

    def _generate_statistics(self, questions: List[QuestionMetadata]) -> Dict[str, Any]:
        """Generate summary statistics for the question set."""
        if not questions:
            return {}

        # Difficulty distribution
        difficulty_counts = {}
        for difficulty in DifficultyLevel:
            difficulty_counts[difficulty.value] = sum(
                1 for q in questions if q.difficulty == difficulty
            )

        # Category distribution
        category_counts = {}
        for category in QuestionCategory:
            category_counts[category.value] = sum(
                1 for q in questions if q.category == category
            )

        # Quality metrics
        quality_scores = [q.quality_score for q in questions]
        complexity_scores = [q.estimated_complexity_score for q in questions]
        word_counts = [q.word_count for q in questions]

        return {
            "difficulty_distribution": difficulty_counts,
            "category_distribution": category_counts,
            "quality_metrics": {
                "mean_quality_score": np.mean(quality_scores),
                "min_quality_score": np.min(quality_scores),
                "max_quality_score": np.max(quality_scores),
                "std_quality_score": np.std(quality_scores)
            },
            "complexity_metrics": {
                "mean_complexity": np.mean(complexity_scores),
                "min_complexity": np.min(complexity_scores),
                "max_complexity": np.max(complexity_scores),
                "std_complexity": np.std(complexity_scores)
            },
            "length_metrics": {
                "mean_word_count": np.mean(word_counts),
                "min_word_count": np.min(word_counts),
                "max_word_count": np.max(word_counts),
                "std_word_count": np.std(word_counts)
            },
            "domain_distribution": self._count_domains(questions),
            "validation_stats": {
                "total_questions": len(questions),
                "validated_questions": sum(1 for q in questions if q.validation_passed),
                "questions_with_specific_knowledge": sum(1 for q in questions if q.has_specific_knowledge),
                "mean_reasoning_steps": np.mean([q.requires_reasoning_steps for q in questions])
            }
        }

    def _count_domains(self, questions: List[QuestionMetadata]) -> Dict[str, int]:
        """Count questions by domain."""
        domain_counts = {}
        for question in questions:
            domain = question.domain or "general"
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        return domain_counts

    def _log_generation_summary(self, questions: List[QuestionMetadata]) -> None:
        """Log summary of generated questions."""
        if not questions:
            logger.warning("No questions generated!")
            return

        stats = self._generate_statistics(questions)

        logger.info(f"Generated {len(questions)} questions with metadata")
        logger.info(f"Difficulty distribution: {stats['difficulty_distribution']}")
        logger.info(f"Category distribution: {stats['category_distribution']}")
        logger.info(f"Average quality score: {stats['quality_metrics']['mean_quality_score']:.2f}")
        logger.info(f"Average complexity: {stats['complexity_metrics']['mean_complexity']:.2f}")
        logger.info(f"Average word count: {stats['length_metrics']['mean_word_count']:.1f}")


def main():
    """Main function for command line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced question generation with difficulty control")
    parser.add_argument("--config", default="default", help="Configuration name to use")
    parser.add_argument("--num-questions", type=int, default=20, help="Number of questions to generate")
    parser.add_argument("--output", default="enhanced_questions.json", help="Output file path")
    parser.add_argument("--difficulty", choices=[d.value for d in DifficultyLevel],
                       help="Target specific difficulty level")
    parser.add_argument("--category", choices=[c.value for c in QuestionCategory],
                       help="Target specific category")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    generator = EnhancedQuestionGenerator(args.config)

    try:
        if args.difficulty and args.category:
            # Generate targeted questions
            difficulty = DifficultyLevel(args.difficulty)
            category = QuestionCategory(args.category)
            questions = generator.generate_questions_with_difficulty(
                difficulty, category, args.num_questions
            )
        else:
            # Generate balanced set
            questions = generator.generate_balanced_question_set(args.num_questions)

        if questions:
            success = generator.save_questions_with_metadata(questions, args.output)
            if success:
                logger.info(f"Successfully generated and saved {len(questions)} questions to {args.output}")
            else:
                logger.error("Failed to save questions")
                exit(1)
        else:
            logger.error("No questions were generated")
            exit(1)

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()