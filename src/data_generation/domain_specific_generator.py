"""Domain-specific question generator for specialized knowledge areas."""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import random

from .enhanced_question_generator import (
    EnhancedQuestionGenerator, DifficultyLevel, QuestionCategory,
    QuestionMetadata, DifficultyAnalyzer, QualityValidator
)


logger = logging.getLogger(__name__)


class Domain(Enum):
    """Enumeration of knowledge domains."""
    SCIENCE = "science"
    MATHEMATICS = "mathematics"
    HISTORY = "history"
    LITERATURE = "literature"
    TECHNOLOGY = "technology"
    PHILOSOPHY = "philosophy"
    MEDICINE = "medicine"
    LAW = "law"
    ECONOMICS = "economics"
    GEOGRAPHY = "geography"
    PSYCHOLOGY = "psychology"
    SOCIOLOGY = "sociology"
    ENGINEERING = "engineering"
    BIOLOGY = "biology"
    CHEMISTRY = "chemistry"
    PHYSICS = "physics"
    COMPUTER_SCIENCE = "computer_science"
    LINGUISTICS = "linguistics"
    ANTHROPOLOGY = "anthropology"
    POLITICAL_SCIENCE = "political_science"


@dataclass
class DomainPromptTemplate:
    """Template for domain-specific question generation."""
    domain: Domain
    instruction: str
    context_keywords: List[str]
    example_questions: Dict[DifficultyLevel, List[str]]
    specialized_concepts: List[str]
    subdomain_areas: List[str]


class DomainPromptTemplateManager:
    """Manages prompt templates for different domains."""

    def __init__(self):
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[Domain, DomainPromptTemplate]:
        """Initialize domain-specific prompt templates."""
        templates = {}

        # Science Domain
        templates[Domain.SCIENCE] = DomainPromptTemplate(
            domain=Domain.SCIENCE,
            instruction="Generate scientific questions covering various fields of natural sciences",
            context_keywords=[
                "experiment", "hypothesis", "theory", "observation", "data", "research",
                "method", "analysis", "evidence", "discovery", "natural", "physical"
            ],
            example_questions={
                DifficultyLevel.BEGINNER: [
                    "What is the difference between a hypothesis and a theory?",
                    "Why do objects fall to the ground?",
                    "What are the three states of matter?"
                ],
                DifficultyLevel.INTERMEDIATE: [
                    "How does natural selection drive evolutionary change?",
                    "What role do enzymes play in biochemical reactions?",
                    "How do greenhouse gases affect global climate?"
                ],
                DifficultyLevel.ADVANCED: [
                    "How do quantum mechanical principles explain chemical bonding?",
                    "What mechanisms drive plate tectonics and continental drift?",
                    "How do epigenetic modifications influence gene expression?"
                ],
                DifficultyLevel.EXPERT: [
                    "How do emergent properties arise from complex adaptive systems in ecology?",
                    "What are the implications of quantum entanglement for our understanding of reality?",
                    "How do prion proteins propagate misfolding in neurodegenerative diseases?"
                ]
            },
            specialized_concepts=[
                "evolution", "photosynthesis", "DNA", "quantum mechanics", "relativity",
                "entropy", "ecosystem", "periodic table", "cell division", "thermodynamics"
            ],
            subdomain_areas=[
                "biology", "chemistry", "physics", "earth science", "astronomy",
                "geology", "meteorology", "oceanography", "ecology", "genetics"
            ]
        )

        # Mathematics Domain
        templates[Domain.MATHEMATICS] = DomainPromptTemplate(
            domain=Domain.MATHEMATICS,
            instruction="Generate mathematical questions covering various areas of mathematics",
            context_keywords=[
                "equation", "proof", "theorem", "calculation", "formula", "function",
                "variable", "constant", "algebra", "geometry", "calculus", "statistics"
            ],
            example_questions={
                DifficultyLevel.BEGINNER: [
                    "What is the Pythagorean theorem?",
                    "How do you calculate the area of a circle?",
                    "What is the difference between mean and median?"
                ],
                DifficultyLevel.INTERMEDIATE: [
                    "How do you find the derivative of a polynomial function?",
                    "What is the relationship between sine and cosine functions?",
                    "How do you solve a system of linear equations?"
                ],
                DifficultyLevel.ADVANCED: [
                    "How do you prove the fundamental theorem of calculus?",
                    "What are the applications of eigenvalues in linear algebra?",
                    "How do you construct a confidence interval for a population mean?"
                ],
                DifficultyLevel.EXPERT: [
                    "How do you prove the Riemann hypothesis for specific cases?",
                    "What are the topological properties of manifolds in differential geometry?",
                    "How do stochastic differential equations model random processes?"
                ]
            },
            specialized_concepts=[
                "calculus", "algebra", "geometry", "statistics", "probability", "topology",
                "number theory", "linear algebra", "differential equations", "complex analysis"
            ],
            subdomain_areas=[
                "pure mathematics", "applied mathematics", "statistics", "mathematical physics",
                "computational mathematics", "discrete mathematics", "analysis", "algebra"
            ]
        )

        # History Domain
        templates[Domain.HISTORY] = DomainPromptTemplate(
            domain=Domain.HISTORY,
            instruction="Generate historical questions covering different periods and civilizations",
            context_keywords=[
                "civilization", "empire", "war", "revolution", "culture", "society",
                "politics", "economy", "religion", "technology", "trade", "exploration"
            ],
            example_questions={
                DifficultyLevel.BEGINNER: [
                    "Who was the first President of the United States?",
                    "When did World War II end?",
                    "What was the Renaissance?"
                ],
                DifficultyLevel.INTERMEDIATE: [
                    "What factors led to the fall of the Roman Empire?",
                    "How did the Industrial Revolution change society?",
                    "What were the main causes of World War I?"
                ],
                DifficultyLevel.ADVANCED: [
                    "How did the Silk Road influence cultural exchange between East and West?",
                    "What role did economic factors play in the French Revolution?",
                    "How did colonialism shape the modern political map of Africa?"
                ],
                DifficultyLevel.EXPERT: [
                    "How did climatic changes influence the collapse of Bronze Age civilizations?",
                    "What were the long-term economic consequences of the Black Death in Europe?",
                    "How did the concept of sovereignty evolve in the Peace of Westphalia system?"
                ]
            },
            specialized_concepts=[
                "ancient civilizations", "medieval period", "renaissance", "industrial revolution",
                "world wars", "cold war", "colonialism", "imperialism", "nationalism", "democracy"
            ],
            subdomain_areas=[
                "ancient history", "medieval history", "modern history", "military history",
                "social history", "economic history", "cultural history", "political history"
            ]
        )

        # Technology Domain
        templates[Domain.TECHNOLOGY] = DomainPromptTemplate(
            domain=Domain.TECHNOLOGY,
            instruction="Generate technology questions covering computing, engineering, and innovation",
            context_keywords=[
                "computer", "software", "hardware", "algorithm", "programming", "internet",
                "artificial intelligence", "machine learning", "data", "network", "system"
            ],
            example_questions={
                DifficultyLevel.BEGINNER: [
                    "What is the difference between hardware and software?",
                    "How does the internet work?",
                    "What is a computer program?"
                ],
                DifficultyLevel.INTERMEDIATE: [
                    "How do machine learning algorithms learn from data?",
                    "What are the principles of object-oriented programming?",
                    "How do databases store and retrieve information?"
                ],
                DifficultyLevel.ADVANCED: [
                    "How do distributed systems maintain consistency across nodes?",
                    "What are the computational complexity implications of quantum algorithms?",
                    "How do neural networks process sequential data?"
                ],
                DifficultyLevel.EXPERT: [
                    "How do consensus algorithms work in blockchain networks?",
                    "What are the theoretical limits of quantum error correction?",
                    "How do advanced compilation techniques optimize code execution?"
                ]
            },
            specialized_concepts=[
                "algorithms", "data structures", "artificial intelligence", "cybersecurity",
                "blockchain", "quantum computing", "robotics", "IoT", "cloud computing"
            ],
            subdomain_areas=[
                "computer science", "software engineering", "artificial intelligence",
                "cybersecurity", "data science", "robotics", "telecommunications"
            ]
        )

        # Literature Domain
        templates[Domain.LITERATURE] = DomainPromptTemplate(
            domain=Domain.LITERATURE,
            instruction="Generate literature questions covering various genres, periods, and styles",
            context_keywords=[
                "novel", "poem", "author", "character", "theme", "style", "genre",
                "narrative", "metaphor", "symbolism", "literary device", "criticism"
            ],
            example_questions={
                DifficultyLevel.BEGINNER: [
                    "Who wrote 'Romeo and Juliet'?",
                    "What is the difference between prose and poetry?",
                    "What is a metaphor?"
                ],
                DifficultyLevel.INTERMEDIATE: [
                    "How does symbolism function in 'The Great Gatsby'?",
                    "What are the characteristics of Romantic poetry?",
                    "How do unreliable narrators affect story interpretation?"
                ],
                DifficultyLevel.ADVANCED: [
                    "How does stream of consciousness technique work in modernist literature?",
                    "What role does intertextuality play in postmodern fiction?",
                    "How do cultural contexts influence literary interpretation?"
                ],
                DifficultyLevel.EXPERT: [
                    "How do deconstructionist readings challenge traditional literary analysis?",
                    "What are the implications of reader-response theory for textual meaning?",
                    "How does the concept of the sublime function in Romantic aesthetics?"
                ]
            },
            specialized_concepts=[
                "romanticism", "modernism", "postmodernism", "narrative theory", "literary criticism",
                "genre theory", "comparative literature", "literary history", "poetics"
            ],
            subdomain_areas=[
                "classical literature", "modern literature", "poetry", "drama", "fiction",
                "literary theory", "comparative literature", "world literature"
            ]
        )

        # Philosophy Domain
        templates[Domain.PHILOSOPHY] = DomainPromptTemplate(
            domain=Domain.PHILOSOPHY,
            instruction="Generate philosophical questions covering ethics, metaphysics, epistemology, and logic",
            context_keywords=[
                "ethics", "morality", "knowledge", "reality", "consciousness", "existence",
                "truth", "justice", "freedom", "logic", "reasoning", "argument"
            ],
            example_questions={
                DifficultyLevel.BEGINNER: [
                    "What is the difference between right and wrong?",
                    "What does it mean to exist?",
                    "How do we know what we know?"
                ],
                DifficultyLevel.INTERMEDIATE: [
                    "What are the main ethical theories and how do they differ?",
                    "How does free will relate to moral responsibility?",
                    "What is the mind-body problem in philosophy of mind?"
                ],
                DifficultyLevel.ADVANCED: [
                    "How does Kant's categorical imperative function as a moral principle?",
                    "What are the implications of logical positivism for metaphysics?",
                    "How do social contract theories justify political authority?"
                ],
                DifficultyLevel.EXPERT: [
                    "How does Heidegger's concept of Being-in-the-world challenge Cartesian dualism?",
                    "What are the epistemological implications of Gettier problems for justified belief?",
                    "How does Wittgenstein's private language argument affect theories of consciousness?"
                ]
            },
            specialized_concepts=[
                "ethics", "metaphysics", "epistemology", "logic", "aesthetics", "political philosophy",
                "philosophy of mind", "philosophy of science", "existentialism", "phenomenology"
            ],
            subdomain_areas=[
                "ancient philosophy", "medieval philosophy", "modern philosophy", "contemporary philosophy",
                "analytic philosophy", "continental philosophy", "applied philosophy"
            ]
        )

        return templates

    def get_template(self, domain: Domain) -> DomainPromptTemplate:
        """Get prompt template for a specific domain."""
        return self.templates.get(domain)

    def get_all_domains(self) -> List[Domain]:
        """Get all available domains."""
        return list(self.templates.keys())


class DomainSpecificQuestionGenerator(EnhancedQuestionGenerator):
    """Enhanced question generator with domain-specific capabilities."""

    def __init__(self, config_name: str = "default"):
        """Initialize domain-specific question generator."""
        super().__init__(config_name)
        self.prompt_manager = DomainPromptTemplateManager()

        # Domain-specific configuration
        domain_config = self.gen_config.get("domain_targeting", {})
        self.domain_enabled = domain_config.get("enabled", False)
        self.target_domains = [Domain(d) for d in domain_config.get("target_domains", [])]
        self.domain_balance = domain_config.get("domain_balance", "equal")

    def generate_domain_questions(
        self,
        domain: Domain,
        difficulty: DifficultyLevel,
        category: QuestionCategory,
        num_questions: int = 5,
        subdomain: Optional[str] = None
    ) -> List[QuestionMetadata]:
        """Generate questions for a specific domain.

        Args:
            domain: Target knowledge domain.
            difficulty: Desired difficulty level.
            category: Desired question category.
            num_questions: Number of questions to generate.
            subdomain: Optional subdomain specification.

        Returns:
            List of generated questions with metadata.
        """
        template = self.prompt_manager.get_template(domain)
        if not template:
            logger.error(f"No template available for domain: {domain.value}")
            return []

        prompt = self._create_domain_prompt(template, difficulty, category, num_questions, subdomain)

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
                    logger.debug(f"Skipping low-quality domain question: {question_text[:50]}...")
                    continue

                # Analyze question properties
                actual_difficulty, complexity_score = self.difficulty_analyzer.estimate_difficulty(question_text)
                actual_category = self.difficulty_analyzer.categorize_question(question_text)
                keywords = self.difficulty_analyzer.extract_keywords(question_text)

                # Add domain-specific keywords
                domain_keywords = [kw for kw in template.context_keywords if kw.lower() in question_text.lower()]
                keywords.extend(domain_keywords)

                # Create metadata with domain information
                metadata = QuestionMetadata(
                    id="",  # Will be auto-generated
                    text=question_text,
                    difficulty=actual_difficulty,
                    category=actual_category,
                    subdomain=subdomain,
                    keywords=list(set(keywords)),  # Remove duplicates
                    estimated_complexity_score=complexity_score,
                    generation_model=self.gen_config["model_name"],
                    quality_score=quality_score,
                    validation_passed=True,
                    domain=domain.value,
                    requires_reasoning_steps=self._estimate_reasoning_steps(question_text),
                    has_specific_knowledge=True  # Domain questions typically require specific knowledge
                )

                # Add domain-specific tags
                metadata.tags.extend([
                    f"domain:{domain.value}",
                    f"target_difficulty:{difficulty.value}",
                    f"target_category:{category.value}",
                    f"actual_difficulty:{actual_difficulty.value}",
                    f"actual_category:{actual_category.value}"
                ])

                if subdomain:
                    metadata.tags.append(f"subdomain:{subdomain}")

                questions_with_metadata.append(metadata)

            return questions_with_metadata

        except Exception as e:
            logger.error(f"Error generating domain questions for {domain.value}: {e}")
            return []

    def generate_multi_domain_questions(
        self,
        domains: List[Domain],
        total_questions: int,
        difficulty_distribution: Optional[Dict[DifficultyLevel, float]] = None,
        category_distribution: Optional[Dict[QuestionCategory, float]] = None
    ) -> List[QuestionMetadata]:
        """Generate questions across multiple domains.

        Args:
            domains: List of target domains.
            total_questions: Total number of questions to generate.
            difficulty_distribution: Optional difficulty distribution override.
            category_distribution: Optional category distribution override.

        Returns:
            List of generated questions with metadata.
        """
        if not domains:
            logger.error("No domains specified for multi-domain generation")
            return []

        # Use provided distributions or defaults
        diff_dist = difficulty_distribution or self.target_difficulty_distribution
        cat_dist = category_distribution or self.target_category_distribution

        # Calculate questions per domain
        questions_per_domain = total_questions // len(domains)
        remaining_questions = total_questions % len(domains)

        all_questions = []

        logger.info(f"Generating questions across {len(domains)} domains...")

        for i, domain in enumerate(domains):
            # Add remainder questions to first few domains
            domain_questions = questions_per_domain + (1 if i < remaining_questions else 0)

            if domain_questions == 0:
                continue

            logger.info(f"Generating {domain_questions} questions for {domain.value}...")

            # Generate questions for each difficulty/category combination
            for difficulty, diff_ratio in diff_dist.items():
                for category, cat_ratio in cat_dist.items():
                    # Calculate number of questions for this combination
                    combo_questions = max(1, int(domain_questions * diff_ratio * cat_ratio))

                    if combo_questions > 0:
                        questions = self.generate_domain_questions(
                            domain, DifficultyLevel(difficulty), QuestionCategory(category), combo_questions
                        )
                        all_questions.extend(questions)

        # Shuffle and trim to exact target
        random.shuffle(all_questions)
        all_questions = all_questions[:total_questions]

        logger.info(f"Generated {len(all_questions)} multi-domain questions")
        self._log_domain_generation_summary(all_questions)

        return all_questions

    def _create_domain_prompt(
        self,
        template: DomainPromptTemplate,
        difficulty: DifficultyLevel,
        category: QuestionCategory,
        num_questions: int,
        subdomain: Optional[str] = None
    ) -> str:
        """Create a domain-specific prompt."""

        # Get example questions for the target difficulty
        examples = template.example_questions.get(difficulty, [])

        # Build subdomain specification
        subdomain_text = ""
        if subdomain:
            subdomain_text = f"\nFocus specifically on: {subdomain}"
        elif template.subdomain_areas:
            # Suggest random subdomain areas
            suggested_areas = random.sample(template.subdomain_areas, min(3, len(template.subdomain_areas)))
            subdomain_text = f"\nConsider these areas: {', '.join(suggested_areas)}"

        # Category-specific instructions
        category_instructions = {
            QuestionCategory.FACTUAL: "asking for specific facts, definitions, or concrete information",
            QuestionCategory.REASONING: "requiring logical thinking, cause-and-effect analysis, or step-by-step reasoning",
            QuestionCategory.CREATIVE: "encouraging imagination, creative thinking, or innovative solutions",
            QuestionCategory.ETHICAL: "exploring moral dilemmas, ethical considerations, or value judgments",
            QuestionCategory.MATHEMATICAL: "involving mathematical concepts, calculations, or quantitative reasoning",
            QuestionCategory.TECHNICAL: "related to technical procedures, methods, or specialized techniques",
            QuestionCategory.ANALYTICAL: "requiring analysis, comparison, evaluation, or critical thinking",
            QuestionCategory.CONCEPTUAL: "exploring abstract concepts, theories, or fundamental principles"
        }

        # Difficulty-specific instructions
        difficulty_instructions = {
            DifficultyLevel.BEGINNER: "accessible to newcomers with basic background knowledge",
            DifficultyLevel.INTERMEDIATE: "requiring moderate expertise and analytical thinking",
            DifficultyLevel.ADVANCED: "challenging for experts, requiring deep understanding and synthesis",
            DifficultyLevel.EXPERT: "requiring specialized knowledge and sophisticated reasoning"
        }

        prompt = f"""Generate {num_questions} high-quality questions in the field of {template.domain.value.replace('_', ' ').title()}.

Domain Focus: {template.instruction}
{subdomain_text}

Requirements:
- Questions should be {difficulty_instructions[difficulty]}
- Questions should be {category_instructions[category]}
- Include domain-specific terminology and concepts
- Ensure proper grammar and clear wording
- End each question with a question mark
- Make questions specific and meaningful to the {template.domain.value.replace('_', ' ')} domain

Key concepts to consider: {', '.join(template.specialized_concepts[:10])}
Context keywords: {', '.join(template.context_keywords[:8])}

Examples of {difficulty.value} level questions in this domain:"""

        # Add examples
        if examples:
            for i, example in enumerate(examples[:3], 1):
                prompt += f"\n{i}. {example}"

        prompt += f"\n\nNow generate {num_questions} new questions following this style and difficulty level:"

        return prompt

    def _log_domain_generation_summary(self, questions: List[QuestionMetadata]) -> None:
        """Log summary of domain-specific generation."""
        if not questions:
            logger.warning("No domain questions generated!")
            return

        # Count by domain
        domain_counts = {}
        for q in questions:
            domain = q.domain or "unknown"
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        # Count by difficulty and category
        difficulty_counts = {}
        category_counts = {}
        for q in questions:
            diff = q.difficulty.value
            cat = q.category.value
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
            category_counts[cat] = category_counts.get(cat, 0) + 1

        logger.info(f"Domain-specific generation summary:")
        logger.info(f"  Total questions: {len(questions)}")
        logger.info(f"  Domain distribution: {domain_counts}")
        logger.info(f"  Difficulty distribution: {difficulty_counts}")
        logger.info(f"  Category distribution: {category_counts}")

        # Quality metrics
        quality_scores = [q.quality_score for q in questions]
        complexity_scores = [q.estimated_complexity_score for q in questions]

        logger.info(f"  Average quality score: {sum(quality_scores)/len(quality_scores):.3f}")
        logger.info(f"  Average complexity: {sum(complexity_scores)/len(complexity_scores):.3f}")


def main():
    """Main function for command line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Domain-specific question generation")
    parser.add_argument("--config", default="default", help="Configuration name to use")
    parser.add_argument("--domains", nargs="+", choices=[d.value for d in Domain],
                       help="Target domains for question generation")
    parser.add_argument("--num-questions", type=int, default=50,
                       help="Total number of questions to generate")
    parser.add_argument("--output", default="domain_specific_questions.json",
                       help="Output file path")
    parser.add_argument("--difficulty", choices=[d.value for d in DifficultyLevel],
                       help="Target specific difficulty level")
    parser.add_argument("--category", choices=[c.value for c in QuestionCategory],
                       help="Target specific category")
    parser.add_argument("--subdomain", help="Focus on specific subdomain")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    generator = DomainSpecificQuestionGenerator(args.config)

    try:
        if args.domains:
            domains = [Domain(d) for d in args.domains]

            if len(domains) == 1 and args.difficulty and args.category:
                # Single domain with specific difficulty/category
                difficulty = DifficultyLevel(args.difficulty)
                category = QuestionCategory(args.category)

                questions = generator.generate_domain_questions(
                    domains[0], difficulty, category, args.num_questions, args.subdomain
                )
            else:
                # Multi-domain generation
                questions = generator.generate_multi_domain_questions(
                    domains, args.num_questions
                )
        else:
            logger.error("Must specify at least one domain with --domains")
            return 1

        if questions:
            success = generator.save_questions_with_metadata(questions, args.output)
            if success:
                logger.info(f"Successfully generated {len(questions)} domain-specific questions")
                return 0
            else:
                logger.error("Failed to save questions")
                return 1
        else:
            logger.error("No questions were generated")
            return 1

    except Exception as e:
        logger.error(f"Domain-specific generation failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())