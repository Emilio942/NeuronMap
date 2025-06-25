"""
Domain Specialization Framework for generating domain-specific questions.
Provides specialized question generators for 20+ academic and professional domains.
"""

import random
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseDomainSpecialist(ABC):
    """Abstract base class for domain-specific question specialists."""

    def __init__(self):
        """Initialize the specialist with domain-specific settings."""
        self.domain = self.__class__.__name__.replace('Specialist', '').lower()
        self.logger = logging.getLogger(f"specialist.{self.domain}")

    @abstractmethod
    def get_domain_vocabulary(self) -> List[str]:
        """Return a list of domain-specific vocabulary terms."""
        pass

    @abstractmethod
    def get_question_patterns(self) -> List[str]:
        """Return a list of question patterns for this domain."""
        pass

    @abstractmethod
    def validate_domain_relevance(self, text: str) -> float:
        """Validate how relevant a text is to this domain (0.0 to 1.0)."""
        pass

    def generate_questions(self, count: int = 5, context: Optional[str] = None) -> List[str]:
        """Generate domain-specific questions."""
        questions = []
        vocabulary = self.get_domain_vocabulary()
        patterns = self.get_question_patterns()

        for _ in range(count):
            pattern = random.choice(patterns)
            question = self._fill_pattern(pattern, vocabulary)
            questions.append(question)

        return questions

    def _fill_pattern(self, pattern: str, vocabulary: List[str]) -> str:
        """Fill a question pattern with domain-specific terms."""
        if not vocabulary:
            vocabulary = ['concept', 'term', 'process', 'phenomenon', 'principle', 'method', 'system', 'element', 'factor', 'example']

        def safe_choice(candidates, fallback='term'):
            """Safely choose from a list of candidates with fallback."""
            if candidates:
                return random.choice(candidates)
            elif vocabulary:
                return random.choice(vocabulary[:5])
            else:
                return fallback

        # Generic placeholder mapping with safer fallbacks
        placeholders = {
            '{concept}': safe_choice(vocabulary[:20], 'concept'),
            '{term}': safe_choice(vocabulary[:30], 'term'),
            '{process}': safe_choice([v for v in vocabulary if 'process' in v.lower() or 'tion' in v], 'process'),
            '{phenomenon}': safe_choice([v for v in vocabulary if any(x in v.lower() for x in ['effect', 'phenomenon', 'behavior'])], 'phenomenon'),
            '{principle}': safe_choice([v for v in vocabulary if any(x in v.lower() for x in ['law', 'principle', 'rule', 'theory'])], 'principle'),
            '{method}': safe_choice([v for v in vocabulary if any(x in v.lower() for x in ['method', 'technique', 'approach'])], 'method'),
            '{system}': safe_choice([v for v in vocabulary if 'system' in v.lower()], 'system'),
            '{element}': safe_choice(vocabulary[:40], 'element'),
            '{factor}': safe_choice([v for v in vocabulary if any(x in v.lower() for x in ['factor', 'element', 'component'])], 'factor'),
            '{example}': safe_choice(vocabulary[:25], 'example'),
        }

        # Fill pattern with placeholders
        filled = pattern
        for placeholder, value in placeholders.items():
            if placeholder in filled:
                filled = filled.replace(placeholder, value)

        return filled


# Domain-specific specialists
class ScienceSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["hypothesis", "experiment", "theory", "data", "observation", "analysis", "method", "research", "evidence", "conclusion", "variable", "control", "measurement", "peer review", "scientific method", "replication", "laboratory", "empirical", "quantitative", "qualitative"]

    def get_question_patterns(self) -> List[str]:
        return ["How does {factor} affect {phenomenon}?", "What is the relationship between {concept} and {concept}?", "Why does {process} occur in {system}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        science_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(science_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class LiteratureSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["narrative", "character", "plot", "theme", "symbolism", "metaphor", "irony", "author", "protagonist", "antagonist", "conflict", "setting", "genre", "poetry", "prose", "fiction", "analysis", "interpretation", "literary device", "criticism"]

    def get_question_patterns(self) -> List[str]:
        return ["How does {element} contribute to {concept}?", "What is the significance of {term} in {example}?", "How does the author use {method} to convey {concept}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        lit_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(lit_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class MathematicsSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["equation", "function", "variable", "coefficient", "derivative", "integral", "theorem", "proof", "algorithm", "formula", "geometry", "algebra", "calculus", "statistics", "probability", "matrix", "vector", "polynomial", "logarithm", "trigonometry"]

    def get_question_patterns(self) -> List[str]:
        return ["What is the {concept} of {element}?", "How do you solve for {term} in {system}?", "What is the relationship between {factor} and {factor}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        math_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(math_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class HistorySpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["civilization", "empire", "revolution", "war", "treaty", "dynasty", "monarchy", "democracy", "republic", "constitution", "legislation", "rebellion", "conquest", "culture", "society", "politics", "economics", "diplomacy", "colonization", "independence"]

    def get_question_patterns(self) -> List[str]:
        return ["What were the causes of {phenomenon}?", "How did {factor} influence {concept}?", "What was the impact of {term} on {system}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        hist_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(hist_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class TechnologySpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["software", "hardware", "algorithm", "programming", "database", "network", "security", "encryption", "artificial intelligence", "machine learning", "cloud computing", "internet", "protocol", "interface", "framework", "architecture", "development", "debugging", "optimization", "innovation"]

    def get_question_patterns(self) -> List[str]:
        return ["How does {system} implement {concept}?", "What are the advantages of {method} over {method}?", "How can {factor} improve {process}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        tech_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(tech_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class PsychologySpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["behavior", "cognition", "emotion", "personality", "development", "learning", "memory", "perception", "motivation", "consciousness", "therapy", "disorder", "treatment", "research", "experiment", "theory", "assessment", "intervention", "social psychology", "neuroscience"]

    def get_question_patterns(self) -> List[str]:
        return ["How does {factor} affect {phenomenon}?", "What is the relationship between {concept} and {concept}?", "How can {method} be used to study {process}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        psych_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(psych_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class EconomicsSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["market", "demand", "supply", "price", "inflation", "recession", "GDP", "unemployment", "investment", "capital", "trade", "competition", "monopoly", "fiscal policy", "monetary policy", "economics", "microeconomics", "macroeconomics", "elasticity", "equilibrium"]

    def get_question_patterns(self) -> List[str]:
        return ["How does {factor} impact {system}?", "What happens to {concept} when {factor} changes?", "How do {element} and {element} interact in {system}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        econ_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(econ_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class PhilosophySpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["ethics", "morality", "logic", "reasoning", "knowledge", "truth", "reality", "existence", "consciousness", "free will", "determinism", "justice", "virtue", "duty", "rights", "metaphysics", "epistemology", "phenomenology", "existentialism", "utilitarianism"]

    def get_question_patterns(self) -> List[str]:
        return ["What is the nature of {concept}?", "How do we define {term}?", "What are the implications of {principle} for {system}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        phil_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(phil_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class ArtSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["painting", "sculpture", "drawing", "color", "composition", "perspective", "style", "technique", "medium", "canvas", "brush", "pigment", "aesthetic", "beauty", "creativity", "expression", "form", "line", "shape", "texture"]

    def get_question_patterns(self) -> List[str]:
        return ["How does {element} contribute to {concept}?", "What {method} did the artist use to create {phenomenon}?", "How does {factor} affect the viewer's perception of {term}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        art_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(art_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class MusicSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["melody", "harmony", "rhythm", "tempo", "pitch", "scale", "chord", "composition", "instrument", "performance", "conductor", "orchestra", "symphony", "concerto", "genre", "notation", "dynamics", "timbre", "counterpoint", "improvisation"]

    def get_question_patterns(self) -> List[str]:
        return ["How does {element} create {phenomenon}?", "What role does {factor} play in {system}?", "How do musicians use {method} to achieve {concept}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        music_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(music_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class GeographySpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["landscape", "climate", "topography", "population", "migration", "urbanization", "environment", "ecosystem", "resources", "sustainability", "cartography", "GIS", "spatial analysis", "region", "territory", "boundary", "location", "place", "scale", "globalization"]

    def get_question_patterns(self) -> List[str]:
        return ["How does {factor} influence {phenomenon}?", "What is the relationship between {concept} and {system}?", "How do {element} patterns vary across {term}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        geo_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(geo_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class SociologySpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["society", "culture", "social structure", "institution", "community", "group", "interaction", "socialization", "inequality", "stratification", "class", "race", "gender", "ethnicity", "identity", "power", "authority", "conflict", "cooperation", "social change"]

    def get_question_patterns(self) -> List[str]:
        return ["How does {factor} shape {phenomenon}?", "What are the {concept} implications of {process}?", "How do {element} and {element} interact in {system}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        soc_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(soc_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class AnthropologySpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["culture", "ethnography", "ritual", "kinship", "tradition", "belief", "custom", "ceremony", "anthropology", "fieldwork", "participant observation", "cultural relativism", "ethnocentrism", "acculturation", "diffusion", "evolution", "adaptation", "archaeology", "linguistics", "primatology"]

    def get_question_patterns(self) -> List[str]:
        return ["How do {factor} reflect {concept}?", "What is the significance of {term} in {system}?", "How do anthropologists study {phenomenon} in {example}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        anthro_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(anthro_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class PoliticalScienceSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["government", "politics", "policy", "democracy", "republic", "sovereignty", "citizenship", "voting", "election", "representation", "legislation", "executive", "judicial", "federalism", "diplomacy", "international relations", "political theory", "public administration", "comparative politics", "political behavior"]

    def get_question_patterns(self) -> List[str]:
        return ["How does {system} affect {process}?", "What are the {concept} consequences of {factor}?", "How do {element} influence {phenomenon} in {system}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        polsci_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(polsci_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class LawSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["statute", "regulation", "precedent", "jurisdiction", "contract", "tort", "criminal law", "civil law", "constitutional law", "procedure", "evidence", "trial", "court", "judge", "jury", "attorney", "legal reasoning", "interpretation", "justice", "rights"]

    def get_question_patterns(self) -> List[str]:
        return ["What are the legal implications of {factor}?", "How does {principle} apply to {system}?", "What {method} do courts use to determine {concept}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        law_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(law_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class MedicineSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["diagnosis", "treatment", "therapy", "medication", "surgery", "patient", "symptom", "disease", "disorder", "health", "prevention", "epidemiology", "pathology", "physiology", "anatomy", "clinical trial", "medical research", "evidence-based medicine", "healthcare", "public health"]

    def get_question_patterns(self) -> List[str]:
        return ["How is {term} diagnosed and treated?", "What are the {concept} effects of {factor}?", "How does {method} improve {system} outcomes?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        med_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(med_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class EngineeringSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["design", "construction", "materials", "structure", "system", "process", "efficiency", "optimization", "safety", "testing", "analysis", "modeling", "simulation", "project", "specification", "standard", "quality", "performance", "maintenance", "innovation"]

    def get_question_patterns(self) -> List[str]:
        return ["How can {system} be designed for optimal {factor}?", "What engineering principles apply to {process}?", "How do {element} affect {system} performance?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        eng_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(eng_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class BusinessSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["management", "strategy", "marketing", "finance", "operations", "leadership", "organization", "profit", "revenue", "customer", "market", "competition", "innovation", "entrepreneurship", "investment", "risk", "planning", "analysis", "decision making", "performance"]

    def get_question_patterns(self) -> List[str]:
        return ["How can {concept} improve {system} performance?", "What {method} should be used to achieve {factor}?", "How does {element} impact {process} outcomes?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        biz_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(biz_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class EnvironmentalScienceSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["environment", "ecosystem", "conservation", "pollution", "sustainability", "climate", "biodiversity", "habitat", "species", "renewable", "carbon", "emissions", "greenhouse", "recycling", "waste", "energy", "water", "air", "soil", "ecology"]

    def get_question_patterns(self) -> List[str]:
        return ["How does {factor} impact environmental {system}?", "What are the environmental effects of {process}?", "How can {method} address {phenomenon}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        env_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(env_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class LinguisticsSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["language", "grammar", "syntax", "semantics", "phonetics", "morphology", "linguistics", "phoneme", "morpheme", "lexicon", "dialect", "sociolinguistics", "psycholinguistics", "pragmatics", "discourse", "etymology", "orthography", "phonology", "bilingual", "multilingual"]

    def get_question_patterns(self) -> List[str]:
        return ["How does {concept} influence {phenomenon}?", "What role does {element} play in {process}?", "How do {factor} affect {system}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        linguistic_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(linguistic_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


# Domain specialist registry
DOMAIN_SPECIALISTS = {
    "science": ScienceSpecialist,
    "literature": LiteratureSpecialist,
    "mathematics": MathematicsSpecialist,
    "history": HistorySpecialist,
    "technology": TechnologySpecialist,
    "psychology": PsychologySpecialist,
    "economics": EconomicsSpecialist,
    "philosophy": PhilosophySpecialist,
    "art": ArtSpecialist,
    "music": MusicSpecialist,
    "geography": GeographySpecialist,
    "sociology": SociologySpecialist,
    "anthropology": AnthropologySpecialist,
    "political_science": PoliticalScienceSpecialist,
    "law": LawSpecialist,
    "medicine": MedicineSpecialist,
    "engineering": EngineeringSpecialist,
    "business": BusinessSpecialist,
    "environmental_science": EnvironmentalScienceSpecialist,
    "linguistics": LinguisticsSpecialist,
}


def get_available_domains() -> List[str]:
    """Get list of all available domain names."""
    return list(DOMAIN_SPECIALISTS.keys())


def create_domain_specialist(domain: str) -> BaseDomainSpecialist:
    """Create a specialist instance for the specified domain."""
    if domain not in DOMAIN_SPECIALISTS:
        raise ValueError(f"No specialist available for domain: {domain}")
    return DOMAIN_SPECIALISTS[domain]()


def generate_domain_specific_questions(domains: List[str], questions_per_domain: int = 3) -> Dict[str, List[str]]:
    """Generate questions for multiple domains."""
    results = {}

    for domain in domains:
        try:
            specialist = create_domain_specialist(domain)
            questions = specialist.generate_questions(count=questions_per_domain)
            results[domain] = questions
        except Exception as e:
            logger.error(f"Failed to generate questions for domain {domain}: {e}")
            results[domain] = []

    return results


class DomainSpecializationFramework:
    """Main framework for managing domain specialists and question generation."""

    def __init__(self):
        """Initialize the framework with all available specialists."""
        self.specialists = {}
        for domain, specialist_class in DOMAIN_SPECIALISTS.items():
            try:
                self.specialists[domain] = specialist_class()
            except Exception as e:
                logger.error(f"Failed to initialize specialist for {domain}: {e}")

    def get_specialist(self, domain: str) -> BaseDomainSpecialist:
        """Get a specialist for the specified domain."""
        if domain not in self.specialists:
            raise ValueError(f"No specialist available for domain: {domain}")
        return self.specialists[domain]

    def generate_questions_for_all_domains(self, questions_per_domain: int = 3) -> Dict[str, List[str]]:
        """Generate questions for all available domains."""
        return generate_domain_specific_questions(
            domains=get_available_domains(),
            questions_per_domain=questions_per_domain
        )
