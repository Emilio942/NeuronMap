"""
Domain Specialization Framework for high-quality, domain-specific question generation.

This module implements specialized question generators for 20+ domains, each with:
- Domain-specific vocabulary and terminology
- Question patterns optimized for the domain
- Domain relevance validation
- Quality assessment integration

Section 4.1 implementation for NeuronMap project.
"""

import random
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseDomainSpecialist(ABC):
    """Abstract base class for domain-specific question generators."""

    def __init__(self):
        self.domain = self.__class__.__name__.replace('Specialist', '').lower()

    @abstractmethod
    def get_domain_vocabulary(self) -> List[str]:
        """Get domain-specific vocabulary terms."""
        pass

    @abstractmethod
    def get_question_patterns(self) -> List[str]:
        """Get domain-specific question patterns."""
        pass

    @abstractmethod
    def validate_domain_relevance(self, text: str) -> float:
        """Validate relevance of text to this domain (0.0-1.0)."""
        pass

    def generate_questions(self, count: int = 3) -> List[str]:
        """Generate domain-specific questions."""
        if count <= 0:
            return []

        questions = []
        patterns = self.get_question_patterns()
        vocabulary = self.get_domain_vocabulary()

        for _ in range(count):
            pattern = random.choice(patterns)
            question = self._fill_pattern(pattern, vocabulary)
            if question and len(question.strip()) > 10:
                questions.append(question)

        return questions

    def _fill_pattern(self, pattern: str, vocabulary: List[str]) -> str:
        """Fill a question pattern with domain-specific terms."""
        # Generic placeholder mapping
        placeholders = {
            '{concept}': random.choice(vocabulary[:20]) if vocabulary else 'concept',
            '{term}': random.choice(vocabulary[:30]) if vocabulary else 'term',
            '{process}': random.choice([v for v in vocabulary if 'process' in v.lower() or 'tion' in v][:10]) if vocabulary else 'process',
            '{phenomenon}': random.choice([v for v in vocabulary if any(x in v.lower() for x in ['effect', 'phenomenon', 'behavior'])][:10]) if vocabulary else 'phenomenon',
            '{principle}': random.choice([v for v in vocabulary if any(x in v.lower() for x in ['law', 'principle', 'rule', 'theory'])][:10]) if vocabulary else 'principle',
            '{method}': random.choice([v for v in vocabulary if any(x in v.lower() for x in ['method', 'technique', 'approach'])][:10]) if vocabulary else 'method',
            '{system}': random.choice([v for v in vocabulary if 'system' in v.lower()][:10]) if vocabulary else 'system',
            '{element}': random.choice(vocabulary[:40]) if vocabulary else 'element',
            '{factor}': random.choice([v for v in vocabulary if any(x in v.lower() for x in ['factor', 'element', 'component'])][:10]) if vocabulary else 'factor',
            '{example}': random.choice(vocabulary[:25]) if vocabulary else 'example',
        }

        # Fill pattern with placeholders
        filled = pattern
        for placeholder, value in placeholders.items():
            if placeholder in filled:
                filled = filled.replace(placeholder, value)

        return filled


class ScienceSpecialist(BaseDomainSpecialist):
    """Science domain specialist for physics, chemistry, biology questions."""

    def get_domain_vocabulary(self) -> List[str]:
        return [
            "atom", "molecule", "electron", "neutron", "proton", "nucleus", "energy", "matter", "force",
            "acceleration", "velocity", "mass", "density", "pressure", "temperature", "heat", "light",
            "photosynthesis", "DNA", "RNA", "protein", "enzyme", "cell", "organism", "evolution",
            "ecosystem", "biodiversity", "gravity", "electromagnetism", "radiation", "chemical reaction",
            "catalyst", "solution", "compound", "element", "periodic table", "hypothesis", "experiment",
            "observation", "theory", "research", "analysis", "scientific method", "data", "measurement",
            "carbon dioxide", "oxygen", "hydrogen", "chlorophyll", "metabolism", "genetics", "mutation",
            "natural selection", "adaptation", "species", "conservation", "entropy", "thermodynamics",
            "quantum mechanics", "wave", "frequency", "amplitude", "electromagnetic spectrum", "laser"
        ]

    def get_question_patterns(self) -> List[str]:
        return [
            "What is the role of {concept} in biological processes?",
            "How does {process} affect {system} behavior?",
            "Explain the relationship between {term} and energy transfer.",
            "What factors influence {phenomenon} in natural systems?",
            "How do scientists measure {concept} in laboratory experiments?",
            "What is the significance of {principle} in modern physics?",
            "Compare the properties of {element} and {term}.",
            "How does {method} help us understand {concept}?",
            "What experimental evidence supports the theory of {phenomenon}?",
            "Analyze the environmental impact of {process} on ecosystems.",
            "What role does {concept} play in {system} function?",
            "How do changes in {factor} affect {process} efficiency?"
        ]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip():
            return 0.0

        science_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())

        overlap = len(science_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class LiteratureSpecialist(BaseDomainSpecialist):
    """Literature domain specialist for literary analysis and criticism."""

    def get_domain_vocabulary(self) -> List[str]:
        return [
            "metaphor", "symbolism", "narrative", "protagonist", "antagonist", "character", "theme",
            "plot", "setting", "conflict", "climax", "resolution", "irony", "foreshadowing",
            "allegory", "alliteration", "imagery", "personification", "simile", "tone", "mood",
            "author", "narrator", "point of view", "dialogue", "monologue", "genre", "style",
            "literary device", "motif", "archetype", "tragedy", "comedy", "drama", "poetry",
            "prose", "verse", "stanza", "rhyme", "meter", "rhythm", "allusion", "paradox",
            "satire", "parody", "critique", "analysis", "interpretation", "context", "meaning",
            "significance", "structure", "form", "technique", "expression", "voice", "perspective"
        ]

    def get_question_patterns(self) -> List[str]:
        return [
            "Analyze the use of {concept} in contemporary literature.",
            "How does the {term} contribute to the overall theme of the work?",
            "What is the significance of {element} in character development?",
            "Compare the {concept} techniques used by different authors.",
            "How does {device} enhance the reader's understanding of {theme}?",
            "What role does {setting} play in developing the {conflict}?",
            "Examine the relationship between {character} and {symbol}.",
            "How do authors use {technique} to convey {meaning}?",
            "What makes {genre} an effective vehicle for exploring {theme}?",
            "Analyze the {perspective} in relation to the work's central message.",
            "How does {style} reflect the historical context of the period?",
            "What literary techniques create emotional effect in the reader?"
        ]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip():
            return 0.0

        lit_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())

        overlap = len(lit_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class MathematicsSpecialist(BaseDomainSpecialist):
    """Mathematics domain specialist for mathematical concepts and problems."""

    def get_domain_vocabulary(self) -> List[str]:
        return [
            "equation", "function", "variable", "constant", "coefficient", "polynomial", "derivative",
            "integral", "limit", "theorem", "proof", "axiom", "lemma", "corollary", "algorithm",
            "matrix", "vector", "scalar", "determinant", "eigenvalue", "transformation", "geometry",
            "algebra", "calculus", "statistics", "probability", "set", "subset", "union", "intersection",
            "graph", "vertex", "edge", "topology", "manifold", "differential", "sequence", "series",
            "convergence", "divergence", "optimization", "minimum", "maximum", "critical point",
            "domain", "range", "mapping", "bijection", "injection", "surjection", "isomorphism",
            "group", "ring", "field", "space", "metric", "norm", "distance", "continuity"
        ]

    def get_question_patterns(self) -> List[str]:
        return [
            "Prove that {concept} satisfies the properties of mathematical structure.",
            "Calculate the derivative of the {function} with respect to variable.",
            "What is the geometric interpretation of {concept} in space?",
            "Find the critical points of the {function} and classify them.",
            "How does {theorem} apply to solving this type of problem?",
            "Determine the convergence properties of the {series}.",
            "What is the relationship between {concept} and mathematical context?",
            "Solve the {equation} using this method and verify your solution.",
            "Explain why {theorem} is fundamental to this area of mathematics.",
            "How can {algorithm} be used to compute mathematical object?",
            "What are the applications of {concept} in applied field?",
            "Derive the formula from first principles using method."
        ]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip():
            return 0.0

        math_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())

        # Also check for mathematical symbols and numbers
        math_symbols = re.findall(r'[+\-*/=<>∫∂∇∆Σ∏√∞π]|\d+', text)
        symbol_bonus = min(0.3, len(math_symbols) * 0.05)

        overlap = len(math_terms.intersection(words))
        base_score = min(1.0, overlap / max(1, len(words) * 0.1))

        return min(1.0, base_score + symbol_bonus)


# Create simplified specialists for the remaining 16 domains
class HistorySpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["civilization", "empire", "war", "revolution", "ancient", "medieval", "modern", "century", "culture", "politics", "society", "economic", "social", "cultural", "historical", "period", "era", "timeline", "primary source", "secondary source"]

    def get_question_patterns(self) -> List[str]:
        return ["What were the causes of {concept}?", "How did {factor} influence {civilization}?", "What role did {element} play in {period}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        hist_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(hist_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class TechnologySpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["algorithm", "software", "hardware", "computer", "programming", "data", "network", "internet", "artificial intelligence", "machine learning", "database", "application", "system", "development", "technology", "digital", "innovation", "automation", "cybersecurity", "cloud"]

    def get_question_patterns(self) -> List[str]:
        return ["How does {algorithm} improve {system}?", "What are the advantages of {technology}?", "How can {concept} be implemented?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        tech_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(tech_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class PsychologySpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["behavior", "cognition", "emotion", "memory", "learning", "personality", "therapy", "mental health", "brain", "psychology", "research", "study", "analysis", "cognitive", "social", "clinical", "developmental", "experimental", "consciousness", "motivation"]

    def get_question_patterns(self) -> List[str]:
        return ["How does {concept} influence {behavior}?", "What factors contribute to {phenomenon}?", "How can {therapy} help treat {condition}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        psych_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(psych_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class EconomicsSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["market", "supply", "demand", "price", "economy", "trade", "business", "finance", "investment", "economics", "economic", "financial", "monetary", "fiscal", "policy", "growth", "inflation", "recession", "competition", "profit"]

    def get_question_patterns(self) -> List[str]:
        return ["How does {concept} affect {indicator}?", "What is the relationship between {factor1} and {factor2}?", "How can {policy} address {problem}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        econ_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(econ_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class PhilosophySpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["ethics", "morality", "logic", "truth", "knowledge", "reality", "existence", "philosophy", "philosophical", "theory", "principle", "concept", "argument", "reasoning", "metaphysics", "epistemology", "consciousness", "belief", "justice", "virtue"]

    def get_question_patterns(self) -> List[str]:
        return ["What are the implications of {concept}?", "How does {theory} approach {issue}?", "What is the structure of {argument}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        phil_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(phil_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class ArtSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["art", "painting", "sculpture", "artist", "creative", "visual", "aesthetic", "design", "style", "technique", "composition", "color", "form", "expression", "artwork", "gallery", "museum", "artistic", "culture", "beauty"]

    def get_question_patterns(self) -> List[str]:
        return ["What techniques are used in {concept}?", "How does {element} contribute to {composition}?", "What is the significance of {style}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        art_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(art_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class MusicSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["music", "sound", "melody", "rhythm", "harmony", "musical", "instrument", "song", "composition", "performance", "audio", "note", "chord", "genre", "artist", "musician", "concert", "recording", "acoustic", "electronic"]

    def get_question_patterns(self) -> List[str]:
        return ["How does {concept} create {effect}?", "What role does {element} play in {composition}?", "How has {genre} evolved?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        music_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(music_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class GeographySpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["geography", "location", "region", "climate", "environment", "landscape", "population", "city", "country", "continent", "map", "geographic", "spatial", "physical", "human", "cultural", "economic", "political", "natural", "urban"]

    def get_question_patterns(self) -> List[str]:
        return ["What are the features of {region}?", "How does {factor} influence {phenomenon}?", "What is the relationship between {location} and {characteristic}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        geo_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(geo_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class SociologySpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["society", "social", "community", "culture", "group", "institution", "sociology", "sociological", "organization", "structure", "class", "inequality", "family", "education", "religion", "politics", "government", "law", "norm", "value"]

    def get_question_patterns(self) -> List[str]:
        return ["How does {concept} affect {outcome}?", "What factors contribute to {phenomenon}?", "How do {institutions} shape {behavior}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        soc_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(soc_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class AnthropologySpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["anthropology", "culture", "human", "society", "cultural", "social", "evolution", "archaeology", "ethnography", "ritual", "tradition", "language", "symbol", "belief", "custom", "artifact", "ancient", "primitive", "civilization", "tribe"]

    def get_question_patterns(self) -> List[str]:
        return ["How do {practices} reflect {values}?", "What can {artifacts} tell us about {society}?", "How has {behavior} evolved?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        anthro_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(anthro_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class PoliticalScienceSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["politics", "political", "government", "policy", "democracy", "authority", "power", "election", "voting", "law", "rights", "citizenship", "state", "nation", "international", "diplomacy", "public", "administration", "legislation", "constitution"]

    def get_question_patterns(self) -> List[str]:
        return ["How does {system} affect {outcome}?", "What factors influence {decision}?", "How do {institutions} shape {process}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        pol_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(pol_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class LawSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["law", "legal", "court", "justice", "rights", "legislation", "regulation", "statute", "contract", "criminal", "civil", "constitutional", "judicial", "lawyer", "attorney", "judge", "trial", "evidence", "case", "ruling"]

    def get_question_patterns(self) -> List[str]:
        return ["What are the implications of {concept}?", "How does {law} protect {rights}?", "What principles apply to {situation}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        law_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(law_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class MedicineSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["medicine", "medical", "health", "disease", "treatment", "patient", "doctor", "therapy", "diagnosis", "clinical", "healthcare", "pharmaceutical", "surgery", "prevention", "care", "hospital", "research", "study", "biological", "physiological"]

    def get_question_patterns(self) -> List[str]:
        return ["How is {condition} treated?", "What factors contribute to {disease}?", "How does {treatment} improve {outcome}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        med_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(med_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class EngineeringSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["engineering", "design", "construction", "technical", "system", "process", "technology", "innovation", "development", "project", "analysis", "solution", "problem", "efficiency", "optimization", "performance", "safety", "quality", "manufacturing", "industrial"]

    def get_question_patterns(self) -> List[str]:
        return ["How can {system} be designed for {performance}?", "What principles apply to {project}?", "How do {materials} affect {structure}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        eng_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(eng_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class BusinessSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["business", "management", "strategy", "marketing", "finance", "organization", "company", "corporate", "commercial", "economic", "profit", "revenue", "customer", "market", "competition", "leadership", "team", "operations", "growth", "success"]

    def get_question_patterns(self) -> List[str]:
        return ["How can {strategy} improve {performance}?", "What factors contribute to {outcome}?", "How do {forces} affect {result}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        bus_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(bus_terms.intersection(words))
        return min(1.0, overlap / max(1, len(words) * 0.1))


class EnvironmentalScienceSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return ["environment", "environmental", "ecology", "ecosystem", "conservation", "sustainability", "climate", "pollution", "natural", "biodiversity", "habitat", "species", "renewable", "energy", "carbon", "emissions", "green", "organic", "waste", "recycling"]

    def get_question_patterns(self) -> List[str]:
        return ["How does {factor} impact {system}?", "What are the effects of {process}?", "How can {solution} address {problem}?"]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip(): return 0.0
        env_terms = set(term.lower() for term in self.get_domain_vocabulary())
        words = set(word.lower().strip('.,!?;:') for word in text.split())
        overlap = len(env_terms.intersection(words))
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
}


def get_available_domains() -> List[str]:
    """Get list of all available domain names."""
    return list(DOMAIN_SPECIALISTS.keys())


def create_domain_specialist(domain: str) -> BaseDomainSpecialist:
    """Create a domain specialist instance for the given domain."""
    if domain not in DOMAIN_SPECIALISTS:
        raise ValueError(f"Unknown domain: {domain}. Available domains: {list(DOMAIN_SPECIALISTS.keys())}")

    specialist_class = DOMAIN_SPECIALISTS[domain]
    return specialist_class()


def generate_domain_specific_questions(domains: List[str], questions_per_domain: int = 3) -> Dict[str, List[str]]:
    """Generate questions for multiple domains."""
    results = {}

    for domain in domains:
        try:
            specialist = create_domain_specialist(domain)
            questions = specialist.generate_questions(count=questions_per_domain)
            results[domain] = questions
            logger.info(f"Generated {len(questions)} questions for domain: {domain}")
        except Exception as e:
            logger.error(f"Failed to generate questions for domain {domain}: {e}")
            results[domain] = []

    return results


class DomainSpecializationFramework:
    """Main framework for managing domain specialists."""

    def __init__(self):
        self.specialists = {}
        self._initialize_specialists()

    def _initialize_specialists(self):
        """Initialize all domain specialists."""
        for domain in get_available_domains():
            try:
                self.specialists[domain] = create_domain_specialist(domain)
                logger.debug(f"Initialized specialist for domain: {domain}")
            except Exception as e:
                logger.error(f"Failed to initialize specialist for domain {domain}: {e}")

    def get_specialist(self, domain: str) -> BaseDomainSpecialist:
        """Get a specialist for the given domain."""
        if domain not in self.specialists:
            raise ValueError(f"No specialist available for domain: {domain}")
        return self.specialists[domain]

    def generate_questions_for_all_domains(self, questions_per_domain: int = 3) -> Dict[str, List[str]]:
        """Generate questions for all available domains."""
        return generate_domain_specific_questions(
            domains=get_available_domains(),
            questions_per_domain=questions_per_domain
        )
