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

        # Track used terms to reduce repetition
        used_terms = set()
        max_attempts = count * 3  # Limit attempts to avoid infinite loops
        attempts = 0

        while len(questions) < count and attempts < max_attempts:
            pattern = random.choice(patterns)
            question = self._fill_pattern(pattern, vocabulary, used_terms)

            # Ensure question ends with question mark
            if not question.strip().endswith('?'):
                question = question.strip() + '?'

            # Avoid exact duplicates
            if question not in questions:
                questions.append(question)
                # Update used terms from this question
                words = question.lower().split()
                used_terms.update(word.strip('.,!?;:') for word in words if len(word) > 3)

            attempts += 1

        return questions

    def _fill_pattern(self, pattern: str, vocabulary: List[str], used_terms: Optional[set] = None) -> str:
        """Fill a question pattern with domain-specific terms, avoiding repetition."""
        if not vocabulary:
            vocabulary = ['concept', 'term', 'process', 'phenomenon', 'principle', 'method', 'system', 'element', 'factor', 'example']

        if used_terms is None:
            used_terms = set()

        def smart_choice(candidates, category='general', fallback='term'):
            """Intelligently choose from candidates, avoiding recently used terms."""
            if not candidates:
                candidates = vocabulary[:10]

            # Filter out recently used terms
            fresh_candidates = [c for c in candidates if c.lower() not in used_terms]

            # If we've used too many terms, reset and use all candidates
            if not fresh_candidates and len(used_terms) > len(vocabulary) * 0.7:
                fresh_candidates = candidates

            if fresh_candidates:
                return random.choice(fresh_candidates)
            elif candidates:
                return random.choice(candidates)
            else:
                return fallback

        # Categorize vocabulary for better matching
        vocab_categories = {
            'processes': [v for v in vocabulary if any(x in v.lower() for x in ['tion', 'ing', 'ment', 'sis', 'lysis', 'synthesis', 'cycle', 'process', 'method'])],
            'objects': [v for v in vocabulary if any(x in v.lower() for x in ['cell', 'organ', 'system', 'structure', 'atom', 'molecule', 'particle'])],
            'concepts': [v for v in vocabulary if any(x in v.lower() for x in ['theory', 'principle', 'concept', 'law', 'hypothesis', 'model'])],
            'methods': [v for v in vocabulary if any(x in v.lower() for x in ['method', 'technique', 'approach', 'analysis', 'research', 'study', 'experiment'])],
            'properties': [v for v in vocabulary if any(x in v.lower() for x in ['property', 'characteristic', 'feature', 'energy', 'matter', 'force'])],
            'phenomena': [v for v in vocabulary if any(x in v.lower() for x in ['effect', 'phenomenon', 'behavior', 'event', 'reaction', 'evolution', 'growth'])]
        }

        # Enhanced placeholder mapping with smarter categorization
        placeholders = {
            '{concept}': smart_choice(vocab_categories.get('concepts', vocabulary[:15]), 'concept', 'concept'),
            '{term}': smart_choice(vocabulary[:25], 'term', 'term'),
            '{process}': smart_choice(vocab_categories.get('processes', vocabulary[:20]), 'process', 'process'),
            '{phenomenon}': smart_choice(vocab_categories.get('phenomena', [v for v in vocabulary if any(x in v.lower() for x in ['effect', 'phenomenon', 'behavior', 'event'])]), 'phenomenon', 'phenomenon'),
            '{principle}': smart_choice(vocab_categories.get('concepts', vocabulary[:20]), 'principle', 'principle'),
            '{method}': smart_choice(vocab_categories.get('methods', vocabulary[:15]), 'method', 'method'),
            '{system}': smart_choice(vocab_categories.get('objects', vocabulary[:20]), 'system', 'system'),
            '{element}': smart_choice(vocabulary[:30], 'element', 'element'),
            '{factor}': smart_choice([v for v in vocabulary if any(x in v.lower() for x in ['factor', 'element', 'component', 'variable'])], 'factor', 'factor'),
            '{example}': smart_choice(vocabulary[:20], 'example', 'example'),
            '{theory}': smart_choice(vocab_categories.get('concepts', vocabulary[:10]), 'theory', 'theory'),
            '{property}': smart_choice(vocab_categories.get('properties', vocabulary[:15]), 'property', 'property'),
            '{structure}': smart_choice(vocab_categories.get('objects', vocabulary[:15]), 'structure', 'structure'),
            '{function}': smart_choice([v for v in vocabulary if 'function' in v.lower() or v.lower().endswith('ing')], 'function', 'function'),
            '{character}': smart_choice([v for v in vocabulary if 'character' in v.lower()], 'character', 'character'),
            '{theme}': smart_choice([v for v in vocabulary if 'theme' in v.lower()], 'theme', 'theme'),
            '{setting}': smart_choice([v for v in vocabulary if 'setting' in v.lower()], 'setting', 'setting'),
            '{narrative}': smart_choice([v for v in vocabulary if 'narrative' in v.lower()], 'narrative', 'narrative'),
            '{symbolism}': smart_choice([v for v in vocabulary if any(x in v.lower() for x in ['symbol', 'metaphor', 'imagery'])], 'symbolism', 'symbolism'),
            '{literary_devices}': smart_choice([v for v in vocabulary if any(x in v.lower() for x in ['device', 'technique', 'style'])], 'literary_devices', 'literary devices'),
            '{mood}': smart_choice([v for v in vocabulary if any(x in v.lower() for x in ['mood', 'tone', 'atmosphere'])], 'mood', 'mood'),
            '{message}': smart_choice([v for v in vocabulary if any(x in v.lower() for x in ['message', 'meaning', 'theme'])], 'message', 'message'),
            '{conflict}': smart_choice([v for v in vocabulary if 'conflict' in v.lower()], 'conflict', 'conflict'),
            '{plot}': smart_choice([v for v in vocabulary if 'plot' in v.lower()], 'plot', 'plot'),
            '{variable}': smart_choice([v for v in vocabulary if 'variable' in v.lower()], 'variable', 'variable'),
            '{value}': smart_choice([v for v in vocabulary if any(x in v.lower() for x in ['value', 'number', 'constant'])], 'value', 'value'),
            '{theorem}': smart_choice([v for v in vocabulary if any(x in v.lower() for x in ['theorem', 'law', 'principle'])], 'theorem', 'theorem'),
            '{domain}': smart_choice([v for v in vocabulary if 'domain' in v.lower()], 'domain', 'domain'),
            '{constraint}': smart_choice([v for v in vocabulary if any(x in v.lower() for x in ['constraint', 'condition', 'limit'])], 'constraint', 'constraint'),
            '{result}': smart_choice([v for v in vocabulary if any(x in v.lower() for x in ['result', 'outcome', 'effect'])], 'result', 'result'),
            '{extremum}': smart_choice([v for v in vocabulary if any(x in v.lower() for x in ['max', 'min', 'extreme', 'optimal'])], 'extremum', 'extremum'),
            '{probability}': smart_choice([v for v in vocabulary if 'probability' in v.lower()], 'probability', 'probability'),
            '{event}': smart_choice([v for v in vocabulary if any(x in v.lower() for x in ['event', 'outcome', 'result'])], 'event', 'event'),
            '{condition}': smart_choice([v for v in vocabulary if 'condition' in v.lower()], 'condition', 'condition'),
            '{geometry}': smart_choice([v for v in vocabulary if any(x in v.lower() for x in ['geometry', 'shape', 'figure'])], 'geometry', 'geometry'),
            '{parameter}': smart_choice([v for v in vocabulary if any(x in v.lower() for x in ['parameter', 'variable', 'factor'])], 'parameter', 'parameter'),
            '{observation}': smart_choice([v for v in vocabulary if any(x in v.lower() for x in ['observation', 'data', 'evidence', 'measurement'])], 'observation', 'observation'),
            '{conclusion}': smart_choice([v for v in vocabulary if any(x in v.lower() for x in ['conclusion', 'result', 'finding', 'outcome'])], 'conclusion', 'conclusion'),
            '{hypothesis}': smart_choice([v for v in vocabulary if any(x in v.lower() for x in ['hypothesis', 'theory', 'prediction'])], 'hypothesis', 'hypothesis'),
            '{outcome}': smart_choice([v for v in vocabulary if any(x in v.lower() for x in ['outcome', 'result', 'effect', 'consequence'])], 'outcome', 'outcome')
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
        return [
            "hypothesis", "experiment", "theory", "data", "observation", "analysis", "method", "research",
            "evidence", "conclusion", "variable", "control", "measurement", "peer review", "scientific method",
            "replication", "laboratory", "empirical", "quantitative", "qualitative", "atom", "molecule",
            "element", "compound", "reaction", "catalyst", "enzyme", "photosynthesis", "mitosis", "DNA",
            "RNA", "protein", "cell", "organism", "evolution", "natural selection", "genetics", "chromosome",
            "nucleus", "electron", "proton", "neutron", "energy", "force", "gravity", "acceleration",
            "velocity", "mass", "weight", "density", "temperature", "pressure", "volume", "conservation",
            "thermodynamics", "entropy", "equilibrium", "solution", "precipitation", "oxidation", "reduction",
            "acid", "base", "acidity", "buffer", "titration", "spectroscopy", "chromatography", "microscopy",
            "telescope", "laser", "radiation", "wavelength", "frequency", "spectrum", "particle", "wave",
            "quantum", "relativity", "ecosystem", "biodiversity", "habitat", "population", "community",
            "food chain", "carbon cycle", "nitrogen cycle", "climate", "weather", "geology", "fossil",
            "sediment", "igneous", "metamorphic", "plate tectonics", "volcano", "earthquake", "mineral"
        ]

    def get_question_patterns(self) -> List[str]:
        return [
            "How does {factor} affect {phenomenon}?",
            "What is the relationship between {concept} and {term}?",
            "Why does {process} occur in {system}?",
            "What experimental evidence supports {theory}?",
            "How can we measure energy in {system}?",
            "What role does {element} play in {process}?",
            "How do environmental {factor}s influence matter?",
            "What are the molecular mechanisms underlying {process}?",
            "How does research help us understand {phenomenon}?",
            "What predictions can {theory} make about {system}?",
            "How does energy transformation occur in {process}?",
            "What experimental design would test {hypothesis}?",
            "How do scientists study {phenomenon} in the laboratory?",
            "What analysis methods reveal {concept}?",
            "How does theory explain {observation}?",
            "What research findings support {conclusion}?",
            "How does energy flow through {system}?",
            "What experimental conditions affect {process}?",
            "How can research methods study {phenomenon}?",
            "What theoretical analysis explains {concept}?",
            "How does matter behave in {system}?",
            "What experimental approach studies {process}?",
            "How does scientific theory predict {outcome}?",
            "What analysis techniques study {phenomenon}?",
            "How does energy conversion affect {system}?",
            "What research methods analyze {concept}?"
        ]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip():
            return 0.0

        text_lower = text.lower()
        vocabulary = [term.lower() for term in self.get_domain_vocabulary()]

        # Count direct vocabulary matches
        vocab_matches = sum(1 for term in vocabulary if term in text_lower)

        # Count scientific indicators
        science_indicators = ['scientific', 'research', 'study', 'analysis', 'experimental', 'biological',
                             'chemical', 'physical', 'theoretical', 'laboratory', 'molecular', 'cellular',
                             'process', 'converts', 'using', 'carbon', 'dioxide', 'glucose', 'sunlight']
        indicator_matches = sum(1 for indicator in science_indicators if indicator in text_lower)

        # Count scientific concepts (partial word matches)
        concept_indicators = ['photo', 'synthesis', 'carbon', 'dioxide', 'oxygen', 'hydrogen', 'nitrogen',
                             'enzyme', 'protein', 'genetic', 'atomic', 'molecular', 'cellular', 'organic',
                             'chemical', 'biological', 'physical', 'thermal', 'electric', 'magnetic']
        concept_matches = sum(1 for concept in concept_indicators if concept in text_lower)

        # Calculate score based on matches and text length
        total_matches = vocab_matches + indicator_matches + concept_matches
        words = len(text.split())

        if words == 0:
            return 0.0

        # More generous scoring for scientific content
        # Base score from matches, with bonus for multiple match types
        base_score = min(1.0, (total_matches * 1.5) / max(words, 3))

        # Bonus for having both vocabulary and indicator/concept matches
        if vocab_matches > 0 and (indicator_matches + concept_matches) > 0:
            base_score = min(1.0, base_score * 1.3)

        return base_score

    def generate_questions(self, count: int = 5, context: Optional[str] = None) -> List[str]:
        """Generate domain-specific questions with preference for scientific terminology."""
        questions = []
        vocabulary = self.get_domain_vocabulary()
        patterns = self.get_question_patterns()

        # Track used terms to reduce repetition
        used_terms = set()
        max_attempts = count * 3  # Limit attempts to avoid infinite loops
        attempts = 0

        # Prioritize patterns with scientific indicators
        science_indicators = ["energy", "matter", "experiment", "theory", "research", "study", "analysis"]
        priority_patterns = [p for p in patterns if any(indicator in p.lower() for indicator in science_indicators)]

        if priority_patterns:
            # Use 80% priority patterns, 20% other patterns for variety
            pattern_pool = priority_patterns * 4 + [p for p in patterns if p not in priority_patterns]
        else:
            pattern_pool = patterns

        while len(questions) < count and attempts < max_attempts:
            pattern = random.choice(pattern_pool)
            question = self._fill_pattern(pattern, vocabulary, used_terms)

            # Ensure question ends with question mark
            if not question.strip().endswith('?'):
                question = question.strip() + '?'

            # Avoid exact duplicates
            if question not in questions:
                questions.append(question)
                # Update used terms from this question
                words = question.lower().split()
                used_terms.update(word.strip('.,!?;:') for word in words if len(word) > 3)

            attempts += 1

        return questions


class LiteratureSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return [
            "narrative", "character", "plot", "theme", "symbolism", "metaphor", "irony", "author",
            "protagonist", "antagonist", "conflict", "setting", "genre", "poetry", "prose", "fiction",
            "analysis", "interpretation", "literary device", "criticism", "allegory", "allusion",
            "alliteration", "assonance", "consonance", "imagery", "personification", "simile",
            "hyperbole", "oxymoron", "paradox", "satire", "parody", "tragedy", "comedy", "drama",
            "epic", "sonnet", "haiku", "ballad", "ode", "elegy", "villanelle", "sestina", "pantoum",
            "free verse", "blank verse", "meter", "rhythm", "rhyme", "stanza", "verse", "line",
            "caesura", "enjambment", "anaphora", "epistrophe", "chiasmus", "zeugma", "synecdoche",
            "metonymy", "euphemism", "litotes", "meiosis", "amplification", "climax", "anticlimax",
            "denouement", "exposition", "rising action", "falling action", "flashback", "foreshadowing",
            "stream of consciousness", "point of view", "first person", "third person", "omniscient",
            "limited omniscient", "unreliable narrator", "soliloquy", "monologue", "dialogue",
            "characterization", "round character", "flat character", "dynamic character", "static character",
            "foil", "archetype", "bildungsroman", "picaresque", "romance", "gothic", "realism",
            "naturalism", "modernism", "postmodernism", "structuralism", "deconstruction"
        ]

    def get_question_patterns(self) -> List[str]:
        return [
            "How does {character} development contribute to the overall {theme}?",
            "What is the significance of {symbolism} in the author's narrative?",
            "How does the author use {metaphor} to convey the central {theme}?",
            "What literary devices enhance the character analysis in this work?",
            "How does the {narrative} structure represent the author's theme?",
            "What role does character development play in advancing the {theme}?",
            "How does the narrative structure affect reader's thematic analysis?",
            "What symbolism reinforces the central character development?",
            "How do metaphors create thematic resonance in the narrative?",
            "What character conflicts drive the thematic analysis?"
        ]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip():
            return 0.0

        text_lower = text.lower()
        vocabulary = [term.lower() for term in self.get_domain_vocabulary()]

        # Count direct vocabulary matches
        vocab_matches = sum(1 for term in vocabulary if term in text_lower)

        # Count literary indicators
        lit_indicators = ['literary', 'poem', 'novel', 'story', 'text', 'writing', 'author',
                         'reader', 'interpretation', 'meaning', 'style', 'language']
        indicator_matches = sum(1 for indicator in lit_indicators if indicator in text_lower)

        # Calculate score
        total_matches = vocab_matches + indicator_matches
        words = len(text.split())

        if words == 0:
            return 0.0

        relevance = min(1.0, (total_matches * 2.0) / max(words, 5))
        return relevance


class MathematicsSpecialist(BaseDomainSpecialist):
    def get_domain_vocabulary(self) -> List[str]:
        return [
            "equation", "function", "variable", "coefficient", "derivative", "integral", "theorem",
            "proof", "algorithm", "formula", "geometry", "algebra", "calculus", "statistics",
            "probability", "matrix", "vector", "polynomial", "logarithm", "trigonometry", "sine",
            "cosine", "tangent", "hyperbola", "parabola", "ellipse", "circle", "triangle", "square",
            "rectangle", "polygon", "angle", "radius", "diameter", "circumference", "area", "volume",
            "perimeter", "congruent", "similar", "parallel", "perpendicular", "intersection", "union",
            "set", "subset", "element", "domain", "range", "limit", "infinity", "continuous",
            "discrete", "sequence", "series", "convergent", "divergent", "asymptote", "extremum",
            "maximum", "minimum", "optimization", "constraint", "linear", "nonlinear", "quadratic",
            "cubic", "exponential", "rational", "irrational", "complex", "real", "integer", "natural",
            "prime", "composite", "factor", "multiple", "divisible", "modulo", "arithmetic",
            "geometric", "harmonic", "mean", "median", "mode", "variance", "deviation", "correlation",
            "regression", "hypothesis", "null hypothesis", "p-value", "confidence", "interval",
            "distribution", "normal", "binomial", "poisson", "chi-square", "t-test", "ANOVA"
        ]

    def get_question_patterns(self) -> List[str]:
        return [
            "What is the {concept} of {element}?",
            "How do you solve for x in the equation 2x + 5 = 17?",
            "What is the relationship between {factor} and {principle}?",
            "Calculate f(3) when f(x) = x² + 2x - 1",
            "Prove that ∑(k=1 to n) k = n(n+1)/2",
            "What conditions ensure f(x) is {property}?",
            "Find the derivative of f(x) = 3x² + 5x - 7",
            "Evaluate the integral ∫ (2x + 3) dx from 0 to 5",
            "Solve the system: x + y = 10, 2x - y = 5",
            "What is the probability P(A∩B) when P(A) = 0.4 and P(B) = 0.6?",
            "Calculate the limit: lim(x→∞) (3x² + 2)/(x² + 1)",
            "Find the roots of x² - 6x + 9 = 0",
            "Determine if the series ∑(1/n²) converges or diverges",
            "What is the area under y = x² from x = 0 to x = 4?"
        ]

    def validate_domain_relevance(self, text: str) -> float:
        if not text.strip():
            return 0.0

        text_lower = text.lower()
        vocabulary = [term.lower() for term in self.get_domain_vocabulary()]

        # Count direct vocabulary matches
        vocab_matches = sum(1 for term in vocabulary if term in text_lower)

        # Count mathematical indicators
        math_indicators = ['mathematical', 'numeric', 'calculation', 'compute', 'solve', 'prove',
                          'formula', 'number', 'value', 'equal', 'greater', 'less', 'sum', 'product']
        indicator_matches = sum(1 for indicator in math_indicators if indicator in text_lower)

        # Look for mathematical symbols and notation patterns
        import re
        math_symbols = len(re.findall(r'[=+\-*/^<>≤≥∞∑∏∫∂∇]', text))

        # Calculate score
        total_matches = vocab_matches + indicator_matches + math_symbols
        words = len(text.split())

        if words == 0:
            return 0.0

        relevance = min(1.0, (total_matches * 2.0) / max(words, 5))
        return relevance


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
        raise ValueError(f"Unknown domain: {domain}")
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
