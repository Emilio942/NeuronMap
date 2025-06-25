"""
Physics Domain Specialist
=========================

Specialized question generator for physics domain with comprehensive terminology,
methodological rigor, and physics-specific reasoning patterns.
"""

import random
import re
from typing import Dict, List, Optional
from datetime import datetime

from ..domain_specialization_framework import (
    DomainSpecialist, DomainType, QuestionType, DomainVocabulary,
    DomainQuestion, DomainComplexityScore, DomainValidationResult
)
from ..difficulty_analyzer import assess_question_difficulty_fast


class PhysicsSpecialist(DomainSpecialist):
    """Domain specialist for physics questions with authentic physics reasoning."""

    def __init__(self):
        super().__init__("Physics", DomainType.STEM)

    def _initialize_vocabulary(self) -> DomainVocabulary:
        """Initialize physics-specific vocabulary and terminology."""
        return DomainVocabulary(
            core_terms=[
                "force", "energy", "momentum", "velocity", "acceleration", "mass", "charge",
                "field", "wave", "frequency", "amplitude", "wavelength", "particle", "photon",
                "electron", "proton", "neutron", "atom", "nucleus", "quantum", "relativity",
                "gravity", "electromagnetic", "thermodynamics", "entropy", "temperature",
                "pressure", "volume", "density", "mechanics", "dynamics", "kinematics"
            ],
            advanced_terms=[
                "hamiltonian", "lagrangian", "eigenvalue", "eigenstate", "superposition",
                "entanglement", "decoherence", "renormalization", "symmetry breaking",
                "gauge invariance", "feynman diagram", "scattering amplitude", "cross section",
                "phase transition", "critical point", "correlation function", "partition function",
                "green's function", "dispersion relation", "lorentz transformation",
                "minkowski spacetime", "riemann tensor", "christoffel symbol", "geodesic"
            ],
            methodology_terms=[
                "experiment", "measurement", "observation", "hypothesis", "theory", "model",
                "simulation", "calculation", "derivation", "approximation", "perturbation",
                "variational principle", "conservation law", "dimensional analysis",
                "order of magnitude", "statistical analysis", "uncertainty principle",
                "fourier transform", "laplace transform", "differential equation",
                "boundary condition", "initial condition", "symmetry analysis"
            ],
            concept_hierarchies={
                "mechanics": ["classical mechanics", "quantum mechanics", "statistical mechanics"],
                "fields": ["electric field", "magnetic field", "gravitational field", "gauge field"],
                "conservation": ["energy conservation", "momentum conservation", "charge conservation"],
                "interactions": ["strong force", "weak force", "electromagnetic force", "gravitational force"],
                "particles": ["fermions", "bosons", "quarks", "leptons", "gauge bosons"],
                "thermodynamics": ["first law", "second law", "third law", "zeroth law"]
            },
            synonyms={
                "velocity": ["speed", "rate"],
                "force": ["interaction"],
                "energy": ["work"],
                "particle": ["corpuscle"],
                "wave": ["oscillation"]
            }
        )

    def _initialize_question_patterns(self) -> Dict[QuestionType, List[str]]:
        """Initialize physics-specific question patterns."""
        return {
            QuestionType.CONCEPTUAL: [
                "Explain why {concept} behaves according to {principle} in {context}.",
                "What is the physical significance of {quantity} in the context of {phenomenon}?",
                "How does {principle} lead to the observed {behavior} in {system}?",
                "Describe the relationship between {concept1} and {concept2} in {framework}.",
                "Why does {symmetry} result in the conservation of {quantity}?"
            ],
            QuestionType.QUANTITATIVE: [
                "Calculate {quantity} for a {system} given {conditions} using {method}.",
                "Derive the expression for {observable} in {context} starting from {principle}.",
                "Estimate the {parameter} for {scenario} using dimensional analysis.",
                "Find the {solution} to the {equation} with {boundary_conditions}.",
                "Determine the {coefficient} from experimental data showing {relationship}."
            ],
            QuestionType.EXPERIMENTAL: [
                "Design an experiment to verify {hypothesis} while controlling for {variables}.",
                "What measurement technique would best determine {quantity} in {system}?",
                "How would you test the prediction that {theory} implies {consequence}?",
                "Propose a method to measure {parameter} with uncertainty less than {precision}.",
                "What experimental signature would distinguish {theory1} from {theory2}?"
            ],
            QuestionType.APPLICATION: [
                "Apply {theory} to predict the behavior of {system} under {conditions}.",
                "How would {principle} be utilized in the design of {device}?",
                "Use {method} to analyze the {phenomenon} observed in {context}.",
                "What implications does {discovery} have for {field} applications?",
                "How does {effect} influence the performance of {technology}?"
            ],
            QuestionType.ANALYTICAL: [
                "Analyze the stability of {system} using {method} when subjected to {perturbation}.",
                "Compare the predictions of {theory1} and {theory2} for {phenomenon}.",
                "What are the limiting cases of {equation} and their physical interpretations?",
                "Investigate how {parameter} affects the {behavior} of {system}.",
                "Examine the validity of {approximation} for {specific_case}."
            ]
        }

    def generate_domain_questions(self, topic: str, count: int = 5,
                                question_type: Optional[QuestionType] = None) -> List[DomainQuestion]:
        """Generate physics-specific questions for a given topic."""
        questions = []

        # Physics-specific topic mapping
        physics_contexts = {
            "quantum": {
                "concepts": ["superposition", "entanglement", "measurement", "uncertainty"],
                "principles": ["correspondence principle", "uncertainty principle", "complementarity"],
                "systems": ["hydrogen atom", "harmonic oscillator", "particle in box", "spin system"],
                "phenomena": ["tunneling", "interference", "decoherence", "entanglement"]
            },
            "relativity": {
                "concepts": ["spacetime", "invariance", "causality", "equivalence"],
                "principles": ["principle of relativity", "equivalence principle", "covariance"],
                "systems": ["reference frame", "light cone", "black hole", "gravitational wave"],
                "phenomena": ["time dilation", "length contraction", "redshift", "lensing"]
            },
            "thermodynamics": {
                "concepts": ["entropy", "temperature", "equilibrium", "irreversibility"],
                "principles": ["zeroth law", "first law", "second law", "third law"],
                "systems": ["heat engine", "refrigerator", "ideal gas", "phase transition"],
                "phenomena": ["heat transfer", "phase change", "critical point", "fluctuation"]
            },
            "electromagnetism": {
                "concepts": ["field", "potential", "flux", "polarization"],
                "principles": ["Gauss law", "Faraday law", "Ampere law", "Lorentz force"],
                "systems": ["capacitor", "inductor", "antenna", "waveguide"],
                "phenomena": ["electromagnetic induction", "wave propagation", "resonance", "radiation"]
            }
        }

        # Select appropriate context
        context_key = self._find_best_context(topic, physics_contexts)
        context = physics_contexts.get(context_key, physics_contexts["quantum"])

        question_types = [question_type] if question_type else list(QuestionType)

        for i in range(count):
            # Select question type
            q_type = random.choice(question_types)

            # Select pattern template
            pattern = random.choice(self.question_patterns[q_type])

            # Fill pattern with physics-specific content
            question_text = self._fill_physics_pattern(pattern, topic, context)

            # Assess difficulty
            difficulty_assessment = assess_question_difficulty_fast(question_text)

            # Assess domain complexity
            domain_complexity = self.assess_domain_complexity(question_text)

            # Validate domain specificity
            validation_result = self.validate_domain_specificity(question_text)

            # Extract terminology used
            terminology_used = self._extract_terminology(question_text)

            # Create domain question
            domain_question = DomainQuestion(
                question=question_text,
                domain=self.domain_name,
                subdomain=context_key,
                question_type=q_type,
                difficulty_assessment=difficulty_assessment,
                domain_complexity=domain_complexity,
                validation_result=validation_result,
                terminology_used=terminology_used,
                concepts_required=context["concepts"][:3],  # Top 3 relevant concepts
                generated_timestamp=datetime.now().isoformat(),
                source_context=topic
            )

            questions.append(domain_question)

            # Update statistics
            self.generation_stats['total_generated'] += 1
            if validation_result.is_domain_appropriate:
                self.generation_stats['validation_passed'] += 1
            self.generation_stats['complexity_scores'].append(domain_complexity.overall_score)

        return questions

    def _find_best_context(self, topic: str, contexts: Dict) -> str:
        """Find the best physics context for the given topic."""
        topic_lower = topic.lower()

        # Direct matching
        for context_key in contexts.keys():
            if context_key in topic_lower:
                return context_key

        # Keyword matching
        keyword_matches = {}
        for context_key, context_data in contexts.items():
            matches = 0
            for concept_list in context_data.values():
                for concept in concept_list:
                    if concept.lower() in topic_lower:
                        matches += 1
            keyword_matches[context_key] = matches

        # Return best match or default
        best_context = max(keyword_matches, key=keyword_matches.get)
        return best_context if keyword_matches[best_context] > 0 else "quantum"

    def _fill_physics_pattern(self, pattern: str, topic: str, context: Dict) -> str:
        """Fill a question pattern with physics-specific content."""
        # Extract placeholders
        placeholders = re.findall(r'\{(\w+)\}', pattern)

        # Content mappings
        fill_options = {
            "concept": context["concepts"] + [topic],
            "concept1": context["concepts"],
            "concept2": context["concepts"],
            "principle": context["principles"],
            "system": context["systems"],
            "phenomenon": context["phenomena"],
            "behavior": ["oscillation", "decay", "acceleration", "rotation", "translation"],
            "framework": ["classical mechanics", "quantum mechanics", "relativity", "field theory"],
            "quantity": ["energy", "momentum", "angular momentum", "charge", "spin"],
            "observable": ["position", "momentum", "energy", "spin", "magnetic moment"],
            "context": ["vacuum", "medium", "field", "potential well", "external field"],
            "method": ["perturbation theory", "variational method", "numerical simulation"],
            "conditions": ["initial conditions", "boundary conditions", "constraints"],
            "variables": ["temperature", "pressure", "field strength", "coupling constant"],
            "symmetry": ["rotational symmetry", "translational symmetry", "gauge symmetry"],
            "equation": ["Schrödinger equation", "Einstein equations", "Maxwell equations"],
            "boundary_conditions": ["periodic", "Dirichlet", "Neumann", "absorbing"],
            "device": ["laser", "transistor", "accelerator", "detector", "antenna"],
            "technology": ["MRI", "GPS", "solar cell", "LED", "superconductor"],
            "effect": ["Doppler effect", "photoelectric effect", "Compton effect"],
            "field": ["medical physics", "astrophysics", "condensed matter", "particle physics"],
            "theory": ["quantum field theory", "general relativity", "statistical mechanics"],
            "theory1": ["classical theory", "Newtonian mechanics", "Maxwell theory"],
            "theory2": ["quantum theory", "relativistic theory", "field theory"],
            "parameter": ["coupling constant", "decay rate", "cross section", "lifetime"],
            "perturbation": ["small oscillations", "weak field", "thermal fluctuations"],
            "approximation": ["dipole approximation", "weak field limit", "low energy limit"],
            "specific_case": ["high temperature", "strong field", "relativistic regime"]
        }

        # Fill placeholders
        filled_pattern = pattern
        for placeholder in placeholders:
            if placeholder in fill_options:
                replacement = random.choice(fill_options[placeholder])
                filled_pattern = filled_pattern.replace(f"{{{placeholder}}}", replacement)

        return filled_pattern

    def _extract_terminology(self, question: str) -> List[str]:
        """Extract physics terminology used in the question."""
        question_lower = question.lower()
        terminology = []

        all_terms = (self.vocabulary.core_terms +
                    self.vocabulary.advanced_terms +
                    self.vocabulary.methodology_terms)

        for term in all_terms:
            if term.lower() in question_lower:
                terminology.append(term)

        return terminology
