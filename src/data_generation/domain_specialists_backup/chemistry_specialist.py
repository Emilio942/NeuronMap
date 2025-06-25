"""
Chemistry Domain Specialist
===========================

Specialized question generator for chemistry domain with molecular reasoning,
reaction mechanisms, and chemical analysis focus.
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


class ChemistrySpecialist(DomainSpecialist):
    """Domain specialist for chemistry questions."""

    def __init__(self):
        super().__init__("Chemistry", DomainType.STEM)

    def _initialize_vocabulary(self) -> DomainVocabulary:
        """Initialize chemistry vocabulary."""
        return DomainVocabulary(
            core_terms=[
                "atom", "molecule", "element", "compound", "mixture", "solution",
                "reaction", "equilibrium", "catalyst", "enzyme", "pH", "acid", "base",
                "oxidation", "reduction", "electron", "proton", "neutron", "orbital",
                "bond", "ionic", "covalent", "metallic", "hydrogen bond", "valence",
                "mole", "molarity", "concentration", "stoichiometry", "yield",
                "thermodynamics", "kinetics", "activation energy", "enthalpy", "entropy"
            ],
            advanced_terms=[
                "quantum chemistry", "molecular orbital theory", "valence bond theory",
                "crystal field theory", "ligand field theory", "coordination chemistry",
                "organometallic chemistry", "polymer chemistry", "biochemistry",
                "physical chemistry", "analytical chemistry", "inorganic chemistry",
                "spectroscopy", "chromatography", "electrochemistry", "photochemistry",
                "computational chemistry", "medicinal chemistry", "green chemistry",
                "supramolecular chemistry", "nanotechnology", "catalysis", "stereochemistry"
            ],
            methodology_terms=[
                "synthesis", "analysis", "purification", "characterization", "identification",
                "spectroscopic analysis", "titration", "extraction", "distillation",
                "crystallization", "chromatographic separation", "mass spectrometry",
                "NMR spectroscopy", "IR spectroscopy", "UV-Vis spectroscopy",
                "experimental design", "data analysis", "error analysis", "calibration"
            ],
            concept_hierarchies={
                "branches": ["organic chemistry", "inorganic chemistry", "physical chemistry", "analytical chemistry"],
                "bonding": ["ionic bonding", "covalent bonding", "metallic bonding", "intermolecular forces"],
                "reactions": ["synthesis reactions", "decomposition reactions", "redox reactions", "acid-base reactions"],
                "analysis": ["qualitative analysis", "quantitative analysis", "instrumental analysis"],
                "states": ["gas phase", "liquid phase", "solid phase", "plasma state"],
                "thermodynamics": ["first law", "second law", "third law", "gibbs free energy"]
            }
        )

    def _initialize_question_patterns(self) -> Dict[QuestionType, List[str]]:
        """Initialize chemistry question patterns."""
        return {
            QuestionType.CONCEPTUAL: [
                "Explain the mechanism of {reaction_type} involving {compound}.",
                "What is the role of {catalyst} in {reaction} and how does it affect {parameter}?",
                "How does {molecular_property} influence {chemical_behavior} in {system}?",
                "Describe the relationship between {structure} and {property} in {compound_class}.",
                "Why does {phenomenon} occur in {chemical_system} under {conditions}?"
            ],
            QuestionType.QUANTITATIVE: [
                "Calculate the {quantity} for {reaction} given {initial_conditions}.",
                "Determine the {parameter} of {solution} with {concentration} of {solute}.",
                "Find the {thermodynamic_property} for {process} at {temperature}.",
                "Compute the {kinetic_parameter} for {reaction} using {data}.",
                "Estimate the {molecular_property} of {compound} using {method}."
            ],
            QuestionType.EXPERIMENTAL: [
                "Design an experiment to determine {property} of {compound}.",
                "How would you synthesize {target_molecule} from {starting_materials}?",
                "What analytical technique would best identify {functional_group} in {sample}?",
                "Propose a method to separate {mixture_components} using {separation_technique}.",
                "Design a procedure to measure {kinetic_parameter} for {reaction}."
            ],
            QuestionType.APPLICATION: [
                "Apply {principle} to predict {outcome} in {chemical_process}.",
                "How would {technique} be used to analyze {sample_type}?",
                "Use {theory} to explain {observation} in {experimental_system}.",
                "Apply {computational_method} to study {molecular_system}.",
                "Utilize {analytical_method} to determine {composition} of {unknown_sample}."
            ],
            QuestionType.ANALYTICAL: [
                "Compare the {property} of {compound1} and {compound2}.",
                "Analyze the effect of {variable} on {reaction_rate} in {system}.",
                "Evaluate the {mechanism} proposed for {reaction} based on {evidence}.",
                "Assess the {environmental_impact} of {industrial_process}.",
                "Investigate the {structure_activity_relationship} in {compound_series}."
            ]
        }

    def generate_domain_questions(self, topic: str, count: int = 5,
                                question_type: Optional[QuestionType] = None) -> List[DomainQuestion]:
        """Generate chemistry questions."""
        questions = []

        chemistry_contexts = {
            "organic": {
                "compounds": ["alkanes", "alkenes", "alcohols", "carboxylic acids", "amines"],
                "reactions": ["substitution", "elimination", "addition", "condensation"],
                "mechanisms": ["SN1", "SN2", "E1", "E2", "radical"],
                "properties": ["stereochemistry", "reactivity", "stability", "aromaticity"]
            },
            "inorganic": {
                "compounds": ["metal complexes", "ionic compounds", "acids", "bases"],
                "reactions": ["redox", "precipitation", "complexation", "acid-base"],
                "mechanisms": ["electron transfer", "ligand exchange", "substitution"],
                "properties": ["coordination number", "oxidation state", "magnetic properties"]
            },
            "physical": {
                "compounds": ["gases", "solutions", "crystals", "polymers"],
                "reactions": ["gas phase reactions", "solution reactions", "surface reactions"],
                "mechanisms": ["collision theory", "transition state theory", "Marcus theory"],
                "properties": ["thermodynamic properties", "kinetic properties", "spectroscopic properties"]
            },
            "analytical": {
                "compounds": ["unknown samples", "mixtures", "trace compounds"],
                "reactions": ["indicator reactions", "complexation reactions", "redox titrations"],
                "mechanisms": ["separation mechanisms", "detection principles"],
                "properties": ["analytical signals", "detection limits", "selectivity"]
            }
        }

        context_key = self._find_best_context(topic, chemistry_contexts)
        context = chemistry_contexts.get(context_key, chemistry_contexts["organic"])

        question_types = [question_type] if question_type else list(QuestionType)

        for i in range(count):
            q_type = random.choice(question_types)
            pattern = random.choice(self.question_patterns[q_type])
            question_text = self._fill_chemistry_pattern(pattern, topic, context)

            difficulty_assessment = assess_question_difficulty_fast(question_text)
            domain_complexity = self.assess_domain_complexity(question_text)
            validation_result = self.validate_domain_specificity(question_text)
            terminology_used = self._extract_terminology(question_text)

            domain_question = DomainQuestion(
                question=question_text,
                domain=self.domain_name,
                subdomain=context_key,
                question_type=q_type,
                difficulty_assessment=difficulty_assessment,
                domain_complexity=domain_complexity,
                validation_result=validation_result,
                terminology_used=terminology_used,
                concepts_required=context["compounds"][:3],
                generated_timestamp=datetime.now().isoformat(),
                source_context=topic
            )

            questions.append(domain_question)

            self.generation_stats['total_generated'] += 1
            if validation_result.is_domain_appropriate:
                self.generation_stats['validation_passed'] += 1
            self.generation_stats['complexity_scores'].append(domain_complexity.overall_score)

        return questions

    def _find_best_context(self, topic: str, contexts: Dict) -> str:
        """Find best chemistry context."""
        topic_lower = topic.lower()

        chemistry_keywords = {
            "organic": ["organic", "carbon", "hydrocarbon", "functional group", "polymer"],
            "inorganic": ["inorganic", "metal", "ionic", "coordination", "crystal"],
            "physical": ["physical", "thermodynamic", "kinetic", "spectroscopic", "quantum"],
            "analytical": ["analytical", "analysis", "detection", "separation", "titration"]
        }

        for context_key, keywords in chemistry_keywords.items():
            if any(keyword in topic_lower for keyword in keywords):
                return context_key

        return "organic"  # Default

    def _fill_chemistry_pattern(self, pattern: str, topic: str, context: Dict) -> str:
        """Fill pattern with chemistry content."""
        placeholders = re.findall(r'\{(\w+)\}', pattern)

        fill_options = {
            "reaction_type": context["reactions"],
            "compound": context["compounds"] + [topic],
            "compound1": context["compounds"],
            "compound2": context["compounds"],
            "catalyst": ["enzyme", "metal catalyst", "acid catalyst", "base catalyst"],
            "reaction": context["reactions"],
            "parameter": ["rate", "yield", "selectivity", "equilibrium constant"],
            "molecular_property": ["polarity", "reactivity", "stability", "aromaticity"],
            "chemical_behavior": ["solubility", "reactivity", "stability", "selectivity"],
            "system": ["aqueous solution", "organic solvent", "gas phase", "solid state"],
            "structure": ["molecular structure", "electronic structure", "crystal structure"],
            "property": context["properties"],
            "compound_class": ["alcohols", "ketones", "acids", "esters", "amines"],
            "phenomenon": ["resonance", "induction", "hyperconjugation", "aromaticity"],
            "chemical_system": ["reaction mixture", "equilibrium system", "catalytic system"],
            "conditions": ["ambient conditions", "elevated temperature", "high pressure"],
            "quantity": ["concentration", "yield", "rate constant", "equilibrium constant"],
            "initial_conditions": ["starting concentrations", "initial temperature", "initial pressure"],
            "solution": ["buffer solution", "standard solution", "unknown solution"],
            "concentration": ["0.1 M", "1.0 M", "saturated", "dilute"],
            "solute": ["strong acid", "weak base", "salt", "organic compound"],
            "thermodynamic_property": ["enthalpy", "entropy", "gibbs free energy", "heat capacity"],
            "process": ["dissolution", "crystallization", "phase transition", "chemical reaction"],
            "temperature": ["298 K", "373 K", "high temperature", "cryogenic conditions"],
            "kinetic_parameter": ["rate constant", "activation energy", "half-life", "reaction order"],
            "data": ["concentration vs time", "temperature dependence", "pH dependence"],
            "molecular_property": ["dipole moment", "polarizability", "ionization energy"],
            "method": ["computational method", "spectroscopic method", "theoretical calculation"],
            "target_molecule": ["pharmaceutical compound", "natural product", "polymer"],
            "starting_materials": ["simple precursors", "commercially available reagents"],
            "functional_group": ["carbonyl", "hydroxyl", "amino", "carboxyl"],
            "sample": ["unknown compound", "mixture", "pharmaceutical sample"],
            "mixture_components": ["organic compounds", "metal ions", "isomers"],
            "separation_technique": ["chromatography", "crystallization", "extraction"],
            "principle": ["Le Chatelier's principle", "conservation of mass", "thermodynamic principles"],
            "outcome": ["product distribution", "reaction rate", "equilibrium position"],
            "chemical_process": ["industrial synthesis", "biological process", "catalytic process"],
            "technique": ["NMR spectroscopy", "mass spectrometry", "X-ray crystallography"],
            "sample_type": ["biological sample", "environmental sample", "pharmaceutical"],
            "theory": ["molecular orbital theory", "transition state theory", "collision theory"],
            "observation": ["color change", "temperature change", "gas evolution"],
            "experimental_system": ["reaction vessel", "electrochemical cell", "catalyst surface"],
            "computational_method": ["DFT calculations", "molecular dynamics", "quantum chemistry"],
            "molecular_system": ["enzyme active site", "metal cluster", "organic molecule"],
            "analytical_method": ["elemental analysis", "functional group analysis", "structural analysis"],
            "composition": ["elemental composition", "molecular composition", "isotopic composition"],
            "unknown_sample": ["forensic sample", "environmental sample", "pharmaceutical"],
            "variable": ["temperature", "pressure", "pH", "concentration"],
            "reaction_rate": ["forward rate", "reverse rate", "net rate"],
            "mechanism": ["concerted mechanism", "stepwise mechanism", "radical mechanism"],
            "evidence": ["kinetic data", "spectroscopic evidence", "isotope labeling"],
            "environmental_impact": ["toxicity", "biodegradability", "carbon footprint"],
            "industrial_process": ["petroleum refining", "pharmaceutical synthesis", "polymer production"],
            "structure_activity_relationship": ["drug activity", "catalyst activity", "biological activity"],
            "compound_series": ["homologous series", "isomer series", "derivative series"]
        }

        filled_pattern = pattern
        for placeholder in placeholders:
            if placeholder in fill_options:
                replacement = random.choice(fill_options[placeholder])
                filled_pattern = filled_pattern.replace(f"{{{placeholder}}}", replacement)

        return filled_pattern

    def _extract_terminology(self, question: str) -> List[str]:
        """Extract chemistry terminology."""
        question_lower = question.lower()
        terminology = []

        all_terms = (self.vocabulary.core_terms +
                    self.vocabulary.advanced_terms +
                    self.vocabulary.methodology_terms)

        for term in all_terms:
            if term.lower() in question_lower:
                terminology.append(term)

        return terminology
