"""
Medicine Domain Specialist
==========================

Specialized question generator for medical domain with clinical reasoning,
pathophysiology, and evidence-based medicine focus.
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


class MedicineSpecialist(DomainSpecialist):
    """Domain specialist for medical questions."""

    def __init__(self):
        super().__init__("Medicine", DomainType.APPLIED_FIELDS)

    def _initialize_vocabulary(self) -> DomainVocabulary:
        """Initialize medical vocabulary."""
        return DomainVocabulary(
            core_terms=[
                "diagnosis", "treatment", "symptom", "syndrome", "disease", "disorder",
                "pathology", "physiology", "anatomy", "patient", "clinical", "therapy",
                "medication", "drug", "dosage", "side effect", "contraindication",
                "examination", "assessment", "intervention", "prognosis", "epidemiology",
                "prevention", "screening", "risk factor", "biomarker", "vital signs",
                "laboratory", "imaging", "biopsy", "surgery", "anesthesia", "recovery"
            ],
            advanced_terms=[
                "pathophysiology", "pharmacokinetics", "pharmacodynamics", "bioavailability",
                "half-life", "metabolism", "excretion", "receptor", "enzyme", "protein",
                "gene expression", "genetic polymorphism", "epigenetics", "biomarker",
                "proteomics", "genomics", "personalized medicine", "precision medicine",
                "immunotherapy", "chemotherapy", "radiotherapy", "targeted therapy",
                "minimal invasive", "robotic surgery", "telemedicine", "AI diagnosis",
                "machine learning", "deep learning", "clinical decision support",
                "evidence-based medicine", "systematic review", "meta-analysis"
            ],
            methodology_terms=[
                "clinical trial", "randomized controlled trial", "cohort study", "case-control",
                "cross-sectional", "longitudinal", "prospective", "retrospective",
                "double-blind", "placebo-controlled", "intention-to-treat", "per-protocol",
                "statistical significance", "confidence interval", "odds ratio", "relative risk",
                "sensitivity", "specificity", "positive predictive value", "negative predictive value",
                "likelihood ratio", "number needed to treat", "clinical guidelines",
                "best practice", "quality improvement", "patient safety", "informed consent"
            ],
            concept_hierarchies={
                "specialties": ["cardiology", "neurology", "oncology", "psychiatry", "pediatrics", "surgery"],
                "systems": ["cardiovascular", "respiratory", "nervous", "endocrine", "immune", "digestive"],
                "diagnostics": ["laboratory tests", "imaging studies", "functional tests", "genetic testing"],
                "therapeutics": ["pharmacotherapy", "surgery", "radiation", "rehabilitation", "prevention"],
                "research": ["basic research", "translational research", "clinical research", "population health"],
                "ethics": ["autonomy", "beneficence", "non-maleficence", "justice", "confidentiality"]
            },
            synonyms={
                "treatment": ["therapy", "intervention"],
                "medication": ["drug", "pharmaceutical"],
                "diagnosis": ["clinical diagnosis"],
                "patient": ["subject", "individual"],
                "disease": ["illness", "condition"]
            }
        )

    def _initialize_question_patterns(self) -> Dict[QuestionType, List[str]]:
        """Initialize medical question patterns."""
        return {
            QuestionType.CONCEPTUAL: [
                "Explain the pathophysiology of {condition} and its impact on {system}.",
                "What is the mechanism of action of {drug} in treating {condition}?",
                "How does {risk_factor} contribute to the development of {disease}?",
                "Describe the relationship between {biomarker} and {outcome} in {condition}.",
                "What are the physiological basis for {symptom} in patients with {disorder}?"
            ],
            QuestionType.QUANTITATIVE: [
                "Calculate the {pharmacokinetic_parameter} for {drug} given {patient_factors}.",
                "Determine the sensitivity and specificity of {diagnostic_test} for {condition}.",
                "Assess the number needed to treat for {intervention} in {population}.",
                "Compute the relative risk of {outcome} for patients with {exposure}.",
                "Analyze the dose-response relationship for {treatment} in {indication}."
            ],
            QuestionType.EXPERIMENTAL: [
                "Design a clinical trial to test {hypothesis} about {intervention} for {condition}.",
                "What study design would best evaluate the efficacy of {treatment}?",
                "How would you investigate the association between {exposure} and {outcome}?",
                "Propose a research methodology to validate {biomarker} for {diagnosis}.",
                "Design a quality improvement study to reduce {adverse_outcome} in {setting}."
            ],
            QuestionType.APPLICATION: [
                "Apply {guideline} to manage a patient with {condition} and {comorbidity}.",
                "How would you use {diagnostic_tool} to evaluate {symptom} in {patient_population}?",
                "Implement {intervention} for {indication} considering {contraindications}.",
                "Utilize {technology} to improve {aspect} of care for {patient_group}.",
                "Adapt {protocol} for use in {clinical_setting} with {resource_constraints}."
            ],
            QuestionType.ANALYTICAL: [
                "Compare the effectiveness of {treatment1} versus {treatment2} for {condition}.",
                "Analyze the cost-effectiveness of {screening_program} for {disease}.",
                "Evaluate the ethical implications of {intervention} in {vulnerable_population}.",
                "Assess the impact of {policy} on {health_outcome} in {population}.",
                "Investigate the factors influencing {adherence} to {treatment} in {patient_group}."
            ]
        }

    def generate_domain_questions(self, topic: str, count: int = 5,
                                question_type: Optional[QuestionType] = None) -> List[DomainQuestion]:
        """Generate medical questions."""
        questions = []

        medical_contexts = {
            "cardiology": {
                "conditions": ["myocardial infarction", "heart failure", "arrhythmia", "hypertension"],
                "treatments": ["ACE inhibitors", "beta blockers", "statins", "anticoagulants"],
                "diagnostics": ["ECG", "echocardiography", "cardiac catheterization", "stress test"],
                "symptoms": ["chest pain", "dyspnea", "palpitations", "syncope"]
            },
            "oncology": {
                "conditions": ["breast cancer", "lung cancer", "colon cancer", "lymphoma"],
                "treatments": ["chemotherapy", "immunotherapy", "radiation therapy", "surgery"],
                "diagnostics": ["biopsy", "CT scan", "PET scan", "tumor markers"],
                "symptoms": ["fatigue", "weight loss", "pain", "nausea"]
            },
            "neurology": {
                "conditions": ["stroke", "epilepsy", "multiple sclerosis", "Parkinson's disease"],
                "treatments": ["anticonvulsants", "dopamine agonists", "thrombolytics", "rehabilitation"],
                "diagnostics": ["MRI", "EEG", "lumbar puncture", "neuropsychological testing"],
                "symptoms": ["headache", "seizures", "weakness", "cognitive impairment"]
            },
            "infectious_disease": {
                "conditions": ["pneumonia", "sepsis", "HIV", "tuberculosis"],
                "treatments": ["antibiotics", "antivirals", "antifungals", "vaccines"],
                "diagnostics": ["blood culture", "PCR", "serology", "chest X-ray"],
                "symptoms": ["fever", "cough", "rash", "lymphadenopathy"]
            }
        }

        context_key = self._find_best_context(topic, medical_contexts)
        context = medical_contexts.get(context_key, medical_contexts["cardiology"])

        question_types = [question_type] if question_type else list(QuestionType)

        for i in range(count):
            q_type = random.choice(question_types)
            pattern = random.choice(self.question_patterns[q_type])
            question_text = self._fill_medical_pattern(pattern, topic, context)

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
                concepts_required=context["conditions"][:3],
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
        """Find best medical context."""
        topic_lower = topic.lower()

        for context_key in contexts.keys():
            if context_key in topic_lower or context_key.replace("_", " ") in topic_lower:
                return context_key

        # Medical specialty keywords
        specialty_keywords = {
            "cardiology": ["heart", "cardiac", "cardiovascular", "coronary"],
            "oncology": ["cancer", "tumor", "malignancy", "chemotherapy"],
            "neurology": ["brain", "neurological", "nervous", "cognitive"],
            "infectious_disease": ["infection", "bacterial", "viral", "antimicrobial"]
        }

        for context_key, keywords in specialty_keywords.items():
            if any(keyword in topic_lower for keyword in keywords):
                return context_key

        return "cardiology"  # Default

    def _fill_medical_pattern(self, pattern: str, topic: str, context: Dict) -> str:
        """Fill pattern with medical content."""
        placeholders = re.findall(r'\{(\w+)\}', pattern)

        fill_options = {
            "condition": context["conditions"] + [topic],
            "treatment": context["treatments"],
            "diagnostic_test": context["diagnostics"],
            "symptom": context["symptoms"],
            "drug": context["treatments"],
            "system": ["cardiovascular", "respiratory", "nervous", "endocrine"],
            "risk_factor": ["smoking", "hypertension", "diabetes", "obesity"],
            "disease": context["conditions"],
            "disorder": context["conditions"],
            "biomarker": ["troponin", "BNP", "PSA", "CA 19-9"],
            "outcome": ["mortality", "morbidity", "quality of life", "functional status"],
            "pharmacokinetic_parameter": ["half-life", "clearance", "bioavailability", "volume of distribution"],
            "patient_factors": ["age", "weight", "renal function", "hepatic function"],
            "population": ["elderly", "pediatric", "pregnant women", "immunocompromised"],
            "intervention": context["treatments"],
            "indication": context["conditions"],
            "exposure": ["smoking", "radiation", "chemical", "infection"],
            "hypothesis": ["drug efficacy", "risk reduction", "diagnostic accuracy"],
            "comorbidity": ["diabetes", "hypertension", "COPD", "kidney disease"],
            "patient_population": ["hospitalized patients", "outpatients", "ICU patients"],
            "contraindications": ["pregnancy", "liver disease", "drug allergy", "renal impairment"],
            "technology": ["telemedicine", "AI diagnosis", "robotic surgery", "point-of-care testing"],
            "aspect": ["diagnosis", "treatment", "monitoring", "prevention"],
            "patient_group": ["diabetic patients", "elderly", "children", "cancer survivors"],
            "protocol": ["treatment protocol", "diagnostic algorithm", "care pathway"],
            "clinical_setting": ["emergency department", "ICU", "outpatient clinic", "community hospital"],
            "resource_constraints": ["limited staff", "budget constraints", "rural setting"],
            "treatment1": context["treatments"][0] if context["treatments"] else "standard care",
            "treatment2": context["treatments"][1] if len(context["treatments"]) > 1 else "alternative therapy",
            "screening_program": ["mammography", "colonoscopy", "Pap smear", "blood pressure screening"],
            "vulnerable_population": ["children", "elderly", "pregnant women", "mentally ill"],
            "policy": ["vaccination policy", "screening guidelines", "antibiotic stewardship"],
            "health_outcome": ["mortality", "hospital readmission", "infection rate", "patient satisfaction"],
            "adherence": ["medication adherence", "treatment compliance", "follow-up attendance"],
            "diagnostic_tool": context["diagnostics"],
            "guideline": ["clinical practice guideline", "treatment protocol", "care standard"],
            "adverse_outcome": ["medication error", "hospital-acquired infection", "readmission"],
            "setting": ["hospital", "clinic", "nursing home", "community"]
        }

        filled_pattern = pattern
        for placeholder in placeholders:
            if placeholder in fill_options:
                replacement = random.choice(fill_options[placeholder])
                filled_pattern = filled_pattern.replace(f"{{{placeholder}}}", replacement)

        return filled_pattern

    def _extract_terminology(self, question: str) -> List[str]:
        """Extract medical terminology."""
        question_lower = question.lower()
        terminology = []

        all_terms = (self.vocabulary.core_terms +
                    self.vocabulary.advanced_terms +
                    self.vocabulary.methodology_terms)

        for term in all_terms:
            if term.lower() in question_lower:
                terminology.append(term)

        return terminology
