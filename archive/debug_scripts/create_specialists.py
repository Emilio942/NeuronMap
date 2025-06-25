#!/usr/bin/env python3
"""
Script to generate skeleton domain specialists
============================================

This script creates skeleton implementations for all remaining domain specialists
following the established pattern but with domain-specific vocabulary and patterns.
"""

import os
from pathlib import Path

# Define all remaining specialists
specialists_data = {
    "biology_specialist.py": {
        "class_name": "BiologySpecialist",
        "domain_name": "Biology",
        "domain_type": "STEM",
        "core_terms": [
            "cell", "organism", "evolution", "genetics", "DNA", "RNA", "protein",
            "ecosystem", "species", "population", "habitat", "biodiversity",
            "photosynthesis", "respiration", "metabolism", "enzyme", "hormone",
            "tissue", "organ", "system", "development", "reproduction", "heredity"
        ],
        "advanced_terms": [
            "genomics", "proteomics", "bioinformatics", "molecular biology",
            "biotechnology", "genetic engineering", "CRISPR", "epigenetics",
            "systems biology", "synthetic biology", "computational biology"
        ],
        "contexts": {
            "molecular": ["DNA", "proteins", "enzymes", "genes"],
            "cellular": ["mitosis", "meiosis", "organelles", "membranes"],
            "organismal": ["anatomy", "physiology", "development", "behavior"],
            "ecological": ["ecosystems", "populations", "communities", "evolution"]
        }
    },
    
    "mathematics_specialist.py": {
        "class_name": "MathematicsSpecialist",
        "domain_name": "Mathematics",
        "domain_type": "STEM",
        "core_terms": [
            "equation", "function", "variable", "constant", "derivative", "integral",
            "limit", "series", "matrix", "vector", "probability", "statistics",
            "geometry", "algebra", "calculus", "topology", "analysis", "proof"
        ],
        "advanced_terms": [
            "differential equations", "complex analysis", "abstract algebra",
            "real analysis", "functional analysis", "topology", "number theory",
            "combinatorics", "graph theory", "optimization", "numerical analysis"
        ],
        "contexts": {
            "algebra": ["equations", "polynomials", "groups", "rings"],
            "analysis": ["limits", "derivatives", "integrals", "series"],
            "geometry": ["shapes", "transformations", "spaces", "manifolds"],
            "applied": ["optimization", "modeling", "statistics", "probability"]
        }
    },
    
    "engineering_specialist.py": {
        "class_name": "EngineeringSpecialist",
        "domain_name": "Engineering",
        "domain_type": "STEM",
        "core_terms": [
            "design", "system", "optimization", "efficiency", "safety", "reliability",
            "material", "structure", "load", "stress", "strain", "force",
            "circuit", "signal", "control", "feedback", "automation", "robotics"
        ],
        "advanced_terms": [
            "finite element analysis", "computational fluid dynamics", "systems engineering",
            "control theory", "signal processing", "machine learning", "AI",
            "sustainable engineering", "bioengineering", "nanotechnology"
        ],
        "contexts": {
            "mechanical": ["machines", "structures", "materials", "thermodynamics"],
            "electrical": ["circuits", "signals", "power", "electronics"],
            "civil": ["structures", "infrastructure", "construction", "transportation"],
            "chemical": ["processes", "reactors", "separation", "safety"]
        }
    },
    
    "philosophy_specialist.py": {
        "class_name": "PhilosophySpecialist",
        "domain_name": "Philosophy",
        "domain_type": "HUMANITIES",
        "core_terms": [
            "logic", "ethics", "metaphysics", "epistemology", "aesthetics",
            "consciousness", "mind", "reality", "truth", "knowledge", "belief",
            "morality", "justice", "freedom", "determinism", "existence"
        ],
        "advanced_terms": [
            "phenomenology", "existentialism", "analytic philosophy", "continental philosophy",
            "philosophy of mind", "philosophy of science", "political philosophy",
            "philosophy of language", "moral philosophy", "applied ethics"
        ],
        "contexts": {
            "metaphysics": ["reality", "existence", "being", "causation"],
            "epistemology": ["knowledge", "belief", "truth", "skepticism"],
            "ethics": ["morality", "virtue", "duty", "consequences"],
            "logic": ["reasoning", "argument", "validity", "soundness"]
        }
    },
    
    "literature_specialist.py": {
        "class_name": "LiteratureSpecialist",
        "domain_name": "Literature",
        "domain_type": "HUMANITIES",
        "core_terms": [
            "narrative", "character", "plot", "theme", "symbolism", "metaphor",
            "genre", "style", "tone", "voice", "perspective", "setting",
            "poetry", "prose", "drama", "fiction", "criticism", "interpretation"
        ],
        "advanced_terms": [
            "literary theory", "postmodernism", "structuralism", "deconstruction",
            "feminist criticism", "postcolonial theory", "new historicism",
            "reader response theory", "psychoanalytic criticism", "formalism"
        ],
        "contexts": {
            "poetry": ["verse", "meter", "rhyme", "imagery"],
            "prose": ["novel", "short story", "essay", "narrative"],
            "drama": ["tragedy", "comedy", "dialogue", "performance"],
            "criticism": ["analysis", "interpretation", "theory", "context"]
        }
    },
    
    "linguistics_specialist.py": {
        "class_name": "LinguisticsSpecialist", 
        "domain_name": "Linguistics",
        "domain_type": "HUMANITIES",
        "core_terms": [
            "phoneme", "morpheme", "syntax", "semantics", "pragmatics",
            "grammar", "language", "dialect", "accent", "pronunciation",
            "meaning", "structure", "communication", "discourse", "conversation"
        ],
        "advanced_terms": [
            "sociolinguistics", "psycholinguistics", "computational linguistics",
            "historical linguistics", "applied linguistics", "corpus linguistics",
            "generative grammar", "cognitive linguistics", "typology"
        ],
        "contexts": {
            "phonetics": ["sounds", "pronunciation", "articulation", "acoustics"],
            "syntax": ["grammar", "structure", "rules", "parsing"],
            "semantics": ["meaning", "interpretation", "reference", "truth"],
            "pragmatics": ["context", "usage", "communication", "speech acts"]
        }
    },
    
    "art_history_specialist.py": {
        "class_name": "ArtHistorySpecialist",
        "domain_name": "Art History", 
        "domain_type": "HUMANITIES",
        "core_terms": [
            "painting", "sculpture", "architecture", "style", "period", "movement",
            "composition", "color", "form", "technique", "medium", "artist",
            "patron", "iconography", "symbolism", "aesthetic", "beauty", "culture"
        ],
        "advanced_terms": [
            "art criticism", "visual culture", "museum studies", "cultural studies",
            "gender studies", "postcolonial studies", "material culture",
            "digital humanities", "conservation", "provenance", "attribution"
        ],
        "contexts": {
            "painting": ["canvas", "oil paint", "fresco", "watercolor"],
            "sculpture": ["marble", "bronze", "wood", "installation"],
            "architecture": ["design", "structure", "space", "function"],
            "theory": ["aesthetics", "criticism", "interpretation", "context"]
        }
    },
    
    "psychology_specialist.py": {
        "class_name": "PsychologySpecialist",
        "domain_name": "Psychology",
        "domain_type": "SOCIAL_SCIENCES",
        "core_terms": [
            "behavior", "cognition", "emotion", "perception", "memory", "learning",
            "personality", "development", "intelligence", "motivation", "stress",
            "therapy", "disorder", "diagnosis", "treatment", "research", "experiment"
        ],
        "advanced_terms": [
            "cognitive psychology", "behavioral psychology", "developmental psychology",
            "social psychology", "clinical psychology", "neuropsychology",
            "psychopathology", "psychotherapy", "psychopharmacology", "psychometrics"
        ],
        "contexts": {
            "cognitive": ["memory", "attention", "perception", "reasoning"],
            "developmental": ["childhood", "adolescence", "aging", "milestones"],
            "social": ["groups", "attitudes", "relationships", "influence"],
            "clinical": ["disorders", "therapy", "assessment", "treatment"]
        }
    },
    
    "sociology_specialist.py": {
        "class_name": "SociologySpecialist",
        "domain_name": "Sociology",
        "domain_type": "SOCIAL_SCIENCES",
        "core_terms": [
            "society", "culture", "social structure", "institution", "group", "role",
            "class", "inequality", "stratification", "mobility", "community",
            "organization", "interaction", "norm", "value", "deviance", "change"
        ],
        "advanced_terms": [
            "sociological theory", "quantitative methods", "qualitative research",
            "social network analysis", "urban sociology", "rural sociology",
            "medical sociology", "education sociology", "criminology", "demography"
        ],
        "contexts": {
            "theory": ["functionalism", "conflict theory", "symbolic interactionism"],
            "methods": ["survey", "interview", "observation", "experiment"],
            "institutions": ["family", "education", "religion", "politics"],
            "inequality": ["class", "race", "gender", "age"]
        }
    },
    
    "political_science_specialist.py": {
        "class_name": "PoliticalScienceSpecialist",
        "domain_name": "Political Science",
        "domain_type": "SOCIAL_SCIENCES", 
        "core_terms": [
            "government", "politics", "power", "authority", "democracy", "state",
            "policy", "election", "voting", "representation", "constitution",
            "law", "rights", "citizenship", "ideology", "party", "institution"
        ],
        "advanced_terms": [
            "political theory", "comparative politics", "international relations",
            "public administration", "public policy", "political economy",
            "political behavior", "political institutions", "governance"
        ],
        "contexts": {
            "theory": ["democracy", "liberalism", "conservatism", "socialism"],
            "comparative": ["systems", "institutions", "cultures", "development"],
            "international": ["diplomacy", "conflict", "cooperation", "organizations"],
            "public": ["policy", "administration", "management", "evaluation"]
        }
    },
    
    "economics_specialist.py": {
        "class_name": "EconomicsSpecialist",
        "domain_name": "Economics",
        "domain_type": "SOCIAL_SCIENCES",
        "core_terms": [
            "market", "price", "supply", "demand", "elasticity", "competition",
            "monopoly", "profit", "cost", "revenue", "investment", "consumption",
            "production", "efficiency", "growth", "inflation", "unemployment"
        ],
        "advanced_terms": [
            "microeconomics", "macroeconomics", "econometrics", "behavioral economics",
            "development economics", "international economics", "public economics",
            "labor economics", "environmental economics", "financial economics"
        ],
        "contexts": {
            "micro": ["consumers", "firms", "markets", "pricing"],
            "macro": ["GDP", "inflation", "unemployment", "policy"],
            "international": ["trade", "exchange rates", "globalization"],
            "development": ["growth", "poverty", "inequality", "institutions"]
        }
    },
    
    "anthropology_specialist.py": {
        "class_name": "AnthropologySpecialist",
        "domain_name": "Anthropology",
        "domain_type": "SOCIAL_SCIENCES",
        "core_terms": [
            "culture", "society", "ethnography", "fieldwork", "ritual", "kinship",
            "language", "belief", "custom", "tradition", "adaptation", "evolution",
            "archaeology", "artifact", "fossil", "human", "primate", "origin"
        ],
        "advanced_terms": [
            "cultural anthropology", "physical anthropology", "linguistic anthropology",
            "archaeological anthropology", "applied anthropology", "medical anthropology",
            "political anthropology", "economic anthropology", "ethnomusicology"
        ],
        "contexts": {
            "cultural": ["customs", "beliefs", "practices", "meaning"],
            "physical": ["evolution", "primates", "fossils", "genetics"],
            "linguistic": ["language", "communication", "cognition", "culture"],
            "archaeological": ["artifacts", "sites", "prehistory", "material culture"]
        }
    },
    
    "law_specialist.py": {
        "class_name": "LawSpecialist",
        "domain_name": "Law",
        "domain_type": "APPLIED_FIELDS",
        "core_terms": [
            "statute", "regulation", "case law", "precedent", "court", "judge",
            "jury", "trial", "evidence", "witness", "contract", "tort", "crime",
            "liability", "damages", "remedy", "justice", "rights", "obligation"
        ],
        "advanced_terms": [
            "constitutional law", "criminal law", "civil law", "international law",
            "corporate law", "intellectual property", "environmental law",
            "human rights law", "comparative law", "legal theory"
        ],
        "contexts": {
            "constitutional": ["rights", "powers", "federalism", "separation"],
            "criminal": ["crime", "punishment", "procedure", "evidence"],
            "civil": ["contracts", "torts", "property", "remedies"],
            "international": ["treaties", "sovereignty", "jurisdiction", "enforcement"]
        }
    },
    
    "education_specialist.py": {
        "class_name": "EducationSpecialist",
        "domain_name": "Education",
        "domain_type": "APPLIED_FIELDS",
        "core_terms": [
            "learning", "teaching", "curriculum", "instruction", "assessment",
            "student", "teacher", "classroom", "school", "education", "pedagogy",
            "development", "motivation", "achievement", "literacy", "numeracy"
        ],
        "advanced_terms": [
            "educational psychology", "curriculum theory", "instructional design",
            "educational technology", "special education", "multicultural education",
            "educational policy", "school administration", "teacher education"
        ],
        "contexts": {
            "learning": ["cognition", "motivation", "development", "assessment"],
            "teaching": ["instruction", "pedagogy", "methods", "technology"],
            "curriculum": ["design", "implementation", "evaluation", "standards"],
            "policy": ["reform", "equity", "access", "accountability"]
        }
    },
    
    "business_specialist.py": {
        "class_name": "BusinessSpecialist",
        "domain_name": "Business",
        "domain_type": "APPLIED_FIELDS",
        "core_terms": [
            "management", "strategy", "marketing", "finance", "accounting",
            "organization", "leadership", "team", "performance", "profit",
            "customer", "product", "service", "quality", "innovation", "competition"
        ],
        "advanced_terms": [
            "strategic management", "operations management", "human resources",
            "organizational behavior", "supply chain management", "business analytics",
            "entrepreneurship", "international business", "business ethics"
        ],
        "contexts": {
            "strategy": ["planning", "competitive advantage", "positioning", "growth"],
            "marketing": ["customers", "branding", "promotion", "pricing"],
            "finance": ["investment", "capital", "risk", "valuation"],
            "operations": ["processes", "quality", "efficiency", "supply chain"]
        }
    }
}

def create_specialist_file(filename, data):
    """Create a specialist file with the given data."""
    template = f'''"""
{data["domain_name"]} Domain Specialist
{"=" * (len(data["domain_name"]) + 20)}

Specialized question generator for {data["domain_name"].lower()} domain.
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


class {data["class_name"]}(DomainSpecialist):
    """Domain specialist for {data["domain_name"].lower()} questions."""
    
    def __init__(self):
        super().__init__("{data["domain_name"]}", DomainType.{data["domain_type"]})
    
    def _initialize_vocabulary(self) -> DomainVocabulary:
        """Initialize {data["domain_name"].lower()} vocabulary."""
        return DomainVocabulary(
            core_terms={data["core_terms"]},
            advanced_terms={data["advanced_terms"]},
            methodology_terms=[
                "analysis", "research", "method", "study", "investigation",
                "evaluation", "assessment", "measurement", "observation",
                "experiment", "data", "evidence", "theory", "hypothesis"
            ],
            concept_hierarchies={{
                {", ".join([f'"{k}": {v}' for k, v in data["contexts"].items()])}
            }}
        )
    
    def _initialize_question_patterns(self) -> Dict[QuestionType, List[str]]:
        """Initialize {data["domain_name"].lower()} question patterns."""
        return {{
            QuestionType.CONCEPTUAL: [
                "Explain the concept of {{concept}} in {{context}}.",
                "What is the relationship between {{concept1}} and {{concept2}}?",
                "How does {{factor}} influence {{outcome}} in {{domain_context}}?",
                "Describe the significance of {{principle}} for {{application}}.",
                "Why is {{concept}} important in {{field_context}}?"
            ],
            QuestionType.ANALYTICAL: [
                "Analyze the {{aspect}} of {{subject}} in {{context}}.",
                "Compare {{item1}} and {{item2}} with respect to {{criterion}}.",
                "Evaluate the {{method}} used to {{purpose}} in {{scenario}}.",
                "Assess the {{quality}} of {{object}} given {{conditions}}.",
                "Examine the {{relationship}} between {{variable1}} and {{variable2}}."
            ],
            QuestionType.APPLICATION: [
                "Apply {{theory}} to solve {{problem}} in {{context}}.",
                "How would you use {{method}} to {{objective}} in {{situation}}?",
                "Implement {{approach}} to address {{challenge}} in {{domain}}.",
                "Utilize {{tool}} to {{purpose}} for {{target}}.",
                "Demonstrate {{principle}} through {{example}} in {{context}}."
            ],
            QuestionType.EVALUATIVE: [
                "Evaluate the effectiveness of {{intervention}} for {{purpose}}.",
                "Assess the {{criterion}} of {{subject}} in {{context}}.",
                "Judge the {{quality}} of {{object}} based on {{standards}}.",
                "Determine the {{value}} of {{resource}} for {{application}}.",
                "Critique the {{approach}} used in {{scenario}}."
            ],
            QuestionType.CREATIVE: [
                "Design a {{solution}} to address {{problem}} in {{context}}.",
                "Propose a {{method}} for {{objective}} considering {{constraints}}.",
                "Create a {{framework}} to {{purpose}} in {{domain}}.",
                "Develop a {{strategy}} to {{goal}} given {{resources}}.",
                "Innovate a {{approach}} for {{challenge}} in {{field}}."
            ]
        }}
    
    def generate_domain_questions(self, topic: str, count: int = 5, 
                                question_type: Optional[QuestionType] = None) -> List[DomainQuestion]:
        """Generate {data["domain_name"].lower()} questions."""
        questions = []
        
        contexts = {data["contexts"]}
        
        context_key = self._find_best_context(topic, contexts)
        context = contexts.get(context_key, list(contexts.values())[0])
        
        question_types = [question_type] if question_type else list(QuestionType)
        
        for i in range(count):
            q_type = random.choice(question_types)
            pattern = random.choice(self.question_patterns[q_type])
            question_text = self._fill_pattern(pattern, topic, context)
            
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
                concepts_required=context[:3],
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
        """Find best context for topic."""
        topic_lower = topic.lower()
        
        for context_key in contexts.keys():
            if context_key in topic_lower:
                return context_key
        
        keyword_matches = {{}}
        for context_key, context_items in contexts.items():
            matches = sum(1 for item in context_items if item.lower() in topic_lower)
            keyword_matches[context_key] = matches
        
        if keyword_matches:
            return max(keyword_matches, key=keyword_matches.get)
        
        return list(contexts.keys())[0]  # Default to first
    
    def _fill_pattern(self, pattern: str, topic: str, context: List[str]) -> str:
        """Fill pattern with domain content."""
        placeholders = re.findall(r'\\{{(\\w+)\\}}', pattern)
        
        # Generic fill options - extend in specific implementations
        fill_options = {{
            "concept": context + [topic],
            "concept1": context,
            "concept2": context,
            "context": ["{data["domain_name"].lower()} context", "academic context", "practical context"],
            "factor": ["key factor", "important element", "significant aspect"],
            "outcome": ["result", "consequence", "effect"],
            "domain_context": ["{data["domain_name"].lower()} field", "academic setting", "professional context"],
            "principle": ["fundamental principle", "key concept", "important rule"],
            "application": ["practical use", "real-world scenario", "case study"],
            "field_context": ["{data["domain_name"].lower()} field", "academic discipline", "professional area"],
            "subject": context + [topic],
            "aspect": ["important aspect", "key feature", "significant element"],
            "item1": context,
            "item2": context,
            "criterion": ["effectiveness", "efficiency", "quality", "accuracy"],
            "method": ["approach", "technique", "procedure", "strategy"],
            "purpose": ["objective", "goal", "aim", "target"],
            "scenario": ["situation", "case", "example", "instance"],
            "quality": ["effectiveness", "reliability", "validity", "usefulness"],
            "object": context + [topic],
            "conditions": ["given circumstances", "specific conditions", "particular context"],
            "relationship": ["connection", "association", "correlation", "interaction"],
            "variable1": context,
            "variable2": context,
            "theory": ["theoretical framework", "conceptual model", "academic theory"],
            "problem": ["challenge", "issue", "difficulty", "question"],
            "objective": ["goal", "aim", "purpose", "target"],
            "situation": ["scenario", "context", "case", "example"],
            "approach": ["method", "strategy", "technique", "procedure"],
            "challenge": ["problem", "difficulty", "issue", "obstacle"],
            "domain": ["{data["domain_name"].lower()} field", "academic area", "professional domain"],
            "tool": ["instrument", "method", "technique", "resource"],
            "target": ["audience", "group", "population", "subject"],
            "example": ["case study", "illustration", "instance", "demonstration"],
            "intervention": ["action", "strategy", "approach", "method"],
            "standards": ["criteria", "benchmarks", "guidelines", "principles"],
            "value": ["worth", "importance", "significance", "utility"],
            "resource": ["tool", "material", "asset", "source"],
            "solution": ["answer", "resolution", "approach", "strategy"],
            "constraints": ["limitations", "restrictions", "boundaries", "conditions"],
            "framework": ["structure", "model", "system", "approach"],
            "goal": ["objective", "aim", "target", "purpose"],
            "resources": ["materials", "tools", "assets", "support"],
            "strategy": ["plan", "approach", "method", "technique"],
            "field": ["{data["domain_name"].lower()} field", "academic area", "professional domain"]
        }}
        
        filled_pattern = pattern
        for placeholder in placeholders:
            if placeholder in fill_options:
                replacement = random.choice(fill_options[placeholder])
                filled_pattern = filled_pattern.replace(f"{{{{{{placeholder}}}}}}", replacement)
        
        return filled_pattern
    
    def _extract_terminology(self, question: str) -> List[str]:
        """Extract {data["domain_name"].lower()} terminology."""
        question_lower = question.lower()
        terminology = []
        
        all_terms = (self.vocabulary.core_terms + 
                    self.vocabulary.advanced_terms + 
                    self.vocabulary.methodology_terms)
        
        for term in all_terms:
            if term.lower() in question_lower:
                terminology.append(term)
        
        return terminology
'''
    
    return template

# Create the specialist files
def main():
    base_dir = Path("/home/emilio/Documents/ai/NeuronMap/src/data_generation/domain_specialists")
    
    print("Creating domain specialist files...")
    
    for filename, data in specialists_data.items():
        file_path = base_dir / filename
        content = create_specialist_file(filename, data)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"Created {filename}")
    
    print(f"Successfully created {len(specialists_data)} domain specialist files!")

if __name__ == "__main__":
    main()
