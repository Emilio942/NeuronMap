"""
History Domain Specialist
=========================

Specialized question generator for history domain with chronological reasoning,
causation analysis, and historical interpretation focus.
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


class HistorySpecialist(DomainSpecialist):
    """Domain specialist for history questions."""

    def __init__(self):
        super().__init__("History", DomainType.HUMANITIES)

    def _initialize_vocabulary(self) -> DomainVocabulary:
        """Initialize history vocabulary."""
        return DomainVocabulary(
            core_terms=[
                "civilization", "empire", "dynasty", "republic", "democracy", "monarchy",
                "revolution", "war", "battle", "treaty", "alliance", "diplomacy",
                "culture", "society", "economy", "politics", "religion", "ideology",
                "colonialism", "imperialism", "nationalism", "feudalism", "capitalism",
                "reformation", "renaissance", "enlightenment", "industrial revolution",
                "primary source", "secondary source", "archaeology", "chronology",
                "period", "era", "century", "decade", "ancient", "medieval", "modern"
            ],
            advanced_terms=[
                "historiography", "periodization", "historical materialism", "social history",
                "cultural history", "microhistory", "oral history", "quantitative history",
                "postcolonial history", "gender history", "environmental history",
                "intellectual history", "diplomatic history", "military history",
                "economic history", "political history", "comparative history",
                "transnational history", "global history", "world systems theory",
                "longue durée", "mentalité", "collective memory", "historical consciousness",
                "anachronism", "presentism", "historical determinism", "contingency"
            ],
            methodology_terms=[
                "analysis", "interpretation", "evidence", "context", "causation", "correlation",
                "primary sources", "archives", "documentation", "oral testimony",
                "archaeological evidence", "material culture", "iconography", "numismatics",
                "paleography", "diplomatic", "sigillography", "chronological analysis",
                "comparative method", "case study", "biographical approach",
                "prosopography", "quantitative analysis", "statistical method",
                "interdisciplinary approach", "historical synthesis", "narrative structure"
            ],
            concept_hierarchies={
                "periods": ["ancient history", "medieval history", "early modern", "modern history", "contemporary"],
                "regions": ["European history", "American history", "Asian history", "African history", "Middle Eastern"],
                "themes": ["political history", "social history", "economic history", "cultural history", "military history"],
                "approaches": ["traditional history", "social history", "cultural history", "new history", "public history"],
                "sources": ["written sources", "material evidence", "oral sources", "visual sources", "digital sources"],
                "methods": ["narrative method", "analytical method", "comparative method", "quantitative method"]
            },
            synonyms={
                "period": ["era", "epoch", "age"],
                "evidence": ["sources", "documentation"],
                "ruler": ["monarch", "sovereign", "emperor"],
                "conflict": ["war", "battle", "struggle"],
                "change": ["transformation", "evolution", "development"]
            }
        )

    def _initialize_question_patterns(self) -> Dict[QuestionType, List[str]]:
        """Initialize history question patterns."""
        return {
            QuestionType.CONCEPTUAL: [
                "Explain the historical significance of {event} in the context of {period}.",
                "What were the underlying causes of {development} in {region} during {timeframe}?",
                "How did {factor} influence the course of {historical_process} in {society}?",
                "Describe the relationship between {concept1} and {concept2} in {historical_context}.",
                "What role did {actor} play in shaping {outcome} during {period}?"
            ],
            QuestionType.ANALYTICAL: [
                "Compare the causes and consequences of {event1} and {event2} in {context}.",
                "Analyze the long-term impact of {policy} on {group} in {region}.",
                "Evaluate the effectiveness of {strategy} employed by {actor} during {conflict}.",
                "Assess the validity of the argument that {factor} was the primary cause of {outcome}.",
                "Examine the different interpretations of {event} by {historian1} and {historian2}."
            ],
            QuestionType.COMPARATIVE: [
                "Compare the development of {institution} in {society1} and {society2}.",
                "How did {phenomenon} manifest differently in {region1} versus {region2}?",
                "Contrast the approaches of {leader1} and {leader2} to {challenge}.",
                "Compare the impact of {event} on {group1} and {group2}.",
                "How did {concept} evolve differently across {timeframe1} and {timeframe2}?"
            ],
            QuestionType.EVALUATIVE: [
                "To what extent was {actor} responsible for {outcome}?",
                "How successfully did {policy} achieve its intended {goals}?",
                "Evaluate the claim that {period} represented a turning point in {aspect}.",
                "Assess the historical significance of {source} for understanding {topic}.",
                "How valid is the interpretation that {factor} determined {development}?"
            ],
            QuestionType.APPLICATION: [
                "Use {historical_method} to analyze {primary_source} about {topic}.",
                "Apply {theoretical_framework} to explain {historical_phenomenon}.",
                "How would you use {type_of_evidence} to investigate {research_question}?",
                "Utilize {comparative_approach} to examine {historical_problem}.",
                "Apply {analytical_tool} to assess the reliability of {historical_account}."
            ]
        }

    def generate_domain_questions(self, topic: str, count: int = 5,
                                question_type: Optional[QuestionType] = None) -> List[DomainQuestion]:
        """Generate history questions."""
        questions = []

        history_contexts = {
            "ancient": {
                "events": ["fall of Rome", "rise of Christianity", "Persian Wars", "Punic Wars"],
                "actors": ["Julius Caesar", "Alexander the Great", "Augustus", "Cleopatra"],
                "societies": ["Roman Empire", "Greek city-states", "Egyptian civilization", "Persian Empire"],
                "concepts": ["democracy", "republic", "empire", "citizenship"]
            },
            "medieval": {
                "events": ["Crusades", "Black Death", "feudal system", "Mongol invasions"],
                "actors": ["Charlemagne", "William the Conqueror", "Thomas Aquinas", "Joan of Arc"],
                "societies": ["Byzantine Empire", "Islamic caliphates", "medieval Europe", "medieval China"],
                "concepts": ["feudalism", "monasticism", "chivalry", "scholasticism"]
            },
            "modern": {
                "events": ["French Revolution", "Industrial Revolution", "World War I", "Russian Revolution"],
                "actors": ["Napoleon", "Lenin", "Churchill", "Gandhi"],
                "societies": ["modern Europe", "colonial America", "industrial Britain", "revolutionary France"],
                "concepts": ["nationalism", "liberalism", "socialism", "imperialism"]
            },
            "contemporary": {
                "events": ["World War II", "Cold War", "decolonization", "civil rights movement"],
                "actors": ["Hitler", "Stalin", "Roosevelt", "Mandela"],
                "societies": ["Nazi Germany", "Soviet Union", "United States", "post-colonial Africa"],
                "concepts": ["totalitarianism", "human rights", "globalization", "democratization"]
            }
        }

        context_key = self._find_best_context(topic, history_contexts)
        context = history_contexts.get(context_key, history_contexts["modern"])

        question_types = [question_type] if question_type else list(QuestionType)

        for i in range(count):
            q_type = random.choice(question_types)
            pattern = random.choice(self.question_patterns[q_type])
            question_text = self._fill_history_pattern(pattern, topic, context, context_key)

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
                concepts_required=context["concepts"][:3],
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
        """Find best historical context."""
        topic_lower = topic.lower()

        # Direct period matching
        period_keywords = {
            "ancient": ["ancient", "classical", "rome", "greece", "egypt"],
            "medieval": ["medieval", "middle ages", "feudal", "crusades", "byzantine"],
            "modern": ["renaissance", "enlightenment", "revolution", "industrial", "napoleon"],
            "contemporary": ["world war", "cold war", "20th century", "21st century", "modern"]
        }

        for context_key, keywords in period_keywords.items():
            if any(keyword in topic_lower for keyword in keywords):
                return context_key

        # Check specific events/actors
        for context_key, context_data in contexts.items():
            for data_list in context_data.values():
                for item in data_list:
                    if item.lower() in topic_lower:
                        return context_key

        return "modern"  # Default

    def _fill_history_pattern(self, pattern: str, topic: str, context: Dict, period: str) -> str:
        """Fill pattern with historical content."""
        placeholders = re.findall(r'\{(\w+)\}', pattern)

        timeframes = {
            "ancient": ["6th century BCE", "5th century BCE", "1st century CE", "late antiquity"],
            "medieval": ["early Middle Ages", "High Middle Ages", "late Middle Ages", "12th century"],
            "modern": ["16th century", "17th century", "18th century", "early 19th century"],
            "contemporary": ["early 20th century", "mid-20th century", "late 20th century", "21st century"]
        }

        fill_options = {
            "event": context["events"] + [topic],
            "event1": context["events"],
            "event2": context["events"],
            "actor": context["actors"],
            "leader1": context["actors"],
            "leader2": context["actors"],
            "society": context["societies"],
            "society1": context["societies"],
            "society2": context["societies"],
            "concept": context["concepts"],
            "concept1": context["concepts"],
            "concept2": context["concepts"],
            "period": [period],
            "timeframe": timeframes.get(period, ["historical period"]),
            "timeframe1": timeframes.get(period, ["early period"]),
            "timeframe2": timeframes.get(period, ["later period"]),
            "region": ["Europe", "Asia", "Africa", "Americas", "Middle East"],
            "region1": ["Western Europe", "Eastern Europe", "North America"],
            "region2": ["Asia", "Africa", "South America"],
            "factor": ["economic factors", "political changes", "social movements", "technological innovation"],
            "development": ["political development", "economic change", "social transformation"],
            "historical_process": ["democratization", "industrialization", "urbanization", "secularization"],
            "outcome": ["political change", "social reform", "economic growth", "cultural shift"],
            "historical_context": [f"{period} period", "wartime", "peacetime", "crisis period"],
            "policy": ["foreign policy", "domestic policy", "economic policy", "social policy"],
            "group": ["nobility", "peasantry", "merchants", "clergy", "working class"],
            "group1": ["upper class", "aristocracy", "elite"],
            "group2": ["common people", "masses", "lower class"],
            "strategy": ["diplomatic strategy", "military strategy", "political strategy"],
            "conflict": ["war", "revolution", "civil conflict", "international crisis"],
            "institution": ["parliament", "monarchy", "church", "guilds", "universities"],
            "phenomenon": ["nationalism", "industrialization", "urbanization", "secularization"],
            "challenge": ["economic crisis", "political instability", "foreign invasion", "social unrest"],
            "aspect": ["political development", "economic growth", "social change", "cultural evolution"],
            "goals": ["political goals", "economic objectives", "social aims", "military objectives"],
            "source": ["chronicle", "memoir", "letter", "official document"],
            "topic": [topic, "historical event", "historical process"],
            "theoretical_framework": ["Marxist analysis", "social history approach", "cultural analysis"],
            "historical_method": ["source criticism", "comparative analysis", "contextual analysis"],
            "primary_source": ["diary", "letter", "official record", "newspaper"],
            "type_of_evidence": ["documentary evidence", "archaeological evidence", "oral testimony"],
            "research_question": ["causes of change", "impact assessment", "interpretation debate"],
            "comparative_approach": ["cross-cultural comparison", "chronological comparison"],
            "historical_problem": ["causation question", "interpretation dispute", "evidence gap"],
            "analytical_tool": ["source criticism", "bias analysis", "contextual analysis"],
            "historical_account": ["official narrative", "witness testimony", "contemporary report"],
            "historian1": ["traditional historian", "revisionist historian"],
            "historian2": ["contemporary scholar", "modern interpreter"]
        }

        filled_pattern = pattern
        for placeholder in placeholders:
            if placeholder in fill_options:
                replacement = random.choice(fill_options[placeholder])
                filled_pattern = filled_pattern.replace(f"{{{placeholder}}}", replacement)

        return filled_pattern

    def _extract_terminology(self, question: str) -> List[str]:
        """Extract historical terminology."""
        question_lower = question.lower()
        terminology = []

        all_terms = (self.vocabulary.core_terms +
                    self.vocabulary.advanced_terms +
                    self.vocabulary.methodology_terms)

        for term in all_terms:
            if term.lower() in question_lower:
                terminology.append(term)

        return terminology
