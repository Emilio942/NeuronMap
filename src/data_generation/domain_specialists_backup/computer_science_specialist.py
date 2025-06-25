"""
Computer Science Domain Specialist
=================================

Specialized question generator for computer science domain with programming,
algorithms, systems, and theoretical computer science focus.
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


class ComputerScienceSpecialist(DomainSpecialist):
    """Domain specialist for computer science questions."""

    def __init__(self):
        super().__init__("Computer Science", DomainType.STEM)

    def _initialize_vocabulary(self) -> DomainVocabulary:
        """Initialize computer science vocabulary."""
        return DomainVocabulary(
            core_terms=[
                "algorithm", "data structure", "complexity", "recursion", "iteration",
                "array", "list", "tree", "graph", "stack", "queue", "hash table",
                "sorting", "searching", "programming", "function", "variable", "loop",
                "conditional", "class", "object", "inheritance", "polymorphism",
                "database", "query", "index", "transaction", "concurrency", "thread",
                "process", "memory", "cache", "processor", "network", "protocol"
            ],
            advanced_terms=[
                "computational complexity", "NP-completeness", "approximation algorithm",
                "dynamic programming", "greedy algorithm", "divide and conquer",
                "machine learning", "neural network", "deep learning", "backpropagation",
                "gradient descent", "overfitting", "regularization", "cross validation",
                "distributed systems", "consensus", "Byzantine fault tolerance",
                "CAP theorem", "eventual consistency", "load balancing", "microservices",
                "compiler optimization", "garbage collection", "virtual memory",
                "cryptography", "public key", "digital signature", "blockchain"
            ],
            methodology_terms=[
                "analysis", "design", "implementation", "testing", "debugging",
                "verification", "validation", "optimization", "profiling", "benchmarking",
                "software engineering", "agile", "waterfall", "version control",
                "code review", "unit testing", "integration testing", "refactoring",
                "design pattern", "architecture", "modeling", "simulation",
                "formal methods", "proof of correctness", "invariant", "precondition"
            ],
            concept_hierarchies={
                "algorithms": ["sorting algorithms", "search algorithms", "graph algorithms", "optimization algorithms"],
                "data_structures": ["linear structures", "tree structures", "graph structures", "hash structures"],
                "programming_paradigms": ["procedural", "object-oriented", "functional", "declarative"],
                "systems": ["operating systems", "database systems", "distributed systems", "embedded systems"],
                "theory": ["computability", "complexity theory", "formal languages", "automata theory"],
                "ai": ["machine learning", "computer vision", "natural language processing", "robotics"]
            },
            synonyms={
                "algorithm": ["procedure", "method"],
                "complexity": ["efficiency"],
                "function": ["procedure", "method", "routine"],
                "variable": ["identifier"],
                "loop": ["iteration"]
            }
        )

    def _initialize_question_patterns(self) -> Dict[QuestionType, List[str]]:
        """Initialize CS-specific question patterns."""
        return {
            QuestionType.CONCEPTUAL: [
                "Explain the difference between {concept1} and {concept2} in {context}.",
                "What are the key properties of {data_structure} that make it suitable for {application}?",
                "How does {algorithm} achieve its {complexity} through {technique}?",
                "Describe the trade-offs between {approach1} and {approach2} for {problem}.",
                "Why is {principle} important in {domain} applications?"
            ],
            QuestionType.QUANTITATIVE: [
                "Analyze the time complexity of {algorithm} operating on {data_structure}.",
                "Calculate the space complexity of {implementation} for input size n.",
                "Determine the worst-case performance of {operation} in {scenario}.",
                "Estimate the memory requirements for {system} handling {workload}.",
                "Compute the expected running time of {randomized_algorithm} on {input}."
            ],
            QuestionType.EXPERIMENTAL: [
                "Design an experiment to compare the performance of {algorithm1} vs {algorithm2}.",
                "How would you benchmark {system} under different {conditions}?",
                "What metrics would you use to evaluate {ml_model} performance?",
                "Propose a testing strategy for {software_component} in {environment}.",
                "Design a study to validate that {optimization} improves {metric}."
            ],
            QuestionType.APPLICATION: [
                "Implement {algorithm} to solve {problem} with constraints {constraints}.",
                "How would you use {data_structure} to optimize {operation} in {system}?",
                "Apply {technique} to improve the {characteristic} of {application}.",
                "Design a {system} that handles {requirement} using {technology}.",
                "Adapt {algorithm} for {specific_domain} requirements."
            ],
            QuestionType.ANALYTICAL: [
                "Compare {algorithm1} and {algorithm2} for {problem_type} considering {factors}.",
                "Analyze the security implications of {protocol} in {threat_model}.",
                "Evaluate the scalability of {architecture} for {use_case}.",
                "Investigate the impact of {parameter} on {system_behavior}.",
                "Examine the correctness of {implementation} using {verification_method}."
            ]
        }

    def generate_domain_questions(self, topic: str, count: int = 5,
                                question_type: Optional[QuestionType] = None) -> List[DomainQuestion]:
        """Generate computer science questions."""
        questions = []

        cs_contexts = {
            "algorithms": {
                "concepts": ["sorting", "searching", "optimization", "graph traversal"],
                "techniques": ["divide and conquer", "dynamic programming", "greedy approach"],
                "applications": ["pathfinding", "scheduling", "resource allocation"],
                "problems": ["shortest path", "minimum spanning tree", "maximum flow"]
            },
            "data structures": {
                "concepts": ["arrays", "linked lists", "trees", "hash tables"],
                "techniques": ["balancing", "hashing", "indexing", "compression"],
                "applications": ["databases", "caching", "symbol tables"],
                "problems": ["collision resolution", "load balancing", "memory management"]
            },
            "machine learning": {
                "concepts": ["supervised learning", "unsupervised learning", "reinforcement learning"],
                "techniques": ["gradient descent", "backpropagation", "cross validation"],
                "applications": ["classification", "regression", "clustering"],
                "problems": ["overfitting", "bias-variance tradeoff", "feature selection"]
            },
            "systems": {
                "concepts": ["concurrency", "distributed computing", "fault tolerance"],
                "techniques": ["locking", "consensus", "replication", "partitioning"],
                "applications": ["web services", "databases", "cloud computing"],
                "problems": ["deadlock", "race conditions", "network partitions"]
            }
        }

        context_key = self._find_best_context(topic, cs_contexts)
        context = cs_contexts.get(context_key, cs_contexts["algorithms"])

        question_types = [question_type] if question_type else list(QuestionType)

        for i in range(count):
            q_type = random.choice(question_types)
            pattern = random.choice(self.question_patterns[q_type])
            question_text = self._fill_cs_pattern(pattern, topic, context)

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
        """Find best CS context for topic."""
        topic_lower = topic.lower()

        for context_key in contexts.keys():
            if context_key in topic_lower:
                return context_key

        keyword_matches = {}
        for context_key, context_data in contexts.items():
            matches = 0
            for concept_list in context_data.values():
                for concept in concept_list:
                    if concept.lower() in topic_lower:
                        matches += 1
            keyword_matches[context_key] = matches

        best_context = max(keyword_matches, key=keyword_matches.get)
        return best_context if keyword_matches[best_context] > 0 else "algorithms"

    def _fill_cs_pattern(self, pattern: str, topic: str, context: Dict) -> str:
        """Fill question pattern with CS content."""
        placeholders = re.findall(r'\{(\w+)\}', pattern)

        fill_options = {
            "concept": context["concepts"] + [topic],
            "concept1": context["concepts"],
            "concept2": context["concepts"],
            "technique": context["techniques"],
            "application": context["applications"],
            "problem": context["problems"],
            "data_structure": ["array", "linked list", "binary tree", "hash table", "graph"],
            "algorithm": ["quicksort", "mergesort", "binary search", "dijkstra", "DFS", "BFS"],
            "algorithm1": ["quicksort", "mergesort", "heapsort"],
            "algorithm2": ["bubblesort", "insertion sort", "selection sort"],
            "complexity": ["O(n log n) time complexity", "O(1) space complexity", "linear performance"],
            "implementation": ["recursive solution", "iterative approach", "in-place algorithm"],
            "operation": ["insertion", "deletion", "search", "traversal"],
            "system": ["database", "web server", "cache", "distributed system"],
            "ml_model": ["neural network", "decision tree", "SVM", "random forest"],
            "software_component": ["API", "module", "service", "library"],
            "environment": ["production", "cloud", "mobile", "embedded"],
            "optimization": ["memory optimization", "performance tuning", "algorithmic improvement"],
            "metric": ["throughput", "latency", "accuracy", "precision"],
            "technology": ["microservices", "containers", "NoSQL", "message queues"],
            "architecture": ["client-server", "peer-to-peer", "microservices", "layered"],
            "use_case": ["high traffic", "real-time processing", "big data", "mobile"],
            "parameter": ["thread count", "cache size", "batch size", "learning rate"],
            "system_behavior": ["performance", "scalability", "reliability", "security"],
            "verification_method": ["formal verification", "testing", "code review", "static analysis"],
            "approach1": ["iterative approach", "recursive solution", "brute force"],
            "approach2": ["dynamic programming", "greedy algorithm", "approximation"],
            "domain": ["software engineering", "cybersecurity", "AI", "HCI"],
            "principle": ["separation of concerns", "DRY principle", "SOLID principles"],
            "context": ["object-oriented programming", "functional programming", "concurrent systems"],
            "scenario": ["worst case", "average case", "best case"],
            "workload": ["high concurrency", "large datasets", "real-time constraints"],
            "randomized_algorithm": ["quicksort", "randomized selection", "bloom filter"],
            "input": ["sorted array", "random data", "adversarial input"],
            "conditions": ["varying load", "network latency", "memory constraints"],
            "constraints": ["time limits", "memory bounds", "API restrictions"],
            "characteristic": ["performance", "security", "usability", "maintainability"],
            "problem_type": ["optimization problems", "search problems", "decision problems"],
            "factors": ["time complexity", "space complexity", "implementation complexity"],
            "protocol": ["HTTP", "TCP", "TLS", "OAuth"],
            "threat_model": ["passive adversary", "active adversary", "insider threat"],
            "requirement": ["scalability", "fault tolerance", "security", "performance"]
        }

        filled_pattern = pattern
        for placeholder in placeholders:
            if placeholder in fill_options:
                replacement = random.choice(fill_options[placeholder])
                filled_pattern = filled_pattern.replace(f"{{{placeholder}}}", replacement)

        return filled_pattern

    def _extract_terminology(self, question: str) -> List[str]:
        """Extract CS terminology used in question."""
        question_lower = question.lower()
        terminology = []

        all_terms = (self.vocabulary.core_terms +
                    self.vocabulary.advanced_terms +
                    self.vocabulary.methodology_terms)

        for term in all_terms:
            if term.lower() in question_lower:
                terminology.append(term)

        return terminology
