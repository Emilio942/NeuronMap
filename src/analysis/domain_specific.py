"""Domain-specific analysis modules for specialized model types.

This module implements analysis methods tailored for specific domains:
- Code understanding for programming models
- Mathematical reasoning for math models
- Multilingual analysis for language models
- Temporal analysis for sequential data
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import re
from dataclasses import dataclass
from abc import ABC, abstractmethod
import keyword
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class CodeAnalysisResult:
    """Results from code understanding analysis."""
    code_snippet: str
    syntax_elements: Dict[str, List[str]]
    complexity_metrics: Dict[str, float]
    activation_patterns: Dict[str, np.ndarray]
    understanding_score: float


@dataclass
class MathAnalysisResult:
    """Results from mathematical reasoning analysis."""
    expression: str
    math_concepts: List[str]
    operation_types: List[str]
    difficulty_score: float
    reasoning_steps: List[str]
    activation_patterns: Dict[str, np.ndarray]


@dataclass
class MultilingualResult:
    """Results from multilingual analysis."""
    text: str
    language: str
    linguistic_features: Dict[str, Any]
    cross_lingual_similarity: Dict[str, float]
    activation_patterns: Dict[str, np.ndarray]


@dataclass
class TemporalResult:
    """Results from temporal sequence analysis."""
    sequence: List[Any]
    temporal_features: Dict[str, Any]
    sequence_patterns: List[str]
    temporal_dependencies: Dict[str, float]
    activation_evolution: np.ndarray


class DomainAnalyzer(ABC):
    """Abstract base class for domain-specific analyzers."""

    def __init__(self, config_name: str = "default"):
        """Initialize domain analyzer."""
        from src.utils.config_manager import get_config
        self.config = get_config().get_experiment_config(config_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def analyze(self, inputs: List[str], model, tokenizer,
                target_layers: Optional[List[str]] = None) -> List[Any]:
        """Perform domain-specific analysis."""
        pass


class CodeUnderstandingAnalyzer(DomainAnalyzer):
    """Analyze how models understand and process code."""

    def __init__(self, config_name: str = "default"):
        """Initialize code understanding analyzer."""
        super().__init__(config_name)
        self.programming_keywords = set(keyword.kwlist)
        self.syntax_patterns = {
            'functions': r'def\s+(\w+)\s*\(',
            'classes': r'class\s+(\w+)\s*[:\(]',
            'variables': r'(\w+)\s*=\s*',
            'imports': r'(?:from\s+(\w+)\s+)?import\s+([\w\s,]+)',
            'control_flow': r'\b(if|else|elif|for|while|try|except|finally|with)\b',
            'operators': r'(\+\+|--|==|!=|<=|>=|&&|\|\||[+\-*/=<>])',
            'comments': r'#.*$',
            'strings': r'["\'].*?["\']',
            'numbers': r'\b\d+\.?\d*\b'
        }

    def analyze(self, code_snippets: List[str], model, tokenizer,
                target_layers: Optional[List[str]] = None) -> List[CodeAnalysisResult]:
        """
        Analyze code understanding in neural models.

        Args:
            code_snippets: List of code snippets to analyze
            model: Language model
            tokenizer: Model tokenizer
            target_layers: Specific layers to analyze

        Returns:
            List of CodeAnalysisResult objects
        """
        results = []

        for code in code_snippets:
            try:
                # Extract syntax elements
                syntax_elements = self._extract_syntax_elements(code)

                # Calculate complexity metrics
                complexity = self._calculate_complexity(code)

                # Get model activations
                activations = self._get_code_activations(
                    code, model, tokenizer, target_layers)

                # Calculate understanding score
                understanding_score = self._calculate_understanding_score(
                    syntax_elements, complexity, activations
                )

                result = CodeAnalysisResult(
                    code_snippet=code,
                    syntax_elements=syntax_elements,
                    complexity_metrics=complexity,
                    activation_patterns=activations,
                    understanding_score=understanding_score
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Code analysis failed for snippet: {e}")
                continue

        return results

    def _extract_syntax_elements(self, code: str) -> Dict[str, List[str]]:
        """Extract syntactic elements from code."""
        elements = {}

        for pattern_name, pattern in self.syntax_patterns.items():
            matches = re.findall(pattern, code, re.MULTILINE)
            if isinstance(matches[0], tuple) if matches else False:
                # Handle tuples from groups
                elements[pattern_name] = [match[0] if match[0] else match[1]
                                          for match in matches if any(match)]
            else:
                elements[pattern_name] = matches

        return elements

    def _calculate_complexity(self, code: str) -> Dict[str, float]:
        """Calculate code complexity metrics."""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]

        complexity = {
            'line_count': len(non_empty_lines),
            'cyclomatic_complexity': self._cyclomatic_complexity(code),
            'nesting_depth': self._max_nesting_depth(code),
            'function_count': len(re.findall(r'def\s+\w+', code)),
            'class_count': len(re.findall(r'class\s+\w+', code)),
            'comment_ratio': self._comment_ratio(code)
        }

        return complexity

    def _cyclomatic_complexity(self, code: str) -> float:
        """Calculate cyclomatic complexity (simplified)."""
        # Count decision points
        decision_patterns = [
            r'\bif\b', r'\belif\b', r'\bwhile\b', r'\bfor\b',
            r'\btry\b', r'\bexcept\b', r'\band\b', r'\bor\b'
        ]

        complexity = 1  # Base complexity
        for pattern in decision_patterns:
            complexity += len(re.findall(pattern, code))

        return float(complexity)

    def _max_nesting_depth(self, code: str) -> float:
        """Calculate maximum nesting depth."""
        lines = code.split('\n')
        max_depth = 0
        current_depth = 0

        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                # Count leading whitespace
                indent = len(line) - len(line.lstrip())
                depth = indent // 4  # Assuming 4-space indentation
                max_depth = max(max_depth, depth)

        return float(max_depth)

    def _comment_ratio(self, code: str) -> float:
        """Calculate ratio of comment lines to total lines."""
        lines = code.split('\n')
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        total_lines = len([line for line in lines if line.strip()])

        return comment_lines / max(total_lines, 1)

    def _get_code_activations(self, code: str, model, tokenizer,
                              target_layers: Optional[List[str]]) -> Dict[str, np.ndarray]:
        """Get model activations for code snippet."""
        try:
            # Tokenize code
            inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get activations
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            activations = {}
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                for i, hidden_state in enumerate(outputs.hidden_states):
                    layer_name = f"layer_{i}"
                    if target_layers is None or layer_name in target_layers:
                        # Average over sequence length
                        activation = hidden_state.mean(dim=1).cpu().numpy()
                        # Remove batch dimension
                        activations[layer_name] = activation[0]

            return activations

        except Exception as e:
            logger.error(f"Failed to get code activations: {e}")
            return {}

    def _calculate_understanding_score(self, syntax_elements: Dict[str, List[str]],
                                       complexity: Dict[str, float],
                                       activations: Dict[str, np.ndarray]) -> float:
        """Calculate code understanding score based on syntax and activations."""
        # Simple heuristic combining syntax richness and activation strength
        syntax_score = sum(len(elements)
                           for elements in syntax_elements.values()) / 10.0
        complexity_score = min(complexity.get('cyclomatic_complexity', 1.0) / 5.0, 1.0)

        activation_score = 0.0
        if activations:
            avg_activation = np.mean([np.mean(np.abs(act))
                                     for act in activations.values()])
            activation_score = min(avg_activation / 1.0, 1.0)  # Normalize

        return (syntax_score + complexity_score + activation_score) / 3.0


class MathematicalReasoningAnalyzer(DomainAnalyzer):
    """Analyze mathematical reasoning capabilities."""

    def __init__(self, config_name: str = "default"):
        """Initialize mathematical reasoning analyzer."""
        super().__init__(config_name)
        self.math_patterns = {
            'arithmetic': r'[\+\-\*/\^]',
            'equations': r'=',
            'fractions': r'\d+/\d+',
            'exponents': r'\d+\^\d+|\d+\*\*\d+',
            'parentheses': r'[\(\)]',
            'variables': r'\b[a-zA-Z]\b',
            'functions': r'(sin|cos|tan|log|exp|sqrt|abs)\s*\(',
            'constants': r'\b(pi|e|\d+\.\d+)\b'
        }

        self.operation_hierarchy = {
            'arithmetic': 1,
            'algebra': 2,
            'calculus': 3,
            'statistics': 2,
            'geometry': 2,
            'logic': 2
        }

    def analyze(self, math_expressions: List[str], model, tokenizer,
                target_layers: Optional[List[str]] = None) -> List[MathAnalysisResult]:
        """
        Analyze mathematical reasoning in neural models.

        Args:
            math_expressions: List of mathematical expressions/problems
            model: Language model
            tokenizer: Model tokenizer
            target_layers: Specific layers to analyze

        Returns:
            List of MathAnalysisResult objects
        """
        results = []

        for expression in math_expressions:
            try:
                # Extract mathematical concepts
                math_concepts = self._extract_math_concepts(expression)

                # Identify operation types
                operation_types = self._identify_operations(expression)

                # Calculate difficulty score
                difficulty = self._calculate_difficulty(math_concepts, operation_types)

                # Extract reasoning steps
                reasoning_steps = self._extract_reasoning_steps(expression)

                # Get model activations
                activations = self._get_math_activations(
                    expression, model, tokenizer, target_layers)

                result = MathAnalysisResult(
                    expression=expression,
                    math_concepts=math_concepts,
                    operation_types=operation_types,
                    difficulty_score=difficulty,
                    reasoning_steps=reasoning_steps,
                    activation_patterns=activations
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Math analysis failed for expression: {e}")
                continue

        return results

    def _extract_math_concepts(self, expression: str) -> List[str]:
        """Extract mathematical concepts from expression."""
        concepts = []

        for concept, pattern in self.math_patterns.items():
            if re.search(pattern, expression):
                concepts.append(concept)

        # Additional concept detection
        if re.search(r'(derivative|integral|limit)', expression.lower()):
            concepts.append('calculus')
        if re.search(r'(mean|median|std|variance)', expression.lower()):
            concepts.append('statistics')
        if re.search(r'(triangle|circle|square|angle)', expression.lower()):
            concepts.append('geometry')

        return list(set(concepts))

    def _identify_operations(self, expression: str) -> List[str]:
        """Identify types of mathematical operations."""
        operations = []

        if re.search(r'[\+\-]', expression):
            operations.append('addition_subtraction')
        if re.search(r'[\*/]', expression):
            operations.append('multiplication_division')
        if re.search(r'[\^\*\*]', expression):
            operations.append('exponentiation')
        if re.search(r'sqrt|log|sin|cos', expression):
            operations.append('transcendental')
        if re.search(r'=', expression):
            operations.append('equation_solving')

        return operations

    def _calculate_difficulty(
            self,
            concepts: List[str],
            operations: List[str]) -> float:
        """Calculate difficulty score for mathematical expression."""
        concept_difficulty = sum(
            self.operation_hierarchy.get(
                concept, 1) for concept in concepts)
        operation_complexity = len(operations)

        # Normalize to 0-1 scale
        difficulty = min((concept_difficulty + operation_complexity) / 10.0, 1.0)

        return difficulty

    def _extract_reasoning_steps(self, expression: str) -> List[str]:
        """Extract reasoning steps from mathematical expression."""
        # Simplified step extraction
        steps = []

        # Look for step indicators
        step_patterns = [
            r'step\s*\d+[:\.]?\s*(.+)',
            r'\d+\)\s*(.+)',
            r'first[,\s]+(.+)',
            r'then[,\s]+(.+)',
            r'finally[,\s]+(.+)',
            r'therefore[,\s]+(.+)'
        ]

        for pattern in step_patterns:
            matches = re.findall(pattern, expression.lower())
            steps.extend(matches)

        if not steps:
            # If no explicit steps, split by sentences
            sentences = re.split(r'[.!?]+', expression)
            steps = [s.strip() for s in sentences if s.strip()]

        return steps[:5]  # Limit to 5 steps

    def _get_math_activations(self,
                              expression: str,
                              model,
                              tokenizer,
                              target_layers: Optional[List[str]]) -> Dict[str,
                                                                          np.ndarray]:
        """Get model activations for mathematical expression."""
        try:
            # Tokenize expression
            inputs = tokenizer(
                expression,
                return_tensors="pt",
                padding=True,
                truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get activations
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            activations = {}
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                for i, hidden_state in enumerate(outputs.hidden_states):
                    layer_name = f"layer_{i}"
                    if target_layers is None or layer_name in target_layers:
                        # Average over sequence length
                        activation = hidden_state.mean(dim=1).cpu().numpy()
                        # Remove batch dimension
                        activations[layer_name] = activation[0]

            return activations

        except Exception as e:
            logger.error(f"Failed to get math activations: {e}")
            return {}


class MultilingualAnalyzer(DomainAnalyzer):
    """Analyze multilingual understanding and cross-lingual transfer."""

    def __init__(self, config_name: str = "default"):
        """Initialize multilingual analyzer."""
        super().__init__(config_name)
        self.language_features = {
            'character_sets': {
                'latin': r'[a-zA-Z]',
                'cyrillic': r'[а-яА-Я]',
                'arabic': r'[\u0600-\u06FF]',
                'chinese': r'[\u4e00-\u9fff]',
                'japanese': r'[\u3040-\u309F\u30A0-\u30FF]'
            },
            'linguistic_markers': {
                'word_order': ['SVO', 'SOV', 'VSO'],
                'morphology': ['agglutinative', 'fusional', 'isolating'],
                'writing_direction': ['LTR', 'RTL']
            }
        }

    def analyze(self, multilingual_texts: List[Tuple[str, str]], model, tokenizer,
                target_layers: Optional[List[str]] = None) -> List[MultilingualResult]:
        """
        Analyze multilingual understanding.

        Args:
            multilingual_texts: List of (text, language) tuples
            model: Language model
            tokenizer: Model tokenizer
            target_layers: Specific layers to analyze

        Returns:
            List of MultilingualResult objects
        """
        results = []
        all_activations = {}

        # First pass: collect all activations
        for text, language in multilingual_texts:
            activations = self._get_multilingual_activations(
                text, model, tokenizer, target_layers)
            all_activations[(text, language)] = activations

        # Second pass: analyze each text with cross-lingual comparisons
        for text, language in multilingual_texts:
            try:
                # Extract linguistic features
                features = self._extract_linguistic_features(text, language)

                # Calculate cross-lingual similarities
                similarities = self._calculate_cross_lingual_similarity(
                    text, language, all_activations
                )

                result = MultilingualResult(
                    text=text,
                    language=language,
                    linguistic_features=features,
                    cross_lingual_similarity=similarities,
                    activation_patterns=all_activations[(text, language)]
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Multilingual analysis failed for text: {e}")
                continue

        return results

    def _extract_linguistic_features(self, text: str, language: str) -> Dict[str, Any]:
        """Extract linguistic features from text."""
        features = {
            'language': language,
            'character_length': len(text),
            'word_count': len(text.split()),
            'character_sets': [],
            'script_diversity': 0.0
        }

        # Detect character sets
        for script, pattern in self.language_features['character_sets'].items():
            if re.search(pattern, text):
                features['character_sets'].append(script)

        # Calculate script diversity
        unique_chars = len(set(text.lower()))
        features['script_diversity'] = unique_chars / max(len(text), 1)

        # Language-specific features
        if language == 'english':
            features['avg_word_length'] = np.mean([len(word) for word in text.split()])
        elif language == 'german':
            # German compound words
            long_words = [word for word in text.split() if len(word) > 10]
            features['compound_word_ratio'] = len(
                long_words) / max(len(text.split()), 1)
        elif language == 'chinese':
            # Character-based language
            features['character_word_ratio'] = len(text) / max(len(text.split()), 1)

        return features

    def _calculate_cross_lingual_similarity(self, text: str, language: str,
                                            all_activations: Dict) -> Dict[str, float]:
        """Calculate similarity with other languages."""
        similarities = {}
        current_activations = all_activations.get((text, language), {})

        if not current_activations:
            return similarities

        for (other_text, other_language), other_activations in all_activations.items():
            if other_language != language and other_activations:
                # Calculate activation similarity
                similarity = self._activation_similarity(
                    current_activations, other_activations)
                if other_language not in similarities:
                    similarities[other_language] = []
                similarities[other_language].append(similarity)

        # Average similarities per language
        for lang, sims in similarities.items():
            similarities[lang] = np.mean(sims)

        return similarities

    def _activation_similarity(self, act1: Dict[str, np.ndarray],
                               act2: Dict[str, np.ndarray]) -> float:
        """Calculate cosine similarity between activation patterns."""
        similarities = []

        for layer in act1.keys():
            if layer in act2:
                # Flatten activations
                vec1 = act1[layer].flatten()
                vec2 = act2[layer].flatten()

                # Calculate cosine similarity
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)

                if norm1 > 0 and norm2 > 0:
                    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
                    similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    def _get_multilingual_activations(self,
                                      text: str,
                                      model,
                                      tokenizer,
                                      target_layers: Optional[List[str]]) -> Dict[str,
                                                                                  np.ndarray]:
        """Get model activations for multilingual text."""
        try:
            # Tokenize text
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get activations
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            activations = {}
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                for i, hidden_state in enumerate(outputs.hidden_states):
                    layer_name = f"layer_{i}"
                    if target_layers is None or layer_name in target_layers:
                        # Average over sequence length
                        activation = hidden_state.mean(dim=1).cpu().numpy()
                        # Remove batch dimension
                        activations[layer_name] = activation[0]

            return activations

        except Exception as e:
            logger.error(f"Failed to get multilingual activations: {e}")
            return {}


class TemporalAnalyzer(DomainAnalyzer):
    """Analyze temporal patterns and dependencies in sequential data."""

    def __init__(self, config_name: str = "default"):
        """Initialize temporal analyzer."""
        super().__init__(config_name)
        self.temporal_patterns = [
            'sequential', 'cyclic', 'trending', 'stationary',
            'seasonal', 'irregular', 'linear', 'exponential'
        ]

    def analyze(self, sequences: List[List[str]], model, tokenizer,
                target_layers: Optional[List[str]] = None) -> List[TemporalResult]:
        """
        Analyze temporal patterns in sequential data.

        Args:
            sequences: List of sequences (each sequence is a list of strings)
            model: Language model
            tokenizer: Model tokenizer
            target_layers: Specific layers to analyze

        Returns:
            List of TemporalResult objects
        """
        results = []

        for sequence in sequences:
            try:
                # Extract temporal features
                features = self._extract_temporal_features(sequence)

                # Identify sequence patterns
                patterns = self._identify_sequence_patterns(sequence)

                # Calculate temporal dependencies
                dependencies = self._calculate_temporal_dependencies(sequence)

                # Get activation evolution
                activation_evolution = self._get_temporal_activations(
                    sequence, model, tokenizer, target_layers
                )

                result = TemporalResult(
                    sequence=sequence,
                    temporal_features=features,
                    sequence_patterns=patterns,
                    temporal_dependencies=dependencies,
                    activation_evolution=activation_evolution
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Temporal analysis failed for sequence: {e}")
                continue

        return results

    def _extract_temporal_features(self, sequence: List[str]) -> Dict[str, Any]:
        """Extract temporal features from sequence."""
        features = {
            'length': len(sequence),
            'unique_elements': len(set(sequence)),
            'repetition_ratio': 1.0 - len(set(sequence)) / max(len(sequence), 1),
            'avg_element_length': np.mean([len(elem) for elem in sequence]),
            'position_entropy': self._calculate_position_entropy(sequence)
        }

        return features

    def _identify_sequence_patterns(self, sequence: List[str]) -> List[str]:
        """Identify patterns in sequence."""
        patterns = []

        # Check for repetitive patterns
        if len(set(sequence)) < len(sequence) * 0.5:
            patterns.append('repetitive')

        # Check for increasing/decreasing patterns (for numeric sequences)
        try:
            numeric_seq = [float(item) for item in sequence]
            if all(x <= y for x, y in zip(numeric_seq, numeric_seq[1:])):
                patterns.append('increasing')
            elif all(x >= y for x, y in zip(numeric_seq, numeric_seq[1:])):
                patterns.append('decreasing')
        except ValueError:
            pass

        # Check for alternating patterns
        if len(sequence) >= 4:
            alternating = all(sequence[i] == sequence[i + 2]
                              for i in range(len(sequence) - 2))
            if alternating:
                patterns.append('alternating')

        return patterns

    def _calculate_temporal_dependencies(self, sequence: List[str]) -> Dict[str, float]:
        """Calculate temporal dependencies in sequence."""
        dependencies = {}

        # Calculate lag-1 autocorrelation (simplified for text)
        if len(sequence) > 1:
            # Convert to numeric representation for correlation
            element_counts = Counter(sequence)
            numeric_seq = [element_counts[elem] for elem in sequence]

            if len(numeric_seq) > 1:
                lag1_corr = np.corrcoef(numeric_seq[:-1], numeric_seq[1:])[0, 1]
                dependencies['lag_1_correlation'] = lag1_corr if not np.isnan(
                    lag1_corr) else 0.0

        # Calculate positional consistency
        position_consistency = self._calculate_position_consistency(sequence)
        dependencies['position_consistency'] = position_consistency

        return dependencies

    def _calculate_position_entropy(self, sequence: List[str]) -> float:
        """Calculate entropy of element positions."""
        if not sequence:
            return 0.0

        element_positions = defaultdict(list)
        for i, elem in enumerate(sequence):
            element_positions[elem].append(i)

        total_entropy = 0.0
        for positions in element_positions.values():
            if len(positions) > 1:
                # Calculate position distribution entropy
                pos_probs = np.array(positions) / len(sequence)
                entropy = -np.sum(pos_probs * np.log2(pos_probs + 1e-8))
                total_entropy += entropy

        return total_entropy / max(len(element_positions), 1)

    def _calculate_position_consistency(self, sequence: List[str]) -> float:
        """Calculate how consistently elements appear in similar positions."""
        if len(sequence) < 2:
            return 1.0

        element_positions = defaultdict(list)
        for i, elem in enumerate(sequence):
            relative_pos = i / (len(sequence) - 1)  # Normalize position
            element_positions[elem].append(relative_pos)

        consistencies = []
        for positions in element_positions.values():
            if len(positions) > 1:
                # Calculate standard deviation of positions
                consistency = 1.0 - np.std(positions)
                consistencies.append(max(consistency, 0.0))

        return np.mean(consistencies) if consistencies else 1.0

    def _get_temporal_activations(self, sequence: List[str], model, tokenizer,
                                  target_layers: Optional[List[str]]) -> np.ndarray:
        """Get activation evolution throughout sequence processing."""
        try:
            activations_over_time = []

            # Process sequence incrementally
            for i in range(1, len(sequence) + 1):
                partial_sequence = " ".join(sequence[:i])

                # Tokenize partial sequence
                inputs = tokenizer(partial_sequence, return_tensors="pt",
                                   padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get activations
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)

                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    # Use last layer's last token activation
                    last_hidden = outputs.hidden_states[-1][:, -1, :].cpu().numpy()
                    activations_over_time.append(last_hidden[0])

            return np.array(activations_over_time)

        except Exception as e:
            logger.error(f"Failed to get temporal activations: {e}")
            return np.array([])


class DomainSpecificPipeline:
    """Pipeline for running all domain-specific analyses."""

    def __init__(self, config_name: str = "default"):
        """Initialize domain-specific analysis pipeline."""
        self.config_name = config_name
        self.code_analyzer = CodeUnderstandingAnalyzer(config_name)
        self.math_analyzer = MathematicalReasoningAnalyzer(config_name)
        self.multilingual_analyzer = MultilingualAnalyzer(config_name)
        self.temporal_analyzer = TemporalAnalyzer(config_name)

    def run_comprehensive_analysis(self, inputs: Dict[str, Any], model, tokenizer,
                                   output_dir: str) -> Dict[str, Any]:
        """
        Run comprehensive domain-specific analysis.

        Args:
            inputs: Dictionary with domain-specific inputs
            model: Language model
            tokenizer: Model tokenizer
            output_dir: Directory to save results

        Returns:
            Dictionary with all analysis results
        """
        logger.info("Running comprehensive domain-specific analysis...")

        results = {
            'code_analysis': [],
            'math_analysis': [],
            'multilingual_analysis': [],
            'temporal_analysis': [],
            'metadata': {
                'config': self.config_name,
                'domains_analyzed': []
            }
        }

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Code understanding analysis
            if 'code_snippets' in inputs:
                logger.info("Running code understanding analysis...")
                code_results = self.code_analyzer.analyze(
                    inputs['code_snippets'], model, tokenizer
                )
                results['code_analysis'] = [result.__dict__ for result in code_results]
                results['metadata']['domains_analyzed'].append('code')

            # Mathematical reasoning analysis
            if 'math_expressions' in inputs:
                logger.info("Running mathematical reasoning analysis...")
                math_results = self.math_analyzer.analyze(
                    inputs['math_expressions'], model, tokenizer
                )
                results['math_analysis'] = [result.__dict__ for result in math_results]
                results['metadata']['domains_analyzed'].append('math')

            # Multilingual analysis
            if 'multilingual_texts' in inputs:
                logger.info("Running multilingual analysis...")
                multilingual_results = self.multilingual_analyzer.analyze(
                    inputs['multilingual_texts'], model, tokenizer
                )
                results['multilingual_analysis'] = [
                    result.__dict__ for result in multilingual_results]
                results['metadata']['domains_analyzed'].append('multilingual')

            # Temporal analysis
            if 'sequences' in inputs:
                logger.info("Running temporal analysis...")
                temporal_results = self.temporal_analyzer.analyze(
                    inputs['sequences'], model, tokenizer
                )
                results['temporal_analysis'] = [
                    result.__dict__ for result in temporal_results]
                results['metadata']['domains_analyzed'].append('temporal')

            # Save results
            with open(output_path / "domain_specific_results.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(
                f"Domain-specific analysis completed. Results saved to {output_dir}")
            return results

        except Exception as e:
            logger.error(f"Domain-specific analysis failed: {e}")
            return results


# Export main classes
__all__ = [
    'CodeUnderstandingAnalyzer',
    'MathematicalReasoningAnalyzer',
    'MultilingualAnalyzer',
    'TemporalAnalyzer',
    'DomainSpecificPipeline',
    'CodeAnalysisResult',
    'MathAnalysisResult',
    'MultilingualResult',
    'TemporalResult'
]
