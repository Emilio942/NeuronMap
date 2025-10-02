"""
Induction Head Scanner Module

This module implements specialized scanners for detecting induction heads
and other specific circuit patterns in transformer models.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from .circuits import (
    NeuralCircuit, CircuitComponent, CircuitConnection, ComponentType,
    create_attention_head_component
)
from .model_integration import ModelManager

# Import config from utils (using relative import within the package)
try:
    from ..utils.config import AnalysisConfig
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.config import AnalysisConfig

logger = logging.getLogger(__name__)


@dataclass
class InductionHeadCandidate:
    """Represents a candidate induction head with its characteristics."""
    model_name: str
    layer: int
    head: int
    induction_score: float
    copying_score: float
    pattern_strength: float
    test_examples: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    @property
    def component_id(self) -> str:
        return f"{self.model_name}_attention_head_L{self.layer}_P{self.head}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'layer': self.layer,
            'head': self.head,
            'induction_score': self.induction_score,
            'copying_score': self.copying_score,
            'pattern_strength': self.pattern_strength,
            'test_examples': self.test_examples,
            'metadata': self.metadata,
            'component_id': self.component_id
        }


@dataclass
class CopyingHeadCandidate:
    """Represents a candidate copying/saliency head."""
    model_name: str
    layer: int
    head: int
    copying_score: float
    target_positions: List[int]  # Positions this head typically copies from
    consistency_score: float
    test_examples: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    @property
    def component_id(self) -> str:
        return f"{self.model_name}_attention_head_L{self.layer}_P{self.head}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'layer': self.layer,
            'head': self.head,
            'copying_score': self.copying_score,
            'target_positions': self.target_positions,
            'consistency_score': self.consistency_score,
            'test_examples': self.test_examples,
            'metadata': self.metadata,
            'component_id': self.component_id
        }


class InductionHeadScanner:
    """
    Scanner for detecting induction heads in transformer models.

    Induction heads are attention heads that look for patterns like:
    [A][B] ... [A][?] -> [B]

    They are crucial for in-context learning and sequence completion.
    """

    def __init__(self, model_manager: ModelManager,
                 config: Optional[AnalysisConfig] = None):
        self.model_manager = model_manager
        self.config = config or AnalysisConfig()
        self.device = model_manager.device

    def generate_induction_test_sequences(self, num_sequences: int = 20,
                                          seq_length: int = 50) -> List[str]:
        """
        Generate test sequences specifically designed to trigger induction heads.

        These sequences contain repeated patterns that induction heads should detect.
        """
        test_sequences = []

        # Pattern 1: Simple AB...AB pattern
        patterns = [
            "cat dog", "red blue", "apple orange", "one two", "hello world",
            "sun moon", "up down", "left right", "hot cold", "big small"
        ]

        for i in range(num_sequences // 2):
            pattern = patterns[i % len(patterns)]
            words = pattern.split()

            # Create sequence with repeated pattern
            sequence_parts = []
            for j in range(seq_length // 4):
                sequence_parts.extend(words)
                if j < seq_length // 4 - 1:
                    # Add some random words between patterns
                    filler_words = ["the", "and", "of", "to", "in", "is", "it", "that"]
                    sequence_parts.extend(np.random.choice(filler_words, size=2))

            test_sequences.append(" ".join(sequence_parts))

        # Pattern 2: More complex patterns with distractors
        for i in range(num_sequences // 2):
            # Create sequences with embedded patterns
            base_words = ["when", "where", "what", "why", "how", "who"]
            pattern_words = ["Alice", "Bob", "Charlie", "David"]

            sequence = []
            for j in range(seq_length // 6):
                # Add base context
                sequence.extend(np.random.choice(base_words, size=3))
                # Add pattern
                sequence.extend(np.random.choice(pattern_words, size=2))

            test_sequences.append(" ".join(sequence))

        return test_sequences

    def analyze_attention_patterns(
            self, model, tokenizer, text: str) -> Dict[str, np.ndarray]:
        """
        Analyze attention patterns for a given text.

        Returns attention weights for each layer and head.
        """
        model.eval()
        with torch.no_grad():
            # Tokenize text
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get model outputs with attention weights
            outputs = model(**inputs, output_attentions=True)

            # Extract attention weights
            # Tuple of (batch, num_heads, seq_len, seq_len)
            attention_weights = outputs.attentions

            # Convert to numpy for analysis
            patterns = {}
            for layer_idx, layer_attention in enumerate(attention_weights):
                layer_attention = layer_attention.cpu().numpy()
                # Remove batch dimension
                patterns[f"layer_{layer_idx}"] = layer_attention[0]

            return patterns

    def calculate_induction_score(self, attention_pattern: np.ndarray) -> float:
        """
        Calculate induction score for a single attention head.

        Induction heads should show strong attention to tokens that appeared
        after the previous occurrence of the current token.
        """
        seq_len = attention_pattern.shape[-1]
        if seq_len < 4:  # Need minimum sequence length
            return 0.0

        # Look for induction patterns
        induction_scores = []

        for pos in range(
                2, seq_len):  # Start from position 2 to allow for previous patterns
            # Get attention distribution for this position
            attention_dist = attention_pattern[pos, :]

            # For each previous position, check if it's followed by high attention
            for prev_pos in range(pos - 1):
                if prev_pos + 1 < pos:  # Ensure we have a next position
                    # Check if attention is high on the position after prev_pos
                    next_pos = prev_pos + 1
                    if next_pos < pos:  # Avoid attending to future positions
                        induction_scores.append(attention_dist[next_pos])

        return np.mean(induction_scores) if induction_scores else 0.0

    def calculate_copying_score(self, attention_pattern: np.ndarray) -> float:
        """
        Calculate copying score - how much this head copies from specific positions.
        """
        seq_len = attention_pattern.shape[-1]
        if seq_len < 2:
            return 0.0

        # Check attention to first token (BOS/important positions)
        first_token_attention = np.mean(attention_pattern[:, 0])

        # Check attention to last few tokens (recent context)
        recent_attention = 0.0
        if seq_len > 3:
            recent_attention = np.mean(attention_pattern[:, -3:])

        return max(first_token_attention, recent_attention)

    def scan_for_induction_heads(
            self,
            model_name: str,
            threshold: float = 0.3) -> List[InductionHeadCandidate]:
        """
        Scan a model for induction heads.

        Args:
            model_name: Name of the model to scan
            threshold: Minimum induction score to consider a head as candidate

        Returns:
            List of induction head candidates
        """
        logger.info(f"Scanning {model_name} for induction heads...")

        # Load model
        model, tokenizer = self.model_manager.get_model(model_name)

        # Generate test sequences
        test_sequences = self.generate_induction_test_sequences(10, 30)

        candidates = []

        # Get model dimensions
        num_layers = model.config.n_layer
        num_heads = model.config.n_head

        logger.info(f"Analyzing {num_layers} layers, {num_heads} heads per layer")

        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                induction_scores = []
                copying_scores = []
                pattern_strengths = []
                test_examples = []

                # Test on multiple sequences
                for seq_idx, test_seq in enumerate(test_sequences):
                    try:
                        # Analyze attention patterns
                        patterns = self.analyze_attention_patterns(
                            model, tokenizer, test_seq)

                        # Get specific head pattern
                        layer_pattern = patterns[f"layer_{layer_idx}"]
                        head_pattern = layer_pattern[head_idx]

                        # Calculate scores
                        induction_score = self.calculate_induction_score(head_pattern)
                        copying_score = self.calculate_copying_score(head_pattern)
                        # Measure of attention concentration
                        pattern_strength = np.std(head_pattern)

                        induction_scores.append(induction_score)
                        copying_scores.append(copying_score)
                        pattern_strengths.append(pattern_strength)

                        # Save example if score is high
                        if induction_score > threshold:
                            test_examples.append({
                                'sequence': test_seq,
                                'induction_score': induction_score,
                                'copying_score': copying_score,
                                'pattern_strength': pattern_strength
                            })

                    except Exception as e:
                        logger.warning(f"Error analyzing sequence {seq_idx} for layer {
                                       layer_idx}, head {head_idx}: {e}")
                        continue

                # Calculate average scores
                if induction_scores:
                    avg_induction = np.mean(induction_scores)
                    avg_copying = np.mean(copying_scores)
                    avg_pattern_strength = np.mean(pattern_strengths)

                    # Check if this head qualifies as induction head
                    if avg_induction > threshold:
                        candidate = InductionHeadCandidate(
                            model_name=model_name,
                            layer=layer_idx,
                            head=head_idx,
                            induction_score=avg_induction,
                            copying_score=avg_copying,
                            pattern_strength=avg_pattern_strength,
                            test_examples=test_examples[:3],  # Keep top 3 examples
                            metadata={
                                'num_test_sequences': len(test_sequences),
                                'threshold_used': threshold,
                                'std_induction_score': np.std(induction_scores),
                                'max_induction_score': np.max(induction_scores)
                            }
                        )
                        candidates.append(candidate)
                        logger.info(f"Found induction head candidate: Layer {layer_idx}, Head {
                                    head_idx} " f"(score: {avg_induction:.3f})")

        logger.info(f"Found {len(candidates)} induction head candidates")
        return candidates

    def create_induction_circuit(
            self, candidates: List[InductionHeadCandidate]) -> NeuralCircuit:
        """
        Create a neural circuit from induction head candidates.

        This identifies connections between induction heads and creates
        a circuit representation.
        """
        circuit = NeuralCircuit(
            circuit_id=f"induction_circuit_{candidates[0].model_name}",
            model_name=candidates[0].model_name,
            description="Circuit containing induction heads and their connections"
        )

        # Add all candidates as components
        for candidate in candidates:
            component = create_attention_head_component(
                candidate.model_name,
                candidate.layer,
                candidate.head,
                metadata={
                    'induction_score': candidate.induction_score,
                    'copying_score': candidate.copying_score,
                    'pattern_strength': candidate.pattern_strength,
                    'type': 'induction_head'
                }
            )
            circuit.add_component(component)

        # Create connections between heads in adjacent layers
        candidates_by_layer = {}
        for candidate in candidates:
            if candidate.layer not in candidates_by_layer:
                candidates_by_layer[candidate.layer] = []
            candidates_by_layer[candidate.layer].append(candidate)

        # Connect heads between consecutive layers
        sorted_layers = sorted(candidates_by_layer.keys())
        for i in range(len(sorted_layers) - 1):
            current_layer = sorted_layers[i]
            next_layer = sorted_layers[i + 1]

            for current_head in candidates_by_layer[current_layer]:
                for next_head in candidates_by_layer[next_layer]:
                    # Create connection with weight based on scores
                    weight = min(
                        current_head.induction_score,
                        next_head.induction_score)

                    source_comp = circuit.get_component(current_head.component_id)
                    target_comp = circuit.get_component(next_head.component_id)

                    if source_comp and target_comp:
                        connection = CircuitConnection(
                            source=source_comp,
                            target=target_comp,
                            weight=weight,
                            connection_type="induction_cascade",
                            metadata={'connection_type': 'induction_head_to_head'}
                        )
                        circuit.add_connection(connection)

        return circuit


class CopyingHeadScanner:
    """
    Scanner for detecting copying/saliency heads in transformer models.

    These heads copy information from specific positions like the first token
    or other salient positions in the sequence.
    """

    def __init__(self, model_manager: ModelManager,
                 config: Optional[AnalysisConfig] = None):
        self.model_manager = model_manager
        self.config = config or AnalysisConfig()
        self.device = model_manager.device

    def generate_copying_test_sequences(self, num_sequences: int = 15) -> List[str]:
        """Generate test sequences for detecting copying heads."""
        sequences = []

        # Sequences with important first tokens
        important_starters = [
            "IMPORTANT: The following information is crucial.",
            "NOTE: Please remember this key point.",
            "ALERT: This is a critical message.",
            "WARNING: Pay attention to this detail.",
            "REMINDER: Don't forget this instruction."
        ]

        for starter in important_starters:
            # Add some filler content
            filler = " ".join([
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
                "Ut enim ad minim veniam, quis nostrud exercitation."
            ])
            sequences.append(starter + " " + filler)

        # Sequences with repeated important tokens
        for i in range(num_sequences - len(important_starters)):
            key_word = ["DECISION", "CONCLUSION", "RESULT", "OUTCOME", "ANSWER"][i % 5]
            sequence = f"The {key_word} is very important. " + \
                "We need to consider many factors. " + \
                "After careful analysis, we can see that " + \
                f"the {key_word} matters most."
            sequences.append(sequence)

        return sequences

    def scan_for_copying_heads(self, model_name: str,
                               threshold: float = 0.5) -> List[CopyingHeadCandidate]:
        """Scan for copying heads in the model."""
        logger.info(f"Scanning {model_name} for copying heads...")

        model, tokenizer = self.model_manager.get_model(model_name)
        test_sequences = self.generate_copying_test_sequences()

        candidates = []
        num_layers = model.config.n_layer
        num_heads = model.config.n_head

        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                copying_scores = []
                position_preferences = []
                test_examples = []

                for test_seq in test_sequences:
                    try:
                        # Analyze attention patterns
                        scanner = InductionHeadScanner(self.model_manager, self.config)
                        patterns = scanner.analyze_attention_patterns(
                            model, tokenizer, test_seq)

                        layer_pattern = patterns[f"layer_{layer_idx}"]
                        head_pattern = layer_pattern[head_idx]

                        # Calculate copying score
                        copying_score = scanner.calculate_copying_score(head_pattern)
                        copying_scores.append(copying_score)

                        # Find preferred positions
                        avg_attention_by_pos = np.mean(head_pattern, axis=0)
                        preferred_positions = np.where(
                            avg_attention_by_pos > 0.3)[0].tolist()
                        position_preferences.append(preferred_positions)

                        if copying_score > threshold:
                            test_examples.append({
                                'sequence': test_seq,
                                'copying_score': copying_score,
                                'preferred_positions': preferred_positions
                            })

                    except Exception as e:
                        logger.warning(
                            f"Error analyzing sequence for copying head detection: {e}")
                        continue

                if copying_scores:
                    avg_copying = np.mean(copying_scores)

                    # Find most common preferred positions
                    all_positions = []
                    for pos_list in position_preferences:
                        all_positions.extend(pos_list)

                    if all_positions:
                        unique_positions, counts = np.unique(
                            all_positions, return_counts=True)
                        common_positions = unique_positions[counts > len(
                            test_sequences) * 0.3].tolist()
                        consistency_score = np.max(
                            counts) / len(test_sequences) if len(test_sequences) > 0 else 0
                    else:
                        common_positions = []
                        consistency_score = 0

                    if avg_copying > threshold and consistency_score > 0.3:
                        candidate = CopyingHeadCandidate(
                            model_name=model_name,
                            layer=layer_idx,
                            head=head_idx,
                            copying_score=avg_copying,
                            target_positions=common_positions,
                            consistency_score=consistency_score,
                            test_examples=test_examples[:3],
                            metadata={
                                'num_test_sequences': len(test_sequences),
                                'threshold_used': threshold,
                                'std_copying_score': np.std(copying_scores)
                            }
                        )
                        candidates.append(candidate)
                        logger.info(
                            f"Found copying head candidate: Layer {layer_idx}, Head {head_idx} " f"(score: {
                                avg_copying:.3f}, consistency: {
                                consistency_score:.3f})")

        logger.info(f"Found {len(candidates)} copying head candidates")
        return candidates


def save_scan_results(results: Dict[str, Any], filepath: Path) -> None:
    """Save scanning results to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Scan results saved to {filepath}")


def load_scan_results(filepath: Path) -> Dict[str, Any]:
    """Load scanning results from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        results = json.load(f)
    logger.info(f"Scan results loaded from {filepath}")
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO)

    # This would be run with a real model manager
    # model_manager = ModelManager()
    # scanner = InductionHeadScanner(model_manager)
    # candidates = scanner.scan_for_induction_heads("gpt2")
    # circuit = scanner.create_induction_circuit(candidates)

    print("Induction head scanner module loaded successfully!")
    print("Use with a ModelManager instance to scan for induction heads.")
