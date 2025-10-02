"""Attention-specific analysis for transformer models."""

import numpy as np
import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict
from collections import defaultdict
from collections import defaultdict

from ..utils.config import get_config

logger = logging.getLogger(__name__)


class AttentionAnalyzer:
    """Specialized analyzer for attention mechanisms in transformer models."""

    def __init__(self, config_name: str = "default"):
        """Initialize attention analyzer.

        Args:
            config_name: Name of experiment configuration.
        """
        # Defensive configuration loading
        self.config = None
        self.experiment_config = {}
        try:
            config_source = get_config()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Failed to load configuration via utils.config: %s", exc)
            config_source = None
        if config_source is not None:
            self.config = config_source
            self.experiment_config = self._extract_experiment_config(config_source, config_name)
        if not self.experiment_config:
            logger.debug(
                "No experiment configuration found for '%s'; using empty defaults",
                config_name,
            )
            self.experiment_config = {}

        self.attention_patterns = {}
        self.attention_weights = {}
        self.logger = logging.getLogger(__name__)

    def create_attention_hook(self, layer_name: str, head_idx: Optional[int] = None):
        """Create hook to capture attention patterns.

        Args:
            layer_name: Name of attention layer.
            head_idx: Specific attention head index (None for all heads).

        Returns:
            Hook function that captures attention weights.
        """
        def attention_hook(module, input_tensor, output_tensor):
            """Hook to capture attention weights."""
            try:
                # Different models have different attention output formats
                if hasattr(module, 'num_heads'):
                    # Multi-head attention layer
                    if isinstance(output_tensor, tuple) and len(output_tensor) > 1:
                        # Output typically includes attention weights as second element
                        attention_weights = output_tensor[1]
                    else:
                        # Some models don't return attention weights by default
                        logger.warning(f"No attention weights found for layer {layer_name}")
                        return
                else:
                    logger.warning(f"Layer {layer_name} doesn't appear to be an attention layer")
                    return

                # Store attention weights
                if attention_weights is not None:
                    # Move to CPU and detach
                    weights = attention_weights.detach().cpu()

                    # Store based on head selection
                    if head_idx is not None and weights.dim() >= 3:
                        # Select specific head: [batch, heads, seq, seq] -> [batch, seq, seq]
                        if head_idx < weights.shape[1]:
                            weights = weights[:, head_idx, :, :]
                        else:
                            logger.warning(f"Head index {head_idx} out of range for layer {layer_name}")
                            return

                    # Store the patterns
                    if layer_name not in self.attention_patterns:
                        self.attention_patterns[layer_name] = []

                    self.attention_patterns[layer_name].append(weights.numpy())

            except Exception as e:
                logger.error(f"Error in attention hook for {layer_name}: {e}")

        return attention_hook

    def analyze_attention_patterns(self, attention_weights: np.ndarray,
                                 tokens: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze attention patterns from weight matrix.

        Args:
            attention_weights: Attention weight matrix [batch, heads, seq_len, seq_len], [heads, seq_len, seq_len] or [seq_len, seq_len].
            tokens: List of tokens corresponding to sequence positions. If None, generic names are used.

        Returns:
            Dictionary with attention pattern analysis.
        """
        # Generate default tokens if none provided
        if tokens is None:
            seq_len = attention_weights.shape[-1]  # Last dimension is sequence length
            tokens = [f"token_{i}" for i in range(seq_len)]

        if attention_weights.ndim == 4:
            # Batch dimension: average across batch and analyze
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            # Average across batch dimension
            attention_weights = np.mean(attention_weights, axis=0)

        if attention_weights.ndim == 3:
            # Multi-head: analyze each head separately
            num_heads, seq_len, _ = attention_weights.shape
            head_analyses = []

            for head_idx in range(num_heads):
                head_weights = attention_weights[head_idx]
                head_analysis = self._analyze_single_attention_matrix(head_weights, tokens)
                head_analysis['head_idx'] = head_idx
                head_analyses.append(head_analysis)

            # Aggregate analysis across heads
            aggregate_weights = np.mean(attention_weights, axis=0)
            aggregate_analysis = self._analyze_single_attention_matrix(aggregate_weights, tokens)

            # Add test-expected keys
            attention_entropy_by_head = self._compute_attention_entropy_by_head(attention_weights)
            head_importance = self._compute_head_importance(attention_weights)
            attention_distance = self._compute_attention_distance(attention_weights)

            return {
                'type': 'multi_head',
                'num_heads': num_heads,
                'head_analyses': head_analyses,
                'aggregate_analysis': aggregate_analysis,
                'attention_entropy_by_head': attention_entropy_by_head,
                'attention_entropy': attention_entropy_by_head,  # Alias for test compatibility
                'head_importance': head_importance,
                'attention_distance': attention_distance
            }

        elif attention_weights.ndim == 2:
            # Single attention matrix
            analysis = self._analyze_single_attention_matrix(attention_weights, tokens)
            analysis['type'] = 'single_head'
            return analysis

        else:
            raise ValueError(f"Unexpected attention weight dimensions: {attention_weights.shape}")

    def _analyze_single_attention_matrix(self, attention_matrix: np.ndarray,
                                       tokens: List[str]) -> Dict[str, Any]:
        """Analyze a single attention matrix.

        Args:
            attention_matrix: 2D attention weight matrix [seq_len, seq_len].
            tokens: List of tokens.

        Returns:
            Analysis results for single attention matrix.
        """
        seq_len = attention_matrix.shape[0]

        # Basic statistics
        analysis = {
            'sequence_length': seq_len,
            'attention_statistics': {
                'mean': float(np.mean(attention_matrix)),
                'std': float(np.std(attention_matrix)),
                'min': float(np.min(attention_matrix)),
                'max': float(np.max(attention_matrix))
            }
        }

        # Attention entropy (how distributed attention is)
        attention_entropy = self._compute_attention_entropy(attention_matrix)
        analysis['attention_entropy'] = {
            'per_position': attention_entropy.tolist(),
            'mean_entropy': float(np.mean(attention_entropy)),
            'std_entropy': float(np.std(attention_entropy))
        }

        # Self-attention strength (diagonal values)
        self_attention = np.diag(attention_matrix)
        analysis['self_attention'] = {
            'values': self_attention.tolist(),
            'mean': float(np.mean(self_attention)),
            'positions_with_high_self_attention': [
                {'position': int(i), 'token': tokens[i] if i < len(tokens) else f"pos_{i}",
                 'attention_weight': float(self_attention[i])}
                for i in np.where(self_attention > np.percentile(self_attention, 90))[0]
            ]
        }

        # Most attended positions for each token
        analysis['most_attended_positions'] = []
        for i in range(seq_len):
            attended_positions = attention_matrix[i, :]
            top_positions = np.argsort(attended_positions)[-3:][::-1]  # Top 3

            position_info = {
                'source_position': i,
                'source_token': tokens[i] if i < len(tokens) else f"pos_{i}",
                'top_attended': [
                    {
                        'position': int(pos),
                        'token': tokens[pos] if pos < len(tokens) else f"pos_{pos}",
                        'attention_weight': float(attended_positions[pos])
                    }
                    for pos in top_positions
                ]
            }
            analysis['most_attended_positions'].append(position_info)

        # Distance-based attention analysis
        analysis['distance_attention'] = self._analyze_attention_distances(attention_matrix)

        # Token-type specific attention patterns
        if tokens:
            analysis['token_patterns'] = self._analyze_token_attention_patterns(
                attention_matrix, tokens
            )

        return analysis

    def _compute_attention_entropy(self, attention_matrix: np.ndarray) -> np.ndarray:
        """Compute attention entropy for each position.

        Args:
            attention_matrix: Attention weight matrix.

        Returns:
            Array of entropy values for each position.
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        attention_with_epsilon = attention_matrix + epsilon

        # Compute entropy for each row (each source position)
        entropy = -np.sum(attention_with_epsilon * np.log(attention_with_epsilon), axis=1)
        return entropy

    def _compute_attention_entropy_by_head(self, attention_weights: np.ndarray) -> List[Dict[str, float]]:
        """Compute attention entropy statistics by head.

        Args:
            attention_weights: Multi-head attention weights [heads, seq, seq].

        Returns:
            List of entropy statistics for each head.
        """
        head_entropies = []

        for head_idx in range(attention_weights.shape[0]):
            head_matrix = attention_weights[head_idx]
            entropy = self._compute_attention_entropy(head_matrix)

            head_entropies.append({
                'head_idx': head_idx,
                'mean_entropy': float(np.mean(entropy)),
                'std_entropy': float(np.std(entropy)),
                'min_entropy': float(np.min(entropy)),
                'max_entropy': float(np.max(entropy))
            })

        return head_entropies

    def _analyze_attention_distances(self, attention_matrix: np.ndarray) -> Dict[str, Any]:
        """Analyze attention as a function of token distance.

        Args:
            attention_matrix: Attention weight matrix.

        Returns:
            Distance-based attention analysis.
        """
        seq_len = attention_matrix.shape[0]

        # Compute attention by distance
        distance_attention = defaultdict(list)

        for i in range(seq_len):
            for j in range(seq_len):
                distance = abs(i - j)
                attention_weight = attention_matrix[i, j]
                distance_attention[distance].append(attention_weight)

        # Compute statistics for each distance
        distance_stats = {}
        for distance, weights in distance_attention.items():
            distance_stats[distance] = {
                'mean_attention': float(np.mean(weights)),
                'std_attention': float(np.std(weights)),
                'count': len(weights)
            }

        # Overall patterns
        distances = sorted(distance_stats.keys())
        mean_attention_by_distance = [distance_stats[d]['mean_attention'] for d in distances]

        return {
            'by_distance': distance_stats,
            'distances': distances,
            'mean_attention_by_distance': mean_attention_by_distance,
            'local_attention_ratio': self._compute_local_attention_ratio(attention_matrix),
            'distant_attention_ratio': self._compute_distant_attention_ratio(attention_matrix)
        }

    def _compute_local_attention_ratio(self, attention_matrix: np.ndarray,
                                     local_window: int = 3) -> float:
        """Compute ratio of attention to nearby tokens.

        Args:
            attention_matrix: Attention weight matrix.
            local_window: Size of local window around each position.

        Returns:
            Ratio of local attention.
        """
        seq_len = attention_matrix.shape[0]
        total_local_attention = 0.0
        total_attention = 0.0

        for i in range(seq_len):
            for j in range(seq_len):
                attention_weight = attention_matrix[i, j]
                total_attention += attention_weight

                if abs(i - j) <= local_window:
                    total_local_attention += attention_weight

        return total_local_attention / total_attention if total_attention > 0 else 0.0

    def _compute_distant_attention_ratio(self, attention_matrix: np.ndarray,
                                       distant_threshold: int = 10) -> float:
        """Compute ratio of attention to distant tokens.

        Args:
            attention_matrix: Attention weight matrix.
            distant_threshold: Minimum distance to be considered distant.

        Returns:
            Ratio of distant attention.
        """
        seq_len = attention_matrix.shape[0]
        total_distant_attention = 0.0
        total_attention = 0.0

        for i in range(seq_len):
            for j in range(seq_len):
                attention_weight = attention_matrix[i, j]
                total_attention += attention_weight

                if abs(i - j) >= distant_threshold:
                    total_distant_attention += attention_weight

        return total_distant_attention / total_attention if total_attention > 0 else 0.0

    def _analyze_token_attention_patterns(self, attention_matrix: np.ndarray,
                                        tokens: List[str]) -> Dict[str, Any]:
        """Analyze attention patterns for different token types.

        Args:
            attention_matrix: Attention weight matrix.
            tokens: List of tokens.

        Returns:
            Token-type specific attention patterns.
        """
        # Simple token categorization
        token_categories = {}
        for i, token in enumerate(tokens):
            if i >= attention_matrix.shape[0]:
                break

            category = self._categorize_token(token)
            if category not in token_categories:
                token_categories[category] = []
            token_categories[category].append(i)

        # Analyze attention for each category
        category_analysis = {}
        for category, positions in token_categories.items():
            if not positions:
                continue

            # Average attention received by this category
            received_attention = np.mean([
                np.sum(attention_matrix[:, pos]) for pos in positions
            ])

            # Average attention given by this category
            given_attention = np.mean([
                np.sum(attention_matrix[pos, :]) for pos in positions
            ])

            # Self-attention within category
            self_attention = np.mean([
                attention_matrix[pos, pos] for pos in positions
            ])

            category_analysis[category] = {
                'count': len(positions),
                'positions': positions,
                'average_received_attention': float(received_attention),
                'average_given_attention': float(given_attention),
                'average_self_attention': float(self_attention)
            }

        return category_analysis

    def _categorize_token(self, token: str) -> str:
        """Categorize a token into a basic type.

        Args:
            token: Input token.

        Returns:
            Token category.
        """
        token = token.strip()

        if token.startswith('<') and token.endswith('>'):
            return 'special'
        elif token in ['.', '!', '?', ',', ';', ':']:
            return 'punctuation'
        elif token.isdigit():
            return 'number'
        elif token.isupper():
            return 'uppercase'
        elif token.islower():
            return 'lowercase'
        elif token.istitle():
            return 'titlecase'
        else:
            return 'other'

    def extract_attention_circuits(self, attention_patterns: Dict[str, List[np.ndarray]],
                                 threshold: float = 0.1) -> Dict[str, Any]:
        """Extract attention circuits from patterns across layers.

        Args:
            attention_patterns: Dictionary of attention patterns by layer.
            threshold: Minimum attention weight to consider significant.

        Returns:
            Extracted attention circuits.
        """
        circuits = {
            'layer_circuits': {},
            'cross_layer_circuits': [],
            'global_patterns': {}
        }

        # Analyze circuits within each layer
        for layer_name, patterns in attention_patterns.items():
            if not patterns:
                continue

            # Average attention pattern across all inputs
            avg_pattern = np.mean(patterns, axis=0)

            # Find strong attention connections
            strong_connections = self._find_strong_connections(avg_pattern, threshold)

            circuits['layer_circuits'][layer_name] = {
                'average_pattern': avg_pattern.tolist(),
                'strong_connections': strong_connections,
                'pattern_consistency': self._compute_pattern_consistency(patterns)
            }

        # Analyze cross-layer patterns (if multiple layers available)
        layer_names = list(attention_patterns.keys())
        if len(layer_names) > 1:
            cross_layer_analysis = self._analyze_cross_layer_attention(attention_patterns)
            circuits['cross_layer_circuits'] = cross_layer_analysis

        return circuits

    def _find_strong_connections(self, attention_matrix: np.ndarray,
                               threshold: float) -> List[Dict[str, Any]]:
        """Find strong attention connections in a matrix.

        Args:
            attention_matrix: Attention weight matrix.
            threshold: Minimum weight threshold.

        Returns:
            List of strong connections.
        """
        connections = []

        if attention_matrix.ndim == 3:
            # Multi-head attention
            for head in range(attention_matrix.shape[0]):
                head_matrix = attention_matrix[head]
                head_connections = self._find_connections_2d(head_matrix, threshold)
                for conn in head_connections:
                    conn['head'] = head
                connections.extend(head_connections)

        elif attention_matrix.ndim == 2:
            # Single attention matrix
            connections = self._find_connections_2d(attention_matrix, threshold)

        return connections

    def _find_connections_2d(self, matrix: np.ndarray, threshold: float) -> List[Dict[str, Any]]:
        """Find strong connections in 2D attention matrix.

        Args:
            matrix: 2D attention matrix.
            threshold: Weight threshold.

        Returns:
            List of connections above threshold.
        """
        connections = []
        rows, cols = np.where(matrix >= threshold)

        for i, j in zip(rows, cols):
            connections.append({
                'from_position': int(i),
                'to_position': int(j),
                'weight': float(matrix[i, j]),
                'is_self_attention': i == j
            })

        # Sort by weight (strongest first)
        connections.sort(key=lambda x: x['weight'], reverse=True)
        return connections

    def _compute_pattern_consistency(self, patterns: List[np.ndarray]) -> float:
        """Compute consistency of attention patterns across inputs.

        Args:
            patterns: List of attention patterns.

        Returns:
            Consistency score (higher = more consistent).
        """
        if len(patterns) < 2:
            return 1.0

        # Compute pairwise correlations
        correlations = []
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                pattern1 = patterns[i].flatten()
                pattern2 = patterns[j].flatten()

                # Compute correlation
                if len(pattern1) == len(pattern2):
                    correlation = np.corrcoef(pattern1, pattern2)[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(correlation)

        return float(np.mean(correlations)) if correlations else 0.0

    def _analyze_cross_layer_attention(self, attention_patterns: Dict[str, List[np.ndarray]]) -> List[Dict[str, Any]]:
        """Analyze attention patterns across layers.

        Args:
            attention_patterns: Attention patterns by layer.

        Returns:
            Cross-layer attention analysis.
        """
        layer_names = sorted(attention_patterns.keys())
        cross_layer_results = []

        # Compare patterns between consecutive layers
        for i in range(len(layer_names) - 1):
            layer1 = layer_names[i]
            layer2 = layer_names[i + 1]

            patterns1 = attention_patterns[layer1]
            patterns2 = attention_patterns[layer2]

            if patterns1 and patterns2:
                # Average patterns for comparison
                avg_pattern1 = np.mean(patterns1, axis=0)
                avg_pattern2 = np.mean(patterns2, axis=0)

                # Compute similarity
                similarity = self._compute_pattern_similarity(avg_pattern1, avg_pattern2)

                cross_layer_results.append({
                    'layer1': layer1,
                    'layer2': layer2,
                    'similarity': similarity,
                    'pattern_evolution': self._analyze_pattern_evolution(avg_pattern1, avg_pattern2)
                })

        return cross_layer_results

    def _compute_pattern_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Compute similarity between two attention patterns.

        Args:
            pattern1: First attention pattern.
            pattern2: Second attention pattern.

        Returns:
            Similarity score.
        """
        # Ensure same shape for comparison
        if pattern1.shape != pattern2.shape:
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(pattern1.shape, pattern2.shape))
            pattern1 = pattern1[:min_shape[0], :min_shape[1]] if pattern1.ndim >= 2 else pattern1[:min_shape[0]]
            pattern2 = pattern2[:min_shape[0], :min_shape[1]] if pattern2.ndim >= 2 else pattern2[:min_shape[0]]

        # Flatten and compute correlation
        flat1 = pattern1.flatten()
        flat2 = pattern2.flatten()

        correlation = np.corrcoef(flat1, flat2)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0

    def _analyze_pattern_evolution(self, pattern1: np.ndarray, pattern2: np.ndarray) -> Dict[str, float]:
        """Analyze how attention patterns evolve between layers.

        Args:
            pattern1: Earlier layer pattern.
            pattern2: Later layer pattern.

        Returns:
            Pattern evolution metrics.
        """
        # Ensure same shape
        if pattern1.shape != pattern2.shape:
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(pattern1.shape, pattern2.shape))
            pattern1 = pattern1[:min_shape[0], :min_shape[1]] if pattern1.ndim >= 2 else pattern1[:min_shape[0]]
            pattern2 = pattern2[:min_shape[0], :min_shape[1]] if pattern2.ndim >= 2 else pattern2[:min_shape[0]]

        # Compute various evolution metrics
        diff = pattern2 - pattern1

        evolution = {
            'mean_change': float(np.mean(diff)),
            'abs_mean_change': float(np.mean(np.abs(diff))),
            'max_increase': float(np.max(diff)),
            'max_decrease': float(np.min(diff)),
            'pattern_sharpening': float(np.std(pattern2) - np.std(pattern1)),
            'entropy_change': float(
                self._compute_attention_entropy(pattern2.reshape(1, -1) if pattern2.ndim == 1 else pattern2)[0] -
                self._compute_attention_entropy(pattern1.reshape(1, -1) if pattern1.ndim == 1 else pattern1)[0]
            )
        }

        return evolution

    def rank_attention_heads(self, attention_weights: Dict[str, np.ndarray]) -> Dict[str, List[Dict[str, Any]]]:
        """Rank attention heads by importance across layers.

        Args:
            attention_weights: Dictionary mapping layer names to attention weight arrays.
                             Each array should have shape [batch, heads, seq_len, seq_len].

        Returns:
            Dictionary mapping layer names to ranked lists of head information.
        """
        try:
            head_rankings = {}

            for layer_name, weights in attention_weights.items():
                if weights is None or weights.size == 0:
                    continue

                # Ensure weights have correct shape
                if len(weights.shape) == 3:
                    weights = weights[np.newaxis, ...]  # Add batch dimension

                batch_size, n_heads, seq_len, _ = weights.shape
                layer_rankings = []

                for head_idx in range(n_heads):
                    head_weights = weights[:, head_idx, :, :]  # [batch, seq_len, seq_len]

                    # Compute importance metrics - average across batch
                    avg_head_weights = np.mean(head_weights, axis=0)  # [seq_len, seq_len]
                    importance_score = self._compute_single_head_importance(avg_head_weights)
                    entropy = np.mean([self._compute_attention_entropy(hw) for hw in head_weights])
                    sparsity = np.mean(head_weights == 0)
                    max_attention = np.max(head_weights)

                    head_info = {
                        'head_idx': head_idx,
                        'importance_score': float(importance_score),
                        'entropy': float(np.mean(entropy)) if hasattr(entropy, '__len__') else float(entropy),
                        'sparsity': float(sparsity),
                        'max_attention': float(max_attention),
                        'layer': layer_name
                    }

                    layer_rankings.append(head_info)

                # Sort by importance score (descending)
                layer_rankings.sort(key=lambda x: x['importance_score'], reverse=True)
                head_rankings[layer_name] = layer_rankings

            return head_rankings

        except Exception as e:
            self.logger.error(f"Head ranking failed: {e}")
            raise

    def prepare_attention_visualization(self, attention_matrix: np.ndarray,
                                      tokens: List[str]) -> Dict[str, Any]:
        """Prepare attention data for visualization.

        Args:
            attention_matrix: Attention weights [heads, seq_len, seq_len] or [seq_len, seq_len].
            tokens: List of tokens corresponding to sequence positions.

        Returns:
            Dictionary containing visualization-ready data.
        """
        try:
            # Handle different input shapes
            if len(attention_matrix.shape) == 2:
                # Single head: [seq_len, seq_len]
                attention_matrix = attention_matrix[np.newaxis, ...]  # Add head dimension

            n_heads, seq_len, _ = attention_matrix.shape

            # Ensure tokens match sequence length
            if len(tokens) != seq_len:
                tokens = tokens[:seq_len] + ['<PAD>'] * max(0, seq_len - len(tokens))

            viz_data = {
                'attention_matrix': attention_matrix.tolist(),
                'tokens': tokens,
                'head_data': [],
                'summary_stats': {},
                'seq_len': seq_len,
                'n_heads': n_heads
            }

            # Prepare per-head data
            for head_idx in range(n_heads):
                head_matrix = attention_matrix[head_idx]

                head_data = {
                    'head_idx': head_idx,
                    'attention_weights': head_matrix.tolist(),
                    'max_attention': float(np.max(head_matrix)),
                    'min_attention': float(np.min(head_matrix)),
                    'entropy': float(np.mean(self._compute_attention_entropy(head_matrix))),
                    'top_attended_tokens': self._get_top_attended_tokens(head_matrix, tokens),
                    'attention_patterns': self._classify_attention_pattern(head_matrix)
                }

                viz_data['head_data'].append(head_data)

            # Compute summary statistics
            viz_data['summary_stats'] = {
                'mean_attention': float(np.mean(attention_matrix)),
                'std_attention': float(np.std(attention_matrix)),
                'max_attention_global': float(np.max(attention_matrix)),
                'total_heads': n_heads,
                'sequence_length': seq_len
            }

            return viz_data

        except Exception as e:
            self.logger.error(f"Attention visualization preparation failed: {e}")
            raise

    def _compute_attention_distance(self, attention_weights: np.ndarray) -> Dict[str, Any]:
        """Compute attention distance statistics.

        Args:
            attention_weights: Attention weights [heads, seq_len, seq_len] or [seq_len, seq_len].

        Returns:
            Dictionary with distance-based attention statistics.
        """
        try:
            if attention_weights.ndim == 3:
                # Multi-head case: analyze each head and aggregate
                num_heads, seq_len, _ = attention_weights.shape
                head_distances = []

                for head_idx in range(num_heads):
                    head_dist = self._analyze_attention_distances(attention_weights[head_idx])
                    head_distances.append(head_dist)

                # Aggregate statistics
                local_means = [hd['local_attention_strength'] for hd in head_distances]
                global_means = [hd['global_attention_strength'] for hd in head_distances]

                return {
                    'head_distances': head_distances,
                    'average_local_attention': float(np.mean(local_means)),
                    'average_global_attention': float(np.mean(global_means)),
                    'local_attention_std': float(np.std(local_means)),
                    'global_attention_std': float(np.std(global_means))
                }
            else:
                # Single head case
                return self._analyze_attention_distances(attention_weights)

        except Exception as e:
            logger.error(f"Attention distance computation failed: {e}")
            return {
                'average_local_attention': 0.0,
                'average_global_attention': 0.0,
                'local_attention_std': 0.0,
                'global_attention_std': 0.0
            }

    def _compute_head_importance(self, attention_weights: np.ndarray) -> List[float]:
        """Compute importance scores for all attention heads.

        Args:
            attention_weights: Attention weights [heads, seq_len, seq_len].

        Returns:
            List of importance scores for each head.
        """
        try:
            if attention_weights.ndim != 3:
                # Single head case
                return [self._compute_single_head_importance(attention_weights)]

            num_heads = attention_weights.shape[0]
            importance_scores = []

            for head_idx in range(num_heads):
                head_weights = attention_weights[head_idx]
                importance = self._compute_single_head_importance(head_weights)
                importance_scores.append(importance)

            return importance_scores

        except Exception as e:
            logger.error(f"Head importance computation failed: {e}")
            return [0.0] * attention_weights.shape[0] if attention_weights.ndim == 3 else [0.0]

    def _compute_single_head_importance(self, head_weights: np.ndarray) -> float:
        """Compute importance score for an attention head.

        Args:
            head_weights: Attention weights for a single head [seq_len, seq_len].

        Returns:
            Importance score (higher = more important).
        """
        try:
            # Multiple metrics for head importance
            variance = np.var(head_weights)
            max_attention = np.max(head_weights)

            # Compute entropy for the head
            if head_weights.ndim == 3:
                # Batch case: average across batch
                entropy_values = [self._compute_attention_entropy(hw) for hw in head_weights]
                entropy = np.mean([np.mean(e) for e in entropy_values])
            else:
                # Single matrix case
                entropy = np.mean(self._compute_attention_entropy(head_weights))

            # Combine metrics (can be tuned)
            importance = variance * 0.4 + max_attention * 0.3 + (1.0 - entropy / 10.0) * 0.3

            return float(importance)

        except Exception as e:
            logger.error(f"Head importance computation failed: {e}")
            return 0.0

    def _get_top_attended_tokens(self, attention_matrix: np.ndarray,
                               tokens: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top attended token pairs for visualization.

        Args:
            attention_matrix: Attention weights [seq_len, seq_len].
            tokens: List of tokens.
            top_k: Number of top attention pairs to return.

        Returns:
            List of dictionaries with top attention pairs.
        """
        try:
            seq_len = attention_matrix.shape[0]
            top_pairs = []

            # Get all attention values with their positions
            for i in range(seq_len):
                for j in range(seq_len):
                    if i != j:  # Skip self-attention
                        attention_value = attention_matrix[i, j]
                        top_pairs.append({
                            'from_token': tokens[i] if i < len(tokens) else '<UNK>',
                            'to_token': tokens[j] if j < len(tokens) else '<UNK>',
                            'from_pos': i,
                            'to_pos': j,
                            'attention_weight': float(attention_value)
                        })

            # Sort by attention weight and return top k
            top_pairs.sort(key=lambda x: x['attention_weight'], reverse=True)
            return top_pairs[:top_k]

        except Exception as e:
            self.logger.error(f"Top attended tokens extraction failed: {e}")
            return []

    def _classify_attention_pattern(self, attention_matrix: np.ndarray) -> Dict[str, Any]:
        """Classify the type of attention pattern.

        Args:
            attention_matrix: Attention weights [seq_len, seq_len].

        Returns:
            Dictionary with pattern classification results.
        """
        try:
            seq_len = attention_matrix.shape[0]

            # Compute different pattern metrics
            diagonal_strength = np.trace(attention_matrix) / seq_len

            # Local attention (near diagonal)
            local_mask = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)) <= 2
            local_attention = np.sum(attention_matrix * local_mask) / np.sum(local_mask)

            # Global attention (far from diagonal)
            global_mask = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)) > 5
            global_attention = np.sum(attention_matrix * global_mask) / max(np.sum(global_mask), 1)

            # Classify pattern
            pattern_type = "uniform"
            if diagonal_strength > 0.3:
                pattern_type = "self-attention"
            elif local_attention > global_attention * 2:
                pattern_type = "local"
            elif global_attention > local_attention * 2:
                pattern_type = "global"

            return {
                'pattern_type': pattern_type,
                'diagonal_strength': float(diagonal_strength),
                'local_attention': float(local_attention),
                'global_attention': float(global_attention),
                'uniformity': float(1.0 - np.std(attention_matrix))
            }

        except Exception as e:
            self.logger.error(f"Attention pattern classification failed: {e}")
            return {'pattern_type': 'unknown', 'error': str(e)}

    def _extract_experiment_config(self, config_source: Any, config_name: str) -> Dict[str, Any]:
        """Return experiment configuration regardless of config backend type."""
        if hasattr(config_source, "get_experiment_config"):
            try:
                experiment_cfg = config_source.get_experiment_config(config_name)
                if isinstance(experiment_cfg, dict):
                    return experiment_cfg
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Config provider failed to return experiment '%s': %s", config_name, exc)

        if isinstance(config_source, dict):
            candidates = [config_source.get("experiments", {}), config_source]
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                if config_name in candidate and isinstance(candidate[config_name], dict):
                    return candidate[config_name]
                nested = candidate.get("experiments") if isinstance(candidate.get("experiments"), dict) else None
                if isinstance(nested, dict) and config_name in nested:
                    nested_cfg = nested[config_name]
                    if isinstance(nested_cfg, dict):
                        return nested_cfg

        try:
            from ..utils.config_manager import get_config_manager  # local import to avoid cycles

            manager = get_config_manager()
            if hasattr(manager, "get_config_model"):
                model = manager.get_config_model()
                if hasattr(model, "get_experiment_config"):
                    return model.get_experiment_config(config_name)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Fallback config manager lookup failed: %s", exc)

        return {}

def main():
    """Command line interface for attention analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Attention pattern analysis")
    parser.add_argument("--input", required=True, help="Path to attention data")
    parser.add_argument("--output", default="data/outputs/attention", help="Output directory")
    parser.add_argument("--layer", help="Specific attention layer to analyze")
    parser.add_argument("--config", default="default", help="Configuration name")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Implementation would depend on the specific data format
    logger.info("Attention analysis functionality available")


if __name__ == "__main__":
    main()
