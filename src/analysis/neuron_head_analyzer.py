"""
Neuron-to-Head Connection Analysis Module

This module implements analysis of the influence of MLP neurons on attention heads,
including gradient-based attribution and activation-based influence measurement.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from .circuits import (
    NeuralCircuit, CircuitComponent, CircuitConnection, ComponentType,
    create_mlp_neuron_component, create_attention_head_component
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
class NeuronHeadInfluence:
    """Result of neuron-to-head influence analysis."""
    source_layer: int
    source_neuron: int
    target_layer: int
    target_head: int
    influence_score: float
    gradient_attribution: float
    activation_correlation: float
    statistical_significance: float
    connection_strength: str  # 'strong', 'moderate', 'weak'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'source_layer': self.source_layer,
            'source_neuron': self.source_neuron,
            'target_layer': self.target_layer,
            'target_head': self.target_head,
            'influence_score': float(self.influence_score),
            'gradient_attribution': float(self.gradient_attribution),
            'activation_correlation': float(self.activation_correlation),
            'statistical_significance': float(self.statistical_significance),
            'connection_strength': self.connection_strength
        }


@dataclass
class NeuronHeadAnalysisResult:
    """Complete result of neuron-to-head analysis."""
    model_name: str
    analysis_type: str
    influences: List[NeuronHeadInfluence]
    summary_stats: Dict[str, float]
    circuit: Optional[NeuralCircuit] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'analysis_type': self.analysis_type,
            'influences': [inf.to_dict() for inf in self.influences],
            'summary_stats': self.summary_stats,
            'circuit': self.circuit.to_dict() if self.circuit else None
        }


class NeuronHeadAnalyzer:
    """Analyzes influence connections between MLP neurons and attention heads."""

    def __init__(self, model_manager: ModelManager, config: AnalysisConfig):
        self.model_manager = model_manager
        self.config = config
        self.device = next(model_manager.model.parameters()).device

    def analyze_neuron_head_influence(
        self,
        input_text: Union[str, List[str]],
        source_layers: Optional[List[int]] = None,
        target_layers: Optional[List[int]] = None,
        top_k_neurons: int = 50,
        top_k_heads: int = 20,
        influence_threshold: float = 0.1
    ) -> NeuronHeadAnalysisResult:
        """
        Analyze influence of MLP neurons on attention heads.

        Args:
            input_text: Text input(s) for analysis
            source_layers: MLP layers to analyze (default: all)
            target_layers: Attention layers to analyze (default: all)
            top_k_neurons: Number of top influential neurons per layer
            top_k_heads: Number of top influenced heads per layer
            influence_threshold: Minimum influence score to include

        Returns:
            Complete analysis results including circuits
        """
        logger.info("Starting neuron-to-head influence analysis")

        # Prepare inputs
        if isinstance(input_text, str):
            input_text = [input_text]

        # Set default layers
        num_layers = self.model_manager.get_model_info()['num_layers']
        if source_layers is None:
            source_layers = list(range(num_layers))
        if target_layers is None:
            target_layers = list(range(num_layers))

        influences = []

        for text in input_text:
            text_influences = self._analyze_single_text(
                text, source_layers, target_layers,
                top_k_neurons, top_k_heads, influence_threshold
            )
            influences.extend(text_influences)

        # Aggregate results across texts
        aggregated_influences = self._aggregate_influences(influences)

        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(aggregated_influences)

        # Build circuit from significant influences
        circuit = self._build_neuron_head_circuit(
            aggregated_influences, influence_threshold)

        result = NeuronHeadAnalysisResult(
            model_name=self.model_manager.model_name,
            analysis_type="neuron_head_influence",
            influences=aggregated_influences,
            summary_stats=summary_stats,
            circuit=circuit
        )

        logger.info(
            f"Analysis complete. Found {
                len(aggregated_influences)} significant connections")
        return result

    def _analyze_single_text(
        self,
        text: str,
        source_layers: List[int],
        target_layers: List[int],
        top_k_neurons: int,
        top_k_heads: int,
        influence_threshold: float
    ) -> List[NeuronHeadInfluence]:
        """Analyze neuron-head influence for a single text."""
        influences = []

        # Tokenize input
        inputs = self.model_manager.model.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        # Get activations with gradients
        activations = self._get_activations_with_gradients(
            inputs, source_layers, target_layers)

        # Calculate influence scores
        for source_layer in source_layers:
            for target_layer in target_layers:
                if target_layer <= source_layer:
                    continue  # Only analyze forward connections

                layer_influences = self._calculate_layer_influence(
                    activations, source_layer, target_layer,
                    top_k_neurons, top_k_heads, influence_threshold
                )
                influences.extend(layer_influences)

        return influences

    def _get_activations_with_gradients(
        self,
        inputs: Dict[str, torch.Tensor],
        source_layers: List[int],
        target_layers: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Get model activations with gradient computation enabled."""
        model = self.model_manager.model

        # Enable gradient computation
        model.train()
        activations = {}
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                activations[name] = output.clone()
                if output.requires_grad:
                    output.retain_grad()
            return hook

        # Register hooks for MLP and attention layers
        for layer_idx in source_layers:
            if hasattr(model, 'transformer'):
                # GPT-style model
                mlp_module = model.transformer.h[layer_idx].mlp
            elif hasattr(model, 'encoder'):
                # BERT/T5-style model
                mlp_module = model.encoder.layer[layer_idx].intermediate
            else:
                logger.warning(f"Unknown model architecture for layer {layer_idx}")
                continue

            hook = mlp_module.register_forward_hook(
                make_hook(f'mlp_{layer_idx}')
            )
            hooks.append(hook)

        for layer_idx in target_layers:
            if hasattr(model, 'transformer'):
                # GPT-style model
                attn_module = model.transformer.h[layer_idx].attn
            elif hasattr(model, 'encoder'):
                # BERT/T5-style model
                attn_module = model.encoder.layer[layer_idx].attention.self
            else:
                logger.warning(f"Unknown model architecture for layer {layer_idx}")
                continue

            hook = attn_module.register_forward_hook(
                make_hook(f'attn_{layer_idx}')
            )
            hooks.append(hook)

        try:
            # Forward pass
            outputs = model(**inputs)

            # Compute gradients w.r.t. final output
            if hasattr(outputs, 'logits'):
                loss = outputs.logits.sum()
            else:
                loss = outputs.last_hidden_state.sum()

            loss.backward()

        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()
            model.eval()

        return activations

    def _calculate_layer_influence(
        self,
        activations: Dict[str, torch.Tensor],
        source_layer: int,
        target_layer: int,
        top_k_neurons: int,
        top_k_heads: int,
        influence_threshold: float
    ) -> List[NeuronHeadInfluence]:
        """Calculate influence scores between a source MLP layer and target attention layer."""
        influences = []

        mlp_key = f'mlp_{source_layer}'
        attn_key = f'attn_{target_layer}'

        if mlp_key not in activations or attn_key not in activations:
            logger.warning(f"Missing activations for layers {
                           source_layer}->{target_layer}")
            return influences

        mlp_activations = activations[mlp_key]  # [batch, seq, hidden]
        attn_activations = activations[attn_key]  # [batch, seq, hidden]

        # Get gradients
        mlp_grads = mlp_activations.grad if mlp_activations.grad is not None else torch.zeros_like(
            mlp_activations)
        attn_grads = attn_activations.grad if attn_activations.grad is not None else torch.zeros_like(
            attn_activations)

        # Calculate per-neuron and per-head statistics
        batch_size, seq_len, hidden_size = mlp_activations.shape

        # Assume attention heads split the hidden dimension
        num_heads = self.model_manager.get_model_info().get('num_attention_heads', 12)
        head_dim = hidden_size // num_heads

        # Aggregate across batch and sequence dimensions
        mlp_neuron_activations = mlp_activations.mean(dim=(0, 1))  # [hidden]
        mlp_neuron_grads = mlp_grads.mean(dim=(0, 1))  # [hidden]

        attn_head_activations = attn_activations.view(
            batch_size,
            seq_len,
            num_heads,
            head_dim).mean(
            dim=(
                0,
                1,
                3))  # [num_heads]
        attn_head_grads = attn_grads.view(
            batch_size,
            seq_len,
            num_heads,
            head_dim).mean(
            dim=(
                0,
                1,
                3))  # [num_heads]

        # Calculate influence scores for top neurons and heads
        top_neuron_indices = torch.topk(
            torch.abs(mlp_neuron_grads),
            top_k_neurons).indices
        top_head_indices = torch.topk(torch.abs(attn_head_grads), top_k_heads).indices

        for neuron_idx in top_neuron_indices:
            for head_idx in top_head_indices:
                # Gradient-based attribution
                gradient_attribution = float(
                    mlp_neuron_grads[neuron_idx] *
                    attn_head_grads[head_idx])

                # Activation correlation (simplified)
                neuron_act = float(mlp_neuron_activations[neuron_idx])
                head_act = float(attn_head_activations[head_idx])
                activation_correlation = neuron_act * head_act

                # Combined influence score
                influence_score = abs(gradient_attribution) + \
                    0.1 * abs(activation_correlation)

                if influence_score >= influence_threshold:
                    # Statistical significance (placeholder - would need proper
                    # statistical testing)
                    statistical_significance = min(
                        influence_score / influence_threshold, 1.0)

                    # Connection strength classification
                    if influence_score > 0.5:
                        strength = 'strong'
                    elif influence_score > 0.2:
                        strength = 'moderate'
                    else:
                        strength = 'weak'

                    influence = NeuronHeadInfluence(
                        source_layer=source_layer,
                        source_neuron=int(neuron_idx),
                        target_layer=target_layer,
                        target_head=int(head_idx),
                        influence_score=influence_score,
                        gradient_attribution=gradient_attribution,
                        activation_correlation=activation_correlation,
                        statistical_significance=statistical_significance,
                        connection_strength=strength
                    )
                    influences.append(influence)

        return influences

    def _aggregate_influences(
            self,
            influences: List[NeuronHeadInfluence]) -> List[NeuronHeadInfluence]:
        """Aggregate influence results across multiple texts."""
        # Group by (source_layer, source_neuron, target_layer, target_head)
        influence_groups = {}

        for influence in influences:
            key = (influence.source_layer, influence.source_neuron,
                   influence.target_layer, influence.target_head)

            if key not in influence_groups:
                influence_groups[key] = []
            influence_groups[key].append(influence)

        # Aggregate each group
        aggregated = []
        for key, group in influence_groups.items():
            # Average the scores
            avg_influence = NeuronHeadInfluence(
                source_layer=key[0],
                source_neuron=key[1],
                target_layer=key[2],
                target_head=key[3],
                influence_score=np.mean([inf.influence_score for inf in group]),
                gradient_attribution=np.mean([inf.gradient_attribution for inf in group]),
                activation_correlation=np.mean([inf.activation_correlation for inf in group]),
                statistical_significance=np.mean([inf.statistical_significance for inf in group]),
                connection_strength=max(set([inf.connection_strength for inf in group]),
                                        key=lambda x: ['weak', 'moderate', 'strong'].index(x))
            )
            aggregated.append(avg_influence)

        # Sort by influence score
        aggregated.sort(key=lambda x: x.influence_score, reverse=True)
        return aggregated

    def _calculate_summary_stats(
            self, influences: List[NeuronHeadInfluence]) -> Dict[str, float]:
        """Calculate summary statistics for the analysis."""
        if not influences:
            return {}

        influence_scores = [inf.influence_score for inf in influences]
        gradient_attrs = [inf.gradient_attribution for inf in influences]

        stats = {
            'total_connections': len(influences),
            'mean_influence_score': float(np.mean(influence_scores)),
            'std_influence_score': float(np.std(influence_scores)),
            'max_influence_score': float(np.max(influence_scores)),
            'min_influence_score': float(np.min(influence_scores)),
            'mean_gradient_attribution': float(np.mean(gradient_attrs)),
            'strong_connections': sum(1 for inf in influences if inf.connection_strength == 'strong'),
            'moderate_connections': sum(1 for inf in influences if inf.connection_strength == 'moderate'),
            'weak_connections': sum(1 for inf in influences if inf.connection_strength == 'weak'),
        }

        return stats

    def _build_neuron_head_circuit(
        self,
        influences: List[NeuronHeadInfluence],
        influence_threshold: float
    ) -> NeuralCircuit:
        """Build a neural circuit from significant neuron-head influences."""
        circuit = NeuralCircuit(
            name=f"neuron_head_circuit_{self.model_manager.model_name}",
            description="Circuit showing MLP neuron to attention head influences",
            model_name=self.model_manager.model_name
        )

        # Add components and connections
        added_components = set()

        for influence in influences:
            if influence.influence_score < influence_threshold:
                continue

            # Add source neuron component
            source_id = f"mlp_{influence.source_layer}_{influence.source_neuron}"
            if source_id not in added_components:
                source_component = create_mlp_neuron_component(
                    influence.source_layer, influence.source_neuron
                )
                circuit.add_component(source_component)
                added_components.add(source_id)

            # Add target head component
            target_id = f"attn_{influence.target_layer}_{influence.target_head}"
            if target_id not in added_components:
                target_component = create_attention_head_component(
                    influence.target_layer, influence.target_head
                )
                circuit.add_component(target_component)
                added_components.add(target_id)

            # Add connection
            connection = CircuitConnection(
                source_id=source_id,
                target_id=target_id,
                weight=influence.influence_score,
                connection_type="neuron_to_head_influence",
                metadata={
                    'gradient_attribution': influence.gradient_attribution,
                    'activation_correlation': influence.activation_correlation,
                    'statistical_significance': influence.statistical_significance,
                    'connection_strength': influence.connection_strength
                }
            )
            circuit.add_connection(connection)

        return circuit

    def find_critical_neuron_head_paths(
        self,
        input_text: Union[str, List[str]],
        min_path_length: int = 2,
        max_path_length: int = 5,
        influence_threshold: float = 0.2
    ) -> List[List[str]]:
        """
        Find critical paths through neuron-head connections.

        Args:
            input_text: Input text for analysis
            min_path_length: Minimum path length to consider
            max_path_length: Maximum path length to consider
            influence_threshold: Minimum influence score for path inclusion

        Returns:
            List of paths, where each path is a list of component IDs
        """
        logger.info("Finding critical neuron-head paths")

        # Perform influence analysis
        result = self.analyze_neuron_head_influence(
            input_text, influence_threshold=influence_threshold
        )

        if not result.circuit:
            return []

        # Find paths in the circuit graph
        paths = result.circuit.find_paths(
            min_length=min_path_length,
            max_length=max_path_length
        )

        # Filter paths by cumulative influence score
        critical_paths = []
        for path in paths:
            path_score = self._calculate_path_influence_score(result.circuit, path)
            if path_score >= influence_threshold:
                critical_paths.append(path)

        # Sort by path influence score
        critical_paths.sort(
            key=lambda p: self._calculate_path_influence_score(result.circuit, p),
            reverse=True
        )

        logger.info(f"Found {len(critical_paths)} critical paths")
        return critical_paths

    def _calculate_path_influence_score(
            self,
            circuit: NeuralCircuit,
            path: List[str]) -> float:
        """Calculate cumulative influence score for a path."""
        if len(path) < 2:
            return 0.0

        total_score = 0.0
        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i + 1]

            # Find connection between these components
            for connection in circuit.connections:
                if connection.source_id == source_id and connection.target_id == target_id:
                    total_score += connection.weight
                    break

        return total_score / (len(path) - 1)  # Average influence score along path


# Utility functions for external usage
def analyze_neuron_head_influence(
    model_manager: ModelManager,
    config: AnalysisConfig,
    input_text: Union[str, List[str]],
    **kwargs
) -> NeuronHeadAnalysisResult:
    """Convenience function for neuron-head influence analysis."""
    analyzer = NeuronHeadAnalyzer(model_manager, config)
    return analyzer.analyze_neuron_head_influence(input_text, **kwargs)


def find_critical_neuron_head_paths(
    model_manager: ModelManager,
    config: AnalysisConfig,
    input_text: Union[str, List[str]],
    **kwargs
) -> List[List[str]]:
    """Convenience function for finding critical neuron-head paths."""
    analyzer = NeuronHeadAnalyzer(model_manager, config)
    return analyzer.find_critical_neuron_head_paths(input_text, **kwargs)
