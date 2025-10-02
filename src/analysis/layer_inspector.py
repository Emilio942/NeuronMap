"""
Layer-specific Analysis Functions for NeuronMap
==============================================

This module provides comprehensive layer-specific analysis capabilities for neural networks,
including layer activation patterns, gradient flows, and structural analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LayerType(Enum):
    """Types of neural network layers."""
    ATTENTION = "attention"
    MLP = "mlp"
    EMBEDDING = "embedding"
    NORMALIZATION = "normalization"
    OUTPUT = "output"
    UNKNOWN = "unknown"


@dataclass
class LayerInfo:
    """Information about a neural network layer."""
    name: str
    layer_type: LayerType
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    parameter_count: Optional[int] = None
    activation_function: Optional[str] = None
    position: Optional[int] = None  # Position in the network
    parent_module: Optional[str] = None


@dataclass
class LayerActivationStats:
    """Statistical information about layer activations."""
    mean: float
    std: float
    min_val: float
    max_val: float
    sparsity: float  # Percentage of zero/near-zero activations
    dead_neurons: int  # Number of consistently inactive neurons
    activation_magnitude: float  # Average magnitude across all activations


class LayerInspector:
    """
    Comprehensive layer analysis and inspection system.

    This class provides tools for:
    - Layer structure analysis
    - Activation pattern inspection
    - Gradient flow analysis
    - Layer importance ranking
    - Dead neuron detection
    """

    def __init__(self, model: torch.nn.Module, tokenizer=None):
        """
        Initialize the layer inspector.

        Args:
            model: The neural network model to inspect
            tokenizer: Optional tokenizer for text models
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer_info: Dict[str, LayerInfo] = {}
        self.activation_hooks: Dict[str, Any] = {}
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}

        # Analyze model structure
        self._analyze_model_structure()

    def _analyze_model_structure(self):
        """Analyze and categorize all layers in the model."""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_type = self._identify_layer_type(name, module)
                param_count = sum(p.numel() for p in module.parameters())

                self.layer_info[name] = LayerInfo(
                    name=name,
                    layer_type=layer_type,
                    parameter_count=param_count,
                    activation_function=self._get_activation_function(module)
                )

        logger.info(f"Analyzed {len(self.layer_info)} layers in the model")

    def _identify_layer_type(self, name: str, module: torch.nn.Module) -> LayerType:
        """Identify the type of a layer based on its name and module type."""
        name_lower = name.lower()
        module_type = type(module).__name__.lower()

        if 'attention' in name_lower or 'attn' in name_lower:
            return LayerType.ATTENTION
        elif 'mlp' in name_lower or 'ffn' in name_lower or 'feed_forward' in name_lower:
            return LayerType.MLP
        elif 'embed' in name_lower:
            return LayerType.EMBEDDING
        elif 'norm' in name_lower or 'layernorm' in module_type:
            return LayerType.NORMALIZATION
        elif 'output' in name_lower or 'classifier' in name_lower:
            return LayerType.OUTPUT
        else:
            return LayerType.UNKNOWN

    def _get_activation_function(self, module: torch.nn.Module) -> Optional[str]:
        """Get the activation function used by a module."""
        module_type = type(module).__name__

        activation_map = {
            'ReLU': 'relu',
            'GELU': 'gelu',
            'Tanh': 'tanh',
            'Sigmoid': 'sigmoid',
            'LeakyReLU': 'leaky_relu',
            'Swish': 'swish',
            'Mish': 'mish'
        }

        return activation_map.get(module_type, None)

    def get_layers_by_type(self, layer_type: LayerType) -> List[str]:
        """Get all layer names of a specific type."""
        return [name for name, info in self.layer_info.items()
                if info.layer_type == layer_type]

    def get_layer_hierarchy(self) -> Dict[str, List[str]]:
        """Get the hierarchical structure of layers."""
        hierarchy = {}
        for name in self.layer_info:
            parts = name.split('.')
            current = hierarchy
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = name
        return hierarchy

    def register_activation_hooks(self, layer_names: Optional[List[str]] = None):
        """Register forward hooks to capture activations."""
        if layer_names is None:
            layer_names = list(self.layer_info.keys())

        def make_hook(layer_name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[layer_name] = output.detach().clone()
                elif isinstance(output, tuple):
                    # For layers that return multiple outputs, take the first
                    self.activations[layer_name] = output[0].detach().clone()
            return hook

        for layer_name in layer_names:
            module = dict(self.model.named_modules())[layer_name]
            handle = module.register_forward_hook(make_hook(layer_name))
            self.activation_hooks[layer_name] = handle

        logger.info(f"Registered activation hooks for {len(layer_names)} layers")

    def register_gradient_hooks(self, layer_names: Optional[List[str]] = None):
        """Register backward hooks to capture gradients."""
        if layer_names is None:
            layer_names = list(self.layer_info.keys())

        def make_hook(layer_name):
            def hook(module, grad_input, grad_output):
                if grad_output and grad_output[0] is not None:
                    self.gradients[layer_name] = grad_output[0].detach().clone()
            return hook

        for layer_name in layer_names:
            module = dict(self.model.named_modules())[layer_name]
            handle = module.register_backward_hook(make_hook(layer_name))
            self.activation_hooks[f"{layer_name}_grad"] = handle

        logger.info(f"Registered gradient hooks for {len(layer_names)} layers")

    def clear_hooks(self):
        """Remove all registered hooks."""
        for handle in self.activation_hooks.values():
            handle.remove()
        self.activation_hooks.clear()
        logger.info("Cleared all activation hooks")

    def analyze_layer_activations(
            self, inputs: torch.Tensor) -> Dict[str, LayerActivationStats]:
        """
        Analyze activation patterns for all registered layers.

        Args:
            inputs: Input tensor to pass through the model

        Returns:
            Dictionary mapping layer names to activation statistics
        """
        self.activations.clear()

        # Forward pass to capture activations
        with torch.no_grad():
            self.model(inputs)

        stats = {}
        for layer_name, activations in self.activations.items():
            # Flatten activations for analysis
            flat_activations = activations.view(-1).cpu().numpy()

            # Calculate statistics
            mean_val = float(np.mean(flat_activations))
            std_val = float(np.std(flat_activations))
            min_val = float(np.min(flat_activations))
            max_val = float(np.max(flat_activations))

            # Calculate sparsity (percentage of near-zero activations)
            threshold = 0.01 * max(abs(min_val), abs(max_val))
            near_zero = np.abs(flat_activations) < threshold
            sparsity = float(np.mean(near_zero)) * 100

            # Calculate dead neurons (consistently zero across batch)
            if len(activations.shape) > 2:
                # For conv layers or attention, average over spatial/sequence dimensions
                neuron_activations = activations.mean(
                    dim=tuple(range(2, len(activations.shape))))
            else:
                neuron_activations = activations

            dead_neurons = int(
                torch.sum(
                    torch.all(
                        torch.abs(neuron_activations) < threshold,
                        dim=0)))

            # Calculate average activation magnitude
            activation_magnitude = float(torch.mean(torch.abs(activations)))

            stats[layer_name] = LayerActivationStats(
                mean=mean_val,
                std=std_val,
                min_val=min_val,
                max_val=max_val,
                sparsity=sparsity,
                dead_neurons=dead_neurons,
                activation_magnitude=activation_magnitude
            )

        return stats

    def find_critical_layers(self,
                             inputs: torch.Tensor,
                             metric: str = "gradient_norm") -> List[Tuple[str, float]]:
        """
        Find the most critical layers based on various metrics.

        Args:
            inputs: Input tensor for analysis
            metric: Metric to use for ranking ("gradient_norm", "activation_magnitude", "sparsity")

        Returns:
            List of (layer_name, importance_score) tuples, sorted by importance
        """
        if metric == "gradient_norm":
            # Requires gradients to be computed
            inputs.requires_grad_(True)
            outputs = self.model(inputs)
            loss = outputs.mean()  # Simple loss for gradient computation
            loss.backward()

            scores = []
            for layer_name, grad in self.gradients.items():
                if grad is not None:
                    grad_norm = float(torch.norm(grad))
                    scores.append((layer_name, grad_norm))

        elif metric == "activation_magnitude":
            stats = self.analyze_layer_activations(inputs)
            scores = [(name, stat.activation_magnitude) for name, stat in stats.items()]

        elif metric == "sparsity":
            stats = self.analyze_layer_activations(inputs)
            scores = [(name, 100 - stat.sparsity)
                      for name, stat in stats.items()]  # Higher score for less sparse

        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Sort by importance (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def compare_layer_activations(self,
                                  inputs1: torch.Tensor,
                                  inputs2: torch.Tensor) -> Dict[str, float]:
        """
        Compare activations between two different inputs.

        Args:
            inputs1: First input tensor
            inputs2: Second input tensor

        Returns:
            Dictionary mapping layer names to cosine similarity scores
        """
        # Get activations for first input
        self.activations.clear()
        with torch.no_grad():
            self.model(inputs1)
        activations1 = {k: v.clone() for k, v in self.activations.items()}

        # Get activations for second input
        self.activations.clear()
        with torch.no_grad():
            self.model(inputs2)
        activations2 = {k: v.clone() for k, v in self.activations.items()}

        # Calculate cosine similarity
        similarities = {}
        for layer_name in activations1:
            if layer_name in activations2:
                act1 = activations1[layer_name].view(-1)
                act2 = activations2[layer_name].view(-1)

                cosine_sim = torch.nn.functional.cosine_similarity(
                    act1.unsqueeze(0), act2.unsqueeze(0)
                )
                similarities[layer_name] = float(cosine_sim)

        return similarities

    def export_layer_analysis(self, output_path: str,
                              stats: Dict[str, LayerActivationStats]):
        """Export layer analysis results to JSON."""
        output_data = {
            "model_info": {
                "total_layers": len(self.layer_info),
                "layer_types": {lt.value: len(self.get_layers_by_type(lt))
                                for lt in LayerType}
            },
            "layer_details": {
                name: {
                    "type": info.layer_type.value,
                    "parameter_count": info.parameter_count,
                    "activation_function": info.activation_function
                }
                for name, info in self.layer_info.items()
            },
            "activation_stats": {
                name: {
                    "mean": stat.mean,
                    "std": stat.std,
                    "min": stat.min_val,
                    "max": stat.max_val,
                    "sparsity": stat.sparsity,
                    "dead_neurons": stat.dead_neurons,
                    "activation_magnitude": stat.activation_magnitude
                }
                for name, stat in stats.items()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Layer analysis exported to {output_path}")

    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        self.clear_hooks()


def analyze_model_layers(model: torch.nn.Module,
                         sample_input: torch.Tensor,
                         output_dir: str = "data/outputs") -> LayerInspector:
    """
    Convenience function to perform comprehensive layer analysis.

    Args:
        model: PyTorch model to analyze
        sample_input: Sample input tensor for the model
        output_dir: Directory to save analysis results

    Returns:
        Configured LayerInspector instance
    """
    inspector = LayerInspector(model)

    # Register hooks for all layers
    inspector.register_activation_hooks()
    inspector.register_gradient_hooks()

    # Analyze activations
    stats = inspector.analyze_layer_activations(sample_input)

    # Find critical layers
    critical_layers = inspector.find_critical_layers(sample_input)

    # Export results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    inspector.export_layer_analysis(
        str(output_path / "layer_analysis.json"),
        stats
    )

    # Export critical layers ranking
    with open(output_path / "critical_layers.json", 'w') as f:
        json.dump({
            "ranking": [{"layer": name, "score": score}
                        for name, score in critical_layers[:20]]  # Top 20
        }, f, indent=2)

    logger.info(f"Layer analysis complete. Results saved to {output_dir}")
    return inspector
