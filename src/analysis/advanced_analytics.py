"""
Advanced Analytics Module for NeuronMap
=======================================

This module provides advanced analytics capabilities including attention flow analysis,
gradient attribution, neuron importance, and cross-layer information flow.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


class AttentionFlowAnalyzer:
    """Analyzes attention flow patterns across layers."""

    def __init__(self, model_adapter, config):
        self.model_adapter = model_adapter
        self.config = config
        self.attention_cache = {}

    def extract_attention_patterns(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Extract attention patterns from the model."""
        attention_patterns = {}

        # Register hooks for attention layers
        hooks = []
        attention_layers = self._find_attention_layers()

        for layer_name, layer_module in attention_layers:
            hook_fn = self._create_attention_hook(layer_name)
            hook = layer_module.register_forward_hook(hook_fn)
            hooks.append(hook)

        try:
            for text in texts:
                inputs = self.model_adapter.prepare_inputs([text])

                with torch.no_grad():
                    self.model_adapter.model(**inputs, output_attentions=True)

                # Process attention patterns
                for layer_name, attention_tensor in self.attention_cache.items():
                    if layer_name not in attention_patterns:
                        attention_patterns[layer_name] = []
                    attention_patterns[layer_name].append(attention_tensor.numpy())

                self.attention_cache.clear()

        finally:
            for hook in hooks:
                hook.remove()

        # Stack patterns
        for layer_name in attention_patterns:
            attention_patterns[layer_name] = np.stack(attention_patterns[layer_name])

        return attention_patterns

    def _find_attention_layers(self) -> List[Tuple[str, nn.Module]]:
        """Find attention layers in the model."""
        attention_layers = []

        for name, module in self.model_adapter.model.named_modules():
            if any(attn_keyword in name.lower() for attn_keyword in ['attention', 'attn', 'self_attn']):
                if hasattr(module, 'forward'):
                    attention_layers.append((name, module))

        return attention_layers

    def _create_attention_hook(self, layer_name: str):
        """Create hook function for attention extraction."""
        def hook_fn(module, input_data, output_data):
            if isinstance(output_data, tuple) and len(output_data) > 1:
                # Extract attention weights if available
                attention_weights = output_data[1] if len(output_data) > 1 else output_data[0]
                self.attention_cache[layer_name] = attention_weights.detach().cpu()
            else:
                self.attention_cache[layer_name] = output_data.detach().cpu()

        return hook_fn

    def analyze_attention_flow(self, attention_patterns: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze attention flow across layers."""
        flow_analysis = {
            'layer_attention_means': {},
            'attention_entropy': {},
            'cross_layer_correlation': {},
            'attention_concentration': {}
        }

        for layer_name, patterns in attention_patterns.items():
            # Mean attention per layer
            flow_analysis['layer_attention_means'][layer_name] = patterns.mean(axis=(0, 1))

            # Attention entropy (measure of attention dispersion)
            entropy = -np.sum(patterns * np.log(patterns + 1e-10), axis=-1)
            flow_analysis['attention_entropy'][layer_name] = entropy.mean()

            # Attention concentration (how focused the attention is)
            concentration = np.max(patterns, axis=-1) - np.min(patterns, axis=-1)
            flow_analysis['attention_concentration'][layer_name] = concentration.mean()

        # Cross-layer correlation
        layer_names = list(attention_patterns.keys())
        for i, layer1 in enumerate(layer_names):
            for j, layer2 in enumerate(layer_names[i+1:], i+1):
                corr, _ = pearsonr(
                    attention_patterns[layer1].flatten(),
                    attention_patterns[layer2].flatten()
                )
                flow_analysis['cross_layer_correlation'][f"{layer1}_{layer2}"] = corr

        return flow_analysis


class GradientAttributionAnalyzer:
    """Analyzes gradient-based attributions for understanding neuron importance."""

    def __init__(self, model_adapter, config):
        self.model_adapter = model_adapter
        self.config = config

    def compute_gradient_attribution(self, texts: List[str], target_layers: List[str]) -> Dict[str, np.ndarray]:
        """Compute gradient-based attributions."""
        if not self.model_adapter.model.training:
            self.model_adapter.model.train()  # Enable gradients

        attributions = {}

        for text in texts:
            inputs = self.model_adapter.prepare_inputs([text])

            # Enable gradient computation
            for param in self.model_adapter.model.parameters():
                param.requires_grad_(True)

            # Forward pass
            outputs = self.model_adapter.model(**inputs)

            # Get logits or last hidden state
            if hasattr(outputs, 'logits'):
                target_output = outputs.logits
            else:
                target_output = outputs.last_hidden_state

            # Compute gradients w.r.t. target metric
            target_metric = target_output.norm()
            target_metric.backward()

            # Extract gradients for target layers
            for layer_name in target_layers:
                try:
                    layer_module = self._get_module_by_name(layer_name)
                    if layer_module is not None and hasattr(layer_module, 'weight'):
                        grad = layer_module.weight.grad
                        if grad is not None:
                            if layer_name not in attributions:
                                attributions[layer_name] = []
                            attributions[layer_name].append(grad.detach().cpu().numpy())
                except Exception as e:
                    logger.warning(f"Could not extract gradient for layer {layer_name}: {e}")

            # Clear gradients
            self.model_adapter.model.zero_grad()

        # Stack attributions
        for layer_name in attributions:
            attributions[layer_name] = np.stack(attributions[layer_name])

        # Switch back to eval mode
        self.model_adapter.model.eval()

        return attributions

    def _get_module_by_name(self, name: str) -> Optional[nn.Module]:
        """Get module by name."""
        try:
            module = self.model_adapter.model
            for attr in name.split('.'):
                module = getattr(module, attr)
            return module
        except AttributeError:
            return None

    def analyze_neuron_importance(self, attributions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze neuron importance based on gradients."""
        importance_analysis = {
            'neuron_importance_scores': {},
            'top_important_neurons': {},
            'importance_distribution': {}
        }

        for layer_name, grad_values in attributions.items():
            # Compute importance as magnitude of gradients
            importance_scores = np.abs(grad_values).mean(axis=0)
            importance_analysis['neuron_importance_scores'][layer_name] = importance_scores

            # Find top important neurons
            flat_importance = importance_scores.flatten()
            top_indices = np.argsort(flat_importance)[-10:]  # Top 10
            importance_analysis['top_important_neurons'][layer_name] = {
                'indices': top_indices.tolist(),
                'scores': flat_importance[top_indices].tolist()
            }

            # Importance distribution statistics
            importance_analysis['importance_distribution'][layer_name] = {
                'mean': float(flat_importance.mean()),
                'std': float(flat_importance.std()),
                'max': float(flat_importance.max()),
                'min': float(flat_importance.min())
            }

        return importance_analysis


class CrossLayerAnalyzer:
    """Analyzes cross-layer information flow and interactions."""

    def __init__(self, model_adapter, config):
        self.model_adapter = model_adapter
        self.config = config

    def _compute_similarity_matrix(self, activations: Dict[str, np.ndarray], layer_names: List[str]) -> np.ndarray:
        """Compute pairwise similarity matrix between layers."""
        similarity_matrix = np.zeros((len(layer_names), len(layer_names)))

        for i, layer1 in enumerate(layer_names):
            for j, layer2 in enumerate(layer_names):
                if i != j:
                    act1 = activations[layer1]
                    act2 = activations[layer2]

                    # Ensure compatible shapes
                    if act1.ndim > 1:
                        act1 = act1.flatten()
                    if act2.ndim > 1:
                        act2 = act2.flatten()

                    # Make sure both have the same length
                    min_len = min(len(act1), len(act2))
                    act1 = act1[:min_len]
                    act2 = act2[:min_len]

                    if len(act1) > 0 and len(act2) > 0:
                        # Compute different similarity metrics
                        try:
                            cosine_sim = 1 - cosine(act1, act2)
                            similarity_matrix[i, j] = cosine_sim
                        except Exception as e:
                            logger.warning(f"Could not compute similarity between {layer1} and {layer2}: {e}")
        return similarity_matrix

    def _identify_information_bottlenecks(self, activations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Identify information bottlenecks in the network."""
        bottlenecks = {}
        for layer_name, activation in activations.items():
            try:
                # Ensure 2D shape for PCA
                if activation.ndim == 1:
                    activation = activation.reshape(1, -1)
                elif activation.ndim > 2:
                    activation = activation.reshape(activation.shape[0], -1)

                if activation.shape[0] > 1 and activation.shape[1] > 1:
                    # Compute effective dimensionality using PCA
                    pca = PCA()
                    pca.fit(activation)

                    # Find number of components explaining 95% variance
                    cumvar = np.cumsum(pca.explained_variance_ratio_)
                    n_components_95 = np.where(cumvar >= 0.95)[0]
                    n_components_95 = n_components_95[0] + 1 if len(n_components_95) > 0 else activation.shape[1]

                    bottlenecks[layer_name] = {
                        'effective_dimensionality': int(n_components_95),
                        'total_dimensionality': int(activation.shape[-1]),
                        'compression_ratio': float(n_components_95 / activation.shape[-1])
                    }
                else:
                    bottlenecks[layer_name] = {
                        'effective_dimensionality': int(activation.shape[-1]),
                        'total_dimensionality': int(activation.shape[-1]),
                        'compression_ratio': 1.0
                    }
            except Exception as e:
                logger.warning(f"Could not analyze information bottleneck for {layer_name}: {e}")
                bottlenecks[layer_name] = {
                    'effective_dimensionality': 'unknown',
                    'total_dimensionality': 'unknown',
                    'compression_ratio': 'unknown'
                }
        return bottlenecks

    def analyze_information_flow(self, activations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze information flow between layers."""
        layer_names = sorted(activations.keys())
        similarity_matrix = self._compute_similarity_matrix(activations, layer_names)
        bottlenecks = self._identify_information_bottlenecks(activations)

        flow_analysis = {
            'layer_similarities': {},
            'information_bottlenecks': bottlenecks,
            'flow_patterns': {}
        }

        for i, layer1 in enumerate(layer_names):
            for j, layer2 in enumerate(layer_names):
                if i != j:
                    flow_analysis['layer_similarities'][f"{layer1}_{layer2}"] = {
                        'cosine': float(similarity_matrix[i, j]) if not np.isnan(similarity_matrix[i, j]) else 0.0
                    }

        # Analyze flow patterns
        flow_analysis['flow_patterns'] = {
            'similarity_matrix': similarity_matrix.tolist(),
            'layer_names': layer_names,
            'average_similarity': float(np.nanmean(similarity_matrix)) if similarity_matrix.size > 0 else 0.0,
            'max_similarity': float(np.nanmax(similarity_matrix)) if similarity_matrix.size > 0 else 0.0,
            'min_similarity': float(np.nanmin(similarity_matrix)) if similarity_matrix.size > 0 else 0.0
        }

        return flow_analysis

    def detect_representational_geometry(self, activations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Detect representational geometry patterns across layers."""
        geometry_analysis = {}

        for layer_name, activation in activations.items():
            try:
                # Ensure proper shape for analysis
                if activation.ndim == 1:
                    # Single sample - create a minimal analysis
                    geometry_analysis[layer_name] = {
                        'rsm_mean': 1.0,  # Self-correlation
                        'rsm_std': 0.0,
                        'eigenvalue_distribution': [1.0],
                        'participation_ratio': 1.0,
                        'effective_rank': 1.0,
                        'note': 'Single sample analysis'
                    }
                    continue

                # Reshape for analysis
                if activation.ndim > 2:
                    reshaped_activation = activation.reshape(activation.shape[0], -1)
                else:
                    reshaped_activation = activation

                if reshaped_activation.shape[0] < 2:
                    # Not enough samples for correlation analysis
                    geometry_analysis[layer_name] = {
                        'rsm_mean': 1.0,
                        'rsm_std': 0.0,
                        'eigenvalue_distribution': [1.0],
                        'participation_ratio': 1.0,
                        'effective_rank': 1.0,
                        'note': 'Insufficient samples for full analysis'
                    }
                    continue

                # Compute representational similarity matrix
                rsm = np.corrcoef(reshaped_activation)

                # Handle NaN values
                rsm = np.nan_to_num(rsm, nan=0.0, posinf=1.0, neginf=-1.0)

                # Eigenvalue analysis
                eigenvals, eigenvecs = np.linalg.eigh(rsm)
                eigenvals = eigenvals[::-1]  # Sort descending
                eigenvals = eigenvals[eigenvals > 0]  # Keep only positive

                if len(eigenvals) == 0:
                    eigenvals = np.array([1.0])

                # Compute participation ratio (measure of dimensionality)
                participation_ratio = (eigenvals.sum() ** 2) / (eigenvals ** 2).sum()

                geometry_analysis[layer_name] = {
                    'rsm_mean': float(np.nanmean(rsm)),
                    'rsm_std': float(np.nanstd(rsm)),
                    'eigenvalue_distribution': eigenvals[:10].tolist(),  # Top 10
                    'participation_ratio': float(participation_ratio),
                    'effective_rank': float(np.sum(eigenvals > 0.01 * eigenvals.max()))
                }

            except Exception as e:
                logger.warning(f"Could not analyze representational geometry for {layer_name}: {e}")
                geometry_analysis[layer_name] = {
                    'error': str(e),
                    'rsm_mean': 0.0,
                    'rsm_std': 0.0,
                    'eigenvalue_distribution': [0.0],
                    'participation_ratio': 0.0,
                    'effective_rank': 0.0
                }

        return geometry_analysis


class AdvancedAnalyticsEngine:
    """Main engine coordinating all advanced analytics."""

    def __init__(self, model_adapter, config):
        self.model_adapter = model_adapter
        self.config = config

        # Initialize analyzers
        self.attention_analyzer = AttentionFlowAnalyzer(model_adapter, config)
        self.gradient_analyzer = GradientAttributionAnalyzer(model_adapter, config)
        self.cross_layer_analyzer = CrossLayerAnalyzer(model_adapter, config)

    def run_comprehensive_analysis(self, texts: List[str], activations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Run comprehensive advanced analysis."""
        logger.info("Running comprehensive advanced analysis...")

        comprehensive_results = {
            'attention_flow': {},
            'gradient_attribution': {},
            'cross_layer_flow': {},
            'representational_geometry': {},
            'summary_statistics': {}
        }

        try:
            # Attention flow analysis
            logger.info("Analyzing attention flow patterns...")
            attention_patterns = self.attention_analyzer.extract_attention_patterns(texts)
            comprehensive_results['attention_flow'] = self.attention_analyzer.analyze_attention_flow(attention_patterns)

        except Exception as e:
            logger.warning(f"Attention flow analysis failed: {e}")
            comprehensive_results['attention_flow'] = {'error': str(e)}

        try:
            # Gradient attribution analysis
            logger.info("Computing gradient attributions...")
            target_layers = list(activations.keys())[:5]  # Limit to first 5 layers
            attributions = self.gradient_analyzer.compute_gradient_attribution(texts, target_layers)
            comprehensive_results['gradient_attribution'] = self.gradient_analyzer.analyze_neuron_importance(attributions)

        except Exception as e:
            logger.warning(f"Gradient attribution analysis failed: {e}")
            comprehensive_results['gradient_attribution'] = {'error': str(e)}

        try:
            # Cross-layer information flow
            logger.info("Analyzing cross-layer information flow...")
            comprehensive_results['cross_layer_flow'] = self.cross_layer_analyzer.analyze_information_flow(activations)

        except Exception as e:
            logger.warning(f"Cross-layer flow analysis failed: {e}")
            comprehensive_results['cross_layer_flow'] = {'error': str(e)}

        try:
            # Representational geometry
            logger.info("Detecting representational geometry...")
            comprehensive_results['representational_geometry'] = self.cross_layer_analyzer.detect_representational_geometry(activations)

        except Exception as e:
            logger.warning(f"Representational geometry analysis failed: {e}")
            comprehensive_results['representational_geometry'] = {'error': str(e)}

        # Summary statistics
        comprehensive_results['summary_statistics'] = self._compute_summary_statistics(comprehensive_results)

        logger.info("Comprehensive advanced analysis completed")
        return comprehensive_results

    def _compute_summary_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute summary statistics across all analyses."""
        summary = {
            'total_layers_analyzed': 0,
            'attention_layers_found': 0,
            'gradient_layers_analyzed': 0,
            'information_bottlenecks_detected': 0,
            'analysis_completeness': {}
        }

        # Count analysis completeness
        for analysis_type, analysis_results in results.items():
            if analysis_type == 'summary_statistics':
                continue

            if isinstance(analysis_results, dict) and 'error' not in analysis_results:
                summary['analysis_completeness'][analysis_type] = 'success'

                # Count specific metrics
                if analysis_type == 'attention_flow' and 'layer_attention_means' in analysis_results:
                    summary['attention_layers_found'] = len(analysis_results['layer_attention_means'])

                elif analysis_type == 'gradient_attribution' and 'neuron_importance_scores' in analysis_results:
                    summary['gradient_layers_analyzed'] = len(analysis_results['neuron_importance_scores'])

                elif analysis_type == 'cross_layer_flow' and 'information_bottlenecks' in analysis_results:
                    summary['information_bottlenecks_detected'] = len(analysis_results['information_bottlenecks'])
                    summary['total_layers_analyzed'] = len(analysis_results['information_bottlenecks'])
            else:
                summary['analysis_completeness'][analysis_type] = 'failed'

        return summary

    def save_advanced_results(self, results: Dict[str, Any], output_path: Path):
        """Save advanced analysis results."""
        output_path.mkdir(parents=True, exist_ok=True)

        # Save main results as JSON
        import json
        results_file = output_path / "advanced_analysis_results.json"

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Advanced analysis results saved to {results_file}")

        # Generate summary report
        self._generate_summary_report(results, output_path)

    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible types."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

    def _generate_summary_report(self, results: Dict[str, Any], output_path: Path):
        """Generate a human-readable summary report."""
        report_file = output_path / "advanced_analysis_summary.txt"

        with open(report_file, 'w') as f:
            f.write("Advanced Neural Network Analysis Report\n")
            f.write("=" * 45 + "\n\n")

            # Summary statistics
            if 'summary_statistics' in results:
                stats = results['summary_statistics']
                f.write("Analysis Overview:\n")
                f.write(f"- Total layers analyzed: {stats.get('total_layers_analyzed', 'N/A')}\n")
                f.write(f"- Attention layers found: {stats.get('attention_layers_found', 'N/A')}\n")
                f.write(f"- Gradient layers analyzed: {stats.get('gradient_layers_analyzed', 'N/A')}\n")
                f.write(f"- Information bottlenecks detected: {stats.get('information_bottlenecks_detected', 'N/A')}\n\n")

            # Attention flow summary
            if 'attention_flow' in results and 'error' not in results['attention_flow']:
                f.write("Attention Flow Analysis:\n")
                flow_data = results['attention_flow']
                if 'attention_entropy' in flow_data:
                    f.write("- Attention entropy by layer:\n")
                    for layer, entropy in flow_data['attention_entropy'].items():
                        f.write(f"  * {layer}: {entropy:.4f}\n")
                f.write("\n")

            # Cross-layer analysis summary
            if 'cross_layer_flow' in results and 'error' not in results['cross_layer_flow']:
                f.write("Cross-Layer Information Flow:\n")
                flow_data = results['cross_layer_flow']
                if 'flow_patterns' in flow_data:
                    patterns = flow_data['flow_patterns']
                    f.write(f"- Average layer similarity: {patterns.get('average_similarity', 'N/A'):.4f}\n")
                    f.write(f"- Maximum layer similarity: {patterns.get('max_similarity', 'N/A'):.4f}\n")
                f.write("\n")

            # Representational geometry summary
            if 'representational_geometry' in results and 'error' not in results['representational_geometry']:
                f.write("Representational Geometry:\n")
                geom_data = results['representational_geometry']
                f.write("- Participation ratios by layer:\n")
                for layer, data in geom_data.items():
                    if isinstance(data, dict) and 'participation_ratio' in data:
                        f.write(f"  * {layer}: {data['participation_ratio']:.4f}\n")
                f.write("\n")

        logger.info(f"Advanced analysis summary report saved to {report_file}")
