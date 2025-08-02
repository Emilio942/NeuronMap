"""
Neuron Coverage Tracker for NeuronMap
====================================

Track active neurons per layer per input for comprehensive coverage analysis.
Helps understand which parts of the network are being utilized.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
import logging
from pathlib import Path
import time
from dataclasses import dataclass

# Minimal imports to avoid dependency issues
from ...core.plugin_interface import InterpretabilityPluginBase, ToolExecutionResult

logger = logging.getLogger(__name__)

@dataclass
class CoverageStats:
    """Statistics for neuron coverage analysis."""
    total_neurons: int
    active_neurons: int
    coverage_percentage: float
    activation_threshold: float
    layer_name: str

class NeuronCoverageTracker(InterpretabilityPluginBase):
    """
    Neuron Coverage Tracker for analyzing network utilization.
    
    Tracks which neurons are active across different inputs to understand
    how thoroughly the network is being tested and which parts remain unused.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(tool_id="neuron_coverage", config=config)
        
        self.version = "1.0.0"
        self.description = "Track active neurons per layer per input for coverage analysis"
        
        # Configuration parameters
        self.activation_threshold = config.get('activation_threshold', 0.1) if config else 0.1
        self.track_all_layers = config.get('track_all_layers', True) if config else True
        self.batch_size = config.get('batch_size', 32) if config else 32
        self.max_samples = config.get('max_samples', 1000) if config else 1000
        
        # Coverage tracking
        self.layer_activations = {}
        self.active_neurons = {}
        self.coverage_stats = {}
        self.activation_hooks = []
        
        logger.info(f"Initialized neuron coverage tracker (threshold: {self.activation_threshold})")
    
    def initialize(self) -> bool:
        """Initialize the neuron coverage tracker."""
        try:
            # Reset tracking state
            self.layer_activations = {}
            self.active_neurons = {}
            self.coverage_stats = {}
            self.activation_hooks = []
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize coverage tracker: {e}")
            return False
    
    def execute(self, model: nn.Module, inputs: torch.Tensor,
                layer_names: Optional[List[str]] = None,
                **kwargs) -> ToolExecutionResult:
        """
        Execute neuron coverage analysis.
        
        Args:
            model: PyTorch model to analyze
            inputs: Input data to test coverage with
            layer_names: Specific layers to track (optional, tracks all if None)
            
        Returns:
            ToolExecutionResult with coverage analysis
        """
        start_time = time.time()
        
        try:
            # Set model to evaluation mode
            model.eval()
            
            # Register hooks for activation tracking
            if layer_names is None:
                layer_names = self._get_all_layer_names(model) if self.track_all_layers else []
            
            self._register_hooks(model, layer_names)
            
            # Process inputs and track activations
            coverage_matrix, active_neuron_sets, total_neurons = self._process_inputs(
                model, inputs
            )
            
            # Compute coverage statistics
            coverage_stats = self._compute_coverage_stats(
                coverage_matrix, active_neuron_sets, total_neurons, layer_names
            )
            
            # Generate coverage report
            coverage_report = self._generate_coverage_report(coverage_stats)
            
            # Clean up hooks
            self._cleanup_hooks()
            
            # Prepare outputs
            outputs = {
                'coverage_matrix': coverage_matrix,
                'active_neurons': {k: list(v) for k, v in active_neuron_sets.items()},
                'coverage_stats': {k: self._serialize_coverage_stats(v) for k, v in coverage_stats.items()},
                'coverage_report': coverage_report,
                'total_neurons_tracked': sum(total_neurons.values()),
                'layers_analyzed': layer_names,
                'tracking_metadata': {
                    'activation_threshold': self.activation_threshold,
                    'samples_processed': inputs.shape[0] if hasattr(inputs, 'shape') else len(inputs),
                    'layers_tracked': len(layer_names)
                }
            }
            
            execution_time = time.time() - start_time
            
            return ToolExecutionResult(
                tool_id=self.tool_id,
                success=True,
                execution_time=execution_time,
                outputs=outputs,
                metadata=self.get_metadata(),
                errors=[],
                warnings=[],
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
        except Exception as e:
            logger.error(f"Coverage tracking execution failed: {e}")
            self._cleanup_hooks()
            execution_time = time.time() - start_time
            
            return ToolExecutionResult(
                tool_id=self.tool_id,
                success=False,
                execution_time=execution_time,
                outputs={},
                metadata=self.get_metadata(),
                errors=[str(e)],
                warnings=[],
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
    
    def _get_all_layer_names(self, model: nn.Module) -> List[str]:
        """Get names of all layers in the model."""
        layer_names = []
        for name, module in model.named_modules():
            # Focus on computational layers
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.LSTM, nn.GRU)):
                layer_names.append(name)
        return layer_names
    
    def _register_hooks(self, model: nn.Module, layer_names: List[str]):
        """Register forward hooks to capture activations."""
        def create_hook(layer_name):
            def hook_fn(module, input, output):
                # Store activations for coverage analysis
                if isinstance(output, torch.Tensor):
                    self.layer_activations[layer_name] = output.detach().cpu()
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    # Handle LSTM/GRU outputs (output, hidden)
                    self.layer_activations[layer_name] = output[0].detach().cpu()
            return hook_fn
        
        # Register hooks for specified layers
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(create_hook(name))
                self.activation_hooks.append(hook)
    
    def _process_inputs(self, model: nn.Module, inputs: torch.Tensor) -> Tuple[Dict, Dict, Dict]:
        """Process inputs and track neuron activations."""
        coverage_matrix = {}
        active_neuron_sets = {}
        total_neurons = {}
        
        # Initialize tracking structures
        for layer_name in self.layer_activations.keys():
            active_neuron_sets[layer_name] = set()
            coverage_matrix[layer_name] = []
        
        # Process inputs in batches
        num_samples = inputs.shape[0] if hasattr(inputs, 'shape') else len(inputs)
        num_batches = min((num_samples + self.batch_size - 1) // self.batch_size, 
                         (self.max_samples + self.batch_size - 1) // self.batch_size)
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, num_samples, self.max_samples)
                
                if hasattr(inputs, '__getitem__'):
                    batch_inputs = inputs[start_idx:end_idx]
                else:
                    batch_inputs = inputs
                
                # Clear previous activations
                self.layer_activations.clear()
                
                # Forward pass
                _ = model(batch_inputs)
                
                # Analyze activations for this batch
                for layer_name, activations in self.layer_activations.items():
                    # Flatten spatial dimensions but keep batch and feature dimensions
                    if activations.dim() > 2:
                        activations = activations.flatten(start_dim=2).mean(dim=2)
                    
                    # Find active neurons (above threshold)
                    active_mask = torch.abs(activations) > self.activation_threshold
                    
                    # Track which neurons are active across samples
                    for sample_idx in range(activations.shape[0]):
                        sample_active = torch.where(active_mask[sample_idx])[0]
                        active_neuron_sets[layer_name].update(sample_active.tolist())
                    
                    # Store batch coverage
                    batch_coverage = active_mask.float().mean(dim=0)  # Average across batch
                    coverage_matrix[layer_name].append(batch_coverage.numpy())
                    
                    # Track total neurons
                    total_neurons[layer_name] = activations.shape[-1]
        
        # Combine batch coverage matrices
        for layer_name in coverage_matrix:
            if coverage_matrix[layer_name]:
                coverage_matrix[layer_name] = np.mean(coverage_matrix[layer_name], axis=0)
            else:
                coverage_matrix[layer_name] = np.array([])
        
        return coverage_matrix, active_neuron_sets, total_neurons
    
    def _compute_coverage_stats(self, coverage_matrix: Dict, active_neuron_sets: Dict,
                               total_neurons: Dict, layer_names: List[str]) -> Dict[str, CoverageStats]:
        """Compute coverage statistics for each layer."""
        coverage_stats = {}
        
        for layer_name in layer_names:
            if layer_name in active_neuron_sets and layer_name in total_neurons:
                active_count = len(active_neuron_sets[layer_name])
                total_count = total_neurons[layer_name]
                coverage_pct = (active_count / total_count * 100) if total_count > 0 else 0
                
                stats = CoverageStats(
                    total_neurons=total_count,
                    active_neurons=active_count,
                    coverage_percentage=coverage_pct,
                    activation_threshold=self.activation_threshold,
                    layer_name=layer_name
                )
                
                coverage_stats[layer_name] = stats
        
        return coverage_stats
    
    def _generate_coverage_report(self, coverage_stats: Dict[str, CoverageStats]) -> Dict[str, Any]:
        """Generate a comprehensive coverage report."""
        if not coverage_stats:
            return {'error': 'No coverage statistics available'}
        
        # Overall statistics
        total_neurons_all = sum(stats.total_neurons for stats in coverage_stats.values())
        active_neurons_all = sum(stats.active_neurons for stats in coverage_stats.values())
        overall_coverage = (active_neurons_all / total_neurons_all * 100) if total_neurons_all > 0 else 0
        
        # Layer-wise analysis
        layer_coverages = [(name, stats.coverage_percentage) for name, stats in coverage_stats.items()]
        layer_coverages.sort(key=lambda x: x[1], reverse=True)
        
        # Coverage distribution
        coverage_values = [stats.coverage_percentage for stats in coverage_stats.values()]
        coverage_distribution = {
            'mean': float(np.mean(coverage_values)),
            'std': float(np.std(coverage_values)),
            'min': float(np.min(coverage_values)),
            'max': float(np.max(coverage_values)),
            'median': float(np.median(coverage_values))
        }
        
        # Coverage categories
        high_coverage_layers = [name for name, cov in layer_coverages if cov >= 80]
        medium_coverage_layers = [name for name, cov in layer_coverages if 50 <= cov < 80]
        low_coverage_layers = [name for name, cov in layer_coverages if cov < 50]
        
        report = {
            'overall_coverage_percentage': overall_coverage,
            'total_neurons_tracked': total_neurons_all,
            'active_neurons_tracked': active_neurons_all,
            'coverage_distribution': coverage_distribution,
            'layer_rankings': layer_coverages,
            'coverage_categories': {
                'high_coverage': high_coverage_layers,
                'medium_coverage': medium_coverage_layers,
                'low_coverage': low_coverage_layers
            },
            'recommendations': self._generate_recommendations(coverage_stats)
        }
        
        return report
    
    def _generate_recommendations(self, coverage_stats: Dict[str, CoverageStats]) -> List[str]:
        """Generate recommendations based on coverage analysis."""
        recommendations = []
        
        # Find layers with very low coverage
        low_coverage_layers = [name for name, stats in coverage_stats.items() 
                              if stats.coverage_percentage < 20]
        if low_coverage_layers:
            recommendations.append(
                f"Consider more diverse inputs - {len(low_coverage_layers)} layers have <20% coverage: "
                f"{', '.join(low_coverage_layers[:3])}{'...' if len(low_coverage_layers) > 3 else ''}"
            )
        
        # Find layers with very high coverage
        high_coverage_layers = [name for name, stats in coverage_stats.items() 
                               if stats.coverage_percentage > 95]
        if high_coverage_layers:
            recommendations.append(
                f"{len(high_coverage_layers)} layers have >95% coverage - "
                "consider if test set is sufficiently challenging"
            )
        
        # Check for imbalanced coverage
        coverage_values = [stats.coverage_percentage for stats in coverage_stats.values()]
        if np.std(coverage_values) > 30:
            recommendations.append(
                "High variance in layer coverage detected - "
                "some layers may be under-utilized"
            )
        
        # Overall coverage assessment
        overall_coverage = np.mean(coverage_values)
        if overall_coverage < 50:
            recommendations.append(
                "Low overall coverage - consider expanding test dataset or reducing activation threshold"
            )
        elif overall_coverage > 90:
            recommendations.append(
                "Very high overall coverage - test set appears comprehensive"
            )
        
        return recommendations
    
    def _serialize_coverage_stats(self, stats: CoverageStats) -> Dict[str, Any]:
        """Serialize CoverageStats to dictionary."""
        return {
            'total_neurons': stats.total_neurons,
            'active_neurons': stats.active_neurons,
            'coverage_percentage': stats.coverage_percentage,
            'activation_threshold': stats.activation_threshold,
            'layer_name': stats.layer_name
        }
    
    def _cleanup_hooks(self):
        """Remove all registered hooks."""
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks = []
        self.layer_activations = {}
    
    def validate_output(self, output: Any) -> bool:
        """Validate that the output contains required coverage data."""
        if not isinstance(output, dict):
            return False
        
        required_keys = ['coverage_matrix', 'active_neurons', 'coverage_stats']
        for key in required_keys:
            if key not in output:
                logger.error(f"Missing required output key: {key}")
                return False
        
        # Check that coverage stats are properly formatted
        coverage_stats = output['coverage_stats']
        if not isinstance(coverage_stats, dict):
            logger.error("Coverage stats must be a dictionary")
            return False
        
        # Check that each stat has required fields
        for layer_name, stats in coverage_stats.items():
            required_stat_keys = ['total_neurons', 'active_neurons', 'coverage_percentage']
            for key in required_stat_keys:
                if key not in stats:
                    logger.error(f"Missing required stat key {key} for layer {layer_name}")
                    return False
        
        return True
    
    def get_visualization_data(self, result: ToolExecutionResult) -> Dict[str, Any]:
        """Prepare data for visualization."""
        if not result.success:
            return {}
        
        outputs = result.outputs
        coverage_stats = outputs['coverage_stats']
        
        # Prepare data for plotting
        layer_names = list(coverage_stats.keys())
        coverage_percentages = [coverage_stats[name]['coverage_percentage'] for name in layer_names]
        active_neurons = [coverage_stats[name]['active_neurons'] for name in layer_names]
        total_neurons = [coverage_stats[name]['total_neurons'] for name in layer_names]
        
        return {
            'layer_names': layer_names,
            'coverage_percentages': coverage_percentages,
            'active_neurons': active_neurons,
            'total_neurons': total_neurons,
            'coverage_matrix': outputs['coverage_matrix'],
            'coverage_report': outputs['coverage_report']
        }

def create_coverage_tracker(config: Optional[Dict[str, Any]] = None) -> NeuronCoverageTracker:
    """Factory function to create neuron coverage tracker."""
    return NeuronCoverageTracker(config=config)
