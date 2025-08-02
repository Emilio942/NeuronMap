"""
Integrated Gradients Implementation for NeuronMap
===============================================

Robust gradient-based attribution analysis with line integral computation
for PyTorch models. Provides post-hoc explanations for neural network decisions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from pathlib import Path
import time

try:
    from captum.attr import IntegratedGradients as CaptumIG
    from captum.attr import LayerIntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    logging.warning("Captum not available, using custom implementation")

from ...core.plugin_interface import AttributionPluginBase, ToolExecutionResult

logger = logging.getLogger(__name__)

class IntegratedGradientsExplainer(AttributionPluginBase):
    """
    Integrated Gradients explainer for neural network interpretability.
    
    Implements the Integrated Gradients method (Sundararajan et al., 2017)
    with both Captum integration and custom implementation fallback.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(tool_id="integrated_gradients", config=config)
        
        self.version = "1.0.0"
        self.description = "Integrated Gradients attribution method with PyTorch compatibility"
        
        # Configuration parameters
        self.n_steps = config.get('n_steps', 50) if config else 50
        self.method = config.get('method', 'gausslegendre') if config else 'gausslegendre'
        self.internal_batch_size = config.get('internal_batch_size', 1) if config else 1
        self.return_convergence_delta = config.get('return_convergence_delta', False) if config else False
        
        # Captum integration
        self.use_captum = CAPTUM_AVAILABLE and config.get('use_captum', True) if config else CAPTUM_AVAILABLE
        self.ig_explainer = None
        self.layer_ig_explainer = None
        
        logger.info(f"Initialized IG explainer (Captum: {self.use_captum})")
    
    def initialize(self) -> bool:
        """Initialize the Integrated Gradients explainer."""
        try:
            if self.use_captum and CAPTUM_AVAILABLE:
                logger.info("Using Captum implementation for Integrated Gradients")
            else:
                logger.info("Using custom implementation for Integrated Gradients")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize IG explainer: {e}")
            return False
    
    def execute(self, model: nn.Module, inputs: torch.Tensor, 
                target: Optional[Union[int, torch.Tensor]] = None,
                baseline: Optional[torch.Tensor] = None,
                target_layer: Optional[str] = None,
                **kwargs) -> ToolExecutionResult:
        """
        Execute Integrated Gradients analysis.
        
        Args:
            model: PyTorch model to analyze
            inputs: Input tensor to explain
            target: Target class index or tensor (for classification)
            baseline: Baseline tensor (default: zeros)
            target_layer: Specific layer to analyze (optional)
            
        Returns:
            ToolExecutionResult with attribution scores and metadata
        """
        start_time = time.time()
        
        try:
            # Set model to evaluation mode
            model.eval()
            
            # Prepare baseline
            if baseline is None:
                baseline = torch.zeros_like(inputs)
            
            # Prepare target
            if target is None and hasattr(model, 'num_classes'):
                # For classification, use predicted class as target
                with torch.no_grad():
                    outputs = model(inputs)
                    target = outputs.argmax(dim=-1)
            
            # Compute attributions
            if target_layer is not None:
                # Layer-specific attribution
                attributions = self._compute_layer_attributions(
                    model, inputs, target, baseline, target_layer
                )
            else:
                # Input-level attribution
                attributions = self._compute_input_attributions(
                    model, inputs, target, baseline
                )
            
            # Compute additional metrics
            convergence_delta = self._compute_convergence_delta(
                model, inputs, baseline, attributions, target
            ) if self.return_convergence_delta else None
            
            # Prepare outputs
            outputs = {
                'attributions': attributions,
                'baseline': baseline,
                'target': target,
                'convergence_delta': convergence_delta,
                'method_config': {
                    'n_steps': self.n_steps,
                    'method': self.method,
                    'use_captum': self.use_captum
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
            logger.error(f"IG execution failed: {e}")
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
    
    def _compute_input_attributions(self, model: nn.Module, inputs: torch.Tensor,
                                   target: Optional[torch.Tensor], 
                                   baseline: torch.Tensor) -> torch.Tensor:
        """Compute input-level attributions."""
        if self.use_captum and CAPTUM_AVAILABLE:
            return self._compute_captum_attributions(model, inputs, target, baseline)
        else:
            return self._compute_custom_attributions(model, inputs, target, baseline)
    
    def _compute_layer_attributions(self, model: nn.Module, inputs: torch.Tensor,
                                   target: Optional[torch.Tensor], baseline: torch.Tensor,
                                   target_layer: str) -> torch.Tensor:
        """Compute layer-specific attributions."""
        if self.use_captum and CAPTUM_AVAILABLE:
            return self._compute_captum_layer_attributions(
                model, inputs, target, baseline, target_layer
            )
        else:
            return self._compute_custom_layer_attributions(
                model, inputs, target, baseline, target_layer
            )
    
    def _compute_captum_attributions(self, model: nn.Module, inputs: torch.Tensor,
                                    target: Optional[torch.Tensor], 
                                    baseline: torch.Tensor) -> torch.Tensor:
        """Compute attributions using Captum library."""
        ig = CaptumIG(model)
        
        attributions = ig.attribute(
            inputs=inputs,
            baselines=baseline,
            target=target,
            n_steps=self.n_steps,
            method=self.method,
            internal_batch_size=self.internal_batch_size,
            return_convergence_delta=False
        )
        
        return attributions
    
    def _compute_captum_layer_attributions(self, model: nn.Module, inputs: torch.Tensor,
                                          target: Optional[torch.Tensor], 
                                          baseline: torch.Tensor,
                                          target_layer: str) -> torch.Tensor:
        """Compute layer attributions using Captum library."""
        # Get target layer
        layer = self._get_layer_by_name(model, target_layer)
        if layer is None:
            raise ValueError(f"Layer {target_layer} not found in model")
        
        layer_ig = LayerIntegratedGradients(model, layer)
        
        attributions = layer_ig.attribute(
            inputs=inputs,
            baselines=baseline,
            target=target,
            n_steps=self.n_steps,
            method=self.method,
            internal_batch_size=self.internal_batch_size
        )
        
        return attributions
    
    def _compute_custom_attributions(self, model: nn.Module, inputs: torch.Tensor,
                                    target: Optional[torch.Tensor], 
                                    baseline: torch.Tensor) -> torch.Tensor:
        """Custom implementation of Integrated Gradients."""
        # Enable gradients for inputs
        inputs.requires_grad_(True)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, self.n_steps, device=inputs.device)
        
        # Initialize gradients accumulator
        gradients = torch.zeros_like(inputs)
        
        for alpha in alphas:
            # Interpolated input
            interpolated = baseline + alpha * (inputs - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            outputs = model(interpolated)
            
            # Compute gradients
            if target is not None:
                # For classification
                if outputs.dim() > 1:
                    target_outputs = outputs.gather(1, target.unsqueeze(1)).squeeze()
                else:
                    target_outputs = outputs[target]
            else:
                # For regression or when no target specified
                target_outputs = outputs.sum()
            
            # Backward pass
            grad = torch.autograd.grad(target_outputs, interpolated, 
                                     retain_graph=False, create_graph=False)[0]
            
            gradients += grad
        
        # Compute integrated gradients
        integrated_gradients = gradients * (inputs - baseline) / self.n_steps
        
        return integrated_gradients
    
    def _compute_custom_layer_attributions(self, model: nn.Module, inputs: torch.Tensor,
                                          target: Optional[torch.Tensor], 
                                          baseline: torch.Tensor,
                                          target_layer: str) -> torch.Tensor:
        """Custom implementation of layer-specific Integrated Gradients."""
        # This is a simplified implementation
        # In practice, you'd need to register hooks to capture layer activations
        raise NotImplementedError("Custom layer attributions not implemented. Use Captum version.")
    
    def _get_layer_by_name(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Get layer by name from model."""
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        return None
    
    def _compute_convergence_delta(self, model: nn.Module, inputs: torch.Tensor,
                                  baseline: torch.Tensor, attributions: torch.Tensor,
                                  target: Optional[torch.Tensor]) -> float:
        """Compute convergence delta to validate attribution quality."""
        with torch.no_grad():
            # Original output
            orig_output = model(inputs)
            baseline_output = model(baseline)
            
            if target is not None:
                if orig_output.dim() > 1:
                    orig_score = orig_output.gather(1, target.unsqueeze(1)).squeeze()
                    baseline_score = baseline_output.gather(1, target.unsqueeze(1)).squeeze()
                else:
                    orig_score = orig_output[target]
                    baseline_score = baseline_output[target]
            else:
                orig_score = orig_output.sum()
                baseline_score = baseline_output.sum()
            
            # Difference between actual and attributed difference
            actual_diff = orig_score - baseline_score
            attributed_diff = attributions.sum()
            
            delta = float(torch.abs(actual_diff - attributed_diff))
            
        return delta
    
    def compute_attributions(self, model: Any, inputs: Any, **kwargs) -> Dict[str, Any]:
        """Interface method for AttributionPluginBase."""
        result = self.execute(model, inputs, **kwargs)
        return result.outputs if result.success else {}
    
    def validate_output(self, output: Any) -> bool:
        """Validate that the output contains required attribution data."""
        if not isinstance(output, dict):
            return False
        
        required_keys = ['attributions', 'baseline']
        for key in required_keys:
            if key not in output:
                logger.error(f"Missing required output key: {key}")
                return False
        
        # Check that attributions is a tensor
        attributions = output['attributions']
        if not torch.is_tensor(attributions):
            logger.error("Attributions must be a torch tensor")
            return False
        
        # Check that attributions are not all zeros
        if torch.all(attributions == 0):
            logger.warning("All attributions are zero - possible implementation issue")
        
        # Check for NaN or infinite values
        if torch.isnan(attributions).any() or torch.isinf(attributions).any():
            logger.error("Attributions contain NaN or infinite values")
            return False
        
        return True
    
    def get_visualization_data(self, result: ToolExecutionResult) -> Dict[str, Any]:
        """Prepare data for visualization."""
        if not result.success:
            return {}
        
        outputs = result.outputs
        attributions = outputs['attributions']
        
        # Compute attribution statistics
        attribution_stats = {
            'mean': float(attributions.mean()),
            'std': float(attributions.std()),
            'min': float(attributions.min()),
            'max': float(attributions.max()),
            'l2_norm': float(torch.norm(attributions)),
            'sparsity': float((attributions == 0).sum() / attributions.numel())
        }
        
        return {
            'attributions': attributions.detach().cpu().numpy(),
            'stats': attribution_stats,
            'convergence_delta': outputs.get('convergence_delta'),
            'method_config': outputs.get('method_config')
        }

def create_ig_explainer(config: Optional[Dict[str, Any]] = None) -> IntegratedGradientsExplainer:
    """Factory function to create IG explainer."""
    return IntegratedGradientsExplainer(config=config)
