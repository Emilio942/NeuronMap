"""
DeepSHAP Implementation for NeuronMap
===================================

Model-agnostic SHAP explanations for deep learning models.
Provides local explainable AI scores using Shapley values.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from pathlib import Path
import time

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP library not available, using fallback implementation")

from ...core.plugin_interface import AttributionPluginBase, ToolExecutionResult

logger = logging.getLogger(__name__)

class DeepSHAPExplainer(AttributionPluginBase):
    """
    DeepSHAP explainer for neural network interpretability.
    
    Implements DeepSHAP method using the SHAP library with
    fallback to custom implementation for basic cases.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(tool_id="deep_shap", config=config)
        
        self.version = "1.0.0"
        self.description = "DeepSHAP explainer for model-agnostic interpretability"
        
        # Configuration parameters
        self.num_background_samples = config.get('num_background_samples', 100) if config else 100
        self.batch_size = config.get('batch_size', 32) if config else 32
        self.use_shap_library = SHAP_AVAILABLE and config.get('use_shap_library', True) if config else SHAP_AVAILABLE
        
        # SHAP explainer instance
        self.explainer = None
        self.background_data = None
        
        logger.info(f"Initialized DeepSHAP explainer (SHAP library: {self.use_shap_library})")
    
    def initialize(self) -> bool:
        """Initialize the DeepSHAP explainer."""
        try:
            if self.use_shap_library and SHAP_AVAILABLE:
                logger.info("Using SHAP library for DeepSHAP implementation")
            else:
                logger.info("Using custom implementation for SHAP approximation")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DeepSHAP explainer: {e}")
            return False
    
    def execute(self, model: nn.Module, test_data: torch.Tensor,
                background_data: Optional[torch.Tensor] = None,
                target_class: Optional[int] = None,
                **kwargs) -> ToolExecutionResult:
        """
        Execute DeepSHAP analysis.
        
        Args:
            model: PyTorch model to analyze
            test_data: Test data to explain
            background_data: Background/reference data for SHAP baseline
            target_class: Target class for classification (optional)
            
        Returns:
            ToolExecutionResult with SHAP values and metadata
        """
        start_time = time.time()
        
        try:
            # Set model to evaluation mode
            model.eval()
            
            # Prepare background data
            if background_data is None:
                background_data = self._generate_background_data(test_data)
            
            # Store background data for later use
            self.background_data = background_data
            
            # Compute SHAP values
            if self.use_shap_library and SHAP_AVAILABLE:
                shap_values, expected_values = self._compute_shap_values(
                    model, test_data, background_data, target_class
                )
            else:
                shap_values, expected_values = self._compute_custom_shap_approximation(
                    model, test_data, background_data, target_class
                )
            
            # Compute additional metrics
            feature_importance = self._compute_feature_importance(shap_values)
            interaction_scores = self._compute_interaction_scores(shap_values)
            
            # Prepare outputs
            outputs = {
                'shap_values': shap_values,
                'expected_values': expected_values,
                'feature_importance': feature_importance,
                'interaction_scores': interaction_scores,
                'background_data': background_data,
                'method_config': {
                    'num_background_samples': self.num_background_samples,
                    'batch_size': self.batch_size,
                    'use_shap_library': self.use_shap_library
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
            logger.error(f"DeepSHAP execution failed: {e}")
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
    
    def _generate_background_data(self, test_data: torch.Tensor) -> torch.Tensor:
        """Generate background data if not provided."""
        # Simple approach: create random samples with same shape
        batch_size = min(self.num_background_samples, test_data.shape[0])
        
        if test_data.dtype in [torch.float32, torch.float64]:
            # For continuous data, use random samples from normal distribution
            background = torch.randn(batch_size, *test_data.shape[1:], 
                                   dtype=test_data.dtype, device=test_data.device)
        else:
            # For discrete data, use zeros
            background = torch.zeros(batch_size, *test_data.shape[1:], 
                                   dtype=test_data.dtype, device=test_data.device)
        
        return background
    
    def _compute_shap_values(self, model: nn.Module, test_data: torch.Tensor,
                            background_data: torch.Tensor, 
                            target_class: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute SHAP values using the SHAP library."""
        # Create model wrapper for SHAP
        def model_wrapper(x):
            with torch.no_grad():
                if isinstance(x, np.ndarray):
                    x = torch.tensor(x, dtype=test_data.dtype, device=test_data.device)
                outputs = model(x)
                return outputs.cpu().numpy()
        
        # Create DeepExplainer
        background_np = background_data.detach().cpu().numpy()
        explainer = shap.DeepExplainer(model_wrapper, background_np)
        
        # Compute SHAP values
        test_np = test_data.detach().cpu().numpy()
        shap_values = explainer.shap_values(test_np)
        expected_values = explainer.expected_value
        
        # Handle different output formats
        if isinstance(shap_values, list):
            if target_class is not None:
                shap_values = shap_values[target_class]
            else:
                # Take the first class or combine classes
                shap_values = shap_values[0]
        
        if isinstance(expected_values, list):
            if target_class is not None:
                expected_values = expected_values[target_class]
            else:
                expected_values = expected_values[0]
        
        return shap_values, expected_values
    
    def _compute_custom_shap_approximation(self, model: nn.Module, test_data: torch.Tensor,
                                          background_data: torch.Tensor,
                                          target_class: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Custom approximation of SHAP values using feature permutation."""
        logger.info("Using custom SHAP approximation (limited functionality)")
        
        # Simplified SHAP approximation using feature ablation
        with torch.no_grad():
            # Get baseline prediction (average over background)
            baseline_outputs = []
            for i in range(0, background_data.shape[0], self.batch_size):
                batch = background_data[i:i + self.batch_size]
                batch_outputs = model(batch)
                baseline_outputs.append(batch_outputs)
            
            baseline_output = torch.cat(baseline_outputs, dim=0).mean(dim=0)
            
            # Get original prediction
            original_output = model(test_data)
            
            # Simple feature attribution using gradients
            test_data.requires_grad_(True)
            if target_class is not None and original_output.dim() > 1:
                target_output = original_output[:, target_class].sum()
            else:
                target_output = original_output.sum()
            
            # Compute gradients
            gradients = torch.autograd.grad(target_output, test_data, 
                                          retain_graph=False, create_graph=False)[0]
            
            # Simple SHAP approximation: gradient * (input - baseline)
            baseline_expanded = baseline_output.unsqueeze(0).expand_as(test_data)
            if test_data.shape != baseline_expanded.shape:
                # Handle shape mismatch by taking mean across appropriate dimensions
                while baseline_expanded.dim() > test_data.dim():
                    baseline_expanded = baseline_expanded.mean(dim=0)
            
            shap_approximation = gradients * (test_data - baseline_expanded)
            
            shap_values = shap_approximation.detach().cpu().numpy()
            expected_values = baseline_output.cpu().numpy()
            
        return shap_values, expected_values
    
    def _compute_feature_importance(self, shap_values: np.ndarray) -> Dict[str, float]:
        """Compute feature importance scores from SHAP values."""
        # Global feature importance
        abs_shap = np.abs(shap_values)
        
        # Average importance across samples and aggregate over feature dimensions
        if abs_shap.ndim > 2:
            # For images/high-dimensional data, sum over spatial dimensions
            importance = abs_shap.mean(axis=0).sum(axis=tuple(range(1, abs_shap.ndim - 1)))
        else:
            # For 1D features
            importance = abs_shap.mean(axis=0)
        
        # Convert to dictionary
        feature_importance = {
            'mean_absolute_shap': float(abs_shap.mean()),
            'total_importance': float(importance.sum()),
            'max_importance': float(importance.max()),
            'std_importance': float(importance.std()),
            'feature_ranking': importance.argsort()[::-1].tolist()[:10]  # Top 10 features
        }
        
        return feature_importance
    
    def _compute_interaction_scores(self, shap_values: np.ndarray) -> Dict[str, Any]:
        """Compute basic interaction scores (simplified version)."""
        # Simplified interaction analysis
        # In a full implementation, you'd use SHAP interaction values
        
        interaction_strength = np.std(shap_values, axis=0)
        
        interaction_scores = {
            'interaction_strength': float(interaction_strength.mean()),
            'max_interaction': float(interaction_strength.max()),
            'min_interaction': float(interaction_strength.min())
        }
        
        return interaction_scores
    
    def compute_attributions(self, model: Any, inputs: Any, **kwargs) -> Dict[str, Any]:
        """Interface method for AttributionPluginBase."""
        # Convert inputs to test_data format
        if 'background_data' not in kwargs:
            kwargs['background_data'] = None
        
        result = self.execute(model, inputs, **kwargs)
        return result.outputs if result.success else {}
    
    def validate_output(self, output: Any) -> bool:
        """Validate that the output contains required SHAP data."""
        if not isinstance(output, dict):
            return False
        
        required_keys = ['shap_values', 'expected_values']
        for key in required_keys:
            if key not in output:
                logger.error(f"Missing required output key: {key}")
                return False
        
        # Check that SHAP values are valid
        shap_values = output['shap_values']
        if not isinstance(shap_values, np.ndarray):
            logger.error("SHAP values must be a numpy array")
            return False
        
        # Check for NaN or infinite values
        if np.isnan(shap_values).any() or np.isinf(shap_values).any():
            logger.error("SHAP values contain NaN or infinite values")
            return False
        
        # Check that expected values are valid
        expected_values = output['expected_values']
        if not isinstance(expected_values, (np.ndarray, float, int)):
            logger.error("Expected values must be numeric")
            return False
        
        return True
    
    def get_visualization_data(self, result: ToolExecutionResult) -> Dict[str, Any]:
        """Prepare data for visualization."""
        if not result.success:
            return {}
        
        outputs = result.outputs
        shap_values = outputs['shap_values']
        
        # Compute SHAP statistics
        shap_stats = {
            'mean': float(np.mean(shap_values)),
            'std': float(np.std(shap_values)),
            'min': float(np.min(shap_values)),
            'max': float(np.max(shap_values)),
            'positive_sum': float(np.sum(shap_values[shap_values > 0])),
            'negative_sum': float(np.sum(shap_values[shap_values < 0]))
        }
        
        return {
            'shap_values': shap_values,
            'expected_values': outputs['expected_values'],
            'feature_importance': outputs['feature_importance'],
            'shap_stats': shap_stats,
            'method_config': outputs.get('method_config')
        }

def create_shap_explainer(config: Optional[Dict[str, Any]] = None) -> DeepSHAPExplainer:
    """Factory function to create SHAP explainer."""
    return DeepSHAPExplainer(config=config)
