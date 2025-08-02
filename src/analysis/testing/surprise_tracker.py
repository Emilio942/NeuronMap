"""
Surprise Coverage Tracker for NeuronMap
======================================

Compare activations to baseline distribution for surprise detection.
Identifies unusual activation patterns that deviate from expected behavior.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import time
from dataclasses import dataclass

# Scipy imports with fallback
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available, using simplified statistical tests")

from ...core.plugin_interface import InterpretabilityPluginBase, ToolExecutionResult

logger = logging.getLogger(__name__)

@dataclass
class SurpriseMetrics:
    """Metrics for surprise analysis."""
    layer_name: str
    surprise_score: float
    baseline_mean: float
    baseline_std: float
    test_mean: float
    test_std: float
    divergence_score: float
    num_outliers: int

class SurpriseCoverageTracker(InterpretabilityPluginBase):
    """
    Surprise Coverage Tracker for detecting unusual activation patterns.
    
    Compares test activations against a baseline distribution to identify
    surprising or unexpected neural behaviors.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(tool_id="surprise_coverage", config=config)
        
        self.version = "1.0.0"
        self.description = "Compare activations to baseline distribution for surprise detection"
        
        # Configuration parameters
        self.surprise_threshold = config.get('surprise_threshold', 2.0) if config else 2.0  # std deviations
        self.outlier_threshold = config.get('outlier_threshold', 3.0) if config else 3.0
        self.min_baseline_samples = config.get('min_baseline_samples', 100) if config else 100
        self.use_kl_divergence = config.get('use_kl_divergence', True) if config else True
        
        # Baseline storage
        self.baseline_distributions = {}
        self.baseline_computed = False
        
        logger.info(f"Initialized surprise coverage tracker (threshold: {self.surprise_threshold})")
    
    def initialize(self) -> bool:
        """Initialize the surprise coverage tracker."""
        try:
            # Reset baseline state
            self.baseline_distributions = {}
            self.baseline_computed = False
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize surprise tracker: {e}")
            return False
    
    def execute(self, model: nn.Module, 
                baseline_distribution: Optional[Dict[str, torch.Tensor]] = None,
                test_inputs: Optional[torch.Tensor] = None,
                **kwargs) -> ToolExecutionResult:
        """
        Execute surprise coverage analysis.
        
        Args:
            model: PyTorch model to analyze
            baseline_distribution: Pre-computed baseline activations or data to compute baseline
            test_inputs: Test inputs to compare against baseline
            
        Returns:
            ToolExecutionResult with surprise analysis
        """
        start_time = time.time()
        
        try:
            # Set model to evaluation mode
            model.eval()
            
            # Prepare baseline distribution
            if baseline_distribution is None:
                if test_inputs is not None:
                    # Use first part of test inputs as baseline
                    split_point = max(1, test_inputs.shape[0] // 2)
                    baseline_inputs = test_inputs[:split_point]
                    test_inputs = test_inputs[split_point:]
                    baseline_distribution = self._extract_baseline_activations(model, baseline_inputs)
                else:
                    raise ValueError("Either baseline_distribution or test_inputs must be provided")
            
            # Ensure we have test inputs
            if test_inputs is None:
                raise ValueError("test_inputs must be provided for comparison")
            
            # Compute baseline statistics
            baseline_stats = self._compute_baseline_statistics(baseline_distribution)
            
            # Extract test activations
            test_activations = self._extract_test_activations(model, test_inputs)
            
            # Compute surprise scores
            surprise_metrics = self._compute_surprise_metrics(baseline_stats, test_activations)
            
            # Detect outliers
            outlier_analysis = self._detect_outliers(baseline_stats, test_activations)
            
            # Generate surprise report
            surprise_report = self._generate_surprise_report(surprise_metrics, outlier_analysis)
            
            # Prepare outputs
            outputs = {
                'surprise_scores': {k: v.surprise_score for k, v in surprise_metrics.items()},
                'distribution_comparison': {k: self._serialize_surprise_metrics(v) 
                                          for k, v in surprise_metrics.items()},
                'outlier_analysis': outlier_analysis,
                'surprise_report': surprise_report,
                'baseline_statistics': {k: self._serialize_baseline_stats(v) 
                                      for k, v in baseline_stats.items()},
                'analysis_metadata': {
                    'surprise_threshold': self.surprise_threshold,
                    'outlier_threshold': self.outlier_threshold,
                    'baseline_samples': sum(len(v['activations']) for v in baseline_stats.values()),
                    'test_samples': test_inputs.shape[0] if hasattr(test_inputs, 'shape') else len(test_inputs)
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
            logger.error(f"Surprise tracking execution failed: {e}")
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
    
    def _extract_baseline_activations(self, model: nn.Module, 
                                     baseline_inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract activations from baseline inputs."""
        activations = {}
        hooks = []
        
        def create_hook(layer_name):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    # Store flattened activations
                    flat_output = output.flatten(start_dim=1) if output.dim() > 2 else output
                    activations[layer_name] = flat_output.detach().cpu()
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    flat_output = output[0].flatten(start_dim=1) if output[0].dim() > 2 else output[0]
                    activations[layer_name] = flat_output.detach().cpu()
            return hook_fn
        
        # Register hooks for key layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                hook = module.register_forward_hook(create_hook(name))
                hooks.append(hook)
        
        try:
            with torch.no_grad():
                _ = model(baseline_inputs)
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()
        
        return activations
    
    def _extract_test_activations(self, model: nn.Module, 
                                 test_inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract activations from test inputs."""
        return self._extract_baseline_activations(model, test_inputs)
    
    def _compute_baseline_statistics(self, baseline_activations: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
        """Compute statistical properties of baseline activations."""
        baseline_stats = {}
        
        for layer_name, activations in baseline_activations.items():
            if len(activations) < self.min_baseline_samples:
                logger.warning(f"Layer {layer_name} has only {len(activations)} baseline samples "
                             f"(minimum recommended: {self.min_baseline_samples})")
            
            # Compute basic statistics
            mean = activations.mean(dim=0)
            std = activations.std(dim=0)
            median = activations.median(dim=0)[0]
            
            # Compute percentiles
            q25 = torch.quantile(activations, 0.25, dim=0)
            q75 = torch.quantile(activations, 0.75, dim=0)
            
            baseline_stats[layer_name] = {
                'activations': activations,
                'mean': mean,
                'std': std,
                'median': median,
                'q25': q25,
                'q75': q75,
                'min': activations.min(dim=0)[0],
                'max': activations.max(dim=0)[0],
                'num_samples': activations.shape[0]
            }
        
        return baseline_stats
    
    def _compute_surprise_metrics(self, baseline_stats: Dict[str, Dict[str, Any]],
                                 test_activations: Dict[str, torch.Tensor]) -> Dict[str, SurpriseMetrics]:
        """Compute surprise metrics for each layer."""
        surprise_metrics = {}
        
        for layer_name in baseline_stats.keys():
            if layer_name not in test_activations:
                continue
            
            baseline = baseline_stats[layer_name]
            test_acts = test_activations[layer_name]
            
            # Compute test statistics
            test_mean = test_acts.mean(dim=0)
            test_std = test_acts.std(dim=0)
            
            # Compute surprise score (normalized difference)
            mean_diff = torch.abs(test_mean - baseline['mean'])
            baseline_std_safe = torch.clamp(baseline['std'], min=1e-8)  # Avoid division by zero
            surprise_score = (mean_diff / baseline_std_safe).mean().item()
            
            # Compute KL divergence if possible
            if self.use_kl_divergence and SCIPY_AVAILABLE:
                try:
                    divergence_score = self._compute_kl_divergence(
                        baseline['activations'], test_acts
                    )
                except Exception:
                    divergence_score = 0.0
            else:
                # Simplified divergence using variance ratio
                var_ratio = (test_std.mean() / baseline_std_safe.mean()).item()
                divergence_score = abs(np.log(var_ratio)) if var_ratio > 0 else 0.0
            
            # Count outliers
            num_outliers = self._count_outliers(baseline, test_acts)
            
            metrics = SurpriseMetrics(
                layer_name=layer_name,
                surprise_score=surprise_score,
                baseline_mean=baseline['mean'].mean().item(),
                baseline_std=baseline['std'].mean().item(),
                test_mean=test_mean.mean().item(),
                test_std=test_std.mean().item(),
                divergence_score=divergence_score,
                num_outliers=num_outliers
            )
            
            surprise_metrics[layer_name] = metrics
        
        return surprise_metrics
    
    def _compute_kl_divergence(self, baseline_acts: torch.Tensor, 
                              test_acts: torch.Tensor) -> float:
        """Compute KL divergence between baseline and test distributions."""
        # Simple histogram-based KL divergence
        try:
            # Flatten and convert to numpy
            baseline_flat = baseline_acts.flatten().numpy()
            test_flat = test_acts.flatten().numpy()
            
            # Create histograms
            bins = 50
            hist_range = (min(baseline_flat.min(), test_flat.min()),
                         max(baseline_flat.max(), test_flat.max()))
            
            baseline_hist, _ = np.histogram(baseline_flat, bins=bins, range=hist_range, density=True)
            test_hist, _ = np.histogram(test_flat, bins=bins, range=hist_range, density=True)
            
            # Add small epsilon to avoid log(0)
            eps = 1e-8
            baseline_hist = baseline_hist + eps
            test_hist = test_hist + eps
            
            # Normalize
            baseline_hist = baseline_hist / baseline_hist.sum()
            test_hist = test_hist / test_hist.sum()
            
            # Compute KL divergence
            kl_div = np.sum(test_hist * np.log(test_hist / baseline_hist))
            return float(kl_div)
            
        except Exception as e:
            logger.warning(f"KL divergence computation failed: {e}")
            return 0.0
    
    def _count_outliers(self, baseline_stats: Dict[str, Any], 
                       test_activations: torch.Tensor) -> int:
        """Count outliers in test activations compared to baseline."""
        baseline_mean = baseline_stats['mean']
        baseline_std = baseline_stats['std']
        
        # Z-score based outlier detection
        z_scores = torch.abs((test_activations - baseline_mean) / torch.clamp(baseline_std, min=1e-8))
        outliers = (z_scores > self.outlier_threshold).any(dim=1)
        
        return int(outliers.sum().item())
    
    def _detect_outliers(self, baseline_stats: Dict[str, Dict[str, Any]],
                        test_activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Detailed outlier analysis."""
        outlier_analysis = {}
        
        for layer_name in baseline_stats.keys():
            if layer_name not in test_activations:
                continue
            
            baseline = baseline_stats[layer_name]
            test_acts = test_activations[layer_name]
            
            # Multiple outlier detection methods
            outlier_methods = {}
            
            # Z-score method
            z_scores = torch.abs((test_acts - baseline['mean']) / 
                               torch.clamp(baseline['std'], min=1e-8))
            z_outliers = (z_scores > self.outlier_threshold).any(dim=1)
            outlier_methods['z_score'] = int(z_outliers.sum().item())
            
            # IQR method
            iqr = baseline['q75'] - baseline['q25']
            lower_bound = baseline['q25'] - 1.5 * iqr
            upper_bound = baseline['q75'] + 1.5 * iqr
            
            iqr_outliers = ((test_acts < lower_bound) | (test_acts > upper_bound)).any(dim=1)
            outlier_methods['iqr'] = int(iqr_outliers.sum().item())
            
            # Magnitude-based (unusually high activations)
            magnitude_threshold = baseline['mean'] + 3 * baseline['std']
            magnitude_outliers = (test_acts > magnitude_threshold).any(dim=1)
            outlier_methods['magnitude'] = int(magnitude_outliers.sum().item())
            
            outlier_analysis[layer_name] = {
                'outlier_counts': outlier_methods,
                'total_test_samples': test_acts.shape[0],
                'most_extreme_indices': self._find_most_extreme_samples(baseline, test_acts)
            }
        
        return outlier_analysis
    
    def _find_most_extreme_samples(self, baseline_stats: Dict[str, Any],
                                  test_activations: torch.Tensor, top_k: int = 5) -> List[int]:
        """Find indices of most extreme test samples."""
        baseline_mean = baseline_stats['mean']
        baseline_std = baseline_stats['std']
        
        # Compute overall deviation for each sample
        deviations = torch.abs((test_activations - baseline_mean) / 
                              torch.clamp(baseline_std, min=1e-8))
        sample_deviations = deviations.mean(dim=1)
        
        # Get top k most extreme samples
        _, extreme_indices = torch.topk(sample_deviations, min(top_k, len(sample_deviations)))
        
        return extreme_indices.tolist()
    
    def _generate_surprise_report(self, surprise_metrics: Dict[str, SurpriseMetrics],
                                 outlier_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive surprise analysis report."""
        if not surprise_metrics:
            return {'error': 'No surprise metrics available'}
        
        # Overall surprise assessment
        surprise_scores = [m.surprise_score for m in surprise_metrics.values()]
        overall_surprise = np.mean(surprise_scores)
        
        # Layer rankings by surprise
        layer_rankings = [(name, metrics.surprise_score) 
                         for name, metrics in surprise_metrics.items()]
        layer_rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Surprise categories
        high_surprise_layers = [name for name, score in layer_rankings if score > self.surprise_threshold]
        medium_surprise_layers = [name for name, score in layer_rankings 
                                if self.surprise_threshold * 0.5 <= score <= self.surprise_threshold]
        low_surprise_layers = [name for name, score in layer_rankings 
                             if score < self.surprise_threshold * 0.5]
        
        # Outlier summary
        total_outliers = sum(max(analysis['outlier_counts'].values()) 
                           for analysis in outlier_analysis.values())
        total_test_samples = sum(analysis['total_test_samples'] 
                               for analysis in outlier_analysis.values())
        outlier_rate = (total_outliers / total_test_samples * 100) if total_test_samples > 0 else 0
        
        report = {
            'overall_surprise_score': overall_surprise,
            'surprise_assessment': self._assess_surprise_level(overall_surprise),
            'layer_rankings': layer_rankings,
            'surprise_categories': {
                'high_surprise': high_surprise_layers,
                'medium_surprise': medium_surprise_layers,
                'low_surprise': low_surprise_layers
            },
            'outlier_summary': {
                'total_outliers': total_outliers,
                'total_samples': total_test_samples,
                'outlier_rate_percent': outlier_rate
            },
            'recommendations': self._generate_surprise_recommendations(
                surprise_metrics, outlier_analysis, overall_surprise
            )
        }
        
        return report
    
    def _assess_surprise_level(self, overall_surprise: float) -> str:
        """Assess the overall level of surprise."""
        if overall_surprise > self.surprise_threshold * 2:
            return "Very High - Test data shows highly unexpected patterns"
        elif overall_surprise > self.surprise_threshold:
            return "High - Significant deviations from baseline behavior"
        elif overall_surprise > self.surprise_threshold * 0.5:
            return "Medium - Some unexpected patterns detected"
        else:
            return "Low - Test data behaves similarly to baseline"
    
    def _generate_surprise_recommendations(self, surprise_metrics: Dict[str, SurpriseMetrics],
                                         outlier_analysis: Dict[str, Any],
                                         overall_surprise: float) -> List[str]:
        """Generate recommendations based on surprise analysis."""
        recommendations = []
        
        # High surprise recommendations
        high_surprise_layers = [name for name, metrics in surprise_metrics.items() 
                              if metrics.surprise_score > self.surprise_threshold]
        if high_surprise_layers:
            recommendations.append(
                f"High surprise detected in {len(high_surprise_layers)} layers. "
                "Consider investigating test data distribution or model behavior."
            )
        
        # Outlier recommendations
        high_outlier_layers = [name for name, analysis in outlier_analysis.items()
                             if max(analysis['outlier_counts'].values()) > 
                             analysis['total_test_samples'] * 0.1]
        if high_outlier_layers:
            recommendations.append(
                f"High outlier rate (>10%) in {len(high_outlier_layers)} layers. "
                "Review data quality or consider model robustness."
            )
        
        # Overall assessment recommendations
        if overall_surprise < 0.5:
            recommendations.append(
                "Low surprise scores suggest test data is very similar to baseline. "
                "Consider more diverse test cases."
            )
        elif overall_surprise > self.surprise_threshold * 3:
            recommendations.append(
                "Extremely high surprise scores may indicate model overfitting "
                "or significant domain shift."
            )
        
        return recommendations
    
    def _serialize_surprise_metrics(self, metrics: SurpriseMetrics) -> Dict[str, Any]:
        """Serialize SurpriseMetrics to dictionary."""
        return {
            'layer_name': metrics.layer_name,
            'surprise_score': metrics.surprise_score,
            'baseline_mean': metrics.baseline_mean,
            'baseline_std': metrics.baseline_std,
            'test_mean': metrics.test_mean,
            'test_std': metrics.test_std,
            'divergence_score': metrics.divergence_score,
            'num_outliers': metrics.num_outliers
        }
    
    def _serialize_baseline_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize baseline statistics to dictionary."""
        return {
            'mean': stats['mean'].mean().item(),
            'std': stats['std'].mean().item(),
            'median': stats['median'].mean().item(),
            'num_samples': stats['num_samples']
        }
    
    def validate_output(self, output: Any) -> bool:
        """Validate that the output contains required surprise analysis data."""
        if not isinstance(output, dict):
            return False
        
        required_keys = ['surprise_scores', 'distribution_comparison']
        for key in required_keys:
            if key not in output:
                logger.error(f"Missing required output key: {key}")
                return False
        
        # Check that surprise scores are valid numbers
        surprise_scores = output['surprise_scores']
        if not isinstance(surprise_scores, dict):
            logger.error("Surprise scores must be a dictionary")
            return False
        
        for layer_name, score in surprise_scores.items():
            if not isinstance(score, (int, float)) or np.isnan(score):
                logger.error(f"Invalid surprise score for layer {layer_name}: {score}")
                return False
        
        return True

def create_surprise_tracker(config: Optional[Dict[str, Any]] = None) -> SurpriseCoverageTracker:
    """Factory function to create surprise coverage tracker."""
    return SurpriseCoverageTracker(config=config)
