"""
Wasserstein Distance Comparator for NeuronMap
============================================

Compute Wasserstein distance for comparing activation distributions
between different models or conditions.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from pathlib import Path
import time
from dataclasses import dataclass

# Optional imports with fallbacks
try:
    import scipy.stats as stats
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available, using simplified distance computation")

try:
    import ot  # Python Optimal Transport library
    OT_AVAILABLE = True
except ImportError:
    OT_AVAILABLE = False
    logging.warning("POT library not available, using simplified Wasserstein approximation")

from ...core.plugin_interface import MetricsPluginBase, ToolExecutionResult

logger = logging.getLogger(__name__)

@dataclass
class WassersteinResult:
    """Result of Wasserstein distance computation."""
    distance: float
    transport_plan: Optional[np.ndarray]
    computation_method: str
    distribution_a_stats: Dict[str, float]
    distribution_b_stats: Dict[str, float]

class WassersteinComparator(MetricsPluginBase):
    """
    Wasserstein Distance Comparator for activation distribution analysis.
    
    Computes Wasserstein distances between activation distributions to measure
    how different two models or conditions are in terms of their internal
    representations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(tool_id="wasserstein_distance", config=config)
        
        self.version = "1.0.0"
        self.description = "Wasserstein distance for comparing activation distributions"
        
        # Configuration parameters
        self.distance_type = config.get('distance_type', 'euclidean') if config else 'euclidean'
        self.reg_param = config.get('regularization', 1e-2) if config else 1e-2  # For Sinkhorn
        self.max_iter = config.get('max_iterations', 1000) if config else 1000
        self.use_gpu = config.get('use_gpu', False) if config else False
        self.approximation_samples = config.get('approximation_samples', 1000) if config else 1000
        
        logger.info(f"Initialized Wasserstein comparator (method: {'exact' if OT_AVAILABLE else 'approximate'})")
    
    def initialize(self) -> bool:
        """Initialize the Wasserstein comparator."""
        try:
            if not SCIPY_AVAILABLE:
                logger.warning("SciPy not available - using basic numpy implementation")
            
            if not OT_AVAILABLE:
                logger.warning("POT library not available - using approximation methods")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Wasserstein comparator: {e}")
            return False
    
    def execute(self, distribution_a: Union[torch.Tensor, np.ndarray],
                distribution_b: Union[torch.Tensor, np.ndarray],
                weights_a: Optional[Union[torch.Tensor, np.ndarray]] = None,
                weights_b: Optional[Union[torch.Tensor, np.ndarray]] = None,
                **kwargs) -> ToolExecutionResult:
        """
        Execute Wasserstein distance computation.
        
        Args:
            distribution_a: First distribution (samples x features)
            distribution_b: Second distribution (samples x features)
            weights_a: Optional weights for distribution_a samples
            weights_b: Optional weights for distribution_b samples
            
        Returns:
            ToolExecutionResult with Wasserstein distance and analysis
        """
        start_time = time.time()
        
        try:
            # Convert inputs to numpy if needed
            dist_a = self._ensure_numpy(distribution_a)
            dist_b = self._ensure_numpy(distribution_b)
            
            # Validate inputs
            if dist_a.shape[1] != dist_b.shape[1]:
                raise ValueError(f"Distributions must have same feature dimension: "
                               f"{dist_a.shape[1]} vs {dist_b.shape[1]}")
            
            # Prepare weights
            if weights_a is None:
                weights_a = np.ones(dist_a.shape[0]) / dist_a.shape[0]
            else:
                weights_a = self._ensure_numpy(weights_a)
                weights_a = weights_a / weights_a.sum()
            
            if weights_b is None:
                weights_b = np.ones(dist_b.shape[0]) / dist_b.shape[0]
            else:
                weights_b = self._ensure_numpy(weights_b)
                weights_b = weights_b / weights_b.sum()
            
            # Compute Wasserstein distance
            if OT_AVAILABLE and dist_a.shape[0] <= self.approximation_samples:
                wasserstein_result = self._compute_exact_wasserstein(
                    dist_a, dist_b, weights_a, weights_b
                )
            else:
                wasserstein_result = self._compute_approximate_wasserstein(
                    dist_a, dist_b, weights_a, weights_b
                )
            
            # Compute additional statistics
            distribution_stats = self._compute_distribution_statistics(dist_a, dist_b)
            
            # Analyze transport plan if available
            transport_analysis = self._analyze_transport_plan(
                wasserstein_result, dist_a, dist_b
            ) if wasserstein_result.transport_plan is not None else {}
            
            # Generate comparison report
            comparison_report = self._generate_comparison_report(
                wasserstein_result, distribution_stats, transport_analysis
            )
            
            # Prepare outputs
            outputs = {
                'wasserstein_distance': wasserstein_result.distance,
                'optimal_transport_plan': wasserstein_result.transport_plan.tolist() 
                    if wasserstein_result.transport_plan is not None else None,
                'computation_method': wasserstein_result.computation_method,
                'distribution_statistics': distribution_stats,
                'transport_analysis': transport_analysis,
                'comparison_report': comparison_report,
                'method_metadata': {
                    'distance_type': self.distance_type,
                    'regularization': self.reg_param,
                    'samples_a': dist_a.shape[0],
                    'samples_b': dist_b.shape[0],
                    'feature_dimension': dist_a.shape[1]
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
            logger.error(f"Wasserstein computation failed: {e}")
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
    
    def _ensure_numpy(self, data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert tensor to numpy array if needed."""
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        return np.asarray(data)
    
    def _compute_exact_wasserstein(self, dist_a: np.ndarray, dist_b: np.ndarray,
                                  weights_a: np.ndarray, weights_b: np.ndarray) -> WassersteinResult:
        """Compute exact Wasserstein distance using POT library."""
        try:
            # Compute cost matrix
            cost_matrix = self._compute_cost_matrix(dist_a, dist_b)
            
            # Solve optimal transport problem
            if self.reg_param > 0:
                # Entropic regularization (Sinkhorn)
                transport_plan = ot.sinkhorn(weights_a, weights_b, cost_matrix, 
                                           self.reg_param, numItermax=self.max_iter)
                method = "sinkhorn_regularized"
            else:
                # Exact solution (slower but exact)
                transport_plan = ot.emd(weights_a, weights_b, cost_matrix)
                method = "exact_emd"
            
            # Compute Wasserstein distance
            wasserstein_dist = np.sum(transport_plan * cost_matrix)
            
            return WassersteinResult(
                distance=float(wasserstein_dist),
                transport_plan=transport_plan,
                computation_method=method,
                distribution_a_stats=self._compute_basic_stats(dist_a),
                distribution_b_stats=self._compute_basic_stats(dist_b)
            )
            
        except Exception as e:
            logger.error(f"Exact Wasserstein computation failed: {e}")
            # Fallback to approximation
            return self._compute_approximate_wasserstein(dist_a, dist_b, weights_a, weights_b)
    
    def _compute_approximate_wasserstein(self, dist_a: np.ndarray, dist_b: np.ndarray,
                                        weights_a: np.ndarray, weights_b: np.ndarray) -> WassersteinResult:
        """Compute approximate Wasserstein distance."""
        try:
            # Sample if distributions are too large
            if dist_a.shape[0] > self.approximation_samples:
                indices_a = np.random.choice(dist_a.shape[0], self.approximation_samples, 
                                           p=weights_a, replace=False)
                dist_a_sample = dist_a[indices_a]
                weights_a_sample = np.ones(len(indices_a)) / len(indices_a)
            else:
                dist_a_sample = dist_a
                weights_a_sample = weights_a
            
            if dist_b.shape[0] > self.approximation_samples:
                indices_b = np.random.choice(dist_b.shape[0], self.approximation_samples, 
                                           p=weights_b, replace=False)
                dist_b_sample = dist_b[indices_b]
                weights_b_sample = np.ones(len(indices_b)) / len(indices_b)
            else:
                dist_b_sample = dist_b
                weights_b_sample = weights_b
            
            # Use sliced Wasserstein if possible
            if SCIPY_AVAILABLE:
                wasserstein_dist = self._compute_sliced_wasserstein(
                    dist_a_sample, dist_b_sample, weights_a_sample, weights_b_sample
                )
                method = "sliced_wasserstein"
            else:
                # Very basic approximation using means and covariances
                wasserstein_dist = self._compute_basic_wasserstein_approximation(
                    dist_a_sample, dist_b_sample
                )
                method = "basic_approximation"
            
            return WassersteinResult(
                distance=float(wasserstein_dist),
                transport_plan=None,  # Not computed for approximations
                computation_method=method,
                distribution_a_stats=self._compute_basic_stats(dist_a),
                distribution_b_stats=self._compute_basic_stats(dist_b)
            )
            
        except Exception as e:
            logger.error(f"Approximate Wasserstein computation failed: {e}")
            # Final fallback
            return WassersteinResult(
                distance=0.0,
                transport_plan=None,
                computation_method="failed",
                distribution_a_stats=self._compute_basic_stats(dist_a),
                distribution_b_stats=self._compute_basic_stats(dist_b)
            )
    
    def _compute_cost_matrix(self, dist_a: np.ndarray, dist_b: np.ndarray) -> np.ndarray:
        """Compute cost matrix between distributions."""
        if SCIPY_AVAILABLE:
            return cdist(dist_a, dist_b, metric=self.distance_type)
        else:
            # Simple Euclidean distance fallback
            cost_matrix = np.zeros((dist_a.shape[0], dist_b.shape[0]))
            for i in range(dist_a.shape[0]):
                for j in range(dist_b.shape[0]):
                    cost_matrix[i, j] = np.linalg.norm(dist_a[i] - dist_b[j])
            return cost_matrix
    
    def _compute_sliced_wasserstein(self, dist_a: np.ndarray, dist_b: np.ndarray,
                                   weights_a: np.ndarray, weights_b: np.ndarray,
                                   n_projections: int = 100) -> float:
        """Compute sliced Wasserstein distance."""
        n_features = dist_a.shape[1]
        wasserstein_distances = []
        
        for _ in range(n_projections):
            # Random projection direction
            projection = np.random.randn(n_features)
            projection = projection / np.linalg.norm(projection)
            
            # Project distributions
            proj_a = dist_a @ projection
            proj_b = dist_b @ projection
            
            # Compute 1D Wasserstein distance
            if SCIPY_AVAILABLE:
                w_dist = stats.wasserstein_distance(proj_a, proj_b, weights_a, weights_b)
            else:
                # Simple approximation for 1D case
                w_dist = abs(np.average(proj_a, weights=weights_a) - 
                           np.average(proj_b, weights=weights_b))
            
            wasserstein_distances.append(w_dist)
        
        return np.mean(wasserstein_distances)
    
    def _compute_basic_wasserstein_approximation(self, dist_a: np.ndarray, 
                                               dist_b: np.ndarray) -> float:
        """Very basic Wasserstein approximation using moments."""
        # Approximate using difference in means and standard deviations
        mean_a = np.mean(dist_a, axis=0)
        mean_b = np.mean(dist_b, axis=0)
        
        std_a = np.std(dist_a, axis=0)
        std_b = np.std(dist_b, axis=0)
        
        # Simple approximation
        mean_diff = np.linalg.norm(mean_a - mean_b)
        std_diff = np.linalg.norm(std_a - std_b)
        
        return mean_diff + 0.5 * std_diff
    
    def _compute_basic_stats(self, distribution: np.ndarray) -> Dict[str, float]:
        """Compute basic statistics for a distribution."""
        return {
            'mean': float(np.mean(distribution)),
            'std': float(np.std(distribution)),
            'min': float(np.min(distribution)),
            'max': float(np.max(distribution)),
            'num_samples': int(distribution.shape[0]),
            'num_features': int(distribution.shape[1])
        }
    
    def _compute_distribution_statistics(self, dist_a: np.ndarray, 
                                       dist_b: np.ndarray) -> Dict[str, Any]:
        """Compute comprehensive distribution statistics."""
        stats_a = self._compute_basic_stats(dist_a)
        stats_b = self._compute_basic_stats(dist_b)
        
        # Compute additional comparative statistics
        mean_distance = np.linalg.norm(np.mean(dist_a, axis=0) - np.mean(dist_b, axis=0))
        std_distance = np.linalg.norm(np.std(dist_a, axis=0) - np.std(dist_b, axis=0))
        
        # Feature-wise statistics
        feature_wise_means_a = np.mean(dist_a, axis=0)
        feature_wise_means_b = np.mean(dist_b, axis=0)
        feature_wise_distances = np.abs(feature_wise_means_a - feature_wise_means_b)
        
        return {
            'distribution_a': stats_a,
            'distribution_b': stats_b,
            'comparative_stats': {
                'mean_distance': float(mean_distance),
                'std_distance': float(std_distance),
                'max_feature_distance': float(np.max(feature_wise_distances)),
                'avg_feature_distance': float(np.mean(feature_wise_distances))
            }
        }
    
    def _analyze_transport_plan(self, wasserstein_result: WassersteinResult,
                               dist_a: np.ndarray, dist_b: np.ndarray) -> Dict[str, Any]:
        """Analyze the optimal transport plan."""
        if wasserstein_result.transport_plan is None:
            return {}
        
        transport_plan = wasserstein_result.transport_plan
        
        # Transport sparsity
        transport_sparsity = np.sum(transport_plan > 1e-6) / transport_plan.size
        
        # Mass conservation check
        row_sums = np.sum(transport_plan, axis=1)
        col_sums = np.sum(transport_plan, axis=0)
        mass_conservation_error = max(
            np.max(np.abs(row_sums - 1/dist_a.shape[0])),
            np.max(np.abs(col_sums - 1/dist_b.shape[0]))
        )
        
        # Most transported mass
        max_transport_indices = np.unravel_index(np.argmax(transport_plan), transport_plan.shape)
        max_transport_value = transport_plan[max_transport_indices]
        
        return {
            'transport_sparsity': float(transport_sparsity),
            'mass_conservation_error': float(mass_conservation_error),
            'max_transport_value': float(max_transport_value),
            'max_transport_indices': max_transport_indices,
            'transport_entropy': float(self._compute_transport_entropy(transport_plan))
        }
    
    def _compute_transport_entropy(self, transport_plan: np.ndarray) -> float:
        """Compute entropy of transport plan."""
        # Add small epsilon to avoid log(0)
        eps = 1e-12
        transport_plan_safe = transport_plan + eps
        
        # Normalize to get probabilities
        transport_probs = transport_plan_safe / np.sum(transport_plan_safe)
        
        # Compute entropy
        entropy = -np.sum(transport_probs * np.log(transport_probs))
        return entropy
    
    def _generate_comparison_report(self, wasserstein_result: WassersteinResult,
                                   distribution_stats: Dict[str, Any],
                                   transport_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        
        # Interpret Wasserstein distance
        distance = wasserstein_result.distance
        interpretation = self._interpret_wasserstein_distance(distance, distribution_stats)
        
        # Summary statistics
        report = {
            'wasserstein_distance': distance,
            'interpretation': interpretation,
            'computation_method': wasserstein_result.computation_method,
            'distribution_comparison': {
                'size_ratio': (distribution_stats['distribution_b']['num_samples'] / 
                             distribution_stats['distribution_a']['num_samples']),
                'mean_distance': distribution_stats['comparative_stats']['mean_distance'],
                'feature_dimension': distribution_stats['distribution_a']['num_features']
            }
        }
        
        # Add transport analysis if available
        if transport_analysis:
            report['transport_properties'] = transport_analysis
        
        # Recommendations
        report['recommendations'] = self._generate_recommendations(
            wasserstein_result, distribution_stats
        )
        
        return report
    
    def _interpret_wasserstein_distance(self, distance: float, 
                                       distribution_stats: Dict[str, Any]) -> str:
        """Interpret the magnitude of Wasserstein distance."""
        # Normalize by typical scale of the data
        typical_scale = max(
            distribution_stats['distribution_a']['std'],
            distribution_stats['distribution_b']['std']
        )
        
        normalized_distance = distance / typical_scale if typical_scale > 0 else distance
        
        if normalized_distance < 0.1:
            return "Very similar distributions"
        elif normalized_distance < 0.5:
            return "Moderately similar distributions"
        elif normalized_distance < 1.0:
            return "Somewhat different distributions"
        elif normalized_distance < 2.0:
            return "Quite different distributions"
        else:
            return "Very different distributions"
    
    def _generate_recommendations(self, wasserstein_result: WassersteinResult,
                                 distribution_stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on Wasserstein analysis."""
        recommendations = []
        
        distance = wasserstein_result.distance
        comp_stats = distribution_stats['comparative_stats']
        
        # Distance-based recommendations
        if distance > comp_stats['mean_distance'] * 2:
            recommendations.append(
                "Large Wasserstein distance suggests significant distributional differences. "
                "Consider investigating what causes this divergence."
            )
        
        if distance < 0.01 * comp_stats['mean_distance']:
            recommendations.append(
                "Very small Wasserstein distance suggests distributions are nearly identical. "
                "Verify if this is expected."
            )
        
        # Method-based recommendations
        if wasserstein_result.computation_method in ["sliced_wasserstein", "basic_approximation"]:
            recommendations.append(
                "Used approximation method due to computational constraints. "
                "Consider reducing sample size for exact computation."
            )
        
        # Feature-based recommendations
        if comp_stats['max_feature_distance'] > comp_stats['avg_feature_distance'] * 3:
            recommendations.append(
                "Large variance in feature-wise differences detected. "
                "Some features may be driving the overall distance."
            )
        
        return recommendations
    
    def compute_distance(self, data_a: Any, data_b: Any, **kwargs) -> Dict[str, Any]:
        """Interface method for MetricsPluginBase."""
        result = self.execute(data_a, data_b, **kwargs)
        return result.outputs if result.success else {}
    
    def validate_output(self, output: Any) -> bool:
        """Validate that the output contains required Wasserstein analysis data."""
        if not isinstance(output, dict):
            return False
        
        required_keys = ['wasserstein_distance', 'computation_method']
        for key in required_keys:
            if key not in output:
                logger.error(f"Missing required output key: {key}")
                return False
        
        # Check that distance is a valid number
        distance = output['wasserstein_distance']
        if not isinstance(distance, (int, float)) or np.isnan(distance) or distance < 0:
            logger.error(f"Invalid Wasserstein distance: {distance}")
            return False
        
        return True

def create_wasserstein_comparator(config: Optional[Dict[str, Any]] = None) -> WassersteinComparator:
    """Factory function to create Wasserstein comparator."""
    return WassersteinComparator(config=config)
