"""
EMD Heatmap Comparator for NeuronMap
===================================

Earth Mover's Distance (EMD) computation for comparing clustermaps
and activation heatmaps between different models or conditions.
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
    from scipy.spatial.distance import cdist, pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available, using simplified distance computation")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/Seaborn not available, heatmap visualization disabled")

try:
    from sklearn.metrics import pairwise_distances
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available, using basic distance computation")

from ...core.plugin_interface import MetricsPluginBase, ToolExecutionResult

logger = logging.getLogger(__name__)

@dataclass
class EMDResult:
    """Result of EMD computation between heatmaps."""
    emd_distance: float
    optimal_flow: Optional[np.ndarray]
    computation_method: str
    heatmap_a_stats: Dict[str, float]
    heatmap_b_stats: Dict[str, float]
    flow_analysis: Dict[str, Any]

class EMDHeatmapComparator(MetricsPluginBase):
    """
    Earth Mover's Distance (EMD) Comparator for heatmaps and clustermaps.
    
    Computes EMD between activation heatmaps to quantify how much "work"
    is needed to transform one activation pattern into another.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(tool_id="emd_heatmap", config=config)
        
        self.version = "1.0.0"
        self.description = "EMD for comparing activation heatmaps and clustermaps"
        
        # Configuration parameters
        self.normalization = config.get('normalization', 'sum') if config else 'sum'  # 'sum', 'max', 'none'
        self.distance_metric = config.get('distance_metric', 'euclidean') if config else 'euclidean'
        self.approximation_threshold = config.get('approximation_threshold', 1000) if config else 1000
        self.bins_for_histogram = config.get('bins', 50) if config else 50
        self.save_visualizations = config.get('save_visualizations', False) if config else False
        
        logger.info(f"Initialized EMD heatmap comparator (metric: {self.distance_metric})")
    
    def initialize(self) -> bool:
        """Initialize the EMD heatmap comparator."""
        try:
            if not SCIPY_AVAILABLE:
                logger.warning("SciPy not available - using basic numpy implementation")
            
            if not PLOTTING_AVAILABLE:
                logger.warning("Matplotlib not available - visualization disabled")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize EMD comparator: {e}")
            return False
    
    def execute(self, heatmap_a: Union[torch.Tensor, np.ndarray],
                heatmap_b: Union[torch.Tensor, np.ndarray],
                positions_a: Optional[np.ndarray] = None,
                positions_b: Optional[np.ndarray] = None,
                **kwargs) -> ToolExecutionResult:
        """
        Execute EMD computation between heatmaps.
        
        Args:
            heatmap_a: First heatmap (2D array)
            heatmap_b: Second heatmap (2D array)
            positions_a: Optional spatial positions for heatmap_a pixels
            positions_b: Optional spatial positions for heatmap_b pixels
            
        Returns:
            ToolExecutionResult with EMD distance and analysis
        """
        start_time = time.time()
        
        try:
            # Convert inputs to numpy
            heat_a = self._ensure_numpy(heatmap_a)
            heat_b = self._ensure_numpy(heatmap_b)
            
            # Validate inputs
            if heat_a.ndim != 2 or heat_b.ndim != 2:
                raise ValueError("Heatmaps must be 2D arrays")
            
            # Prepare heatmaps for EMD computation
            processed_a, processed_b, pos_a, pos_b = self._preprocess_heatmaps(
                heat_a, heat_b, positions_a, positions_b
            )
            
            # Compute EMD
            emd_result = self._compute_emd(processed_a, processed_b, pos_a, pos_b)
            
            # Analyze the optimal flow if available
            flow_analysis = self._analyze_optimal_flow(
                emd_result, processed_a, processed_b, pos_a, pos_b
            )
            emd_result.flow_analysis = flow_analysis
            
            # Generate comparison visualizations
            visualizations = {}
            if PLOTTING_AVAILABLE and self.save_visualizations:
                visualizations = self._create_comparison_visualizations(
                    heat_a, heat_b, emd_result
                )
            
            # Compute additional heatmap statistics
            heatmap_stats = self._compute_heatmap_statistics(heat_a, heat_b)
            
            # Generate EMD interpretation
            interpretation = self._interpret_emd_result(emd_result, heatmap_stats)
            
            # Prepare outputs
            outputs = {
                'emd_distance': emd_result.emd_distance,
                'computation_method': emd_result.computation_method,
                'optimal_flow': emd_result.optimal_flow.tolist() 
                    if emd_result.optimal_flow is not None else None,
                'flow_analysis': flow_analysis,
                'heatmap_statistics': heatmap_stats,
                'interpretation': interpretation,
                'visualizations': visualizations,
                'method_metadata': {
                    'normalization': self.normalization,
                    'distance_metric': self.distance_metric,
                    'heatmap_a_shape': heat_a.shape,
                    'heatmap_b_shape': heat_b.shape,
                    'total_mass_a': float(np.sum(processed_a)),
                    'total_mass_b': float(np.sum(processed_b))
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
            logger.error(f"EMD computation failed: {e}")
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
    
    def _preprocess_heatmaps(self, heat_a: np.ndarray, heat_b: np.ndarray,
                            positions_a: Optional[np.ndarray] = None,
                            positions_b: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess heatmaps for EMD computation."""
        
        # Flatten heatmaps
        flat_a = heat_a.flatten()
        flat_b = heat_b.flatten()
        
        # Generate positions if not provided
        if positions_a is None:
            h_a, w_a = heat_a.shape
            y_a, x_a = np.meshgrid(np.arange(h_a), np.arange(w_a), indexing='ij')
            positions_a = np.column_stack([y_a.flatten(), x_a.flatten()])
        
        if positions_b is None:
            h_b, w_b = heat_b.shape
            y_b, x_b = np.meshgrid(np.arange(h_b), np.arange(w_b), indexing='ij')
            positions_b = np.column_stack([y_b.flatten(), x_b.flatten()])
        
        # Normalize values to create proper probability distributions
        if self.normalization == 'sum':
            # Ensure non-negative and normalize to sum to 1
            flat_a = np.maximum(flat_a, 0)
            flat_b = np.maximum(flat_b, 0)
            
            sum_a = np.sum(flat_a)
            sum_b = np.sum(flat_b)
            
            if sum_a > 0:
                flat_a = flat_a / sum_a
            else:
                flat_a = np.ones_like(flat_a) / len(flat_a)
            
            if sum_b > 0:
                flat_b = flat_b / sum_b
            else:
                flat_b = np.ones_like(flat_b) / len(flat_b)
                
        elif self.normalization == 'max':
            # Normalize by maximum value
            max_a = np.max(np.abs(flat_a))
            max_b = np.max(np.abs(flat_b))
            
            if max_a > 0:
                flat_a = flat_a / max_a
            if max_b > 0:
                flat_b = flat_b / max_b
                
            # Make non-negative and normalize
            flat_a = np.maximum(flat_a, 0)
            flat_b = np.maximum(flat_b, 0)
            flat_a = flat_a / np.sum(flat_a) if np.sum(flat_a) > 0 else flat_a
            flat_b = flat_b / np.sum(flat_b) if np.sum(flat_b) > 0 else flat_b
        
        # Filter out zero-mass pixels for efficiency
        nonzero_a = flat_a > 1e-10
        nonzero_b = flat_b > 1e-10
        
        return (flat_a[nonzero_a], flat_b[nonzero_b], 
                positions_a[nonzero_a], positions_b[nonzero_b])
    
    def _compute_emd(self, mass_a: np.ndarray, mass_b: np.ndarray,
                     pos_a: np.ndarray, pos_b: np.ndarray) -> EMDResult:
        """Compute EMD between preprocessed heatmaps."""
        
        try:
            # For large problems, use approximation
            if len(mass_a) > self.approximation_threshold or len(mass_b) > self.approximation_threshold:
                return self._compute_approximate_emd(mass_a, mass_b, pos_a, pos_b)
            
            # Compute distance matrix
            if SKLEARN_AVAILABLE:
                distance_matrix = pairwise_distances(pos_a, pos_b, metric=self.distance_metric)
            elif SCIPY_AVAILABLE:
                distance_matrix = cdist(pos_a, pos_b, metric=self.distance_metric)
            else:
                distance_matrix = self._compute_distance_matrix_basic(pos_a, pos_b)
            
            # Solve EMD using simple iterative algorithm
            optimal_flow, emd_distance = self._solve_emd_iterative(
                mass_a, mass_b, distance_matrix
            )
            
            return EMDResult(
                emd_distance=float(emd_distance),
                optimal_flow=optimal_flow,
                computation_method="iterative_solver",
                heatmap_a_stats=self._compute_mass_stats(mass_a),
                heatmap_b_stats=self._compute_mass_stats(mass_b),
                flow_analysis={}
            )
            
        except Exception as e:
            logger.error(f"EMD computation failed: {e}")
            # Fallback to simple approximation
            return self._compute_simple_emd_approximation(mass_a, mass_b, pos_a, pos_b)
    
    def _compute_distance_matrix_basic(self, pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
        """Compute distance matrix using basic numpy operations."""
        n_a, n_b = len(pos_a), len(pos_b)
        distance_matrix = np.zeros((n_a, n_b))
        
        for i in range(n_a):
            for j in range(n_b):
                if self.distance_metric == 'euclidean':
                    distance_matrix[i, j] = np.linalg.norm(pos_a[i] - pos_b[j])
                elif self.distance_metric == 'manhattan':
                    distance_matrix[i, j] = np.sum(np.abs(pos_a[i] - pos_b[j]))
                else:
                    # Default to Euclidean
                    distance_matrix[i, j] = np.linalg.norm(pos_a[i] - pos_b[j])
        
        return distance_matrix
    
    def _solve_emd_iterative(self, mass_a: np.ndarray, mass_b: np.ndarray,
                            distance_matrix: np.ndarray, max_iter: int = 1000) -> Tuple[np.ndarray, float]:
        """Solve EMD using iterative algorithm (simplified transport)."""
        
        n_a, n_b = len(mass_a), len(mass_b)
        flow = np.zeros((n_a, n_b))
        
        # Simple greedy algorithm: always transport to nearest available destination
        remaining_supply = mass_a.copy()
        remaining_demand = mass_b.copy()
        
        while np.sum(remaining_supply) > 1e-10 and np.sum(remaining_demand) > 1e-10:
            # Find minimum cost edge
            valid_mask = np.outer(remaining_supply > 1e-10, remaining_demand > 1e-10)
            if not np.any(valid_mask):
                break
            
            masked_costs = np.where(valid_mask, distance_matrix, np.inf)
            min_idx = np.unravel_index(np.argmin(masked_costs), masked_costs.shape)
            i, j = min_idx
            
            # Transport as much as possible on this edge
            transport_amount = min(remaining_supply[i], remaining_demand[j])
            flow[i, j] = transport_amount
            
            remaining_supply[i] -= transport_amount
            remaining_demand[j] -= transport_amount
        
        # Compute total cost
        total_cost = np.sum(flow * distance_matrix)
        
        return flow, total_cost
    
    def _compute_approximate_emd(self, mass_a: np.ndarray, mass_b: np.ndarray,
                                pos_a: np.ndarray, pos_b: np.ndarray) -> EMDResult:
        """Compute approximate EMD for large problems."""
        
        # Sample points for approximation
        n_samples = min(self.approximation_threshold // 2, len(mass_a), len(mass_b))
        
        if len(mass_a) > n_samples:
            indices_a = np.random.choice(len(mass_a), n_samples, p=mass_a, replace=False)
            sampled_mass_a = mass_a[indices_a]
            sampled_pos_a = pos_a[indices_a]
            sampled_mass_a = sampled_mass_a / np.sum(sampled_mass_a)
        else:
            sampled_mass_a = mass_a
            sampled_pos_a = pos_a
        
        if len(mass_b) > n_samples:
            indices_b = np.random.choice(len(mass_b), n_samples, p=mass_b, replace=False)
            sampled_mass_b = mass_b[indices_b]
            sampled_pos_b = pos_b[indices_b]
            sampled_mass_b = sampled_mass_b / np.sum(sampled_mass_b)
        else:
            sampled_mass_b = mass_b
            sampled_pos_b = pos_b
        
        # Compute EMD on sampled data
        return self._compute_emd(sampled_mass_a, sampled_mass_b, sampled_pos_a, sampled_pos_b)
    
    def _compute_simple_emd_approximation(self, mass_a: np.ndarray, mass_b: np.ndarray,
                                         pos_a: np.ndarray, pos_b: np.ndarray) -> EMDResult:
        """Simple EMD approximation using centroids."""
        
        # Compute centroids
        centroid_a = np.average(pos_a, weights=mass_a, axis=0)
        centroid_b = np.average(pos_b, weights=mass_b, axis=0)
        
        # Distance between centroids as EMD approximation
        centroid_distance = np.linalg.norm(centroid_a - centroid_b)
        
        # Add spread difference
        spread_a = np.average(np.linalg.norm(pos_a - centroid_a, axis=1), weights=mass_a)
        spread_b = np.average(np.linalg.norm(pos_b - centroid_b, axis=1), weights=mass_b)
        spread_difference = abs(spread_a - spread_b)
        
        approximate_emd = centroid_distance + 0.5 * spread_difference
        
        return EMDResult(
            emd_distance=float(approximate_emd),
            optimal_flow=None,
            computation_method="centroid_approximation",
            heatmap_a_stats=self._compute_mass_stats(mass_a),
            heatmap_b_stats=self._compute_mass_stats(mass_b),
            flow_analysis={}
        )
    
    def _compute_mass_stats(self, mass: np.ndarray) -> Dict[str, float]:
        """Compute statistics for mass distribution."""
        return {
            'total_mass': float(np.sum(mass)),
            'max_mass': float(np.max(mass)),
            'min_mass': float(np.min(mass)),
            'mean_mass': float(np.mean(mass)),
            'std_mass': float(np.std(mass)),
            'num_points': int(len(mass))
        }
    
    def _analyze_optimal_flow(self, emd_result: EMDResult,
                             mass_a: np.ndarray, mass_b: np.ndarray,
                             pos_a: np.ndarray, pos_b: np.ndarray) -> Dict[str, Any]:
        """Analyze the optimal transport flow."""
        
        if emd_result.optimal_flow is None:
            return {'analysis_available': False}
        
        flow = emd_result.optimal_flow
        
        # Flow statistics
        total_flow = np.sum(flow)
        nonzero_flows = flow[flow > 1e-10]
        flow_sparsity = len(nonzero_flows) / flow.size
        
        # Most significant flows
        max_flow_idx = np.unravel_index(np.argmax(flow), flow.shape)
        max_flow_value = flow[max_flow_idx]
        
        # Average transport distance
        if len(pos_a) > 0 and len(pos_b) > 0:
            transport_distances = []
            for i in range(len(mass_a)):
                for j in range(len(mass_b)):
                    if flow[i, j] > 1e-10:
                        distance = np.linalg.norm(pos_a[i] - pos_b[j])
                        transport_distances.extend([distance] * int(flow[i, j] * 1000))  # Weight by flow
            
            avg_transport_distance = np.mean(transport_distances) if transport_distances else 0.0
        else:
            avg_transport_distance = 0.0
        
        return {
            'analysis_available': True,
            'total_flow': float(total_flow),
            'flow_sparsity': float(flow_sparsity),
            'max_flow_value': float(max_flow_value),
            'max_flow_indices': max_flow_idx,
            'average_transport_distance': float(avg_transport_distance),
            'num_active_transports': int(len(nonzero_flows))
        }
    
    def _compute_heatmap_statistics(self, heat_a: np.ndarray, heat_b: np.ndarray) -> Dict[str, Any]:
        """Compute comprehensive heatmap statistics."""
        
        # Basic statistics
        stats_a = {
            'mean': float(np.mean(heat_a)),
            'std': float(np.std(heat_a)),
            'min': float(np.min(heat_a)),
            'max': float(np.max(heat_a)),
            'shape': heat_a.shape
        }
        
        stats_b = {
            'mean': float(np.mean(heat_b)),
            'std': float(np.std(heat_b)),
            'min': float(np.min(heat_b)),
            'max': float(np.max(heat_b)),
            'shape': heat_b.shape
        }
        
        # Comparative statistics
        if heat_a.shape == heat_b.shape:
            pixel_wise_diff = np.abs(heat_a - heat_b)
            comparative_stats = {
                'pixel_wise_mean_diff': float(np.mean(pixel_wise_diff)),
                'pixel_wise_max_diff': float(np.max(pixel_wise_diff)),
                'correlation': float(np.corrcoef(heat_a.flatten(), heat_b.flatten())[0, 1])
                    if not np.isnan(np.corrcoef(heat_a.flatten(), heat_b.flatten())[0, 1]) else 0.0
            }
        else:
            comparative_stats = {
                'shape_mismatch': True,
                'size_ratio': heat_b.size / heat_a.size
            }
        
        return {
            'heatmap_a': stats_a,
            'heatmap_b': stats_b,
            'comparative': comparative_stats
        }
    
    def _interpret_emd_result(self, emd_result: EMDResult, heatmap_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Interpret EMD result in context of heatmap properties."""
        
        distance = emd_result.emd_distance
        
        # Normalize distance by heatmap properties
        typical_scale = max(
            heatmap_stats['heatmap_a']['std'],
            heatmap_stats['heatmap_b']['std']
        )
        
        normalized_distance = distance / typical_scale if typical_scale > 0 else distance
        
        # Generate interpretation
        if normalized_distance < 0.1:
            similarity = "Very similar activation patterns"
        elif normalized_distance < 0.5:
            similarity = "Moderately similar activation patterns"
        elif normalized_distance < 1.0:
            similarity = "Somewhat different activation patterns"
        elif normalized_distance < 2.0:
            similarity = "Quite different activation patterns"
        else:
            similarity = "Very different activation patterns"
        
        # Generate recommendations
        recommendations = []
        
        if distance < 0.01:
            recommendations.append("Extremely low EMD suggests near-identical patterns")
        
        if emd_result.computation_method == "centroid_approximation":
            recommendations.append("Used approximation method - consider reducing data size for exact computation")
        
        if emd_result.flow_analysis.get('flow_sparsity', 1.0) < 0.1:
            recommendations.append("Very sparse transport flow - patterns may have distinct concentrated regions")
        
        return {
            'similarity_assessment': similarity,
            'normalized_distance': float(normalized_distance),
            'recommendations': recommendations,
            'distance_interpretation': f"EMD of {distance:.4f} using {emd_result.computation_method}"
        }
    
    def _create_comparison_visualizations(self, heat_a: np.ndarray, heat_b: np.ndarray,
                                         emd_result: EMDResult) -> Dict[str, str]:
        """Create comparison visualizations if plotting available."""
        
        if not PLOTTING_AVAILABLE:
            return {}
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot original heatmaps
            im1 = axes[0, 0].imshow(heat_a, cmap='viridis')
            axes[0, 0].set_title('Heatmap A')
            plt.colorbar(im1, ax=axes[0, 0])
            
            im2 = axes[0, 1].imshow(heat_b, cmap='viridis')
            axes[0, 1].set_title('Heatmap B')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # Plot difference
            if heat_a.shape == heat_b.shape:
                diff = heat_a - heat_b
                im3 = axes[1, 0].imshow(diff, cmap='RdBu')
                axes[1, 0].set_title('Difference (A - B)')
                plt.colorbar(im3, ax=axes[1, 0])
            
            # Summary statistics
            axes[1, 1].text(0.1, 0.8, f"EMD Distance: {emd_result.emd_distance:.4f}", 
                           transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.6, f"Method: {emd_result.computation_method}", 
                           transform=axes[1, 1].transAxes, fontsize=10)
            axes[1, 1].set_title('EMD Summary')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Save plot
            visualization_path = f"emd_comparison_{int(time.time())}.png"
            plt.savefig(visualization_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return {'comparison_plot': visualization_path}
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
            return {}
    
    def compute_distance(self, data_a: Any, data_b: Any, **kwargs) -> Dict[str, Any]:
        """Interface method for MetricsPluginBase."""
        result = self.execute(data_a, data_b, **kwargs)
        return result.outputs if result.success else {}
    
    def validate_output(self, output: Any) -> bool:
        """Validate EMD computation output."""
        if not isinstance(output, dict):
            return False
        
        required_keys = ['emd_distance', 'computation_method']
        for key in required_keys:
            if key not in output:
                logger.error(f"Missing required output key: {key}")
                return False
        
        # Check distance validity
        distance = output['emd_distance']
        if not isinstance(distance, (int, float)) or np.isnan(distance) or distance < 0:
            logger.error(f"Invalid EMD distance: {distance}")
            return False
        
        return True

def create_emd_comparator(config: Optional[Dict[str, Any]] = None) -> EMDHeatmapComparator:
    """Factory function to create EMD heatmap comparator."""
    return EMDHeatmapComparator(config=config)
