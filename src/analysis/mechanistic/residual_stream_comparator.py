"""
Residual Stream Comparator for NeuronMap
========================================

Advanced residual stream analysis combining TransformerLens and NeuronMap
data for comprehensive mechanistic interpretability.
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
    from scipy.spatial.distance import cosine
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available, using basic numpy operations")

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available, dimensionality reduction disabled")

from ...core.plugin_interface import MechanisticPluginBase, ToolExecutionResult

logger = logging.getLogger(__name__)

@dataclass
class ResidualStreamComparison:
    """Result of residual stream comparison."""
    similarity_scores: Dict[str, float]
    dimensionality_analysis: Dict[str, Any]
    flow_analysis: Dict[str, Any]
    layer_wise_comparison: Dict[str, Dict[str, float]]
    combined_insights: Dict[str, Any]

class ResidualStreamComparator(MechanisticPluginBase):
    """
    Residual Stream Comparator for mechanistic interpretability.
    
    Combines TransformerLens residual stream data with NeuronMap analysis
    to provide comprehensive comparison and flow analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(tool_id="residual_stream_comparator", config=config)
        
        self.version = "1.0.0"
        self.description = "Advanced residual stream analysis and comparison"
        
        # Configuration
        self.similarity_metrics = config.get('similarity_metrics', ['cosine', 'euclidean', 'correlation']) if config else ['cosine', 'euclidean', 'correlation']
        self.dimensionality_reduction = config.get('dimensionality_reduction', 'pca') if config else 'pca'
        self.n_components = config.get('n_components', 50) if config else 50
        self.cluster_analysis = config.get('cluster_analysis', True) if config else True
        self.flow_threshold = config.get('flow_threshold', 0.1) if config else 0.1
        self.layer_grouping = config.get('layer_grouping', 'all') if config else 'all'  # 'all', 'attention', 'mlp'
        
        logger.info("Initialized Residual Stream Comparator")
    
    def initialize(self) -> bool:
        """Initialize the residual stream comparator."""
        try:
            if not SCIPY_AVAILABLE:
                logger.warning("SciPy not available - using limited similarity metrics")
            
            if not SKLEARN_AVAILABLE:
                logger.warning("Scikit-learn not available - advanced analysis disabled")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize residual stream comparator: {e}")
            return False
    
    def execute(self, 
                tl_data: Dict[str, Any],
                neuronmap_data: Dict[str, Any],
                comparison_type: str = "full",
                **kwargs) -> ToolExecutionResult:
        """
        Execute residual stream comparison.
        
        Args:
            tl_data: TransformerLens residual stream data
            neuronmap_data: NeuronMap activation data
            comparison_type: Type of comparison ('similarity', 'flow', 'dimensionality', 'full')
            
        Returns:
            ToolExecutionResult with comparison results
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            self._validate_inputs(tl_data, neuronmap_data)
            
            # Extract and align residual stream data
            tl_streams, nm_streams = self._extract_and_align_streams(tl_data, neuronmap_data)
            
            # Initialize comparison result
            comparison_result = ResidualStreamComparison(
                similarity_scores={},
                dimensionality_analysis={},
                flow_analysis={},
                layer_wise_comparison={},
                combined_insights={}
            )
            
            # Compute similarity scores
            if comparison_type in ["similarity", "full"]:
                comparison_result.similarity_scores = self._compute_similarity_scores(
                    tl_streams, nm_streams
                )
            
            # Perform dimensionality analysis
            if comparison_type in ["dimensionality", "full"]:
                comparison_result.dimensionality_analysis = self._analyze_dimensionality(
                    tl_streams, nm_streams
                )
            
            # Analyze information flow
            if comparison_type in ["flow", "full"]:
                comparison_result.flow_analysis = self._analyze_information_flow(
                    tl_streams, nm_streams
                )
            
            # Layer-wise comparison
            comparison_result.layer_wise_comparison = self._compare_layers(
                tl_streams, nm_streams
            )
            
            # Generate combined insights
            comparison_result.combined_insights = self._generate_combined_insights(
                comparison_result
            )
            
            # Convert to output format
            outputs = self._format_outputs(comparison_result, tl_data, neuronmap_data)
            
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
            logger.error(f"Residual stream comparison failed: {e}")
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
    
    def _validate_inputs(self, tl_data: Dict[str, Any], neuronmap_data: Dict[str, Any]):
        """Validate input data formats."""
        # Check TransformerLens data
        if 'residual_stream_data' not in tl_data:
            raise ValueError("TransformerLens data missing residual_stream_data")
        
        # Check NeuronMap data
        if 'neuron_activations' not in neuronmap_data:
            raise ValueError("NeuronMap data missing neuron_activations")
    
    def _extract_and_align_streams(self, tl_data: Dict[str, Any], 
                                  neuronmap_data: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Extract and align residual stream data from both sources."""
        
        # Extract TransformerLens streams
        tl_streams = {}
        for stream_name, stream_data in tl_data['residual_stream_data'].items():
            if 'resid' in stream_name:
                # Convert to numpy array
                data = np.array(stream_data['data'])
                tl_streams[stream_name] = data
        
        # Extract corresponding NeuronMap activations
        nm_streams = {}
        for layer_name, activation_data in neuronmap_data['neuron_activations'].items():
            if 'resid' in layer_name or self._is_comparable_layer(layer_name):
                activations = np.array(activation_data['activations'])
                nm_streams[layer_name] = activations
        
        # Align dimensions and shapes
        aligned_tl, aligned_nm = self._align_stream_dimensions(tl_streams, nm_streams)
        
        return aligned_tl, aligned_nm
    
    def _is_comparable_layer(self, layer_name: str) -> bool:
        """Check if a layer is comparable to residual stream."""
        comparable_patterns = ['resid', 'output', 'final', 'hidden']
        return any(pattern in layer_name.lower() for pattern in comparable_patterns)
    
    def _align_stream_dimensions(self, tl_streams: Dict[str, np.ndarray], 
                                nm_streams: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Align dimensions between TransformerLens and NeuronMap streams."""
        
        aligned_tl = {}
        aligned_nm = {}
        
        # Find common layers or create mappings
        for tl_name, tl_data in tl_streams.items():
            # Try to find corresponding NeuronMap layer
            corresponding_nm = None
            for nm_name, nm_data in nm_streams.items():
                if self._layers_correspond(tl_name, nm_name):
                    corresponding_nm = nm_data
                    break
            
            if corresponding_nm is not None:
                # Align shapes
                tl_aligned, nm_aligned = self._align_tensor_shapes(tl_data, corresponding_nm)
                aligned_tl[tl_name] = tl_aligned
                aligned_nm[tl_name] = nm_aligned
        
        return aligned_tl, aligned_nm
    
    def _layers_correspond(self, tl_name: str, nm_name: str) -> bool:
        """Check if TransformerLens and NeuronMap layers correspond."""
        # Extract layer numbers
        tl_layer = self._extract_layer_number(tl_name)
        nm_layer = self._extract_layer_number(nm_name)
        
        if tl_layer is not None and nm_layer is not None:
            return tl_layer == nm_layer
        
        # Fallback to pattern matching
        return any(pattern in tl_name and pattern in nm_name 
                  for pattern in ['resid', 'final', 'output'])
    
    def _extract_layer_number(self, layer_name: str) -> Optional[int]:
        """Extract layer number from layer name."""
        import re
        match = re.search(r'(\d+)', layer_name)
        return int(match.group(1)) if match else None
    
    def _align_tensor_shapes(self, tensor_a: np.ndarray, tensor_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align shapes of two tensors for comparison."""
        
        # Flatten if needed for compatibility
        if tensor_a.ndim > 2:
            tensor_a = tensor_a.reshape(tensor_a.shape[0], -1)
        if tensor_b.ndim > 2:
            tensor_b = tensor_b.reshape(tensor_b.shape[0], -1)
        
        # Ensure same batch size (take minimum)
        min_batch = min(tensor_a.shape[0], tensor_b.shape[0])
        tensor_a = tensor_a[:min_batch]
        tensor_b = tensor_b[:min_batch]
        
        # Ensure same feature dimension (truncate or pad)
        if tensor_a.shape[1] != tensor_b.shape[1]:
            min_features = min(tensor_a.shape[1], tensor_b.shape[1])
            tensor_a = tensor_a[:, :min_features]
            tensor_b = tensor_b[:, :min_features]
        
        return tensor_a, tensor_b
    
    def _compute_similarity_scores(self, tl_streams: Dict[str, np.ndarray], 
                                  nm_streams: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute similarity scores between aligned streams."""
        
        similarity_scores = {}
        
        for layer_name in tl_streams.keys():
            if layer_name in nm_streams:
                tl_data = tl_streams[layer_name]
                nm_data = nm_streams[layer_name]
                
                layer_similarities = {}
                
                # Cosine similarity
                if 'cosine' in self.similarity_metrics:
                    if SCIPY_AVAILABLE:
                        cosine_sim = 1 - cosine(tl_data.flatten(), nm_data.flatten())
                    else:
                        # Basic cosine similarity
                        dot_product = np.dot(tl_data.flatten(), nm_data.flatten())
                        norm_a = np.linalg.norm(tl_data.flatten())
                        norm_b = np.linalg.norm(nm_data.flatten())
                        cosine_sim = dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
                    
                    layer_similarities['cosine'] = float(cosine_sim)
                
                # Euclidean distance (converted to similarity)
                if 'euclidean' in self.similarity_metrics:
                    euclidean_dist = np.linalg.norm(tl_data - nm_data)
                    # Convert to similarity (higher is more similar)
                    euclidean_sim = 1 / (1 + euclidean_dist)
                    layer_similarities['euclidean'] = float(euclidean_sim)
                
                # Correlation
                if 'correlation' in self.similarity_metrics:
                    correlation = np.corrcoef(tl_data.flatten(), nm_data.flatten())[0, 1]
                    if not np.isnan(correlation):
                        layer_similarities['correlation'] = float(correlation)
                    else:
                        layer_similarities['correlation'] = 0.0
                
                # Mean Squared Error (converted to similarity)
                mse = np.mean((tl_data - nm_data) ** 2)
                layer_similarities['mse_similarity'] = float(1 / (1 + mse))
                
                similarity_scores[layer_name] = layer_similarities
        
        # Compute overall similarity
        if similarity_scores:
            overall_similarities = {}
            for metric in ['cosine', 'euclidean', 'correlation', 'mse_similarity']:
                metric_values = [scores.get(metric, 0) for scores in similarity_scores.values()]
                if metric_values:
                    overall_similarities[f'overall_{metric}'] = float(np.mean(metric_values))
            
            similarity_scores['overall'] = overall_similarities
        
        return similarity_scores
    
    def _analyze_dimensionality(self, tl_streams: Dict[str, np.ndarray], 
                               nm_streams: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze dimensionality properties of residual streams."""
        
        dimensionality_analysis = {}
        
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available - dimensionality analysis limited")
            return {'sklearn_unavailable': True}
        
        for layer_name in tl_streams.keys():
            if layer_name in nm_streams:
                tl_data = tl_streams[layer_name]
                nm_data = nm_streams[layer_name]
                
                layer_analysis = {}
                
                # PCA analysis
                if self.dimensionality_reduction == 'pca':
                    try:
                        # TL data PCA
                        pca_tl = PCA(n_components=min(self.n_components, tl_data.shape[1]))
                        tl_reduced = pca_tl.fit_transform(tl_data)
                        
                        # NM data PCA
                        pca_nm = PCA(n_components=min(self.n_components, nm_data.shape[1]))
                        nm_reduced = pca_nm.fit_transform(nm_data)
                        
                        layer_analysis['pca'] = {
                            'tl_explained_variance_ratio': pca_tl.explained_variance_ratio_[:10].tolist(),
                            'nm_explained_variance_ratio': pca_nm.explained_variance_ratio_[:10].tolist(),
                            'tl_cumulative_variance': np.cumsum(pca_tl.explained_variance_ratio_)[:10].tolist(),
                            'nm_cumulative_variance': np.cumsum(pca_nm.explained_variance_ratio_)[:10].tolist()
                        }
                        
                        # Compare reduced representations
                        if tl_reduced.shape == nm_reduced.shape:
                            reduced_similarity = np.corrcoef(tl_reduced.flatten(), nm_reduced.flatten())[0, 1]
                            layer_analysis['pca']['reduced_similarity'] = float(reduced_similarity) if not np.isnan(reduced_similarity) else 0.0
                    
                    except Exception as e:
                        logger.warning(f"PCA analysis failed for {layer_name}: {e}")
                
                # Intrinsic dimensionality estimation
                layer_analysis['intrinsic_dim'] = {
                    'tl_effective_rank': float(np.linalg.matrix_rank(tl_data)),
                    'nm_effective_rank': float(np.linalg.matrix_rank(nm_data)),
                    'tl_stable_rank': float(np.linalg.norm(tl_data, 'fro') ** 2 / np.linalg.norm(tl_data, 2) ** 2),
                    'nm_stable_rank': float(np.linalg.norm(nm_data, 'fro') ** 2 / np.linalg.norm(nm_data, 2) ** 2)
                }
                
                dimensionality_analysis[layer_name] = layer_analysis
        
        return dimensionality_analysis
    
    def _analyze_information_flow(self, tl_streams: Dict[str, np.ndarray], 
                                 nm_streams: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze information flow through residual streams."""
        
        flow_analysis = {}
        
        # Get layer ordering
        tl_layers = sorted(tl_streams.keys(), key=self._extract_layer_number)
        nm_layers = sorted(nm_streams.keys(), key=self._extract_layer_number)
        
        # Analyze flow between consecutive layers
        tl_flows = []
        nm_flows = []
        
        for i in range(len(tl_layers) - 1):
            current_layer = tl_layers[i]
            next_layer = tl_layers[i + 1]
            
            if current_layer in tl_streams and next_layer in tl_streams:
                # Compute flow magnitude
                current_data = tl_streams[current_layer]
                next_data = tl_streams[next_layer]
                
                if current_data.shape == next_data.shape:
                    flow_magnitude = np.linalg.norm(next_data - current_data)
                    tl_flows.append(flow_magnitude)
        
        # Same for NeuronMap
        for i in range(len(nm_layers) - 1):
            current_layer = nm_layers[i]
            next_layer = nm_layers[i + 1]
            
            if current_layer in nm_streams and next_layer in nm_streams:
                current_data = nm_streams[current_layer]
                next_data = nm_streams[next_layer]
                
                if current_data.shape == next_data.shape:
                    flow_magnitude = np.linalg.norm(next_data - current_data)
                    nm_flows.append(flow_magnitude)
        
        # Flow analysis
        flow_analysis['layer_to_layer_flow'] = {
            'tl_flows': [float(f) for f in tl_flows],
            'nm_flows': [float(f) for f in nm_flows],
            'tl_mean_flow': float(np.mean(tl_flows)) if tl_flows else 0.0,
            'nm_mean_flow': float(np.mean(nm_flows)) if nm_flows else 0.0,
            'flow_correlation': float(np.corrcoef(tl_flows, nm_flows)[0, 1]) 
                if len(tl_flows) == len(nm_flows) and len(tl_flows) > 1 
                and not np.isnan(np.corrcoef(tl_flows, nm_flows)[0, 1]) else 0.0
        }
        
        # Information bottleneck analysis
        bottleneck_analysis = {}
        for layer_name in tl_streams.keys():
            if layer_name in nm_streams:
                tl_data = tl_streams[layer_name]
                nm_data = nm_streams[layer_name]
                
                # Compute effective dimensionality
                tl_svd = np.linalg.svd(tl_data, compute_uv=False)
                nm_svd = np.linalg.svd(nm_data, compute_uv=False)
                
                # Information content (entropy approximation)
                tl_entropy = float(-np.sum(tl_svd / np.sum(tl_svd) * np.log(tl_svd / np.sum(tl_svd) + 1e-10)))
                nm_entropy = float(-np.sum(nm_svd / np.sum(nm_svd) * np.log(nm_svd / np.sum(nm_svd) + 1e-10)))
                
                bottleneck_analysis[layer_name] = {
                    'tl_information_content': tl_entropy,
                    'nm_information_content': nm_entropy,
                    'information_ratio': nm_entropy / tl_entropy if tl_entropy > 0 else 0.0
                }
        
        flow_analysis['information_bottlenecks'] = bottleneck_analysis
        
        return flow_analysis
    
    def _compare_layers(self, tl_streams: Dict[str, np.ndarray], 
                       nm_streams: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Perform detailed layer-wise comparison."""
        
        layer_comparison = {}
        
        for layer_name in tl_streams.keys():
            if layer_name in nm_streams:
                tl_data = tl_streams[layer_name]
                nm_data = nm_streams[layer_name]
                
                comparison = {
                    'shape_match': tl_data.shape == nm_data.shape,
                    'tl_norm': float(np.linalg.norm(tl_data)),
                    'nm_norm': float(np.linalg.norm(nm_data)),
                    'norm_ratio': float(np.linalg.norm(nm_data) / np.linalg.norm(tl_data)) 
                        if np.linalg.norm(tl_data) > 0 else 0.0,
                    'mean_absolute_difference': float(np.mean(np.abs(tl_data - nm_data))),
                    'max_absolute_difference': float(np.max(np.abs(tl_data - nm_data))),
                    'relative_error': float(np.linalg.norm(tl_data - nm_data) / np.linalg.norm(tl_data))
                        if np.linalg.norm(tl_data) > 0 else float('inf')
                }
                
                layer_comparison[layer_name] = comparison
        
        return layer_comparison
    
    def _generate_combined_insights(self, comparison_result: ResidualStreamComparison) -> Dict[str, Any]:
        """Generate combined insights from all analyses."""
        
        insights = {}
        
        # Overall similarity assessment
        if comparison_result.similarity_scores.get('overall'):
            overall_sim = comparison_result.similarity_scores['overall']
            avg_similarity = np.mean(list(overall_sim.values()))
            
            if avg_similarity > 0.8:
                similarity_assessment = "High similarity between TL and NeuronMap streams"
            elif avg_similarity > 0.5:
                similarity_assessment = "Moderate similarity between streams"
            else:
                similarity_assessment = "Low similarity between streams"
            
            insights['similarity_assessment'] = similarity_assessment
            insights['average_similarity'] = float(avg_similarity)
        
        # Dimensionality insights
        if comparison_result.dimensionality_analysis:
            dim_insights = []
            for layer_name, layer_analysis in comparison_result.dimensionality_analysis.items():
                if 'intrinsic_dim' in layer_analysis:
                    tl_rank = layer_analysis['intrinsic_dim']['tl_effective_rank']
                    nm_rank = layer_analysis['intrinsic_dim']['nm_effective_rank']
                    
                    if abs(tl_rank - nm_rank) > 0.2 * max(tl_rank, nm_rank):
                        dim_insights.append(f"Significant rank difference in {layer_name}")
            
            insights['dimensionality_insights'] = dim_insights
        
        # Flow insights
        if comparison_result.flow_analysis.get('layer_to_layer_flow'):
            flow_data = comparison_result.flow_analysis['layer_to_layer_flow']
            flow_correlation = flow_data.get('flow_correlation', 0)
            
            if flow_correlation > 0.7:
                flow_assessment = "Similar information flow patterns"
            elif flow_correlation > 0.3:
                flow_assessment = "Moderately similar flow patterns"
            else:
                flow_assessment = "Different flow patterns"
            
            insights['flow_assessment'] = flow_assessment
            insights['flow_correlation'] = float(flow_correlation)
        
        # Generate recommendations
        recommendations = []
        
        if insights.get('average_similarity', 0) < 0.5:
            recommendations.append("Low similarity suggests significant differences in representations")
        
        if insights.get('flow_correlation', 0) < 0.3:
            recommendations.append("Different flow patterns may indicate architectural differences")
        
        if len(insights.get('dimensionality_insights', [])) > 0:
            recommendations.append("Dimensionality differences detected - investigate layer capacities")
        
        insights['recommendations'] = recommendations
        
        return insights
    
    def _format_outputs(self, comparison_result: ResidualStreamComparison,
                       tl_data: Dict[str, Any], neuronmap_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format comparison results for output."""
        
        return {
            'comparison_summary': {
                'similarity_scores': comparison_result.similarity_scores,
                'combined_insights': comparison_result.combined_insights
            },
            'detailed_analysis': {
                'dimensionality_analysis': comparison_result.dimensionality_analysis,
                'flow_analysis': comparison_result.flow_analysis,
                'layer_wise_comparison': comparison_result.layer_wise_comparison
            },
            'metadata': {
                'tl_model_name': tl_data.get('model_name', 'unknown'),
                'nm_model_info': neuronmap_data.get('model_info', {}),
                'comparison_config': {
                    'similarity_metrics': self.similarity_metrics,
                    'dimensionality_reduction': self.dimensionality_reduction,
                    'n_components': self.n_components
                },
                'layers_compared': len(comparison_result.layer_wise_comparison),
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
    
    def analyze_model(self, input_data: Any, analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """Interface method for MechanisticPluginBase."""
        tl_data = analysis_config.get('tl_data', {})
        neuronmap_data = analysis_config.get('neuronmap_data', {})
        comparison_type = analysis_config.get('comparison_type', 'full')
        
        result = self.execute(tl_data, neuronmap_data, comparison_type)
        return result.outputs if result.success else {}
    
    def extract_mechanisms(self, model_data: Any) -> Dict[str, Any]:
        """Extract mechanistic insights from comparison data."""
        if not isinstance(model_data, dict):
            return {}
        
        mechanisms = {}
        
        if 'comparison_summary' in model_data:
            summary = model_data['comparison_summary']
            mechanisms['residual_stream_mechanisms'] = {
                'similarity_patterns': summary.get('similarity_scores', {}),
                'information_flow': summary.get('combined_insights', {})
            }
        
        return mechanisms
    
    def validate_output(self, output: Any) -> bool:
        """Validate residual stream comparison output."""
        if not isinstance(output, dict):
            return False
        
        required_keys = ['comparison_summary', 'detailed_analysis', 'metadata']
        for key in required_keys:
            if key not in output:
                logger.error(f"Missing required output key: {key}")
                return False
        
        return True

def create_residual_stream_comparator(config: Optional[Dict[str, Any]] = None) -> ResidualStreamComparator:
    """Factory function to create residual stream comparator."""
    return ResidualStreamComparator(config=config)
