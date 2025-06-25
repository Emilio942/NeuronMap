"""
Statistical Analysis Plugin
==========================

Provides statistical analysis of neural network activations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import sys

# Add src directory to path
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from core.plugin_system import AnalysisPlugin, PluginMetadata

class StatisticalAnalysisPlugin(AnalysisPlugin):
    """Statistical analysis plugin for activation patterns."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Statistical Analysis",
            version="1.0.0",
            author="NeuronMap Team",
            description="Comprehensive statistical analysis of neural activations",
            plugin_type="analysis",
            dependencies=["numpy", "pandas", "scipy", "scikit-learn"],
            tags=["statistics", "analysis", "distributions"],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'data/outputs/plugins/statistical'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return True
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute statistical analysis."""
        return self.analyze(*args, **kwargs)
    
    def analyze(self, activations: Dict[str, np.ndarray], **kwargs) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        results = {
            'layer_statistics': {},
            'distribution_analysis': {},
            'correlation_analysis': {},
            'dimensionality_analysis': {},
            'outlier_analysis': {},
            'summary_report': {}
        }
        
        try:
            for layer_name, activation_data in activations.items():
                if activation_data is None or len(activation_data) == 0:
                    continue
                
                # Flatten activation data if needed
                flat_data = activation_data.flatten() if activation_data.ndim > 1 else activation_data
                
                # Basic statistics
                layer_stats = {
                    'mean': float(np.mean(flat_data)),
                    'std': float(np.std(flat_data)),
                    'min': float(np.min(flat_data)),
                    'max': float(np.max(flat_data)),
                    'median': float(np.median(flat_data)),
                    'q25': float(np.percentile(flat_data, 25)),
                    'q75': float(np.percentile(flat_data, 75)),
                    'skewness': float(stats.skew(flat_data)),
                    'kurtosis': float(stats.kurtosis(flat_data)),
                    'variance': float(np.var(flat_data)),
                    'range': float(np.max(flat_data) - np.min(flat_data))
                }
                results['layer_statistics'][layer_name] = layer_stats
                
                # Distribution analysis
                # Test for normality
                _, p_value_shapiro = stats.shapiro(flat_data[:5000])  # Sample for large datasets
                _, p_value_ks = stats.kstest(flat_data, 'norm')
                
                results['distribution_analysis'][layer_name] = {
                    'shapiro_wilk_p': float(p_value_shapiro),
                    'kolmogorov_smirnov_p': float(p_value_ks),
                    'is_normal': bool(p_value_shapiro > 0.05),
                    'histogram_bins': 50
                }
                
                # Outlier detection using IQR method
                q1, q3 = np.percentile(flat_data, [25, 75])
                iqr = q3 - q1
                outlier_threshold = 1.5 * iqr
                outliers = flat_data[(flat_data < q1 - outlier_threshold) | 
                                   (flat_data > q3 + outlier_threshold)]
                
                results['outlier_analysis'][layer_name] = {
                    'outlier_count': int(len(outliers)),
                    'outlier_percentage': float(len(outliers) / len(flat_data) * 100),
                    'outlier_threshold': float(outlier_threshold)
                }
            
            # Cross-layer correlation analysis
            if len(activations) > 1:
                correlation_matrix = self._compute_cross_layer_correlations(activations)
                results['correlation_analysis'] = {
                    'correlation_matrix': correlation_matrix.tolist() if correlation_matrix is not None else None,
                    'layer_names': list(activations.keys())
                }
            
            # Dimensionality analysis
            results['dimensionality_analysis'] = self._analyze_dimensionality(activations)
            
            # Generate summary report
            results['summary_report'] = self._generate_summary_report(results)
            
            # Save detailed results
            self._save_results(results)
            
            return results
            
        except Exception as e:
            return {'error': f"Statistical analysis failed: {str(e)}"}
    
    def _compute_cross_layer_correlations(self, activations: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute correlations between layers."""
        try:
            layer_means = []
            layer_names = []
            
            for layer_name, activation_data in activations.items():
                if activation_data is not None and len(activation_data) > 0:
                    # Compute mean activation per sample
                    if activation_data.ndim > 1:
                        mean_activation = np.mean(activation_data, axis=tuple(range(1, activation_data.ndim)))
                    else:
                        mean_activation = activation_data
                    layer_means.append(mean_activation)
                    layer_names.append(layer_name)
            
            if len(layer_means) > 1:
                # Ensure all arrays have the same length
                min_length = min(len(arr) for arr in layer_means)
                layer_means = [arr[:min_length] for arr in layer_means]
                
                correlation_matrix = np.corrcoef(layer_means)
                return correlation_matrix
            
            return None
        except Exception:
            return None
    
    def _analyze_dimensionality(self, activations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze the effective dimensionality of activations."""
        dimensionality_results = {}
        
        for layer_name, activation_data in activations.items():
            if activation_data is None or len(activation_data) == 0:
                continue
            
            try:
                # Reshape for PCA
                if activation_data.ndim > 2:
                    # Flatten spatial dimensions, keep batch dimension
                    reshaped_data = activation_data.reshape(activation_data.shape[0], -1)
                else:
                    reshaped_data = activation_data
                
                # Skip if not enough samples or features
                if reshaped_data.shape[0] < 2 or reshaped_data.shape[1] < 2:
                    continue
                
                # PCA analysis
                n_components = min(10, reshaped_data.shape[0] - 1, reshaped_data.shape[1])
                pca = PCA(n_components=n_components)
                pca.fit(reshaped_data)
                
                # Calculate effective dimensionality (95% variance)
                cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
                effective_dim = np.argmax(cumsum_variance >= 0.95) + 1
                
                dimensionality_results[layer_name] = {
                    'total_dimensions': int(reshaped_data.shape[1]),
                    'effective_dimensions_95': int(effective_dim),
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                    'cumulative_variance_ratio': cumsum_variance.tolist()
                }
                
            except Exception as e:
                dimensionality_results[layer_name] = {'error': str(e)}
        
        return dimensionality_results
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary report of key findings."""
        summary = {
            'total_layers_analyzed': len(results['layer_statistics']),
            'key_insights': [],
            'recommendations': []
        }
        
        # Analyze patterns across layers
        if results['layer_statistics']:
            # Find layers with highest/lowest activation
            layer_means = {layer: stats['mean'] for layer, stats in results['layer_statistics'].items()}
            highest_activation_layer = max(layer_means, key=layer_means.get)
            lowest_activation_layer = min(layer_means, key=layer_means.get)
            
            summary['key_insights'].extend([
                f"Layer '{highest_activation_layer}' shows highest mean activation: {layer_means[highest_activation_layer]:.4f}",
                f"Layer '{lowest_activation_layer}' shows lowest mean activation: {layer_means[lowest_activation_layer]:.4f}"
            ])
            
            # Check for distribution normality
            normal_layers = []
            for layer, dist_info in results['distribution_analysis'].items():
                if dist_info.get('is_normal', False):
                    normal_layers.append(layer)
            
            if normal_layers:
                summary['key_insights'].append(f"{len(normal_layers)} layers show approximately normal distributions")
            
            # Outlier analysis
            high_outlier_layers = []
            for layer, outlier_info in results['outlier_analysis'].items():
                if outlier_info.get('outlier_percentage', 0) > 5:  # More than 5% outliers
                    high_outlier_layers.append(layer)
            
            if high_outlier_layers:
                summary['recommendations'].append(f"Consider investigating layers with high outlier rates: {', '.join(high_outlier_layers)}")
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]):
        """Save analysis results to files."""
        try:
            # Save as JSON
            import json
            with open(self.output_dir / 'statistical_analysis.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save summary as text
            with open(self.output_dir / 'analysis_summary.txt', 'w') as f:
                f.write("Statistical Analysis Summary\\n")
                f.write("=" * 50 + "\\n\\n")
                
                summary = results.get('summary_report', {})
                f.write(f"Total layers analyzed: {summary.get('total_layers_analyzed', 0)}\\n\\n")
                
                if summary.get('key_insights'):
                    f.write("Key Insights:\\n")
                    for insight in summary['key_insights']:
                        f.write(f"• {insight}\\n")
                    f.write("\\n")
                
                if summary.get('recommendations'):
                    f.write("Recommendations:\\n")
                    for rec in summary['recommendations']:
                        f.write(f"• {rec}\\n")
                    f.write("\\n")
                
                # Layer statistics
                if results.get('layer_statistics'):
                    f.write("Layer Statistics:\\n")
                    f.write("-" * 20 + "\\n")
                    for layer, stats in results['layer_statistics'].items():
                        f.write(f"\\n{layer}:\\n")
                        f.write(f"  Mean: {stats['mean']:.4f}\\n")
                        f.write(f"  Std:  {stats['std']:.4f}\\n")
                        f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\\n")
                        f.write(f"  Skewness: {stats['skewness']:.4f}\\n")
                        f.write(f"  Kurtosis: {stats['kurtosis']:.4f}\\n")
        
        except Exception as e:
            print(f"Error saving statistical analysis results: {e}")
