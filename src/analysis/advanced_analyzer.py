"""
Advanced Analysis Methods for NeuronMap
======================================

This module provides sophisticated analysis techniques for neural network activations.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)

# Conditional imports
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import PCA, FastICA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Advanced analysis methods will be limited.")

try:
    import scipy.stats as stats
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import dendrogram, linkage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available. Statistical analysis will be limited.")


class AdvancedAnalyzer:
    """Advanced analysis methods for neural network activations."""

    def __init__(self, config=None):
        self.config = config
        if config and hasattr(config, 'data') and hasattr(config.data, 'outputs_dir'):
            self.output_dir = Path(config.data.outputs_dir) / "advanced_analysis"
        else:
            self.output_dir = Path("outputs") / "advanced_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def cluster_activations(self, activations: np.ndarray,
                          method: str = "kmeans",
                          n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """Cluster activation patterns."""
        if not SKLEARN_AVAILABLE:
            logger.error("Scikit-learn required for clustering")
            return {}

        if activations.ndim != 2:
            logger.error(f"Expected 2D array, got {activations.ndim}D")
            return {}

        # Standardize features
        scaler = StandardScaler()
        scaled_activations = scaler.fit_transform(activations)

        results = {
            'method': method,
            'n_samples': activations.shape[0],
            'n_features': activations.shape[1]
        }

        try:
            if method.lower() == "kmeans":
                # Determine optimal number of clusters if not specified
                if n_clusters is None:
                    n_clusters = self._find_optimal_clusters(scaled_activations, max_k=min(10, activations.shape[0]//2))

                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = clusterer.fit_predict(scaled_activations)

                results.update({
                    'n_clusters': n_clusters,
                    'labels': labels.tolist(),
                    'cluster_centers': clusterer.cluster_centers_.tolist(),
                    'inertia': clusterer.inertia_
                })

            elif method.lower() == "dbscan":
                # Use adaptive eps based on data scale
                eps = self._estimate_dbscan_eps(scaled_activations)
                clusterer = DBSCAN(eps=eps, min_samples=max(2, activations.shape[0]//10))
                labels = clusterer.fit_predict(scaled_activations)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise_points = list(labels).count(-1)

                results.update({
                    'n_clusters': n_clusters,
                    'n_noise_points': noise_points,
                    'labels': labels.tolist(),
                    'eps': eps
                })

            elif method.lower() == "hierarchical":
                if n_clusters is None:
                    n_clusters = min(5, activations.shape[0]//2)

                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                labels = clusterer.fit_predict(scaled_activations)

                results.update({
                    'n_clusters': n_clusters,
                    'labels': labels.tolist()
                })

            # Compute clustering metrics
            if len(set(labels)) > 1:  # Need at least 2 clusters for metrics
                valid_indices = labels != -1  # Exclude noise points for DBSCAN
                if np.sum(valid_indices) > 0:
                    try:
                        silhouette = silhouette_score(scaled_activations[valid_indices],
                                                    labels[valid_indices])
                        results['silhouette_score'] = silhouette
                    except:
                        pass

                    try:
                        calinski_harabasz = calinski_harabasz_score(scaled_activations[valid_indices],
                                                                  labels[valid_indices])
                        results['calinski_harabasz_score'] = calinski_harabasz
                    except:
                        pass

            logger.info(f"Clustering completed: {method}, {results.get('n_clusters', 0)} clusters")

        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            results['error'] = str(e)

        return results

    def _find_optimal_clusters(self, data: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method."""
        if max_k < 2:
            return 2

        inertias = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
            except:
                break

        if len(inertias) < 2:
            return 2

        # Find elbow using second derivative
        if len(inertias) >= 3:
            second_derivatives = []
            for i in range(1, len(inertias) - 1):
                second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
                second_derivatives.append(second_deriv)

            if second_derivatives:
                elbow_idx = np.argmax(second_derivatives) + 1
                return list(k_range)[elbow_idx]

        # Fallback: choose k with largest drop in inertia
        drops = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        return list(k_range)[np.argmax(drops)]

    def _estimate_dbscan_eps(self, data: np.ndarray) -> float:
        """Estimate eps parameter for DBSCAN."""
        from sklearn.neighbors import NearestNeighbors

        k = min(4, data.shape[0] - 1)  # k=4 is common choice
        if k < 1:
            return 0.5

        nbrs = NearestNeighbors(n_neighbors=k).fit(data)
        distances, indices = nbrs.kneighbors(data)
        distances = np.sort(distances[:, k-1], axis=0)

        # Use knee detection or percentile
        return np.percentile(distances, 80)  # 80th percentile

    def dimensionality_analysis(self, activations: np.ndarray) -> Dict[str, Any]:
        """Analyze dimensionality characteristics of activations."""
        if not SKLEARN_AVAILABLE:
            logger.error("Scikit-learn required for dimensionality analysis")
            return {}

        results = {
            'original_shape': activations.shape,
            'n_samples': activations.shape[0],
            'n_features': activations.shape[1] if activations.ndim > 1 else 1
        }

        if activations.ndim != 2 or activations.shape[0] < 2:
            logger.warning("Insufficient data for dimensionality analysis")
            return results

        try:
            # PCA Analysis
            pca = PCA()
            pca_transformed = pca.fit_transform(activations)

            # Find effective dimensionality (95% variance explained)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            effective_dim = np.argmax(cumvar >= 0.95) + 1

            results.update({
                'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'pca_cumulative_variance': cumvar.tolist(),
                'effective_dimensionality_95': int(effective_dim),
                'effective_dimensionality_99': int(np.argmax(cumvar >= 0.99) + 1),
                'intrinsic_dimensionality': self._estimate_intrinsic_dimension(activations)
            })

            # Sparsity analysis
            sparsity_threshold = 0.01 * np.std(activations)
            sparsity = np.mean(np.abs(activations) < sparsity_threshold)

            results.update({
                'sparsity': float(sparsity),
                'mean_activation': float(np.mean(activations)),
                'std_activation': float(np.std(activations)),
                'activation_range': [float(np.min(activations)), float(np.max(activations))]
            })

            logger.info(f"Dimensionality analysis completed: effective dim = {effective_dim}")

        except Exception as e:
            logger.error(f"Dimensionality analysis failed: {e}")
            results['error'] = str(e)

        return results

    def _estimate_intrinsic_dimension(self, data: np.ndarray) -> float:
        """Estimate intrinsic dimensionality using correlation dimension."""
        if not SCIPY_AVAILABLE:
            return -1.0

        try:
            # Use subset for computational efficiency
            max_samples = min(500, data.shape[0])
            if data.shape[0] > max_samples:
                indices = np.random.choice(data.shape[0], max_samples, replace=False)
                data_subset = data[indices]
            else:
                data_subset = data

            # Compute pairwise distances
            distances = pdist(data_subset)

            # Use correlation dimension estimation
            r_values = np.logspace(-3, 0, 20) * np.median(distances)
            correlations = []

            for r in r_values:
                count = np.sum(distances < r)
                correlation = count / len(distances)
                correlations.append(max(correlation, 1e-10))  # Avoid log(0)

            # Fit line to log-log plot
            log_r = np.log(r_values)
            log_c = np.log(correlations)

            # Use middle section of the curve
            valid_indices = (np.array(correlations) > 0.01) & (np.array(correlations) < 0.99)
            if np.sum(valid_indices) > 3:
                slope, _ = np.polyfit(log_r[valid_indices], log_c[valid_indices], 1)
                return max(0.0, slope)  # Slope is the dimension estimate

            return -1.0  # Could not estimate

        except Exception as e:
            logger.warning(f"Intrinsic dimension estimation failed: {e}")
            return -1.0

    def compute_activation_statistics(self, activations: np.ndarray) -> Dict[str, Any]:
        """Compute basic activation statistics for test compatibility."""
        flat_activations = activations.flatten()

        try:
            from scipy import stats
            scipy_available = True
        except ImportError:
            scipy_available = False

        # Basic statistics
        stats_dict = {
            'mean': float(np.mean(flat_activations)),
            'std': float(np.std(flat_activations)),
            'min': float(np.min(flat_activations)),
            'max': float(np.max(flat_activations)),
            'sparsity': float(np.mean(flat_activations == 0))
        }

        # Add scipy-based statistics if available
        if scipy_available:
            stats_dict.update({
                'skewness': float(stats.skew(flat_activations)),
                'kurtosis': float(stats.kurtosis(flat_activations))
            })
        else:
            stats_dict.update({
                'skewness': 0.0,
                'kurtosis': 0.0
            })

        return stats_dict

    def statistical_analysis(self, activations: np.ndarray,
                           layer_name: str = "unknown") -> Dict[str, Any]:
        """Comprehensive statistical analysis of activations."""
        results = {
            'layer_name': layer_name,
            'shape': activations.shape
        }

        try:
            # Basic statistics
            results.update({
                'mean': float(np.mean(activations)),
                'std': float(np.std(activations)),
                'var': float(np.var(activations)),
                'min': float(np.min(activations)),
                'max': float(np.max(activations)),
                'median': float(np.median(activations)),
                'skewness': float(stats.skew(activations.flatten())) if SCIPY_AVAILABLE else None,
                'kurtosis': float(stats.kurtosis(activations.flatten())) if SCIPY_AVAILABLE else None
            })

            # Percentiles
            percentiles = [1, 5, 10, 25, 75, 90, 95, 99]
            percentile_values = np.percentile(activations, percentiles)
            results['percentiles'] = {f'p{p}': float(v) for p, v in zip(percentiles, percentile_values)}

            # Distribution analysis
            flat_activations = activations.flatten()

            # Test for normality
            if SCIPY_AVAILABLE and len(flat_activations) > 3:
                try:
                    if len(flat_activations) <= 5000:  # Shapiro-Wilk test
                        shapiro_stat, shapiro_p = stats.shapiro(flat_activations)
                        results['normality_test'] = {
                            'test': 'shapiro_wilk',
                            'statistic': float(shapiro_stat),
                            'p_value': float(shapiro_p),
                            'is_normal': shapiro_p > 0.05
                        }
                    else:  # Kolmogorov-Smirnov test for larger samples
                        ks_stat, ks_p = stats.kstest(flat_activations, 'norm',
                                                    args=(np.mean(flat_activations), np.std(flat_activations)))
                        results['normality_test'] = {
                            'test': 'kolmogorov_smirnov',
                            'statistic': float(ks_stat),
                            'p_value': float(ks_p),
                            'is_normal': ks_p > 0.05
                        }
                except:
                    pass

            # Activation patterns
            if activations.ndim == 2:
                # Per-neuron statistics
                neuron_means = np.mean(activations, axis=0)
                neuron_stds = np.std(activations, axis=0)

                results.update({
                    'neuron_statistics': {
                        'mean_neuron_activation': float(np.mean(neuron_means)),
                        'std_neuron_activation': float(np.std(neuron_means)),
                        'mean_neuron_variability': float(np.mean(neuron_stds)),
                        'most_active_neuron_idx': int(np.argmax(neuron_means)),
                        'most_variable_neuron_idx': int(np.argmax(neuron_stds))
                    }
                })

                # Sample correlations
                if activations.shape[0] > 1:
                    sample_correlations = np.corrcoef(activations)
                    results['sample_correlations'] = {
                        'mean_correlation': float(np.mean(sample_correlations[np.triu_indices_from(sample_correlations, k=1)])),
                        'max_correlation': float(np.max(sample_correlations[np.triu_indices_from(sample_correlations, k=1)])),
                        'min_correlation': float(np.min(sample_correlations[np.triu_indices_from(sample_correlations, k=1)]))
                    }

            logger.info(f"Statistical analysis completed for {layer_name}")

        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            results['error'] = str(e)

        return results

    def activation_similarity_analysis(self, activations_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze similarities between different activation sets."""
        if not SKLEARN_AVAILABLE or not SCIPY_AVAILABLE:
            logger.error("Scikit-learn and SciPy required for similarity analysis")
            return {}

        layer_names = list(activations_dict.keys())
        if len(layer_names) < 2:
            logger.warning("Need at least 2 activation sets for similarity analysis")
            return {}

        results = {
            'n_layers': len(layer_names),
            'layer_names': layer_names,
            'pairwise_similarities': {}
        }

        try:
            from sklearn.metrics.pairwise import cosine_similarity

            # Compute all pairwise similarities
            for i, layer1 in enumerate(layer_names):
                for j, layer2 in enumerate(layer_names):
                    if i < j:  # Avoid duplicates
                        act1 = activations_dict[layer1]
                        act2 = activations_dict[layer2]

                        # Ensure same number of samples
                        min_samples = min(act1.shape[0], act2.shape[0])
                        act1_subset = act1[:min_samples]
                        act2_subset = act2[:min_samples]

                        # Compute similarity metrics
                        similarities = {}

                        # Cosine similarity
                        cos_sims = []
                        for k in range(min_samples):
                            cos_sim = cosine_similarity([act1_subset[k].flatten()],
                                                      [act2_subset[k].flatten()])[0][0]
                            cos_sims.append(cos_sim)

                        similarities['cosine_similarity'] = {
                            'mean': float(np.mean(cos_sims)),
                            'std': float(np.std(cos_sims)),
                            'min': float(np.min(cos_sims)),
                            'max': float(np.max(cos_sims))
                        }

                        # Correlation
                        correlations = []
                        for k in range(min_samples):
                            corr = np.corrcoef(act1_subset[k].flatten(),
                                             act2_subset[k].flatten())[0, 1]
                            if not np.isnan(corr):
                                correlations.append(corr)

                        if correlations:
                            similarities['correlation'] = {
                                'mean': float(np.mean(correlations)),
                                'std': float(np.std(correlations)),
                                'min': float(np.min(correlations)),
                                'max': float(np.max(correlations))
                            }

                        results['pairwise_similarities'][f"{layer1}_vs_{layer2}"] = similarities

            logger.info(f"Similarity analysis completed for {len(layer_names)} layers")

        except Exception as e:
            logger.error(f"Similarity analysis failed: {e}")
            results['error'] = str(e)

        return results

    def perform_clustering_analysis(self, activations: np.ndarray, n_clusters: int = 5,
                                  algorithm: str = 'kmeans', methods: List[str] = None) -> Dict[str, Any]:
        """Perform clustering analysis with multiple methods."""
        if methods is None:
            methods = [algorithm]

        results = {}
        for method in methods:
            if method.lower() in ['kmeans', 'k-means']:
                cluster_result = self.cluster_activations(activations, method='kmeans', n_clusters=n_clusters)
                results[method] = cluster_result
            elif method.lower() in ['hierarchical', 'agglomerative']:
                cluster_result = self.cluster_activations(activations, method='hierarchical', n_clusters=n_clusters)
                results[method] = cluster_result

        # Return all method results for compatibility
        return results

    def perform_pca_analysis(self, activations: np.ndarray, n_components: int = 10) -> Dict[str, Any]:
        """Perform PCA analysis on activation patterns."""
        if not SKLEARN_AVAILABLE:
            logger.error("Scikit-learn required for PCA analysis")
            return {}

        try:
            # Ensure we don't request more components than features
            n_features = activations.shape[1] if activations.ndim > 1 else 1
            n_components = min(n_components, n_features, activations.shape[0])

            pca = PCA(n_components=n_components)
            transformed_data = pca.fit_transform(activations)

            return {
                'components': pca.components_.tolist(),
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'transformed_data': transformed_data,  # Keep as numpy array
                'singular_values': pca.singular_values_.tolist(),
                'mean': pca.mean_.tolist(),
                'n_components': n_components
            }
        except Exception as e:
            logger.error(f"PCA analysis failed: {e}")
            return {}

    def analyze_layer_correlations(self, activations_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze correlations between different layers."""
        try:
            layer_names = list(activations_dict.keys())
            n_layers = len(layer_names)

            # Compute correlation matrix between layers
            correlation_matrix = np.zeros((n_layers, n_layers))

            for i, layer1 in enumerate(layer_names):
                for j, layer2 in enumerate(layer_names):
                    if i <= j:  # Only compute upper triangle
                        act1 = activations_dict[layer1].flatten()
                        act2 = activations_dict[layer2].flatten()

                        # Ensure same length
                        min_len = min(len(act1), len(act2))
                        corr_coef = np.corrcoef(act1[:min_len], act2[:min_len])[0, 1]
                        correlation_matrix[i, j] = corr_coef
                        correlation_matrix[j, i] = corr_coef  # Make symmetric

            # Find highly correlated pairs
            highly_correlated_pairs = []
            threshold = 0.7
            for i in range(n_layers):
                for j in range(i+1, n_layers):
                    if abs(correlation_matrix[i, j]) > threshold:
                        highly_correlated_pairs.append({
                            'layer1': layer_names[i],
                            'layer2': layer_names[j],
                            'correlation': float(correlation_matrix[i, j])
                        })

            return {
                'correlation_matrix': correlation_matrix.tolist(),
                'layer_names': layer_names,
                'highly_correlated_pairs': highly_correlated_pairs,
                'average_correlation': float(np.mean(correlation_matrix[np.triu_indices(n_layers, k=1)]))
            }

        except Exception as e:
            logger.error(f"Layer correlation analysis failed: {e}")
            return {}

    def generate_advanced_report(self, analyses: Dict[str, Any]) -> None:
        """Generate comprehensive advanced analysis report."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"advanced_analysis_report_{timestamp}.json"

        # Save detailed JSON report
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(analyses, f, indent=2, ensure_ascii=False)

        # Generate text summary
        summary_file = self.output_dir / f"advanced_analysis_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Advanced NeuronMap Analysis Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Clustering results
            if 'clustering' in analyses:
                f.write("Clustering Analysis:\n")
                f.write("-" * 20 + "\n")
                clustering = analyses['clustering']
                f.write(f"Method: {clustering.get('method', 'unknown')}\n")
                f.write(f"Clusters found: {clustering.get('n_clusters', 'N/A')}\n")
                if 'silhouette_score' in clustering:
                    f.write(f"Silhouette score: {clustering['silhouette_score']:.3f}\n")
                f.write("\n")

            # Dimensionality analysis
            if 'dimensionality' in analyses:
                f.write("Dimensionality Analysis:\n")
                f.write("-" * 25 + "\n")
                dim = analyses['dimensionality']
                f.write(f"Original dimensions: {dim.get('n_features', 'N/A')}\n")
                f.write(f"Effective dim (95% var): {dim.get('effective_dimensionality_95', 'N/A')}\n")
                f.write(f"Intrinsic dimension: {dim.get('intrinsic_dimensionality', 'N/A'):.2f}\n")
                f.write(f"Sparsity: {dim.get('sparsity', 0):.2%}\n")
                f.write("\n")

            # Statistical analysis
            if 'statistics' in analyses:
                f.write("Statistical Analysis:\n")
                f.write("-" * 20 + "\n")
                stats = analyses['statistics']
                f.write(f"Mean: {stats.get('mean', 0):.4f}\n")
                f.write(f"Std: {stats.get('std', 0):.4f}\n")
                f.write(f"Skewness: {stats.get('skewness', 'N/A')}\n")
                f.write(f"Kurtosis: {stats.get('kurtosis', 'N/A')}\n")
                if 'normality_test' in stats:
                    norm_test = stats['normality_test']
                    f.write(f"Normal distribution: {norm_test.get('is_normal', 'N/A')}\n")
                f.write("\n")

        logger.info(f"Advanced analysis report saved to {report_file}")
        logger.info(f"Summary report saved to {summary_file}")


def run_advanced_analysis_demo():
    """Demonstration of advanced analysis capabilities."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from utils.config_manager import get_config
    from analysis.activation_analyzer import ActivationAnalyzer
    from data_processing.question_loader import QuestionLoader

    # Load config and data
    config = get_config()
    analyzer = ActivationAnalyzer(config)
    question_loader = QuestionLoader(config)

    # Load sample questions
    questions = question_loader.load_questions()[:5]  # Limit for demo

    print("Running advanced analysis demo...")

    # Get activations
    results = analyzer.analyze_questions(questions)

    # Extract activation arrays
    layer_activations = {}
    for result in results:
        if result.get('success', False):
            for layer_name, activation in result.get('activations', {}).items():
                if layer_name not in layer_activations:
                    layer_activations[layer_name] = []
                layer_activations[layer_name].append(activation)

    # Convert to numpy arrays
    for layer_name in layer_activations:
        layer_activations[layer_name] = np.array(layer_activations[layer_name])

    if not layer_activations:
        print("No activations found for analysis")
        return

    # Initialize advanced analyzer
    advanced_analyzer = AdvancedAnalyzer(config)

    # Run analyses
    analyses = {}

    # Use first layer for individual analyses
    first_layer = list(layer_activations.keys())[0]
    first_activations = layer_activations[first_layer]

    print(f"Analyzing layer: {first_layer}")
    print(f"Activation shape: {first_activations.shape}")

    # Clustering
    print("Running clustering analysis...")
    analyses['clustering'] = advanced_analyzer.cluster_activations(first_activations)

    # Dimensionality analysis
    print("Running dimensionality analysis...")
    analyses['dimensionality'] = advanced_analyzer.dimensionality_analysis(first_activations)

    # Statistical analysis
    print("Running statistical analysis...")
    analyses['statistics'] = advanced_analyzer.statistical_analysis(first_activations, first_layer)

    # Similarity analysis (if multiple layers)
    if len(layer_activations) > 1:
        print("Running similarity analysis...")
        analyses['similarity'] = advanced_analyzer.activation_similarity_analysis(layer_activations)

    # Generate report
    advanced_analyzer.generate_advanced_report(analyses)

    print("Advanced analysis completed!")
    print(f"Results saved to: {advanced_analyzer.output_dir}")

    return analyses


if __name__ == "__main__":
    run_advanced_analysis_demo()
