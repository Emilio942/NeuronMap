"""Advanced analysis methods for neural network activation patterns."""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
import logging
import json
from pathlib import Path
import h5py

from ..utils.config import get_config


logger = logging.getLogger(__name__)


class ActivationAnalyzer:
    """Advanced analysis of neural network activation patterns."""

    def __init__(self, config_name: str = "default"):
        """Initialize analyzer with configuration.

        Args:
            config_name: Name of experiment configuration.
        """
        self.config = get_config()
        self.experiment_config = self.config.get_experiment_config(config_name)

    def load_activations_hdf5(self, filepath: str) -> Dict[str, Any]:
        """Load activation data from HDF5 file.

        Args:
            filepath: Path to HDF5 file with activations.

        Returns:
            Dictionary with loaded activation data.
        """
        data = {
            'questions': [],
            'activations': {},
            'metadata': {}
        }

        with h5py.File(filepath, 'r') as f:
            # Load metadata
            data['metadata'] = dict(f.attrs)

            # Load questions
            questions_group = f['questions']
            for key in sorted(questions_group.keys(), key=lambda x: int(x.split('_')[1])):
                data['questions'].append(questions_group[key][()].decode('utf-8'))

            # Load activations
            activations_group = f['activations']
            for question_key in sorted(activations_group.keys(), key=lambda x: int(x.split('_')[1])):
                question_idx = int(question_key.split('_')[1])
                question_group = activations_group[question_key]

                for layer_key in question_group.keys():
                    layer_name = layer_key.replace('_', '.')

                    if layer_name not in data['activations']:
                        data['activations'][layer_name] = []

                    layer_group = question_group[layer_key]
                    activation_vector = layer_group['vector'][()]
                    stats = json.loads(layer_group.attrs['stats'])

                    data['activations'][layer_name].append({
                        'question_idx': question_idx,
                        'vector': activation_vector,
                        'stats': stats
                    })

        logger.info(f"Loaded activations for {len(data['questions'])} questions, "
                   f"{len(data['activations'])} layers")
        return data

    def compute_activation_statistics(self, activations: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive statistics for activation vectors.

        Args:
            activations: Array of activation vectors (n_samples, n_features).

        Returns:
            Dictionary with statistical measures.
        """
        stats = {
            # Basic statistics
            'mean': float(np.mean(activations)),
            'std': float(np.std(activations)),
            'min': float(np.min(activations)),
            'max': float(np.max(activations)),
            'median': float(np.median(activations)),

            # Distribution characteristics
            'skewness': float(self._compute_skewness(activations)),
            'kurtosis': float(self._compute_kurtosis(activations)),
            'sparsity': float(np.mean(activations == 0)),
            'density': float(np.mean(activations != 0)),

            # Activation patterns
            'positive_ratio': float(np.mean(activations > 0)),
            'negative_ratio': float(np.mean(activations < 0)),
            'high_activation_ratio': float(np.mean(activations > np.percentile(activations, 90))),

            # Variability measures
            'coefficient_of_variation': float(np.std(activations) / (np.mean(activations) + 1e-8)),
            'interquartile_range': float(np.percentile(activations, 75) - np.percentile(activations, 25)),
        }

        return stats

    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

    def analyze_layer_activations(self, data: Dict[str, Any], layer_name: str) -> Dict[str, Any]:
        """Analyze activations from a specific layer.

        Args:
            data: Loaded activation data.
            layer_name: Name of layer to analyze.

        Returns:
            Analysis results for the layer.
        """
        if layer_name not in data['activations']:
            raise ValueError(f"Layer {layer_name} not found in data")

        layer_data = data['activations'][layer_name]

        # Extract activation vectors
        activation_matrix = np.array([item['vector'] for item in layer_data])

        # Compute statistics
        overall_stats = self.compute_activation_statistics(activation_matrix)

        # Per-question statistics
        per_question_stats = []
        for i, item in enumerate(layer_data):
            question_stats = self.compute_activation_statistics(np.array(item['vector']))
            question_stats['question_idx'] = item['question_idx']
            question_stats['question'] = data['questions'][item['question_idx']]
            per_question_stats.append(question_stats)

        # Neuron-wise analysis
        neuron_stats = self._analyze_neurons(activation_matrix)

        # Correlation analysis
        correlation_analysis = self._analyze_correlations(activation_matrix)

        return {
            'layer_name': layer_name,
            'overall_statistics': overall_stats,
            'per_question_statistics': per_question_stats,
            'neuron_analysis': neuron_stats,
            'correlation_analysis': correlation_analysis,
            'activation_matrix_shape': activation_matrix.shape
        }

    def _analyze_neurons(self, activation_matrix: np.ndarray) -> Dict[str, Any]:
        """Analyze individual neurons across all questions.

        Args:
            activation_matrix: Matrix of activations (n_questions, n_neurons).

        Returns:
            Neuron-wise analysis results.
        """
        n_questions, n_neurons = activation_matrix.shape

        neuron_stats = {
            'total_neurons': n_neurons,
            'most_active_neurons': [],
            'most_variable_neurons': [],
            'sparsest_neurons': [],
            'neuron_statistics': []
        }

        # Analyze each neuron
        for neuron_idx in range(n_neurons):
            neuron_activations = activation_matrix[:, neuron_idx]

            stats = self.compute_activation_statistics(neuron_activations)
            stats['neuron_idx'] = neuron_idx
            neuron_stats['neuron_statistics'].append(stats)

        # Find most active neurons (highest mean activation)
        neuron_means = [stats['mean'] for stats in neuron_stats['neuron_statistics']]
        most_active_indices = np.argsort(neuron_means)[-10:]  # Top 10
        neuron_stats['most_active_neurons'] = [
            {'neuron_idx': int(idx), 'mean_activation': neuron_means[idx]}
            for idx in most_active_indices
        ]

        # Find most variable neurons (highest std)
        neuron_stds = [stats['std'] for stats in neuron_stats['neuron_statistics']]
        most_variable_indices = np.argsort(neuron_stds)[-10:]  # Top 10
        neuron_stats['most_variable_neurons'] = [
            {'neuron_idx': int(idx), 'std_activation': neuron_stds[idx]}
            for idx in most_variable_indices
        ]

        # Find sparsest neurons (highest sparsity)
        neuron_sparsities = [stats['sparsity'] for stats in neuron_stats['neuron_statistics']]
        sparsest_indices = np.argsort(neuron_sparsities)[-10:]  # Top 10
        neuron_stats['sparsest_neurons'] = [
            {'neuron_idx': int(idx), 'sparsity': neuron_sparsities[idx]}
            for idx in sparsest_indices
        ]

        return neuron_stats

    def _analyze_correlations(self, activation_matrix: np.ndarray) -> Dict[str, Any]:
        """Analyze correlations between questions and neurons.

        Args:
            activation_matrix: Matrix of activations (n_questions, n_neurons).

        Returns:
            Correlation analysis results.
        """
        # Question-to-question correlations
        question_correlations = np.corrcoef(activation_matrix)

        # Neuron-to-neuron correlations (transpose to get neuron x neuron)
        neuron_correlations = np.corrcoef(activation_matrix.T)

        # Summary statistics
        question_corr_stats = self.compute_activation_statistics(
            question_correlations[np.triu_indices_from(question_correlations, k=1)]
        )

        neuron_corr_stats = self.compute_activation_statistics(
            neuron_correlations[np.triu_indices_from(neuron_correlations, k=1)]
        )

        return {
            'question_correlation_matrix': question_correlations.tolist(),
            'neuron_correlation_matrix': neuron_correlations.tolist(),
            'question_correlation_stats': question_corr_stats,
            'neuron_correlation_stats': neuron_corr_stats,
            'highly_correlated_questions': self._find_highly_correlated_indices(
                question_correlations, threshold=0.8
            ),
            'highly_correlated_neurons': self._find_highly_correlated_indices(
                neuron_correlations, threshold=0.8
            )
        }

    def _find_highly_correlated_indices(self, correlation_matrix: np.ndarray, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find pairs of indices with high correlation within a single matrix.

        Args:
            correlation_matrix: Correlation matrix.
            threshold: Correlation threshold for high correlation.

        Returns:
            List of highly correlated pairs.
        """
        highly_correlated = []
        n_items = correlation_matrix.shape[0]

        for i in range(n_items):
            for j in range(i + 1, n_items):  # Avoid diagonal and duplicate pairs
                correlation = correlation_matrix[i, j]
                if abs(correlation) >= threshold:
                    highly_correlated.append({
                        'idx1': int(i),
                        'idx2': int(j),
                        'correlation': float(correlation)
                    })
        return highly_correlated

    def _find_highly_correlated_pairs(
        self,
        correlation_matrix: np.ndarray,
        threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Return index-based correlation pairs for backwards compatibility."""

        raw_pairs = self._find_highly_correlated_indices(correlation_matrix, threshold)
        return [
            {
                'index_1': pair['idx1'],
                'index_2': pair['idx2'],
                'correlation': pair['correlation'],
            }
            for pair in raw_pairs
        ]

    def _find_highly_correlated_layer_pairs(self, correlation_matrix: np.ndarray,
                                     layer_names: List[str], threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find pairs of layers with high correlation.

        Args:
            correlation_matrix: Correlation matrix between layers.
            layer_names: Names of the layers.
            threshold: Correlation threshold for high correlation.

        Returns:
            List of highly correlated layer pairs.
        """
        highly_correlated = []
        n_layers = len(layer_names)

        for i in range(n_layers):
            for j in range(i + 1, n_layers):  # Avoid diagonal and duplicate pairs
                correlation = correlation_matrix[i, j]
                if abs(correlation) >= threshold:
                    highly_correlated.append({
                        'layer1': layer_names[i],
                        'layer2': layer_names[j],
                        'correlation': float(correlation)
                    })

        return highly_correlated

    def perform_dimensionality_reduction(self, activation_matrix: np.ndarray,
                                       method: str = "pca",
                                       n_components: int = 2) -> Dict[str, Any]:
        """Perform dimensionality reduction on activation data.

        Args:
            activation_matrix: Matrix of activations (n_questions, n_neurons).
            method: Reduction method ('pca', 'tsne').
            n_components: Number of components to reduce to.

        Returns:
            Dimensionality reduction results.
        """
        results = {
            'method': method,
            'n_components': n_components,
            'original_shape': activation_matrix.shape
        }

        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components)
            reduced_data = reducer.fit_transform(activation_matrix)

            results['reduced_data'] = reduced_data.tolist()
            results['explained_variance_ratio'] = reducer.explained_variance_ratio_.tolist()
            results['cumulative_variance_ratio'] = np.cumsum(reducer.explained_variance_ratio_).tolist()
            results['principal_components'] = reducer.components_.tolist()

        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, activation_matrix.shape[0]-1))
            reduced_data = reducer.fit_transform(activation_matrix)

            results['reduced_data'] = reduced_data.tolist()

        else:
            raise ValueError(f"Unsupported reduction method: {method}")

        return results

    def cluster_activations(self, activation_matrix: np.ndarray,
                           n_clusters: int = 5) -> Dict[str, Any]:
        """Cluster activation patterns.

        Args:
            activation_matrix: Matrix of activations (n_questions, n_neurons).
            n_clusters: Number of clusters.

        Returns:
            Clustering results.
        """
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(activation_matrix)

        # Analyze clusters
        cluster_stats = []
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            cluster_data = activation_matrix[mask]

            if len(cluster_data) > 0:
                stats = self.compute_activation_statistics(cluster_data)
                stats['cluster_id'] = cluster_id
                stats['cluster_size'] = int(np.sum(mask))
                stats['question_indices'] = np.where(mask)[0].tolist()
                cluster_stats.append(stats)

        return {
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels.tolist(),
            'cluster_centers': clusterer.cluster_centers_.tolist(),
            'cluster_statistics': cluster_stats,
            'inertia': float(clusterer.inertia_)
        }

    def _align_activation_matrices(self, data: Dict[str, Any], layer_names: List[str]) -> Tuple[Dict[str, np.ndarray], List[int]]:
        """Align activation matrices to common questions."""
        layer_matrices = {}
        common_questions = None

        for layer_name in layer_names:
            if layer_name not in data['activations']:
                logger.warning(f"Layer {layer_name} not found in data")
                continue

            layer_data = data['activations'][layer_name]
            question_indices = [item['question_idx'] for item in layer_data]

            if common_questions is None:
                common_questions = set(question_indices)
            else:
                common_questions = common_questions.intersection(set(question_indices))

            # Sort by question index to ensure alignment
            sorted_data = sorted(layer_data, key=lambda x: x['question_idx'])
            layer_matrices[layer_name] = np.array([item['vector'] for item in sorted_data])

        if not common_questions:
            raise ValueError("No common questions found across all layers")

        # Filter to common questions only
        common_indices = sorted(list(common_questions))
        aligned_matrices = {}

        for layer_name, matrix in layer_matrices.items():
            # Find indices of common questions in this layer's data
            layer_data = data['activations'][layer_name]
            layer_question_indices = [item['question_idx'] for item in
                                    sorted(layer_data, key=lambda x: x['question_idx'])]

            mask = [i for i, q_idx in enumerate(layer_question_indices) if q_idx in common_questions]
            aligned_matrices[layer_name] = matrix[mask]

        return aligned_matrices, common_indices

    def compare_layers(self, data: Dict[str, Any],
                      layer_names: List[str],
                      metric: str = "cosine") -> Dict[str, Any]:
        """Compare activation patterns between different layers.

        Args:
            data: Loaded activation data.
            layer_names: List of layer names to compare.
            metric: Similarity metric ('cosine', 'pearson').

        Returns:
            Layer comparison results.
        """
        aligned_matrices, common_indices = self._align_activation_matrices(data, layer_names)

        # Compute layer similarities
        similarities = {}
        for i, layer1 in enumerate(layer_names):
            if layer1 not in aligned_matrices:
                continue
            for j, layer2 in enumerate(layer_names):
                if j <= i or layer2 not in aligned_matrices:
                    continue

                matrix1 = aligned_matrices[layer1]
                matrix2 = aligned_matrices[layer2]

                if metric == "cosine":
                    # Average cosine similarity across questions
                    similarities_per_question = []
                    for q_idx in range(len(matrix1)):
                        sim = cosine_similarity([matrix1[q_idx]], [matrix2[q_idx]])[0, 0]
                        similarities_per_question.append(sim)

                    avg_similarity = np.mean(similarities_per_question)
                    similarities[f"{layer1}_vs_{layer2}"] = {
                        'metric': metric,
                        'average_similarity': float(avg_similarity),
                        'per_question_similarities': similarities_per_question
                    }

                elif metric == "pearson":
                    # Flatten matrices and compute correlation
                    flat1 = matrix1.flatten()
                    flat2 = matrix2.flatten()
                    corr, p_value = pearsonr(flat1, flat2)

                    similarities[f"{layer1}_vs_{layer2}"] = {
                        'metric': metric,
                        'correlation': float(corr),
                        'p_value': float(p_value)
                    }

        return {
            'layer_names': [name for name in layer_names if name in aligned_matrices],
            'common_questions': len(common_indices),
            'similarities': similarities,
            'layer_shapes': {name: matrix.shape for name, matrix in aligned_matrices.items()}
        }

    def generate_analysis_report(self, data: Dict[str, Any],
                               output_dir: str = "data/outputs/analysis") -> str:
        """Generate a comprehensive analysis report.

        Args:
            data: Loaded activation data.
            output_dir: Directory to save the report.

        Returns:
            Path to the generated report.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report = {
            'metadata': data['metadata'],
            'dataset_overview': {
                'num_questions': len(data['questions']),
                'num_layers': len(data['activations']),
                'layer_names': list(data['activations'].keys())
            },
            'layer_analyses': {},
            'cross_layer_analysis': {},
            'dimensionality_reduction': {},
            'clustering': {}
        }

        # Analyze each layer
        for layer_name in data['activations'].keys():
            logger.info(f"Analyzing layer: {layer_name}")
            layer_analysis = self.analyze_layer_activations(data, layer_name)
            report['layer_analyses'][layer_name] = layer_analysis

            # Perform dimensionality reduction
            activation_matrix = np.array([item['vector'] for item in data['activations'][layer_name]])

            # PCA
            pca_results = self.perform_dimensionality_reduction(activation_matrix, 'pca', 2)
            report['dimensionality_reduction'][f"{layer_name}_pca"] = pca_results

            # t-SNE (only if we have enough samples)
            if activation_matrix.shape[0] > 30:
                tsne_results = self.perform_dimensionality_reduction(activation_matrix, 'tsne', 2)
                report['dimensionality_reduction'][f"{layer_name}_tsne"] = tsne_results

            # Clustering
            n_clusters = min(5, activation_matrix.shape[0] // 2)
            if n_clusters >= 2:
                clustering_results = self.cluster_activations(activation_matrix, n_clusters)
                report['clustering'][layer_name] = clustering_results

        # Cross-layer comparison
        layer_names = list(data['activations'].keys())
        if len(layer_names) > 1:
            comparison_results = self.compare_layers(data, layer_names, 'cosine')
            report['cross_layer_analysis'] = comparison_results

        # Save report
        report_file = output_path / "analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Analysis report saved to: {report_file}")
        return str(report_file)


class AdvancedAnalyzer:
    """Advanced analysis methods for neural network activation patterns."""

    def __init__(self, config_name: str = "default"):
        """Initialize advanced analyzer.

        Args:
            config_name: Configuration name to use.
        """
        self.config_name = config_name
        self.logger = logging.getLogger(__name__)

    def perform_pca_analysis(self, activations: np.ndarray, n_components: int = 10) -> Dict[str, Any]:
        """Perform PCA analysis on activation patterns.

        Args:
            activations: Activation data matrix [samples, features].
            n_components: Number of principal components to extract.

        Returns:
            Dictionary containing PCA results.
        """
        try:
            pca = PCA(n_components=n_components)
            transformed_data = pca.fit_transform(activations)

            return {
                'components': pca.components_,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'transformed_data': transformed_data,
                'singular_values': pca.singular_values_,
                'mean': pca.mean_
            }
        except Exception as e:
            self.logger.error(f"PCA analysis failed: {e}")
            raise

    def perform_clustering_analysis(self, activations: np.ndarray, n_clusters: int = 5,
                                  algorithm: str = 'kmeans', methods: List[str] = None) -> Dict[str, Any]:
        """Perform clustering analysis on activations.

        Args:
            activations: Activation data matrix [samples, features].
            n_clusters: Number of clusters.
            algorithm: Single clustering algorithm ('kmeans', 'hierarchical') - for backwards compatibility.
            methods: List of clustering algorithms to run - preferred parameter.

        Returns:
            Dictionary containing clustering results for each method.
        """
        try:
            # Handle both single algorithm and multiple methods parameters
            if methods is not None:
                algorithms_to_run = methods
            else:
                algorithms_to_run = [algorithm]

            results = {}

            for alg in algorithms_to_run:
                if alg == 'kmeans':
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(activations)

                    results[alg] = {
                        'labels': labels,
                        'cluster_centers': kmeans.cluster_centers_,
                        'inertia': kmeans.inertia_,
                        'n_clusters': n_clusters,
                        'algorithm': alg
                    }

                elif alg == 'hierarchical':
                    from sklearn.cluster import AgglomerativeClustering
                    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
                    labels = hierarchical.fit_predict(activations)

                    results[alg] = {
                        'labels': labels,
                        'n_clusters': n_clusters,
                        'algorithm': alg
                    }

                else:
                    self.logger.warning(f"Unsupported clustering algorithm: {alg}")

            # For backwards compatibility, if only one algorithm was run, return its results directly
            if len(results) == 1 and methods is None:
                return list(results.values())[0]

            return results

        except Exception as e:
            self.logger.error(f"Clustering analysis failed: {e}")
            raise

    def compute_similarity_matrix(self, activations: np.ndarray,
                                metric: str = 'cosine') -> np.ndarray:
        """Compute similarity matrix between activation patterns.

        Args:
            activations: Activation data matrix [samples, features].
            metric: Similarity metric ('cosine', 'euclidean', 'correlation').

        Returns:
            Similarity matrix.
        """
        try:
            if metric == 'cosine':
                return cosine_similarity(activations)
            elif metric == 'euclidean':
                distances = pdist(activations, metric='euclidean')
                return squareform(distances)
            elif metric == 'correlation':
                n_samples = activations.shape[0]
                similarity_matrix = np.zeros((n_samples, n_samples))
                for i in range(n_samples):
                    for j in range(n_samples):
                        correlation, _ = pearsonr(activations[i], activations[j])
                        similarity_matrix[i, j] = correlation
                return similarity_matrix
            else:
                raise ValueError(f"Unsupported similarity metric: {metric}")

        except Exception as e:
            self.logger.error(f"Similarity computation failed: {e}")
            raise

    def perform_tsne_analysis(self, activations: np.ndarray, n_components: int = 2,
                            perplexity: float = 30.0) -> Dict[str, Any]:
        """Perform t-SNE dimensionality reduction.

        Args:
            activations: Activation data matrix [samples, features].
            n_components: Number of dimensions for t-SNE output.
            perplexity: t-SNE perplexity parameter.

        Returns:
            Dictionary containing t-SNE results.
        """
        try:
            tsne = TSNE(n_components=n_components, perplexity=perplexity,
                       random_state=42, n_iter=1000)
            embedded = tsne.fit_transform(activations)

            return {
                'embedded': embedded,
                'n_components': n_components,
                'perplexity': perplexity,
                'kl_divergence': tsne.kl_divergence_
            }
        except Exception as e:
            self.logger.error(f"t-SNE analysis failed: {e}")
            raise

    def analyze_layer_progression(self, layer_activations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze how representations change across layers.

        Args:
            layer_activations: Dictionary mapping layer names to activation matrices.

        Returns:
            Dictionary containing layer progression analysis.
        """
        try:
            results = {
                'layer_similarities': {},
                'representational_drift': {},
                'layer_statistics': {}
            }

            layer_names = sorted(layer_activations.keys())

            # Compute inter-layer similarities
            for i, layer1 in enumerate(layer_names):
                for j, layer2 in enumerate(layer_names[i+1:], i+1):
                    # Compute CCA or other similarity measures
                    similarity = self._compute_layer_similarity(
                        layer_activations[layer1],
                        layer_activations[layer2]
                    )
                    results['layer_similarities'][f"{layer1}_{layer2}"] = similarity

            # Compute layer statistics
            for layer_name, activations in layer_activations.items():
                stats = {
                    'mean_activation': np.mean(activations),
                    'std_activation': np.std(activations),
                    'sparsity': np.mean(activations == 0),
                    'max_activation': np.max(activations),
                    'min_activation': np.min(activations)
                }
                results['layer_statistics'][layer_name] = stats

            return results

        except Exception as e:
            self.logger.error(f"Layer progression analysis failed: {e}")
            raise

    def _compute_layer_similarity(self, activations1: np.ndarray,
                                activations2: np.ndarray) -> float:
        """Compute similarity between two layer activations."""
        try:
            # Use centered kernel alignment (CKA) as similarity measure
            def _center_gram_matrix(gram_matrix):
                n = gram_matrix.shape[0]
                unit = np.ones([n, n]) / n
                return gram_matrix - unit @ gram_matrix - gram_matrix @ unit + unit @ gram_matrix @ unit

            # Compute Gram matrices
            gram1 = activations1 @ activations1.T
            gram2 = activations2 @ activations2.T

            # Center the Gram matrices
            centered_gram1 = _center_gram_matrix(gram1)
            centered_gram2 = _center_gram_matrix(gram2)

            # Compute CKA similarity
            numerator = np.trace(centered_gram1 @ centered_gram2)
            denominator = np.sqrt(np.trace(centered_gram1 @ centered_gram1) *
                                np.trace(centered_gram2 @ centered_gram2))

            return numerator / (denominator + 1e-10)

        except Exception as e:
            self.logger.error(f"Layer similarity computation failed: {e}")
            return 0.0

    def compute_activation_statistics(self, activations: np.ndarray) -> Dict[str, Any]:
        """Compute comprehensive statistics for activation patterns.

        Args:
            activations: Activation data matrix [samples, features].

        Returns:
            Dictionary containing activation statistics.
        """
        try:
            stats = {
                'mean': np.mean(activations, axis=0),
                'std': np.std(activations, axis=0),
                'min': np.min(activations, axis=0),
                'max': np.max(activations, axis=0),
                'median': np.median(activations, axis=0),
                'percentile_25': np.percentile(activations, 25, axis=0),
                'percentile_75': np.percentile(activations, 75, axis=0),
                'sparsity': np.mean(activations == 0, axis=0),
                'kurtosis': self._compute_kurtosis(activations),
                'skewness': self._compute_skewness(activations),
                'variance': np.var(activations, axis=0),
                'l1_norm': np.mean(np.abs(activations), axis=0),
                'l2_norm': np.mean(np.square(activations), axis=0)
            }

            # Global statistics
            stats['global_mean'] = np.mean(activations)
            stats['global_std'] = np.std(activations)
            stats['global_sparsity'] = np.mean(activations == 0)

            return stats

        except Exception as e:
            self.logger.error(f"Activation statistics computation failed: {e}")
            raise

    def analyze_layer_correlations(self, layer_activations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze correlations between different layers.

        Args:
            layer_activations: Dictionary mapping layer names to activation matrices.

        Returns:
            Dictionary containing correlation analysis results.
        """
        try:
            layer_names = list(layer_activations.keys())
            n_layers = len(layer_names)

            # Compute correlation matrix between layers
            correlation_matrix = np.zeros((n_layers, n_layers))

            for i, layer1 in enumerate(layer_names):
                for j, layer2 in enumerate(layer_names):
                    # Average correlation across samples
                    correlations = []
                    act1 = layer_activations[layer1]
                    act2 = layer_activations[layer2]

                    # Ensure same number of samples
                    min_samples = min(act1.shape[0], act2.shape[0])
                    act1 = act1[:min_samples]
                    act2 = act2[:min_samples]

                    for sample_idx in range(min_samples):
                        corr, _ = pearsonr(act1[sample_idx], act2[sample_idx])
                        if not np.isnan(corr):
                            correlations.append(corr)

                    correlation_matrix[i, j] = np.mean(correlations) if correlations else 0.0

            return {
                'correlation_matrix': correlation_matrix,
                'layer_names': layer_names,
                'mean_correlation': np.mean(correlation_matrix),
                'max_correlation': np.max(correlation_matrix),
                'min_correlation': np.min(correlation_matrix),
                'highly_correlated_pairs': self._find_highly_correlated_pairs(
                    correlation_matrix, layer_names, threshold=0.7
                )
            }

        except Exception as e:
            self.logger.error(f"Layer correlation analysis failed: {e}")
            raise

    def _find_highly_correlated_pairs(
        self,
        correlation_matrix: np.ndarray,
        layer_names: Optional[List[str]] = None,
        threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Find pairs of activations with correlation above ``threshold``.

        When ``layer_names`` are provided and match the matrix dimensions the
        result includes both the layer identifiers and their positional indices.
        Otherwise the method falls back to reporting indices only.  This keeps
        compatibility with legacy tests that expect ``index_1``/``index_2``
        fields while supporting the richer descriptive output used elsewhere in
        the module.
        """

        if correlation_matrix.ndim != 2:
            raise ValueError("correlation_matrix must be a 2D array")
        if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
            raise ValueError("correlation_matrix must be square")

        num_layers = correlation_matrix.shape[0]
        include_names = isinstance(layer_names, list) and len(layer_names) == num_layers

        results: List[Dict[str, Any]] = []
        for i in range(num_layers):
            for j in range(i + 1, num_layers):
                correlation = float(correlation_matrix[i, j])
                if abs(correlation) < threshold:
                    continue

                entry: Dict[str, Any] = {
                    'index_1': i,
                    'index_2': j,
                    'correlation': correlation,
                }

                if include_names:
                    entry['layer1'] = layer_names[i]
                    entry['layer2'] = layer_names[j]

                results.append(entry)

        return results

    def _compute_kurtosis(self, data: np.ndarray) -> np.ndarray:
        """Compute kurtosis for each feature."""
        from scipy.stats import kurtosis
        return kurtosis(data, axis=0)

    def _compute_skewness(self, data: np.ndarray) -> np.ndarray:
        """Compute skewness for each feature."""
        from scipy.stats import skew
        return skew(data, axis=0)


def main():
    """Command line interface for activation analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Advanced activation analysis")
    parser.add_argument("--input", required=True, help="Path to HDF5 activation file")
    parser.add_argument("--output", default="data/outputs/analysis", help="Output directory")
    parser.add_argument("--layer", help="Specific layer to analyze")
    parser.add_argument("--compare-layers", nargs="+", help="Layers to compare")
    parser.add_argument("--config", default="default", help="Configuration name")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    analyzer = ActivationAnalyzer(args.config)

    # Load data
    logger.info(f"Loading activation data from: {args.input}")
    data = analyzer.load_activations_hdf5(args.input)

    if args.layer:
        # Analyze specific layer
        results = analyzer.analyze_layer_activations(data, args.layer)
        output_file = Path(args.output) / f"{args.layer}_analysis.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Layer analysis saved to: {output_file}")

    elif args.compare_layers:
        # Compare specific layers
        results = analyzer.compare_layers(data, args.compare_layers)
        output_file = Path(args.output) / "layer_comparison.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Layer comparison saved to: {output_file}")

    else:
        # Generate comprehensive report
        report_path = analyzer.generate_analysis_report(data, args.output)
        logger.info(f"Comprehensive analysis completed: {report_path}")


if __name__ == "__main__":
    main()
