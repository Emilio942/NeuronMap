"""
Functional Neuron Groups Finder for NeuronMap
==============================================

This module implements advanced algorithms to identify and analyze functional groups
of neurons that work together to perform specific cognitive tasks in transformer models.

Author: GitHub Copilot
Date: July 29, 2025
Purpose: Extend NeuronMap with sophisticated neuron group discovery capabilities
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisTaskType(Enum):
    """Types of cognitive tasks for functional group analysis."""
    TOKEN_CLASSIFICATION = "token_classification"
    ARITHMETIC_OPERATIONS = "arithmetic_operations"
    CAUSAL_REASONING = "causal_reasoning"
    SYNTAX_PARSING = "syntax_parsing"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    ENTITY_COMPARISON = "entity_comparison"
    LOGICAL_OPERATIONS = "logical_operations"
    ATTENTION_PATTERNS = "attention_patterns"
    SEQUENCE_CONTINUATION = "sequence_continuation"
    FACTUAL_RECALL = "factual_recall"


class ClusteringMethod(Enum):
    """Available clustering algorithms."""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    SPECTRAL = "spectral"
    GAUSSIAN_MIXTURE = "gaussian_mixture"


@dataclass
class NeuronGroup:
    """Represents a functional group of neurons."""
    group_id: str
    neurons: List[int]
    layer: int
    function: str
    activation_trigger: List[str]
    ablation_effect: str
    confidence: float
    task_type: AnalysisTaskType
    statistical_metrics: Dict[str, float] = field(default_factory=dict)
    co_activation_strength: float = 0.0
    cluster_coherence: float = 0.0
    examples: List[Dict[str, Any]] = field(default_factory=list)
    visualization_data: Optional[Dict[str, Any]] = None


@dataclass
class ActivationPattern:
    """Stores activation patterns for analysis."""
    activations: np.ndarray  # Shape: (n_samples, n_neurons)
    inputs: List[str]
    labels: Optional[List[str]] = None
    layer: int = 0
    task_type: Optional[AnalysisTaskType] = None


class FunctionalGroupsFinder:
    """
    Advanced system for discovering functional neuron groups in transformer models.
    
    This class implements multiple algorithms to identify clusters of neurons that
    work together to perform specific cognitive functions.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.7,
                 min_group_size: int = 3,
                 max_group_size: int = 50,
                 clustering_method: ClusteringMethod = ClusteringMethod.KMEANS):
        """
        Initialize the Functional Groups Finder.
        
        Args:
            similarity_threshold: Minimum correlation for neuron co-activation
            min_group_size: Minimum number of neurons in a functional group
            max_group_size: Maximum number of neurons in a functional group
            clustering_method: Clustering algorithm to use
        """
        self.similarity_threshold = similarity_threshold
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self.clustering_method = clustering_method
        
        # Storage for discovered groups
        self.discovered_groups: Dict[str, List[NeuronGroup]] = {}
        self.activation_patterns: Dict[str, ActivationPattern] = {}
        
        logger.info(f"Initialized FunctionalGroupsFinder with threshold={similarity_threshold}")
    
    def add_activation_pattern(self, 
                             pattern_id: str,
                             activations: np.ndarray,
                             inputs: List[str],
                             layer: int,
                             task_type: AnalysisTaskType,
                             labels: Optional[List[str]] = None) -> None:
        """
        Add activation patterns for a specific task and layer.
        
        Args:
            pattern_id: Unique identifier for this activation pattern
            activations: Neuron activations array (n_samples, n_neurons)
            inputs: Input texts that generated these activations
            layer: Layer number
            task_type: Type of cognitive task
            labels: Optional labels for supervised analysis
        """
        if activations.ndim != 2:
            raise ValueError(f"Activations must be 2D, got shape {activations.shape}")
        
        pattern = ActivationPattern(
            activations=activations,
            inputs=inputs,
            labels=labels,
            layer=layer,
            task_type=task_type
        )
        
        self.activation_patterns[pattern_id] = pattern
        logger.info(f"Added activation pattern '{pattern_id}' for layer {layer}, "
                   f"shape: {activations.shape}, task: {task_type.value}")
    
    def calculate_neuron_correlations(self, activations: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise correlations between neurons.
        
        Args:
            activations: Neuron activations (n_samples, n_neurons)
            
        Returns:
            Correlation matrix (n_neurons, n_neurons)
        """
        logger.info("Calculating neuron correlations...")
        
        # Calculate Pearson correlations
        correlation_matrix = np.corrcoef(activations.T)
        
        # Handle NaN values (neurons with zero variance)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        
        return correlation_matrix
    
    def find_co_activation_clusters(self, 
                                   activations: np.ndarray,
                                   method: Optional[ClusteringMethod] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Find clusters of co-activated neurons using various clustering methods.
        
        Args:
            activations: Neuron activations (n_samples, n_neurons)
            method: Clustering method to use (defaults to self.clustering_method)
            
        Returns:
            Tuple of (cluster_labels, clustering_metrics)
        """
        if method is None:
            method = self.clustering_method
            
        logger.info(f"Finding co-activation clusters using {method.value}...")
        
        n_neurons = activations.shape[1]
        correlation_matrix = self.calculate_neuron_correlations(activations)
        
        # Convert correlation to distance matrix
        distance_matrix = 1 - np.abs(correlation_matrix)
        
        metrics = {}
        
        if method == ClusteringMethod.KMEANS:
            # Estimate optimal number of clusters using elbow method
            n_clusters = self._estimate_optimal_clusters(activations)
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(activations.T)
            
        elif method == ClusteringMethod.DBSCAN:
            # Use distance matrix for DBSCAN
            clusterer = DBSCAN(eps=0.3, min_samples=self.min_group_size, metric='precomputed')
            labels = clusterer.fit_predict(distance_matrix)
            
        elif method == ClusteringMethod.HIERARCHICAL:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1 - self.similarity_threshold,
                linkage='ward'
            )
            labels = clusterer.fit_predict(activations.T)
            
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        # Calculate clustering quality metrics
        if len(np.unique(labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(activations.T, labels)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(activations.T, labels)
        
        logger.info(f"Found {len(np.unique(labels))} clusters")
        return labels, metrics
    
    def _estimate_optimal_clusters(self, activations: np.ndarray) -> int:
        """Estimate optimal number of clusters using elbow method."""
        max_clusters = min(20, activations.shape[1] // self.min_group_size)
        if max_clusters < 2:
            return 2
            
        inertias = []
        K = range(2, max_clusters + 1)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(activations.T)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point
        if len(inertias) >= 3:
            diffs = np.diff(inertias)
            diff2 = np.diff(diffs)
            elbow_idx = np.argmax(diff2) + 2  # +2 because we start from k=2
            return K[elbow_idx] if elbow_idx < len(K) else K[-1]
        
        return K[0]
    
    def analyze_task_specificity(self, 
                                pattern_id: str,
                                target_inputs: List[str]) -> Dict[str, float]:
        """
        Analyze how specific neuron groups are to particular task inputs.
        
        Args:
            pattern_id: ID of the activation pattern to analyze
            target_inputs: Specific inputs to analyze for task specificity
            
        Returns:
            Dictionary mapping neuron IDs to specificity scores
        """
        if pattern_id not in self.activation_patterns:
            raise ValueError(f"Pattern '{pattern_id}' not found")
        
        pattern = self.activation_patterns[pattern_id]
        activations = pattern.activations
        inputs = pattern.inputs
        
        # Find indices of target inputs
        target_indices = [i for i, inp in enumerate(inputs) if inp in target_inputs]
        if not target_indices:
            logger.warning("No target inputs found in pattern")
            return {}
        
        # Calculate specificity for each neuron
        specificity_scores = {}
        
        for neuron_idx in range(activations.shape[1]):
            neuron_activations = activations[:, neuron_idx]
            
            # Activation for target inputs
            target_activations = neuron_activations[target_indices]
            # Activation for all other inputs
            other_activations = np.delete(neuron_activations, target_indices)
            
            if len(other_activations) == 0:
                specificity_scores[neuron_idx] = 1.0
                continue
            
            # Calculate specificity as ratio of mean target activation to mean other activation
            target_mean = np.mean(target_activations)
            other_mean = np.mean(other_activations)
            
            if other_mean == 0:
                specificity = float('inf') if target_mean > 0 else 0
            else:
                specificity = target_mean / other_mean
            
            specificity_scores[neuron_idx] = min(specificity, 10.0)  # Cap at 10
        
        return specificity_scores
    
    def perform_ablation_analysis(self,
                                 pattern_id: str,
                                 neuron_groups: List[List[int]],
                                 model_forward_fn: callable) -> Dict[str, float]:
        """
        Perform ablation analysis on neuron groups to measure their functional impact.
        
        Args:
            pattern_id: ID of the activation pattern
            neuron_groups: List of neuron index lists to ablate
            model_forward_fn: Function that takes (inputs, ablated_neurons) and returns predictions
            
        Returns:
            Dictionary mapping group IDs to performance impact scores
        """
        if pattern_id not in self.activation_patterns:
            raise ValueError(f"Pattern '{pattern_id}' not found")
        
        pattern = self.activation_patterns[pattern_id]
        inputs = pattern.inputs
        
        logger.info(f"Performing ablation analysis on {len(neuron_groups)} groups...")
        
        # Get baseline performance (no ablation)
        baseline_outputs = model_forward_fn(inputs, [])
        
        ablation_effects = {}
        
        for group_idx, group in enumerate(neuron_groups):
            group_id = f"group_{group_idx}"
            
            # Ablate this group
            ablated_outputs = model_forward_fn(inputs, group)
            
            # Calculate performance difference
            if hasattr(baseline_outputs, 'logits'):
                baseline_probs = F.softmax(baseline_outputs.logits, dim=-1)
                ablated_probs = F.softmax(ablated_outputs.logits, dim=-1)
                
                # KL divergence as a measure of impact
                kl_div = F.kl_div(
                    ablated_probs.log(),
                    baseline_probs,
                    reduction='mean'
                ).item()
                
                ablation_effects[group_id] = kl_div
            else:
                # Simple MSE for regression-like outputs
                mse = F.mse_loss(baseline_outputs, ablated_outputs).item()
                ablation_effects[group_id] = mse
        
        return ablation_effects
    
    def discover_functional_groups(self, 
                                  pattern_id: str,
                                  task_type: AnalysisTaskType,
                                  generate_visualizations: bool = True) -> List[NeuronGroup]:
        """
        Main method to discover functional neuron groups for a specific task.
        
        Args:
            pattern_id: ID of the activation pattern to analyze
            task_type: Type of cognitive task
            generate_visualizations: Whether to generate visualization data
            
        Returns:
            List of discovered NeuronGroup objects
        """
        if pattern_id not in self.activation_patterns:
            raise ValueError(f"Pattern '{pattern_id}' not found")
        
        pattern = self.activation_patterns[pattern_id]
        activations = pattern.activations
        inputs = pattern.inputs
        layer = pattern.layer
        
        logger.info(f"Discovering functional groups for {task_type.value} in layer {layer}")
        
        # Step 1: Find co-activation clusters
        cluster_labels, clustering_metrics = self.find_co_activation_clusters(activations)
        
        # Step 2: Analyze each cluster
        discovered_groups = []
        unique_labels = np.unique(cluster_labels)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Noise cluster in DBSCAN
                continue
                
            # Get neurons in this cluster
            cluster_neurons = np.where(cluster_labels == cluster_id)[0].tolist()
            
            # Filter by size constraints
            if len(cluster_neurons) < self.min_group_size or len(cluster_neurons) > self.max_group_size:
                continue
            
            # Calculate cluster statistics
            cluster_activations = activations[:, cluster_neurons]
            correlation_matrix = self.calculate_neuron_correlations(cluster_activations)
            
            # Co-activation strength (mean pairwise correlation)
            upper_triangle = np.triu(correlation_matrix, k=1)
            co_activation_strength = np.mean(upper_triangle[upper_triangle != 0])
            
            # Cluster coherence (how well neurons cluster together)
            cluster_coherence = self._calculate_cluster_coherence(cluster_activations)
            
            # Statistical metrics
            statistical_metrics = {
                'mean_activation': float(np.mean(cluster_activations)),
                'std_activation': float(np.std(cluster_activations)),
                'max_activation': float(np.max(cluster_activations)),
                'sparsity': float(np.mean(cluster_activations == 0)),
                'correlation_strength': float(co_activation_strength)
            }
            
            # Identify activation triggers (inputs that strongly activate this group)
            activation_triggers = self._identify_activation_triggers(
                cluster_activations, inputs, top_k=5
            )
            
            # Estimate functional role
            functional_role = self._estimate_functional_role(
                cluster_activations, inputs, task_type
            )
            
            # Create neuron group
            group = NeuronGroup(
                group_id=f"{pattern_id}_cluster_{cluster_id}",
                neurons=cluster_neurons,
                layer=layer,
                function=functional_role,
                activation_trigger=activation_triggers,
                ablation_effect="Not measured",  # Will be filled by ablation analysis
                confidence=co_activation_strength,
                task_type=task_type,
                statistical_metrics=statistical_metrics,
                co_activation_strength=co_activation_strength,
                cluster_coherence=cluster_coherence
            )
            
            # Generate visualizations if requested
            if generate_visualizations:
                group.visualization_data = self._generate_visualization_data(
                    cluster_activations, inputs, cluster_neurons
                )
            
            discovered_groups.append(group)
        
        # Store discovered groups
        if pattern_id not in self.discovered_groups:
            self.discovered_groups[pattern_id] = []
        self.discovered_groups[pattern_id].extend(discovered_groups)
        
        logger.info(f"Discovered {len(discovered_groups)} functional groups")
        return discovered_groups
    
    def _calculate_cluster_coherence(self, cluster_activations: np.ndarray) -> float:
        """Calculate how coherent the cluster is (how similar neurons behave)."""
        if cluster_activations.shape[1] < 2:
            return 1.0
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(cluster_activations.shape[1]):
            for j in range(i + 1, cluster_activations.shape[1]):
                corr, _ = pearsonr(cluster_activations[:, i], cluster_activations[:, j])
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _identify_activation_triggers(self, 
                                    cluster_activations: np.ndarray,
                                    inputs: List[str],
                                    top_k: int = 5) -> List[str]:
        """Identify input patterns that most strongly activate this cluster."""
        # Calculate mean activation across all neurons in cluster for each input
        mean_activations = np.mean(cluster_activations, axis=1)
        
        # Get top-k most activating inputs
        top_indices = np.argsort(mean_activations)[-top_k:][::-1]
        
        triggers = []
        for idx in top_indices:
            if idx < len(inputs):
                triggers.append(inputs[idx])
        
        return triggers
    
    def _estimate_functional_role(self,
                                cluster_activations: np.ndarray,
                                inputs: List[str],
                                task_type: AnalysisTaskType) -> str:
        """Estimate the functional role of a neuron cluster based on task type and activation patterns."""
        
        # Analyze activation patterns to infer function
        mean_activation = np.mean(cluster_activations)
        sparsity = np.mean(cluster_activations == 0)
        
        # Task-specific role estimation
        if task_type == AnalysisTaskType.TOKEN_CLASSIFICATION:
            if sparsity > 0.8:
                return "Sparse token-specific feature detector"
            elif mean_activation > 0.5:
                return "General token classification mechanism"
            else:
                return "Contextual token modifier"
                
        elif task_type == AnalysisTaskType.ARITHMETIC_OPERATIONS:
            # Look for numerical patterns in triggers
            triggers = self._identify_activation_triggers(cluster_activations, inputs)
            if any(any(char.isdigit() for char in trigger) for trigger in triggers):
                return "Numerical computation processor"
            else:
                return "Arithmetic context analyzer"
                
        elif task_type == AnalysisTaskType.CAUSAL_REASONING:
            return "Causal relationship detector"
            
        elif task_type == AnalysisTaskType.SYNTAX_PARSING:
            if sparsity > 0.7:
                return "Specific syntactic pattern detector"
            else:
                return "General syntactic structure analyzer"
                
        elif task_type == AnalysisTaskType.SEMANTIC_SIMILARITY:
            return "Semantic relationship encoder"
            
        elif task_type == AnalysisTaskType.ENTITY_COMPARISON:
            return "Entity comparison mechanism"
            
        else:
            return f"Function related to {task_type.value}"
    
    def _generate_visualization_data(self,
                                   cluster_activations: np.ndarray,
                                   inputs: List[str],
                                   neuron_indices: List[int]) -> Dict[str, Any]:
        """Generate data for visualizing the functional group."""
        
        viz_data = {}
        
        # 1. Activation heatmap data
        viz_data['heatmap'] = {
            'activations': cluster_activations.tolist(),
            'neuron_indices': neuron_indices,
            'input_labels': inputs[:len(cluster_activations)]  # Ensure matching length
        }
        
        # 2. PCA projection for dimensionality reduction
        if cluster_activations.shape[1] > 2:
            pca = PCA(n_components=2)
            pca_coords = pca.fit_transform(cluster_activations.T)
            viz_data['pca'] = {
                'coordinates': pca_coords.tolist(),
                'explained_variance': pca.explained_variance_ratio_.tolist()
            }
        
        # 3. Correlation matrix
        corr_matrix = self.calculate_neuron_correlations(cluster_activations)
        viz_data['correlation_matrix'] = corr_matrix.tolist()
        
        # 4. Activation statistics over time/inputs
        viz_data['activation_timeline'] = {
            'mean_activation': np.mean(cluster_activations, axis=1).tolist(),
            'std_activation': np.std(cluster_activations, axis=1).tolist()
        }
        
        return viz_data
    
    def export_groups_to_json(self, 
                             pattern_id: str,
                             output_path: Union[str, Path]) -> None:
        """
        Export discovered functional groups to JSON format.
        
        Args:
            pattern_id: ID of the pattern whose groups to export
            output_path: Path to save the JSON file
        """
        if pattern_id not in self.discovered_groups:
            raise ValueError(f"No groups found for pattern '{pattern_id}'")
        
        groups_data = []
        for group in self.discovered_groups[pattern_id]:
            group_dict = {
                'group_id': group.group_id,
                'neurons': group.neurons,
                'layer': group.layer,
                'function': group.function,
                'activation_trigger': group.activation_trigger,
                'ablation_effect': group.ablation_effect,
                'confidence': group.confidence,
                'task_type': group.task_type.value,
                'statistical_metrics': group.statistical_metrics,
                'co_activation_strength': group.co_activation_strength,
                'cluster_coherence': group.cluster_coherence
            }
            groups_data.append(group_dict)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(groups_data, f, indent=2)
        
        logger.info(f"Exported {len(groups_data)} groups to {output_path}")
    
    def visualize_functional_groups(self, 
                                   pattern_id: str,
                                   save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Create comprehensive visualizations of discovered functional groups.
        
        Args:
            pattern_id: ID of the pattern whose groups to visualize
            save_path: Optional path to save the visualization
        """
        if pattern_id not in self.discovered_groups:
            raise ValueError(f"No groups found for pattern '{pattern_id}'")
        
        groups = self.discovered_groups[pattern_id]
        
        # Create a comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Functional Neuron Groups - {pattern_id}', fontsize=16)
        
        # 1. Group size distribution
        group_sizes = [len(group.neurons) for group in groups]
        axes[0, 0].hist(group_sizes, bins=10, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Group Size Distribution')
        axes[0, 0].set_xlabel('Number of Neurons')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Co-activation strength vs Confidence
        co_activation = [group.co_activation_strength for group in groups]
        confidence = [group.confidence for group in groups]
        scatter = axes[0, 1].scatter(co_activation, confidence, 
                                   c=range(len(groups)), cmap='viridis', alpha=0.7)
        axes[0, 1].set_title('Co-activation Strength vs Confidence')
        axes[0, 1].set_xlabel('Co-activation Strength')
        axes[0, 1].set_ylabel('Confidence')
        plt.colorbar(scatter, ax=axes[0, 1])
        
        # 3. Cluster coherence distribution
        coherence = [group.cluster_coherence for group in groups]
        axes[1, 0].hist(coherence, bins=10, alpha=0.7, color='lightcoral')
        axes[1, 0].set_title('Cluster Coherence Distribution')
        axes[1, 0].set_xlabel('Coherence Score')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Function type distribution
        functions = [group.function for group in groups]
        function_counts = pd.Series(functions).value_counts()
        axes[1, 1].pie(function_counts.values, labels=function_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Function Type Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def generate_analysis_report(self, pattern_id: str) -> str:
        """
        Generate a comprehensive analysis report for discovered functional groups.
        
        Args:
            pattern_id: ID of the pattern to analyze
            
        Returns:
            Formatted analysis report as string
        """
        if pattern_id not in self.discovered_groups:
            return f"No groups found for pattern '{pattern_id}'"
        
        groups = self.discovered_groups[pattern_id]
        pattern = self.activation_patterns[pattern_id]
        
        report = []
        report.append(f"🧠 FUNCTIONAL NEURON GROUPS ANALYSIS REPORT")
        report.append(f"=" * 50)
        report.append(f"Pattern ID: {pattern_id}")
        report.append(f"Layer: {pattern.layer}")
        report.append(f"Task Type: {pattern.task_type.value}")
        report.append(f"Total Neurons Analyzed: {pattern.activations.shape[1]}")
        report.append(f"Number of Samples: {pattern.activations.shape[0]}")
        report.append(f"Discovered Groups: {len(groups)}")
        report.append("")
        
        # Summary statistics
        if groups:
            group_sizes = [len(g.neurons) for g in groups]
            confidences = [g.confidence for g in groups]
            
            report.append(f"📊 SUMMARY STATISTICS:")
            report.append(f"Average Group Size: {np.mean(group_sizes):.1f} ± {np.std(group_sizes):.1f}")
            report.append(f"Average Confidence: {np.mean(confidences):.3f} ± {np.std(confidences):.3f}")
            report.append(f"Size Range: {min(group_sizes)} - {max(group_sizes)} neurons")
            report.append("")
        
        # Detailed group analysis
        report.append(f"🔍 DETAILED GROUP ANALYSIS:")
        report.append("-" * 30)
        
        for i, group in enumerate(groups, 1):
            report.append(f"Group {i}: {group.group_id}")
            report.append(f"  📍 Neurons: {group.neurons[:10]}{'...' if len(group.neurons) > 10 else ''} ({len(group.neurons)} total)")
            report.append(f"  🎯 Function: {group.function}")
            report.append(f"  📈 Confidence: {group.confidence:.3f}")
            report.append(f"  🔗 Co-activation: {group.co_activation_strength:.3f}")
            report.append(f"  🎲 Coherence: {group.cluster_coherence:.3f}")
            
            # Top activation triggers
            if group.activation_trigger:
                report.append(f"  💡 Top Triggers:")
                for trigger in group.activation_trigger[:3]:
                    report.append(f"     • {trigger[:50]}{'...' if len(trigger) > 50 else ''}")
            
            # Key statistics
            if group.statistical_metrics:
                metrics = group.statistical_metrics
                report.append(f"  📊 Statistics:")
                report.append(f"     Mean activation: {metrics.get('mean_activation', 0):.3f}")
                report.append(f"     Sparsity: {metrics.get('sparsity', 0):.3f}")
                report.append(f"     Max activation: {metrics.get('max_activation', 0):.3f}")
            
            report.append("")
        
        # Recommendations
        report.append(f"💡 RECOMMENDATIONS:")
        report.append("-" * 20)
        
        high_confidence_groups = [g for g in groups if g.confidence > 0.8]
        if high_confidence_groups:
            report.append(f"• {len(high_confidence_groups)} high-confidence groups identified for detailed ablation studies")
        
        large_groups = [g for g in groups if len(g.neurons) > 20]
        if large_groups:
            report.append(f"• {len(large_groups)} large groups may benefit from sub-clustering analysis")
        
        sparse_groups = [g for g in groups if g.statistical_metrics.get('sparsity', 0) > 0.8]
        if sparse_groups:
            report.append(f"• {len(sparse_groups)} sparse groups may be specialized feature detectors")
        
        report.append("")
        report.append(f"🎯 NEXT STEPS:")
        report.append("1. Perform ablation analysis to measure functional impact")
        report.append("2. Validate groups across different input sets")
        report.append("3. Compare functional groups across different layers")
        report.append("4. Investigate cross-task functional overlap")
        
        return "\n".join(report)


# Example usage and testing functions
def create_sample_data() -> Tuple[np.ndarray, List[str]]:
    """Create sample activation data for testing."""
    np.random.seed(42)
    
    # Simulate activations for 100 samples and 50 neurons
    n_samples, n_neurons = 100, 50
    
    # Create some structured groups
    activations = np.random.randn(n_samples, n_neurons) * 0.1
    
    # Group 1: Neurons 0-9 activated together for arithmetic
    arithmetic_samples = list(range(0, 30))
    activations[arithmetic_samples, 0:10] += np.random.randn(len(arithmetic_samples), 10) * 0.5 + 1.0
    
    # Group 2: Neurons 20-29 activated for semantic similarity
    semantic_samples = list(range(30, 60))
    activations[semantic_samples, 20:30] += np.random.randn(len(semantic_samples), 10) * 0.3 + 0.8
    
    # Group 3: Neurons 35-44 activated for causal reasoning
    causal_samples = list(range(60, 90))
    activations[causal_samples, 35:45] += np.random.randn(len(causal_samples), 10) * 0.4 + 1.2
    
    # Create corresponding inputs
    inputs = []
    for i in range(n_samples):
        if i < 30:
            inputs.append(f"Calculate {i} + {i+1}")
        elif i < 60:
            inputs.append(f"Compare meaning of word{i} and word{i+1}")
        elif i < 90:
            inputs.append(f"Because of A, then B happened, therefore {i}")
        else:
            inputs.append(f"Random input {i}")
    
    return activations, inputs


def demo_functional_groups_finder():
    """Demonstrate the Functional Groups Finder with sample data."""
    print("🧠 Functional Groups Finder Demo")
    print("=" * 40)
    
    # Create sample data
    activations, inputs = create_sample_data()
    print(f"✓ Created sample data: {activations.shape[0]} samples, {activations.shape[1]} neurons")
    
    # Initialize finder
    finder = FunctionalGroupsFinder(
        similarity_threshold=0.6,
        min_group_size=3,
        max_group_size=20,
        clustering_method=ClusteringMethod.KMEANS
    )
    
    # Add activation pattern
    finder.add_activation_pattern(
        pattern_id="demo_mixed_tasks",
        activations=activations,
        inputs=inputs,
        layer=6,
        task_type=AnalysisTaskType.ARITHMETIC_OPERATIONS
    )
    
    # Discover functional groups
    groups = finder.discover_functional_groups(
        pattern_id="demo_mixed_tasks",
        task_type=AnalysisTaskType.ARITHMETIC_OPERATIONS,
        generate_visualizations=True
    )
    
    print(f"✓ Discovered {len(groups)} functional groups")
    
    # Generate and print analysis report
    report = finder.generate_analysis_report("demo_mixed_tasks")
    print("\n" + report)
    
    # Export results
    output_dir = Path("outputs/functional_groups")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    finder.export_groups_to_json("demo_mixed_tasks", output_dir / "demo_groups.json")
    print(f"✓ Exported results to {output_dir / 'demo_groups.json'}")
    
    # Create visualizations
    finder.visualize_functional_groups("demo_mixed_tasks", output_dir / "demo_visualization.png")
    print(f"✓ Generated visualization at {output_dir / 'demo_visualization.png'}")
    
    return finder, groups


if __name__ == "__main__":
    # Run demonstration
    demo_functional_groups_finder()
