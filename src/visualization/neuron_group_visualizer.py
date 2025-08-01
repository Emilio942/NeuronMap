"""
Neuron Group Visualizer for NeuronMap
====================================

This module provides advanced visualization capabilities for identifying and 
visualizing groups/clusters of neurons that activate together during learning.
Focuses on detecting functional neuron groups and their learning patterns.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

# Conditional imports for plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    import matplotlib.cm as cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available. Some visualizations will be disabled.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    logger.warning("Seaborn not available. Some visualizations will be disabled.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Interactive visualizations will be disabled.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available. Network visualizations will be disabled.")

try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Advanced analysis will be disabled.")


@dataclass
class NeuronGroup:
    """Represents a group of neurons that activate together."""
    group_id: int
    neuron_indices: List[int]
    activation_pattern: np.ndarray
    group_center: np.ndarray
    group_size: int
    cohesion_score: float
    learning_phase: Optional[str] = None
    skill_category: Optional[str] = None
    temporal_consistency: Optional[float] = None


@dataclass
class LearningEvent:
    """Represents a learning event or pattern."""
    event_id: int
    question_indices: List[int]
    activated_groups: List[int]
    learning_strength: float
    skill_type: str
    temporal_position: int


class NeuronGroupVisualizer:
    """Advanced visualizer for neuron groups and learning patterns."""
    
    def __init__(self, config=None, output_dir: str = "data/outputs/neuron_groups"):
        """Initialize the neuron group visualizer.
        
        Args:
            config: Configuration object
            output_dir: Directory to save visualizations
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes for different groups
        self.color_schemes = {
            'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
            'learning': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9CA24', '#6C5CE7',
                        '#A0E7E5', '#FEE75C', '#FF9FF3', '#54A0FF', '#5F27CD'],
            'skill': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E',
                     '#7209B7', '#F72585', '#4361EE', '#F77F00', '#FCBF49']
        }
        
        self._setup_plotting_style()
    
    def _setup_plotting_style(self):
        """Setup plotting style and parameters."""
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('default')
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['figure.dpi'] = 150
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.titlesize'] = 14
            plt.rcParams['axes.labelsize'] = 11
            plt.rcParams['legend.fontsize'] = 9

        if SEABORN_AVAILABLE:
            sns.set_style("whitegrid")
            sns.set_palette("Set2")
    
    def identify_neuron_groups(self, 
                             activation_matrix: np.ndarray,
                             method: str = 'correlation_clustering',
                             n_groups: Optional[int] = None,
                             correlation_threshold: float = 0.7,
                             min_group_size: int = 3) -> List[NeuronGroup]:
        """Identify groups of neurons that consistently activate together.
        
        Args:
            activation_matrix: Matrix of shape (n_samples, n_neurons)
            method: Clustering method ('correlation_clustering', 'kmeans', 'hierarchical')
            n_groups: Number of groups (auto-detected if None)
            correlation_threshold: Minimum correlation for grouping
            min_group_size: Minimum neurons per group
            
        Returns:
            List of identified neuron groups
        """
        if not SKLEARN_AVAILABLE:
            logger.error("Scikit-learn required for neuron group identification")
            return []
        
        logger.info(f"Identifying neuron groups using {method}")
        
        # Transpose to get neuron correlations across samples
        neuron_data = activation_matrix.T  # (n_neurons, n_samples)
        
        groups = []
        
        if method == 'correlation_clustering':
            groups = self._correlation_based_clustering(
                neuron_data, correlation_threshold, min_group_size
            )
        elif method == 'kmeans':
            groups = self._kmeans_clustering(
                neuron_data, n_groups, min_group_size
            )
        elif method == 'hierarchical':
            groups = self._hierarchical_clustering(
                neuron_data, n_groups, min_group_size
            )
        else:
            logger.error(f"Unknown clustering method: {method}")
            return []
        
        logger.info(f"Identified {len(groups)} neuron groups")
        return groups
    
    def _correlation_based_clustering(self, 
                                    neuron_data: np.ndarray,
                                    threshold: float,
                                    min_size: int) -> List[NeuronGroup]:
        """Group neurons based on correlation similarity."""
        n_neurons = neuron_data.shape[0]
        
        # Compute correlation matrix
        correlation_matrix = np.corrcoef(neuron_data)
        
        # Find groups using correlation threshold
        visited = set()
        groups = []
        group_id = 0
        
        for i in range(n_neurons):
            if i in visited:
                continue
                
            # Find all neurons correlated with neuron i
            correlated_neurons = []
            for j in range(n_neurons):
                if i != j and abs(correlation_matrix[i, j]) >= threshold:
                    correlated_neurons.append(j)
            
            # Add the seed neuron itself
            correlated_neurons.append(i)
            
            # Check if group is large enough
            if len(correlated_neurons) >= min_size:
                # Mark all neurons as visited
                for neuron_idx in correlated_neurons:
                    visited.add(neuron_idx)
                
                # Calculate group statistics
                group_activation_patterns = neuron_data[correlated_neurons]
                group_center = np.mean(group_activation_patterns, axis=0)
                cohesion = self._calculate_group_cohesion(group_activation_patterns)
                
                group = NeuronGroup(
                    group_id=group_id,
                    neuron_indices=correlated_neurons,
                    activation_pattern=group_activation_patterns,
                    group_center=group_center,
                    group_size=len(correlated_neurons),
                    cohesion_score=cohesion
                )
                
                groups.append(group)
                group_id += 1
        
        return groups
    
    def _kmeans_clustering(self, 
                          neuron_data: np.ndarray,
                          n_groups: Optional[int],
                          min_size: int) -> List[NeuronGroup]:
        """Group neurons using K-means clustering."""
        if n_groups is None:
            n_groups = max(2, min(10, neuron_data.shape[0] // 5))
        
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(neuron_data)
        
        kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(normalized_data)
        
        groups = []
        group_id = 0
        
        for cluster_id in range(n_groups):
            neuron_indices = np.where(cluster_labels == cluster_id)[0].tolist()
            
            if len(neuron_indices) >= min_size:
                group_activation_patterns = neuron_data[neuron_indices]
                group_center = np.mean(group_activation_patterns, axis=0)
                cohesion = self._calculate_group_cohesion(group_activation_patterns)
                
                group = NeuronGroup(
                    group_id=group_id,
                    neuron_indices=neuron_indices,
                    activation_pattern=group_activation_patterns,
                    group_center=group_center,
                    group_size=len(neuron_indices),
                    cohesion_score=cohesion
                )
                
                groups.append(group)
                group_id += 1
        
        return groups
    
    def _hierarchical_clustering(self, 
                               neuron_data: np.ndarray,
                               n_groups: Optional[int],
                               min_size: int) -> List[NeuronGroup]:
        """Group neurons using hierarchical clustering."""
        if n_groups is None:
            n_groups = max(2, min(8, neuron_data.shape[0] // 4))
        
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(neuron_data)
        
        hierarchical = AgglomerativeClustering(n_clusters=n_groups, linkage='ward')
        cluster_labels = hierarchical.fit_predict(normalized_data)
        
        groups = []
        group_id = 0
        
        for cluster_id in range(n_groups):
            neuron_indices = np.where(cluster_labels == cluster_id)[0].tolist()
            
            if len(neuron_indices) >= min_size:
                group_activation_patterns = neuron_data[neuron_indices]
                group_center = np.mean(group_activation_patterns, axis=0)
                cohesion = self._calculate_group_cohesion(group_activation_patterns)
                
                group = NeuronGroup(
                    group_id=group_id,
                    neuron_indices=neuron_indices,
                    activation_pattern=group_activation_patterns,
                    group_center=group_center,
                    group_size=len(neuron_indices),
                    cohesion_score=cohesion
                )
                
                groups.append(group)
                group_id += 1
        
        return groups
    
    def _calculate_group_cohesion(self, group_patterns: np.ndarray) -> float:
        """Calculate cohesion score for a neuron group."""
        if group_patterns.shape[0] < 2:
            return 1.0
        
        # Calculate pairwise correlations within the group
        correlations = np.corrcoef(group_patterns)
        
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(correlations, dtype=bool), k=1)
        correlation_values = correlations[mask]
        
        # Return mean correlation as cohesion score
        return np.mean(correlation_values) if len(correlation_values) > 0 else 0.0
    
    def analyze_learning_patterns(self,
                                activation_matrix: np.ndarray,
                                neuron_groups: List[NeuronGroup],
                                question_metadata: Optional[pd.DataFrame] = None) -> List[LearningEvent]:
        """Analyze learning patterns from neuron group activations.
        
        Args:
            activation_matrix: Matrix of shape (n_samples, n_neurons)
            neuron_groups: Identified neuron groups
            question_metadata: Optional metadata about questions/tasks
            
        Returns:
            List of identified learning events
        """
        logger.info("Analyzing learning patterns from neuron groups")
        
        learning_events = []
        event_id = 0
        
        # For each sample (question/task), determine which groups are active
        for sample_idx in range(activation_matrix.shape[0]):
            sample_activations = activation_matrix[sample_idx]
            
            # Determine which groups are strongly activated
            activated_groups = []
            group_activations = []
            
            for group in neuron_groups:
                # Calculate mean activation for this group
                group_mean_activation = np.mean(sample_activations[group.neuron_indices])
                group_activations.append(group_mean_activation)
                
                # Consider group active if above threshold
                activation_threshold = np.mean(sample_activations) + 0.5 * np.std(sample_activations)
                if group_mean_activation > activation_threshold:
                    activated_groups.append(group.group_id)
            
            # Create learning event if significant activation
            if activated_groups:
                learning_strength = np.max(group_activations)
                
                # Determine skill type from metadata if available
                skill_type = "unknown"
                if question_metadata is not None and len(question_metadata) > sample_idx:
                    # Try to infer skill type from question content or category
                    if 'category' in question_metadata.columns:
                        skill_type = question_metadata.iloc[sample_idx]['category']
                    elif 'question' in question_metadata.columns:
                        skill_type = self._infer_skill_type(question_metadata.iloc[sample_idx]['question'])
                
                event = LearningEvent(
                    event_id=event_id,
                    question_indices=[sample_idx],
                    activated_groups=activated_groups,
                    learning_strength=learning_strength,
                    skill_type=skill_type,
                    temporal_position=sample_idx
                )
                
                learning_events.append(event)
                event_id += 1
        
        logger.info(f"Identified {len(learning_events)} learning events")
        return learning_events
    
    def _infer_skill_type(self, question_text: str) -> str:
        """Infer skill type from question text."""
        question_lower = question_text.lower()
        
        if any(word in question_lower for word in ['math', 'calculate', 'number', 'equation']):
            return 'mathematical'
        elif any(word in question_lower for word in ['language', 'word', 'grammar', 'sentence']):
            return 'linguistic'
        elif any(word in question_lower for word in ['logic', 'reason', 'cause', 'effect']):
            return 'logical'
        elif any(word in question_lower for word in ['memory', 'remember', 'recall', 'fact']):
            return 'memory'
        else:
            return 'general'
    
    def visualize_neuron_groups(self,
                              activation_matrix: np.ndarray,
                              neuron_groups: List[NeuronGroup],
                              method: str = 'heatmap') -> str:
        """Visualize neuron groups and their activation patterns.
        
        Args:
            activation_matrix: Original activation matrix
            neuron_groups: Identified neuron groups
            method: Visualization method ('heatmap', 'network', 'scatter')
            
        Returns:
            Path to saved visualization
        """
        if method == 'heatmap':
            return self._visualize_groups_heatmap(activation_matrix, neuron_groups)
        elif method == 'network':
            return self._visualize_groups_network(activation_matrix, neuron_groups)
        elif method == 'scatter':
            return self._visualize_groups_scatter(activation_matrix, neuron_groups)
        else:
            logger.error(f"Unknown visualization method: {method}")
            return ""
    
    def _visualize_groups_heatmap(self,
                                activation_matrix: np.ndarray,
                                neuron_groups: List[NeuronGroup]) -> str:
        """Create heatmap visualization of neuron groups."""
        if not MATPLOTLIB_AVAILABLE or not SEABORN_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not available for heatmap visualization")
            return ""
        
        # Reorganize matrix by groups
        grouped_indices = []
        group_boundaries = []
        current_pos = 0
        
        for group in neuron_groups:
            grouped_indices.extend(group.neuron_indices)
            current_pos += len(group.neuron_indices)
            group_boundaries.append(current_pos)
        
        # Add ungrouped neurons
        all_grouped = set(grouped_indices)
        ungrouped = [i for i in range(activation_matrix.shape[1]) if i not in all_grouped]
        grouped_indices.extend(ungrouped)
        
        # Reorder matrix
        reordered_matrix = activation_matrix[:, grouped_indices]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[1, 4])
        
        # Top plot: Group labels
        colors = self.color_schemes['default'][:len(neuron_groups)]
        group_colors = []
        
        for i, group in enumerate(neuron_groups):
            group_colors.extend([colors[i % len(colors)]] * len(group.neuron_indices))
        
        # Add gray for ungrouped neurons
        if ungrouped:
            group_colors.extend(['lightgray'] * len(ungrouped))
        
        # Create color bar for groups
        color_array = np.array(group_colors).reshape(1, -1)
        ax1.imshow(color_array, aspect='auto')
        ax1.set_xlim(-0.5, len(grouped_indices) - 0.5)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title('Neuron Groups', fontsize=14, fontweight='bold')
        
        # Add group boundaries
        for boundary in group_boundaries[:-1]:  # Don't draw last boundary
            ax1.axvline(x=boundary - 0.5, color='white', linewidth=2)
        
        # Main heatmap
        sns.heatmap(
            reordered_matrix[:50, :],  # Limit to first 50 samples
            ax=ax2,
            cmap='viridis',
            cbar_kws={'label': 'Activation Strength'},
            xticklabels=False,
            yticklabels=False
        )
        
        ax2.set_title('Neuron Group Activation Patterns', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Neurons (Grouped)', fontsize=12)
        ax2.set_ylabel('Samples/Questions', fontsize=12)
        
        # Add group boundaries to main heatmap
        for boundary in group_boundaries[:-1]:
            ax2.axvline(x=boundary - 0.5, color='white', linewidth=1)
        
        # Add legend
        legend_elements = []
        for i, group in enumerate(neuron_groups):
            color = colors[i % len(colors)]
            legend_elements.append(
                mpatches.Rectangle((0, 0), 1, 1, facecolor=color, 
                                 label=f'Group {group.group_id} ({group.group_size} neurons)')
            )
        
        if ungrouped:
            legend_elements.append(
                mpatches.Rectangle((0, 0), 1, 1, facecolor='lightgray',
                                 label=f'Ungrouped ({len(ungrouped)} neurons)')
            )
        
        ax1.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "neuron_groups_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Neuron groups heatmap saved to {output_path}")
        return str(output_path)
    
    def _visualize_groups_network(self,
                                activation_matrix: np.ndarray,
                                neuron_groups: List[NeuronGroup]) -> str:
        """Create network visualization of neuron groups."""
        if not NETWORKX_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            logger.warning("NetworkX/Matplotlib not available for network visualization")
            return ""
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes for each neuron group
        for group in neuron_groups:
            G.add_node(f"Group_{group.group_id}", 
                      size=group.group_size,
                      cohesion=group.cohesion_score,
                      type='group')
        
        # Add edges based on group interactions
        for i, group1 in enumerate(neuron_groups):
            for j, group2 in enumerate(neuron_groups[i+1:], i+1):
                # Calculate interaction strength between groups
                interaction = self._calculate_group_interaction(
                    activation_matrix, group1, group2
                )
                
                if interaction > 0.3:  # Threshold for significant interaction
                    G.add_edge(f"Group_{group1.group_id}", 
                             f"Group_{group2.group_id}",
                             weight=interaction)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        node_sizes = [G.nodes[node]['size'] * 100 for node in G.nodes()]
        node_colors = [G.nodes[node]['cohesion'] for node in G.nodes()]
        
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap='viridis',
            alpha=0.8
        )
        
        # Draw edges
        edges = G.edges()
        edge_weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(
            G, pos,
            width=[w * 5 for w in edge_weights],
            alpha=0.6,
            edge_color='gray'
        )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title('Neuron Group Interaction Network', fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # Add colorbar for cohesion
        if nodes:
            cbar = plt.colorbar(nodes, shrink=0.8)
            cbar.set_label('Group Cohesion', fontsize=12)
        
        # Save plot
        output_path = self.output_dir / "neuron_groups_network.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Neuron groups network saved to {output_path}")
        return str(output_path)
    
    def _calculate_group_interaction(self,
                                   activation_matrix: np.ndarray,
                                   group1: NeuronGroup,
                                   group2: NeuronGroup) -> float:
        """Calculate interaction strength between two neuron groups."""
        # Get group activations across all samples
        group1_activations = np.mean(activation_matrix[:, group1.neuron_indices], axis=1)
        group2_activations = np.mean(activation_matrix[:, group2.neuron_indices], axis=1)
        
        # Calculate correlation between group activations
        correlation = np.corrcoef(group1_activations, group2_activations)[0, 1]
        
        # Return absolute correlation as interaction strength
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def _visualize_groups_scatter(self,
                                activation_matrix: np.ndarray,
                                neuron_groups: List[NeuronGroup]) -> str:
        """Create scatter plot visualization of neuron groups using dimensionality reduction."""
        if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            logger.warning("Scikit-learn/Matplotlib not available for scatter visualization")
            return ""
        
        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=2)
        activation_2d = pca.fit_transform(activation_matrix)
        
        plt.figure(figsize=(12, 10))
        
        # Create group assignment array
        neuron_to_group = {}
        for group in neuron_groups:
            for neuron_idx in group.neuron_indices:
                neuron_to_group[neuron_idx] = group.group_id
        
        # Calculate group centers in 2D space
        group_centers_2d = {}
        colors = self.color_schemes['default']
        
        for group in neuron_groups:
            # Get samples where this group is highly active
            group_activations = np.mean(activation_matrix[:, group.neuron_indices], axis=1)
            active_threshold = np.percentile(group_activations, 75)
            active_samples = group_activations > active_threshold
            
            if np.any(active_samples):
                group_samples_2d = activation_2d[active_samples]
                color = colors[group.group_id % len(colors)]
                
                plt.scatter(
                    group_samples_2d[:, 0], 
                    group_samples_2d[:, 1],
                    c=color,
                    label=f'Group {group.group_id} ({group.group_size} neurons)',
                    alpha=0.6,
                    s=60
                )
                
                # Calculate and plot group center
                center_2d = np.mean(group_samples_2d, axis=0)
                group_centers_2d[group.group_id] = center_2d
                
                plt.scatter(
                    center_2d[0], center_2d[1],
                    c=color,
                    marker='X',
                    s=200,
                    edgecolors='black',
                    linewidth=2
                )
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
        plt.title('Neuron Groups in 2D Activation Space', fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_path = self.output_dir / "neuron_groups_scatter.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Neuron groups scatter plot saved to {output_path}")
        return str(output_path)
    
    def create_interactive_group_dashboard(self,
                                         activation_matrix: np.ndarray,
                                         neuron_groups: List[NeuronGroup],
                                         learning_events: List[LearningEvent],
                                         question_metadata: Optional[pd.DataFrame] = None) -> str:
        """Create interactive dashboard for neuron groups and learning patterns."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for interactive dashboard")
            return ""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Group Activation Timeline',
                'Group Size Distribution',
                'Learning Event Analysis',
                'Group Cohesion Scores'
            ],
            specs=[[{"secondary_y": True}, {"type": "pie"}],
                   [{"colspan": 2}, None]],
            vertical_spacing=0.15
        )
        
        # 1. Group activation timeline
        colors = self.color_schemes['learning']
        
        for i, group in enumerate(neuron_groups[:8]):  # Limit to 8 groups for clarity
            group_activations = np.mean(activation_matrix[:, group.neuron_indices], axis=1)
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(group_activations))),
                    y=group_activations,
                    mode='lines',
                    name=f'Group {group.group_id}',
                    line=dict(color=colors[i % len(colors)]),
                    hovertemplate=f'Group {group.group_id}<br>Sample: %{{x}}<br>Activation: %{{y:.3f}}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Group size distribution
        group_sizes = [group.group_size for group in neuron_groups]
        group_labels = [f'Group {group.group_id}' for group in neuron_groups]
        
        fig.add_trace(
            go.Pie(
                labels=group_labels,
                values=group_sizes,
                name="Group Sizes",
                hovertemplate='%{label}<br>Size: %{value} neurons<br>Percentage: %{percent}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Learning event analysis
        if learning_events:
            event_positions = [event.temporal_position for event in learning_events]
            event_strengths = [event.learning_strength for event in learning_events]
            event_skills = [event.skill_type for event in learning_events]
            
            # Create skill color mapping
            unique_skills = list(set(event_skills))
            skill_colors = {skill: colors[i % len(colors)] for i, skill in enumerate(unique_skills)}
            
            for skill in unique_skills:
                skill_indices = [i for i, s in enumerate(event_skills) if s == skill]
                skill_positions = [event_positions[i] for i in skill_indices]
                skill_strengths = [event_strengths[i] for i in skill_indices]
                
                fig.add_trace(
                    go.Scatter(
                        x=skill_positions,
                        y=skill_strengths,
                        mode='markers',
                        name=f'{skill} learning',
                        marker=dict(
                            color=skill_colors[skill],
                            size=10,
                            opacity=0.7
                        ),
                        hovertemplate=f'{skill}<br>Position: %{{x}}<br>Strength: %{{y:.3f}}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            title_text="Neuron Group Analysis Dashboard",
            height=800,
            showlegend=True,
            hovermode='closest'
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Sample/Time", row=1, col=1)
        fig.update_yaxes(title_text="Mean Group Activation", row=1, col=1)
        fig.update_xaxes(title_text="Learning Event Position", row=2, col=1)
        fig.update_yaxes(title_text="Learning Strength", row=2, col=1)
        
        # Save interactive dashboard
        output_path = self.output_dir / "interactive_group_dashboard.html"
        fig.write_html(str(output_path))
        
        logger.info(f"Interactive group dashboard saved to {output_path}")
        return str(output_path)
    
    def generate_group_analysis_report(self,
                                     activation_matrix: np.ndarray,
                                     neuron_groups: List[NeuronGroup],
                                     learning_events: List[LearningEvent],
                                     output_format: str = 'json') -> str:
        """Generate comprehensive analysis report of neuron groups and learning patterns.
        
        Args:
            activation_matrix: Original activation matrix
            neuron_groups: Identified neuron groups
            learning_events: Identified learning events
            output_format: Format for report ('json', 'text')
            
        Returns:
            Path to saved report
        """
        logger.info("Generating neuron group analysis report")
        
        # Compile analysis results
        report = {
            'summary': {
                'total_neurons': activation_matrix.shape[1],
                'total_samples': activation_matrix.shape[0],
                'identified_groups': len(neuron_groups),
                'learning_events': len(learning_events),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            },
            'groups': [],
            'learning_patterns': {},
            'statistics': {}
        }
        
        # Group details
        for group in neuron_groups:
            group_data = {
                'group_id': group.group_id,
                'neuron_count': group.group_size,
                'neuron_indices': group.neuron_indices,
                'cohesion_score': float(group.cohesion_score),
                'mean_activation': float(np.mean(group.activation_pattern)),
                'activation_variance': float(np.var(group.activation_pattern))
            }
            report['groups'].append(group_data)
        
        # Learning patterns analysis
        if learning_events:
            skill_types = [event.skill_type for event in learning_events]
            skill_counts = pd.Series(skill_types).value_counts().to_dict()
            
            report['learning_patterns'] = {
                'skill_distribution': skill_counts,
                'total_events': len(learning_events),
                'average_learning_strength': float(np.mean([e.learning_strength for e in learning_events])),
                'temporal_distribution': self._analyze_temporal_patterns(learning_events)
            }
        
        # Overall statistics
        grouped_neurons = set()
        for group in neuron_groups:
            grouped_neurons.update(group.neuron_indices)
        
        report['statistics'] = {
            'neurons_in_groups': len(grouped_neurons),
            'ungrouped_neurons': activation_matrix.shape[1] - len(grouped_neurons),
            'grouping_efficiency': len(grouped_neurons) / activation_matrix.shape[1],
            'average_group_size': np.mean([g.group_size for g in neuron_groups]) if neuron_groups else 0,
            'average_group_cohesion': np.mean([g.cohesion_score for g in neuron_groups]) if neuron_groups else 0
        }
        
        # Save report
        if output_format == 'json':
            output_path = self.output_dir / "neuron_group_analysis_report.json"
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        else:
            output_path = self.output_dir / "neuron_group_analysis_report.txt"
            self._save_text_report(report, output_path)
        
        logger.info(f"Analysis report saved to {output_path}")
        return str(output_path)
    
    def _analyze_temporal_patterns(self, learning_events: List[LearningEvent]) -> Dict[str, Any]:
        """Analyze temporal patterns in learning events."""
        if not learning_events:
            return {}
        
        positions = [event.temporal_position for event in learning_events]
        strengths = [event.learning_strength for event in learning_events]
        
        return {
            'learning_progression': {
                'early_phase': np.mean([s for p, s in zip(positions, strengths) if p < len(positions) * 0.33]),
                'middle_phase': np.mean([s for p, s in zip(positions, strengths) if len(positions) * 0.33 <= p < len(positions) * 0.66]),
                'late_phase': np.mean([s for p, s in zip(positions, strengths) if p >= len(positions) * 0.66])
            },
            'peak_learning_position': positions[np.argmax(strengths)],
            'learning_trend': 'increasing' if np.corrcoef(positions, strengths)[0, 1] > 0 else 'decreasing'
        }
    
    def _save_text_report(self, report: Dict[str, Any], output_path: Path):
        """Save report in text format."""
        with open(output_path, 'w') as f:
            f.write("Neuron Group Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            summary = report['summary']
            f.write(f"Total Neurons: {summary['total_neurons']}\n")
            f.write(f"Total Samples: {summary['total_samples']}\n")
            f.write(f"Identified Groups: {summary['identified_groups']}\n")
            f.write(f"Learning Events: {summary['learning_events']}\n")
            f.write(f"Analysis Time: {summary['analysis_timestamp']}\n\n")
            
            # Groups
            f.write("NEURON GROUPS\n")
            f.write("-" * 20 + "\n")
            for group in report['groups']:
                f.write(f"Group {group['group_id']}:\n")
                f.write(f"  Neurons: {group['neuron_count']}\n")
                f.write(f"  Cohesion: {group['cohesion_score']:.3f}\n")
                f.write(f"  Mean Activation: {group['mean_activation']:.3f}\n")
                f.write(f"  Activation Variance: {group['activation_variance']:.3f}\n\n")
            
            # Learning patterns
            if 'learning_patterns' in report and report['learning_patterns']:
                f.write("LEARNING PATTERNS\n")
                f.write("-" * 20 + "\n")
                patterns = report['learning_patterns']
                f.write(f"Total Learning Events: {patterns['total_events']}\n")
                f.write(f"Average Learning Strength: {patterns['average_learning_strength']:.3f}\n")
                f.write("Skill Distribution:\n")
                for skill, count in patterns['skill_distribution'].items():
                    f.write(f"  {skill}: {count}\n")
                f.write("\n")
            
            # Statistics
            f.write("STATISTICS\n")
            f.write("-" * 20 + "\n")
            stats = report['statistics']
            f.write(f"Neurons in Groups: {stats['neurons_in_groups']}\n")
            f.write(f"Ungrouped Neurons: {stats['ungrouped_neurons']}\n")
            f.write(f"Grouping Efficiency: {stats['grouping_efficiency']:.2%}\n")
            f.write(f"Average Group Size: {stats['average_group_size']:.1f}\n")
            f.write(f"Average Group Cohesion: {stats['average_group_cohesion']:.3f}\n")


def create_neuron_group_analysis(activation_matrix: np.ndarray,
                                question_metadata: Optional[pd.DataFrame] = None,
                                output_dir: str = "data/outputs/neuron_groups",
                                config=None) -> Dict[str, Any]:
    """Convenience function to run complete neuron group analysis.
    
    Args:
        activation_matrix: Matrix of shape (n_samples, n_neurons)
        question_metadata: Optional metadata about questions/tasks
        output_dir: Directory to save results
        config: Configuration object
        
    Returns:
        Dictionary containing analysis results and file paths
    """
    logger.info("Starting complete neuron group analysis")
    
    # Initialize visualizer
    visualizer = NeuronGroupVisualizer(config=config, output_dir=output_dir)
    
    # Identify neuron groups
    neuron_groups = visualizer.identify_neuron_groups(
        activation_matrix, 
        method='correlation_clustering',
        correlation_threshold=0.6,
        min_group_size=3
    )
    
    # Analyze learning patterns
    learning_events = visualizer.analyze_learning_patterns(
        activation_matrix, neuron_groups, question_metadata
    )
    
    # Create visualizations
    heatmap_path = visualizer.visualize_neuron_groups(
        activation_matrix, neuron_groups, method='heatmap'
    )
    
    network_path = visualizer.visualize_neuron_groups(
        activation_matrix, neuron_groups, method='network'
    )
    
    scatter_path = visualizer.visualize_neuron_groups(
        activation_matrix, neuron_groups, method='scatter'
    )
    
    # Create interactive dashboard
    dashboard_path = visualizer.create_interactive_group_dashboard(
        activation_matrix, neuron_groups, learning_events, question_metadata
    )
    
    # Generate analysis report
    report_path = visualizer.generate_group_analysis_report(
        activation_matrix, neuron_groups, learning_events
    )
    
    results = {
        'neuron_groups': neuron_groups,
        'learning_events': learning_events,
        'visualizations': {
            'heatmap': heatmap_path,
            'network': network_path,
            'scatter': scatter_path,
            'dashboard': dashboard_path
        },
        'report': report_path,
        'summary': {
            'total_groups': len(neuron_groups),
            'total_learning_events': len(learning_events),
            'analysis_complete': True
        }
    }
    
    logger.info("Neuron group analysis completed successfully")
    return results
