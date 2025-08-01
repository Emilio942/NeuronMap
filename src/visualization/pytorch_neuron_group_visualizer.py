"""
PyTorch Neuron Group Visualizer for NeuronMap
============================================

This module provides PyTorch-integrated visualization capabilities for identifying 
and visualizing groups/clusters of neurons that activate together during learning.
Specifically designed for PyTorch models and tensor operations.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import json
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Some features will be disabled.")

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
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Advanced analysis will be disabled.")


@dataclass
class PyTorchNeuronGroup:
    """Represents a group of neurons that activate together in PyTorch models."""
    group_id: int
    neuron_indices: List[int]
    layer_name: str
    activation_pattern: torch.Tensor
    group_center: torch.Tensor
    group_size: int
    cohesion_score: float
    learning_phase: Optional[str] = None
    skill_category: Optional[str] = None
    temporal_consistency: Optional[float] = None
    device: str = 'cpu'


@dataclass
class PyTorchLearningEvent:
    """Represents a learning event or pattern in PyTorch context."""
    event_id: int
    sample_indices: List[int]
    activated_groups: List[int]
    learning_strength: float
    skill_type: str
    temporal_position: int
    model_epoch: Optional[int] = None
    loss_value: Optional[float] = None


class PyTorchNeuronGroupVisualizer:
    """Advanced PyTorch-integrated visualizer for neuron groups and learning patterns."""
    
    def __init__(self, 
                 config=None, 
                 output_dir: str = "data/outputs/pytorch_neuron_groups",
                 device: str = 'auto'):
        """Initialize the PyTorch neuron group visualizer.
        
        Args:
            config: Configuration object
            output_dir: Directory to save visualizations
            device: PyTorch device ('auto', 'cpu', 'cuda')
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set PyTorch device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Using PyTorch device: {self.device}")
        
        # Color schemes for different groups
        self.color_schemes = {
            'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
            'learning': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9CA24', '#6C5CE7',
                        '#A0E7E5', '#FEE75C', '#FF9FF3', '#54A0FF', '#5F27CD'],
            'pytorch': ['#EE4C2C', '#FF9500', '#007ACC', '#00D2FF', '#8B5A2B',
                       '#FF6B35', '#004225', '#7209B7', '#F39C12', '#9B59B6']
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
    
    def extract_activations_from_model(self,
                                     model: nn.Module,
                                     dataloader: DataLoader,
                                     layer_names: List[str],
                                     max_batches: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Extract activations from specific layers of a PyTorch model.
        
        Args:
            model: PyTorch model
            dataloader: DataLoader with input data
            layer_names: Names of layers to extract activations from
            max_batches: Maximum number of batches to process
            
        Returns:
            Dictionary mapping layer names to activation tensors
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available for activation extraction")
            return {}
        
        logger.info(f"Extracting activations from {len(layer_names)} layers")
        
        model.eval()
        model.to(self.device)
        
        # Dictionary to store activations
        activations = {name: [] for name in layer_names}
        
        # Register hooks to capture activations
        hooks = []
        
        def get_activation_hook(name):
            def hook(module, input, output):
                # Store activation on CPU to save GPU memory
                if isinstance(output, torch.Tensor):
                    activations[name].append(output.detach().cpu())
                elif isinstance(output, (list, tuple)):
                    # Handle layers that return multiple outputs
                    activations[name].append(output[0].detach().cpu())
            return hook
        
        # Register hooks for specified layers
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(get_activation_hook(name))
                hooks.append(hook)
                logger.info(f"Registered hook for layer: {name}")
        
        # Extract activations
        with torch.no_grad():
            batch_count = 0
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                # Forward pass to trigger hooks
                _ = model(inputs)
                batch_count += 1
                
                if batch_count % 10 == 0:
                    logger.info(f"Processed {batch_count} batches")
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Concatenate activations from all batches
        final_activations = {}
        for name in layer_names:
            if activations[name]:
                # Concatenate along batch dimension
                concatenated = torch.cat(activations[name], dim=0)
                
                # Flatten spatial dimensions if needed (e.g., for conv layers)
                if concatenated.dim() > 2:
                    batch_size = concatenated.size(0)
                    concatenated = concatenated.view(batch_size, -1)
                
                final_activations[name] = concatenated
                logger.info(f"Layer {name}: {concatenated.shape}")
        
        logger.info(f"Activation extraction completed for {len(final_activations)} layers")
        return final_activations
    
    def identify_neuron_groups_pytorch(self,
                                     activation_tensor: torch.Tensor,
                                     layer_name: str,
                                     method: str = 'correlation_clustering',
                                     n_groups: Optional[int] = None,
                                     correlation_threshold: float = 0.7,
                                     min_group_size: int = 3) -> List[PyTorchNeuronGroup]:
        """Identify neuron groups from PyTorch activation tensors.
        
        Args:
            activation_tensor: Tensor of shape (n_samples, n_neurons)
            layer_name: Name of the layer
            method: Clustering method
            n_groups: Number of groups (auto-detected if None)
            correlation_threshold: Minimum correlation for grouping
            min_group_size: Minimum neurons per group
            
        Returns:
            List of identified PyTorch neuron groups
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available for neuron group identification")
            return []
        
        logger.info(f"Identifying neuron groups in layer {layer_name} using {method}")
        
        # Convert to numpy for clustering (most clustering libraries expect numpy)
        if isinstance(activation_tensor, torch.Tensor):
            activation_np = activation_tensor.cpu().numpy()
        else:
            activation_np = activation_tensor
        
        # Transpose to get neuron correlations across samples
        neuron_data = activation_np.T  # (n_neurons, n_samples)
        
        groups = []
        
        if method == 'correlation_clustering':
            groups = self._pytorch_correlation_clustering(
                neuron_data, activation_tensor, layer_name, 
                correlation_threshold, min_group_size
            )
        elif method == 'kmeans':
            groups = self._pytorch_kmeans_clustering(
                neuron_data, activation_tensor, layer_name,
                n_groups, min_group_size
            )
        elif method == 'hierarchical':
            groups = self._pytorch_hierarchical_clustering(
                neuron_data, activation_tensor, layer_name,
                n_groups, min_group_size
            )
        else:
            logger.error(f"Unknown clustering method: {method}")
            return []
        
        logger.info(f"Identified {len(groups)} neuron groups in layer {layer_name}")
        return groups
    
    def _pytorch_correlation_clustering(self,
                                      neuron_data: np.ndarray,
                                      activation_tensor: torch.Tensor,
                                      layer_name: str,
                                      threshold: float,
                                      min_size: int) -> List[PyTorchNeuronGroup]:
        """PyTorch-specific correlation-based clustering."""
        n_neurons = neuron_data.shape[0]
        
        # Compute correlation matrix using PyTorch
        activation_tensor_normalized = F.normalize(activation_tensor.T, dim=1)
        correlation_matrix = torch.mm(activation_tensor_normalized, activation_tensor_normalized.T)
        correlation_matrix = correlation_matrix.cpu().numpy()
        
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
                
                # Extract group activations as PyTorch tensor
                group_activation_tensor = activation_tensor[:, correlated_neurons]
                group_center = torch.mean(group_activation_tensor, dim=1)
                cohesion = self._calculate_pytorch_cohesion(group_activation_tensor)
                
                group = PyTorchNeuronGroup(
                    group_id=group_id,
                    neuron_indices=correlated_neurons,
                    layer_name=layer_name,
                    activation_pattern=group_activation_tensor,
                    group_center=group_center,
                    group_size=len(correlated_neurons),
                    cohesion_score=cohesion,
                    device=self.device
                )
                
                groups.append(group)
                group_id += 1
        
        return groups
    
    def _pytorch_kmeans_clustering(self,
                                 neuron_data: np.ndarray,
                                 activation_tensor: torch.Tensor,
                                 layer_name: str,
                                 n_groups: Optional[int],
                                 min_size: int) -> List[PyTorchNeuronGroup]:
        """PyTorch-specific K-means clustering."""
        if not SKLEARN_AVAILABLE:
            logger.error("Scikit-learn required for K-means clustering")
            return []
        
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
                group_activation_tensor = activation_tensor[:, neuron_indices]
                group_center = torch.mean(group_activation_tensor, dim=1)
                cohesion = self._calculate_pytorch_cohesion(group_activation_tensor)
                
                group = PyTorchNeuronGroup(
                    group_id=group_id,
                    neuron_indices=neuron_indices,
                    layer_name=layer_name,
                    activation_pattern=group_activation_tensor,
                    group_center=group_center,
                    group_size=len(neuron_indices),
                    cohesion_score=cohesion,
                    device=self.device
                )
                
                groups.append(group)
                group_id += 1
        
        return groups
    
    def _pytorch_hierarchical_clustering(self,
                                       neuron_data: np.ndarray,
                                       activation_tensor: torch.Tensor,
                                       layer_name: str,
                                       n_groups: Optional[int],
                                       min_size: int) -> List[PyTorchNeuronGroup]:
        """PyTorch-specific hierarchical clustering."""
        if not SKLEARN_AVAILABLE:
            logger.error("Scikit-learn required for hierarchical clustering")
            return []
        
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
                group_activation_tensor = activation_tensor[:, neuron_indices]
                group_center = torch.mean(group_activation_tensor, dim=1)
                cohesion = self._calculate_pytorch_cohesion(group_activation_tensor)
                
                group = PyTorchNeuronGroup(
                    group_id=group_id,
                    neuron_indices=neuron_indices,
                    layer_name=layer_name,
                    activation_pattern=group_activation_tensor,
                    group_center=group_center,
                    group_size=len(neuron_indices),
                    cohesion_score=cohesion,
                    device=self.device
                )
                
                groups.append(group)
                group_id += 1
        
        return groups
    
    def _calculate_pytorch_cohesion(self, group_tensor: torch.Tensor) -> float:
        """Calculate cohesion score for a neuron group using PyTorch operations."""
        if group_tensor.size(1) < 2:
            return 1.0
        
        # Normalize group activations
        normalized = F.normalize(group_tensor.T, dim=1)
        
        # Calculate correlation matrix
        correlation_matrix = torch.mm(normalized, normalized.T)
        
        # Get upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones_like(correlation_matrix, dtype=torch.bool), diagonal=1)
        correlation_values = correlation_matrix[mask]
        
        # Return mean correlation as cohesion score
        return float(torch.mean(correlation_values)) if len(correlation_values) > 0 else 0.0
    
    def analyze_pytorch_learning_patterns(self,
                                        activations_dict: Dict[str, torch.Tensor],
                                        neuron_groups_dict: Dict[str, List[PyTorchNeuronGroup]],
                                        question_metadata: Optional[pd.DataFrame] = None,
                                        model_epoch: Optional[int] = None,
                                        loss_values: Optional[List[float]] = None) -> List[PyTorchLearningEvent]:
        """Analyze learning patterns from PyTorch neuron groups across multiple layers.
        
        Args:
            activations_dict: Dictionary mapping layer names to activation tensors
            neuron_groups_dict: Dictionary mapping layer names to neuron group lists
            question_metadata: Optional metadata about questions/tasks
            model_epoch: Current training epoch
            loss_values: Optional loss values for each sample
            
        Returns:
            List of identified PyTorch learning events
        """
        logger.info("Analyzing PyTorch learning patterns from neuron groups")
        
        learning_events = []
        event_id = 0
        
        # Process each layer
        for layer_name, activation_tensor in activations_dict.items():
            if layer_name not in neuron_groups_dict:
                continue
            
            neuron_groups = neuron_groups_dict[layer_name]
            n_samples = activation_tensor.size(0)
            
            # For each sample, determine which groups are active
            for sample_idx in range(n_samples):
                sample_activations = activation_tensor[sample_idx]
                
                # Determine which groups are strongly activated
                activated_groups = []
                group_activations = []
                
                for group in neuron_groups:
                    # Calculate mean activation for this group using PyTorch
                    group_indices = torch.tensor(group.neuron_indices, dtype=torch.long)
                    group_activations_sample = sample_activations[group_indices]
                    group_mean_activation = torch.mean(group_activations_sample)
                    group_activations.append(float(group_mean_activation))
                    
                    # Consider group active if above threshold
                    activation_threshold = torch.mean(sample_activations) + 0.5 * torch.std(sample_activations)
                    if group_mean_activation > activation_threshold:
                        activated_groups.append(group.group_id)
                
                # Create learning event if significant activation
                if activated_groups:
                    learning_strength = max(group_activations)
                    
                    # Determine skill type from metadata if available
                    skill_type = "unknown"
                    if question_metadata is not None and len(question_metadata) > sample_idx:
                        if 'category' in question_metadata.columns:
                            skill_type = question_metadata.iloc[sample_idx]['category']
                        elif 'question' in question_metadata.columns:
                            skill_type = self._infer_skill_type(question_metadata.iloc[sample_idx]['question'])
                    
                    # Get loss value if available
                    loss_value = None
                    if loss_values and len(loss_values) > sample_idx:
                        loss_value = loss_values[sample_idx]
                    
                    event = PyTorchLearningEvent(
                        event_id=event_id,
                        sample_indices=[sample_idx],
                        activated_groups=activated_groups,
                        learning_strength=learning_strength,
                        skill_type=skill_type,
                        temporal_position=sample_idx,
                        model_epoch=model_epoch,
                        loss_value=loss_value
                    )
                    
                    learning_events.append(event)
                    event_id += 1
        
        logger.info(f"Identified {len(learning_events)} PyTorch learning events")
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
    
    def visualize_pytorch_groups(self,
                               activations_dict: Dict[str, torch.Tensor],
                               neuron_groups_dict: Dict[str, List[PyTorchNeuronGroup]],
                               method: str = 'heatmap',
                               layer_name: Optional[str] = None) -> str:
        """Visualize PyTorch neuron groups.
        
        Args:
            activations_dict: Dictionary of activation tensors
            neuron_groups_dict: Dictionary of neuron groups
            method: Visualization method
            layer_name: Specific layer to visualize (if None, visualize all)
            
        Returns:
            Path to saved visualization
        """
        if layer_name:
            # Visualize specific layer
            if layer_name in activations_dict and layer_name in neuron_groups_dict:
                activation_matrix = activations_dict[layer_name].cpu().numpy()
                neuron_groups = neuron_groups_dict[layer_name]
                return self._visualize_single_layer(activation_matrix, neuron_groups, layer_name, method)
            else:
                logger.error(f"Layer {layer_name} not found in activations or groups")
                return ""
        else:
            # Visualize all layers
            return self._visualize_all_layers(activations_dict, neuron_groups_dict, method)
    
    def _visualize_single_layer(self,
                              activation_matrix: np.ndarray,
                              neuron_groups: List[PyTorchNeuronGroup],
                              layer_name: str,
                              method: str) -> str:
        """Visualize neuron groups for a single layer."""
        if method == 'heatmap':
            return self._visualize_pytorch_heatmap(activation_matrix, neuron_groups, layer_name)
        elif method == 'scatter':
            return self._visualize_pytorch_scatter(activation_matrix, neuron_groups, layer_name)
        elif method == 'network':
            return self._visualize_pytorch_network(activation_matrix, neuron_groups, layer_name)
        else:
            logger.error(f"Unknown visualization method: {method}")
            return ""
    
    def _visualize_all_layers(self,
                            activations_dict: Dict[str, torch.Tensor],
                            neuron_groups_dict: Dict[str, List[PyTorchNeuronGroup]],
                            method: str) -> str:
        """Visualize neuron groups across all layers."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for multi-layer visualization")
            return ""
        
        n_layers = len(activations_dict)
        if n_layers == 0:
            return ""
        
        # Create subplot layout
        cols = min(3, n_layers)
        rows = (n_layers + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_layers == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        plot_idx = 0
        for layer_name, activation_tensor in activations_dict.items():
            if layer_name in neuron_groups_dict:
                activation_matrix = activation_tensor.cpu().numpy()
                neuron_groups = neuron_groups_dict[layer_name]
                
                # Create simple group scatter plot for each layer
                if plot_idx < len(axes):
                    ax = axes[plot_idx]
                    self._plot_layer_summary(ax, activation_matrix, neuron_groups, layer_name)
                    plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / f"pytorch_all_layers_{method}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Multi-layer PyTorch visualization saved to {output_path}")
        return str(output_path)
    
    def _plot_layer_summary(self,
                          ax,
                          activation_matrix: np.ndarray,
                          neuron_groups: List[PyTorchNeuronGroup],
                          layer_name: str):
        """Plot summary for a single layer."""
        if not neuron_groups:
            ax.text(0.5, 0.5, f'{layer_name}\nNo groups found', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(layer_name)
            return
        
        # Simple visualization: group sizes and cohesion
        group_sizes = [group.group_size for group in neuron_groups]
        group_cohesions = [group.cohesion_score for group in neuron_groups]
        group_ids = [f'G{group.group_id}' for group in neuron_groups]
        
        colors = self.color_schemes['pytorch'][:len(neuron_groups)]
        
        # Scatter plot: size vs cohesion
        scatter = ax.scatter(group_sizes, group_cohesions, 
                           c=colors[:len(neuron_groups)], 
                           s=100, alpha=0.7)
        
        # Add group labels
        for i, (size, cohesion, group_id) in enumerate(zip(group_sizes, group_cohesions, group_ids)):
            ax.annotate(group_id, (size, cohesion), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Group Size')
        ax.set_ylabel('Cohesion Score')
        ax.set_title(f'{layer_name}\n{len(neuron_groups)} groups')
        ax.grid(True, alpha=0.3)
    
    def _visualize_pytorch_heatmap(self,
                                 activation_matrix: np.ndarray,
                                 neuron_groups: List[PyTorchNeuronGroup],
                                 layer_name: str) -> str:
        """Create PyTorch-specific heatmap visualization."""
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
        
        # Top plot: Group labels with PyTorch colors
        colors = self.color_schemes['pytorch'][:len(neuron_groups)]
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
        ax1.set_title(f'PyTorch Neuron Groups - {layer_name}', fontsize=14, fontweight='bold')
        
        # Add group boundaries
        for boundary in group_boundaries[:-1]:
            ax1.axvline(x=boundary - 0.5, color='white', linewidth=2)
        
        # Main heatmap
        sns.heatmap(
            reordered_matrix[:50, :],  # Limit to first 50 samples
            ax=ax2,
            cmap='plasma',  # PyTorch-like colormap
            cbar_kws={'label': 'Activation Strength'},
            xticklabels=False,
            yticklabels=False
        )
        
        ax2.set_title(f'PyTorch Neuron Group Activation Patterns - {layer_name}', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Neurons (Grouped)', fontsize=12)
        ax2.set_ylabel('Samples', fontsize=12)
        
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
        output_path = self.output_dir / f"pytorch_groups_heatmap_{layer_name.replace('.', '_')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"PyTorch neuron groups heatmap saved to {output_path}")
        return str(output_path)
    
    def _visualize_pytorch_scatter(self,
                                 activation_matrix: np.ndarray,
                                 neuron_groups: List[PyTorchNeuronGroup],
                                 layer_name: str) -> str:
        """Create PyTorch-specific scatter visualization using PCA."""
        if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            logger.warning("Scikit-learn/Matplotlib not available for scatter visualization")
            return ""
        
        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=2)
        activation_2d = pca.fit_transform(activation_matrix)
        
        plt.figure(figsize=(12, 10))
        
        # Use PyTorch color scheme
        colors = self.color_schemes['pytorch']
        
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
        plt.title(f'PyTorch Neuron Groups in 2D Activation Space - {layer_name}', 
                 fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_path = self.output_dir / f"pytorch_groups_scatter_{layer_name.replace('.', '_')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"PyTorch neuron groups scatter plot saved to {output_path}")
        return str(output_path)
    
    def create_pytorch_interactive_dashboard(self,
                                           activations_dict: Dict[str, torch.Tensor],
                                           neuron_groups_dict: Dict[str, List[PyTorchNeuronGroup]],
                                           learning_events: List[PyTorchLearningEvent],
                                           question_metadata: Optional[pd.DataFrame] = None) -> str:
        """Create interactive PyTorch dashboard for neuron groups and learning patterns."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for interactive dashboard")
            return ""
        
        # Calculate dashboard data
        layer_names = list(activations_dict.keys())
        n_layers = len(layer_names)
        
        # Create subplots layout
        subplot_specs = []
        subplot_titles = []
        
        if n_layers > 0:
            subplot_specs.append([{"colspan": 2}, None])
            subplot_titles.extend(['Layer-wise Group Analysis', ''])
            
            subplot_specs.append([{"type": "pie"}, {"type": "bar"}])
            subplot_titles.extend(['Group Size Distribution', 'Cohesion Scores'])
            
            if learning_events:
                subplot_specs.append([{"colspan": 2}, None])
                subplot_titles.extend(['Learning Events Timeline', ''])
        
        fig = make_subplots(
            rows=len(subplot_specs), cols=2,
            subplot_titles=subplot_titles,
            specs=subplot_specs,
            vertical_spacing=0.15
        )
        
        # PyTorch color scheme
        colors = self.color_schemes['pytorch']
        
        row = 1
        
        # 1. Layer-wise group analysis
        if n_layers > 0:
            for i, layer_name in enumerate(layer_names):
                if layer_name in neuron_groups_dict:
                    groups = neuron_groups_dict[layer_name]
                    activation_tensor = activations_dict[layer_name]
                    
                    # Calculate mean activation per group over time
                    group_means = []
                    for group in groups:
                        group_indices = torch.tensor(group.neuron_indices, dtype=torch.long)
                        group_activations = torch.mean(activation_tensor[:, group_indices], dim=1)
                        group_means.append(group_activations.cpu().numpy())
                    
                    # Plot each group's activation timeline
                    for j, group in enumerate(groups):
                        fig.add_trace(
                            go.Scatter(
                                x=list(range(len(group_means[j]))),
                                y=group_means[j],
                                mode='lines',
                                name=f'{layer_name} - Group {group.group_id}',
                                line=dict(color=colors[j % len(colors)]),
                                hovertemplate=f'{layer_name} - Group {group.group_id}<br>'
                                           f'Sample: %{{x}}<br>Activation: %{{y:.3f}}<extra></extra>'
                            ),
                            row=row, col=1
                        )
            row += 1
        
        # 2. Group size distribution (pie chart)
        if neuron_groups_dict:
            all_groups = []
            for groups in neuron_groups_dict.values():
                all_groups.extend(groups)
            
            if all_groups:
                group_sizes = [group.group_size for group in all_groups]
                group_labels = [f'{group.layer_name} - G{group.group_id}' for group in all_groups]
                
                fig.add_trace(
                    go.Pie(
                        labels=group_labels,
                        values=group_sizes,
                        name="Group Sizes",
                        hovertemplate='%{label}<br>Size: %{value} neurons<br>Percentage: %{percent}<extra></extra>'
                    ),
                    row=row, col=1
                )
                
                # 3. Cohesion scores (bar chart)
                cohesion_scores = [group.cohesion_score for group in all_groups]
                
                fig.add_trace(
                    go.Bar(
                        x=group_labels,
                        y=cohesion_scores,
                        name="Cohesion Scores",
                        marker_color=colors[:len(all_groups)],
                        hovertemplate='%{x}<br>Cohesion: %{y:.3f}<extra></extra>'
                    ),
                    row=row, col=2
                )
            row += 1
        
        # 4. Learning events timeline
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
                    row=row, col=1
                )
        
        # Update layout
        fig.update_layout(
            title_text="PyTorch Neuron Group Analysis Dashboard",
            height=800 if learning_events else 600,
            showlegend=True,
            hovermode='closest'
        )
        
        # Update axis labels
        if n_layers > 0:
            fig.update_xaxes(title_text="Sample/Time", row=1, col=1)
            fig.update_yaxes(title_text="Mean Group Activation", row=1, col=1)
        
        if learning_events:
            fig.update_xaxes(title_text="Learning Event Position", row=row, col=1)
            fig.update_yaxes(title_text="Learning Strength", row=row, col=1)
        
        # Save interactive dashboard
        output_path = self.output_dir / "pytorch_interactive_dashboard.html"
        fig.write_html(str(output_path))
        
        logger.info(f"PyTorch interactive dashboard saved to {output_path}")
        return str(output_path)
    
    def generate_pytorch_report(self,
                              activations_dict: Dict[str, torch.Tensor],
                              neuron_groups_dict: Dict[str, List[PyTorchNeuronGroup]],
                              learning_events: List[PyTorchLearningEvent],
                              model_info: Optional[Dict[str, Any]] = None,
                              output_format: str = 'json') -> str:
        """Generate comprehensive PyTorch analysis report.
        
        Args:
            activations_dict: Dictionary of activation tensors
            neuron_groups_dict: Dictionary of neuron groups per layer
            learning_events: List of learning events
            model_info: Optional model information
            output_format: Format for report ('json', 'text')
            
        Returns:
            Path to saved report
        """
        logger.info("Generating PyTorch neuron group analysis report")
        
        # Compile analysis results
        report = {
            'pytorch_analysis': {
                'device': self.device,
                'torch_version': torch.__version__ if TORCH_AVAILABLE else 'N/A',
                'total_layers': len(activations_dict),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            },
            'model_info': model_info or {},
            'layer_analysis': {},
            'learning_patterns': {},
            'statistics': {}
        }
        
        # Analyze each layer
        total_groups = 0
        total_neurons = 0
        
        for layer_name, activation_tensor in activations_dict.items():
            layer_stats = {
                'activation_shape': list(activation_tensor.shape),
                'device': str(activation_tensor.device),
                'dtype': str(activation_tensor.dtype),
                'groups': []
            }
            
            if layer_name in neuron_groups_dict:
                groups = neuron_groups_dict[layer_name]
                total_groups += len(groups)
                
                for group in groups:
                    group_data = {
                        'group_id': group.group_id,
                        'neuron_count': group.group_size,
                        'neuron_indices': group.neuron_indices,
                        'cohesion_score': float(group.cohesion_score),
                        'mean_activation': float(torch.mean(group.activation_pattern)),
                        'activation_variance': float(torch.var(group.activation_pattern))
                    }
                    layer_stats['groups'].append(group_data)
                    total_neurons += group.group_size
            
            report['layer_analysis'][layer_name] = layer_stats
        
        # Learning patterns analysis
        if learning_events:
            skill_types = [event.skill_type for event in learning_events]
            skill_counts = pd.Series(skill_types).value_counts().to_dict()
            
            # Analyze epochs if available
            epochs = [event.model_epoch for event in learning_events if event.model_epoch is not None]
            epoch_analysis = {}
            if epochs:
                epoch_analysis = {
                    'epoch_range': [min(epochs), max(epochs)],
                    'total_epochs': len(set(epochs)),
                    'events_per_epoch': len(learning_events) / len(set(epochs))
                }
            
            # Analyze loss values if available
            loss_analysis = {}
            loss_values = [event.loss_value for event in learning_events if event.loss_value is not None]
            if loss_values:
                loss_analysis = {
                    'mean_loss': float(np.mean(loss_values)),
                    'loss_std': float(np.std(loss_values)),
                    'min_loss': float(min(loss_values)),
                    'max_loss': float(max(loss_values))
                }
            
            report['learning_patterns'] = {
                'skill_distribution': skill_counts,
                'total_events': len(learning_events),
                'average_learning_strength': float(np.mean([e.learning_strength for e in learning_events])),
                'epoch_analysis': epoch_analysis,
                'loss_analysis': loss_analysis
            }
        
        # Overall statistics
        report['statistics'] = {
            'total_groups': total_groups,
            'total_neurons_in_groups': total_neurons,
            'average_group_size': total_neurons / total_groups if total_groups > 0 else 0,
            'layers_analyzed': len(activations_dict),
            'device_used': self.device
        }
        
        # Save report
        if output_format == 'json':
            output_path = self.output_dir / "pytorch_neuron_group_analysis_report.json"
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        else:
            output_path = self.output_dir / "pytorch_neuron_group_analysis_report.txt"
            self._save_pytorch_text_report(report, output_path)
        
        logger.info(f"PyTorch analysis report saved to {output_path}")
        return str(output_path)
    
    def _save_pytorch_text_report(self, report: Dict[str, Any], output_path: Path):
        """Save PyTorch report in text format."""
        with open(output_path, 'w') as f:
            f.write("PyTorch Neuron Group Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            # PyTorch analysis info
            f.write("PYTORCH ANALYSIS INFO\n")
            f.write("-" * 30 + "\n")
            pytorch_info = report['pytorch_analysis']
            f.write(f"Device: {pytorch_info['device']}\n")
            f.write(f"PyTorch Version: {pytorch_info['torch_version']}\n")
            f.write(f"Total Layers: {pytorch_info['total_layers']}\n")
            f.write(f"Analysis Time: {pytorch_info['analysis_timestamp']}\n\n")
            
            # Model info
            if report['model_info']:
                f.write("MODEL INFORMATION\n")
                f.write("-" * 30 + "\n")
                for key, value in report['model_info'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Layer analysis
            f.write("LAYER ANALYSIS\n")
            f.write("-" * 30 + "\n")
            for layer_name, layer_stats in report['layer_analysis'].items():
                f.write(f"\nLayer: {layer_name}\n")
                f.write(f"  Activation Shape: {layer_stats['activation_shape']}\n")
                f.write(f"  Device: {layer_stats['device']}\n")
                f.write(f"  Data Type: {layer_stats['dtype']}\n")
                f.write(f"  Groups Found: {len(layer_stats['groups'])}\n")
                
                for group in layer_stats['groups']:
                    f.write(f"    Group {group['group_id']}:\n")
                    f.write(f"      Neurons: {group['neuron_count']}\n")
                    f.write(f"      Cohesion: {group['cohesion_score']:.3f}\n")
                    f.write(f"      Mean Activation: {group['mean_activation']:.3f}\n")
            
            # Learning patterns
            if 'learning_patterns' in report and report['learning_patterns']:
                f.write("\nLEARNING PATTERNS\n")
                f.write("-" * 30 + "\n")
                patterns = report['learning_patterns']
                f.write(f"Total Learning Events: {patterns['total_events']}\n")
                f.write(f"Average Learning Strength: {patterns['average_learning_strength']:.3f}\n")
                
                if patterns.get('epoch_analysis'):
                    epoch_info = patterns['epoch_analysis']
                    f.write(f"Epoch Range: {epoch_info['epoch_range']}\n")
                    f.write(f"Events per Epoch: {epoch_info['events_per_epoch']:.2f}\n")
                
                if patterns.get('loss_analysis'):
                    loss_info = patterns['loss_analysis']
                    f.write(f"Mean Loss: {loss_info['mean_loss']:.4f}\n")
                    f.write(f"Loss Range: {loss_info['min_loss']:.4f} - {loss_info['max_loss']:.4f}\n")
                
                f.write("Skill Distribution:\n")
                for skill, count in patterns['skill_distribution'].items():
                    f.write(f"  {skill}: {count}\n")
            
            # Statistics
            f.write("\nSTATISTICS\n")
            f.write("-" * 30 + "\n")
            stats = report['statistics']
            f.write(f"Total Groups: {stats['total_groups']}\n")
            f.write(f"Total Neurons in Groups: {stats['total_neurons_in_groups']}\n")
            f.write(f"Average Group Size: {stats['average_group_size']:.1f}\n")
            f.write(f"Layers Analyzed: {stats['layers_analyzed']}\n")
            f.write(f"Device Used: {stats['device_used']}\n")


def create_pytorch_neuron_group_analysis(model: nn.Module,
                                        dataloader: DataLoader,
                                        layer_names: List[str],
                                        question_metadata: Optional[pd.DataFrame] = None,
                                        output_dir: str = "data/outputs/pytorch_neuron_groups",
                                        device: str = 'auto',
                                        max_batches: Optional[int] = None,
                                        model_epoch: Optional[int] = None,
                                        model_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function to run complete PyTorch neuron group analysis.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with input data
        layer_names: Names of layers to analyze
        question_metadata: Optional metadata about questions/tasks
        output_dir: Directory to save results
        device: PyTorch device ('auto', 'cpu', 'cuda')
        max_batches: Maximum number of batches to process
        model_epoch: Current training epoch
        model_info: Optional model information
        
    Returns:
        Dictionary containing analysis results and file paths
    """
    if not TORCH_AVAILABLE:
        logger.error("PyTorch not available for analysis")
        return {}
    
    logger.info("Starting complete PyTorch neuron group analysis")
    
    # Initialize visualizer
    visualizer = PyTorchNeuronGroupVisualizer(output_dir=output_dir, device=device)
    
    # Extract activations from model
    activations_dict = visualizer.extract_activations_from_model(
        model, dataloader, layer_names, max_batches
    )
    
    if not activations_dict:
        logger.error("No activations extracted")
        return {}
    
    # Identify neuron groups for each layer
    neuron_groups_dict = {}
    for layer_name, activation_tensor in activations_dict.items():
        groups = visualizer.identify_neuron_groups_pytorch(
            activation_tensor, layer_name,
            method='correlation_clustering',
            correlation_threshold=0.6,
            min_group_size=3
        )
        neuron_groups_dict[layer_name] = groups
    
    # Analyze learning patterns
    learning_events = visualizer.analyze_pytorch_learning_patterns(
        activations_dict, neuron_groups_dict, question_metadata, model_epoch
    )
    
    # Create visualizations
    visualization_paths = {}
    
    # Multi-layer visualization
    multi_layer_path = visualizer.visualize_pytorch_groups(
        activations_dict, neuron_groups_dict, method='heatmap'
    )
    visualization_paths['multi_layer'] = multi_layer_path
    
    # Individual layer visualizations
    for layer_name in layer_names:
        if layer_name in activations_dict:
            layer_path = visualizer.visualize_pytorch_groups(
                activations_dict, neuron_groups_dict, 
                method='scatter', layer_name=layer_name
            )
            visualization_paths[f'{layer_name}_scatter'] = layer_path
    
    # Create interactive dashboard
    dashboard_path = visualizer.create_pytorch_interactive_dashboard(
        activations_dict, neuron_groups_dict, learning_events, question_metadata
    )
    
    # Generate analysis report
    report_path = visualizer.generate_pytorch_report(
        activations_dict, neuron_groups_dict, learning_events, model_info
    )
    
    # Calculate summary statistics
    total_groups = sum(len(groups) for groups in neuron_groups_dict.values())
    total_learning_events = len(learning_events)
    
    results = {
        'activations': activations_dict,
        'neuron_groups': neuron_groups_dict,
        'learning_events': learning_events,
        'visualizations': visualization_paths,
        'dashboard': dashboard_path,
        'report': report_path,
        'summary': {
            'total_layers': len(activations_dict),
            'total_groups': total_groups,
            'total_learning_events': total_learning_events,
            'device_used': visualizer.device,
            'analysis_complete': True
        }
    }
    
    logger.info("PyTorch neuron group analysis completed successfully")
    logger.info(f"Analyzed {len(activations_dict)} layers, found {total_groups} groups, "
               f"identified {total_learning_events} learning events")
    
    return results
