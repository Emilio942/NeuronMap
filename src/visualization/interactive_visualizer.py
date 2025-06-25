"""Interactive visualization tools for neural network activation analysis."""

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import h5py
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import networkx as nx
from datetime import datetime

from ..utils.config import get_config
from ..utils.error_handling import with_retry, safe_execute


logger = logging.getLogger(__name__)


class InteractiveVisualizer:
    """Create interactive visualizations for activation analysis."""

    def __init__(self, config_name: str = "default", output_dir: str = None):
        """Initialize the interactive visualizer.

        Args:
            config_name: Name of experiment configuration.
            output_dir: Output directory (for test compatibility)
        """
        self.config = get_config()
        self.experiment_config = self.config.get_experiment_config(config_name)

        # Store output_dir for test compatibility
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True)
        else:
            self.output_dir = Path("outputs")
        self.experiment_config = self.config.get_experiment_config(config_name)

        # Store output_dir for test compatibility
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True)
        else:
            self.output_dir = Path("outputs")

        # Enhanced visualization themes
        self.themes = {
            'neuronmap': {
                'background': '#f8f9fa',
                'paper_bgcolor': 'white',
                'plot_bgcolor': 'white',
                'colorscale': 'Viridis',
                'font_color': '#2c3e50'
            },
            'dark': {
                'background': '#2c3e50',
                'paper_bgcolor': '#34495e',
                'plot_bgcolor': '#34495e',
                'colorscale': 'Plasma',
                'font_color': '#ecf0f1'
            },
            'scientific': {
                'background': '#ffffff',
                'paper_bgcolor': 'white',
                'plot_bgcolor': 'white',
                'colorscale': 'RdBu',
                'font_color': '#000000'
            }
        }
        self.current_theme = 'neuronmap'

    def load_activation_data(self, filepath: str) -> Dict[str, Any]:
        """Load activation data from HDF5 file.

        Args:
            filepath: Path to HDF5 activation file.

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

        logger.info(f"Loaded data: {len(data['questions'])} questions, {len(data['activations'])} layers")
        return data

    def create_activation_heatmap(self, data: Dict[str, Any],
                                layer_name: str) -> go.Figure:
        """Create interactive heatmap of activations.

        Args:
            data: Loaded activation data.
            layer_name: Name of layer to visualize.

        Returns:
            Plotly figure with interactive heatmap.
        """
        if layer_name not in data['activations']:
            raise ValueError(f"Layer {layer_name} not found in data")

        layer_data = data['activations'][layer_name]

        # Create activation matrix
        activation_matrix = np.array([item['vector'] for item in layer_data])
        questions = [data['questions'][item['question_idx']] for item in layer_data]

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=activation_matrix,
            x=[f"Neuron {i}" for i in range(activation_matrix.shape[1])],
            y=[f"Q{i}: {q[:50]}..." if len(q) > 50 else f"Q{i}: {q}"
               for i, q in enumerate(questions)],
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Activation Strength"),
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>' +
                         'Neuron: %{x}<br>' +
                         'Activation: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=f"Activation Heatmap - {layer_name}",
            xaxis_title="Neurons",
            yaxis_title="Questions",
            width=1000,
            height=600,
            font=dict(size=10)
        )

        return fig

    def create_dimensionality_reduction_plot(self, data: Dict[str, Any],
                                           layer_name: str,
                                           method: str = "pca",
                                           n_components: int = 2) -> go.Figure:
        """Create interactive dimensionality reduction plot.

        Args:
            data: Loaded activation data.
            layer_name: Name of layer to visualize.
            method: Reduction method ('pca', 'tsne', 'umap').
            n_components: Number of components (2 or 3).

        Returns:
            Plotly figure with interactive plot.
        """
        if layer_name not in data['activations']:
            raise ValueError(f"Layer {layer_name} not found in data")

        layer_data = data['activations'][layer_name]
        activation_matrix = np.array([item['vector'] for item in layer_data])
        questions = [data['questions'][item['question_idx']] for item in layer_data]

        # Perform dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components)
            reduced_data = reducer.fit_transform(activation_matrix)
            explained_var = reducer.explained_variance_ratio_
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(questions)-1))
            reduced_data = reducer.fit_transform(activation_matrix)
            explained_var = None
        elif method.lower() == 'umap':
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            reduced_data = reducer.fit_transform(activation_matrix)
            explained_var = None
        else:
            raise ValueError(f"Unsupported method: {method}")

        # Create dataframe for plotting
        plot_df = pd.DataFrame({
            'x': reduced_data[:, 0],
            'y': reduced_data[:, 1],
            'question': questions,
            'question_idx': [item['question_idx'] for item in layer_data],
            'question_short': [f"Q{item['question_idx']}: {data['questions'][item['question_idx']][:30]}..."
                              for item in layer_data]
        })

        if n_components == 3:
            plot_df['z'] = reduced_data[:, 2]

        # Create plot
        if n_components == 2:
            fig = px.scatter(
                plot_df,
                x='x', y='y',
                hover_data=['question'],
                title=f"{method.upper()} - {layer_name}",
                labels={'x': f'{method.upper()} 1', 'y': f'{method.upper()} 2'}
            )

            # Add question labels on hover
            fig.update_traces(
                hovertemplate='<b>%{customdata[0]}</b><br>' +
                             f'{method.upper()} 1: %{{x:.3f}}<br>' +
                             f'{method.upper()} 2: %{{y:.3f}}<extra></extra>',
                customdata=plot_df[['question']]
            )

        else:  # 3D plot
            fig = px.scatter_3d(
                plot_df,
                x='x', y='y', z='z',
                hover_data=['question'],
                title=f"{method.upper()} 3D - {layer_name}",
                labels={'x': f'{method.upper()} 1', 'y': f'{method.upper()} 2', 'z': f'{method.upper()} 3'}
            )

        # Add explained variance to title if available
        if explained_var is not None:
            if n_components == 2:
                title_suffix = f" (Explained Variance: {explained_var[0]:.1%}, {explained_var[1]:.1%})"
            else:
                title_suffix = f" (Total Explained Variance: {sum(explained_var):.1%})"

            fig.update_layout(title=fig.layout.title.text + title_suffix)

        fig.update_layout(
            width=800,
            height=600,
            font=dict(size=12)
        )

        return fig

    def create_layer_comparison_plot(self, data: Dict[str, Any],
                                   layer_names: List[str]) -> go.Figure:
        """Create interactive comparison plot between layers.

        Args:
            data: Loaded activation data.
            layer_names: List of layer names to compare.

        Returns:
            Plotly figure with layer comparison.
        """
        # Calculate statistics for each layer
        layer_stats = {}

        for layer_name in layer_names:
            if layer_name not in data['activations']:
                logger.warning(f"Layer {layer_name} not found, skipping")
                continue

            layer_data = data['activations'][layer_name]
            activation_matrix = np.array([item['vector'] for item in layer_data])

            # Calculate statistics
            stats = {
                'mean': np.mean(activation_matrix, axis=1),  # Mean per question
                'std': np.std(activation_matrix, axis=1),   # Std per question
                'sparsity': np.mean(activation_matrix == 0, axis=1),  # Sparsity per question
                'max_activation': np.max(activation_matrix, axis=1)   # Max per question
            }

            layer_stats[layer_name] = stats

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mean Activation', 'Standard Deviation', 'Sparsity', 'Max Activation'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        colors = px.colors.qualitative.Set1[:len(layer_stats)]

        # Plot each statistic
        for i, (stat_name, subplot_pos) in enumerate([
            ('mean', (1, 1)), ('std', (1, 2)), ('sparsity', (2, 1)), ('max_activation', (2, 2))
        ]):
            for j, (layer_name, stats) in enumerate(layer_stats.items()):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(stats[stat_name]))),
                        y=stats[stat_name],
                        mode='lines+markers',
                        name=f"{layer_name} - {stat_name}",
                        line=dict(color=colors[j % len(colors)]),
                        showlegend=(i == 0),  # Only show legend for first subplot
                        hovertemplate=f'<b>{layer_name}</b><br>' +
                                     f'Question: %{{x}}<br>' +
                                     f'{stat_name}: %{{y:.3f}}<extra></extra>'
                    ),
                    row=subplot_pos[0], col=subplot_pos[1]
                )

        fig.update_layout(
            title="Layer Comparison - Activation Statistics",
            height=600,
            width=1000,
            font=dict(size=10)
        )

        # Update axis labels
        fig.update_xaxes(title_text="Question Index")
        fig.update_yaxes(title_text="Value")

        return fig

    def create_neuron_activity_distribution(self, data: Dict[str, Any],
                                          layer_name: str) -> go.Figure:
        """Create interactive distribution plot of neuron activities.

        Args:
            data: Loaded activation data.
            layer_name: Name of layer to analyze.

        Returns:
            Plotly figure with neuron activity distributions.
        """
        if layer_name not in data['activations']:
            raise ValueError(f"Layer {layer_name} not found in data")

        layer_data = data['activations'][layer_name]
        activation_matrix = np.array([item['vector'] for item in layer_data])

        # Calculate neuron-wise statistics
        neuron_means = np.mean(activation_matrix, axis=0)
        neuron_stds = np.std(activation_matrix, axis=0)
        neuron_sparsity = np.mean(activation_matrix == 0, axis=0)

        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Mean Activation per Neuron',
                'Standard Deviation per Neuron',
                'Sparsity per Neuron',
                'Overall Activation Distribution'
            )
        )

        # Mean activation histogram
        fig.add_trace(
            go.Histogram(
                x=neuron_means,
                nbinsx=50,
                name="Mean Activation",
                opacity=0.7,
                hovertemplate='Mean Activation: %{x:.3f}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )

        # Standard deviation histogram
        fig.add_trace(
            go.Histogram(
                x=neuron_stds,
                nbinsx=50,
                name="Std Deviation",
                opacity=0.7,
                hovertemplate='Std Deviation: %{x:.3f}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=2
        )

        # Sparsity histogram
        fig.add_trace(
            go.Histogram(
                x=neuron_sparsity,
                nbinsx=50,
                name="Sparsity",
                opacity=0.7,
                hovertemplate='Sparsity: %{x:.3f}<br>Count: %{y}<extra></extra>'
            ),
            row=2, col=1
        )

        # Overall activation distribution
        fig.add_trace(
            go.Histogram(
                x=activation_matrix.flatten(),
                nbinsx=100,
                name="All Activations",
                opacity=0.7,
                hovertemplate='Activation: %{x:.3f}<br>Count: %{y}<extra></extra>'
            ),
            row=2, col=2
        )

        fig.update_layout(
            title=f"Neuron Activity Analysis - {layer_name}",
            height=700,
            width=1000,
            showlegend=False,
            font=dict(size=10)
        )

        return fig

    def create_activation_animation(self, data: Dict[str, Any],
                                  layer_names: List[str]) -> go.Figure:
        """Create animated plot showing activation evolution across layers.

        Args:
            data: Loaded activation data.
            layer_names: List of layer names to animate through.

        Returns:
            Plotly figure with animation.
        """
        # Prepare data for animation
        frames = []

        for layer_name in layer_names:
            if layer_name not in data['activations']:
                continue

            layer_data = data['activations'][layer_name]
            activation_matrix = np.array([item['vector'] for item in layer_data])

            # Use PCA for dimensionality reduction
            if activation_matrix.shape[1] > 2:
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(activation_matrix)
            else:
                reduced_data = activation_matrix

            # Create frame
            frame = go.Frame(
                data=[go.Scatter(
                    x=reduced_data[:, 0],
                    y=reduced_data[:, 1],
                    mode='markers+text',
                    text=[f"Q{i}" for i in range(len(reduced_data))],
                    textposition="top center",
                    marker=dict(
                        size=10,
                        color=np.arange(len(reduced_data)),
                        colorscale='viridis',
                        showscale=True
                    ),
                    hovertemplate='Question %{text}<br>' +
                                 'PC1: %{x:.3f}<br>' +
                                 'PC2: %{y:.3f}<extra></extra>'
                )],
                name=layer_name,
                layout=go.Layout(title_text=f"Activation Evolution - {layer_name}")
            )
            frames.append(frame)

        # Create initial plot
        if frames:
            fig = go.Figure(
                data=frames[0].data,
                frames=frames
            )

            # Add animation controls
            fig.update_layout(
                title="Activation Evolution Across Layers",
                xaxis_title="Principal Component 1",
                yaxis_title="Principal Component 2",
                width=800,
                height=600,
                updatemenus=[dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(label="Play",
                             method="animate",
                             args=[None, {"frame": {"duration": 1000, "redraw": True},
                                         "fromcurrent": True, "transition": {"duration": 300}}]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], {"frame": {"duration": 0, "redraw": True},
                                           "mode": "immediate",
                                           "transition": {"duration": 0}}])
                    ]),
                    pad={"r": 10, "t": 87},
                    showactive=False,
                    x=0.011,
                    xanchor="right",
                    y=0,
                    yanchor="top"
                )],
                sliders=[dict(
                    active=0,
                    yanchor="top",
                    xanchor="left",
                    currentvalue={"font": {"size": 20}, "prefix": "Layer: ",
                                "visible": True, "xanchor": "right"},
                    transition={"duration": 300, "easing": "cubic-in-out"},
                    pad={"b": 10, "t": 50},
                    len=0.9,
                    x=0.1,
                    y=0,
                    steps=[dict(args=[[f.name],
                                     {"frame": {"duration": 300, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 300}}],
                               label=f.name,
                               method="animate") for f in frames]
                )]
            )

            return fig
        else:
            # Return empty figure if no valid layers
            return go.Figure().add_annotation(
                text="No valid layers found for animation",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

    def save_interactive_html(self, fig: go.Figure, filepath: str):
        """Save interactive plot as HTML file.

        Args:
            fig: Plotly figure to save.
            filepath: Output HTML file path.
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.write_html(str(output_path), include_plotlyjs=True)
        logger.info(f"Interactive plot saved to: {output_path}")

    def create_dashboard(self, data: Dict[str, Any], output_dir: str = "data/outputs/dashboard"):
        """Create a comprehensive interactive dashboard.

        Args:
            data: Loaded activation data.
            output_dir: Directory to save dashboard files.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        layer_names = list(data['activations'].keys())

        # Create different visualizations
        visualizations = {}

        # Heatmaps for each layer
        for layer_name in layer_names[:3]:  # Limit to first 3 layers for demo
            heatmap = self.create_activation_heatmap(data, layer_name)
            visualizations[f'heatmap_{layer_name.replace(".", "_")}'] = heatmap

        # Dimensionality reduction plots
        if layer_names:
            for method in ['pca', 'tsne']:
                dim_plot = self.create_dimensionality_reduction_plot(data, layer_names[0], method)
                visualizations[f'{method}_{layer_names[0].replace(".", "_")}'] = dim_plot

        # Layer comparison
        if len(layer_names) > 1:
            comparison = self.create_layer_comparison_plot(data, layer_names[:3])
            visualizations['layer_comparison'] = comparison

        # Neuron distributions
        if layer_names:
            neuron_dist = self.create_neuron_activity_distribution(data, layer_names[0])
            visualizations['neuron_distribution'] = neuron_dist

        # Animation
        if len(layer_names) > 1:
            animation = self.create_activation_animation(data, layer_names)
            visualizations['layer_animation'] = animation

        # Save all visualizations
        for name, fig in visualizations.items():
            self.save_interactive_html(fig, output_path / f"{name}.html")

        # Create index HTML file
        self._create_dashboard_index(visualizations.keys(), output_path)

        logger.info(f"Dashboard created with {len(visualizations)} visualizations in {output_path}")

    def _create_dashboard_index(self, visualization_names: List[str], output_path: Path):
        """Create HTML index page for dashboard.

        Args:
            visualization_names: List of visualization file names.
            output_path: Output directory path.
        """
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuronMap Interactive Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .header { text-align: center; margin-bottom: 30px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .card h3 { margin-top: 0; color: #333; }
        .card a { display: inline-block; background: #007bff; color: white; padding: 10px 20px;
                  text-decoration: none; border-radius: 4px; margin-top: 10px; }
        .card a:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 NeuronMap Interactive Dashboard</h1>
        <p>Explore neural network activation patterns through interactive visualizations</p>
    </div>

    <div class="grid">
"""

        # Add cards for each visualization
        viz_descriptions = {
            'heatmap': 'Activation Heatmap - View activation patterns across neurons and questions',
            'pca': 'PCA Analysis - Principal component analysis of activation space',
            'tsne': 't-SNE Analysis - Non-linear dimensionality reduction visualization',
            'layer_comparison': 'Layer Comparison - Compare statistics across different layers',
            'neuron_distribution': 'Neuron Analysis - Distribution of neuron activities',
            'layer_animation': 'Layer Animation - Animated evolution across network layers'
        }

        for viz_name in visualization_names:
            # Determine visualization type and description
            viz_type = None
            for key in viz_descriptions:
                if key in viz_name:
                    viz_type = key
                    break

            if viz_type:
                description = viz_descriptions[viz_type]
                title = viz_name.replace('_', ' ').title()
            else:
                description = f"Interactive visualization: {viz_name}"
                title = viz_name.replace('_', ' ').title()

            html_content += f"""
        <div class="card">
            <h3>{title}</h3>
            <p>{description}</p>
            <a href="{viz_name}.html" target="_blank">Open Visualization</a>
        </div>
"""

        html_content += """
    </div>

    <div style="text-align: center; margin-top: 40px; color: #666;">
        <p>Generated by NeuronMap - Neural Network Activation Analysis Toolkit</p>
    </div>
</body>
</html>
"""

        index_file = output_path / "index.html"
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Dashboard index created: {index_file}")

    def create_3d_activation_landscape(self,
                                     activations: np.ndarray,
                                     layer_names: List[str],
                                     questions: List[str],
                                     title: str = "3D Activation Landscape") -> str:
        """Create interactive 3D visualization of activation patterns."""
        try:
            # Prepare data for 3D visualization
            if activations.ndim > 2:
                # Flatten higher dimensional activations
                activations = activations.reshape(activations.shape[0], -1)

            # Create 3D surface plot
            fig = go.Figure()

            # Add surface plot
            z_data = activations[:min(50, len(activations))]  # Limit for performance

            fig.add_trace(go.Surface(
                z=z_data,
                colorscale=self.themes[self.current_theme]['colorscale'],
                showscale=True,
                hovertemplate='Layer: %{y}<br>Neuron: %{x}<br>Activation: %{z:.4f}<extra></extra>'
            ))

            # Update layout
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                scene=dict(
                    xaxis_title="Neuron Index",
                    yaxis_title="Question Index",
                    zaxis_title="Activation Value",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                paper_bgcolor=self.themes[self.current_theme]['paper_bgcolor'],
                plot_bgcolor=self.themes[self.current_theme]['plot_bgcolor'],
                font_color=self.themes[self.current_theme]['font_color'],
                height=600
            )

            # Save to file
            output_file = Path(f"data/outputs/visualizations/3d_landscape_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_file))

            logger.info(f"3D activation landscape saved to {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"Error creating 3D activation landscape: {e}")
            return ""

    def create_network_graph(self,
                           attention_weights: np.ndarray,
                           token_labels: List[str],
                           title: str = "Attention Network Graph") -> str:
        """Create interactive network graph of attention patterns."""
        try:
            # Create network graph
            G = nx.Graph()

            # Add nodes
            for i, label in enumerate(token_labels):
                G.add_node(i, label=label)

            # Add edges based on attention weights
            threshold = np.percentile(attention_weights, 90)  # Top 10% connections
            for i in range(len(token_labels)):
                for j in range(i+1, len(token_labels)):
                    if i < attention_weights.shape[0] and j < attention_weights.shape[1]:
                        weight = attention_weights[i, j]
                        if weight > threshold:
                            G.add_edge(i, j, weight=weight)

            # Get node positions using spring layout
            pos = nx.spring_layout(G, k=1, iterations=50)

            # Extract coordinates
            node_x, node_y, node_text, node_size = [], [], [], []

            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(token_labels[node])
                # Node size based on degree centrality
                degree = G.degree(node)
                node_size.append(10 + degree * 5)

            # Create edge traces
            edge_x, edge_y = [], []

            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            # Create figure
            fig = go.Figure()

            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='rgba(125, 125, 125, 0.5)'),
                hoverinfo='none',
                mode='lines',
                name='Connections'
            ))

            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                hovertext=node_text,
                marker=dict(
                    size=node_size,
                    color='rgba(50, 150, 200, 0.8)',
                    line=dict(width=2, color='white')
                ),
                name='Tokens'
            ))

            # Update layout
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor=self.themes[self.current_theme]['paper_bgcolor'],
                plot_bgcolor=self.themes[self.current_theme]['plot_bgcolor'],
                height=600
            )

            # Save to file
            output_file = Path(f"data/outputs/visualizations/network_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_file))

            logger.info(f"Network graph saved to {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"Error creating network graph: {e}")
            return ""

    def create_comparative_dashboard(self,
                                   model_results: Dict[str, Dict[str, Any]],
                                   title: str = "Model Comparison Dashboard") -> str:
        """Create comprehensive comparison dashboard for multiple models."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Model Performance', 'Activation Distributions',
                              'Layer-wise Analysis', 'Attention Patterns'),
                specs=[[{"secondary_y": True}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "heatmap"}]]
            )

            models = list(model_results.keys())
            colors = px.colors.qualitative.Set1[:len(models)]

            # Plot 1: Model Performance Metrics
            metrics = ['accuracy', 'processing_time', 'memory_usage']
            for i, model in enumerate(models):
                if 'metrics' in model_results[model]:
                    model_metrics = model_results[model]['metrics']
                    fig.add_trace(
                        go.Scatter(
                            x=metrics,
                            y=[model_metrics.get(m, 0) for m in metrics],
                            mode='lines+markers',
                            name=f"{model} Performance",
                            line=dict(color=colors[i])
                        ),
                        row=1, col=1
                    )

            # Update layout
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                height=800,
                paper_bgcolor=self.themes[self.current_theme]['paper_bgcolor'],
                plot_bgcolor=self.themes[self.current_theme]['plot_bgcolor'],
                font_color=self.themes[self.current_theme]['font_color'],
                showlegend=True
            )

            # Save to file
            output_file = Path(f"data/outputs/visualizations/comparative_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_file))

            logger.info(f"Comparative dashboard saved to {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"Error creating comparative dashboard: {e}")
            return ""

    def set_theme(self, theme_name: str):
        """Set visualization theme."""
        if theme_name in self.themes:
            self.current_theme = theme_name
            logger.info(f"Theme set to: {theme_name}")
        else:
            logger.warning(f"Unknown theme: {theme_name}")
