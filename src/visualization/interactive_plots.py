"""
Interactive Visualization Module for NeuronMap
=============================================

This module provides Plotly-based interactive visualization capabilities
for neural network activation analysis, including 3D plots, interactive
heatmaps, and dynamic filtering options.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import json
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

logger = logging.getLogger(__name__)


@dataclass
class PlotConfig:
    """Configuration for plot appearance and behavior."""
    width: int = 800
    height: int = 600
    theme: str = "plotly_white"  # "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"
    color_palette: str = "viridis"
    show_legend: bool = True
    show_toolbar: bool = True
    auto_open: bool = False


class InteractivePlots:
    """
    Interactive visualization system using Plotly for neural activation analysis.

    Provides capabilities for:
    - Interactive scatter plots with dimension reduction
    - 3D activation space visualization
    - Dynamic heatmaps with filtering
    - Layer-by-layer activation analysis
    - Comparative visualizations
    """

    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize the interactive plotting system.

        Args:
            config: Plot configuration settings
        """
        self.config = config or PlotConfig()
        self.figures: Dict[str, go.Figure] = {}

    def create_activation_scatter(self,
                                activations: np.ndarray,
                                labels: Optional[List[str]] = None,
                                reduction_method: str = "pca",
                                dimensions: int = 2,
                                title: str = "Neural Activation Scatter Plot") -> go.Figure:
        """
        Create an interactive scatter plot of neural activations.

        Args:
            activations: Activation matrix (samples x features)
            labels: Optional labels for each sample
            reduction_method: Dimensionality reduction method ("pca", "tsne", "umap")
            dimensions: Number of dimensions (2 or 3)
            title: Plot title

        Returns:
            Plotly figure object
        """
        # Perform dimensionality reduction
        if reduction_method == "pca":
            reducer = PCA(n_components=dimensions, random_state=42)
            reduced_data = reducer.fit_transform(activations)
            explained_var = reducer.explained_variance_ratio_
            axis_labels = [f"PC{i+1} ({var:.1%})" for i, var in enumerate(explained_var)]
        elif reduction_method == "tsne":
            reducer = TSNE(n_components=dimensions, random_state=42, perplexity=min(30, len(activations)-1))
            reduced_data = reducer.fit_transform(activations)
            axis_labels = [f"t-SNE {i+1}" for i in range(dimensions)]
        elif reduction_method == "umap":
            reducer = umap.UMAP(n_components=dimensions, random_state=42)
            reduced_data = reducer.fit_transform(activations)
            axis_labels = [f"UMAP {i+1}" for i in range(dimensions)]
        else:
            raise ValueError(f"Unknown reduction method: {reduction_method}")

        # Create labels if not provided
        if labels is None:
            labels = [f"Sample {i}" for i in range(len(activations))]

        # Create scatter plot
        if dimensions == 2:
            fig = px.scatter(
                x=reduced_data[:, 0],
                y=reduced_data[:, 1],
                hover_name=labels,
                title=f"{title} ({reduction_method.upper()})",
                labels={'x': axis_labels[0], 'y': axis_labels[1]},
                template=self.config.theme
            )

            fig.update_traces(
                marker=dict(size=8, opacity=0.7),
                hovertemplate='<b>%{hovertext}</b><br>' +
                            f'{axis_labels[0]}: %{{x:.3f}}<br>' +
                            f'{axis_labels[1]}: %{{y:.3f}}<extra></extra>'
            )

        elif dimensions == 3:
            fig = px.scatter_3d(
                x=reduced_data[:, 0],
                y=reduced_data[:, 1],
                z=reduced_data[:, 2],
                hover_name=labels,
                title=f"{title} ({reduction_method.upper()})",
                labels={'x': axis_labels[0], 'y': axis_labels[1], 'z': axis_labels[2]},
                template=self.config.theme
            )

            fig.update_traces(
                marker=dict(size=5, opacity=0.8),
                hovertemplate='<b>%{hovertext}</b><br>' +
                            f'{axis_labels[0]}: %{{x:.3f}}<br>' +
                            f'{axis_labels[1]}: %{{y:.3f}}<br>' +
                            f'{axis_labels[2]}: %{{z:.3f}}<extra></extra>'
            )

        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
            showlegend=self.config.show_legend
        )

        return fig

    def create_activation_heatmap(self,
                                activations: np.ndarray,
                                sample_names: Optional[List[str]] = None,
                                neuron_names: Optional[List[str]] = None,
                                title: str = "Neural Activation Heatmap") -> go.Figure:
        """
        Create an interactive heatmap of neural activations.

        Args:
            activations: Activation matrix (samples x neurons)
            sample_names: Names for each sample
            neuron_names: Names for each neuron
            title: Plot title

        Returns:
            Plotly figure object
        """
        if sample_names is None:
            sample_names = [f"Sample {i}" for i in range(activations.shape[0])]
        if neuron_names is None:
            neuron_names = [f"Neuron {i}" for i in range(activations.shape[1])]

        fig = go.Figure(data=go.Heatmap(
            z=activations,
            x=neuron_names,
            y=sample_names,
            colorscale=self.config.color_palette,
            hoverongaps=False,
            hovertemplate='Sample: %{y}<br>Neuron: %{x}<br>Activation: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Neurons",
            yaxis_title="Samples",
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme
        )

        # Add dropdown for colorscale selection
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(label="Viridis", method="restyle", args=["colorscale", "Viridis"]),
                        dict(label="Plasma", method="restyle", args=["colorscale", "Plasma"]),
                        dict(label="Inferno", method="restyle", args=["colorscale", "Inferno"]),
                        dict(label="RdYlBu", method="restyle", args=["colorscale", "RdYlBu"]),
                        dict(label="Blues", method="restyle", args=["colorscale", "Blues"]),
                    ]),
                    direction="down",
                    showactive=True,
                    x=1.0,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                ),
            ]
        )

        return fig

    def create_layer_comparison(self,
                              layer_activations: Dict[str, np.ndarray],
                              sample_index: int = 0,
                              title: str = "Layer-by-Layer Activation Comparison") -> go.Figure:
        """
        Create a comparative visualization of activations across layers.

        Args:
            layer_activations: Dictionary mapping layer names to activation arrays
            sample_index: Index of the sample to visualize
            title: Plot title

        Returns:
            Plotly figure object
        """
        # Create subplots
        layer_names = list(layer_activations.keys())
        n_layers = len(layer_names)

        # Determine subplot layout
        cols = min(3, n_layers)
        rows = (n_layers + cols - 1) // cols

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=layer_names,
            specs=[[{"type": "scatter"}] * cols for _ in range(rows)]
        )

        for i, (layer_name, activations) in enumerate(layer_activations.items()):
            row = i // cols + 1
            col = i % cols + 1

            # Get activations for the specified sample
            if len(activations.shape) > 1:
                sample_activations = activations[sample_index]
            else:
                sample_activations = activations

            # Create bar plot for this layer
            fig.add_trace(
                go.Bar(
                    x=list(range(len(sample_activations))),
                    y=sample_activations,
                    name=layer_name,
                    showlegend=False,
                    hovertemplate=f'Layer: {layer_name}<br>Neuron: %{{x}}<br>Activation: %{{y:.3f}}<extra></extra>'
                ),
                row=row,
                col=col
            )

        fig.update_layout(
            title_text=title,
            width=self.config.width * 1.5,
            height=self.config.height * rows / 2,
            template=self.config.theme
        )

        # Add dropdown to select different samples
        if layer_activations and len(list(layer_activations.values())[0].shape) > 1:
            n_samples = list(layer_activations.values())[0].shape[0]

            buttons = []
            for sample_idx in range(min(n_samples, 10)):  # Limit to first 10 samples
                button_data = []
                for i, (layer_name, activations) in enumerate(layer_activations.items()):
                    if len(activations.shape) > 1:
                        sample_acts = activations[sample_idx]
                    else:
                        sample_acts = activations
                    button_data.append(sample_acts)

                buttons.append(
                    dict(
                        label=f"Sample {sample_idx}",
                        method="restyle",
                        args=[{"y": button_data}]
                    )
                )

            fig.update_layout(
                updatemenus=[
                    dict(
                        buttons=buttons,
                        direction="down",
                        showactive=True,
                        x=1.0,
                        xanchor="left",
                        y=1.02,
                        yanchor="top"
                    )
                ]
            )

        return fig

    def create_activation_timeline(self,
                                 activations: np.ndarray,
                                 layer_names: List[str],
                                 sample_names: Optional[List[str]] = None,
                                 title: str = "Activation Timeline") -> go.Figure:
        """
        Create a timeline visualization showing activation evolution across layers.

        Args:
            activations: 3D array (samples x layers x features)
            layer_names: Names of the layers
            sample_names: Names of the samples
            title: Plot title

        Returns:
            Plotly figure object
        """
        if sample_names is None:
            sample_names = [f"Sample {i}" for i in range(activations.shape[0])]

        # Calculate average activation per layer per sample
        avg_activations = np.mean(activations, axis=2)  # Average over features

        fig = go.Figure()

        # Add a trace for each sample
        for i, sample_name in enumerate(sample_names):
            fig.add_trace(
                go.Scatter(
                    x=layer_names,
                    y=avg_activations[i],
                    mode='lines+markers',
                    name=sample_name,
                    hovertemplate=f'Sample: {sample_name}<br>Layer: %{{x}}<br>Avg Activation: %{{y:.3f}}<extra></extra>'
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Layers",
            yaxis_title="Average Activation",
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme,
            hovermode='x unified'
        )

        return fig

    def create_neuron_importance_plot(self,
                                    importance_scores: np.ndarray,
                                    neuron_names: Optional[List[str]] = None,
                                    top_k: int = 20,
                                    title: str = "Neuron Importance Ranking") -> go.Figure:
        """
        Create a bar plot showing the most important neurons.

        Args:
            importance_scores: Array of importance scores for each neuron
            neuron_names: Names of the neurons
            top_k: Number of top neurons to display
            title: Plot title

        Returns:
            Plotly figure object
        """
        if neuron_names is None:
            neuron_names = [f"Neuron {i}" for i in range(len(importance_scores))]

        # Get top k neurons
        top_indices = np.argsort(importance_scores)[-top_k:][::-1]
        top_scores = importance_scores[top_indices]
        top_names = [neuron_names[i] for i in top_indices]

        fig = go.Figure(data=[
            go.Bar(
                x=top_names,
                y=top_scores,
                marker_color=px.colors.sequential.Viridis[::len(px.colors.sequential.Viridis)//len(top_scores)],
                hovertemplate='Neuron: %{x}<br>Importance: %{y:.3f}<extra></extra>'
            )
        ])

        fig.update_layout(
            title=title,
            xaxis_title="Neurons",
            yaxis_title="Importance Score",
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme,
            xaxis_tickangle=-45
        )

        return fig

    def create_similarity_matrix(self,
                               similarity_matrix: np.ndarray,
                               labels: List[str],
                               title: str = "Activation Similarity Matrix") -> go.Figure:
        """
        Create an interactive similarity matrix heatmap.

        Args:
            similarity_matrix: Square matrix of similarity scores
            labels: Labels for rows/columns
            title: Plot title

        Returns:
            Plotly figure object
        """
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=labels,
            y=labels,
            colorscale='RdBu',
            zmid=0,
            hoverongaps=False,
            hovertemplate='Row: %{y}<br>Col: %{x}<br>Similarity: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Samples",
            yaxis_title="Samples",
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme
        )

        return fig

    def save_figure(self,
                   figure: go.Figure,
                   filename: str,
                   format: str = "html",
                   output_dir: str = "data/outputs") -> str:
        """
        Save a Plotly figure to file.

        Args:
            figure: Plotly figure to save
            filename: Output filename (without extension)
            format: Output format ("html", "png", "pdf", "svg")
            output_dir: Output directory

        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / f"{filename}.{format}"

        if format == "html":
            figure.write_html(str(file_path))
        elif format in ["png", "pdf", "svg"]:
            figure.write_image(str(file_path))
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Figure saved to {file_path}")
        return str(file_path)

    def create_dashboard(self,
                        layer_activations: Dict[str, np.ndarray],
                        sample_names: List[str],
                        output_path: str = "data/outputs/activation_dashboard.html") -> str:
        """
        Create a comprehensive dashboard with multiple visualizations.

        Args:
            layer_activations: Dictionary mapping layer names to activation arrays
            sample_names: Names of the samples
            output_path: Path to save the dashboard

        Returns:
            Path to saved dashboard
        """
        from plotly.subplots import make_subplots
        import plotly.offline as pyo

        # Create multiple visualizations
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NeuronMap Interactive Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .plot-container {{ margin: 20px 0; }}
                h1, h2 {{ color: #333; }}
            </style>
        </head>
        <body>
            <h1>NeuronMap Interactive Analysis Dashboard</h1>
            <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """

        # Add each visualization
        for i, (layer_name, activations) in enumerate(layer_activations.items()):
            # Create scatter plot
            if activations.shape[1] > 2:  # Only if we have enough dimensions
                scatter_fig = self.create_activation_scatter(
                    activations,
                    labels=sample_names,
                    title=f"Activation Scatter - {layer_name}"
                )

                dashboard_html += f"""
                <div class="plot-container">
                    <h2>Layer: {layer_name} - Activation Scatter</h2>
                    <div id="scatter_{i}"></div>
                </div>
                """

                # Add JavaScript to render the plot
                dashboard_html += f"""
                <script>
                    var scatter_data_{i} = {scatter_fig.to_json()};
                    Plotly.newPlot('scatter_{i}', scatter_data_{i}.data, scatter_data_{i}.layout);
                </script>
                """

            # Create heatmap
            heatmap_fig = self.create_activation_heatmap(
                activations,
                sample_names=sample_names,
                title=f"Activation Heatmap - {layer_name}"
            )

            dashboard_html += f"""
            <div class="plot-container">
                <h2>Layer: {layer_name} - Activation Heatmap</h2>
                <div id="heatmap_{i}"></div>
            </div>
            """

            dashboard_html += f"""
            <script>
                var heatmap_data_{i} = {heatmap_fig.to_json()};
                Plotly.newPlot('heatmap_{i}', heatmap_data_{i}.data, heatmap_data_{i}.layout);
            </script>
            """

        dashboard_html += """
        </body>
        </html>
        """

        # Save dashboard
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)

        logger.info(f"Interactive dashboard saved to {output_path}")
        return str(output_path)


def create_interactive_analysis(activations: np.ndarray,
                              labels: Optional[List[str]] = None,
                              output_dir: str = "data/outputs",
                              config: Optional[PlotConfig] = None) -> Dict[str, str]:
    """
    Convenience function to create a complete set of interactive visualizations.

    Args:
        activations: Activation matrix (samples x features)
        labels: Optional labels for samples
        output_dir: Output directory for saved plots
        config: Plot configuration

    Returns:
        Dictionary mapping plot types to file paths
    """
    plotter = InteractivePlots(config)
    saved_files = {}

    # Create different visualizations
    plots = {
        "scatter_pca": plotter.create_activation_scatter(activations, labels, "pca", 2),
        "scatter_tsne": plotter.create_activation_scatter(activations, labels, "tsne", 2),
        "scatter_3d": plotter.create_activation_scatter(activations, labels, "pca", 3),
        "heatmap": plotter.create_activation_heatmap(activations, labels)
    }

    # Save all plots
    for plot_name, figure in plots.items():
        file_path = plotter.save_figure(figure, plot_name, "html", output_dir)
        saved_files[plot_name] = file_path

    logger.info(f"Created {len(saved_files)} interactive visualizations in {output_dir}")
    return saved_files
