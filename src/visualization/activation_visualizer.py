"""
Activation Visualizer Module for NeuronMap
=========================================

This module provides visualization capabilities for neural network activations.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json

logger = logging.getLogger(__name__)

# Conditional imports for plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
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
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Interactive visualizations will be disabled.")

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Dimensionality reduction and clustering will be disabled.")


class ActivationVisualizer:
    """Creates visualizations of neural network activations."""

    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.data.outputs_dir) / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set visualization backend
        self.backend = config.visualization.backend
        self._setup_plotting_style()

    def _setup_plotting_style(self):
        """Setup plotting style and parameters."""
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('default')
            plt.rcParams['figure.figsize'] = (
                self.config.visualization.figure_width,
                self.config.visualization.figure_height
            )
            plt.rcParams['figure.dpi'] = self.config.visualization.dpi
            plt.rcParams['font.size'] = 10
            plt.rcParams['axes.titlesize'] = 12
            plt.rcParams['axes.labelsize'] = 10
            plt.rcParams['legend.fontsize'] = 9

        if SEABORN_AVAILABLE:
            sns.set_style("whitegrid")
            sns.set_palette("husl")

    def load_results(self) -> pd.DataFrame:
        """Load analysis results from file."""
        results_file = Path(self.config.data.output_file)

        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")

        try:
            if results_file.suffix.lower() == '.csv':
                df = pd.read_csv(results_file)
            else:
                # Assume JSON
                with open(results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)

            logger.info(f"Loaded {len(df)} results from {results_file}")
            return df

        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            raise

    def extract_activation_data(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract activation arrays from the DataFrame."""
        activation_data = {}

        # Find activation columns
        activation_columns = [col for col in df.columns if col.endswith('_activation')]

        for col in activation_columns:
            layer_name = col.replace('_activation', '')

            # Extract activations
            activations = []
            for idx, row in df.iterrows():
                if pd.notna(row[col]) and row.get('success', True):
                    try:
                        if isinstance(row[col], str):
                            # Parse JSON string
                            activation = json.loads(row[col])
                        else:
                            activation = row[col]

                        if isinstance(activation, list):
                            activations.append(np.array(activation))
                        elif isinstance(activation, np.ndarray):
                            activations.append(activation)
                    except Exception as e:
                        logger.warning(f"Failed to parse activation for row {idx}: {e}")

            if activations:
                activation_data[layer_name] = np.array(activations)
                logger.info(f"Extracted {len(activations)} activations for layer {layer_name}")

        return activation_data

    def plot_activation_statistics(self, activation_data: Dict[str, np.ndarray]) -> None:
        """Plot basic statistics of activations."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping activation statistics plot.")
            return

        n_layers = len(activation_data)
        if n_layers == 0:
            logger.warning("No activation data available for plotting.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Activation Statistics by Layer', fontsize=16)

        layer_names = list(activation_data.keys())

        # 1. Mean activation values
        means = []
        stds = []
        for layer_name in layer_names:
            activations = activation_data[layer_name]
            means.append(activations.mean())
            stds.append(activations.std())

        axes[0, 0].bar(range(len(layer_names)), means, yerr=stds, capsize=5)
        axes[0, 0].set_title('Mean Activation Values')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Mean Activation')
        axes[0, 0].set_xticks(range(len(layer_names)))
        axes[0, 0].set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45)

        # 2. Activation distributions (first layer as example)
        if layer_names:
            first_layer = layer_names[0]
            activations = activation_data[first_layer]
            axes[0, 1].hist(activations.flatten(), bins=50, alpha=0.7, density=True)
            axes[0, 1].set_title(f'Activation Distribution - {first_layer.split(".")[-1]}')
            axes[0, 1].set_xlabel('Activation Value')
            axes[0, 1].set_ylabel('Density')

        # 3. Activation variance across neurons
        variances = []
        for layer_name in layer_names:
            activations = activation_data[layer_name]
            if activations.ndim > 1:
                var_per_neuron = activations.var(axis=0)
                variances.append(var_per_neuron.mean())
            else:
                variances.append(activations.var())

        axes[1, 0].bar(range(len(layer_names)), variances)
        axes[1, 0].set_title('Mean Variance per Neuron')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Mean Variance')
        axes[1, 0].set_xticks(range(len(layer_names)))
        axes[1, 0].set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45)

        # 4. Sparsity (percentage of near-zero activations)
        sparsities = []
        for layer_name in layer_names:
            activations = activation_data[layer_name]
            threshold = 0.01 * activations.std()
            sparsity = (np.abs(activations) < threshold).mean()
            sparsities.append(sparsity * 100)

        axes[1, 1].bar(range(len(layer_names)), sparsities)
        axes[1, 1].set_title('Activation Sparsity')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Sparsity (%)')
        axes[1, 1].set_xticks(range(len(layer_names)))
        axes[1, 1].set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45)

        plt.tight_layout()
        output_path = self.output_dir / f"activation_statistics.{self.config.visualization.save_format}"
        plt.savefig(output_path, dpi=self.config.visualization.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Activation statistics plot saved to {output_path}")

    def plot_activation_heatmap(self, activation_data: Dict[str, np.ndarray]) -> None:
        """Create heatmap of activations."""
        if not SEABORN_AVAILABLE:
            logger.warning("Seaborn not available. Skipping activation heatmap.")
            return

        for layer_name, activations in activation_data.items():
            if activations.ndim < 2:
                continue

            # Limit to first 100 neurons and 50 samples for visualization
            plot_activations = activations[:50, :100]

            plt.figure(figsize=(12, 8))
            sns.heatmap(
                plot_activations,
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Activation Value'},
                xticklabels=False,
                yticklabels=False
            )

            plt.title(f'Activation Heatmap - {layer_name}')
            plt.xlabel('Neuron Index')
            plt.ylabel('Sample Index')

            output_path = self.output_dir / f"heatmap_{layer_name.replace('.', '_')}.{self.config.visualization.save_format}"
            plt.savefig(output_path, dpi=self.config.visualization.dpi, bbox_inches='tight')
            plt.close()

            logger.info(f"Heatmap for {layer_name} saved to {output_path}")

    def plot_dimensionality_reduction(self, activation_data: Dict[str, np.ndarray]) -> None:
        """Create dimensionality reduction plots."""
        if not SKLEARN_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            logger.warning("Scikit-learn or Matplotlib not available. Skipping dimensionality reduction plots.")
            return

        for layer_name, activations in activation_data.items():
            if activations.ndim < 2 or activations.shape[1] < 2:
                continue

            # Apply PCA
            try:
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(activations)

                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, s=50)
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                plt.title(f'PCA - {layer_name}')
                plt.colorbar(scatter, label='Sample Index')

                output_path = self.output_dir / f"pca_{layer_name.replace('.', '_')}.{self.config.visualization.save_format}"
                plt.savefig(output_path, dpi=self.config.visualization.dpi, bbox_inches='tight')
                plt.close()

                logger.info(f"PCA plot for {layer_name} saved to {output_path}")

            except Exception as e:
                logger.warning(f"PCA failed for {layer_name}: {e}")

            # Apply t-SNE (only for smaller datasets)
            if activations.shape[0] <= 1000:
                try:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, activations.shape[0]-1))
                    tsne_result = tsne.fit_transform(activations)

                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.6, s=50)
                    plt.xlabel('t-SNE 1')
                    plt.ylabel('t-SNE 2')
                    plt.title(f't-SNE - {layer_name}')
                    plt.colorbar(scatter, label='Sample Index')

                    output_path = self.output_dir / f"tsne_{layer_name.replace('.', '_')}.{self.config.visualization.save_format}"
                    plt.savefig(output_path, dpi=self.config.visualization.dpi, bbox_inches='tight')
                    plt.close()

                    logger.info(f"t-SNE plot for {layer_name} saved to {output_path}")

                except Exception as e:
                    logger.warning(f"t-SNE failed for {layer_name}: {e}")

    def create_interactive_dashboard(self, df: pd.DataFrame, activation_data: Dict[str, np.ndarray]) -> None:
        """Create an interactive dashboard using Plotly."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Skipping interactive dashboard.")
            return

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Activation Statistics',
                'Success Rate',
                'Layer Comparison',
                'Sample Distribution'
            ],
            specs=[[{"secondary_y": True}, {"type": "pie"}],
                   [{"colspan": 2}, None]],
            vertical_spacing=0.15
        )

        # 1. Activation statistics
        layer_names = list(activation_data.keys())
        if layer_names:
            means = [activation_data[layer].mean() for layer in layer_names]
            stds = [activation_data[layer].std() for layer in layer_names]

            fig.add_trace(
                go.Bar(x=layer_names, y=means, name='Mean', yaxis='y'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=layer_names, y=stds, mode='lines+markers', name='Std Dev', yaxis='y2'),
                row=1, col=1, secondary_y=True
            )

        # 2. Success rate pie chart
        success_count = df['success'].sum() if 'success' in df.columns else len(df)
        total_count = len(df)
        fig.add_trace(
            go.Pie(
                labels=['Success', 'Failed'],
                values=[success_count, total_count - success_count],
                name="Success Rate"
            ),
            row=1, col=2
        )

        # 3. Layer comparison (if multiple layers)
        if len(layer_names) > 1:
            for i, layer_name in enumerate(layer_names[:3]):  # Limit to 3 layers
                activations = activation_data[layer_name]
                if activations.ndim > 1:
                    # Show first few dimensions
                    fig.add_trace(
                        go.Scatter(
                            y=activations[:, 0] if activations.shape[1] > 0 else activations.flatten()[:50],
                            mode='lines',
                            name=f'{layer_name.split(".")[-1]} (dim 0)',
                            line=dict(width=2)
                        ),
                        row=2, col=1
                    )

        # Update layout
        fig.update_layout(
            title_text="NeuronMap Activation Analysis Dashboard",
            height=800,
            showlegend=True
        )

        # Save interactive plot
        output_path = self.output_dir / "interactive_dashboard.html"
        fig.write_html(str(output_path))
        logger.info(f"Interactive dashboard saved to {output_path}")

    def generate_summary_report(self, df: pd.DataFrame, activation_data: Dict[str, np.ndarray]) -> None:
        """Generate a text summary report."""
        report_path = self.output_dir / "analysis_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("NeuronMap Analysis Report\\n")
            f.write("=" * 50 + "\\n\\n")

            # Basic statistics
            f.write(f"Total samples: {len(df)}\\n")
            if 'success' in df.columns:
                success_rate = df['success'].mean()
                f.write(f"Success rate: {success_rate:.2%}\\n")

            f.write(f"Number of layers analyzed: {len(activation_data)}\\n\\n")

            # Layer details
            f.write("Layer Analysis:\\n")
            f.write("-" * 20 + "\\n")

            for layer_name, activations in activation_data.items():
                f.write(f"\\nLayer: {layer_name}\\n")
                f.write(f"  Shape: {activations.shape}\\n")
                f.write(f"  Mean: {activations.mean():.4f}\\n")
                f.write(f"  Std: {activations.std():.4f}\\n")
                f.write(f"  Min: {activations.min():.4f}\\n")
                f.write(f"  Max: {activations.max():.4f}\\n")

                if activations.ndim > 1:
                    sparsity = (np.abs(activations) < 0.01 * activations.std()).mean()
                    f.write(f"  Sparsity: {sparsity:.2%}\\n")

            # Configuration summary
            f.write(f"\\n\\nConfiguration Used:\\n")
            f.write("-" * 20 + "\\n")
            f.write(f"Model: {self.config.model.name}\\n")
            f.write(f"Device: {self.config.model.device}\\n")
            f.write(f"Aggregation: {self.config.analysis.aggregation_method}\\n")
            f.write(f"Target layers: {', '.join(self.config.model.target_layers)}\\n")

        logger.info(f"Summary report saved to {report_path}")

    def generate_all_visualizations(self) -> None:
        """Generate all available visualizations."""
        logger.info("Generating visualizations...")

        try:
            # Load data
            df = self.load_results()
            activation_data = self.extract_activation_data(df)

            if not activation_data:
                logger.warning("No activation data found. Cannot generate visualizations.")
                return

            # Generate plots
            self.plot_activation_statistics(activation_data)
            self.plot_activation_heatmap(activation_data)
            self.plot_dimensionality_reduction(activation_data)
            self.create_interactive_dashboard(df, activation_data)
            self.generate_summary_report(df, activation_data)

            logger.info(f"All visualizations saved to {self.output_dir}")

        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
            raise


def main():
    """Test the visualizer."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from utils.config_manager import get_config

    config = get_config()
    visualizer = ActivationVisualizer(config)

    # Check if results file exists
    results_file = Path(config.data.output_file)
    if results_file.exists():
        print(f"Generating visualizations for {results_file}")
        visualizer.generate_all_visualizations()
    else:
        print(f"Results file {results_file} not found. Run analysis first.")


if __name__ == "__main__":
    main()
