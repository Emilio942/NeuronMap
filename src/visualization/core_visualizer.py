"""
Core Visualization Module for NeuronMap
======================================

Production-ready visualization system with class-based architecture.
Migrated from visualizer.py with enhanced modularity and configuration support.
"""

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys

# Import our error handling and config systems
try:
    from ..utils.error_handling import NeuronMapException, ValidationError
    from ..utils.config_manager import VisualizationConfig
except ImportError:
    # Fallback for standalone usage
    class NeuronMapException(Exception):
        pass
    class ValidationError(Exception):
        pass
    VisualizationConfig = None

# --- Configuration Defaults ---
DEFAULT_INPUT_CSV = "activation_results.csv"
DEFAULT_OUTPUT_PCA = "activation_pca_scatter.png"
DEFAULT_OUTPUT_TSNE = "activation_tsne_scatter.png"
DEFAULT_OUTPUT_HEATMAP = "activation_heatmap.png"
DEFAULT_N_COMPONENTS = 2
DEFAULT_TSNE_PERPLEXITY = 30
DEFAULT_TSNE_LEARNING_RATE = 200
DEFAULT_TSNE_N_ITER = 1000
DEFAULT_HEATMAP_MAX_QUESTIONS = 50
DEFAULT_HEATMAP_MAX_NEURONS = 100

logger = logging.getLogger(__name__)


class CoreVisualizer:
    """
    Core visualization class for neural network activation analysis.

    Provides PCA, t-SNE, heatmaps and other visualization methods with
    robust error handling and configurable parameters.

    Features:
    - Automated data loading and preprocessing
    - Multiple dimensionality reduction techniques
    - Interactive and static plotting options
    - Configurable output formats
    - Command-line interface
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize the visualizer with configuration."""
        self.config = config or self._get_default_config()
        self.scaler = StandardScaler()

        # Set up matplotlib for better quality
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                # Use default style if seaborn not available
                pass

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if none provided."""
        return {
            'dpi': 300,
            'figsize': (10, 8),
            'style': 'seaborn',
            'colormap': 'viridis',
            'n_components': 2,
            'random_state': 42
        }

    # load_and_prepare_data method is defined below with better implementation

    def load_and_prepare_data(self, filepath: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Loads CSV data, parses vectors, and prepares the matrix.

        Args:
            filepath: Path to the CSV file with activation data

        Returns:
            Tuple of (DataFrame, scaled_activation_matrix)
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Data loaded from '{filepath}'. Number of rows: {len(df)}")
        except FileNotFoundError:
            logger.error(f"File '{filepath}' not found. Make sure the activation extraction script ran successfully.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading CSV: {e}")
            raise

        # Remove rows where activation vector is missing
        original_len = len(df)
        df.dropna(subset=['activation_vector'], inplace=True)
        if len(df) < original_len:
            logger.warning(f"{original_len - len(df)} rows without activation vector removed.")

        if df.empty:
            raise ValidationError("No valid activation vectors found in the file.")

        # Parse activation_vector column from string lists to actual lists
        try:
            # ast.literal_eval is safer than eval()
            df['activation_list'] = df['activation_vector'].apply(ast.literal_eval)
        except (ValueError, SyntaxError) as e:
            logger.error("Could not parse 'activation_vector' column as lists.")
            logger.error("Make sure vectors are stored in format '[num, num, ...]'.")
            logger.error(f"Error details: {e}")

            # Show first few problematic entries
            logger.error("Sample entries from 'activation_vector':")
            for i, vec_str in enumerate(df['activation_vector'].head()):
                logger.error(f"Row {i}: {str(vec_str)[:100]}...")
            raise

        # Check if all vectors have the same length
        vector_lengths = df['activation_list'].apply(len)
        if vector_lengths.nunique() > 1:
            logger.warning("Activation vectors have different lengths!")
            logger.warning(f"Found lengths: {vector_lengths.unique()}")

            # Use first length as reference and filter
            target_len = vector_lengths.iloc[0]
            logger.info(f"Filtering to vectors with length {target_len}")
            df = df[df['activation_list'].apply(len) == target_len].copy()

            if df.empty:
                raise ValidationError("No data remaining after filtering for consistent vector length.")

        # Convert list of vectors to NumPy matrix
        # Each row of the matrix is an activation vector
        activation_matrix = np.array(df['activation_list'].tolist())

        logger.info(f"Activation matrix created with shape: {activation_matrix.shape}")

        # Normalize data (standardization) - helps PCA and t-SNE perform better
        activation_matrix_scaled = self.scaler.fit_transform(activation_matrix)
        logger.info("Activation matrix normalized (StandardScaler).")

        return df, activation_matrix_scaled

    def run_pca(self, data_matrix: np.ndarray, n_components: int = DEFAULT_N_COMPONENTS) -> np.ndarray:
        """
        Performs PCA on the data matrix.

        Args:
            data_matrix: Input activation matrix
            n_components: Number of components to reduce to

        Returns:
            PCA-transformed data
        """
        logger.info(f"Running PCA with {n_components} components...")
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(data_matrix)

        # Log explained variance
        explained_variance = pca.explained_variance_ratio_
        logger.info(f"PCA explained variance: {explained_variance}")
        logger.info(f"Total explained variance: {explained_variance.sum():.3f}")

        return pca_result

    def run_tsne(self, data_matrix: np.ndarray,
                 n_components: int = DEFAULT_N_COMPONENTS,
                 perplexity: float = DEFAULT_TSNE_PERPLEXITY,
                 learning_rate: float = DEFAULT_TSNE_LEARNING_RATE,
                 n_iter: int = DEFAULT_TSNE_N_ITER) -> np.ndarray:
        """
        Performs t-SNE on the data matrix.

        Args:
            data_matrix: Input activation matrix
            n_components: Number of components to reduce to
            perplexity: t-SNE perplexity parameter
            learning_rate: t-SNE learning rate
            n_iter: Number of iterations

        Returns:
            t-SNE transformed data
        """
        logger.info(f"Running t-SNE with {n_components} components, perplexity={perplexity}...")

        # t-SNE can be slow for large datasets
        if data_matrix.shape[0] > 1000:
            logger.warning(f"Large dataset ({data_matrix.shape[0]} samples). t-SNE may be slow.")

        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=42  # For reproducibility
        )

        tsne_result = tsne.fit_transform(data_matrix)
        logger.info("t-SNE completed successfully.")

        return tsne_result

    def create_scatter_plot(self, data_2d: np.ndarray, title: str, output_file: str) -> None:
        """Create a scatter plot from 2D data."""
        try:
            logger.info(f"Creating scatter plot: {title}")

            fig, ax = plt.subplots(figsize=self.config.get('figsize', (10, 8)))
            scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1],
                               c=range(len(data_2d)), cmap='viridis', alpha=0.7)

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Component 1', fontsize=12)
            ax.set_ylabel('Component 2', fontsize=12)
            plt.colorbar(scatter, ax=ax, label='Data Point Index')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_file, dpi=self.config.get('dpi', 300), bbox_inches='tight')
            plt.close()

            logger.info(f"Scatter plot saved to {output_file}")

        except Exception as e:
            raise NeuronMapException(f"Failed to create scatter plot: {e}",
                                   error_code="PLOT_ERROR")

    def plot_scatter(self, data_2d: np.ndarray, title: str, filename: str,
                    df_questions: Optional[pd.DataFrame] = None) -> None:
        """
        Creates a 2D scatter plot and saves it.

        Args:
            data_2d: 2D data points to plot
            title: Plot title
            filename: Output filename
            df_questions: Optional DataFrame with questions for hover text
        """
        plt.figure(figsize=(12, 8))

        # Create scatter plot
        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1],
                            alpha=0.6, s=30, c=range(len(data_2d)),
                            cmap='viridis')

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Component 1', fontsize=12)
        plt.ylabel('Component 2', fontsize=12)
        plt.colorbar(scatter, label='Question Index')
        plt.grid(True, alpha=0.3)

        # Add some sample annotations if questions available
        if df_questions is not None and len(df_questions) <= 20:
            for i, (x, y) in enumerate(data_2d[:min(5, len(data_2d))]):
                question_preview = df_questions.iloc[i]['question'][:30] + "..."
                plt.annotate(question_preview, (x, y),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Scatter plot saved to '{filename}'")

    def plot_heatmap(self, activation_matrix: np.ndarray, df_questions: pd.DataFrame,
                    filename: str, max_questions: int = DEFAULT_HEATMAP_MAX_QUESTIONS,
                    max_neurons: int = DEFAULT_HEATMAP_MAX_NEURONS) -> None:
        """
        Creates a heatmap of activations (subset) and saves it.

        Args:
            activation_matrix: Full activation matrix
            df_questions: DataFrame with questions
            filename: Output filename
            max_questions: Maximum number of questions to show
            max_neurons: Maximum number of neurons/dimensions to show
        """
        # Take subset for readability
        matrix_subset = activation_matrix[:max_questions, :max_neurons]

        plt.figure(figsize=(15, 10))

        # Create heatmap
        sns.heatmap(matrix_subset,
                   cmap='RdYlBu_r',
                   center=0,
                   annot=False,
                   fmt='.2f',
                   cbar_kws={'label': 'Activation Strength'})

        plt.title(f'Activation Heatmap (First {max_questions} Questions, First {max_neurons} Neurons)',
                 fontsize=16, fontweight='bold')
        plt.xlabel('Neuron/Dimension Index', fontsize=12)
        plt.ylabel('Question Index', fontsize=12)

        # Add question labels on y-axis if not too many
        if max_questions <= 20:
            question_labels = [f"Q{i}: {df_questions.iloc[i]['question'][:20]}..."
                             for i in range(min(max_questions, len(df_questions)))]
            plt.yticks(range(len(question_labels)), question_labels, fontsize=8)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Heatmap saved to '{filename}'")

    def analyze_activations(self, csv_file: str, output_dir: str = "visualizations") -> Dict[str, Any]:
        """Complete activation analysis pipeline."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            df, activation_matrix = self.load_and_prepare_data(csv_file)

            # Normalize data
            activation_normalized = self.scaler.fit_transform(activation_matrix)

            # PCA
            pca = PCA(n_components=2, random_state=42)
            pca_result = pca.fit_transform(activation_normalized)
            pca_file = output_path / "activation_pca_scatter.png"
            self.create_scatter_plot(pca_result, "PCA of Neural Activations", str(pca_file))

            # t-SNE
            tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
            tsne_result = tsne.fit_transform(activation_normalized)
            tsne_file = output_path / "activation_tsne_scatter.png"
            self.create_scatter_plot(tsne_result, "t-SNE of Neural Activations", str(tsne_file))

            results = {
                'data_shape': activation_matrix.shape,
                'num_samples': len(activation_matrix),
                'output_files': {
                    'pca': str(pca_file),
                    'tsne': str(tsne_file)
                }
            }

            logger.info("✅ Complete activation analysis finished successfully!")
            return results

        except Exception as e:
            raise NeuronMapException(f"Analysis pipeline failed: {e}",
                                   error_code="ANALYSIS_ERROR")

    def run_full_visualization(self, input_file: str,
                             output_pca: str = DEFAULT_OUTPUT_PCA,
                             output_tsne: str = DEFAULT_OUTPUT_TSNE,
                             output_heatmap: str = DEFAULT_OUTPUT_HEATMAP,
                             tsne_perplexity: float = DEFAULT_TSNE_PERPLEXITY,
                             tsne_learning_rate: float = DEFAULT_TSNE_LEARNING_RATE,
                             tsne_iterations: int = DEFAULT_TSNE_N_ITER,
                             heatmap_max_questions: int = DEFAULT_HEATMAP_MAX_QUESTIONS,
                             heatmap_max_neurons: int = DEFAULT_HEATMAP_MAX_NEURONS) -> bool:
        """
        Runs the complete visualization pipeline.

        Args:
            input_file: Path to input CSV file
            output_pca: Path for PCA plot
            output_tsne: Path for t-SNE plot
            output_heatmap: Path for heatmap
            tsne_perplexity: t-SNE perplexity parameter
            tsne_learning_rate: t-SNE learning rate
            tsne_iterations: Number of t-SNE iterations
            heatmap_max_questions: Max questions to show in heatmap
            heatmap_max_neurons: Max neurons to show in heatmap

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load and prepare data
            df_questions, activation_matrix = self.load_and_prepare_data(input_file)

            # --- Visualization 1: PCA ---
            pca_result_2d = self.run_pca(activation_matrix, n_components=DEFAULT_N_COMPONENTS)
            self.plot_scatter(pca_result_2d,
                            f"PCA of Neural Activations ({DEFAULT_N_COMPONENTS} Components)",
                            output_pca,
                            df_questions)

            # --- Visualization 2: t-SNE ---
            tsne_result_2d = self.run_tsne(activation_matrix,
                                         n_components=DEFAULT_N_COMPONENTS,
                                         perplexity=tsne_perplexity,
                                         learning_rate=tsne_learning_rate,
                                         n_iter=tsne_iterations)
            self.plot_scatter(tsne_result_2d,
                            f"t-SNE of Neural Activations ({DEFAULT_N_COMPONENTS} Components, Perplexity={tsne_perplexity})",
                            output_tsne,
                            df_questions)

            # --- Visualization 3: Heatmap ---
            self.plot_heatmap(activation_matrix,
                            df_questions,
                            output_heatmap,
                            heatmap_max_questions,
                            heatmap_max_neurons)

            logger.info("Visualization pipeline completed successfully.")
            return True

        except Exception as e:
            logger.error(f"Visualization pipeline failed: {e}")
            return False

def main():
    """Command-line interface for the core visualizer."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations for neural network activation analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--input-file", type=str, default=DEFAULT_INPUT_CSV,
                       help="Path to input CSV file with activation data")
    parser.add_argument("--output-pca", type=str, default=DEFAULT_OUTPUT_PCA,
                       help="Path for PCA scatter plot output")
    parser.add_argument("--output-tsne", type=str, default=DEFAULT_OUTPUT_TSNE,
                       help="Path for t-SNE scatter plot output")
    parser.add_argument("--output-heatmap", type=str, default=DEFAULT_OUTPUT_HEATMAP,
                       help="Path for activation heatmap output")
    parser.add_argument("--tsne-perplexity", type=float, default=DEFAULT_TSNE_PERPLEXITY,
                       help="t-SNE perplexity parameter")
    parser.add_argument("--tsne-learning-rate", type=float, default=DEFAULT_TSNE_LEARNING_RATE,
                       help="t-SNE learning rate")
    parser.add_argument("--tsne-iterations", type=int, default=DEFAULT_TSNE_N_ITER,
                       help="Number of t-SNE iterations")
    parser.add_argument("--heatmap-max-questions", type=int, default=DEFAULT_HEATMAP_MAX_QUESTIONS,
                       help="Maximum questions to show in heatmap")
    parser.add_argument("--heatmap-max-neurons", type=int, default=DEFAULT_HEATMAP_MAX_NEURONS,
                       help="Maximum neurons to show in heatmap")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create visualizer
    visualizer = CoreVisualizer()

    # Update visualizer defaults with command-line arguments (avoiding global variables)
    # Pass parameters directly to the visualization method

    # Run visualization pipeline
    success = visualizer.run_full_visualization(
        args.input_file,
        args.output_pca,
        args.output_tsne,
        args.output_heatmap,
        args.tsne_perplexity,
        args.tsne_learning_rate,
        args.tsne_iterations,
        args.heatmap_max_questions,
        args.heatmap_max_neurons
    )

    sys.exit(0 if success else 1)


# Backward-compatible script entry point
if __name__ == "__main__":
    main()
