"""Visualization tools for neural network activations in NeuronMap."""

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from ..utils.config import get_config


logger = logging.getLogger(__name__)


class ActivationVisualizer:
    """Visualize neural network activation patterns."""

    def __init__(self, config_name: str = "default", output_dir: str = None):
        """Initialize visualizer with configuration.

        Args:
            config_name: Name of experiment configuration to use.
            output_dir: Output directory (for test compatibility)
        """
        try:
            self.config = get_config()
            self.experiment_config = self.config.get_experiment_config(config_name)
            self.viz_config = self.experiment_config.get("visualization", {})
        except (AttributeError, KeyError) as e:
            logger.warning(f"Could not load visualization config: {e}. Using defaults.")
            self.config = None
            self.experiment_config = {}
            self.viz_config = {}

        # Store output_dir for test compatibility
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True)
        else:
            self.output_dir = Path("outputs")

        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")

    def load_and_prepare_data(self, filepath: str) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load activation data and prepare for visualization.

        Args:
            filepath: Path to CSV file containing activation data.

        Returns:
            Tuple of (activation_matrix, dataframe).
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded data from '{filepath}'. Rows: {len(df)}")
        except FileNotFoundError:
            logger.error(f"File '{filepath}' not found")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise

        # Remove rows with missing activations
        original_len = len(df)
        df.dropna(subset=['activation_vector'], inplace=True)
        if len(df) < original_len:
            logger.warning(f"Removed {original_len - len(df)} rows with missing activations")

        if df.empty:
            raise ValueError("No valid activation data found")

        # Parse activation vectors
        logger.info("Parsing activation vectors...")
        activation_matrix = []
        valid_indices = []

        for idx, row in df.iterrows():
            try:
                # Parse string representation of list
                if isinstance(row['activation_vector'], str):
                    activation_vector = ast.literal_eval(row['activation_vector'])
                else:
                    activation_vector = row['activation_vector']

                if isinstance(activation_vector, list):
                    activation_matrix.append(activation_vector)
                    valid_indices.append(idx)
                else:
                    logger.warning(f"Row {idx}: Invalid activation vector type")

            except (ValueError, SyntaxError) as e:
                logger.warning(f"Row {idx}: Error parsing activation vector - {e}")

        if not activation_matrix:
            raise ValueError("No valid activation vectors found")

        # Convert to numpy array and filter dataframe
        activation_matrix = np.array(activation_matrix)
        df_filtered = df.loc[valid_indices].reset_index(drop=True)

        logger.info(f"Activation matrix shape: {activation_matrix.shape}")
        logger.info(f"Valid samples: {len(df_filtered)}")

        return activation_matrix, df_filtered

    def apply_pca(self, activation_matrix: np.ndarray,
                  n_components: int = 2) -> Tuple[np.ndarray, PCA]:
        """Apply PCA dimensionality reduction.

        Args:
            activation_matrix: Input activation matrix.
            n_components: Number of components to keep.

        Returns:
            Tuple of (reduced_data, pca_object).
        """
        logger.info(f"Applying PCA with {n_components} components...")

        # Standardize the data
        scaler = StandardScaler()
        activation_scaled = scaler.fit_transform(activation_matrix)

        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(activation_scaled)

        explained_variance = pca.explained_variance_ratio_
        logger.info(f"PCA explained variance ratio: {explained_variance}")
        logger.info(f"Total explained variance: {explained_variance.sum():.3f}")

        return pca_result, pca

    def apply_tsne(self, activation_matrix: np.ndarray,
                   n_components: int = 2, perplexity: int = 30,
                   learning_rate: int = 200, n_iter: int = 1000) -> np.ndarray:
        """Apply t-SNE dimensionality reduction.

        Args:
            activation_matrix: Input activation matrix.
            n_components: Number of components to keep.
            perplexity: t-SNE perplexity parameter.
            learning_rate: t-SNE learning rate.
            n_iter: Number of iterations.

        Returns:
            Reduced data array.
        """
        logger.info(f"Applying t-SNE with {n_components} components...")

        # Standardize the data first
        scaler = StandardScaler()
        activation_scaled = scaler.fit_transform(activation_matrix)

        # Apply t-SNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=42,
            verbose=1
        )

        tsne_result = tsne.fit_transform(activation_scaled)
        logger.info("t-SNE completed")

        return tsne_result

    # plot_scatter method defined below with legacy compatibility

    def plot_scatter(self, data_2d: np.ndarray, title: str, filename: str,
                     df_questions: Optional[pd.DataFrame] = None) -> None:
        """
        Legacy-compatible scatter plot method.

        Args:
            data_2d: 2D reduced data array
            title: Plot title
            filename: Output filename
            df_questions: DataFrame with question metadata (optional)
        """
        plt.figure(figsize=(12, 10))

        # Create scatter plot with enhanced styling
        scatter = sns.scatterplot(
            x=data_2d[:, 0],
            y=data_2d[:, 1],
            s=50,
            alpha=0.7
        )

        plt.title(title, fontsize=16)
        plt.xlabel("Component 1", fontsize=12)
        plt.ylabel("Component 2", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        logger.info(f"Scatter plot saved as '{filename}'")
        plt.close()

    # plot_heatmap method defined below with improved implementation

    def plot_heatmap(self, activation_matrix: np.ndarray, df: pd.DataFrame,
                     output_file: str, max_questions: int = 50,
                     max_neurons: int = 100) -> None:
        """Create heatmap of activation patterns.

        Args:
            activation_matrix: Full activation matrix.
            df: DataFrame with metadata.
            output_file: Output file path.
            max_questions: Maximum number of questions to show.
            max_neurons: Maximum number of neurons to show.
        """
        # Limit data for readability
        n_questions = min(max_questions, activation_matrix.shape[0])
        n_neurons = min(max_neurons, activation_matrix.shape[1])

        subset_data = activation_matrix[:n_questions, :n_neurons]

        plt.figure(figsize=(15, 10))

        # Create heatmap
        sns.heatmap(subset_data,
                   cmap='viridis',
                   xticklabels=False,
                   yticklabels=False,
                   cbar_kws={'label': 'Activation Value'})

        plt.title(f'Activation Heatmap (First {n_questions} questions, {n_neurons} neurons)')
        plt.xlabel('Neuron Index')
        plt.ylabel('Question Index')

        # Save plot
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Heatmap saved to {output_path}")

    def plot_activation_statistics(self, activation_matrix: np.ndarray,
                                 output_file: str) -> None:
        """Plot activation statistics and distributions.

        Args:
            activation_matrix: Activation matrix.
            output_file: Output file path.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Activation value distribution
        axes[0, 0].hist(activation_matrix.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Distribution of Activation Values')
        axes[0, 0].set_xlabel('Activation Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Mean activation per neuron
        neuron_means = np.mean(activation_matrix, axis=0)
        axes[0, 1].plot(neuron_means)
        axes[0, 1].set_title('Mean Activation per Neuron')
        axes[0, 1].set_xlabel('Neuron Index')
        axes[0, 1].set_ylabel('Mean Activation')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Activation variance per neuron
        neuron_vars = np.var(activation_matrix, axis=0)
        axes[1, 0].plot(neuron_vars)
        axes[1, 0].set_title('Activation Variance per Neuron')
        axes[1, 0].set_xlabel('Neuron Index')
        axes[1, 0].set_ylabel('Activation Variance')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Mean activation per question
        question_means = np.mean(activation_matrix, axis=1)
        axes[1, 1].hist(question_means, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Distribution of Mean Activation per Question')
        axes[1, 1].set_xlabel('Mean Activation')
        axes[1, 1].set_ylabel('Number of Questions')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Statistics plot saved to {output_path}")

    def run_visualization(self, input_file: Optional[str] = None) -> bool:
        """Run the complete visualization process.

        Args:
            input_file: Path to input CSV file. If None, uses config.

        Returns:
            True if visualization completed successfully.
        """
        if input_file is None:
            input_file = self.viz_config["input_file"]

        output_dir = Path(self.viz_config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting visualization process")
        logger.info(f"Input file: {input_file}")
        logger.info(f"Output directory: {output_dir}")

        try:
            # Load and prepare data
            activation_matrix, df = self.load_and_prepare_data(input_file)

            # Get visualization methods from config
            methods = self.viz_config.get("methods", ["pca", "tsne", "heatmap"])

            # Apply PCA if requested
            if "pca" in methods:
                pca_config = self.viz_config.get("pca", {})
                n_components = pca_config.get("n_components", 2)

                pca_result, pca_object = self.apply_pca(activation_matrix, n_components)

                self.plot_scatter(
                    pca_result, df,
                    title=f"PCA Visualization of Neural Activations\\n"
                          f"(Explained Variance: {pca_object.explained_variance_ratio_.sum():.3f})",
                    output_file=output_dir / "activation_pca_scatter.png",
                    method="PCA"
                )

            # Apply t-SNE if requested
            if "tsne" in methods:
                tsne_config = self.viz_config.get("tsne", {})

                tsne_result = self.apply_tsne(
                    activation_matrix,
                    n_components=tsne_config.get("n_components", 2),
                    perplexity=tsne_config.get("perplexity", 30),
                    learning_rate=tsne_config.get("learning_rate", 200),
                    n_iter=tsne_config.get("n_iter", 1000)
                )

                self.plot_scatter(
                    tsne_result, df,
                    title="t-SNE Visualization of Neural Activations",
                    output_file=output_dir / "activation_tsne_scatter.png",
                    method="t-SNE"
                )

            # Create heatmap if requested
            if "heatmap" in methods:
                heatmap_config = self.viz_config.get("heatmap", {})

                self.plot_heatmap(
                    activation_matrix, df,
                    output_file=output_dir / "activation_heatmap.png",
                    max_questions=heatmap_config.get("max_questions", 50),
                    max_neurons=heatmap_config.get("max_neurons", 100)
                )

            # Create statistics plot
            self.plot_activation_statistics(
                activation_matrix,
                output_file=output_dir / "activation_statistics.png"
            )

            logger.info("Visualization completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return False


def main():
    """Main function for command line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize neural network activations")
    parser.add_argument("--config", default="default", help="Configuration name to use")
    parser.add_argument("--input-file", help="Path to input CSV file")
    parser.add_argument("--methods", nargs="+", choices=["pca", "tsne", "heatmap"],
                       default=["pca", "tsne", "heatmap"], help="Visualization methods to use")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    visualizer = ActivationVisualizer(args.config)

    # Override methods if specified
    if hasattr(args, 'methods') and args.methods:
        visualizer.viz_config["methods"] = args.methods

    success = visualizer.run_visualization(args.input_file)

    if success:
        logger.info("Visualization completed successfully!")
    else:
        logger.error("Visualization failed!")
        exit(1)


if __name__ == "__main__":
    main()
