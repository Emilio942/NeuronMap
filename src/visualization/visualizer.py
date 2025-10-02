"""Visualization tools for neural network activations in NeuronMap."""

import ast
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency handling
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - fallback when matplotlib is missing
    plt = None  # type: ignore

try:  # pragma: no cover - optional dependency handling
    import seaborn as sns
except Exception:  # pragma: no cover - fallback when seaborn is missing
    sns = None  # type: ignore

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from ..utils.config import get_config


logger = logging.getLogger(__name__)


_DEFAULT_PLOT_CONFIG: Dict[str, Any] = {
    "figure_size": (12, 8),
    "dpi": 120,
    "scatter_size": 60,
    "scatter_alpha": 0.75,
    "color_palette": "husl",
    "heatmap_size": (12, 8),
    "heatmap_cmap": "viridis",
    "comparison_size": (12, 6),
    "max_questions": 50,
    "max_neurons": 100,
    "show_grid": True,
    "annotate_points": False,
    "methods": ["pca", "tsne", "heatmap"],
    "output_dir": "outputs",
    "input_file": "data/activations.csv",
    "generate_statistics": True,
    "x_label": "Component 1",
    "y_label": "Component 2",
}


def get_default_plot_config() -> Dict[str, Any]:
    """Return a shallow copy of the default plot configuration."""
    return dict(_DEFAULT_PLOT_CONFIG)


def _rgb_to_hex(color: Tuple[float, float, float]) -> str:
    r, g, b = [int(round(max(0.0, min(1.0, channel)) * 255)) for channel in color[:3]]
    return f"#{r:02x}{g:02x}{b:02x}"


def generate_color_palette(count: int) -> List[str]:
    """Generate a palette of distinct colors for plotting."""
    if count <= 0:
        return []

    palette_name = _DEFAULT_PLOT_CONFIG.get("color_palette", "husl")

    if sns is not None:
        try:
            palette = sns.color_palette(palette_name, count)
            return [_rgb_to_hex(color) for color in palette]
        except Exception:  # pragma: no cover - fallback when palette not available
            warnings.warn("Falling back to static palette; seaborn palette unavailable.")

    base_colors = [
        "#4C78A8",
        "#F58518",
        "#E45756",
        "#72B7B2",
        "#54A24B",
        "#EECA3B",
        "#B279A2",
        "#FF9DA6",
        "#9C755F",
        "#BAB0AC",
    ]

    while len(base_colors) < count:
        base_colors.extend(base_colors)

    return base_colors[:count]


class ActivationVisualizer:
    """Visualize neural network activation patterns."""

    def __init__(self, config_name: str = "default", output_dir: Optional[str] = None):
        self._matplotlib_available = plt is not None
        self._seaborn_available = sns is not None

        self.config, self.experiment_config = self._load_configuration(config_name)
        self.viz_config = get_default_plot_config()

        if isinstance(self.experiment_config, dict):
            viz_settings = self.experiment_config.get("visualization", {})
            if isinstance(viz_settings, dict):
                self.viz_config.update(viz_settings)

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(self.viz_config.get("output_dir", "outputs"))

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self._matplotlib_available:
            plt.style.use("default")

        if self._seaborn_available:
            try:
                sns.set_palette(self.viz_config.get("color_palette", "husl"))
            except Exception:  # pragma: no cover - palette application failure
                warnings.warn("Failed to set seaborn palette; continuing with defaults.")

        self._last_pca_model: Optional[PCA] = None
        self._last_tsne_result: Optional[np.ndarray] = None

    def _load_configuration(self, config_name: str) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Load visualization configuration from the global config system."""
        try:
            from ..utils.config_manager import get_config_manager as _get_config_manager  # type: ignore

            config_manager = _get_config_manager()
            config_candidate = config_manager.get_config_model()

            if hasattr(config_candidate, "get_experiment_config"):
                experiment = config_candidate.get_experiment_config(config_name)
                return config_candidate, experiment
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            logger.debug("Typed visualization config unavailable: %s", exc)

        try:
            config_candidate = get_config()
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            logger.warning("Failed to load raw visualization config: %s", exc)
            return None, {}

        if hasattr(config_candidate, "get_experiment_config"):
            try:
                experiment = config_candidate.get_experiment_config(config_name)
            except Exception:
                experiment = {}
            return config_candidate, experiment

        if isinstance(config_candidate, dict):
            experiment = self._extract_experiment_config(config_candidate, config_name)
            return config_candidate, experiment

        return config_candidate, {}

    @staticmethod
    def _extract_experiment_config(config_dict: Dict[str, Any], config_name: str) -> Dict[str, Any]:
        experiments = config_dict.get("experiments", {})
        if isinstance(experiments, dict):
            if config_name in experiments and isinstance(experiments[config_name], dict):
                return experiments[config_name]
            default_experiment = experiments.get("default")
            if isinstance(default_experiment, dict):
                return default_experiment

        top_level_default = config_dict.get("default")
        if isinstance(top_level_default, dict):
            return top_level_default

        experiment = config_dict.get("experiment")
        if isinstance(experiment, dict):
            return experiment

        return {"name": config_name or "default", "visualization": {}}

    def _ensure_matplotlib(self) -> None:
        if not self._matplotlib_available:
            raise ImportError("matplotlib is required for this visualization but is not available.")

    def _ensure_seaborn(self) -> None:
        if not self._seaborn_available:
            raise ImportError("seaborn is required for this visualization but is not available.")

    @staticmethod
    def _validate_array(data: np.ndarray) -> np.ndarray:
        if data is None:
            raise ValueError("Activation data cannot be None.")

        array = np.asarray(data)
        if array.ndim != 2:
            raise ValueError("Activation data must be a 2D array.")

        if array.size == 0:
            raise ValueError("Activation data cannot be empty.")

        return array

    def _standardize_activations(self, activations: np.ndarray) -> np.ndarray:
        data = self._validate_array(activations)
        scaler = StandardScaler()
        return scaler.fit_transform(data)

    def load_and_prepare_data(self, filepath: str) -> Tuple[np.ndarray, pd.DataFrame]:
        try:
            df = pd.read_csv(filepath)
            logger.info("Loaded activation data from %s (rows=%s)", filepath, len(df))
        except FileNotFoundError:
            logger.error("Activation file '%s' not found", filepath)
            raise
        except Exception as exc:
            logger.error("Failed to load activation file '%s': %s", filepath, exc)
            raise

        original_len = len(df)
        df.dropna(subset=["activation_vector"], inplace=True)
        if len(df) < original_len:
            logger.warning("Removed %s rows with missing activations", original_len - len(df))

        if df.empty:
            raise ValueError("No valid activation data found")

        activation_matrix: List[List[float]] = []
        valid_indices: List[int] = []

        for idx, row in df.iterrows():
            try:
                value = row["activation_vector"]
                if isinstance(value, str):
                    value = ast.literal_eval(value)

                if isinstance(value, list):
                    activation_matrix.append(value)
                    valid_indices.append(idx)
                else:
                    logger.debug("Row %s: invalid activation vector type %s", idx, type(value))
            except (ValueError, SyntaxError) as exc:
                logger.debug("Row %s: failed to parse activation vector (%s)", idx, exc)

        if not activation_matrix:
            raise ValueError("No valid activation vectors found")

        matrix = np.asarray(activation_matrix)
        df_filtered = df.loc[valid_indices].reset_index(drop=True)

        logger.info("Prepared activation matrix with shape %s", matrix.shape)
        return matrix, df_filtered

    def _compute_pca(self, activations: np.ndarray, n_components: int = 2,
                      return_model: bool = False) -> Any:
        data = self._standardize_activations(activations)
        max_components = max(1, min(data.shape[0], data.shape[1]))
        components = max(1, min(n_components, max_components))

        pca = PCA(n_components=components)
        transformed = pca.fit_transform(data)

        self._last_pca_model = pca

        if return_model:
            return transformed, pca
        return transformed

    def _compute_tsne(self, activations: np.ndarray, n_components: int = 2,
                       perplexity: int = 30, learning_rate: int = 200,
                       n_iter: int = 1000) -> np.ndarray:
        data = self._standardize_activations(activations)
        n_samples = data.shape[0]

        if n_samples < 2:
            raise ValueError("t-SNE requires at least two samples")

        max_perplexity = min(perplexity, max(1, (n_samples - 1) // 3))
        if max_perplexity < 1:
            max_perplexity = 1

        tsne_kwargs = {
            "n_components": min(n_components, data.shape[1], 3),
            "perplexity": max_perplexity,
            "learning_rate": learning_rate,
            "random_state": 42,
            "init": "pca" if data.shape[1] > 3 else "random",
            "verbose": 0,
        }

        try:
            tsne = TSNE(**tsne_kwargs, n_iter=n_iter)
        except TypeError:  # pragma: no cover - compatibility with older sklearn
            tsne = TSNE(**tsne_kwargs)

        embedding = tsne.fit_transform(data)
        self._last_tsne_result = embedding
        return embedding

    def apply_pca(self, activation_matrix: np.ndarray,
                  n_components: int = 2) -> Tuple[np.ndarray, PCA]:
        logger.info("Applying PCA with %s components", n_components)
        result, model = self._compute_pca(activation_matrix, n_components=n_components, return_model=True)
        explained = getattr(model, "explained_variance_ratio_", None)
        if explained is not None:
            logger.info("Total explained variance: %.3f", float(np.sum(explained)))
        return result, model

    def apply_tsne(self, activation_matrix: np.ndarray,
                   n_components: int = 2, perplexity: int = 30,
                   learning_rate: int = 200, n_iter: int = 1000) -> np.ndarray:
        logger.info("Applying t-SNE with %s components", n_components)
        return self._compute_tsne(
            activation_matrix,
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
        )

    def plot_scatter(self, data_2d: np.ndarray, labels: Optional[List[str]] = None,
                     title: Optional[str] = None, save_path: Optional[str] = None,
                     method: str = "Scatter") -> Dict[str, Any]:
        self._ensure_matplotlib()

        data = self._validate_array(data_2d)
        if data.shape[1] < 2:
            raise ValueError("Reduced data must have at least 2 components for plotting")

        figure_size = tuple(self.viz_config.get("figure_size", (12, 8)))
        dpi = int(self.viz_config.get("dpi", 120))
        scatter_size = int(self.viz_config.get("scatter_size", 60))

        plt.figure(figsize=figure_size)

        colors = None
        if labels:
            unique_labels = list(dict.fromkeys(labels))
            palette = generate_color_palette(len(unique_labels))
            color_map = {label: palette[idx % len(palette)] for idx, label in enumerate(unique_labels)}
            colors = [color_map.get(label, palette[0]) for label in labels]

        plt.scatter(
            data[:, 0],
            data[:, 1],
            c=colors,
            s=scatter_size,
            alpha=self.viz_config.get("scatter_alpha", 0.75),
            edgecolors="none",
        )

        if labels and self.viz_config.get("annotate_points", False):
            for idx, label in enumerate(labels):
                plt.annotate(label, (data[idx, 0], data[idx, 1]), fontsize=8, alpha=0.7)

        plot_title = title or f"{method} Visualization"
        plt.title(plot_title)
        plt.xlabel(self.viz_config.get("x_label", "Component 1"))
        plt.ylabel(self.viz_config.get("y_label", "Component 2"))
        if self.viz_config.get("show_grid", True):
            plt.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()

        if save_path:
            save_path = str(save_path)
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=dpi)
            logger.info("Scatter plot saved to %s", save_path)

        plt.close()
        return {"data": data, "labels": labels, "title": plot_title, "path": save_path}

    def create_pca_plot(self, activations: np.ndarray, labels: Optional[List[str]] = None,
                        save_path: Optional[str] = None, n_components: int = 2) -> Dict[str, Any]:
        reduced, model = self._compute_pca(activations, n_components=n_components, return_model=True)
        explained = getattr(model, "explained_variance_ratio_", None)

        title = "PCA Visualization"
        if explained is not None:
            title += f" (Explained Variance: {float(np.sum(explained)):.2f})"

        return {
            "reduced": reduced,
            "plot": self.plot_scatter(reduced, labels=labels, title=title,
                                       save_path=save_path, method="PCA"),
            "explained_variance": explained,
        }

    def create_tsne_plot(self, activations: np.ndarray, labels: Optional[List[str]] = None,
                         save_path: Optional[str] = None, n_components: int = 2,
                         perplexity: int = 30, learning_rate: int = 200,
                         n_iter: int = 1000) -> Dict[str, Any]:
        reduced = self._compute_tsne(
            activations,
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
        )

        return {
            "reduced": reduced,
            "plot": self.plot_scatter(reduced, labels=labels, title="t-SNE Visualization",
                                       save_path=save_path, method="t-SNE"),
        }

    def create_activation_heatmap(self, activations: np.ndarray,
                                   title: str = "Activation Heatmap",
                                   save_path: Optional[str] = None,
                                   max_questions: Optional[int] = None,
                                   max_neurons: Optional[int] = None) -> Dict[str, Any]:
        self._ensure_matplotlib()

        data = self._validate_array(activations)
        questions_limit = max_questions or self.viz_config.get("max_questions", min(50, data.shape[0]))
        neurons_limit = max_neurons or self.viz_config.get("max_neurons", min(100, data.shape[1]))

        trimmed = data[:questions_limit, :neurons_limit]

        plt.figure(figsize=tuple(self.viz_config.get("heatmap_size", (12, 8))))

        if self._seaborn_available:
            ax = sns.heatmap(
                trimmed,
                cmap=self.viz_config.get("heatmap_cmap", "viridis"),
                cbar_kws={"label": "Activation Value"},
                xticklabels=False,
                yticklabels=False,
            )
        else:
            warnings.warn("Seaborn unavailable; using matplotlib.imshow for heatmap.")
            ax = plt.imshow(trimmed, aspect="auto", cmap=self.viz_config.get("heatmap_cmap", "viridis"))
            plt.colorbar(label="Activation Value")

        plt.title(title)
        plt.xlabel("Neuron Index")
        plt.ylabel("Sample Index")
        plt.tight_layout()

        if save_path:
            save_path = str(save_path)
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=int(self.viz_config.get("dpi", 120)))
            logger.info("Heatmap saved to %s", save_path)

        plt.close()
        return {"data": trimmed, "path": save_path, "axes": ax}

    def create_layer_comparison_plot(self, layer_activations: Dict[str, np.ndarray],
                                     save_path: Optional[str] = None,
                                     metric: str = "mean_abs") -> Dict[str, Any]:
        self._ensure_matplotlib()

        if not layer_activations:
            raise ValueError("layer_activations must contain at least one layer")

        layer_names = list(layer_activations.keys())
        values: List[float] = []

        for layer in layer_names:
            activations = self._validate_array(layer_activations[layer])
            if metric == "mean_abs":
                values.append(float(np.mean(np.abs(activations))))
            elif metric == "std":
                values.append(float(np.std(activations)))
            else:
                values.append(float(np.mean(activations)))

        plt.figure(figsize=tuple(self.viz_config.get("comparison_size", (12, 6))))
        plt.plot(layer_names, values, marker="o")
        plt.title("Layer Activation Comparison")
        plt.xlabel("Layer")
        plt.ylabel(f"Activation ({metric})")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        if save_path:
            save_path = str(save_path)
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=int(self.viz_config.get("dpi", 120)))
            logger.info("Layer comparison plot saved to %s", save_path)

        plt.close()
        return {"layers": layer_names, "values": values, "path": save_path}

    def plot_activation_statistics(self, activation_matrix: np.ndarray,
                                   output_file: str) -> None:
        self._ensure_matplotlib()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].hist(activation_matrix.flatten(), bins=50, alpha=0.7, edgecolor="black")
        axes[0, 0].set_title("Distribution of Activation Values")
        axes[0, 0].set_xlabel("Activation Value")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].grid(True, alpha=0.3)

        neuron_means = np.mean(activation_matrix, axis=0)
        axes[0, 1].plot(neuron_means)
        axes[0, 1].set_title("Mean Activation per Neuron")
        axes[0, 1].set_xlabel("Neuron Index")
        axes[0, 1].set_ylabel("Mean Activation")
        axes[0, 1].grid(True, alpha=0.3)

        neuron_vars = np.var(activation_matrix, axis=0)
        axes[1, 0].plot(neuron_vars)
        axes[1, 0].set_title("Activation Variance per Neuron")
        axes[1, 0].set_xlabel("Neuron Index")
        axes[1, 0].set_ylabel("Variance")
        axes[1, 0].grid(True, alpha=0.3)

        question_means = np.mean(activation_matrix, axis=1)
        axes[1, 1].hist(question_means, bins=30, alpha=0.7, edgecolor="black")
        axes[1, 1].set_title("Distribution of Mean Activation per Question")
        axes[1, 1].set_xlabel("Mean Activation")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=int(self.viz_config.get("dpi", 120)))
        plt.close(fig)
        logger.info("Activation statistics saved to %s", output_path)

    def plot_heatmap(self, activation_matrix: np.ndarray, df: pd.DataFrame,
                     output_file: str, max_questions: int = 50,
                     max_neurons: int = 100) -> None:
        """Backward-compatible heatmap helper for legacy callers."""
        self.create_activation_heatmap(
            activation_matrix,
            title="Activation Heatmap",
            save_path=str(output_file),
            max_questions=max_questions,
            max_neurons=max_neurons,
        )

    def run_visualization(self, input_file: Optional[str] = None) -> bool:
        if input_file is None:
            input_file = self.viz_config.get("input_file")

        if not input_file:
            raise ValueError("No input file provided for visualization")

        output_dir = self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting visualization pipeline")
        logger.info("Input file: %s", input_file)
        logger.info("Output directory: %s", output_dir)

        try:
            activations, df = self.load_and_prepare_data(input_file)
            methods = self.viz_config.get("methods", ["pca", "tsne", "heatmap"])
            if isinstance(methods, str):
                methods = [methods]

            labels: Optional[List[str]] = None
            if isinstance(df, pd.DataFrame):
                for column in ("question", "prompt", "text", "content"):
                    if column in df.columns:
                        labels = df[column].astype(str).tolist()
                        break

            if "pca" in methods:
                pca_config = self.viz_config.get("pca", {})
                self.create_pca_plot(
                    activations,
                    labels=labels,
                    save_path=str(output_dir / "activation_pca_scatter.png"),
                    n_components=pca_config.get("n_components", 2),
                )

            if "tsne" in methods:
                tsne_config = self.viz_config.get("tsne", {})
                self.create_tsne_plot(
                    activations,
                    labels=labels,
                    save_path=str(output_dir / "activation_tsne_scatter.png"),
                    n_components=tsne_config.get("n_components", 2),
                    perplexity=tsne_config.get("perplexity", 30),
                    learning_rate=tsne_config.get("learning_rate", 200),
                    n_iter=tsne_config.get("n_iter", 1000),
                )

            if "heatmap" in methods:
                heatmap_config = self.viz_config.get("heatmap", {})
                self.create_activation_heatmap(
                    activations,
                    title="Activation Heatmap",
                    save_path=str(output_dir / "activation_heatmap.png"),
                    max_questions=heatmap_config.get("max_questions"),
                    max_neurons=heatmap_config.get("max_neurons"),
                )

            if self.viz_config.get("generate_statistics", True):
                self.plot_activation_statistics(
                    activations,
                    output_file=str(output_dir / "activation_statistics.png"),
                )

            logger.info("Visualization pipeline completed successfully")
            return True

        except Exception as exc:
            logger.error("Visualization pipeline failed: %s", exc)
            return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize neural network activations")
    parser.add_argument("--config", default="default", help="Configuration name to use")
    parser.add_argument("--input-file", help="Path to input CSV file")
    parser.add_argument("--methods", nargs="+", choices=["pca", "tsne", "heatmap"],
                        default=["pca", "tsne", "heatmap"], help="Visualization methods to use")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    visualizer = ActivationVisualizer(args.config)
    if args.methods:
        visualizer.viz_config["methods"] = args.methods

    success = visualizer.run_visualization(args.input_file)
    if success:
        logger.info("Visualization completed successfully")
    else:
        logger.error("Visualization failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
