"""
TensorBoard Integration for NeuronMap

Provides seamless integration with TensorBoard for visualization and logging.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import tempfile

try:
    from torch.utils.tensorboard import SummaryWriter
    import torch
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    torch = None
    TENSORBOARD_AVAILABLE = False

# Internal imports
from utils.error_handling import NeuronMapError
from utils.monitoring import setup_monitoring

logger = logging.getLogger(__name__)

class TensorBoardIntegration:
    """
    Integration with TensorBoard for visualization and experiment tracking.

    Provides functionality to:
    - Log activation patterns and statistics
    - Visualize model architectures
    - Track experiment metrics
    - Create interactive dashboards
    - Export NeuronMap results to TensorBoard format
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
        enable_graph_logging: bool = True
    ):
        """
        Initialize TensorBoard integration.

        Args:
            log_dir: Directory for TensorBoard logs
            experiment_name: Name of the experiment
            enable_graph_logging: Whether to log model graphs
        """
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard integration requires PyTorch and tensorboard")

        self.log_dir = Path(log_dir) if log_dir else Path("./tensorboard_logs")
        self.experiment_name = experiment_name or f"neuronmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.enable_graph_logging = enable_graph_logging

        # Create experiment directory
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SummaryWriter
        self.writer = SummaryWriter(log_dir=str(self.experiment_dir))

        # Monitoring
        self.monitor = setup_monitoring()

        # Tracking variables
        self.global_step = 0
        self.logged_models = set()

        logger.info(f"TensorBoard integration initialized: {self.experiment_dir}")

    def log_model_architecture(
        self,
        model: Any,
        model_name: str,
        input_sample: Optional[Any] = None
    ) -> bool:
        """
        Log model architecture to TensorBoard.

        Args:
            model: PyTorch model to log
            model_name: Name of the model
            input_sample: Sample input for the model

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.enable_graph_logging:
                return True

            if model_name in self.logged_models:
                logger.debug(f"Model {model_name} already logged")
                return True

            if input_sample is not None:
                # Log model graph
                self.writer.add_graph(model, input_sample)
                logger.info(f"Logged model graph for {model_name}")

            # Log model summary
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.writer.add_text(
                f"model_info/{model_name}",
                f"""
                **Model**: {model_name}
                **Total Parameters**: {total_params:,}
                **Trainable Parameters**: {trainable_params:,}
                **Architecture**: {type(model).__name__}
                """,
                self.global_step
            )

            self.logged_models.add(model_name)
            return True

        except Exception as e:
            logger.error(f"Failed to log model architecture: {e}")
            return False

    def log_activation_statistics(
        self,
        activations: Dict[str, Any],
        model_name: str,
        layer_names: Optional[List[str]] = None,
        step: Optional[int] = None
    ):
        """
        Log activation statistics to TensorBoard.

        Args:
            activations: Dictionary of layer activations
            model_name: Name of the model
            layer_names: Names of layers (optional)
            step: Global step for logging
        """
        if step is None:
            step = self.global_step

        try:
            for layer_idx, activation in activations.items():
                layer_name = f"layer_{layer_idx}"
                if layer_names and layer_idx < len(layer_names):
                    layer_name = layer_names[layer_idx]

                # Convert to numpy if torch tensor
                if hasattr(activation, 'detach'):
                    activation = activation.detach().cpu().numpy()

                # Flatten activation for statistics
                flat_activation = activation.flatten()

                # Log statistics
                self.writer.add_scalar(
                    f"{model_name}/activations/{layer_name}/mean",
                    np.mean(flat_activation),
                    step
                )

                self.writer.add_scalar(
                    f"{model_name}/activations/{layer_name}/std",
                    np.std(flat_activation),
                    step
                )

                self.writer.add_scalar(
                    f"{model_name}/activations/{layer_name}/max",
                    np.max(flat_activation),
                    step
                )

                self.writer.add_scalar(
                    f"{model_name}/activations/{layer_name}/min",
                    np.min(flat_activation),
                    step
                )

                # Log activation distribution
                self.writer.add_histogram(
                    f"{model_name}/activations/{layer_name}/distribution",
                    flat_activation,
                    step
                )

                # Log sparsity (percentage of near-zero activations)
                sparsity = np.mean(np.abs(flat_activation) < 1e-6)
                self.writer.add_scalar(
                    f"{model_name}/activations/{layer_name}/sparsity",
                    sparsity,
                    step
                )

            logger.info(f"Logged activation statistics for {len(activations)} layers")

        except Exception as e:
            logger.error(f"Failed to log activation statistics: {e}")

    def log_attention_patterns(
        self,
        attention_weights: Dict[str, Any],
        model_name: str,
        input_tokens: Optional[List[str]] = None,
        step: Optional[int] = None
    ):
        """
        Log attention patterns to TensorBoard.

        Args:
            attention_weights: Attention weight matrices
            model_name: Name of the model
            input_tokens: Input tokens for labeling
            step: Global step for logging
        """
        if step is None:
            step = self.global_step

        try:
            for layer_idx, attention in attention_weights.items():
                if hasattr(attention, 'detach'):
                    attention = attention.detach().cpu().numpy()

                # Average across heads if multi-head attention
                if len(attention.shape) > 2:
                    attention_avg = np.mean(attention, axis=0)  # Average over heads
                else:
                    attention_avg = attention

                # Log attention heatmap as image
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(attention_avg, cmap='Blues', aspect='auto')

                if input_tokens:
                    ax.set_xticks(range(len(input_tokens)))
                    ax.set_yticks(range(len(input_tokens)))
                    ax.set_xticklabels(input_tokens, rotation=45)
                    ax.set_yticklabels(input_tokens)

                ax.set_title(f"Layer {layer_idx} Attention Pattern")
                plt.colorbar(im)
                plt.tight_layout()

                # Convert to image and log
                self.writer.add_figure(
                    f"{model_name}/attention/layer_{layer_idx}",
                    fig,
                    step
                )
                plt.close(fig)

                # Log attention statistics
                self.writer.add_scalar(
                    f"{model_name}/attention/layer_{layer_idx}/entropy",
                    -np.sum(attention_avg * np.log(attention_avg + 1e-12)),
                    step
                )

                # Log max attention weight per position
                max_attention = np.max(attention_avg, axis=1)
                self.writer.add_histogram(
                    f"{model_name}/attention/layer_{layer_idx}/max_weights",
                    max_attention,
                    step
                )

            logger.info(f"Logged attention patterns for {len(attention_weights)} layers")

        except Exception as e:
            logger.error(f"Failed to log attention patterns: {e}")

    def log_experiment_metrics(
        self,
        metrics: Dict[str, float],
        experiment_id: str,
        step: Optional[int] = None
    ):
        """
        Log experiment metrics to TensorBoard.

        Args:
            metrics: Dictionary of metrics to log
            experiment_id: Identifier for the experiment
            step: Global step for logging
        """
        if step is None:
            step = self.global_step

        try:
            for metric_name, value in metrics.items():
                self.writer.add_scalar(
                    f"experiments/{experiment_id}/{metric_name}",
                    value,
                    step
                )

            logger.info(f"Logged {len(metrics)} metrics for experiment {experiment_id}")

        except Exception as e:
            logger.error(f"Failed to log experiment metrics: {e}")

    def _log_activations_to_tensorboard(self, activations: Dict[str, Any], model_name: str, step: Optional[int]):
        if 'activations' in activations:
            self.log_activation_statistics(
                activations['activations'],
                model_name,
                step=step
            )

    def _log_attention_to_tensorboard(self, results: Dict[str, Any], model_name: str, step: Optional[int]):
        if 'attention_weights' in results:
            self.log_attention_patterns(
                results['attention_weights'],
                model_name,
                step=step
            )

    def _log_layer_stats_to_tensorboard(self, results: Dict[str, Any], model_name: str, analysis_type: str, step: Optional[int]):
        if 'layer_statistics' in results:
            for layer, stats in results['layer_statistics'].items():
                for stat_name, value in stats.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(
                            f"{model_name}/{analysis_type}/layer_{layer}/{stat_name}",
                            value,
                            step
                        )

    def _log_discriminative_neurons_to_tensorboard(self, results: Dict[str, Any], model_name: str, analysis_type: str, step: Optional[int]):
        if 'discriminative_neurons' in results:
            for layer, scores in results['discriminative_neurons'].items():
                if scores:
                    self.writer.add_histogram(
                        f"{model_name}/{analysis_type}/discriminative_neurons/layer_{layer}",
                        np.array(scores),
                        step
                    )

    def _log_metadata_to_tensorboard(self, analysis_type: str, model_name: str, step: Optional[int]):
        metadata = {
            'analysis_type': analysis_type,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'step': step
        }
        self.writer.add_text(
            f"{model_name}/{analysis_type}/metadata",
            json.dumps(metadata, indent=2),
            step
        )

    def log_analysis_results(
        self,
        results: Dict[str, Any],
        analysis_type: str,
        model_name: str,
        step: Optional[int] = None
    ):
        """Log comprehensive analysis results to TensorBoard."""
        if step is None:
            step = self.global_step

        try:
            self._log_activations_to_tensorboard(results, model_name, step)
            self._log_attention_to_tensorboard(results, model_name, step)
            self._log_layer_stats_to_tensorboard(results, model_name, analysis_type, step)
            self._log_discriminative_neurons_to_tensorboard(results, model_name, analysis_type, step)
            self._log_metadata_to_tensorboard(analysis_type, model_name, step)

            logger.info(f"Logged analysis results for {analysis_type}")

        except Exception as e:
            logger.error(f"Failed to log analysis results: {e}")

    def create_comparison_dashboard(
        self,
        comparison_results: Dict[str, Dict[str, Any]],
        comparison_name: str,
        step: Optional[int] = None
    ):
        """
        Create a comparison dashboard in TensorBoard.

        Args:
            comparison_results: Results from multiple models/experiments
            comparison_name: Name for the comparison
            step: Global step for logging
        """
        if step is None:
            step = self.global_step

        try:
            # Create comparison metrics
            import matplotlib.pyplot as plt

            # Performance comparison
            model_names = list(comparison_results.keys())

            # Extract common metrics for comparison
            common_metrics = set()
            for results in comparison_results.values():
                if 'metrics' in results:
                    common_metrics.update(results['metrics'].keys())

            for metric in common_metrics:
                metric_values = []
                valid_models = []

                for model_name, results in comparison_results.items():
                    if 'metrics' in results and metric in results['metrics']:
                        metric_values.append(results['metrics'][metric])
                        valid_models.append(model_name)

                if metric_values:
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bars = ax.bar(valid_models, metric_values)
                    ax.set_title(f"{comparison_name}: {metric.replace('_', ' ').title()}")
                    ax.set_ylabel(metric)
                    plt.xticks(rotation=45)

                    # Add value labels on bars
                    for bar, value in zip(bars, metric_values):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{value:.4f}', ha='center', va='bottom')

                    plt.tight_layout()

                    self.writer.add_figure(
                        f"comparisons/{comparison_name}/{metric}",
                        fig,
                        step
                    )
                    plt.close(fig)

            # Log comparison summary
            summary = {
                'comparison_name': comparison_name,
                'models_compared': model_names,
                'metrics_compared': list(common_metrics),
                'timestamp': datetime.now().isoformat()
            }

            self.writer.add_text(
                f"comparisons/{comparison_name}/summary",
                json.dumps(summary, indent=2),
                step
            )

            logger.info(f"Created comparison dashboard: {comparison_name}")

        except Exception as e:
            logger.error(f"Failed to create comparison dashboard: {e}")

    def export_neuronmap_session(
        self,
        session_data: Dict[str, Any],
        session_name: str
    ):
        """
        Export a complete NeuronMap session to TensorBoard.

        Args:
            session_data: Complete session data from NeuronMap
            session_name: Name for the session
        """
        try:
            # Log session metadata
            self.writer.add_text(
                f"sessions/{session_name}/metadata",
                json.dumps(session_data.get('metadata', {}), indent=2),
                self.global_step
            )

            # Log each analysis in the session
            if 'analyses' in session_data:
                for analysis_id, analysis_data in session_data['analyses'].items():
                    self.log_analysis_results(
                        analysis_data['results'],
                        analysis_data.get('analysis_type', 'unknown'),
                        analysis_data.get('model_name', 'unknown'),
                        step=self.global_step
                    )
                    self.global_step += 1

            # Log session summary
            summary = {
                'session_name': session_name,
                'num_analyses': len(session_data.get('analyses', {})),
                'export_timestamp': datetime.now().isoformat()
            }

            self.writer.add_text(
                f"sessions/{session_name}/summary",
                json.dumps(summary, indent=2),
                self.global_step
            )

            logger.info(f"Exported NeuronMap session: {session_name}")

        except Exception as e:
            logger.error(f"Failed to export session: {e}")

    def start_tensorboard_server(
        self,
        port: int = 6006,
        host: str = "localhost"
    ) -> bool:
        """
        Start TensorBoard server for viewing logs.

        Args:
            port: Port to run TensorBoard on
            host: Host to bind to

        Returns:
            True if server started successfully
        """
        try:
            import subprocess
            import threading

            cmd = [
                "tensorboard",
                "--logdir", str(self.log_dir),
                "--port", str(port),
                "--host", host
            ]

            def run_tensorboard():
                subprocess.run(cmd)

            # Start in background thread
            thread = threading.Thread(target=run_tensorboard, daemon=True)
            thread.start()

            logger.info(f"TensorBoard server starting on http://{host}:{port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start TensorBoard server: {e}")
            return False

    def close(self):
        """Close the TensorBoard writer and clean up."""
        try:
            if self.writer:
                self.writer.close()
            logger.info("TensorBoard integration closed")
        except Exception as e:
            logger.error(f"Error closing TensorBoard integration: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

# Utility functions for TensorBoard integration
def create_tensorboard_config(
    log_dir: str = "./tensorboard_logs",
    experiment_name: Optional[str] = None,
    enable_graph_logging: bool = True
) -> Dict[str, Any]:
    """Create configuration for TensorBoard integration."""
    return {
        'log_dir': log_dir,
        'experiment_name': experiment_name,
        'enable_graph_logging': enable_graph_logging
    }

def export_to_tensorboard(
    neuronmap_results: Dict[str, Any],
    tensorboard_config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Quick function to export NeuronMap results to TensorBoard.

    Args:
        neuronmap_results: Results from NeuronMap analysis
        tensorboard_config: TensorBoard configuration

    Returns:
        True if export successful
    """
    config = tensorboard_config or create_tensorboard_config()

    try:
        with TensorBoardIntegration(**config) as tb:
            tb.log_analysis_results(
                neuronmap_results,
                neuronmap_results.get('analysis_type', 'unknown'),
                neuronmap_results.get('model_name', 'unknown')
            )
        return True
    except Exception as e:
        logger.error(f"Failed to export to TensorBoard: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Example TensorBoard integration
    with TensorBoardIntegration(experiment_name="example_experiment") as tb:
        # Mock data for demonstration
        mock_activations = {
            0: np.random.randn(32, 768),
            6: np.random.randn(32, 768),
            11: np.random.randn(32, 768)
        }

        mock_attention = {
            6: np.random.rand(12, 32, 32),  # 12 heads, 32x32 attention
            11: np.random.rand(12, 32, 32)
        }

        # Log examples
        tb.log_activation_statistics(mock_activations, "bert-base-uncased")
        tb.log_attention_patterns(mock_attention, "bert-base-uncased")

        # Log experiment metrics
        tb.log_experiment_metrics({
            'accuracy': 0.95,
            'interpretability_score': 0.78,
            'efficiency': 0.85
        }, "experiment_1")

        print("Example data logged to TensorBoard")
        print(f"View at: tensorboard --logdir {tb.experiment_dir}")
