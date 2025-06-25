"""
MLflow Integration for NeuronMap

Provides integration with MLflow for experiment tracking and model management.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import tempfile
import pickle

try:
    import mlflow
    import mlflow.pytorch
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False

# Internal imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.error_handling import NeuronMapError
from utils.monitoring import setup_monitoring

logger = logging.getLogger(__name__)

class MLFlowIntegration:
    """
    Integration with MLflow for experiment tracking and model management.

    Provides functionality to:
    - Track NeuronMap experiments and parameters
    - Log models and artifacts
    - Store analysis results and visualizations
    - Compare experiments across runs
    - Manage model versions
    """

    def __init__(
        self,
        experiment_name: str = "neuronmap-analysis",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        run_name: Optional[str] = None
    ):
        """
        Initialize MLflow integration.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
            artifact_location: Location for storing artifacts
            run_name: Name for the current run
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow integration requires mlflow package: pip install mlflow")

        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location
                )
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            logger.warning(f"Could not set experiment: {e}")
            experiment_id = None

        if experiment_id:
            mlflow.set_experiment(experiment_name)

        self.experiment_name = experiment_name
        self.run_name = run_name or f"neuronmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Start MLflow run
        self.run = mlflow.start_run(run_name=self.run_name)

        # Monitoring
        self.monitor = setup_monitoring()

        # Tracking variables
        self.logged_models = set()
        self.artifact_paths = {}

        logger.info(f"MLflow integration initialized: {self.run.info.run_name}")

    def log_parameters(self, params: Dict[str, Any]):
        """
        Log parameters for the experiment.

        Args:
            params: Dictionary of parameters to log
        """
        try:
            # Flatten nested parameters
            flat_params = self._flatten_params(params)

            for key, value in flat_params.items():
                # MLflow parameters must be strings
                mlflow.log_param(key, str(value))

            logger.info(f"Logged {len(flat_params)} parameters")

        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics for the experiment.

        Args:
            metrics: Dictionary of metrics to log
            step: Step number for the metrics
        """
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)
                else:
                    logger.warning(f"Skipping non-numeric metric: {key}")

            logger.info(f"Logged {len(metrics)} metrics")

        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def log_model_info(
        self,
        model: Any,
        model_name: str,
        model_type: str = "pytorch",
        input_example: Optional[Any] = None
    ):
        """
        Log model information and artifacts.

        Args:
            model: Model to log
            model_name: Name of the model
            model_type: Type of model (pytorch, sklearn, etc.)
            input_example: Example input for the model
        """
        try:
            if model_name in self.logged_models:
                logger.debug(f"Model {model_name} already logged")
                return

            # Log model parameters
            model_params = {
                f"model_{model_name}_type": type(model).__name__,
                f"model_{model_name}_name": model_name
            }

            # Try to get parameter count for PyTorch models
            try:
                if hasattr(model, 'parameters'):
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    model_params.update({
                        f"model_{model_name}_total_params": total_params,
                        f"model_{model_name}_trainable_params": trainable_params
                    })
            except Exception:
                pass

            self.log_parameters(model_params)

            # Log model artifact
            try:
                if model_type == "pytorch" and hasattr(mlflow, 'pytorch'):
                    mlflow.pytorch.log_model(
                        model,
                        f"models/{model_name}",
                        input_example=input_example
                    )
                elif model_type == "sklearn" and hasattr(mlflow, 'sklearn'):
                    mlflow.sklearn.log_model(
                        model,
                        f"models/{model_name}",
                        input_example=input_example
                    )
                else:
                    # Generic model logging using pickle
                    temp_path = tempfile.mktemp(suffix='.pkl')
                    with open(temp_path, 'wb') as f:
                        pickle.dump(model, f)
                    mlflow.log_artifact(temp_path, f"models/{model_name}")

                logger.info(f"Logged model: {model_name}")

            except Exception as e:
                logger.warning(f"Could not log model artifact: {e}")

            self.logged_models.add(model_name)

        except Exception as e:
            logger.error(f"Failed to log model info: {e}")

    def log_activation_analysis(
        self,
        activations: Dict[str, Any],
        model_name: str,
        analysis_config: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None
    ):
        """
        Log activation analysis results.

        Args:
            activations: Dictionary of layer activations
            model_name: Name of the analyzed model
            analysis_config: Configuration used for analysis
            step: Step number for metrics
        """
        try:
            # Log analysis configuration
            if analysis_config:
                config_params = {f"analysis_{k}": v for k, v in analysis_config.items()}
                self.log_parameters(config_params)

            # Compute and log activation metrics
            metrics = {}
            artifacts = {}

            for layer_idx, activation in activations.items():
                # Convert to numpy if needed
                if hasattr(activation, 'detach'):
                    activation = activation.detach().cpu().numpy()

                # Flatten for statistics
                flat_activation = activation.flatten()

                layer_prefix = f"{model_name}_layer_{layer_idx}"

                # Compute statistics
                metrics.update({
                    f"{layer_prefix}_mean": float(np.mean(flat_activation)),
                    f"{layer_prefix}_std": float(np.std(flat_activation)),
                    f"{layer_prefix}_max": float(np.max(flat_activation)),
                    f"{layer_prefix}_min": float(np.min(flat_activation)),
                    f"{layer_prefix}_sparsity": float(np.mean(np.abs(flat_activation) < 1e-6))
                })

                # Save activation data as artifact
                temp_path = tempfile.mktemp(suffix=f'_layer_{layer_idx}_activations.npy')
                np.save(temp_path, activation)
                artifacts[f"activations/layer_{layer_idx}"] = temp_path

            # Log metrics
            self.log_metrics(metrics, step=step)

            # Log artifacts
            for artifact_path, file_path in artifacts.items():
                mlflow.log_artifact(file_path, artifact_path)

            logger.info(f"Logged activation analysis for {len(activations)} layers")

        except Exception as e:
            logger.error(f"Failed to log activation analysis: {e}")

    def log_attention_analysis(
        self,
        attention_weights: Dict[str, Any],
        model_name: str,
        input_tokens: Optional[List[str]] = None,
        step: Optional[int] = None
    ):
        """
        Log attention analysis results.

        Args:
            attention_weights: Attention weight matrices
            model_name: Name of the analyzed model
            input_tokens: Input tokens for context
            step: Step number for metrics
        """
        try:
            metrics = {}
            artifacts = {}

            for layer_idx, attention in attention_weights.items():
                if hasattr(attention, 'detach'):
                    attention = attention.detach().cpu().numpy()

                layer_prefix = f"{model_name}_attention_layer_{layer_idx}"

                # Average across heads if multi-head
                if len(attention.shape) > 2:
                    attention_avg = np.mean(attention, axis=0)
                else:
                    attention_avg = attention

                # Compute attention metrics
                entropy = -np.sum(attention_avg * np.log(attention_avg + 1e-12))
                max_attention = np.max(attention_avg, axis=1)

                metrics.update({
                    f"{layer_prefix}_entropy": float(entropy),
                    f"{layer_prefix}_max_attention_mean": float(np.mean(max_attention)),
                    f"{layer_prefix}_max_attention_std": float(np.std(max_attention)),
                    f"{layer_prefix}_attention_concentration": float(np.std(attention_avg))
                })

                # Save attention data
                temp_path = tempfile.mktemp(suffix=f'_layer_{layer_idx}_attention.npy')
                np.save(temp_path, attention)
                artifacts[f"attention/layer_{layer_idx}"] = temp_path

                # Create attention visualization if possible
                try:
                    viz_path = self._create_attention_visualization(
                        attention_avg, input_tokens, f"Layer {layer_idx}"
                    )
                    if viz_path:
                        artifacts[f"attention_viz/layer_{layer_idx}"] = viz_path
                except Exception as e:
                    logger.debug(f"Could not create attention visualization: {e}")

            # Log metrics and artifacts
            self.log_metrics(metrics, step=step)

            for artifact_path, file_path in artifacts.items():
                mlflow.log_artifact(file_path, artifact_path)

            logger.info(f"Logged attention analysis for {len(attention_weights)} layers")

        except Exception as e:
            logger.error(f"Failed to log attention analysis: {e}")

    def log_experiment_results(
        self,
        results: Dict[str, Any],
        experiment_config: Dict[str, Any],
        model_name: str,
        step: Optional[int] = None
    ):
        """
        Log comprehensive experiment results.

        Args:
            results: Complete experiment results
            experiment_config: Configuration used for the experiment
            model_name: Name of the model
            step: Step number for metrics
        """
        try:
            # Log experiment configuration
            self.log_parameters(experiment_config)

            # Log basic metrics
            if 'metrics' in results:
                self.log_metrics(results['metrics'], step=step)

            # Log activation analysis
            if 'activations' in results:
                self.log_activation_analysis(
                    results['activations'],
                    model_name,
                    experiment_config,
                    step=step
                )

            # Log attention analysis
            if 'attention_weights' in results:
                self.log_attention_analysis(
                    results['attention_weights'],
                    model_name,
                    results.get('input_tokens'),
                    step=step
                )

            # Log layer statistics
            if 'layer_statistics' in results:
                layer_metrics = {}
                for layer, stats in results['layer_statistics'].items():
                    for stat_name, value in stats.items():
                        if isinstance(value, (int, float)):
                            layer_metrics[f"layer_{layer}_{stat_name}"] = value

                self.log_metrics(layer_metrics, step=step)

            # Log interpretability results
            if 'interpretability' in results:
                interp_metrics = {}
                for method, method_results in results['interpretability'].items():
                    if isinstance(method_results, dict):
                        for metric, value in method_results.items():
                            if isinstance(value, (int, float)):
                                interp_metrics[f"interpretability_{method}_{metric}"] = value

                self.log_metrics(interp_metrics, step=step)

            # Save complete results as artifact
            self._save_results_artifact(results)

            logger.info("Logged comprehensive experiment results")

        except Exception as e:
            logger.error(f"Failed to log experiment results: {e}")

    def log_model_comparison(
        self,
        comparison_results: Dict[str, Dict[str, Any]],
        comparison_name: str
    ):
        """
        Log model comparison results.

        Args:
            comparison_results: Results from multiple models
            comparison_name: Name for the comparison
        """
        try:
            # Create comparison metrics
            comparison_metrics = {}

            for model_name, results in comparison_results.items():
                if 'metrics' in results:
                    for metric_name, value in results['metrics'].items():
                        if isinstance(value, (int, float)):
                            comparison_metrics[f"{comparison_name}_{model_name}_{metric_name}"] = value

            # Log comparison metrics
            self.log_metrics(comparison_metrics)

            # Save comparison results as artifact
            comparison_artifact_path = tempfile.mktemp(suffix='_comparison.json')

            # Make results JSON serializable
            serializable_results = self._make_json_serializable(comparison_results)

            with open(comparison_artifact_path, 'w') as f:
                json.dump({
                    'comparison_name': comparison_name,
                    'results': serializable_results,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)

            mlflow.log_artifact(comparison_artifact_path, f"comparisons/{comparison_name}")

            # Create comparison visualization if possible
            try:
                viz_path = self._create_comparison_visualization(
                    comparison_results, comparison_name
                )
                if viz_path:
                    mlflow.log_artifact(viz_path, f"comparisons/{comparison_name}/visualizations")
            except Exception as e:
                logger.debug(f"Could not create comparison visualization: {e}")

            logger.info(f"Logged model comparison: {comparison_name}")

        except Exception as e:
            logger.error(f"Failed to log model comparison: {e}")

    def _flatten_params(self, params: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested parameter dictionary."""
        flat_params = {}

        for key, value in params.items():
            full_key = f"{prefix}{key}" if prefix else key

            if isinstance(value, dict):
                flat_params.update(self._flatten_params(value, f"{full_key}_"))
            elif isinstance(value, (list, tuple)):
                # Convert lists/tuples to strings
                flat_params[full_key] = str(value)
            else:
                flat_params[full_key] = value

        return flat_params

    def _make_json_serializable(self, obj):
        """Convert object to JSON serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'detach'):  # PyTorch tensor
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

    def _save_results_artifact(self, results: Dict[str, Any]):
        """Save complete results as MLflow artifact."""
        try:
            results_path = tempfile.mktemp(suffix='_results.json')

            # Make results JSON serializable
            serializable_results = self._make_json_serializable(results)

            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)

            mlflow.log_artifact(results_path, "analysis_results")

        except Exception as e:
            logger.debug(f"Could not save results artifact: {e}")

    def _create_attention_visualization(
        self,
        attention: np.ndarray,
        tokens: Optional[List[str]],
        title: str
    ) -> Optional[str]:
        """Create attention visualization."""
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(attention, cmap='Blues', aspect='auto')

            if tokens:
                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45)
                ax.set_yticklabels(tokens)

            ax.set_title(title)
            plt.colorbar(im)
            plt.tight_layout()

            viz_path = tempfile.mktemp(suffix='_attention_viz.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            return viz_path

        except ImportError:
            return None

    def _create_comparison_visualization(
        self,
        comparison_results: Dict[str, Dict[str, Any]],
        comparison_name: str
    ) -> Optional[str]:
        """Create comparison visualization."""
        try:
            import matplotlib.pyplot as plt

            model_names = list(comparison_results.keys())

            # Extract common metrics
            all_metrics = set()
            for results in comparison_results.values():
                if 'metrics' in results:
                    all_metrics.update(results['metrics'].keys())

            if not all_metrics:
                return None

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()

            for i, metric in enumerate(list(all_metrics)[:4]):  # Show up to 4 metrics
                if i >= len(axes):
                    break

                values = []
                valid_models = []

                for model_name, results in comparison_results.items():
                    if 'metrics' in results and metric in results['metrics']:
                        values.append(results['metrics'][metric])
                        valid_models.append(model_name)

                if values:
                    axes[i].bar(valid_models, values)
                    axes[i].set_title(f"{metric}")
                    axes[i].tick_params(axis='x', rotation=45)

            plt.suptitle(f"Model Comparison: {comparison_name}")
            plt.tight_layout()

            viz_path = tempfile.mktemp(suffix='_comparison_viz.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            return viz_path

        except ImportError:
            return None

    def create_model_registry_entry(
        self,
        model: Any,
        model_name: str,
        model_version: str = "1",
        description: Optional[str] = None
    ):
        """
        Register model in MLflow Model Registry.

        Args:
            model: Model to register
            model_name: Name for the registered model
            model_version: Version of the model
            description: Description of the model
        """
        try:
            # Log the model first
            mlflow.pytorch.log_model(
                model,
                f"models/{model_name}",
                registered_model_name=model_name
            )

            # Add model description if provided
            if description:
                client = mlflow.tracking.MlflowClient()
                client.update_registered_model(
                    name=model_name,
                    description=description
                )

            logger.info(f"Registered model: {model_name}")

        except Exception as e:
            logger.error(f"Failed to register model: {e}")

    def end_run(self):
        """End the current MLflow run."""
        try:
            mlflow.end_run()
            logger.info("MLflow run ended successfully")
        except Exception as e:
            logger.error(f"Error ending MLflow run: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()

# Utility functions
def create_mlflow_config(
    experiment_name: str = "neuronmap-analysis",
    tracking_uri: Optional[str] = None,
    artifact_location: Optional[str] = None
) -> Dict[str, Any]:
    """Create configuration for MLflow integration."""
    return {
        'experiment_name': experiment_name,
        'tracking_uri': tracking_uri,
        'artifact_location': artifact_location
    }

def quick_log_to_mlflow(
    results: Dict[str, Any],
    experiment_name: str = "neuronmap-analysis",
    run_name: Optional[str] = None
) -> bool:
    """
    Quick function to log NeuronMap results to MLflow.

    Args:
        results: Results from NeuronMap analysis
        experiment_name: MLflow experiment name
        run_name: Run name

    Returns:
        True if logging successful
    """
    try:
        with MLFlowIntegration(
            experiment_name=experiment_name,
            run_name=run_name
        ) as mlflow_integration:

            experiment_config = {
                'model_name': results.get('model_name', 'unknown'),
                'analysis_type': results.get('analysis_type', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }

            mlflow_integration.log_experiment_results(
                results,
                experiment_config,
                results.get('model_name', 'unknown')
            )

        return True

    except Exception as e:
        logger.error(f"Failed to log to MLflow: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Example MLflow integration
    with MLFlowIntegration(
        experiment_name="neuronmap-demo",
        run_name="example_analysis"
    ) as mlflow_integration:

        # Mock data for demonstration
        mock_results = {
            'model_name': 'bert-base-uncased',
            'analysis_type': 'sentiment',
            'activations': {
                0: np.random.randn(32, 768),
                6: np.random.randn(32, 768),
                11: np.random.randn(32, 768)
            },
            'attention_weights': {
                6: np.random.rand(12, 32, 32),
                11: np.random.rand(12, 32, 32)
            },
            'metrics': {
                'accuracy': 0.95,
                'interpretability_score': 0.78,
                'processing_time': 45.2
            }
        }

        experiment_config = {
            'model_name': 'bert-base-uncased',
            'dataset': 'sentiment_samples',
            'layers_analyzed': [0, 6, 11],
            'analysis_method': 'basic'
        }

        # Log example results
        mlflow_integration.log_experiment_results(
            mock_results,
            experiment_config,
            'bert-base-uncased'
        )

        print("Example data logged to MLflow")
        print("View MLflow UI with: mlflow ui")
