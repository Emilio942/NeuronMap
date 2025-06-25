"""
Weights & Biases Integration for NeuronMap

Provides seamless integration with wandb for experiment tracking and visualization.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import tempfile

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

# Internal imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.error_handling import NeuronMapError
from utils.monitoring import setup_monitoring

logger = logging.getLogger(__name__)

class WandBIntegration:
    """
    Integration with Weights & Biases for experiment tracking and visualization.

    Provides functionality to:
    - Track NeuronMap experiments and metrics
    - Log activation patterns and model architectures
    - Create interactive visualizations
    - Compare models and experiments
    - Share results with team
    """

    def __init__(
        self,
        project_name: str = "neuronmap-analysis",
        experiment_name: Optional[str] = None,
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Initialize Weights & Biases integration.

        Args:
            project_name: W&B project name
            experiment_name: Name of the experiment run
            entity: W&B entity (username or team)
            config: Experiment configuration
            tags: Tags for the experiment
        """
        if not WANDB_AVAILABLE:
            raise ImportError("wandb integration requires wandb package: pip install wandb")

        self.project_name = project_name
        self.experiment_name = experiment_name or f"neuronmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.entity = entity
        self.config = config or {}
        self.tags = tags or []

        # Initialize wandb run
        self.run = wandb.init(
            project=project_name,
            name=self.experiment_name,
            entity=entity,
            config=self.config,
            tags=self.tags
        )

        # Monitoring
        self.monitor = setup_monitoring()

        # Tracking variables
        self.step_counter = 0
        self.logged_models = set()

        logger.info(f"W&B integration initialized: {self.run.name}")

    def log_model_architecture(
        self,
        model: Any,
        model_name: str,
        input_sample: Optional[Any] = None
    ) -> bool:
        """
        Log model architecture to W&B.

        Args:
            model: Model to log
            model_name: Name of the model
            input_sample: Sample input for the model

        Returns:
            True if successful, False otherwise
        """
        try:
            if model_name in self.logged_models:
                logger.debug(f"Model {model_name} already logged")
                return True

            # Create model summary
            model_info = {
                'model_name': model_name,
                'model_type': type(model).__name__,
            }

            # Try to get parameter count
            try:
                if hasattr(model, 'parameters'):
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    model_info.update({
                        'total_parameters': total_params,
                        'trainable_parameters': trainable_params
                    })
            except Exception:
                pass

            # Log model info
            wandb.config.update({f"model_{model_name}": model_info})

            # Try to log model graph (if torch model)
            try:
                if input_sample is not None and hasattr(model, 'forward'):
                    # Create a temporary onnx file for visualization
                    import torch
                    temp_path = tempfile.mktemp(suffix='.onnx')
                    torch.onnx.export(
                        model, input_sample, temp_path,
                        input_names=['input'], output_names=['output'],
                        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                    )
                    wandb.save(temp_path)
            except Exception as e:
                logger.debug(f"Could not log model graph: {e}")

            self.logged_models.add(model_name)
            return True

        except Exception as e:
            logger.error(f"Failed to log model architecture: {e}")
            return False

    def log_activation_analysis(
        self,
        activations: Dict[str, Any],
        model_name: str,
        analysis_type: str = "basic",
        step: Optional[int] = None
    ):
        """
        Log activation analysis results to W&B.

        Args:
            activations: Dictionary of layer activations
            model_name: Name of the model
            analysis_type: Type of analysis
            step: Step number for logging
        """
        if step is None:
            step = self.step_counter
            self.step_counter += 1

        try:
            metrics = {}

            for layer_idx, activation in activations.items():
                # Convert to numpy if needed
                if hasattr(activation, 'detach'):
                    activation = activation.detach().cpu().numpy()

                # Flatten for statistics
                flat_activation = activation.flatten()

                layer_prefix = f"{model_name}/{analysis_type}/layer_{layer_idx}"

                # Basic statistics
                metrics.update({
                    f"{layer_prefix}/mean": np.mean(flat_activation),
                    f"{layer_prefix}/std": np.std(flat_activation),
                    f"{layer_prefix}/max": np.max(flat_activation),
                    f"{layer_prefix}/min": np.min(flat_activation),
                    f"{layer_prefix}/sparsity": np.mean(np.abs(flat_activation) < 1e-6)
                })

                # Create histogram
                hist_data = wandb.Histogram(flat_activation)
                metrics[f"{layer_prefix}/distribution"] = hist_data

                # Create heatmap for 2D activations
                if len(activation.shape) >= 2:
                    # Take a sample if too large
                    sample_size = min(100, activation.shape[0])
                    activation_sample = activation[:sample_size, :sample_size] if len(activation.shape) == 2 else activation[0, :sample_size, :sample_size]

                    metrics[f"{layer_prefix}/heatmap"] = wandb.Image(
                        self._create_heatmap_plot(activation_sample, f"Layer {layer_idx} Activations")
                    )

            # Log all metrics
            wandb.log(metrics, step=step)

            logger.info(f"Logged activation analysis for {len(activations)} layers")

        except Exception as e:
            logger.error(f"Failed to log activation analysis: {e}")

    def log_attention_patterns(
        self,
        attention_weights: Dict[str, Any],
        model_name: str,
        input_tokens: Optional[List[str]] = None,
        step: Optional[int] = None
    ):
        """
        Log attention patterns to W&B.

        Args:
            attention_weights: Attention weight matrices
            model_name: Name of the model
            input_tokens: Input tokens for labeling
            step: Step number for logging
        """
        if step is None:
            step = self.step_counter
            self.step_counter += 1

        try:
            metrics = {}

            for layer_idx, attention in attention_weights.items():
                if hasattr(attention, 'detach'):
                    attention = attention.detach().cpu().numpy()

                layer_prefix = f"{model_name}/attention/layer_{layer_idx}"

                # Average across heads if multi-head
                if len(attention.shape) > 2:
                    attention_avg = np.mean(attention, axis=0)
                else:
                    attention_avg = attention

                # Attention statistics
                entropy = -np.sum(attention_avg * np.log(attention_avg + 1e-12))
                max_attention = np.max(attention_avg, axis=1)

                metrics.update({
                    f"{layer_prefix}/entropy": entropy,
                    f"{layer_prefix}/max_attention_mean": np.mean(max_attention),
                    f"{layer_prefix}/max_attention_std": np.std(max_attention)
                })

                # Create attention visualization
                attention_plot = self._create_attention_plot(
                    attention_avg, input_tokens, f"Layer {layer_idx} Attention"
                )
                metrics[f"{layer_prefix}/pattern"] = wandb.Image(attention_plot)

                # Log attention distribution
                metrics[f"{layer_prefix}/weight_distribution"] = wandb.Histogram(attention_avg.flatten())

            wandb.log(metrics, step=step)

            logger.info(f"Logged attention patterns for {len(attention_weights)} layers")

        except Exception as e:
            logger.error(f"Failed to log attention patterns: {e}")

    def log_experiment_results(
        self,
        results: Dict[str, Any],
        experiment_config: Dict[str, Any],
        step: Optional[int] = None
    ):
        """
        Log comprehensive experiment results.

        Args:
            results: Experiment results dictionary
            experiment_config: Configuration used for the experiment
            step: Step number for logging
        """
        if step is None:
            step = self.step_counter
            self.step_counter += 1

        try:
            # Update run config with experiment config
            wandb.config.update(experiment_config)

            # Log different types of results
            metrics = {}

            # Basic metrics
            if 'metrics' in results:
                for metric_name, value in results['metrics'].items():
                    if isinstance(value, (int, float)):
                        metrics[f"experiment/{metric_name}"] = value

            # Activation analysis
            if 'activations' in results:
                self.log_activation_analysis(
                    results['activations'],
                    experiment_config.get('model_name', 'unknown'),
                    'experiment',
                    step
                )

            # Attention analysis
            if 'attention_weights' in results:
                self.log_attention_patterns(
                    results['attention_weights'],
                    experiment_config.get('model_name', 'unknown'),
                    results.get('input_tokens'),
                    step
                )

            # Layer statistics
            if 'layer_statistics' in results:
                for layer, stats in results['layer_statistics'].items():
                    for stat_name, value in stats.items():
                        if isinstance(value, (int, float)):
                            metrics[f"layers/layer_{layer}/{stat_name}"] = value

            # Interpretability results
            if 'interpretability' in results:
                interp_results = results['interpretability']
                for method, method_results in interp_results.items():
                    if isinstance(method_results, dict):
                        for metric, value in method_results.items():
                            if isinstance(value, (int, float)):
                                metrics[f"interpretability/{method}/{metric}"] = value

            # Log all metrics
            if metrics:
                wandb.log(metrics, step=step)

            # Log analysis artifacts
            self._log_analysis_artifacts(results)

            logger.info("Logged comprehensive experiment results")

        except Exception as e:
            logger.error(f"Failed to log experiment results: {e}")

    def log_model_comparison(
        self,
        comparison_results: Dict[str, Dict[str, Any]],
        comparison_name: str,
        step: Optional[int] = None
    ):
        """
        Log model comparison results.

        Args:
            comparison_results: Results from multiple models
            comparison_name: Name for the comparison
            step: Step number for logging
        """
        if step is None:
            step = self.step_counter
            self.step_counter += 1

        try:
            # Create comparison table
            comparison_data = []

            for model_name, results in comparison_results.items():
                row = {'model': model_name}

                # Extract metrics for comparison
                if 'metrics' in results:
                    row.update(results['metrics'])

                if 'layer_statistics' in results:
                    # Add summary statistics
                    layer_stats = results['layer_statistics']
                    if layer_stats:
                        avg_activation = np.mean([
                            stats.get('mean', 0) for stats in layer_stats.values()
                        ])
                        row['avg_activation'] = avg_activation

                comparison_data.append(row)

            # Log comparison table
            comparison_table = wandb.Table(
                columns=list(comparison_data[0].keys()) if comparison_data else ['model'],
                data=[list(row.values()) for row in comparison_data]
            )

            wandb.log({
                f"comparisons/{comparison_name}/table": comparison_table,
                f"comparisons/{comparison_name}/num_models": len(comparison_results)
            }, step=step)

            # Create comparison plots
            self._create_comparison_plots(comparison_results, comparison_name, step)

            logger.info(f"Logged model comparison: {comparison_name}")

        except Exception as e:
            logger.error(f"Failed to log model comparison: {e}")

    def _create_heatmap_plot(self, data: np.ndarray, title: str):
        """Create a heatmap plot."""
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(data, cmap='viridis', aspect='auto')
            ax.set_title(title)
            plt.colorbar(im)
            plt.tight_layout()

            return fig

        except ImportError:
            # Fallback to simple array representation
            return data

    def _create_attention_plot(
        self,
        attention: np.ndarray,
        tokens: Optional[List[str]],
        title: str
    ):
        """Create attention pattern visualization."""
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

            return fig

        except ImportError:
            return attention

    def _log_analysis_artifacts(self, results: Dict[str, Any]):
        """Log analysis artifacts as W&B artifacts."""
        try:
            # Create artifact for storing detailed results
            artifact = wandb.Artifact(
                name=f"analysis_results_{self.run.id}",
                type="analysis_results",
                description="Detailed NeuronMap analysis results"
            )

            # Save results as JSON
            temp_path = tempfile.mktemp(suffix='.json')

            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(results)

            with open(temp_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)

            artifact.add_file(temp_path, name="results.json")
            wandb.log_artifact(artifact)

        except Exception as e:
            logger.debug(f"Could not log artifacts: {e}")

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
        else:
            return obj

    def _create_comparison_plots(
        self,
        comparison_results: Dict[str, Dict[str, Any]],
        comparison_name: str,
        step: int
    ):
        """Create comparison visualization plots."""
        try:
            import matplotlib.pyplot as plt

            model_names = list(comparison_results.keys())

            # Extract common metrics
            all_metrics = set()
            for results in comparison_results.values():
                if 'metrics' in results:
                    all_metrics.update(results['metrics'].keys())

            for metric in all_metrics:
                values = []
                valid_models = []

                for model_name, results in comparison_results.items():
                    if 'metrics' in results and metric in results['metrics']:
                        values.append(results['metrics'][metric])
                        valid_models.append(model_name)

                if values:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(valid_models, values)
                    ax.set_title(f"{comparison_name}: {metric}")
                    ax.set_ylabel(metric)
                    plt.xticks(rotation=45)

                    # Add value labels
                    for bar, value in zip(bars, values):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{value:.3f}', ha='center', va='bottom')

                    plt.tight_layout()

                    wandb.log({
                        f"comparisons/{comparison_name}/plots/{metric}": wandb.Image(fig)
                    }, step=step)

                    plt.close(fig)

        except Exception as e:
            logger.debug(f"Could not create comparison plots: {e}")

    def create_report(
        self,
        title: str,
        description: str,
        results: Dict[str, Any]
    ) -> str:
        """
        Create a W&B report for sharing results.

        Args:
            title: Report title
            description: Report description
            results: Results to include in report

        Returns:
            URL of the created report
        """
        try:
            # This would create a W&B report programmatically
            # For now, we'll log the report data and provide instructions

            report_data = {
                'title': title,
                'description': description,
                'results_summary': self._create_results_summary(results),
                'run_id': self.run.id,
                'project': self.project_name
            }

            wandb.log({"report_data": report_data})

            # Return URL to the run (manual report creation needed)
            return f"https://wandb.ai/{self.entity}/{self.project_name}/runs/{self.run.id}"

        except Exception as e:
            logger.error(f"Failed to create report: {e}")
            return ""

    def _create_results_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of results for reporting."""
        summary = {
            'num_layers_analyzed': 0,
            'analysis_types': [],
            'key_metrics': {}
        }

        if 'activations' in results:
            summary['num_layers_analyzed'] = len(results['activations'])
            summary['analysis_types'].append('activation_analysis')

        if 'attention_weights' in results:
            summary['analysis_types'].append('attention_analysis')

        if 'metrics' in results:
            summary['key_metrics'] = results['metrics']

        return summary

    def finish(self):
        """Finish the W&B run."""
        try:
            wandb.finish()
            logger.info("W&B run finished successfully")
        except Exception as e:
            logger.error(f"Error finishing W&B run: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()

# Utility functions
def create_wandb_config(
    project_name: str = "neuronmap-analysis",
    experiment_name: Optional[str] = None,
    entity: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create configuration for W&B integration."""
    return {
        'project_name': project_name,
        'experiment_name': experiment_name,
        'entity': entity,
        'tags': tags or []
    }

def quick_log_to_wandb(
    results: Dict[str, Any],
    project_name: str = "neuronmap-analysis",
    experiment_name: Optional[str] = None
) -> bool:
    """
    Quick function to log NeuronMap results to W&B.

    Args:
        results: Results from NeuronMap analysis
        project_name: W&B project name
        experiment_name: Experiment name

    Returns:
        True if logging successful
    """
    try:
        with WandBIntegration(
            project_name=project_name,
            experiment_name=experiment_name
        ) as wandb_integration:

            experiment_config = {
                'model_name': results.get('model_name', 'unknown'),
                'analysis_type': results.get('analysis_type', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }

            wandb_integration.log_experiment_results(results, experiment_config)

        return True

    except Exception as e:
        logger.error(f"Failed to log to W&B: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Example W&B integration
    with WandBIntegration(
        project_name="neuronmap-demo",
        experiment_name="example_analysis",
        tags=["demo", "bert", "sentiment"]
    ) as wandb_integration:

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
            'layers_analyzed': [0, 6, 11]
        }

        # Log example results
        wandb_integration.log_experiment_results(mock_results, experiment_config)

        print("Example data logged to W&B")
        print(f"View at: https://wandb.ai/your-entity/neuronmap-demo")
