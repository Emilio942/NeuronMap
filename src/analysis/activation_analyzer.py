"""
Activation Analyzer Module for NeuronMap
=======================================

This module handles the extraction and analysis of neural network activations.
"""

import torch
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
from .universal_model_adapter import UniversalModelAdapter

logger = logging.getLogger(__name__)


class ActivationAnalyzer:
    """Analyzes neural network activations for given inputs."""

    def __init__(self, config):
        self.config = config
        self.model_adapter = None
        self.device = self._get_device()
        self.activation_cache = {}
        self.universal_adapter = UniversalModelAdapter(config)

        # Load model and tokenizer
        self._load_model()

    def _get_device(self) -> torch.device:
        """Determine the appropriate device to use."""
        device_config = self.config.model.device

        if device_config == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_config)

        logger.info(f"Using device: {device}")
        return device

    def _load_model(self):
        """Load the model and tokenizer using universal adapter."""
        model_name = self.config.model.name
        logger.info(f"Loading model: {model_name}")

        try:
            self.model_adapter = self.universal_adapter.load_model(model_name)
            logger.info(f"Successfully loaded model {model_name} using universal adapter")

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def get_model_layers(self) -> List[str]:
        """Get all available layer names in the model."""
        if self.model_adapter is None:
            raise RuntimeError("Model not loaded")

        return self.model_adapter.get_layer_names()

    def find_target_layers(self) -> List[Tuple[str, torch.nn.Module]]:
        """Find target layers specified in configuration."""
        if self.model_adapter is None:
            raise RuntimeError("Model not loaded")

        # Try to get layers from model configuration
        model_config = self.universal_adapter._find_model_config(self.config.model.name)
        if model_config and 'layers' in model_config:
            layer_config = model_config['layers']
            found_layers = self.model_adapter.get_target_layers(layer_config)
        else:
            # Fallback to target_layers from config
            target_layer_names = self.config.model.target_layers
            found_layers = []

            for target_name in target_layer_names:
                found = False
                for name, module in self.model_adapter.model.named_modules():
                    if name == target_name:
                        found_layers.append((name, module))
                        found = True
                        break

                if not found:
                    logger.warning(f"Target layer '{target_name}' not found in model")

        if not found_layers:
            logger.error("No target layers found!")
            available_layers = self.get_model_layers()[:10]  # Show first 10
            logger.error(f"Available layers (first 10): {available_layers}")
            raise ValueError("No valid target layers specified")

        logger.info(f"Found {len(found_layers)} target layers")
        return found_layers

    def _create_hook_fn(self, layer_name: str):
        """Create a hook function for capturing activations."""
        def hook_fn(module, input_data, output_data):
            # Handle different output types
            if isinstance(output_data, torch.Tensor):
                output_tensor = output_data
            elif isinstance(output_data, tuple) and len(output_data) > 0:
                output_tensor = output_data[0]
            else:
                logger.warning(f"Unexpected output type from layer {layer_name}: {type(output_data)}")
                return

            # Detach and move to CPU
            detached_output = output_tensor.detach().cpu()

            # Apply aggregation method
            aggregation_method = self.config.analysis.aggregation_method
            if detached_output.ndim >= 2:
                if detached_output.shape[0] == 1:  # Remove batch dimension
                    detached_output = detached_output[0]

                # Aggregate over sequence dimension
                if aggregation_method == "mean":
                    aggregated = detached_output.mean(dim=0)
                elif aggregation_method == "max":
                    aggregated = detached_output.max(dim=0)[0]
                elif aggregation_method == "sum":
                    aggregated = detached_output.sum(dim=0)
                elif aggregation_method == "first":
                    aggregated = detached_output[0]
                elif aggregation_method == "last":
                    aggregated = detached_output[-1]
                else:
                    aggregated = detached_output.mean(dim=0)  # Default to mean

                self.activation_cache[layer_name] = aggregated
            else:
                self.activation_cache[layer_name] = detached_output

        return hook_fn

    def extract_activations(self, text: str) -> Dict[str, np.ndarray]:
        """Extract activations for a single text input."""
        if self.model_adapter is None:
            raise RuntimeError("Model adapter not loaded")

        # Clear activation cache
        self.activation_cache.clear()

        # Find target layers
        target_layers = self.find_target_layers()

        # Register hooks
        hooks = []
        for layer_name, layer_module in target_layers:
            hook_fn = self._create_hook_fn(layer_name)
            hook_handle = layer_module.register_forward_hook(hook_fn)
            hooks.append(hook_handle)

        try:
            # Prepare inputs using model adapter
            inputs = self.model_adapter.prepare_inputs([text])

            # Forward pass
            with torch.no_grad():
                self.model_adapter.model(**inputs)

            # Convert cached activations to numpy
            activations = {}
            for layer_name in [name for name, _ in target_layers]:
                if layer_name in self.activation_cache:
                    activations[layer_name] = self.activation_cache[layer_name].numpy()
                else:
                    logger.warning(f"No activation captured for layer {layer_name}")

            return activations

        except Exception as e:
            logger.error(f"Failed to extract activations for text: {text[:50]}... Error: {e}")
            return {}

        finally:
            # Always remove hooks
            for hook_handle in hooks:
                hook_handle.remove()

    def analyze_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Analyze a list of questions and extract activations."""
        results = []

        logger.info(f"Analyzing {len(questions)} questions...")

        # Process questions with progress bar
        for question in tqdm(questions, desc="Extracting activations", unit="question"):
            try:
                activations = self.extract_activations(question)

                result = {
                    'question': question,
                    'activations': activations,
                    'success': len(activations) > 0
                }

                # Add metadata
                if activations:
                    for layer_name, activation in activations.items():
                        result[f'{layer_name}_shape'] = activation.shape
                        result[f'{layer_name}_mean'] = float(activation.mean())
                        result[f'{layer_name}_std'] = float(activation.std())

                results.append(result)

            except Exception as e:
                logger.error(f"Failed to analyze question: {question[:50]}... Error: {e}")
                results.append({
                    'question': question,
                    'activations': {},
                    'success': False,
                    'error': str(e)
                })

        successful_analyses = sum(1 for r in results if r['success'])
        logger.info(f"Successfully analyzed {successful_analyses}/{len(questions)} questions")

        return results

    def save_results(self, results: List[Dict[str, Any]]) -> None:
        """Save analysis results to file."""
        output_file = Path(self.config.data.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for saving
        save_data = []

        for result in results:
            row = {
                'question': result['question'],
                'success': result['success']
            }

            # Add error if present
            if 'error' in result:
                row['error'] = result['error']

            # Add activation data
            for layer_name, activation in result.get('activations', {}).items():
                if isinstance(activation, np.ndarray):
                    # Convert to list for JSON serialization
                    row[f'{layer_name}_activation'] = activation.tolist()
                    row[f'{layer_name}_shape'] = activation.shape
                    row[f'{layer_name}_mean'] = float(activation.mean())
                    row[f'{layer_name}_std'] = float(activation.std())

            save_data.append(row)

        # Save as CSV
        if output_file.suffix.lower() == '.csv':
            df = pd.DataFrame(save_data)
            df.to_csv(output_file, index=False)
        else:
            # Save as JSON
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_file}")

    def compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics on the analysis results."""
        stats = {
            'total_questions': len(results),
            'successful_analyses': sum(1 for r in results if r['success']),
            'failed_analyses': sum(1 for r in results if not r['success']),
            'success_rate': 0.0
        }

        if stats['total_questions'] > 0:
            stats['success_rate'] = stats['successful_analyses'] / stats['total_questions']

        # Layer-specific statistics
        layer_stats = {}
        successful_results = [r for r in results if r['success']]

        if successful_results:
            # Get all layer names
            all_layers = set()
            for result in successful_results:
                all_layers.update(result.get('activations', {}).keys())

            for layer_name in all_layers:
                activations = []
                for result in successful_results:
                    if layer_name in result.get('activations', {}):
                        activation = result['activations'][layer_name]
                        if isinstance(activation, np.ndarray):
                            activations.append(activation)

                if activations:
                    all_activations = np.stack(activations)
                    layer_stats[layer_name] = {
                        'count': len(activations),
                        'shape': activations[0].shape,
                        'mean': float(all_activations.mean()),
                        'std': float(all_activations.std()),
                        'min': float(all_activations.min()),
                        'max': float(all_activations.max())
                    }

        stats['layer_statistics'] = layer_stats
        return stats

    def _find_highly_correlated_pairs(
        self,
        correlation_matrix: np.ndarray,
        layer_names: List[str],
        threshold: float = 0.8,
    ) -> List[Dict[str, Any]]:
        """Identify pairs of layers whose correlation exceeds ``threshold``.

        The method mirrors the helper available in :mod:`advanced_analysis` so
        that the multi-model test-suite can exercise this analyser in
        isolation.
        """

        if correlation_matrix.ndim != 2:
            raise ValueError("correlation_matrix must be 2-dimensional")

        if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
            raise ValueError("correlation_matrix must be square")

        if len(layer_names) != correlation_matrix.shape[0]:
            raise ValueError("layer_names length must match correlation_matrix dimensions")

        pairs: List[Dict[str, Any]] = []
        for i in range(len(layer_names)):
            for j in range(i + 1, len(layer_names)):
                value = float(correlation_matrix[i, j])
                if abs(value) >= threshold:
                    pairs.append({
                        "layer1": layer_names[i],
                        "layer2": layer_names[j],
                        "correlation": value,
                    })

        return pairs


def main():
    """Test the activation analyzer."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from utils.config_manager import get_config

    # Load configuration
    config = get_config()

    # Test with sample questions
    sample_questions = [
        "What is the capital of France?",
        "How does machine learning work?",
        "Explain quantum physics."
    ]

    # Initialize analyzer
    analyzer = ActivationAnalyzer(config)

    # List available layers
    print("Available layers:")
    layers = analyzer.get_model_layers()
    for i, layer in enumerate(layers[:10]):  # Show first 10
        print(f"  {i+1}. {layer}")

    # Run analysis
    results = analyzer.analyze_questions(sample_questions)

    # Print statistics
    stats = analyzer.compute_statistics(results)
    print(f"\nAnalysis Statistics:")
    print(f"  Total questions: {stats['total_questions']}")
    print(f"  Successful: {stats['successful_analyses']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")


if __name__ == "__main__":
    main()
