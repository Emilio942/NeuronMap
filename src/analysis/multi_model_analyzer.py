"""
Multi-Model Support for NeuronMap
================================

This module enables comparative analysis across multiple neural network models.
"""

import logging
import torch
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .activation_analyzer import ActivationAnalyzer

logger = logging.getLogger(__name__)


class MultiModelAnalyzer:
    """Analyzes activations across multiple models for comparative studies."""

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.analyzers = {}

    def add_model(self, model_name: str, model_config: Optional[Dict] = None) -> bool:
        """Add a model to the multi-model analysis."""
        try:
            # Create a copy of config for this model
            model_specific_config = self._create_model_config(model_name, model_config)

            # Initialize analyzer for this model
            analyzer = ActivationAnalyzer(model_specific_config)
            self.analyzers[model_name] = analyzer

            logger.info(f"Successfully added model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to add model {model_name}: {e}")
            return False

    def _create_model_config(
            self,
            model_name: str,
            model_config: Optional[Dict] = None):
        """Create model-specific configuration."""
        from utils.config_manager import NeuronMapConfig

        # Start with base config
        config_dict = self.config.model_dump()

        # Update model name
        config_dict['model']['name'] = model_name

        # Apply model-specific overrides
        if model_config:
            for key, value in model_config.items():
                if key in config_dict['model']:
                    config_dict['model'][key] = value

        # Create new config object
        return NeuronMapConfig(**config_dict)

    def list_supported_models(self) -> List[str]:
        """List commonly supported models for analysis."""
        return [
            # GPT Models
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "distilgpt2",

            # BERT Models
            "bert-base-uncased",
            "bert-base-cased",
            "bert-large-uncased",
            "distilbert-base-uncased",

            # RoBERTa Models
            "roberta-base",
            "roberta-large",
            "distilroberta-base",

            # T5 Models
            "t5-small",
            "t5-base",

            # Other Models
            "albert-base-v2",
            "electra-base-discriminator",
        ]

    def analyze_single_model(self, model_name: str,
                             questions: List[str]) -> Dict[str, Any]:
        """Analyze questions with a single model."""
        if model_name not in self.analyzers:
            logger.error(f"Model {model_name} not initialized")
            return {}

        analyzer = self.analyzers[model_name]

        start_time = time.time()
        results = analyzer.analyze_questions(questions)
        end_time = time.time()

        # Compute statistics
        stats = analyzer.compute_statistics(results)
        stats['analysis_time'] = end_time - start_time
        stats['model_name'] = model_name

        return {
            'model_name': model_name,
            'results': results,
            'statistics': stats
        }

    def analyze_multiple_models(self, questions: List[str],
                                model_names: Optional[List[str]] = None,
                                parallel: bool = True) -> Dict[str, Any]:
        """Analyze questions across multiple models."""
        if model_names is None:
            model_names = list(self.analyzers.keys())

        if not model_names:
            logger.error("No models available for analysis")
            return {}

        logger.info(f"Starting multi-model analysis with {len(model_names)} models")

        results = {}

        if parallel and len(model_names) > 1:
            # Parallel processing (if multiple models and enough resources)
            with ThreadPoolExecutor(max_workers=min(len(model_names), 3)) as executor:
                future_to_model = {
                    executor.submit(self.analyze_single_model, model_name, questions): model_name
                    for model_name in model_names
                }

                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        model_results = future.result()
                        results[model_name] = model_results
                        logger.info(f"Completed analysis for {model_name}")
                    except Exception as e:
                        logger.error(f"Analysis failed for {model_name}: {e}")
        else:
            # Sequential processing
            for model_name in model_names:
                try:
                    model_results = self.analyze_single_model(model_name, questions)
                    results[model_name] = model_results
                    logger.info(f"Completed analysis for {model_name}")
                except Exception as e:
                    logger.error(f"Analysis failed for {model_name}: {e}")

        return results

    def compare_models(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative analysis between models."""
        if len(results) < 2:
            logger.warning("Need at least 2 models for comparison")
            return {}

        comparison = {
            'model_count': len(results),
            'models': list(results.keys()),
            'comparison_metrics': {}
        }

        # Performance comparison
        performance_metrics = {}
        for model_name, model_data in results.items():
            stats = model_data.get('statistics', {})
            performance_metrics[model_name] = {
                'success_rate': stats.get('success_rate', 0.0),
                'analysis_time': stats.get('analysis_time', 0.0),
                'total_questions': stats.get('total_questions', 0),
                'layer_count': len(stats.get('layer_statistics', {}))
            }

        comparison['performance_metrics'] = performance_metrics

        # Layer comparison (if same layers exist)
        layer_comparison = self._compare_layers(results)
        if layer_comparison:
            comparison['layer_comparison'] = layer_comparison

        # Activation similarity
        similarity_analysis = self._compute_activation_similarity(results)
        if similarity_analysis:
            comparison['similarity_analysis'] = similarity_analysis

        return comparison

    def _compare_layers(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare similar layers across models."""
        # Find common layer patterns
        all_layers = {}
        for model_name, model_data in results.items():
            stats = model_data.get('statistics', {})
            layer_stats = stats.get('layer_statistics', {})
            for layer_name in layer_stats.keys():
                if layer_name not in all_layers:
                    all_layers[layer_name] = []
                all_layers[layer_name].append(model_name)

        # Find layers present in multiple models
        common_layers = {layer: models for layer, models in all_layers.items()
                         if len(models) > 1}

        if not common_layers:
            return {}

        layer_comparison = {}
        for layer_name, model_list in common_layers.items():
            layer_stats = {}
            for model_name in model_list:
                model_layer_stats = results[model_name]['statistics']['layer_statistics'][layer_name]
                layer_stats[model_name] = {
                    'mean': model_layer_stats.get('mean', 0.0),
                    'std': model_layer_stats.get('std', 0.0),
                    'shape': model_layer_stats.get('shape', []),
                    'sparsity': model_layer_stats.get('sparsity', 0.0)
                }

            layer_comparison[layer_name] = layer_stats

        return layer_comparison

    def _compute_pairwise_similarities(self, model_activations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compute pairwise similarities between model activations."""
        similarities = {}
        model_names = list(model_activations.keys())

        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:
                    try:
                        min_samples = min(len(model_activations[model1]),
                                        len(model_activations[model2]))

                        act1 = model_activations[model1][:min_samples]
                        act2 = model_activations[model2][:min_samples]

                        similarities_per_sample = []
                        for k in range(min_samples):
                            sim = cosine_similarity([act1[k]], [act2[k]])[0][0]
                            similarities_per_sample.append(sim)

                        avg_similarity = np.mean(similarities_per_sample)
                        similarities[f"{model1}_vs_{model2}"] = {
                            'cosine_similarity': float(avg_similarity),
                            'sample_count': min_samples
                        }

                    except Exception as e:
                        logger.warning(f"Failed to compute similarity between {model1} and {model2}: {e}")
        return similarities

    def _compute_activation_similarity(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute similarity between model activations."""
        try:
            from sklearn.metrics.pairwise import cosine_similarity

            model_activations = {}

            for model_name, model_data in results.items():
                model_results = model_data.get('results', [])
                if not model_results:
                    continue

                activations = []
                for result in model_results:
                    if result.get('success', False):
                        result_activations = result.get('activations', {})
                        if result_activations:
                            first_layer = list(result_activations.keys())[0]
                            activation = result_activations[first_layer]
                            if isinstance(activation, np.ndarray):
                                activations.append(activation.flatten())

                if activations:
                    model_activations[model_name] = np.array(activations)

            if len(model_activations) < 2:
                return {}

            return self._compute_pairwise_similarities(model_activations)

        except ImportError:
            logger.warning("Scikit-learn not available for similarity computation")
            return {}
        except Exception as e:
            logger.error(f"Failed to compute activation similarity: {e}")
            return {}

    def save_multi_model_results(self, results: Dict[str, Any],
                                 comparison: Dict[str, Any]) -> None:
        """Save multi-model analysis results."""
        output_dir = Path(self.config.data.outputs_dir) / "multi_model_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save individual model results
        for model_name, model_data in results.items():
            model_file = output_dir / f"{model_name}_{timestamp}.json"
            with open(model_file, 'w', encoding='utf-8') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_data = self._make_json_serializable(model_data)
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)

        # Save comparison results
        comparison_file = output_dir / f"comparison_{timestamp}.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            serializable_comparison = self._make_json_serializable(comparison)
            json.dump(serializable_comparison, f, indent=2, ensure_ascii=False)

        # Save summary report
        summary_file = output_dir / f"summary_{timestamp}.txt"
        self._generate_summary_report(results, comparison, summary_file)

        logger.info(f"Multi-model results saved to {output_dir}")

    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible types."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(
                value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

    def _generate_summary_report(self, results: Dict[str, Any],
                                 comparison: Dict[str, Any],
                                 output_file: Path) -> None:
        """Generate a text summary of multi-model analysis."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Multi-Model Analysis Summary\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models Analyzed: {len(results)}\n\n")

            # Model performance summary
            f.write("Model Performance:\n")
            f.write("-" * 20 + "\n")

            for model_name, model_data in results.items():
                stats = model_data.get('statistics', {})
                f.write(f"\n{model_name}:\n")
                f.write(f"  Success Rate: {stats.get('success_rate', 0):.2%}\n")
                f.write(f"  Analysis Time: {stats.get('analysis_time', 0):.2f}s\n")
                f.write(f"  Questions: {stats.get('total_questions', 0)}\n")
                f.write(f"  Layers: {len(stats.get('layer_statistics', {}))}\n")

            # Similarity analysis
            if 'similarity_analysis' in comparison:
                f.write(f"\n\nModel Similarity Analysis:\n")
                f.write("-" * 30 + "\n")

                for pair, similarity_data in comparison['similarity_analysis'].items():
                    cosine_sim = similarity_data.get('cosine_similarity', 0)
                    f.write(f"{pair}: {cosine_sim:.3f}\n")

        logger.info(f"Summary report saved to {output_file}")


def create_multi_model_demo():
    """Create a demonstration of multi-model analysis."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from utils.config_manager import get_config
    from data_processing.question_loader import QuestionLoader

    # Load configuration
    config = get_config()

    # Initialize multi-model analyzer
    analyzer = MultiModelAnalyzer(config)

    # Add some lightweight models for comparison
    models_to_test = ["distilgpt2", "distilbert-base-uncased"]

    print("Adding models to multi-model analyzer...")
    for model_name in models_to_test:
        success = analyzer.add_model(model_name)
        print(f"  {model_name}: {'✅' if success else '❌'}")

    # Load questions
    question_loader = QuestionLoader(config)
    questions = question_loader.load_questions()

    if not questions:
        print("No questions found. Creating sample questions...")
        question_loader.save_sample_questions()
        questions = question_loader.load_questions()

    # Limit to first 3 questions for demo
    demo_questions = questions[:3]

    print(
        f"\nAnalyzing {
            len(demo_questions)} questions across {
            len(models_to_test)} models...")

    # Run multi-model analysis
    results = analyzer.analyze_multiple_models(demo_questions)

    # Generate comparison
    comparison = analyzer.compare_models(results)

    # Save results
    analyzer.save_multi_model_results(results, comparison)

    print("Multi-model analysis completed!")
    return results, comparison


if __name__ == "__main__":
    create_multi_model_demo()
