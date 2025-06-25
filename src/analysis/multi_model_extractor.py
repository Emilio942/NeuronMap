"""Enhanced activation extraction supporting multiple models and multi-layer analysis."""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModel,
    GPT2Model, BertModel, T5Model, LlamaForCausalLM
)
import pandas as pd
import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from tqdm import tqdm
from collections import defaultdict
import h5py

from ..utils.config import get_config
from ..utils.error_handling import with_retry, safe_execute
from ..utils.monitoring import SystemMonitor


logger = logging.getLogger(__name__)


class ModelLayerMapper:
    """Maps layer names for different model architectures."""

    def __init__(self, model_config: Dict[str, Any]):
        """Initialize with model configuration.

        Args:
            model_config: Model configuration dictionary.
        """
        self.model_config = model_config
        self.model_name = model_config["name"]
        self.model_type = model_config.get("type", "auto")

    def get_layer_patterns(self) -> Dict[str, List[str]]:
        """Get layer naming patterns for the model type.

        Returns:
            Dictionary with attention and MLP layer patterns.
        """
        config = get_config()
        layer_patterns = config.data.get("layer_patterns", {})

        if self.model_type in layer_patterns:
            return layer_patterns[self.model_type]

        # Default patterns based on model name
        if "gpt" in self.model_name.lower():
            return layer_patterns.get("gpt", {})
        elif "bert" in self.model_name.lower():
            return layer_patterns.get("bert", {})
        elif "t5" in self.model_name.lower():
            return layer_patterns.get("t5", {})
        else:
            logger.warning(f"Unknown model type for {self.model_name}, using generic patterns")
            return {"attention_patterns": [], "mlp_patterns": []}

    def get_all_layers(self, layer_range: Optional[Tuple[int, int]] = None) -> List[str]:
        """Get all layer names for the model.

        Args:
            layer_range: Optional tuple (start, end) to limit layer range.

        Returns:
            List of layer names.
        """
        total_layers = self.model_config.get("layers", {}).get("total_layers", 12)
        patterns = self.get_layer_patterns()

        # Determine layer range
        if layer_range:
            start, end = layer_range
            start = max(0, start)
            end = min(total_layers, end)
        else:
            start, end = 0, total_layers

        layer_names = []

        # Add attention layers
        for pattern in patterns.get("attention_patterns", []):
            for layer_idx in range(start, end):
                layer_names.append(pattern.format(layer=layer_idx))

        # Add MLP layers
        for pattern in patterns.get("mlp_patterns", []):
            for layer_idx in range(start, end):
                layer_names.append(pattern.format(layer=layer_idx))

        return layer_names

    def categorize_layer(self, layer_name: str) -> str:
        """Categorize a layer as attention, mlp, or other.

        Args:
            layer_name: Name of the layer.

        Returns:
            Category string: 'attention', 'mlp', or 'other'.
        """
        layer_lower = layer_name.lower()

        if any(keyword in layer_lower for keyword in ["attn", "attention", "self"]):
            return "attention"
        elif any(keyword in layer_lower for keyword in ["mlp", "dense", "fc", "intermediate"]):
            return "mlp"
        else:
            return "other"


class MultiModelActivationExtractor:
    """Enhanced activation extractor with multi-model and multi-layer support."""

    def __init__(self, config_name: str = "default"):
        """Initialize the extractor.

        Args:
            config_name: Name of experiment configuration to use.
        """
        self.config = get_config()
        self.experiment_config = self.config.get_experiment_config(config_name)
        self.extract_config = self.experiment_config["activation_extraction"]

        # Setup device
        self.device = self.config.get_device(self.extract_config["device"])
        logger.info(f"Using device: {self.device}")

        # Model components
        self.model = None
        self.tokenizer = None
        self.model_config = None
        self.layer_mapper = None

        # Activation storage
        self.activation_hooks = {}
        self.activation_storage = defaultdict(list)

    @with_retry()
    def load_model(self, model_config_name: Optional[str] = None) -> bool:
        """Load a transformer model and tokenizer.

        Args:
            model_config_name: Name of model config. If None, uses config from experiment.

        Returns:
            True if model loaded successfully.
        """
        if model_config_name is None:
            model_config_name = self.extract_config["model_config"]

        self.model_config = self.config.get_model_config(model_config_name)
        self.layer_mapper = ModelLayerMapper(self.model_config)

        model_name = self.model_config["name"]
        model_type = self.model_config.get("type", "auto")

        logger.info(f"Loading model: {model_name} (type: {model_type})")

        # Check GPU memory before loading
        if torch.cuda.is_available():
            monitor = SystemMonitor()
            memory_info = monitor.get_system_metrics()
            logger.info(f"GPU memory before loading: {memory_info.gpu_memory_percent}% used")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model based on type
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            if model_type in ["gpt", "causal"]:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )
            elif model_type == "bert":
                self.model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )
            else:
                # Auto-detect model type
                self.model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True
                )

            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)

            self.model.eval()
            logger.info("Model loaded successfully")

            # Log memory usage after loading
            if torch.cuda.is_available():
                monitor = SystemMonitor()
                memory_info = monitor.get_system_metrics()
                logger.info(f"GPU memory after loading: {memory_info.gpu_memory_percent}% used")

            return True

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise

    def discover_model_layers(self) -> Dict[str, List[str]]:
        """Discover all layers in the loaded model.

        Returns:
            Dictionary categorizing layers by type.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        discovered_layers = {
            "attention": [],
            "mlp": [],
            "other": []
        }

        logger.info("Discovering model layers...")
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') or hasattr(module, 'bias'):
                category = self.layer_mapper.categorize_layer(name)
                discovered_layers[category].append(name)

        # Log discovered layers
        for category, layers in discovered_layers.items():
            logger.info(f"Found {len(layers)} {category} layers")
            if layers and len(layers) <= 10:  # Show first few layers
                for layer in layers[:5]:
                    logger.debug(f"  {layer}")

        return discovered_layers

    def create_activation_hook(self, layer_name: str):
        """Create a hook function to capture activations from a specific layer.

        Args:
            layer_name: Name of the layer to hook.

        Returns:
            Hook function.
        """
        def hook_fn(module, input_tensor, output_tensor):
            """Hook function to capture and store activations."""
            try:
                # Handle different output types
                if isinstance(output_tensor, torch.Tensor):
                    tensor = output_tensor
                elif isinstance(output_tensor, tuple) and len(output_tensor) > 0:
                    tensor = output_tensor[0]
                else:
                    logger.warning(f"Unexpected output type from {layer_name}: {type(output_tensor)}")
                    return

                # Move to CPU and detach
                detached_tensor = tensor.detach().cpu()

                # Different aggregation strategies based on tensor shape
                if detached_tensor.ndim >= 3:  # (batch, seq_len, hidden)
                    # Aggregate over sequence dimension
                    if detached_tensor.shape[0] == 1:  # Single batch
                        aggregated = detached_tensor[0].mean(dim=0)  # Mean over sequence
                    else:
                        aggregated = detached_tensor.mean(dim=(0, 1))  # Mean over batch and sequence
                elif detached_tensor.ndim == 2:  # (batch, hidden) or (seq_len, hidden)
                    aggregated = detached_tensor.mean(dim=0)
                else:
                    aggregated = detached_tensor

                # Store the activation
                self.activation_storage[layer_name].append(aggregated.numpy())

            except Exception as e:
                logger.error(f"Error in hook for layer {layer_name}: {e}")

        return hook_fn

    def register_hooks(self, target_layers: List[str]) -> Dict[str, Any]:
        """Register hooks for multiple layers.

        Args:
            target_layers: List of layer names to hook.

        Returns:
            Dictionary of hook handles.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        hook_handles = {}

        for layer_name in target_layers:
            try:
                # Find the layer module
                module = None
                for name, mod in self.model.named_modules():
                    if name == layer_name:
                        module = mod
                        break

                if module is None:
                    logger.warning(f"Layer '{layer_name}' not found in model")
                    continue

                # Create and register hook
                hook_fn = self.create_activation_hook(layer_name)
                hook_handle = module.register_forward_hook(hook_fn)
                hook_handles[layer_name] = hook_handle

                logger.debug(f"Registered hook for layer: {layer_name}")

            except Exception as e:
                logger.error(f"Failed to register hook for layer {layer_name}: {e}")

        logger.info(f"Registered hooks for {len(hook_handles)} layers")
        return hook_handles

    def clear_activation_storage(self):
        """Clear stored activations."""
        self.activation_storage.clear()

    def process_batch(self, questions: List[str], target_layers: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of questions and extract activations.

        Args:
            questions: List of questions to process.
            target_layers: List of layer names to extract from.

        Returns:
            List of results with activations.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Register hooks
        hook_handles = self.register_hooks(target_layers)

        try:
            results = []

            for i, question in enumerate(questions):
                # Clear previous activations
                self.clear_activation_storage()

                try:
                    # Tokenize input
                    inputs = self.tokenizer(
                        question,
                        return_tensors="pt",
                        max_length=self.extract_config.get("max_length", 512),
                        truncation=True,
                        padding=True
                    )

                    # Move to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Forward pass
                    with torch.no_grad():
                        _ = self.model(**inputs)

                    # Collect activations from all layers
                    question_result = {
                        'question_id': i,
                        'question': question,
                        'model_name': self.model_config['name'],
                        'activations': {}
                    }

                    for layer_name in target_layers:
                        if layer_name in self.activation_storage and self.activation_storage[layer_name]:
                            activation = self.activation_storage[layer_name][-1]  # Get latest activation
                            question_result['activations'][layer_name] = {
                                'vector': activation.tolist(),
                                'shape': activation.shape,
                                'stats': {
                                    'mean': float(np.mean(activation)),
                                    'std': float(np.std(activation)),
                                    'min': float(np.min(activation)),
                                    'max': float(np.max(activation)),
                                    'sparsity': float(np.mean(activation == 0))
                                }
                            }

                    results.append(question_result)

                except Exception as e:
                    logger.error(f"Error processing question {i}: {e}")
                    continue

            return results

        finally:
            # Always remove hooks
            for handle in hook_handles.values():
                handle.remove()

    def save_results_hdf5(self, results: List[Dict[str, Any]], output_file: str):
        """Save results to HDF5 format for efficient storage.

        Args:
            results: List of extraction results.
            output_file: Path to output HDF5 file.
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, 'w') as f:
            # Metadata
            f.attrs['model_name'] = self.model_config['name']
            f.attrs['num_questions'] = len(results)
            f.attrs['extraction_config'] = json.dumps(self.extract_config)

            # Questions
            questions_group = f.create_group('questions')
            for i, result in enumerate(results):
                questions_group.create_dataset(f'question_{i}', data=result['question'])

            # Activations
            activations_group = f.create_group('activations')
            for i, result in enumerate(results):
                question_group = activations_group.create_group(f'question_{i}')

                for layer_name, layer_data in result['activations'].items():
                    layer_group = question_group.create_group(layer_name.replace('.', '_'))
                    layer_group.create_dataset('vector', data=layer_data['vector'])
                    layer_group.attrs['shape'] = layer_data['shape']
                    layer_group.attrs['stats'] = json.dumps(layer_data['stats'])

        logger.info(f"Results saved to HDF5: {output_path}")

    def save_results_csv(self, results: List[Dict[str, Any]], output_file: str):
        """Save results to CSV format (flattened).

        Args:
            results: List of extraction results.
            output_file: Path to output CSV file.
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Flatten results for CSV
        csv_rows = []
        for result in results:
            base_row = {
                'question_id': result['question_id'],
                'question': result['question'],
                'model_name': result['model_name']
            }

            for layer_name, layer_data in result['activations'].items():
                row = base_row.copy()
                row['layer_name'] = layer_name
                row['layer_category'] = self.layer_mapper.categorize_layer(layer_name)
                row.update(layer_data['stats'])
                row['activation_vector'] = layer_data['vector']
                csv_rows.append(row)

        df = pd.DataFrame(csv_rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to CSV: {output_path}")

    def run_multi_layer_extraction(self,
                                  questions_file: Optional[str] = None,
                                  target_layers: Optional[List[str]] = None,
                                  layer_range: Optional[Tuple[int, int]] = None,
                                  batch_size: int = 1,
                                  output_format: str = "hdf5") -> bool:
        """Run multi-layer activation extraction.

        Args:
            questions_file: Path to questions file.
            target_layers: Specific layers to extract from.
            layer_range: Range of layer indices to extract from.
            batch_size: Number of questions to process at once.
            output_format: Output format ('hdf5', 'csv', or 'both').

        Returns:
            True if extraction completed successfully.
        """
        # Load questions
        if questions_file is None:
            questions_file = self.experiment_config["question_generation"]["output_file"]

        questions = self._load_questions(questions_file)
        if not questions:
            logger.error("No questions loaded")
            return False

        # Determine target layers
        if target_layers is None:
            if layer_range:
                target_layers = self.layer_mapper.get_all_layers(layer_range)
            else:
                # Use discovered layers
                discovered = self.discover_model_layers()
                target_layers = discovered["attention"] + discovered["mlp"]

        if not target_layers:
            logger.error("No target layers specified")
            return False

        logger.info(f"Extracting from {len(target_layers)} layers")
        logger.info(f"Processing {len(questions)} questions")

        # Process in batches
        all_results = []
        failed_count = 0

        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]

            try:
                batch_results = self.process_batch(batch_questions, target_layers)
                all_results.extend(batch_results)

                logger.info(f"Processed batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")

            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                failed_count += len(batch_questions)

        # Save results
        if all_results:
            output_base = self.extract_config["output_file"].replace(".csv", "")

            if output_format in ["hdf5", "both"]:
                self.save_results_hdf5(all_results, f"{output_base}.h5")

            if output_format in ["csv", "both"]:
                self.save_results_csv(all_results, f"{output_base}.csv")

            logger.info(f"Multi-layer extraction completed!")
            logger.info(f"Total questions: {len(questions)}")
            logger.info(f"Successful extractions: {len(all_results)}")
            logger.info(f"Failed extractions: {failed_count}")

            return True
        else:
            logger.error("No activations extracted successfully")
            return False

    def _load_questions(self, filepath: str) -> List[str]:
        """Load questions from JSONL file.

        Args:
            filepath: Path to questions file.

        Returns:
            List of questions.
        """
        questions = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if 'question' in data:
                                questions.append(data['question'])
                            else:
                                logger.warning(f"Line {line_num}: No 'question' field found")
                        except json.JSONDecodeError as e:
                            logger.warning(f"Line {line_num}: Invalid JSON - {e}")
        except Exception as e:
            logger.error(f"Error loading questions: {e}")
            raise

        logger.info(f"Loaded {len(questions)} questions")
        return questions


def main():
    """Command line interface for multi-model extraction."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-model activation extraction")
    parser.add_argument("--config", default="default", help="Configuration name")
    parser.add_argument("--model", help="Model configuration name to use")
    parser.add_argument("--questions-file", help="Path to questions file")
    parser.add_argument("--layers", nargs="+", help="Specific layers to extract from")
    parser.add_argument("--layer-range", nargs=2, type=int, help="Layer range (start end)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--output-format", choices=["hdf5", "csv", "both"], default="hdf5",
                        help="Output format")
    parser.add_argument("--discover-layers", action="store_true",
                        help="Discover and list model layers")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    extractor = MultiModelActivationExtractor(args.config)

    # Load model
    extractor.load_model(args.model)

    if args.discover_layers:
        layers = extractor.discover_model_layers()
        print("\nDiscovered layers:")
        for category, layer_list in layers.items():
            print(f"\n{category.upper()} layers ({len(layer_list)}):")
            for layer in layer_list[:10]:  # Show first 10
                print(f"  {layer}")
            if len(layer_list) > 10:
                print(f"  ... and {len(layer_list) - 10} more")
        return

    # Run extraction
    layer_range = tuple(args.layer_range) if args.layer_range else None

    success = extractor.run_multi_layer_extraction(
        questions_file=args.questions_file,
        target_layers=args.layers,
        layer_range=layer_range,
        batch_size=args.batch_size,
        output_format=args.output_format
    )

    if success:
        logger.info("Multi-layer extraction completed successfully!")
    else:
        logger.error("Multi-layer extraction failed!")
        exit(1)


if __name__ == "__main__":
    main()
