"""
Model Integration for NeuronMap Intervention System
==================================================
This module provides integration between the intervention system and actual model loading,
enabling real model execution for ablation and path patching experiments.
"""
import torch
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from .universal_model_adapter import GPTAdapter, BERTAdapter, ModelAdapter
from .interventions import (
    ModifiableHookManager,
    InterventionSpec,
    InterventionType,
    run_with_ablation,
    run_with_patching,
    calculate_causal_effect
)
from .intervention_cache import InterventionCache
logger = logging.getLogger(__name__)
import yaml
from .universal_model_adapter import GPTAdapter, BERTAdapter, T5Adapter, LlamaAdapter

class ModelManager:
    """Manages model loading and integration with intervention system."""

    def __init__(self, device: Optional[torch.device] = None):
        """Initialize model manager."""
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loaded_models: Dict[str, ModelAdapter] = {}
        self.hook_managers: Dict[str, ModifiableHookManager] = {}
        self.SUPPORTED_MODELS = self._load_models_from_config()
        logger.info(f"ModelManager initialized with device: {self.device}")

    def _load_models_from_config(self) -> Dict[str, Any]:
        """Load model configurations from the YAML file."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "models.yaml"
        if not config_path.exists():
            logger.error(f"Model config file not found at {config_path}")
            return {}

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        supported_models = {}
        adapter_map = {
            "gpt": GPTAdapter,
            "bert": BERTAdapter,
            "t5": T5Adapter,
            "llama": LlamaAdapter,
        }

        for model_key, model_info in config.get('models', {}).items():
            adapter_class = adapter_map.get(model_info.get('type'))
            if adapter_class:
                supported_models[model_key] = {
                    'adapter': adapter_class,
                    'type': model_info.get('type'),
                    'name': model_info.get('name')
                }
        return supported_models
    def load_model(self, model_name: str, config: Optional[Dict[str, Any]] = None) -> ModelAdapter:
        """Load a model if not already loaded."""
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded")
            return self.loaded_models[model_name]
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Supported: {list(self.SUPPORTED_MODELS.keys())}")
        
        model_info = self.SUPPORTED_MODELS[model_name]
        adapter_class = model_info['adapter']
        model_hf_name = model_info.get('name', model_name)

        # Default configuration
        default_config = {
            'max_sequence_length': 512 if 'bert' in model_hf_name else 1024,
            'preferred_dtype': 'float32',
            'batch_size': 1
        }
        if config:
            default_config.update(config)

        logger.info(f"Loading model {model_hf_name} with {adapter_class.__name__}")
        try:
            adapter = adapter_class(model_hf_name, self.device, default_config)
            adapter.load_model()
            # Create hook manager for this model
            hook_manager = ModifiableHookManager()
            self.loaded_models[model_name] = adapter
            self.hook_managers[model_name] = hook_manager
            logger.info(f"Successfully loaded {model_hf_name}")
            return adapter
        except Exception as e:
            logger.error(f"Failed to load model {model_hf_name}: {e}")
            raise
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model."""
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")
        adapter = self.load_model(model_name)
        layer_names = adapter.get_layer_names()
        # Filter for common intervention targets
        attention_layers = [name for name in layer_names if 'attn' in name or 'attention' in name]
        mlp_layers = [name for name in layer_names if 'mlp' in name or 'feed_forward' in name]
        return {
            'model_name': model_name,
            'model_type': self.SUPPORTED_MODELS[model_name]['type'],
            'total_parameters': sum(p.numel() for p in adapter.model.parameters()),
            'device': str(self.device),
            'layer_count': len(layer_names),
            'hidden_size': adapter.model.config.hidden_size if hasattr(adapter.model.config, 'hidden_size') else None,
            'attention_layers': attention_layers[:5],  # First 5 for brevity
            'mlp_layers': mlp_layers[:5],
            'sample_layers': layer_names[:10]
        }
    def run_ablation_analysis(
        self,
        model_name: str,
        prompt: str,
        layer_name: str,
        neuron_indices: Optional[List[int]] = None,
        cache: Optional[InterventionCache] = None
    ) -> Dict[str, Any]:
        """Run actual ablation analysis on a real model."""
        logger.info(f"Running ablation analysis: {model_name}, layer: {layer_name}")
        # Load model
        adapter = self.load_model(model_name)
        hook_manager = self.hook_managers[model_name]
        # Prepare inputs
        inputs = adapter.prepare_inputs([prompt])
        # Get baseline output
        with torch.no_grad():
            baseline_outputs = adapter.model(**inputs)
        # Create intervention specification
        intervention_spec = InterventionSpec(
            layer_name=layer_name,
            intervention_type=InterventionType.ABLATION,
            target_indices=neuron_indices
        )
        # Run ablation
        try:
            # Get the actual input tensor (assuming single input)
            input_tensor = inputs['input_ids']  # For text models
            ablated_results = run_with_ablation(
                model=adapter.model,
                input_tensor=input_tensor,
                layer_name=layer_name,
                neuron_indices=neuron_indices,
                return_activations=True
            )
            # Extract outputs from results
            ablated_outputs = ablated_results['output']
            # Calculate effect for ablation (simpler than path patching)
            effect_size = self._calculate_ablation_effect(baseline_outputs, ablated_outputs)
            # Interpret outputs for text generation models
            baseline_text = self._decode_output(adapter, baseline_outputs, inputs)
            ablated_text = self._decode_output(adapter, ablated_outputs, inputs)
            results = {
                'experiment_type': 'ablation',
                'model': model_name,
                'prompt': prompt,
                'layer': layer_name,
                'neurons': neuron_indices or 'all',
                'effect_size': float(effect_size),
                'interpretation': self._interpret_effect_size(effect_size),
                'baseline_output': baseline_text,
                'ablated_output': ablated_text,
                'baseline_logits_shape': list(baseline_outputs.logits.shape) if hasattr(baseline_outputs, 'logits') else None,
                'success': True
            }
            logger.info(f"Ablation analysis completed with effect size: {effect_size:.3f}")
            return results
        except Exception as e:
            logger.error(f"Ablation analysis failed: {e}")
            return {
                'experiment_type': 'ablation',
                'model': model_name,
                'prompt': prompt,
                'layer': layer_name,
                'neurons': neuron_indices or 'all',
                'error': str(e),
                'success': False
            }
    def run_patching_analysis(
        self,
        model_name: str,
        clean_prompt: str,
        corrupted_prompt: str,
        patch_layers: List[str],
        cache: Optional[InterventionCache] = None
    ) -> Dict[str, Any]:
        """Run actual path patching analysis on a real model."""
        logger.info(f"Running path patching analysis: {model_name}")
        # Load model
        adapter = self.load_model(model_name)
        hook_manager = self.hook_managers[model_name]
        # Prepare inputs
        clean_inputs = adapter.prepare_inputs([clean_prompt])
        corrupted_inputs = adapter.prepare_inputs([corrupted_prompt])
        results = {
            'experiment_type': 'path_patching',
            'model': model_name,
            'clean_prompt': clean_prompt,
            'corrupted_prompt': corrupted_prompt,
            'patch_layers': patch_layers,
            'results': []
        }
        try:
            for layer_name in patch_layers:
                # Create intervention specification
                intervention_spec = InterventionSpec(
                    layer_name=layer_name,
                    intervention_type=InterventionType.PATCHING,
                    target_indices=None  # Patch entire layer
                )
                # Run patching
                patched_outputs = run_with_patching(
                    model=adapter.model,
                    clean_inputs=clean_inputs,
                    corrupted_inputs=corrupted_inputs,
                    intervention_spec=intervention_spec,
                    hook_manager=hook_manager,
                    cache=cache
                )
                # Get baseline corrupted output for comparison
                with torch.no_grad():
                    corrupted_outputs = adapter.model(**corrupted_inputs)
                # Calculate causal effect
                effect_size = calculate_causal_effect(corrupted_outputs, patched_outputs)
                layer_result = {
                    'layer': layer_name,
                    'causal_effect': float(effect_size),
                    'interpretation': self._interpret_effect_size(effect_size)
                }
                results['results'].append(layer_result)
                logger.info(f"Layer {layer_name} patching effect: {effect_size:.3f}")
            results['success'] = True
            logger.info("Path patching analysis completed successfully")
            return results
        except Exception as e:
            logger.error(f"Path patching analysis failed: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results
    def _decode_output(self, adapter: ModelAdapter, outputs, inputs: Dict[str, torch.Tensor]) -> str:
        """Decode model outputs to human-readable text."""
        try:
            if hasattr(outputs, 'logits'):
                # For causal LM models, get the next token prediction
                logits = outputs.logits[0, -1, :]  # Last token prediction
                predicted_token_id = torch.argmax(logits).item()
                predicted_token = adapter.tokenizer.decode([predicted_token_id])
                # Also get the original prompt
                original_text = adapter.tokenizer.decode(inputs['input_ids'][0])
                return f"{original_text.strip()} → {predicted_token.strip()}"
            else:
                return "No logits available"
        except Exception as e:
            logger.warning(f"Failed to decode output: {e}")
            return "Decoding failed"
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret the magnitude of the effect size."""
        abs_effect = abs(effect_size)
        if abs_effect < 0.1:
            return "MINIMAL effect - component has little impact"
        elif abs_effect < 0.3:
            return "SMALL effect - component has minor impact"
        elif abs_effect < 0.7:
            return "MODERATE effect - component has noticeable impact"
        elif abs_effect < 1.0:
            return "LARGE effect - component is important"
        else:
            return "VERY LARGE effect - component is critical"
    def list_available_layers(self, model_name: str, layer_type: Optional[str] = None) -> List[str]:
        """List available layers for intervention."""
        adapter = self.load_model(model_name)
        all_layers = adapter.get_layer_names()
        if layer_type == 'attention':
            return [name for name in all_layers if 'attn' in name or 'attention' in name]
        elif layer_type == 'mlp':
            return [name for name in all_layers if 'mlp' in name or 'feed_forward' in name]
        else:
            return all_layers
    def cleanup(self):
        """Clean up loaded models and free memory."""
        logger.info("Cleaning up loaded models")
        for model_name in list(self.loaded_models.keys()):
            del self.loaded_models[model_name]
            del self.hook_managers[model_name]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model cleanup completed")
    def _calculate_ablation_effect(self, baseline_output, ablated_output) -> float:
        """Calculate the effect size of ablation intervention."""
        try:
            # Extract logits from outputs
            baseline_logits = baseline_output.logits if hasattr(baseline_output, 'logits') else baseline_output
            ablated_logits = ablated_output.logits if hasattr(ablated_output, 'logits') else ablated_output
            # Calculate normalized difference between outputs
            diff = torch.norm(baseline_logits - ablated_logits)
            baseline_norm = torch.norm(baseline_logits)
            # Return relative change
            return (diff / (baseline_norm + 1e-8)).item()
        except Exception as e:
            logger.warning(f"Failed to calculate ablation effect: {e}")
            return 0.0
# Global model manager instance
_model_manager = None
def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
