"""
Model Adapter Plugin for BERT-like Models
========================================

Specialized adapter for BERT and similar transformer models.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AutoConfig

import sys

# Add src directory to path
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from core.plugin_system import ModelAdapterPlugin, PluginMetadata

class BertModelAdapterPlugin(ModelAdapterPlugin):
    """Model adapter plugin specifically for BERT-like transformer models."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="BERT Model Adapter",
            version="1.0.0",
            author="NeuronMap Team",
            description="Specialized adapter for BERT and similar transformer models",
            plugin_type="model_adapter",
            dependencies=["torch", "transformers", "numpy"],
            tags=["bert", "transformer", "nlp"],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.supported_models = [
            'bert-base-uncased', 'bert-large-uncased',
            'distilbert-base-uncased', 'roberta-base',
            'albert-base-v2', 'electra-base-discriminator'
        ]
        return True
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute model adapter functionality."""
        action = kwargs.get('action', 'load_model')
        if action == 'load_model':
            return self.load_model(*args, **kwargs)
        elif action == 'extract_activations':
            return self.extract_activations(*args, **kwargs)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def load_model(self, model_name: str, config: Dict[str, Any]) -> Any:
        """Load and return a BERT-like model."""
        try:
            # Check if model is supported
            if not any(supported in model_name.lower() for supported in 
                      ['bert', 'distilbert', 'roberta', 'albert', 'electra']):
                raise ValueError(f"Model {model_name} is not supported by this adapter")
            
            # Load model and tokenizer
            model_config = AutoConfig.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, config=model_config)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Move to device
            model = model.to(self.device)
            model.eval()  # Set to evaluation mode
            
            # Store model info
            model_info = {
                'model': model,
                'tokenizer': tokenizer,
                'config': model_config,
                'device': self.device,
                'model_name': model_name,
                'num_layers': model_config.num_hidden_layers,
                'hidden_size': model_config.hidden_size,
                'num_attention_heads': model_config.num_attention_heads
            }
            
            return model_info
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")
    
    def extract_activations(self, model_info: Dict[str, Any], inputs: Any, 
                          layer_names: List[str]) -> Dict[str, np.ndarray]:
        """Extract activations from BERT-like model."""
        try:
            model = model_info['model']
            tokenizer = model_info['tokenizer']
            device = model_info['device']
            
            # Prepare inputs
            if isinstance(inputs, str):
                inputs = [inputs]
            elif isinstance(inputs, list) and all(isinstance(item, str) for item in inputs):
                pass  # Already a list of strings
            else:
                raise ValueError("Inputs must be string or list of strings")
            
            # Tokenize inputs
            encoded_inputs = tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            # Prepare hooks for activation extraction
            activations = {}
            hooks = []
            
            def create_hook(layer_name):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        # For some models, output is a tuple
                        activation = output[0].detach().cpu().numpy()
                    else:
                        activation = output.detach().cpu().numpy()
                    activations[layer_name] = activation
                return hook_fn
            
            # Register hooks for requested layers
            for layer_name in layer_names:
                try:
                    layer_module = self._get_layer_module(model, layer_name)
                    if layer_module is not None:
                        hook = layer_module.register_forward_hook(create_hook(layer_name))
                        hooks.append(hook)
                except Exception as e:
                    print(f"Warning: Could not register hook for layer {layer_name}: {e}")
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**encoded_inputs, output_hidden_states=True, output_attentions=True)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Add special extractions for BERT-specific outputs
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                # Extract all hidden states
                for i, hidden_state in enumerate(outputs.hidden_states):
                    layer_key = f"hidden_state_{i}"
                    if layer_key in layer_names or f"layer_{i}" in layer_names:
                        activations[layer_key] = hidden_state.detach().cpu().numpy()
            
            if hasattr(outputs, 'attentions') and outputs.attentions:
                # Extract attention weights
                for i, attention in enumerate(outputs.attentions):
                    layer_key = f"attention_{i}"
                    if layer_key in layer_names:
                        activations[layer_key] = attention.detach().cpu().numpy()
            
            # Extract pooler output if available
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                if 'pooler_output' in layer_names:
                    activations['pooler_output'] = outputs.pooler_output.detach().cpu().numpy()
            
            # Extract last hidden state
            if hasattr(outputs, 'last_hidden_state'):
                if 'last_hidden_state' in layer_names:
                    activations['last_hidden_state'] = outputs.last_hidden_state.detach().cpu().numpy()
            
            return activations
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract activations: {str(e)}")
    
    def _get_layer_module(self, model, layer_name: str):
        """Get the actual layer module from the model."""
        try:
            # Handle different naming conventions
            if layer_name.startswith('encoder.layer.'):
                # BERT style: encoder.layer.0.attention.self
                parts = layer_name.split('.')
                module = model
                for part in parts:
                    if part.isdigit():
                        module = module[int(part)]
                    else:
                        module = getattr(module, part)
                return module
            
            elif layer_name.startswith('layer_'):
                # Simple layer indexing: layer_0, layer_1, etc.
                layer_num = int(layer_name.split('_')[1])
                if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
                    return model.encoder.layer[layer_num]
                elif hasattr(model, 'transformer') and hasattr(model.transformer, 'layer'):
                    return model.transformer.layer[layer_num]
            
            elif hasattr(model, layer_name):
                # Direct attribute access
                return getattr(model, layer_name)
            
            # Try to navigate the model structure
            parts = layer_name.split('.')
            module = model
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part, None)
                    if module is None:
                        return None
            
            return module
            
        except Exception as e:
            print(f"Error getting layer module for {layer_name}: {e}")
            return None
    
    def get_available_layers(self, model_info: Dict[str, Any]) -> List[str]:
        """Get list of available layer names for this model."""
        try:
            model = model_info['model']
            config = model_info['config']
            
            layer_names = []
            
            # Add hidden states
            for i in range(config.num_hidden_layers + 1):  # +1 for embedding layer
                layer_names.append(f"hidden_state_{i}")
            
            # Add attention layers
            for i in range(config.num_hidden_layers):
                layer_names.append(f"attention_{i}")
            
            # Add special outputs
            layer_names.extend([
                'last_hidden_state',
                'pooler_output'
            ])
            
            # Add encoder layers with full paths
            for i in range(config.num_hidden_layers):
                layer_names.extend([
                    f"encoder.layer.{i}.attention.self",
                    f"encoder.layer.{i}.attention.output",
                    f"encoder.layer.{i}.intermediate",
                    f"encoder.layer.{i}.output"
                ])
            
            return layer_names
            
        except Exception as e:
            print(f"Error getting available layers: {e}")
            return []
    
    def get_model_summary(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of the model."""
        try:
            config = model_info['config']
            model = model_info['model']
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            summary = {
                'model_name': model_info['model_name'],
                'model_type': config.model_type,
                'num_layers': config.num_hidden_layers,
                'hidden_size': config.hidden_size,
                'num_attention_heads': config.num_attention_heads,
                'vocab_size': config.vocab_size,
                'max_position_embeddings': getattr(config, 'max_position_embeddings', 'N/A'),
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'device': str(model_info['device']),
                'available_layers': len(self.get_available_layers(model_info))
            }
            
            return summary
            
        except Exception as e:
            return {'error': f"Could not generate model summary: {str(e)}"}
