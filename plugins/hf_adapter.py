"""
Hugging Face Model Adapter Plugin
================================

Provides support for loading and analyzing Hugging Face Transformers models.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import torch
import numpy as np

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from src.core.plugin_system import ModelAdapterPlugin, PluginMetadata

logger = logging.getLogger(__name__)

class HuggingFaceAdapter(ModelAdapterPlugin):
    """Adapter for Hugging Face Transformers models."""

    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="HuggingFace Adapter",
            version="1.0.0",
            author="NeuronMap Team",
            description="Support for Hugging Face Transformers models",
            plugin_type="model_adapter",
            dependencies=["transformers", "torch"],
            tags=["model", "nlp", "transformer"],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )

    def initialize(self, config: Dict[str, Any]) -> bool:
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers library not found. Please install it.")
            return False
        
        self.device = config.get("device", self.device)
        logger.info(f"HuggingFace Adapter initialized on {self.device}")
        return True

    def execute(self, *args, **kwargs) -> Any:
        """
        Generic execute method. 
        For adapters, we usually call specific methods directly, 
        but this can serve as a shortcut for extraction.
        """
        if len(args) >= 2:
            # Assume execute(model_name, input_text)
            model = self.load_model(args[0], kwargs.get("config", {}))
            return self.extract_activations(model, args[1], kwargs.get("layers", []))
        return None

    def load_model(self, model_name: str, config: Dict[str, Any]) -> Any:
        """Load and return the model and tokenizer."""
        if model_name in self.models:
            return self.models[model_name]

        try:
            logger.info(f"Loading model: {model_name}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model configuration
            model_config = AutoConfig.from_pretrained(model_name)
            model_config.output_hidden_states = True
            model_config.output_attentions = config.get("output_attentions", False)
            
            # Load model
            model = AutoModel.from_pretrained(model_name, config=model_config)
            model.to(self.device)
            model.eval()
            
            # Cache model components
            model_bundle = {
                "model": model,
                "tokenizer": tokenizer,
                "config": model_config
            }
            
            self.models[model_name] = model_bundle
            return model_bundle
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def extract_activations(self, model_bundle: Any, inputs: Union[str, List[str]], layer_names: List[str] = None) -> Dict[str, Any]:
        """
        Extract activations from the model.
        
        Args:
            model_bundle: Dict containing 'model' and 'tokenizer'
            inputs: Input text or list of texts
            layer_names: List of layer indices (integers) or names to extract. 
                         If None, extracts all layers.
        """
        model = model_bundle["model"]
        tokenizer = model_bundle["tokenizer"]
        
        # Prepare inputs
        if isinstance(inputs, str):
            inputs = [inputs]
            
        encoded_inputs = tokenizer(
            inputs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        try:
            with torch.no_grad():
                outputs = model(**encoded_inputs)
            
            # Extract hidden states
            # hidden_states is a tuple of (batch, seq_len, hidden_size) for each layer
            hidden_states = outputs.hidden_states
            
            results = {}
            
            # Determine which layers to keep
            num_layers = len(hidden_states)
            target_indices = []
            
            if layer_names:
                # Parse layer requests (handling ints and strings)
                for l in layer_names:
                    try:
                        idx = int(l)
                        if -num_layers <= idx < num_layers:
                            target_indices.append(idx)
                    except ValueError:
                        logger.warning(f"Skipping invalid layer index: {l}")
            else:
                # Default: All layers
                target_indices = list(range(num_layers))
            
            # Process selected layers
            activations = []
            for idx in target_indices:
                # Get tensor for layer
                layer_tensor = hidden_states[idx] # [batch, seq, hidden]
                
                # Convert to numpy/list for serialization
                # We take the last token for simple representation, or mean pooling
                # For this implementation, let's do mean pooling over sequence length
                # to get a single vector per input per layer
                
                # Mask padding tokens
                mask = encoded_inputs.attention_mask.unsqueeze(-1).expand(layer_tensor.size()).float()
                masked_embeddings = layer_tensor * mask
                summed = torch.sum(masked_embeddings, 1)
                counts = torch.clamp(mask.sum(1), min=1e-9)
                mean_pooled = summed / counts # [batch, hidden]
                
                activations.append(mean_pooled.cpu().numpy().tolist())
            
            # Structure results
            # activations is now [num_selected_layers, batch, hidden]
            # We might want to flatten or structure differently based on analysis needs
            
            results["activations"] = activations # Raw list structure
            results["layer_indices"] = target_indices
            results["model_name"] = model.config.name_or_path
            
            return results
            
        except Exception as e:
            logger.error(f"Activation extraction failed: {e}")
            raise

