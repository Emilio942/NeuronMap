"""
Base Model Handler for NeuronMap
Provides common interface and utilities for all model handlers.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple, Optional, Any, Union, Protocol
import numpy as np
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model initialization and processing."""
    model_name: str
    d_model: int
    num_layers: int
    num_heads: int
    max_length: int
    vocab_size: Optional[int] = None
    architecture_type: str = "transformer"
    special_features: Dict[str, Any] = None
    memory_optimization: bool = False


@dataclass
class ActivationResult:
    """Container for model activation analysis results."""
    layer_activations: Dict[str, torch.Tensor]
    attention_weights: Dict[str, torch.Tensor]
    hidden_states: Dict[str, torch.Tensor]
    metadata: Dict[str, Any]
    input_ids: torch.Tensor
    input_text: str


class BaseModelHandler(ABC):
    """
    Abstract base class for all model handlers in NeuronMap.

    Provides common interface and utilities for model loading,
    activation extraction, and analysis.
    """

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model handler.

        Args:
            model_name: Name/path of the model to load
            config: Optional configuration dictionary
        """
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False

        # Load model configuration
        self.model_config = self._get_model_config(model_name)

        logger.info(f"Initialized {self.__class__.__name__} for {model_name}")

    @abstractmethod
    def _get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for the specific model."""
        pass

    @abstractmethod
    def load_model(self, **kwargs) -> bool:
        """Load the model and tokenizer."""
        pass

    @abstractmethod
    def extract_activations(
        self,
        input_text: str,
        target_text: Optional[str] = None,
        layer_indices: Optional[List[int]] = None,
        return_attention: bool = True
    ) -> ActivationResult:
        """Extract activations for given input."""
        pass

    def get_layer_names(self) -> List[str]:
        """Get names of all layers in the model."""
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded first")

        layer_names = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.MultiheadAttention, nn.LayerNorm)):
                layer_names.append(name)

        return layer_names

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the loaded model."""
        if not self.is_loaded:
            return {"error": "Model not loaded"}

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "model_name": self.model_name,
            "architecture": self.model_config.architecture_type,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "d_model": self.model_config.d_model,
            "num_layers": self.model_config.num_layers,
            "num_heads": self.model_config.num_heads,
            "max_length": self.model_config.max_length,
            "vocab_size": self.model_config.vocab_size,
            "device": str(self.device),
            "memory_usage_mb": self._get_memory_usage(),
            "special_features": self.model_config.special_features or {}
        }

    def _get_memory_usage(self) -> float:
        """Get approximate memory usage in MB."""
        if not self.is_loaded:
            return 0.0

        total_memory = 0
        for param in self.model.parameters():
            total_memory += param.numel() * param.element_size()

        return total_memory / (1024 * 1024)  # Convert to MB

    def _validate_input(self, input_text: str) -> bool:
        """Validate input text."""
        if not input_text or not isinstance(input_text, str):
            raise ValueError("Input text must be a non-empty string")

        if len(input_text.strip()) == 0:
            raise ValueError("Input text cannot be empty or whitespace only")

        return True

    def _prepare_inputs(
        self,
        input_text: str,
        target_text: Optional[str] = None,
        max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Prepare model inputs from text."""
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded first")

        self._validate_input(input_text)

        max_len = max_length or self.model_config.max_length

        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_len,
            truncation=True,
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs

    def _register_hooks(
        self,
        layer_indices: Optional[List[int]] = None,
        hook_type: str = "forward"
    ) -> Dict[str, Any]:
        """Register hooks for activation extraction."""
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded first")

        hooks = {}
        activations = {}

        def forward_hook(name):
            def hook(module, input, output):
                activations[name] = output.detach().cpu() if isinstance(output, torch.Tensor) else output
            return hook

        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                activations[f"{name}_grad"] = grad_output[0].detach().cpu() if grad_output[0] is not None else None
            return hook

        # Register hooks on specified layers
        target_layers = self._get_target_layers(layer_indices)

        for name, module in target_layers:
            if hook_type == "forward":
                handle = module.register_forward_hook(forward_hook(name))
            else:
                handle = module.register_backward_hook(backward_hook(name))

            hooks[name] = handle

        return hooks, activations

    def _get_target_layers(self, layer_indices: Optional[List[int]] = None) -> List[Tuple[str, nn.Module]]:
        """Get target layers for hook registration."""
        all_layers = [(name, module) for name, module in self.model.named_modules()]

        if layer_indices is None:
            return all_layers

        # Filter by indices (assuming ordered layers)
        filtered_layers = []
        layer_count = 0

        for name, module in all_layers:
            if any(isinstance(module, layer_type) for layer_type in [nn.Linear, nn.MultiheadAttention, nn.LayerNorm]):
                if layer_count in layer_indices:
                    filtered_layers.append((name, module))
                layer_count += 1

        return filtered_layers

    def cleanup(self):
        """Cleanup resources and move model to CPU if needed."""
        if self.is_loaded and self.model is not None:
            # Move model to CPU to free GPU memory
            self.model.cpu()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info(f"Cleaned up {self.model_name}")


class ModelFactory:
    """Factory for creating model handlers based on model type."""

    _handlers = {}

    @classmethod
    def register_handler(cls, model_type: str, handler_class):
        """Register a model handler for a specific model type."""
        cls._handlers[model_type] = handler_class

    @classmethod
    def create_handler(cls, model_name: str, **kwargs) -> BaseModelHandler:
        """Create appropriate model handler based on model name."""
        # Determine model type from name
        model_type = cls._detect_model_type(model_name)

        if model_type not in cls._handlers:
            raise ValueError(f"No handler registered for model type: {model_type}")

        handler_class = cls._handlers[model_type]
        return handler_class(model_name, **kwargs)

    @classmethod
    def _detect_model_type(cls, model_name: str) -> str:
        """Detect model type from model name."""
        model_name_lower = model_name.lower()

        if any(variant in model_name_lower for variant in ['t5', 'ul2', 'flan']):
            return 't5'
        elif any(variant in model_name_lower for variant in ['bert', 'roberta', 'electra']):
            return 'bert'
        elif any(variant in model_name_lower for variant in ['gpt', 'gpt2', 'gpt-neo']):
            return 'gpt'
        elif any(variant in model_name_lower for variant in ['llama', 'alpaca', 'vicuna']):
            return 'llama'
        else:
            return 'auto'  # Fallback to auto-detection
