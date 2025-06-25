"""Universal Multi-Model Support System for NeuronMap.

This module implements comprehensive support for multiple neural network architectures
including GPT, BERT, T5, LLaMA families and domain-specific models according to
roadmap section 3.1.
"""

import logging
import torch
import transformers
from typing import Dict, List, Optional, Any, Union, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import psutil
import gc
from pathlib import Path

from .error_handling import NeuronMapException, ModelCompatibilityError
from .robust_decorators import robust_execution

logger = logging.getLogger(__name__)


class ModelFamily(Enum):
    """Supported model families."""
    GPT = "gpt"
    BERT = "bert"
    T5 = "t5"
    LLAMA = "llama"
    DOMAIN_SPECIFIC = "domain_specific"


class ModelArchitecture(Enum):
    """Specific model architectures."""
    # GPT Family
    GPT2 = "gpt2"
    GPT_NEO = "gpt-neo"
    GPT_J = "gpt-j"
    CODEGEN = "codegen"

    # BERT Family
    BERT = "bert"
    ROBERTA = "roberta"
    DEBERTA = "deberta"
    DISTILBERT = "distilbert"

    # T5 Family
    T5 = "t5"
    UL2 = "ul2"
    FLAN_T5 = "flan-t5"

    # LLaMA Family
    LLAMA = "llama"
    ALPACA = "alpaca"
    VICUNA = "vicuna"

    # Domain-Specific
    CODEBERT = "codebert"
    SCIBERT = "scibert"
    BIOBERT = "biobert"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    family: ModelFamily
    architecture: ModelArchitecture

    # Architecture-specific parameters
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    max_position_embeddings: int

    # Model-specific features
    vocab_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    is_encoder_decoder: bool = False
    uses_rms_norm: bool = False
    has_relative_attention: bool = False

    # Loading and optimization
    memory_requirements_gb: float = 2.0
    supports_gradient_checkpointing: bool = True
    recommended_batch_size: int = 8

    # Domain-specific features
    domain: Optional[str] = None
    special_tokens: List[str] = field(default_factory=list)
    tokenizer_type: str = "wordpiece"

    # Performance optimizations
    supports_mixed_precision: bool = True
    supports_model_parallel: bool = False
    min_gpu_memory_gb: float = 4.0


@dataclass
class LayerMapping:
    """Mapping for accessing layers in different architectures."""
    transformer_layers_path: str  # e.g., "transformer.h" for GPT-2
    attention_path: str  # e.g., "attn.c_attn" for GPT-2
    mlp_path: str  # e.g., "mlp.c_fc" for GPT-2
    layer_norm_path: str  # e.g., "ln_1" for GPT-2

    # For encoder-decoder models
    encoder_path: Optional[str] = None
    decoder_path: Optional[str] = None
    cross_attention_path: Optional[str] = None


class UniversalModelAdapter(ABC):
    """Abstract base class for model-specific adapters."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.layer_mapping = None

    @abstractmethod
    def load_model(self, model_name: str, device: str = "auto") -> None:
        """Load the model and tokenizer."""
        pass

    @abstractmethod
    def get_layer_mapping(self) -> LayerMapping:
        """Get layer mapping for this architecture."""
        pass

    @abstractmethod
    def extract_activations(self, inputs: List[str], layer_indices: List[int]) -> Dict[str, torch.Tensor]:
        """Extract activations from specified layers."""
        pass

    @abstractmethod
    def get_attention_patterns(self, inputs: List[str], layer_indices: List[int]) -> Dict[str, torch.Tensor]:
        """Extract attention patterns from specified layers."""
        pass

    def optimize_memory(self) -> None:
        """Apply memory optimization strategies."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class GPTModelAdapter(UniversalModelAdapter):
    """Adapter for GPT-family models."""

    GPT_CONFIGS = {
        'gpt2': ModelConfig(
            name='gpt2', family=ModelFamily.GPT, architecture=ModelArchitecture.GPT2,
            num_layers=12, hidden_size=768, num_attention_heads=12,
            max_position_embeddings=1024, vocab_size=50257,
            memory_requirements_gb=1.5, recommended_batch_size=16
        ),
        'gpt2-medium': ModelConfig(
            name='gpt2-medium', family=ModelFamily.GPT, architecture=ModelArchitecture.GPT2,
            num_layers=24, hidden_size=1024, num_attention_heads=16,
            max_position_embeddings=1024, vocab_size=50257,
            memory_requirements_gb=3.0, recommended_batch_size=8
        ),
        'gpt2-large': ModelConfig(
            name='gpt2-large', family=ModelFamily.GPT, architecture=ModelArchitecture.GPT2,
            num_layers=36, hidden_size=1280, num_attention_heads=20,
            max_position_embeddings=1024, vocab_size=50257,
            memory_requirements_gb=6.0, recommended_batch_size=4
        ),
        'gpt-neo-125M': ModelConfig(
            name='gpt-neo-125M', family=ModelFamily.GPT, architecture=ModelArchitecture.GPT_NEO,
            num_layers=12, hidden_size=768, num_attention_heads=12,
            max_position_embeddings=2048, vocab_size=50257,
            memory_requirements_gb=1.0, recommended_batch_size=16
        ),
        'gpt-neo-1.3B': ModelConfig(
            name='gpt-neo-1.3B', family=ModelFamily.GPT, architecture=ModelArchitecture.GPT_NEO,
            num_layers=24, hidden_size=2048, num_attention_heads=16,
            max_position_embeddings=2048, vocab_size=50257,
            memory_requirements_gb=8.0, recommended_batch_size=4,
            supports_model_parallel=True, min_gpu_memory_gb=8.0
        ),
        'gpt-j-6B': ModelConfig(
            name='gpt-j-6B', family=ModelFamily.GPT, architecture=ModelArchitecture.GPT_J,
            num_layers=28, hidden_size=4096, num_attention_heads=16,
            max_position_embeddings=2048, vocab_size=50400,
            memory_requirements_gb=24.0, recommended_batch_size=1,
            supports_model_parallel=True, min_gpu_memory_gb=16.0
        )
    }

    def load_model(self, model_name: str, device: str = "auto") -> None:
        """Load GPT model and tokenizer."""
        try:
            # Update config if we have a predefined one
            if model_name in self.GPT_CONFIGS:
                self.config = self.GPT_CONFIGS[model_name]

            # Determine device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load tokenizer
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with memory optimization
            model_kwargs = {"torch_dtype": torch.float16 if device == "cuda" else torch.float32}

            if self.config.memory_requirements_gb > 8.0 and torch.cuda.is_available():
                # Use device mapping for large models
                model_kwargs["device_map"] = "auto"
                model_kwargs["low_cpu_mem_usage"] = True

            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name, **model_kwargs
            )

            if "device_map" not in model_kwargs:
                self.model = self.model.to(device)

            # Apply optimizations
            self.optimize_memory()

            logger.info(f"Successfully loaded GPT model: {model_name}")

        except Exception as e:
            raise ModelCompatibilityError(f"Failed to load GPT model {model_name}: {e}")

    def get_layer_mapping(self) -> LayerMapping:
        """Get layer mapping for GPT architectures."""
        if self.config.architecture == ModelArchitecture.GPT2:
            return LayerMapping(
                transformer_layers_path="transformer.h",
                attention_path="attn",
                mlp_path="mlp",
                layer_norm_path="ln_1"
            )
        elif self.config.architecture in [ModelArchitecture.GPT_NEO, ModelArchitecture.GPT_J]:
            return LayerMapping(
                transformer_layers_path="transformer.h",
                attention_path="attn.attention",
                mlp_path="mlp",
                layer_norm_path="ln_1"
            )
        else:
            # Generic GPT mapping
            return LayerMapping(
                transformer_layers_path="transformer.h",
                attention_path="attn",
                mlp_path="mlp",
                layer_norm_path="ln_1"
            )

    @robust_execution(max_retries=3)
    def extract_activations(self, inputs: List[str], layer_indices: List[int]) -> Dict[str, torch.Tensor]:
        """Extract activations from GPT model layers."""
        if not self.model or not self.tokenizer:
            raise ModelCompatibilityError("Model not loaded. Call load_model first.")

        activations = {}

        # Tokenize inputs
        encoded = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True,
            max_length=self.config.max_position_embeddings
        )

        if torch.cuda.is_available() and next(self.model.parameters()).device.type == "cuda":
            encoded = {k: v.cuda() for k, v in encoded.items()}

        # Hook function to capture activations
        captured_activations = {}

        def capture_activation(name):
            def hook(module, input, output):
                # Handle tuple outputs
                if isinstance(output, tuple):
                    output = output[0]  # Take the first element
                elif hasattr(output, "last_hidden_state"):
                    output = output.last_hidden_state
                captured_activations[name] = output.detach().cpu()
            return hook

        # Register hooks for specified layers
        hooks = []
        layer_mapping = self.get_layer_mapping()

        for layer_idx in layer_indices:
            if layer_idx < self.config.num_layers:
                layer_name = f"{layer_mapping.transformer_layers_path}.{layer_idx}"
                layer_module = self.model.get_submodule(layer_name)
                hook = layer_module.register_forward_hook(capture_activation(f"layer_{layer_idx}"))
                hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            self.model(**encoded)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Process captured activations
        for layer_idx in layer_indices:
            key = f"layer_{layer_idx}"
            if key in captured_activations:
                activations[key] = captured_activations[key]

        return activations

    @robust_execution(max_retries=3)
    def get_attention_patterns(self, inputs: List[str], layer_indices: List[int]) -> Dict[str, torch.Tensor]:
        """Extract attention patterns from GPT model layers."""
        if not self.model or not self.tokenizer:
            raise ModelCompatibilityError("Model not loaded. Call load_model first.")

        attention_patterns = {}

        # Enable attention output
        self.model.config.output_attentions = True

        # Tokenize inputs
        encoded = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True,
            max_length=self.config.max_position_embeddings
        )

        if torch.cuda.is_available() and next(self.model.parameters()).device.type == "cuda":
            encoded = {k: v.cuda() for k, v in encoded.items()}

        # Forward pass with attention output
        with torch.no_grad():
            outputs = self.model(**encoded)

        # Extract attention patterns for specified layers
        if hasattr(outputs, 'attentions') and outputs.attentions:
            for layer_idx in layer_indices:
                if layer_idx < len(outputs.attentions):
                    attention_patterns[f"layer_{layer_idx}"] = outputs.attentions[layer_idx].cpu()

        # Disable attention output to save memory
        self.model.config.output_attentions = False

        return attention_patterns


class BERTModelAdapter(UniversalModelAdapter):
    """Adapter for BERT-family models."""

    BERT_CONFIGS = {
        'bert-base-uncased': ModelConfig(
            name='bert-base-uncased', family=ModelFamily.BERT, architecture=ModelArchitecture.BERT,
            num_layers=12, hidden_size=768, num_attention_heads=12,
            max_position_embeddings=512, vocab_size=30522,
            memory_requirements_gb=1.5, recommended_batch_size=16,
            tokenizer_type="wordpiece"
        ),
        'roberta-base': ModelConfig(
            name='roberta-base', family=ModelFamily.BERT, architecture=ModelArchitecture.ROBERTA,
            num_layers=12, hidden_size=768, num_attention_heads=12,
            max_position_embeddings=514, vocab_size=50265,
            memory_requirements_gb=1.5, recommended_batch_size=16,
            tokenizer_type="bpe"
        ),
        'distilbert-base-uncased': ModelConfig(
            name='distilbert-base-uncased', family=ModelFamily.BERT, architecture=ModelArchitecture.DISTILBERT,
            num_layers=6, hidden_size=768, num_attention_heads=12,
            max_position_embeddings=512, vocab_size=30522,
            memory_requirements_gb=1.0, recommended_batch_size=32,
            tokenizer_type="wordpiece"
        )
    }

    def load_model(self, model_name: str, device: str = "auto") -> None:
        """Load BERT model and tokenizer."""
        try:
            # Update config if we have a predefined one
            if model_name in self.BERT_CONFIGS:
                self.config = self.BERT_CONFIGS[model_name]

            # Determine device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load tokenizer
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

            # Load model
            model_kwargs = {"torch_dtype": torch.float16 if device == "cuda" else torch.float32}

            self.model = transformers.AutoModel.from_pretrained(model_name, **model_kwargs)
            self.model = self.model.to(device)

            # Apply optimizations
            self.optimize_memory()

            logger.info(f"Successfully loaded BERT model: {model_name}")

        except Exception as e:
            raise ModelCompatibilityError(f"Failed to load BERT model {model_name}: {e}")

    def get_layer_mapping(self) -> LayerMapping:
        """Get layer mapping for BERT architectures."""
        if self.config.architecture == ModelArchitecture.DISTILBERT:
            return LayerMapping(
                transformer_layers_path="transformer.layer",
                attention_path="attention",
                mlp_path="ffn",
                layer_norm_path="sa_layer_norm"
            )
        else:
            # Standard BERT/RoBERTa mapping
            return LayerMapping(
                transformer_layers_path="encoder.layer",
                attention_path="attention.self",
                mlp_path="intermediate",
                layer_norm_path="attention.output.LayerNorm"
            )

    @robust_execution(max_retries=3)
    def extract_activations(self, inputs: List[str], layer_indices: List[int]) -> Dict[str, torch.Tensor]:
        """Extract activations from BERT model layers."""
        if not self.model or not self.tokenizer:
            raise ModelCompatibilityError("Model not loaded. Call load_model first.")

        activations = {}

        # Tokenize inputs
        encoded = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True,
            max_length=self.config.max_position_embeddings
        )

        if torch.cuda.is_available() and next(self.model.parameters()).device.type == "cuda":
            encoded = {k: v.cuda() for k, v in encoded.items()}

        # Hook function to capture activations
        captured_activations = {}

        def capture_activation(name):
            def hook(module, input, output):
                # Handle tuple outputs
                if isinstance(output, tuple):
                    output = output[0]  # Take the first element
                elif hasattr(output, "last_hidden_state"):
                    output = output.last_hidden_state
                captured_activations[name] = output.detach().cpu()
            return hook

        # Register hooks for specified layers
        hooks = []
        layer_mapping = self.get_layer_mapping()

        for layer_idx in layer_indices:
            if layer_idx < self.config.num_layers:
                layer_name = f"{layer_mapping.transformer_layers_path}.{layer_idx}"
                layer_module = self.model.get_submodule(layer_name)
                hook = layer_module.register_forward_hook(capture_activation(f"layer_{layer_idx}"))
                hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            self.model(**encoded)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Process captured activations
        for layer_idx in layer_indices:
            key = f"layer_{layer_idx}"
            if key in captured_activations:
                activations[key] = captured_activations[key]

        return activations

    @robust_execution(max_retries=3)
    def get_attention_patterns(self, inputs: List[str], layer_indices: List[int]) -> Dict[str, torch.Tensor]:
        """Extract attention patterns from BERT model layers."""
        if not self.model or not self.tokenizer:
            raise ModelCompatibilityError("Model not loaded. Call load_model first.")

        attention_patterns = {}

        # Enable attention output
        self.model.config.output_attentions = True

        # Tokenize inputs
        encoded = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True,
            max_length=self.config.max_position_embeddings
        )

        if torch.cuda.is_available() and next(self.model.parameters()).device.type == "cuda":
            encoded = {k: v.cuda() for k, v in encoded.items()}

        # Forward pass with attention output
        with torch.no_grad():
            outputs = self.model(**encoded)

        # Extract attention patterns for specified layers
        if hasattr(outputs, 'attentions') and outputs.attentions:
            for layer_idx in layer_indices:
                if layer_idx < len(outputs.attentions):
                    attention_patterns[f"layer_{layer_idx}"] = outputs.attentions[layer_idx].cpu()

        # Disable attention output to save memory
        self.model.config.output_attentions = False

        return attention_patterns


class T5ModelAdapter(UniversalModelAdapter):
    """Adapter for T5-family encoder-decoder models."""

    T5_CONFIGS = {
        't5-small': ModelConfig(
            name='t5-small', family=ModelFamily.T5, architecture=ModelArchitecture.T5,
            num_layers=6, hidden_size=512, num_attention_heads=8,
            max_position_embeddings=512, vocab_size=32128,
            is_encoder_decoder=True, has_relative_attention=True,
            memory_requirements_gb=1.0, recommended_batch_size=16
        ),
        't5-base': ModelConfig(
            name='t5-base', family=ModelFamily.T5, architecture=ModelArchitecture.T5,
            num_layers=12, hidden_size=768, num_attention_heads=12,
            max_position_embeddings=512, vocab_size=32128,
            is_encoder_decoder=True, has_relative_attention=True,
            memory_requirements_gb=2.0, recommended_batch_size=8
        ),
        't5-large': ModelConfig(
            name='t5-large', family=ModelFamily.T5, architecture=ModelArchitecture.T5,
            num_layers=24, hidden_size=1024, num_attention_heads=16,
            max_position_embeddings=512, vocab_size=32128,
            is_encoder_decoder=True, has_relative_attention=True,
            memory_requirements_gb=6.0, recommended_batch_size=4
        ),
        'flan-t5-base': ModelConfig(
            name='flan-t5-base', family=ModelFamily.T5, architecture=ModelArchitecture.FLAN_T5,
            num_layers=12, hidden_size=768, num_attention_heads=12,
            max_position_embeddings=512, vocab_size=32128,
            is_encoder_decoder=True, has_relative_attention=True,
            memory_requirements_gb=2.0, recommended_batch_size=8,
            domain="instruction_following"
        )
    }

    def load_model(self, model_name: str, device: str = "auto") -> None:
        """Load T5 model and tokenizer."""
        try:
            # Update config if we have a predefined one
            if model_name in self.T5_CONFIGS:
                self.config = self.T5_CONFIGS[model_name]

            # Determine device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load tokenizer
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

            # Load model with memory optimization
            model_kwargs = {"torch_dtype": torch.float16 if device == "cuda" else torch.float32}

            if self.config.memory_requirements_gb > 4.0 and torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
                model_kwargs["low_cpu_mem_usage"] = True

            self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                model_name, **model_kwargs
            )

            if "device_map" not in model_kwargs:
                self.model = self.model.to(device)

            # Apply optimizations
            self.optimize_memory()

            logger.info(f"Successfully loaded T5 model: {model_name}")

        except Exception as e:
            raise ModelCompatibilityError(f"Failed to load T5 model {model_name}: {e}")

    def get_layer_mapping(self) -> LayerMapping:
        """Get layer mapping for T5 architectures."""
        return LayerMapping(
            transformer_layers_path="encoder.block",
            attention_path="layer.0.SelfAttention",
            mlp_path="layer.1.DenseReluDense",
            layer_norm_path="layer.0.layer_norm",
            encoder_path="encoder.block",
            decoder_path="decoder.block",
            cross_attention_path="layer.1.EncDecAttention"
        )

    @robust_execution(max_retries=3)
    def extract_activations(self, inputs: List[str], layer_indices: List[int]) -> Dict[str, torch.Tensor]:
        """Extract activations from T5 encoder-decoder layers."""
        if not self.model or not self.tokenizer:
            raise ModelCompatibilityError("Model not loaded. Call load_model first.")

        activations = {}

        # Tokenize inputs for encoder
        encoded = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True,
            max_length=self.config.max_position_embeddings
        )

        if torch.cuda.is_available() and next(self.model.parameters()).device.type == "cuda":
            encoded = {k: v.cuda() for k, v in encoded.items()}

        # Hook function to capture activations
        captured_activations = {}

        def capture_activation(name):
            def hook(module, input, output):
                # Handle tuple outputs
                if isinstance(output, tuple):
                    output = output[0]  # Take the first element
                elif hasattr(output, "last_hidden_state"):
                    output = output.last_hidden_state
                captured_activations[name] = output.detach().cpu()
            return hook

        # Register hooks for encoder layers
        hooks = []
        layer_mapping = self.get_layer_mapping()

        for layer_idx in layer_indices:
            if layer_idx < self.config.num_layers:
                # Encoder layer
                encoder_layer_name = f"{layer_mapping.encoder_path}.{layer_idx}"
                encoder_module = self.model.get_submodule(encoder_layer_name)
                hook = encoder_module.register_forward_hook(capture_activation(f"encoder_layer_{layer_idx}"))
                hooks.append(hook)

                # Decoder layer (if exists)
                try:
                    decoder_layer_name = f"{layer_mapping.decoder_path}.{layer_idx}"
                    decoder_module = self.model.get_submodule(decoder_layer_name)
                    hook = decoder_module.register_forward_hook(capture_activation(f"decoder_layer_{layer_idx}"))
                    hooks.append(hook)
                except:
                    pass  # Decoder layer might not exist or be accessible

        # Forward pass through encoder
        with torch.no_grad():
            # Generate dummy decoder input for full forward pass
            decoder_input_ids = torch.zeros((encoded['input_ids'].shape[0], 1), dtype=torch.long)
            if torch.cuda.is_available() and next(self.model.parameters()).device.type == "cuda":
                decoder_input_ids = decoder_input_ids.cuda()

            self.model(input_ids=encoded['input_ids'],
                      attention_mask=encoded['attention_mask'],
                      decoder_input_ids=decoder_input_ids)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Process captured activations
        for layer_idx in layer_indices:
            encoder_key = f"encoder_layer_{layer_idx}"
            decoder_key = f"decoder_layer_{layer_idx}"

            if encoder_key in captured_activations:
                activations[encoder_key] = captured_activations[encoder_key]
            if decoder_key in captured_activations:
                activations[decoder_key] = captured_activations[decoder_key]

        return activations

    @robust_execution(max_retries=3)
    def get_attention_patterns(self, inputs: List[str], layer_indices: List[int]) -> Dict[str, torch.Tensor]:
        """Extract attention patterns from T5 encoder-decoder layers."""
        if not self.model or not self.tokenizer:
            raise ModelCompatibilityError("Model not loaded. Call load_model first.")

        attention_patterns = {}

        # Enable attention output
        self.model.config.output_attentions = True

        # Tokenize inputs
        encoded = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True,
            max_length=self.config.max_position_embeddings
        )

        if torch.cuda.is_available() and next(self.model.parameters()).device.type == "cuda":
            encoded = {k: v.cuda() for k, v in encoded.items()}

        # Forward pass with attention output
        with torch.no_grad():
            # Generate dummy decoder input
            decoder_input_ids = torch.zeros((encoded['input_ids'].shape[0], 1), dtype=torch.long)
            if torch.cuda.is_available() and next(self.model.parameters()).device.type == "cuda":
                decoder_input_ids = decoder_input_ids.cuda()

            outputs = self.model(input_ids=encoded['input_ids'],
                               attention_mask=encoded['attention_mask'],
                               decoder_input_ids=decoder_input_ids)

        # Extract encoder attention patterns
        if hasattr(outputs, 'encoder_attentions') and outputs.encoder_attentions:
            for layer_idx in layer_indices:
                if layer_idx < len(outputs.encoder_attentions):
                    attention_patterns[f"encoder_layer_{layer_idx}"] = outputs.encoder_attentions[layer_idx].cpu()

        # Extract decoder attention patterns
        if hasattr(outputs, 'decoder_attentions') and outputs.decoder_attentions:
            for layer_idx in layer_indices:
                if layer_idx < len(outputs.decoder_attentions):
                    attention_patterns[f"decoder_layer_{layer_idx}"] = outputs.decoder_attentions[layer_idx].cpu()

        # Extract cross attention patterns
        if hasattr(outputs, 'cross_attentions') and outputs.cross_attentions:
            for layer_idx in layer_indices:
                if layer_idx < len(outputs.cross_attentions):
                    attention_patterns[f"cross_attention_layer_{layer_idx}"] = outputs.cross_attentions[layer_idx].cpu()

        # Disable attention output to save memory
        self.model.config.output_attentions = False

        return attention_patterns


class LLaMAModelAdapter(UniversalModelAdapter):
    """Adapter for LLaMA-family models."""

    LLAMA_CONFIGS = {
        'llama-7b': ModelConfig(
            name='llama-7b', family=ModelFamily.LLAMA, architecture=ModelArchitecture.LLAMA,
            num_layers=32, hidden_size=4096, num_attention_heads=32,
            max_position_embeddings=2048, vocab_size=32000,
            uses_rms_norm=True, memory_requirements_gb=16.0, recommended_batch_size=2,
            supports_model_parallel=True, min_gpu_memory_gb=16.0
        ),
        'llama-13b': ModelConfig(
            name='llama-13b', family=ModelFamily.LLAMA, architecture=ModelArchitecture.LLAMA,
            num_layers=40, hidden_size=5120, num_attention_heads=40,
            max_position_embeddings=2048, vocab_size=32000,
            uses_rms_norm=True, memory_requirements_gb=28.0, recommended_batch_size=1,
            supports_model_parallel=True, min_gpu_memory_gb=24.0
        ),
        'alpaca-7b': ModelConfig(
            name='alpaca-7b', family=ModelFamily.LLAMA, architecture=ModelArchitecture.ALPACA,
            num_layers=32, hidden_size=4096, num_attention_heads=32,
            max_position_embeddings=2048, vocab_size=32000,
            uses_rms_norm=True, memory_requirements_gb=16.0, recommended_batch_size=2,
            supports_model_parallel=True, min_gpu_memory_gb=16.0,
            domain="instruction_following"
        ),
        'vicuna-7b': ModelConfig(
            name='vicuna-7b', family=ModelFamily.LLAMA, architecture=ModelArchitecture.VICUNA,
            num_layers=32, hidden_size=4096, num_attention_heads=32,
            max_position_embeddings=2048, vocab_size=32000,
            uses_rms_norm=True, memory_requirements_gb=16.0, recommended_batch_size=2,
            supports_model_parallel=True, min_gpu_memory_gb=16.0,
            domain="conversation"
        )
    }

    def load_model(self, model_name: str, device: str = "auto") -> None:
        """Load LLaMA model and tokenizer."""
        try:
            # Update config if we have a predefined one
            if model_name in self.LLAMA_CONFIGS:
                self.config = self.LLAMA_CONFIGS[model_name]

            # Determine device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load tokenizer
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with aggressive memory optimization for large models
            model_kwargs = {
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "trust_remote_code": True
            }

            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name, **model_kwargs
            )

            # Apply optimizations
            self.optimize_memory()

            logger.info(f"Successfully loaded LLaMA model: {model_name}")

        except Exception as e:
            raise ModelCompatibilityError(f"Failed to load LLaMA model {model_name}: {e}")

    def get_layer_mapping(self) -> LayerMapping:
        """Get layer mapping for LLaMA architectures."""
        return LayerMapping(
            transformer_layers_path="model.layers",
            attention_path="self_attn",
            mlp_path="mlp",
            layer_norm_path="input_layernorm"
        )

    @robust_execution(max_retries=3)
    def extract_activations(self, inputs: List[str], layer_indices: List[int]) -> Dict[str, torch.Tensor]:
        """Extract activations from LLaMA model layers."""
        if not self.model or not self.tokenizer:
            raise ModelCompatibilityError("Model not loaded. Call load_model first.")

        activations = {}

        # Tokenize inputs
        encoded = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True,
            max_length=self.config.max_position_embeddings
        )

        if torch.cuda.is_available() and next(self.model.parameters()).device.type == "cuda":
            encoded = {k: v.cuda() for k, v in encoded.items()}

        # Hook function to capture activations
        captured_activations = {}

        def capture_activation(name):
            def hook(module, input, output):
                # Handle tuple outputs
                if isinstance(output, tuple):
                    output = output[0]  # Take the first element
                elif hasattr(output, "last_hidden_state"):
                    output = output.last_hidden_state
                captured_activations[name] = output.detach().cpu()
            return hook

        # Register hooks for specified layers
        hooks = []
        layer_mapping = self.get_layer_mapping()

        for layer_idx in layer_indices:
            if layer_idx < self.config.num_layers:
                layer_name = f"{layer_mapping.transformer_layers_path}.{layer_idx}"
                layer_module = self.model.get_submodule(layer_name)
                hook = layer_module.register_forward_hook(capture_activation(f"layer_{layer_idx}"))
                hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            self.model(**encoded)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Process captured activations
        for layer_idx in layer_indices:
            key = f"layer_{layer_idx}"
            if key in captured_activations:
                activations[key] = captured_activations[key]

        return activations

    @robust_execution(max_retries=3)
    def get_attention_patterns(self, inputs: List[str], layer_indices: List[int]) -> Dict[str, torch.Tensor]:
        """Extract attention patterns from LLaMA model layers."""
        if not self.model or not self.tokenizer:
            raise ModelCompatibilityError("Model not loaded. Call load_model first.")

        attention_patterns = {}

        # Enable attention output
        self.model.config.output_attentions = True

        # Tokenize inputs
        encoded = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True,
            max_length=self.config.max_position_embeddings
        )

        if torch.cuda.is_available() and next(self.model.parameters()).device.type == "cuda":
            encoded = {k: v.cuda() for k, v in encoded.items()}

        # Forward pass with attention output
        with torch.no_grad():
            outputs = self.model(**encoded)

        # Extract attention patterns for specified layers
        if hasattr(outputs, 'attentions') and outputs.attentions:
            for layer_idx in layer_indices:
                if layer_idx < len(outputs.attentions):
                    attention_patterns[f"layer_{layer_idx}"] = outputs.attentions[layer_idx].cpu()

        # Disable attention output to save memory
        self.model.config.output_attentions = False

        return attention_patterns


class DomainSpecificModelAdapter(UniversalModelAdapter):
    """Adapter for domain-specific BERT variants."""

    DOMAIN_CONFIGS = {
        'microsoft/codebert-base': ModelConfig(
            name='microsoft/codebert-base', family=ModelFamily.DOMAIN_SPECIFIC,
            architecture=ModelArchitecture.CODEBERT,
            num_layers=12, hidden_size=768, num_attention_heads=12,
            max_position_embeddings=512, vocab_size=50265,
            memory_requirements_gb=1.5, recommended_batch_size=16,
            domain="programming", tokenizer_type="bpe",
            special_tokens=['<s>', '</s>', '<unk>', '<pad>', '<mask>']
        ),
        'allenai/scibert_scivocab_uncased': ModelConfig(
            name='allenai/scibert_scivocab_uncased', family=ModelFamily.DOMAIN_SPECIFIC,
            architecture=ModelArchitecture.SCIBERT,
            num_layers=12, hidden_size=768, num_attention_heads=12,
            max_position_embeddings=512, vocab_size=31090,
            memory_requirements_gb=1.5, recommended_batch_size=16,
            domain="scientific", tokenizer_type="wordpiece"
        ),
        'dmis-lab/biobert-base-cased-v1.1': ModelConfig(
            name='dmis-lab/biobert-base-cased-v1.1', family=ModelFamily.DOMAIN_SPECIFIC,
            architecture=ModelArchitecture.BIOBERT,
            num_layers=12, hidden_size=768, num_attention_heads=12,
            max_position_embeddings=512, vocab_size=28996,
            memory_requirements_gb=1.5, recommended_batch_size=16,
            domain="biomedical", tokenizer_type="wordpiece"
        )
    }

    def load_model(self, model_name: str, device: str = "auto") -> None:
        """Load domain-specific model and tokenizer."""
        try:
            # Update config if we have a predefined one
            if model_name in self.DOMAIN_CONFIGS:
                self.config = self.DOMAIN_CONFIGS[model_name]

            # Determine device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load tokenizer
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

            # Load model
            model_kwargs = {"torch_dtype": torch.float16 if device == "cuda" else torch.float32}

            self.model = transformers.AutoModel.from_pretrained(model_name, **model_kwargs)
            self.model = self.model.to(device)

            # Apply optimizations
            self.optimize_memory()

            logger.info(f"Successfully loaded domain-specific model: {model_name}")

        except Exception as e:
            raise ModelCompatibilityError(f"Failed to load domain-specific model {model_name}: {e}")

    def get_layer_mapping(self) -> LayerMapping:
        """Get layer mapping for domain-specific models (similar to BERT)."""
        return LayerMapping(
            transformer_layers_path="encoder.layer",
            attention_path="attention.self",
            mlp_path="intermediate",
            layer_norm_path="attention.output.LayerNorm"
        )

    @robust_execution(max_retries=3)
    def extract_activations(self, inputs: List[str], layer_indices: List[int]) -> Dict[str, torch.Tensor]:
        """Extract activations from domain-specific model layers."""
        # Use same implementation as BERT adapter since these are BERT-based
        if not self.model or not self.tokenizer:
            raise ModelCompatibilityError("Model not loaded. Call load_model first.")

        activations = {}

        # Tokenize inputs
        encoded = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True,
            max_length=self.config.max_position_embeddings
        )

        if torch.cuda.is_available() and next(self.model.parameters()).device.type == "cuda":
            encoded = {k: v.cuda() for k, v in encoded.items()}

        # Hook function to capture activations
        captured_activations = {}

        def capture_activation(name):
            def hook(module, input, output):
                # Handle tuple outputs
                if isinstance(output, tuple):
                    output = output[0]  # Take the first element
                elif hasattr(output, "last_hidden_state"):
                    output = output.last_hidden_state
                captured_activations[name] = output.detach().cpu()
            return hook

        # Register hooks for specified layers
        hooks = []
        layer_mapping = self.get_layer_mapping()

        for layer_idx in layer_indices:
            if layer_idx < self.config.num_layers:
                layer_name = f"{layer_mapping.transformer_layers_path}.{layer_idx}"
                layer_module = self.model.get_submodule(layer_name)
                hook = layer_module.register_forward_hook(capture_activation(f"layer_{layer_idx}"))
                hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            self.model(**encoded)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Process captured activations
        for layer_idx in layer_indices:
            key = f"layer_{layer_idx}"
            if key in captured_activations:
                activations[key] = captured_activations[key]

        return activations

    @robust_execution(max_retries=3)
    def get_attention_patterns(self, inputs: List[str], layer_indices: List[int]) -> Dict[str, torch.Tensor]:
        """Extract attention patterns from domain-specific model layers."""
        # Use same implementation as BERT adapter
        if not self.model or not self.tokenizer:
            raise ModelCompatibilityError("Model not loaded. Call load_model first.")

        attention_patterns = {}

        # Enable attention output
        self.model.config.output_attentions = True

        # Tokenize inputs
        encoded = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True,
            max_length=self.config.max_position_embeddings
        )

        if torch.cuda.is_available() and next(self.model.parameters()).device.type == "cuda":
            encoded = {k: v.cuda() for k, v in encoded.items()}

        # Forward pass with attention output
        with torch.no_grad():
            outputs = self.model(**encoded)

        # Extract attention patterns for specified layers
        if hasattr(outputs, 'attentions') and outputs.attentions:
            for layer_idx in layer_indices:
                if layer_idx < len(outputs.attentions):
                    attention_patterns[f"layer_{layer_idx}"] = outputs.attentions[layer_idx].cpu()

        # Disable attention output to save memory
        self.model.config.output_attentions = False

        return attention_patterns

class UniversalModelRegistry:
    """Registry for managing all supported models."""

    def __init__(self):
        self.adapters = {
            ModelFamily.GPT: GPTModelAdapter,
            ModelFamily.BERT: BERTModelAdapter,
            ModelFamily.T5: T5ModelAdapter,
            ModelFamily.LLAMA: LLaMAModelAdapter,
            ModelFamily.DOMAIN_SPECIFIC: DomainSpecificModelAdapter,
        }
        self.model_configs = {}
        self._initialize_configs()

    def _initialize_configs(self):
        """Initialize all model configurations."""
        # Add GPT configs
        self.model_configs.update(GPTModelAdapter.GPT_CONFIGS)
        # Add BERT configs
        self.model_configs.update(BERTModelAdapter.BERT_CONFIGS)
        # Add T5 configs
        self.model_configs.update(T5ModelAdapter.T5_CONFIGS)
        # Add LLaMA configs
        self.model_configs.update(LLaMAModelAdapter.LLAMA_CONFIGS)
        # Add Domain-Specific configs
        self.model_configs.update(DomainSpecificModelAdapter.DOMAIN_CONFIGS)

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a model."""
        return self.model_configs.get(model_name)

    def detect_model_family(self, model_name: str) -> ModelFamily:
        """Detect which family a model belongs to."""
        model_name_lower = model_name.lower()

        # Check domain-specific models first (as they often contain "bert" but are specialized)
        if any(domain_type in model_name_lower for domain_type in ['codebert', 'scibert', 'biobert']):
            return ModelFamily.DOMAIN_SPECIFIC
        elif any(gpt_type in model_name_lower for gpt_type in ['gpt', 'codegen']):
            return ModelFamily.GPT
        elif any(bert_type in model_name_lower for bert_type in ['bert', 'roberta', 'distilbert']):
            return ModelFamily.BERT
        elif any(t5_type in model_name_lower for t5_type in ['t5', 'ul2', 'flan']):
            return ModelFamily.T5
        elif any(llama_type in model_name_lower for llama_type in ['llama', 'alpaca', 'vicuna']):
            return ModelFamily.LLAMA
        else:
            return ModelFamily.DOMAIN_SPECIFIC

    def create_adapter(self, model_name: str) -> UniversalModelAdapter:
        """Create appropriate adapter for a model."""
        # Try to get existing config
        config = self.get_model_config(model_name)

        if config:
            family = config.family
        else:
            # Detect family and create basic config
            family = self.detect_model_family(model_name)
            config = ModelConfig(
                name=model_name,
                family=family,
                architecture=ModelArchitecture.GPT2,  # Default, will be updated
                num_layers=12,  # Default, will be detected
                hidden_size=768,  # Default, will be detected
                num_attention_heads=12,  # Default, will be detected
                max_position_embeddings=1024  # Default, will be detected
            )

        if family not in self.adapters:
            raise ModelCompatibilityError(f"Unsupported model family: {family}")

        return self.adapters[family](config)


class MultiModelAnalyzer:
    """Main interface for multi-model analysis."""

    def __init__(self):
        self.registry = UniversalModelRegistry()
        self.loaded_models = {}

    def load_model(self, model_name: str, device: str = "auto") -> str:
        """Load a model and return its identifier."""
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded")
            return model_name

        adapter = self.registry.create_adapter(model_name)
        adapter.load_model(model_name, device)

        self.loaded_models[model_name] = adapter
        logger.info(f"Successfully loaded model: {model_name}")

        return model_name

    def extract_activations(self, model_name: str, inputs: List[str], layer_indices: List[int]) -> Dict[str, torch.Tensor]:
        """Extract activations from a loaded model."""
        if model_name not in self.loaded_models:
            raise ModelCompatibilityError(f"Model {model_name} not loaded. Call load_model first.")

        adapter = self.loaded_models[model_name]
        return adapter.extract_activations(inputs, layer_indices)

    def get_attention_patterns(self, model_name: str, inputs: List[str], layer_indices: List[int]) -> Dict[str, torch.Tensor]:
        """Get attention patterns from a loaded model."""
        if model_name not in self.loaded_models:
            raise ModelCompatibilityError(f"Model {model_name} not loaded. Call load_model first.")

        adapter = self.loaded_models[model_name]
        return adapter.get_attention_patterns(inputs, layer_indices)

    def get_model_info(self, model_name: str) -> ModelConfig:
        """Get information about a model."""
        config = self.registry.get_model_config(model_name)
        if not config:
            # Try to load and detect config
            adapter = self.registry.create_adapter(model_name)
            config = adapter.config

        return config

    def list_supported_models(self) -> Dict[str, List[str]]:
        """List all supported models by family."""
        models_by_family = {}

        for model_name, config in self.registry.model_configs.items():
            family_name = config.family.value
            if family_name not in models_by_family:
                models_by_family[family_name] = []
            models_by_family[family_name].append(model_name)

        return models_by_family

    def unload_model(self, model_name: str) -> None:
        """Unload a model to free memory."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded model: {model_name}")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = {
            "system_memory_gb": psutil.virtual_memory().used / (1024**3),
            "system_memory_percent": psutil.virtual_memory().percent
        }

        if torch.cuda.is_available():
            memory_info["gpu_memory_gb"] = torch.cuda.memory_allocated() / (1024**3)
            memory_info["gpu_memory_percent"] = (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100 if torch.cuda.max_memory_allocated() > 0 else 0

        return memory_info


# Global instance for easy access
multi_model_analyzer = MultiModelAnalyzer()

def create_activation_hook(captured_activations: dict, name: str):
    """Create a robust activation hook that handles various output types."""
    def hook(module, input, output):
        try:
            # Handle tuple outputs (some modules return tuples)
            if isinstance(output, tuple):
                # For most transformers, the first element is the hidden states
                tensor_output = output[0]
            elif hasattr(output, 'last_hidden_state'):
                # For some model outputs that have specific attributes
                tensor_output = output.last_hidden_state
            else:
                # Direct tensor output
                tensor_output = output

            # Ensure it's a tensor and detach it
            if hasattr(tensor_output, 'detach'):
                captured_activations[name] = tensor_output.detach().cpu()
            else:
                logger.warning(f"Output for {name} is not a tensor: {type(tensor_output)}")

        except Exception as e:
            logger.error(f"Failed to capture activation for {name}: {e}")

    return hook
