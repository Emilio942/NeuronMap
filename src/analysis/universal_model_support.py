"""
Universal Model Support Framework
Provides automatic layer mapping and cross-architecture compatibility for all supported models.
"""

import json
import logging
import inspect
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import re

# Set up logging
logger = logging.getLogger(__name__)


class ArchitectureType(Enum):
    """Supported model architecture types."""
    GPT = "gpt"
    BERT = "bert"
    T5 = "t5"
    LLAMA = "llama"
    DOMAIN_SPECIFIC = "domain_specific"
    UNKNOWN = "unknown"


class LayerType(Enum):
    """Types of layers that can be found in transformer models."""
    EMBEDDING = "embedding"
    ATTENTION = "attention"
    FEED_FORWARD = "feed_forward"
    CROSS_ATTENTION = "cross_attention"
    NORMALIZATION = "normalization"
    OUTPUT = "output"
    POOLER = "pooler"
    UNKNOWN = "unknown"


@dataclass
class LayerInfo:
    """Information about a discovered layer."""
    name: str
    layer_type: LayerType
    index: int
    module: nn.Module
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    num_heads: Optional[int] = None
    is_encoder: bool = True
    is_decoder: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterConfig:
    """Configuration for model adapters."""
    architecture_type: ArchitectureType
    layer_patterns: List[str]
    attention_patterns: List[str]
    special_tokens: List[str] = field(default_factory=list)
    max_sequence_length: int = 512
    supports_encoder_decoder: bool = False
    supports_bidirectional: bool = False
    normalization_type: str = "layer_norm"
    position_embedding_type: str = "absolute"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ArchitectureRegistry:
    """Registry for different model architectures and their configurations."""

    def __init__(self):
        self._registry: Dict[str, AdapterConfig] = {}
        self._patterns: Dict[ArchitectureType, List[str]] = {}
        self._initialize_default_configs()

    def _initialize_default_configs(self):
        """Initialize default configurations for known architectures."""

        # GPT family configurations
        gpt_config = AdapterConfig(
            architecture_type=ArchitectureType.GPT,
            layer_patterns=[
                r".*transformer\.h\.(\d+).*",
                r".*layers\.(\d+).*",
                r".*block\.(\d+).*"
            ],
            attention_patterns=[
                r".*attn.*",
                r".*attention.*",
                r".*self_attn.*"
            ],
            special_tokens=["<|endoftext|>", "<pad>", "<unk>"],
            max_sequence_length=2048,
            supports_encoder_decoder=False,
            supports_bidirectional=False,
            metadata={"autoregressive": True, "causal_mask": True}
        )

        # BERT family configurations
        bert_config = AdapterConfig(
            architecture_type=ArchitectureType.BERT,
            layer_patterns=[
                r".*encoder\.layer\.(\d+).*",
                r".*layer\.(\d+).*"
            ],
            attention_patterns=[
                r".*attention.*",
                r".*self.*"
            ],
            special_tokens=["[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]"],
            max_sequence_length=512,
            supports_encoder_decoder=False,
            supports_bidirectional=True,
            metadata={"masked_lm": True, "nsp": True}
        )

        # T5 family configurations
        t5_config = AdapterConfig(
            architecture_type=ArchitectureType.T5,
            layer_patterns=[
                r".*encoder\.block\.(\d+).*",
                r".*decoder\.block\.(\d+).*"
            ],
            attention_patterns=[
                r".*SelfAttention.*",
                r".*EncDecAttention.*",
                r".*attention.*"
            ],
            special_tokens=["</s>", "<pad>", "<unk>"],
            max_sequence_length=512,
            supports_encoder_decoder=True,
            supports_bidirectional=False,
            normalization_type="rms_norm",
            position_embedding_type="relative",
            metadata={"text_to_text": True, "relative_attention": True}
        )

        # LLaMA family configurations
        llama_config = AdapterConfig(
            architecture_type=ArchitectureType.LLAMA,
            layer_patterns=[
                r".*layers\.(\d+).*",
                r".*decoder\.layers\.(\d+).*"
            ],
            attention_patterns=[
                r".*self_attn.*",
                r".*attention.*"
            ],
            special_tokens=["</s>", "<unk>", "<pad>"],
            max_sequence_length=2048,
            supports_encoder_decoder=False,
            supports_bidirectional=False,
            normalization_type="rms_norm",
            metadata={"rotary_embeddings": True, "grouped_query_attention": True}
        )

        self.register("gpt", gpt_config)
        self.register("bert", bert_config)
        self.register("t5", t5_config)
        self.register("llama", llama_config)

    def register(self, model_family: str, config: AdapterConfig):
        """Register a model family with its configuration."""
        self._registry[model_family.lower()] = config

        # Update pattern mapping
        if config.architecture_type not in self._patterns:
            self._patterns[config.architecture_type] = []
        self._patterns[config.architecture_type].extend(config.layer_patterns)

        logger.info(f"Registered model family: {model_family}")

    def get_config(self, model_family: str) -> Optional[AdapterConfig]:
        """Get configuration for a model family."""
        return self._registry.get(model_family.lower())

    def detect_architecture(self, model_name: str) -> ArchitectureType:
        """Detect architecture type from model name."""
        model_name_lower = model_name.lower()

        # Check for known patterns
        if any(pattern in model_name_lower for pattern in ["gpt", "codegen", "gpt-j", "gpt-neo"]):
            return ArchitectureType.GPT
        elif any(pattern in model_name_lower for pattern in ["bert", "roberta", "deberta", "distilbert"]):
            return ArchitectureType.BERT
        elif any(pattern in model_name_lower for pattern in ["t5", "ul2", "flan"]):
            return ArchitectureType.T5
        elif any(pattern in model_name_lower for pattern in ["llama", "alpaca", "vicuna"]):
            return ArchitectureType.LLAMA
        elif any(pattern in model_name_lower for pattern in ["codebert", "scibert", "biobert"]):
            return ArchitectureType.DOMAIN_SPECIFIC
        else:
            return ArchitectureType.UNKNOWN

    def list_supported_families(self) -> List[str]:
        """List all registered model families."""
        return list(self._registry.keys())


class UniversalLayerMapper:
    """Universal layer mapping system for automatic layer discovery."""

    def __init__(self, registry: ArchitectureRegistry):
        self.registry = registry
        self._layer_cache: Dict[str, List[LayerInfo]] = {}

    def discover_layers(self, model: nn.Module, model_name: str) -> List[LayerInfo]:
        """Discover all layers in a model automatically."""
        if model_name in self._layer_cache:
            return self._layer_cache[model_name]

        architecture_type = self.registry.detect_architecture(model_name)
        config = self.registry.get_config(architecture_type.value)

        if config is None:
            logger.warning(f"No configuration found for {model_name}, using generic discovery")
            layers = self._generic_layer_discovery(model)
        else:
            layers = self._pattern_based_discovery(model, config)

        # Cache the results
        self._layer_cache[model_name] = layers
        logger.info(f"Discovered {len(layers)} layers in {model_name}")

        return layers

    def _pattern_based_discovery(self, model: nn.Module, config: AdapterConfig) -> List[LayerInfo]:
        """Discover layers using architecture-specific patterns."""
        layers = []

        for name, module in model.named_modules():
            layer_info = self._analyze_module(name, module, config)
            if layer_info:
                layers.append(layer_info)

        # Sort by layer index
        layers.sort(key=lambda x: x.index)
        return layers

    def _generic_layer_discovery(self, model: nn.Module) -> List[LayerInfo]:
        """Generic layer discovery for unknown architectures."""
        layers = []
        layer_index = 0

        for name, module in model.named_modules():
            layer_type = self._classify_layer_type(name, module)
            if layer_type != LayerType.UNKNOWN:
                layer_info = LayerInfo(
                    name=name,
                    layer_type=layer_type,
                    index=layer_index,
                    module=module
                )
                layers.append(layer_info)
                layer_index += 1

        return layers

    def _analyze_module(self, name: str, module: nn.Module, config: AdapterConfig) -> Optional[LayerInfo]:
        """Analyze a module to determine if it's a relevant layer."""
        # Extract layer index from name using patterns
        layer_index = self._extract_layer_index(name, config.layer_patterns)
        if layer_index is None:
            return None

        layer_type = self._classify_layer_type(name, module)
        if layer_type == LayerType.UNKNOWN:
            return None

        # Determine encoder/decoder status for T5-like models
        is_encoder = "encoder" in name.lower() or not config.supports_encoder_decoder
        is_decoder = "decoder" in name.lower()

        # Extract additional metadata
        input_dim, output_dim = self._extract_dimensions(module)
        num_heads = self._extract_attention_heads(module)

        return LayerInfo(
            name=name,
            layer_type=layer_type,
            index=layer_index,
            module=module,
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            is_encoder=is_encoder,
            is_decoder=is_decoder,
            metadata={"config_type": config.architecture_type.value}
        )

    def _extract_layer_index(self, name: str, patterns: List[str]) -> Optional[int]:
        """Extract layer index from module name using regex patterns."""
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                try:
                    return int(match.group(1))
                except (IndexError, ValueError):
                    continue
        return None

    def _classify_layer_type(self, name: str, module: nn.Module) -> LayerType:
        """Classify the type of a layer based on name and module type."""
        name_lower = name.lower()

        # Check for attention layers
        if any(pattern in name_lower for pattern in ["attention", "attn", "self_attn"]):
            if "cross" in name_lower or "enc_dec" in name_lower:
                return LayerType.CROSS_ATTENTION
            return LayerType.ATTENTION

        # Check for feed-forward layers
        if any(pattern in name_lower for pattern in ["mlp", "ffn", "feed_forward", "intermediate"]):
            return LayerType.FEED_FORWARD

        # Check for normalization layers
        if any(pattern in name_lower for pattern in ["norm", "layernorm", "rmsnorm"]):
            return LayerType.NORMALIZATION

        # Check for embedding layers
        if any(pattern in name_lower for pattern in ["embed", "wte", "wpe", "position"]):
            return LayerType.EMBEDDING

        # Check for output layers
        if any(pattern in name_lower for pattern in ["output", "head", "classifier", "lm_head"]):
            return LayerType.OUTPUT

        # Check for pooler layers
        if "pooler" in name_lower:
            return LayerType.POOLER

        return LayerType.UNKNOWN

    def _extract_dimensions(self, module: nn.Module) -> Tuple[Optional[int], Optional[int]]:
        """Extract input and output dimensions from a module."""
        if isinstance(module, nn.Linear):
            return module.in_features, module.out_features
        elif isinstance(module, nn.Embedding):
            return module.num_embeddings, module.embedding_dim
        elif hasattr(module, 'config'):
            config = module.config
            if hasattr(config, 'hidden_size'):
                return config.hidden_size, config.hidden_size

        return None, None

    def _extract_attention_heads(self, module: nn.Module) -> Optional[int]:
        """Extract number of attention heads from a module."""
        if hasattr(module, 'num_heads'):
            return module.num_heads
        elif hasattr(module, 'num_attention_heads'):
            return module.num_attention_heads
        elif hasattr(module, 'config'):
            config = module.config
            if hasattr(config, 'num_attention_heads'):
                return config.num_attention_heads
            elif hasattr(config, 'num_heads'):
                return config.num_heads

        return None

    def get_layers_by_type(self, layers: List[LayerInfo], layer_type: LayerType) -> List[LayerInfo]:
        """Get all layers of a specific type."""
        return [layer for layer in layers if layer.layer_type == layer_type]

    def get_encoder_layers(self, layers: List[LayerInfo]) -> List[LayerInfo]:
        """Get all encoder layers."""
        return [layer for layer in layers if layer.is_encoder and not layer.is_decoder]

    def get_decoder_layers(self, layers: List[LayerInfo]) -> List[LayerInfo]:
        """Get all decoder layers."""
        return [layer for layer in layers if layer.is_decoder]


class DomainAdapterRegistry:
    """Registry for domain-specific adapters."""

    def __init__(self):
        self._adapters: Dict[str, Any] = {}

    def register_adapter(self, domain: str, adapter_class: type):
        """Register a domain-specific adapter."""
        self._adapters[domain.lower()] = adapter_class
        logger.info(f"Registered domain adapter: {domain}")

    def get_adapter(self, domain: str) -> Optional[type]:
        """Get a domain-specific adapter."""
        return self._adapters.get(domain.lower())

    def list_domains(self) -> List[str]:
        """List all registered domains."""
        return list(self._adapters.keys())


class UniversalModelSupport:
    """Universal model support system with automatic layer mapping and cross-architecture compatibility."""

    def __init__(self):
        self.architecture_registry = ArchitectureRegistry()
        self.layer_mapper = UniversalLayerMapper(self.architecture_registry)
        self.domain_adapters = DomainAdapterRegistry()
        self._model_cache: Dict[str, Any] = {}

    def add_model_support(self, model_family: str, adapter_config: AdapterConfig):
        """Add support for a new model family."""
        self.architecture_registry.register(model_family, adapter_config)
        logger.info(f"Added support for model family: {model_family}")

    def analyze_model_architecture(self, model: nn.Module, model_name: str) -> Dict[str, Any]:
        """Analyze a model's architecture and return comprehensive information."""
        # Discover layers
        layers = self.layer_mapper.discover_layers(model, model_name)

        # Detect architecture type
        architecture_type = self.architecture_registry.detect_architecture(model_name)

        # Get configuration
        config = self.architecture_registry.get_config(architecture_type.value)

        # Analyze layer distribution
        layer_types = {}
        for layer in layers:
            layer_type = layer.layer_type.value
            if layer_type not in layer_types:
                layer_types[layer_type] = 0
            layer_types[layer_type] += 1

        # Check for encoder-decoder structure
        encoder_layers = self.layer_mapper.get_encoder_layers(layers)
        decoder_layers = self.layer_mapper.get_decoder_layers(layers)

        analysis = {
            "model_name": model_name,
            "architecture_type": architecture_type.value,
            "total_layers": len(layers),
            "layer_types": layer_types,
            "encoder_layers": len(encoder_layers),
            "decoder_layers": len(decoder_layers),
            "is_encoder_decoder": len(decoder_layers) > 0,
            "supports_bidirectional": config.supports_bidirectional if config else False,
            "max_sequence_length": config.max_sequence_length if config else None,
            "special_tokens": config.special_tokens if config else [],
            "layers": [
                {
                    "name": layer.name,
                    "type": layer.layer_type.value,
                    "index": layer.index,
                    "is_encoder": layer.is_encoder,
                    "is_decoder": layer.is_decoder,
                    "input_dim": layer.input_dim,
                    "output_dim": layer.output_dim,
                    "num_heads": layer.num_heads
                }
                for layer in layers
            ]
        }

        return analysis

    def get_layer_mapping(self, model: nn.Module, model_name: str, target_layers: Optional[List[int]] = None) -> Dict[int, LayerInfo]:
        """Get a mapping of layer indices to LayerInfo objects."""
        layers = self.layer_mapper.discover_layers(model, model_name)

        layer_mapping = {}
        for layer in layers:
            if target_layers is None or layer.index in target_layers:
                layer_mapping[layer.index] = layer

        return layer_mapping

    def validate_cross_architecture_compatibility(self, model1_name: str, model2_name: str) -> Dict[str, Any]:
        """Validate compatibility between two different architectures."""
        arch1 = self.architecture_registry.detect_architecture(model1_name)
        arch2 = self.architecture_registry.detect_architecture(model2_name)

        config1 = self.architecture_registry.get_config(arch1.value)
        config2 = self.architecture_registry.get_config(arch2.value)

        compatibility = {
            "architectures": [arch1.value, arch2.value],
            "are_compatible": arch1 == arch2,
            "common_features": [],
            "differences": [],
            "recommendations": []
        }

        if config1 and config2:
            # Check common features
            if config1.supports_encoder_decoder == config2.supports_encoder_decoder:
                compatibility["common_features"].append("encoder_decoder_structure")
            else:
                compatibility["differences"].append("encoder_decoder_structure")

            if config1.supports_bidirectional == config2.supports_bidirectional:
                compatibility["common_features"].append("bidirectional_support")
            else:
                compatibility["differences"].append("bidirectional_support")

            if config1.normalization_type == config2.normalization_type:
                compatibility["common_features"].append("normalization_type")
            else:
                compatibility["differences"].append("normalization_type")

            # Generate recommendations
            if arch1 != arch2:
                compatibility["recommendations"].append(
                    "Consider using architecture-specific analysis methods"
                )
                compatibility["recommendations"].append(
                    "Use universal interface for consistent results"
                )

        return compatibility

    def get_supported_models(self) -> Dict[str, Any]:
        """Get information about all supported models."""
        return {
            "model_families": self.architecture_registry.list_supported_families(),
            "architecture_types": [arch.value for arch in ArchitectureType],
            "domain_adapters": self.domain_adapters.list_domains(),
            "total_supported": len(self.architecture_registry.list_supported_families())
        }

    def clear_cache(self):
        """Clear all cached model information."""
        self.layer_mapper._layer_cache.clear()
        self._model_cache.clear()
        logger.info("Cleared model caches")


# Convenience functions for easy access
def create_universal_model_support() -> UniversalModelSupport:
    """Create a new UniversalModelSupport instance with default configurations."""
    return UniversalModelSupport()


def analyze_model(model: nn.Module, model_name: str) -> Dict[str, Any]:
    """Quick function to analyze a model's architecture."""
    ums = create_universal_model_support()
    return ums.analyze_model_architecture(model, model_name)


def get_layer_info(model: nn.Module, model_name: str, layer_index: int) -> Optional[LayerInfo]:
    """Get information about a specific layer."""
    ums = create_universal_model_support()
    layer_mapping = ums.get_layer_mapping(model, model_name)
    return layer_mapping.get(layer_index)
