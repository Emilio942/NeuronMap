"""
Universal Model Adapter for NeuronMap
=====================================

This module provides a universal interface for loading and analyzing different
neural network architectures including GPT, BERT, T5, LLaMA, and domain-specific models.
"""

import torch
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
from abc import ABC, abstractmethod

from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification, AutoModelForMaskedLM,
    GPT2Model, GPT2LMHeadModel, GPTNeoModel, GPTJModel,
    BertModel, RobertaModel, DistilBertModel,
    T5Model, T5ForConditionalGeneration,
    LlamaModel, LlamaForCausalLM
)

logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""

    def __init__(self, model_name: str, device: torch.device, config: Dict[str, Any]):
        self.model_name = model_name
        self.device = device
        self.config = config
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer."""
        pass

    @abstractmethod
    def prepare_inputs(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the model."""
        pass

    @abstractmethod
    def get_layer_names(self) -> List[str]:
        """Get all layer names in the model."""
        pass

    @abstractmethod
    def get_target_layers(self, layer_config: Dict[str, Any]) -> List[Tuple[str, torch.nn.Module]]:
        """Get target layers based on configuration."""
        pass


class GPTAdapter(ModelAdapter):
    """Adapter for GPT-style models (GPT-2, GPT-Neo, GPT-J, CodeGen)."""

    def load_model(self):
        """Load GPT-style model and tokenizer."""
        logger.info(f"Loading GPT model: {self.model_name}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=getattr(torch, self.config.get('preferred_dtype', 'float32')),
                device_map="auto" if self.device.type == "cuda" else None
            )

            if self.device.type != "cuda" or "device_map" not in locals():
                self.model.to(self.device)

            self.model.eval()
            logger.info(f"Successfully loaded GPT model on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load GPT model {self.model_name}: {e}")
            raise

    def prepare_inputs(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for GPT models."""
        max_length = self.config.get('max_sequence_length', 1024)

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        return {k: v.to(self.device) for k, v in inputs.items()}

    def get_layer_names(self) -> List[str]:
        """Get all layer names in GPT model."""
        layer_names = []
        for name, module in self.model.named_modules():
            layer_names.append(name)
        return layer_names

    def get_target_layers(self, layer_config: Dict[str, Any]) -> List[Tuple[str, torch.nn.Module]]:
        """Get target layers for GPT models."""
        found_layers = []
        total_layers = layer_config.get('total_layers', 12)

        attention_pattern = layer_config.get('attention', '')
        mlp_pattern = layer_config.get('mlp', '')

        for layer_idx in range(total_layers):
            # Add attention layers
            if attention_pattern:
                attn_name = attention_pattern.format(layer=layer_idx)
                attn_module = self._get_module_by_name(attn_name)
                if attn_module is not None:
                    found_layers.append((attn_name, attn_module))

            # Add MLP layers
            if mlp_pattern:
                mlp_name = mlp_pattern.format(layer=layer_idx)
                mlp_module = self._get_module_by_name(mlp_name)
                if mlp_module is not None:
                    found_layers.append((mlp_name, mlp_module))

        return found_layers

    def _get_module_by_name(self, name: str) -> Optional[torch.nn.Module]:
        """Get module by name."""
        try:
            module = self.model
            for attr in name.split('.'):
                module = getattr(module, attr)
            return module
        except AttributeError:
            return None


class BERTAdapter(ModelAdapter):
    """Adapter for BERT-style models (BERT, RoBERTa, DistilBERT, SciBERT, BioBERT)."""

    def load_model(self):
        """Load BERT-style model and tokenizer."""
        logger.info(f"Loading BERT model: {self.model_name}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Load model - try different model classes
            try:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=getattr(torch, self.config.get('preferred_dtype', 'float32'))
                )
            except:
                # Fallback to masked LM model
                self.model = AutoModelForMaskedLM.from_pretrained(
                    self.model_name,
                    torch_dtype=getattr(torch, self.config.get('preferred_dtype', 'float32'))
                )

            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Successfully loaded BERT model on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load BERT model {self.model_name}: {e}")
            raise

    def prepare_inputs(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for BERT models."""
        max_length = self.config.get('max_sequence_length', 512)

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        return {k: v.to(self.device) for k, v in inputs.items()}

    def get_layer_names(self) -> List[str]:
        """Get all layer names in BERT model."""
        layer_names = []
        for name, module in self.model.named_modules():
            layer_names.append(name)
        return layer_names

    def get_target_layers(self, layer_config: Dict[str, Any]) -> List[Tuple[str, torch.nn.Module]]:
        """Get target layers for BERT models."""
        found_layers = []
        total_layers = layer_config.get('total_layers', 12)

        attention_pattern = layer_config.get('attention', '')
        mlp_pattern = layer_config.get('mlp', '')

        for layer_idx in range(total_layers):
            # Add attention layers
            if attention_pattern:
                attn_name = attention_pattern.format(layer=layer_idx)
                attn_module = self._get_module_by_name(attn_name)
                if attn_module is not None:
                    found_layers.append((attn_name, attn_module))

            # Add MLP layers
            if mlp_pattern:
                mlp_name = mlp_pattern.format(layer=layer_idx)
                mlp_module = self._get_module_by_name(mlp_name)
                if mlp_module is not None:
                    found_layers.append((mlp_name, mlp_module))

        return found_layers

    def _get_module_by_name(self, name: str) -> Optional[torch.nn.Module]:
        """Get module by name."""
        try:
            module = self.model
            for attr in name.split('.'):
                module = getattr(module, attr)
            return module
        except AttributeError:
            return None


class T5Adapter(ModelAdapter):
    """Adapter for T5-style models (T5, FLAN-T5, CodeT5)."""

    def load_model(self):
        """Load T5-style model and tokenizer."""
        logger.info(f"Loading T5 model: {self.model_name}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=getattr(torch, self.config.get('preferred_dtype', 'float32'))
            )

            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Successfully loaded T5 model on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load T5 model {self.model_name}: {e}")
            raise

    def prepare_inputs(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for T5 models."""
        max_length = self.config.get('max_sequence_length', 512)

        # T5 expects "task prefix" for many tasks
        processed_texts = []
        for text in texts:
            if not any(text.startswith(prefix) for prefix in ["translate", "summarize", "question"]):
                text = f"analyze: {text}"
            processed_texts.append(text)

        inputs = self.tokenizer(
            processed_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        return {k: v.to(self.device) for k, v in inputs.items()}

    def get_layer_names(self) -> List[str]:
        """Get all layer names in T5 model."""
        layer_names = []
        for name, module in self.model.named_modules():
            layer_names.append(name)
        return layer_names

    def get_target_layers(self, layer_config: Dict[str, Any]) -> List[Tuple[str, torch.nn.Module]]:
        """Get target layers for T5 models."""
        found_layers = []
        total_layers = layer_config.get('total_layers', 6)

        attention_pattern = layer_config.get('attention', '')
        mlp_pattern = layer_config.get('mlp', '')

        for layer_idx in range(total_layers):
            # Add attention layers
            if attention_pattern:
                attn_name = attention_pattern.format(layer=layer_idx)
                attn_module = self._get_module_by_name(attn_name)
                if attn_module is not None:
                    found_layers.append((attn_name, attn_module))

            # Add MLP layers
            if mlp_pattern:
                mlp_name = mlp_pattern.format(layer=layer_idx)
                mlp_module = self._get_module_by_name(mlp_name)
                if mlp_module is not None:
                    found_layers.append((mlp_name, mlp_module))

        return found_layers

    def _get_module_by_name(self, name: str) -> Optional[torch.nn.Module]:
        """Get module by name."""
        try:
            module = self.model
            for attr in name.split('.'):
                module = getattr(module, attr)
            return module
        except AttributeError:
            return None


class LlamaAdapter(ModelAdapter):
    """Adapter for LLaMA-style models."""

    def load_model(self):
        """Load LLaMA-style model and tokenizer."""
        logger.info(f"Loading LLaMA model: {self.model_name}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with special handling for LLaMA
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=getattr(torch, self.config.get('preferred_dtype', 'float16')),
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True
            )

            if self.device.type != "cuda" or "device_map" not in locals():
                self.model.to(self.device)

            self.model.eval()
            logger.info(f"Successfully loaded LLaMA model on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load LLaMA model {self.model_name}: {e}")
            raise

    def prepare_inputs(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for LLaMA models."""
        max_length = self.config.get('max_sequence_length', 2048)

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        return {k: v.to(self.device) for k, v in inputs.items()}

    def get_layer_names(self) -> List[str]:
        """Get all layer names in LLaMA model."""
        layer_names = []
        for name, module in self.model.named_modules():
            layer_names.append(name)
        return layer_names

    def get_target_layers(self, layer_config: Dict[str, Any]) -> List[Tuple[str, torch.nn.Module]]:
        """Get target layers for LLaMA models."""
        found_layers = []
        total_layers = layer_config.get('total_layers', 32)

        attention_pattern = layer_config.get('attention', '')
        mlp_pattern = layer_config.get('mlp', '')

        for layer_idx in range(total_layers):
            # Add attention layers
            if attention_pattern:
                attn_name = attention_pattern.format(layer=layer_idx)
                attn_module = self._get_module_by_name(attn_name)
                if attn_module is not None:
                    found_layers.append((attn_name, attn_module))

            # Add MLP layers
            if mlp_pattern:
                mlp_name = mlp_pattern.format(layer=layer_idx)
                mlp_module = self._get_module_by_name(mlp_name)
                if mlp_module is not None:
                    found_layers.append((mlp_name, mlp_module))

        return found_layers

    def _get_module_by_name(self, name: str) -> Optional[torch.nn.Module]:
        """Get module by name."""
        try:
            module = self.model
            for attr in name.split('.'):
                module = getattr(module, attr)
            return module
        except AttributeError:
            return None


class UniversalModelAdapter:
    """Universal adapter that can handle multiple model architectures."""

    def __init__(self, config):
        self.config = config
        self.models_config = self._load_models_config()
        self.device = self._get_device()
        self.adapter = None

    def _get_device(self) -> torch.device:
        """Determine the appropriate device to use."""
        device_config = self.config.model.device

        if device_config == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_config)

        logger.info(f"Using device: {device}")
        return device

    def _load_models_config(self) -> Dict[str, Any]:
        """Load models configuration from YAML file."""
        # Try multiple possible paths for models config
        possible_paths = [
            Path("configs/models.yaml"),
            Path("models.yaml"),
            Path(__file__).parent.parent.parent / "configs" / "models.yaml"
        ]

        for models_config_path in possible_paths:
            try:
                if models_config_path.exists():
                    with open(models_config_path, 'r') as f:
                        return yaml.safe_load(f)
            except Exception as e:
                logger.debug(f"Could not load models config from {models_config_path}: {e}")
                continue

        logger.warning("Could not load models config from any location, using empty config")
        return {}

    def load_model(self, model_name: str) -> ModelAdapter:
        """Load a model using the appropriate adapter."""
        # Find model configuration
        model_config = self._find_model_config(model_name)
        if not model_config:
            logger.warning(f"No configuration found for model {model_name}, attempting auto-detection")
            model_config = self._auto_detect_model_type(model_name)

        model_type = model_config.get('type', 'auto')
        extraction_config = self.models_config.get('extraction_settings', {}).get(model_type, {})

        # Create appropriate adapter
        if model_type == 'gpt':
            self.adapter = GPTAdapter(model_name, self.device, extraction_config)
        elif model_type == 'bert':
            self.adapter = BERTAdapter(model_name, self.device, extraction_config)
        elif model_type == 't5':
            self.adapter = T5Adapter(model_name, self.device, extraction_config)
        elif model_type == 'llama':
            self.adapter = LlamaAdapter(model_name, self.device, extraction_config)
        else:
            # Auto-detect based on model name
            self.adapter = self._auto_create_adapter(model_name, extraction_config)

        # Load the model
        self.adapter.load_model()
        return self.adapter

    def _find_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Find configuration for a specific model."""
        models = self.models_config.get('models', {})

        # Direct match
        for key, config in models.items():
            if config.get('name') == model_name:
                return config

        # Partial match
        for key, config in models.items():
            if model_name in config.get('name', ''):
                return config

        return None

    def _auto_detect_model_type(self, model_name: str) -> Dict[str, Any]:
        """Auto-detect model type based on model name."""
        model_name_lower = model_name.lower()

        if any(name in model_name_lower for name in ['gpt', 'gpt2', 'gpt-neo', 'gpt-j', 'codegen']):
            return {'type': 'gpt', 'total_layers': 12}
        elif any(name in model_name_lower for name in ['bert', 'roberta', 'distilbert', 'scibert', 'biobert']):
            return {'type': 'bert', 'total_layers': 12}
        elif any(name in model_name_lower for name in ['t5', 'flan-t5', 'codet5']):
            return {'type': 't5', 'total_layers': 6}
        elif any(name in model_name_lower for name in ['llama', 'alpaca', 'vicuna']):
            return {'type': 'llama', 'total_layers': 32}
        else:
            # Default to GPT-style
            logger.warning(f"Could not auto-detect type for {model_name}, defaulting to GPT")
            return {'type': 'gpt', 'total_layers': 12}

    def _auto_create_adapter(self, model_name: str, extraction_config: Dict[str, Any]) -> ModelAdapter:
        """Auto-create adapter based on model name patterns."""
        model_type_config = self._auto_detect_model_type(model_name)
        model_type = model_type_config['type']

        if model_type == 'gpt':
            return GPTAdapter(model_name, self.device, extraction_config)
        elif model_type == 'bert':
            return BERTAdapter(model_name, self.device, extraction_config)
        elif model_type == 't5':
            return T5Adapter(model_name, self.device, extraction_config)
        elif model_type == 'llama':
            return LlamaAdapter(model_name, self.device, extraction_config)
        else:
            # Fallback to GPT
            return GPTAdapter(model_name, self.device, extraction_config)

    def get_available_models(self) -> List[str]:
        """Get list of available preconfigured models."""
        models = self.models_config.get('models', {})
        return [config.get('name', key) for key, config in models.items()]

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model."""
        model_config = self._find_model_config(model_name)
        if model_config:
            return {
                'name': model_config.get('name', model_name),
                'type': model_config.get('type', 'unknown'),
                'total_layers': model_config.get('layers', {}).get('total_layers', 'unknown'),
                'attention_pattern': model_config.get('layers', {}).get('attention', ''),
                'mlp_pattern': model_config.get('layers', {}).get('mlp', '')
            }
        else:
            auto_config = self._auto_detect_model_type(model_name)
            return {
                'name': model_name,
                'type': auto_config.get('type', 'unknown'),
                'total_layers': auto_config.get('total_layers', 'unknown'),
                'attention_pattern': 'auto-detected',
                'mlp_pattern': 'auto-detected'
            }
