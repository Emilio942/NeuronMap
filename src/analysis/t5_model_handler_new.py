"""
T5 Model Family Handler for NeuronMap
Supports T5, UL2, Flan-T5 with encoder-decoder architecture analysis.
"""

import torch
import torch.nn as nn
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass
import re

from ..utils.config import get_config_manager
from .base_model_handler import BaseModelHandler, ModelConfig, ActivationResult

logger = logging.getLogger(__name__)


@dataclass
class T5ActivationResult(ActivationResult):
    """Extended activation result for T5 models with encoder-decoder specific data."""
    encoder_activations: Dict[str, torch.Tensor]
    decoder_activations: Dict[str, torch.Tensor]
    cross_attention_weights: Dict[str, torch.Tensor]
    encoder_hidden_states: List[torch.Tensor]
    decoder_hidden_states: List[torch.Tensor]
    position_bias: Optional[torch.Tensor] = None
    task_prefix: Optional[str] = None
    target_text: Optional[str] = None


class T5ModelHandler(BaseModelHandler):
    """
    Specialized handler for T5-family models with encoder-decoder architecture support.

    Supports:
    - T5 (small, base, large, xl, xxl)
    - UL2 (base, large)
    - Flan-T5 (small, base, large, xl, xxl)

    Features:
    - Encoder-decoder cross-attention analysis
    - Text-to-text format processing
    - Relative position embedding analysis
    - Task-specific activation patterns
    """

    T5_VARIANTS = {
        't5-small': {
            'd_model': 512,
            'num_layers': 6,
            'd_ff': 2048,
            'num_heads': 8,
            'max_length': 512,
            'vocab_size': 32128,
            'relative_attention_num_buckets': 32
        },
        't5-base': {
            'd_model': 768,
            'num_layers': 12,
            'd_ff': 3072,
            'num_heads': 12,
            'max_length': 512,
            'vocab_size': 32128,
            'relative_attention_num_buckets': 32
        },
        't5-large': {
            'd_model': 1024,
            'num_layers': 24,
            'd_ff': 4096,
            'num_heads': 16,
            'max_length': 512,
            'vocab_size': 32128,
            'relative_attention_num_buckets': 32
        },
        't5-xl': {
            'd_model': 2048,
            'num_layers': 24,
            'd_ff': 5120,
            'num_heads': 32,
            'max_length': 512,
            'vocab_size': 32128,
            'relative_attention_num_buckets': 32
        },
        't5-xxl': {
            'd_model': 4096,
            'num_layers': 24,
            'd_ff': 10240,
            'num_heads': 64,
            'max_length': 512,
            'vocab_size': 32128,
            'relative_attention_num_buckets': 32
        },
        'ul2-base': {
            'd_model': 768,
            'num_layers': 12,
            'd_ff': 2048,
            'num_heads': 12,
            'max_length': 512,
            'vocab_size': 32128,
            'unified_architecture': True,
            'prefix_lm': True
        },
        'flan-t5-small': {
            'd_model': 512,
            'num_layers': 6,
            'd_ff': 2048,
            'num_heads': 8,
            'max_length': 512,
            'vocab_size': 32128,
            'instruction_tuned': True
        },
        'flan-t5-base': {
            'd_model': 768,
            'num_layers': 12,
            'd_ff': 3072,
            'num_heads': 12,
            'max_length': 512,
            'vocab_size': 32128,
            'instruction_tuned': True
        },
        'flan-t5-large': {
            'd_model': 1024,
            'num_layers': 24,
            'd_ff': 4096,
            'num_heads': 16,
            'max_length': 512,
            'vocab_size': 32128,
            'instruction_tuned': True
        },
        'flan-t5-xl': {
            'd_model': 2048,
            'num_layers': 24,
            'd_ff': 5120,
            'num_heads': 32,
            'max_length': 512,
            'vocab_size': 32128,
            'instruction_tuned': True
        },
        'flan-t5-xxl': {
            'd_model': 4096,
            'num_layers': 24,
            'd_ff': 10240,
            'num_heads': 64,
            'max_length': 512,
            'vocab_size': 32128,
            'instruction_tuned': True
        }
    }

    # Common T5 task prefixes for automatic detection
    TASK_PREFIXES = {
        'translation': ['translate English to', 'translate German to', 'translate to'],
        'summarization': ['summarize:', 'summarize this:', 'summary:'],
        'question_answering': ['question:', 'answer the question:', 'qa:'],
        'text_classification': ['classify:', 'sentiment:'],
        'paraphrasing': ['paraphrase:', 'rephrase:'],
        'text_generation': ['generate:', 'continue:'],
        'code_generation': ['code:', 'python:', 'javascript:'],
        'reasoning': ['reason:', 'explain:', 'because:']
    }

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize T5 model handler."""
        super().__init__(model_name, config)
        self.encoder_hooks = {}
        self.decoder_hooks = {}
        self.cross_attention_hooks = {}

    def _get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for T5 model variant."""
        # Normalize model name
        normalized_name = self._normalize_model_name(model_name)

        if normalized_name not in self.T5_VARIANTS:
            logger.warning(f"Unknown T5 variant: {model_name}, using t5-base config")
            normalized_name = 't5-base'

        variant_config = self.T5_VARIANTS[normalized_name]

        return ModelConfig(
            model_name=model_name,
            d_model=variant_config['d_model'],
            num_layers=variant_config['num_layers'],
            num_heads=variant_config['num_heads'],
            max_length=variant_config['max_length'],
            vocab_size=variant_config.get('vocab_size'),
            architecture_type="encoder-decoder",
            special_features={
                'encoder_decoder': True,
                'relative_attention': True,
                'instruction_tuned': variant_config.get('instruction_tuned', False),
                'unified_architecture': variant_config.get('unified_architecture', False),
                'relative_attention_num_buckets': variant_config.get('relative_attention_num_buckets', 32),
                'd_ff': variant_config.get('d_ff'),
                'prefix_lm': variant_config.get('prefix_lm', False)
            }
        )

    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name for config lookup."""
        # Handle HuggingFace model names
        if 'google/' in model_name:
            model_name = model_name.replace('google/', '')

        # Normalize common variations
        model_name = model_name.lower()

        # Map common variations to standard names
        name_mappings = {
            't5_small': 't5-small',
            't5_base': 't5-base',
            't5_large': 't5-large',
            'flan_t5_small': 'flan-t5-small',
            'flan_t5_base': 'flan-t5-base',
            'flan_t5_large': 'flan-t5-large',
        }

        return name_mappings.get(model_name, model_name)

    def load_model(self,
                   trust_remote_code: bool = False,
                   torch_dtype: torch.dtype = torch.float32,
                   device_map: Optional[str] = None,
                   **kwargs) -> bool:
        """
        Load T5 model and tokenizer.

        Args:
            trust_remote_code: Whether to trust remote code
            torch_dtype: Data type for model weights
            device_map: Device mapping for model sharding
            **kwargs: Additional arguments for model loading

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Loading T5 model: {self.model_name}")

            # Load tokenizer
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=trust_remote_code
                )
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=trust_remote_code
                )

            # Load model with memory optimization if needed
            model_kwargs = {
                'torch_dtype': torch_dtype,
                'trust_remote_code': trust_remote_code,
                **kwargs
            }

            if device_map:
                model_kwargs['device_map'] = device_map

            try:
                self.model = T5ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            except Exception:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )

            # Move to device if not using device_map
            if not device_map:
                self.model.to(self.device)

            self.model.eval()
            self.is_loaded = True

            logger.info(f"Successfully loaded T5 model: {self.model_name}")
            logger.info(f"Model info: {self.get_model_info()}")

            return True

        except Exception as e:
            logger.error(f"Failed to load T5 model {self.model_name}: {str(e)}")
            return False

    def detect_task_prefix(self, input_text: str) -> Optional[str]:
        """
        Detect T5 task prefix from input text.

        Args:
            input_text: Input text to analyze

        Returns:
            Detected task type or None
        """
        input_lower = input_text.lower().strip()

        for task_type, prefixes in self.TASK_PREFIXES.items():
            for prefix in prefixes:
                if input_lower.startswith(prefix.lower()):
                    logger.debug(f"Detected task type: {task_type} from prefix: {prefix}")
                    return task_type

        # Try to detect from pattern matching
        if re.search(r'\btranslate\b.*\bto\b', input_lower):
            return 'translation'
        elif re.search(r'\bsummariz\w*\b', input_lower):
            return 'summarization'
        elif input_lower.endswith('?'):
            return 'question_answering'

        return None

    def extract_activations(
        self,
        input_text: str,
        target_text: Optional[str] = None,
        layer_indices: Optional[List[int]] = None,
        return_attention: bool = True,
        max_new_tokens: int = 50
    ) -> T5ActivationResult:
        """
        Extract activations from T5 model with encoder-decoder analysis.

        Args:
            input_text: Input text for encoding
            target_text: Optional target text for decoding
            layer_indices: Specific layers to analyze
            return_attention: Whether to return attention weights
            max_new_tokens: Maximum tokens to generate if no target provided

        Returns:
            T5ActivationResult with comprehensive activation data
        """
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded first")

        self._validate_input(input_text)

        # Detect task prefix
        task_prefix = self.detect_task_prefix(input_text)

        # Prepare inputs
        encoder_inputs = self._prepare_inputs(input_text)

        # Prepare decoder inputs if target text provided
        decoder_inputs = None
        if target_text:
            decoder_inputs = self._prepare_inputs(target_text)

        # Storage for activations
        encoder_activations = {}
        decoder_activations = {}
        cross_attention_weights = {}
        encoder_hidden_states = []
        decoder_hidden_states = []

        # Register hooks for encoder
        encoder_hooks = self._register_encoder_hooks(
            encoder_activations,
            encoder_hidden_states,
            layer_indices
        )

        # Register hooks for decoder and cross-attention
        decoder_hooks = self._register_decoder_hooks(
            decoder_activations,
            decoder_hidden_states,
            cross_attention_weights,
            layer_indices,
            return_attention
        )

        try:
            with torch.no_grad():
                if target_text:
                    # Teacher forcing mode with target
                    outputs = self.model(
                        input_ids=encoder_inputs['input_ids'],
                        attention_mask=encoder_inputs.get('attention_mask'),
                        decoder_input_ids=decoder_inputs['input_ids'],
                        decoder_attention_mask=decoder_inputs.get('attention_mask'),
                        output_attentions=return_attention,
                        output_hidden_states=True,
                        return_dict=True
                    )
                else:
                    # Generation mode
                    outputs = self.model.generate(
                        input_ids=encoder_inputs['input_ids'],
                        attention_mask=encoder_inputs.get('attention_mask'),
                        max_new_tokens=max_new_tokens,
                        output_attentions=return_attention,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                        do_sample=False,
                        num_beams=1
                    )

        finally:
            # Clean up hooks
            for hook in encoder_hooks.values():
                hook.remove()
            for hook in decoder_hooks.values():
                hook.remove()

        # Extract position bias if available
        position_bias = None
        if hasattr(outputs, 'encoder_attentions') and outputs.encoder_attentions:
            # T5 stores position bias in the first attention layer
            first_encoder_attention = outputs.encoder_attentions[0]
            if hasattr(first_encoder_attention, 'position_bias'):
                position_bias = first_encoder_attention.position_bias

        # Create result
        result = T5ActivationResult(
            layer_activations={**encoder_activations, **decoder_activations},
            attention_weights=cross_attention_weights,
            hidden_states={'encoder': encoder_hidden_states, 'decoder': decoder_hidden_states},
            metadata={
                'model_name': self.model_name,
                'task_prefix': task_prefix,
                'encoder_layers': len(encoder_hidden_states),
                'decoder_layers': len(decoder_hidden_states),
                'cross_attention_layers': len(cross_attention_weights),
                'input_length': encoder_inputs['input_ids'].size(1),
                'target_length': decoder_inputs['input_ids'].size(1) if decoder_inputs else None,
                'architecture': 'encoder-decoder',
                'relative_attention': True
            },
            input_ids=encoder_inputs['input_ids'],
            input_text=input_text,
            encoder_activations=encoder_activations,
            decoder_activations=decoder_activations,
            cross_attention_weights=cross_attention_weights,
            encoder_hidden_states=encoder_hidden_states,
            decoder_hidden_states=decoder_hidden_states,
            position_bias=position_bias,
            task_prefix=task_prefix,
            target_text=target_text
        )

        return result

    def _register_encoder_hooks(
        self,
        activations_dict: Dict[str, torch.Tensor],
        hidden_states_list: List[torch.Tensor],
        layer_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Register hooks for encoder layers."""
        hooks = {}

        def encoder_hook(name, layer_idx):
            def hook(module, input, output):
                # Store layer activation
                if isinstance(output, tuple):
                    hidden_state = output[0]
                else:
                    hidden_state = output

                activations_dict[f"encoder_layer_{layer_idx}"] = hidden_state.detach().cpu()

                # Store in sequential list
                if len(hidden_states_list) <= layer_idx:
                    hidden_states_list.extend([None] * (layer_idx - len(hidden_states_list) + 1))
                hidden_states_list[layer_idx] = hidden_state.detach().cpu()

            return hook

        # Register on encoder layers
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'block'):
            encoder_layers = self.model.encoder.block
            target_indices = layer_indices if layer_indices else range(len(encoder_layers))

            for i in target_indices:
                if i < len(encoder_layers):
                    hook_handle = encoder_layers[i].register_forward_hook(
                        encoder_hook(f"encoder_{i}", i)
                    )
                    hooks[f"encoder_{i}"] = hook_handle

        return hooks

    def _register_decoder_hooks(
        self,
        activations_dict: Dict[str, torch.Tensor],
        hidden_states_list: List[torch.Tensor],
        cross_attention_dict: Dict[str, torch.Tensor],
        layer_indices: Optional[List[int]] = None,
        return_attention: bool = True
    ) -> Dict[str, Any]:
        """Register hooks for decoder layers and cross-attention."""
        hooks = {}

        def decoder_hook(name, layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_state = output[0]
                    if return_attention and len(output) > 1:
                        # Store cross-attention if available
                        attention_weights = output[1]
                        if attention_weights is not None:
                            cross_attention_dict[f"cross_attention_layer_{layer_idx}"] = attention_weights.detach().cpu()
                else:
                    hidden_state = output

                activations_dict[f"decoder_layer_{layer_idx}"] = hidden_state.detach().cpu()

                # Store in sequential list
                if len(hidden_states_list) <= layer_idx:
                    hidden_states_list.extend([None] * (layer_idx - len(hidden_states_list) + 1))
                hidden_states_list[layer_idx] = hidden_state.detach().cpu()

            return hook

        # Register on decoder layers
        if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'block'):
            decoder_layers = self.model.decoder.block
            target_indices = layer_indices if layer_indices else range(len(decoder_layers))

            for i in target_indices:
                if i < len(decoder_layers):
                    hook_handle = decoder_layers[i].register_forward_hook(
                        decoder_hook(f"decoder_{i}", i)
                    )
                    hooks[f"decoder_{i}"] = hook_handle

        return hooks

    def analyze_encoder_decoder_flow(
        self,
        input_text: str,
        target_text: str,
        layer_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Analyze information flow between encoder and decoder.

        Args:
            input_text: Source text
            target_text: Target text
            layer_indices: Specific layers to analyze

        Returns:
            Analysis results with cross-attention patterns and flow metrics
        """
        # Extract activations
        result = self.extract_activations(
            input_text,
            target_text,
            layer_indices,
            return_attention=True
        )

        analysis = {
            'input_text': input_text,
            'target_text': target_text,
            'task_prefix': result.task_prefix,
            'cross_attention_analysis': {},
            'information_flow_metrics': {},
            'attention_patterns': {}
        }

        # Analyze cross-attention patterns
        for layer_name, attention_weights in result.cross_attention_weights.items():
            layer_analysis = self._analyze_cross_attention_layer(
                attention_weights,
                input_text,
                target_text
            )
            analysis['cross_attention_analysis'][layer_name] = layer_analysis

        # Calculate information flow metrics
        analysis['information_flow_metrics'] = self._calculate_information_flow(
            result.encoder_hidden_states,
            result.decoder_hidden_states,
            result.cross_attention_weights
        )

        # Analyze attention patterns
        analysis['attention_patterns'] = self._analyze_attention_patterns(
            result.cross_attention_weights
        )

        return analysis

    def _analyze_cross_attention_layer(
        self,
        attention_weights: torch.Tensor,
        input_text: str,
        target_text: str
    ) -> Dict[str, Any]:
        """Analyze cross-attention patterns for a single layer."""
        # attention_weights shape: [batch, num_heads, tgt_len, src_len]

        if attention_weights.dim() == 4:
            batch_size, num_heads, tgt_len, src_len = attention_weights.shape

            # Average across batch and heads for analysis
            avg_attention = attention_weights.mean(dim=(0, 1))  # [tgt_len, src_len]

            analysis = {
                'shape': list(attention_weights.shape),
                'num_heads': num_heads,
                'source_length': src_len,
                'target_length': tgt_len,
                'attention_entropy': float(self._calculate_attention_entropy(avg_attention)),
                'attention_concentration': float(avg_attention.max()),
                'attention_distribution': avg_attention.numpy().tolist(),
                'head_specialization': self._analyze_head_specialization(attention_weights)
            }

            return analysis

        return {'error': f'Unexpected attention shape: {attention_weights.shape}'}

    def _calculate_attention_entropy(self, attention_matrix: torch.Tensor) -> float:
        """Calculate entropy of attention distribution."""
        # Flatten and normalize
        flat_attention = attention_matrix.flatten()
        flat_attention = flat_attention / flat_attention.sum()

        # Calculate entropy
        epsilon = 1e-10
        entropy = -(flat_attention * torch.log(flat_attention + epsilon)).sum()

        return entropy.item()

    def _analyze_head_specialization(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """Analyze specialization patterns across attention heads."""
        if attention_weights.dim() != 4:
            return {}

        batch_size, num_heads, tgt_len, src_len = attention_weights.shape

        # Analyze each head separately
        head_entropies = []
        head_concentrations = []

        for head_idx in range(num_heads):
            head_attention = attention_weights[0, head_idx]  # Take first batch

            # Calculate entropy for this head
            entropy = self._calculate_attention_entropy(head_attention)
            head_entropies.append(entropy)

            # Calculate concentration (max attention value)
            concentration = float(head_attention.max())
            head_concentrations.append(concentration)

        return {
            'head_entropies': head_entropies,
            'head_concentrations': head_concentrations,
            'entropy_variance': float(np.var(head_entropies)),
            'concentration_variance': float(np.var(head_concentrations)),
            'most_focused_head': int(np.argmax(head_concentrations)),
            'most_distributed_head': int(np.argmax(head_entropies))
        }

    def _calculate_information_flow(
        self,
        encoder_states: List[torch.Tensor],
        decoder_states: List[torch.Tensor],
        cross_attention: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Calculate information flow metrics between encoder and decoder."""
        flow_metrics = {
            'encoder_decoder_similarity': {},
            'layer_wise_flow': {},
            'bottleneck_analysis': {}
        }

        # Calculate layer-wise similarities
        min_layers = min(len(encoder_states), len(decoder_states))

        for i in range(min_layers):
            if encoder_states[i] is not None and decoder_states[i] is not None:
                # Calculate cosine similarity between corresponding layers
                enc_state = encoder_states[i].flatten()
                dec_state = decoder_states[i].flatten()

                similarity = torch.cosine_similarity(
                    enc_state.unsqueeze(0),
                    dec_state.unsqueeze(0)
                )

                flow_metrics['encoder_decoder_similarity'][f'layer_{i}'] = float(similarity)

        # Analyze attention-based flow
        for layer_name, attention in cross_attention.items():
            if attention.dim() == 4:
                # Sum attention across heads and batch
                flow_strength = attention.sum(dim=(0, 1)).mean(dim=0)  # Average over target positions
                flow_metrics['layer_wise_flow'][layer_name] = {
                    'mean_flow': float(flow_strength.mean()),
                    'max_flow': float(flow_strength.max()),
                    'flow_concentration': float(flow_strength.std())
                }

        return flow_metrics

    def _analyze_attention_patterns(
        self,
        cross_attention: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Analyze attention patterns across layers."""
        patterns = {
            'layer_patterns': {},
            'global_patterns': {}
        }

        all_attentions = []

        for layer_name, attention in cross_attention.items():
            if attention.dim() == 4:
                # Average across batch and heads
                avg_attention = attention.mean(dim=(0, 1))
                all_attentions.append(avg_attention)

                # Analyze this layer's patterns
                patterns['layer_patterns'][layer_name] = {
                    'diagonal_strength': self._measure_diagonal_attention(avg_attention),
                    'dispersion': float(avg_attention.std()),
                    'max_attention_position': avg_attention.argmax().item()
                }

        # Global pattern analysis
        if all_attentions:
            stacked_attentions = torch.stack(all_attentions)
            patterns['global_patterns'] = {
                'layer_consistency': float(torch.std(stacked_attentions, dim=0).mean()),
                'evolution_trend': self._analyze_layer_evolution(stacked_attentions)
            }

        return patterns

    def _measure_diagonal_attention(self, attention_matrix: torch.Tensor) -> float:
        """Measure strength of diagonal attention pattern."""
        min_dim = min(attention_matrix.shape)
        diagonal_sum = torch.diagonal(attention_matrix[:min_dim, :min_dim]).sum()
        total_sum = attention_matrix.sum()

        return float(diagonal_sum / total_sum) if total_sum > 0 else 0.0

    def _analyze_layer_evolution(self, stacked_attentions: torch.Tensor) -> Dict[str, float]:
        """Analyze how attention patterns evolve across layers."""
        num_layers = stacked_attentions.shape[0]

        if num_layers < 2:
            return {}

        # Calculate layer-to-layer changes
        layer_changes = []
        for i in range(1, num_layers):
            change = torch.norm(stacked_attentions[i] - stacked_attentions[i-1])
            layer_changes.append(float(change))

        return {
            'mean_change': float(np.mean(layer_changes)),
            'max_change': float(np.max(layer_changes)),
            'change_variance': float(np.var(layer_changes)),
            'convergence_trend': float(np.polyfit(range(len(layer_changes)), layer_changes, 1)[0])
        }


# Register T5 handler with the factory
from .base_model_handler import ModelFactory
ModelFactory.register_handler('t5', T5ModelHandler)
