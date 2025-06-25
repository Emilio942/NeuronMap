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

    # __init__ method defined below with more comprehensive implementation

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

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize T5 model handler.

        Args:
            model_name: Name of the T5 model variant
            config: Optional configuration override
        """
        super().__init__(model_name, config)

        if model_name not in self.T5_VARIANTS:
            raise ValueError(f"Unsupported T5 model: {model_name}. "
                           f"Supported models: {list(self.T5_VARIANTS.keys())}")

        self.model_config = self.T5_VARIANTS[model_name].copy()
        if config:
            self.model_config.update(config)

        self.model = None
        self.tokenizer = None
        self.device = None

        # Activation capture storage
        self.encoder_activations = {}
        self.decoder_activations = {}
        self.cross_attention_matrices = {}

        logger.info(f"Initialized T5ModelHandler for {model_name}")

    # def load_model(self, device: str = "auto") -> Tuple[nn.Module, Any]:
    #     """
    #     Load T5 model and tokenizer with memory optimization.
    #     (Commented out to avoid F811 redefinition error - use first load_model method)
    #     """
    #     pass

    # Note: get_layer_names method is defined below with correct implementation

    def get_layer_names(self) -> Dict[str, List[str]]:
        """
        Get available layer names for T5 encoder-decoder architecture.

        Returns:
            Dictionary with encoder and decoder layer names
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        encoder_layers = []
        decoder_layers = []
        cross_attention_layers = []

        # Encoder layers
        for i in range(self.model_config['num_layers']):
            encoder_layers.extend([
                f"encoder.block.{i}.layer.0.SelfAttention",  # Self-attention
                f"encoder.block.{i}.layer.1.DenseReluDense",  # Feed-forward
                f"encoder.block.{i}.layer.0.layer_norm",     # Layer norm
                f"encoder.block.{i}.layer.1.layer_norm"      # Layer norm
            ])

        # Decoder layers
        for i in range(self.model_config['num_layers']):
            decoder_layers.extend([
                f"decoder.block.{i}.layer.0.SelfAttention",      # Self-attention
                f"decoder.block.{i}.layer.1.EncDecAttention",    # Cross-attention
                f"decoder.block.{i}.layer.2.DenseReluDense",     # Feed-forward
                f"decoder.block.{i}.layer.0.layer_norm",         # Layer norm
                f"decoder.block.{i}.layer.1.layer_norm",         # Layer norm
                f"decoder.block.{i}.layer.2.layer_norm"          # Layer norm
            ])

            # Cross-attention layers for analysis
            cross_attention_layers.append(f"decoder.block.{i}.layer.1.EncDecAttention")

        return {
            'encoder': encoder_layers,
            'decoder': decoder_layers,
            'cross_attention': cross_attention_layers
        }

    def extract_activations(self,
                          input_texts: Union[str, List[str]],
                          target_texts: Optional[Union[str, List[str]]] = None,
                          layer_names: Optional[List[str]] = None,
                          include_cross_attention: bool = True) -> Dict[str, torch.Tensor]:
        """
        Extract activations from T5 encoder-decoder model.

        Args:
            input_texts: Input text(s) for encoding
            target_texts: Target text(s) for decoding (optional for encoder-only analysis)
            layer_names: Specific layers to extract from
            include_cross_attention: Whether to extract cross-attention matrices

        Returns:
            Dictionary of layer activations
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if isinstance(input_texts, str):
            input_texts = [input_texts]

        if target_texts is not None and isinstance(target_texts, str):
            target_texts = [target_texts]

        # Get all layer names if not specified
        if layer_names is None:
            all_layers = self.get_layer_names()
            layer_names = all_layers['encoder'][:6] + all_layers['decoder'][:6]  # Subset for efficiency

        # Clear activation storage
        self.encoder_activations.clear()
        self.decoder_activations.clear()
        self.cross_attention_matrices.clear()

        # Register hooks for specified layers
        hooks = []
        for layer_name in layer_names:
            layer = self._get_layer_by_name(layer_name)
            if layer is not None:
                hook = layer.register_forward_hook(
                    lambda module, input, output, name=layer_name:
                    self._activation_hook(module, input, output, name)
                )
                hooks.append(hook)

        # Register cross-attention hooks if requested
        if include_cross_attention:
            cross_attn_layers = self.get_layer_names()['cross_attention'][:6]  # Subset
            for layer_name in cross_attn_layers:
                layer = self._get_layer_by_name(layer_name)
                if layer is not None:
                    hook = layer.register_forward_hook(
                        lambda module, input, output, name=layer_name:
                        self._cross_attention_hook(module, input, output, name)
                    )
                    hooks.append(hook)

        activations = {}

        try:
            for i, input_text in enumerate(input_texts):
                # Prepare inputs
                input_ids = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=self.model_config['max_length'],
                    truncation=True,
                    padding=True
                ).input_ids.to(self.device)

                # Prepare decoder inputs if targets provided
                if target_texts is not None and i < len(target_texts):
                    # For training-style forward pass with targets
                    target_ids = self.tokenizer(
                        target_texts[i],
                        return_tensors="pt",
                        max_length=self.model_config['max_length'],
                        truncation=True,
                        padding=True
                    ).input_ids.to(self.device)

                    # Shift targets for decoder input
                    decoder_input_ids = torch.cat([
                        torch.zeros((target_ids.shape[0], 1), dtype=torch.long, device=self.device),
                        target_ids[:, :-1]
                    ], dim=1)

                    # Forward pass
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=input_ids,
                            decoder_input_ids=decoder_input_ids,
                            labels=target_ids,
                            output_attentions=include_cross_attention
                        )
                else:
                    # For generation-style forward pass (encoder only or generate)
                    with torch.no_grad():
                        # Just encode for encoder activations
                        encoder_outputs = self.model.encoder(input_ids=input_ids)

                        # Generate a short sequence to get decoder activations
                        generated = self.model.generate(
                            input_ids=input_ids,
                            max_length=min(50, self.model_config['max_length']),
                            num_beams=1,
                            do_sample=False,
                            output_attentions=include_cross_attention,
                            return_dict_in_generate=True
                        )

                # Collect activations for this input
                for layer_name in layer_names:
                    if layer_name in self.encoder_activations:
                        key = f"{layer_name}_sample_{i}"
                        activations[key] = self.encoder_activations[layer_name].clone()
                    elif layer_name in self.decoder_activations:
                        key = f"{layer_name}_sample_{i}"
                        activations[key] = self.decoder_activations[layer_name].clone()

                # Collect cross-attention matrices
                if include_cross_attention:
                    for layer_name, attn_matrix in self.cross_attention_matrices.items():
                        key = f"{layer_name}_cross_attn_sample_{i}"
                        activations[key] = attn_matrix.clone()

                # Clear for next iteration
                self.encoder_activations.clear()
                self.decoder_activations.clear()
                self.cross_attention_matrices.clear()

        finally:
            # Remove all hooks
            for hook in hooks:
                hook.remove()

        logger.info(f"Extracted activations from {len(layer_names)} layers for {len(input_texts)} inputs")
        return activations

    def analyze_encoder_decoder_flow(self,
                                   input_text: str,
                                   target_text: str) -> Dict[str, Any]:
        """
        Analyze information flow between encoder and decoder.

        Args:
            input_text: Input text for encoding
            target_text: Target text for decoding

        Returns:
            Analysis results including cross-attention patterns
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Extract activations with cross-attention
        activations = self.extract_activations(
            input_text,
            target_text,
            include_cross_attention=True
        )

        # Analyze cross-attention patterns
        cross_attn_analysis = self._analyze_cross_attention_patterns(activations)

        # Analyze encoder-decoder information flow
        flow_analysis = self._analyze_information_flow(activations)

        # Analyze position dependencies
        position_analysis = self._analyze_position_dependencies(activations)

        return {
            'cross_attention_patterns': cross_attn_analysis,
            'information_flow': flow_analysis,
            'position_dependencies': position_analysis,
            'model_config': self.model_config
        }

    def detect_task_prefix(self, input_text: str) -> Dict[str, Any]:
        """
        Detect and analyze task prefixes in T5-style text-to-text format.

        Args:
            input_text: Input text potentially containing task prefix

        Returns:
            Task detection results
        """
        # Common T5 task prefixes
        task_prefixes = {
            'translate': ['translate English to German:', 'translate German to English:',
                         'translate to'],
            'summarize': ['summarize:', 'summarization:'],
            'question_answering': ['question:', 'answer the question:'],
            'sentiment': ['sentiment:', 'sentiment analysis:'],
            'classification': ['classify:', 'classification:'],
            'generation': ['generate:', 'complete:'],
            'cola': ['cola sentence:'],
            'stsb': ['stsb sentence'],
            'rte': ['rte premise:', 'rte hypothesis:'],
            'wnli': ['wnli sentence']
        }

        detected_tasks = []
        confidence_scores = {}

        input_lower = input_text.lower()

        for task_type, prefixes in task_prefixes.items():
            for prefix in prefixes:
                if input_lower.startswith(prefix.lower()):
                    detected_tasks.append(task_type)
                    confidence_scores[task_type] = 1.0  # Exact match
                    break
                elif prefix.lower() in input_lower:
                    if task_type not in detected_tasks:
                        detected_tasks.append(task_type)
                        confidence_scores[task_type] = 0.5  # Partial match

        # If instruction tuned model, check for instruction patterns
        if self.model_config.get('instruction_tuned', False):
            instruction_patterns = [
                'please', 'can you', 'help me', 'explain', 'describe',
                'what is', 'how to', 'why', 'when', 'where'
            ]

            for pattern in instruction_patterns:
                if pattern in input_lower:
                    if 'instruction_following' not in detected_tasks:
                        detected_tasks.append('instruction_following')
                        confidence_scores['instruction_following'] = 0.7
                    break

        return {
            'detected_tasks': detected_tasks,
            'confidence_scores': confidence_scores,
            'is_instruction_tuned': self.model_config.get('instruction_tuned', False),
            'original_text': input_text
        }

    def _get_layer_by_name(self, layer_name: str) -> Optional[nn.Module]:
        """Get layer module by name."""
        try:
            parts = layer_name.split('.')
            layer = self.model
            for part in parts:
                layer = getattr(layer, part)
            return layer
        except AttributeError:
            logger.warning(f"Layer not found: {layer_name}")
            return None

    def _activation_hook(self, module: nn.Module, input: Tuple, output: torch.Tensor, name: str):
        """Hook to capture layer activations."""
        if isinstance(output, tuple):
            activation = output[0]  # Take first element if tuple
        else:
            activation = output

        # Store activation (detached from computation graph)
        if 'encoder' in name:
            self.encoder_activations[name] = activation.detach().cpu()
        elif 'decoder' in name:
            self.decoder_activations[name] = activation.detach().cpu()

    def _cross_attention_hook(self, module: nn.Module, input: Tuple, output: torch.Tensor, name: str):
        """Hook to capture cross-attention matrices."""
        if isinstance(output, tuple) and len(output) > 1:
            # T5 attention modules return (output, attention_weights)
            attention_weights = output[1]
            if attention_weights is not None:
                self.cross_attention_matrices[name] = attention_weights.detach().cpu()

    def _analyze_cross_attention_patterns(self, activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze cross-attention patterns between encoder and decoder."""
        cross_attn_layers = [k for k in activations.keys() if 'cross_attn' in k]

        if not cross_attn_layers:
            return {'error': 'No cross-attention matrices found'}

        patterns = {}

        for layer_key in cross_attn_layers:
            attn_matrix = activations[layer_key]

            # Analyze attention patterns
            patterns[layer_key] = {
                'attention_entropy': self._compute_attention_entropy(attn_matrix),
                'attention_concentration': self._compute_attention_concentration(attn_matrix),
                'max_attention_positions': self._find_max_attention_positions(attn_matrix),
                'attention_distribution': self._analyze_attention_distribution(attn_matrix)
            }

        return patterns

    def _analyze_information_flow(self, activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze information flow from encoder to decoder."""
        encoder_acts = {k: v for k, v in activations.items() if 'encoder' in k}
        decoder_acts = {k: v for k, v in activations.items() if 'decoder' in k}

        if not encoder_acts or not decoder_acts:
            return {'error': 'Insufficient activations for flow analysis'}

        # Compute information flow metrics
        flow_metrics = {
            'encoder_decoder_similarity': self._compute_encoder_decoder_similarity(
                encoder_acts, decoder_acts
            ),
            'information_bottleneck': self._analyze_information_bottleneck(
                encoder_acts, decoder_acts
            ),
            'layer_wise_influence': self._compute_layer_wise_influence(
                encoder_acts, decoder_acts
            )
        }

        return flow_metrics

    def _analyze_position_dependencies(self, activations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze relative position embedding effects."""
        # This is a simplified analysis - full implementation would require
        # access to position embeddings and bias matrices

        position_analysis = {
            'sequence_length_effects': self._analyze_sequence_length_effects(activations),
            'position_bias_patterns': self._analyze_position_bias_patterns(activations),
            'relative_position_influence': self._compute_relative_position_influence(activations)
        }

        return position_analysis

    # Helper methods for analysis (simplified implementations)

    def _compute_attention_entropy(self, attention_matrix: torch.Tensor) -> float:
        """Compute entropy of attention distribution."""
        # Flatten attention weights and compute entropy
        attn_flat = attention_matrix.flatten()
        attn_flat = attn_flat / attn_flat.sum()
        entropy = -(attn_flat * torch.log(attn_flat + 1e-8)).sum()
        return entropy.item()

    def _compute_attention_concentration(self, attention_matrix: torch.Tensor) -> float:
        """Compute how concentrated the attention is."""
        attn_flat = attention_matrix.flatten()
        max_attn = attn_flat.max()
        mean_attn = attn_flat.mean()
        return (max_attn / mean_attn).item()

    def _find_max_attention_positions(self, attention_matrix: torch.Tensor) -> List[Tuple[int, int]]:
        """Find positions with maximum attention."""
        # Simplified: return top 5 attention positions
        attn_2d = attention_matrix.squeeze()
        flat_indices = torch.topk(attn_2d.flatten(), k=5).indices
        positions = [(idx.item() // attn_2d.shape[1], idx.item() % attn_2d.shape[1])
                    for idx in flat_indices]
        return positions

    def _analyze_attention_distribution(self, attention_matrix: torch.Tensor) -> Dict[str, float]:
        """Analyze distribution properties of attention."""
        attn_flat = attention_matrix.flatten()
        return {
            'mean': attn_flat.mean().item(),
            'std': attn_flat.std().item(),
            'max': attn_flat.max().item(),
            'min': attn_flat.min().item()
        }

    def _compute_encoder_decoder_similarity(self, encoder_acts: Dict, decoder_acts: Dict) -> float:
        """Compute similarity between encoder and decoder representations."""
        # Simplified cosine similarity computation
        if not encoder_acts or not decoder_acts:
            return 0.0

        # Take first available activations from each
        enc_act = list(encoder_acts.values())[0].mean(dim=1).flatten()
        dec_act = list(decoder_acts.values())[0].mean(dim=1).flatten()

        # Ensure same dimension
        min_dim = min(enc_act.shape[0], dec_act.shape[0])
        enc_act = enc_act[:min_dim]
        dec_act = dec_act[:min_dim]

        similarity = torch.cosine_similarity(enc_act, dec_act, dim=0)
        return similarity.item()

    def _analyze_information_bottleneck(self, encoder_acts: Dict, decoder_acts: Dict) -> Dict[str, float]:
        """Analyze information bottleneck between encoder and decoder."""
        return {
            'encoder_variance': self._compute_activation_variance(encoder_acts),
            'decoder_variance': self._compute_activation_variance(decoder_acts),
            'information_compression': 0.5  # Placeholder
        }

    def _compute_layer_wise_influence(self, encoder_acts: Dict, decoder_acts: Dict) -> Dict[str, float]:
        """Compute influence of encoder layers on decoder."""
        influences = {}

        for enc_layer, enc_act in encoder_acts.items():
            layer_num = self._extract_layer_number(enc_layer)
            if layer_num is not None:
                # Simplified influence metric
                influences[f"encoder_layer_{layer_num}"] = enc_act.abs().mean().item()

        return influences

    def _analyze_sequence_length_effects(self, activations: Dict) -> Dict[str, Any]:
        """Analyze how sequence length affects activations."""
        return {
            'sequence_length_correlation': 0.0,  # Placeholder
            'position_dependent_variance': 0.0   # Placeholder
        }

    def _analyze_position_bias_patterns(self, activations: Dict) -> Dict[str, Any]:
        """Analyze position bias patterns in attention."""
        return {
            'local_attention_bias': 0.0,    # Placeholder
            'global_attention_bias': 0.0,   # Placeholder
            'distance_decay_pattern': 'linear'  # Placeholder
        }

    def _compute_relative_position_influence(self, activations: Dict) -> float:
        """Compute influence of relative positions."""
        return 0.5  # Placeholder

    def _compute_activation_variance(self, activations: Dict) -> float:
        """Compute variance across activations."""
        if not activations:
            return 0.0

        all_variances = []
        for act in activations.values():
            all_variances.append(act.var().item())

        return np.mean(all_variances)

    def _extract_layer_number(self, layer_name: str) -> Optional[int]:
        """Extract layer number from layer name."""
        import re
        match = re.search(r'\.(\d+)\.', layer_name)
        return int(match.group(1)) if match else None


def create_t5_handler(model_name: str, config: Optional[Dict[str, Any]] = None) -> T5ModelHandler:
    """
    Factory function to create T5 model handler.

    Args:
        model_name: Name of T5 model variant
        config: Optional configuration override

    Returns:
        Configured T5ModelHandler instance
    """
    return T5ModelHandler(model_name, config)


# Export for use in other modules
__all__ = ['T5ModelHandler', 'create_t5_handler']
