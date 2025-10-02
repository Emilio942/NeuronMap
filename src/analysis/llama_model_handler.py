"""
LLaMA Model Family Handler for NeuronMap
Supports LLaMA, Alpaca, Vicuna with large-scale autoregressive model analysis.
"""

import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM
)
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
import psutil

from .base_model_handler import BaseModelHandler, ModelConfig, ActivationResult, ModelFactory

logger = logging.getLogger(__name__)


@dataclass
class LlamaActivationResult(ActivationResult):
    """Extended activation result for LLaMA models with autoregressive specific data."""
    rms_norm_stats: Dict[str, torch.Tensor]
    instruction_attention_patterns: Optional[Dict[str, torch.Tensor]] = None
    conversation_state: Optional[Dict[str, Any]] = None
    memory_usage: Optional[Dict[str, float]] = None
    instruction_compliance_score: Optional[float] = None


class LlamaModelHandler(BaseModelHandler):
    """
    Specialized handler for LLaMA-family models with large-scale autoregressive analysis.

    Supports:
    - LLaMA (7B, 13B, 30B, 65B)
    - Alpaca (instruction-tuned variants)
    - Vicuna (conversation-tuned variants)

    Features:
    - Large-model memory optimization
    - RMS normalization analysis
    - Instruction-following behavior investigation
    - Multi-turn conversation state tracking
    """

    LLAMA_CONFIGS = {
        'llama-7b': {
            'hidden_size': 4096,
            'num_layers': 32,
            'num_heads': 32,
            'intermediate_size': 11008,
            'max_length': 2048,
            'vocab_size': 32000,
            'rms_norm': True,
            'rope': True,  # Rotary Position Embedding
            'approximate_size_gb': 13.5
        },
        'llama-13b': {
            'hidden_size': 5120,
            'num_layers': 40,
            'num_heads': 40,
            'intermediate_size': 13824,
            'max_length': 2048,
            'vocab_size': 32000,
            'rms_norm': True,
            'rope': True,
            'approximate_size_gb': 26.0
        },
        'llama-30b': {
            'hidden_size': 6656,
            'num_layers': 60,
            'num_heads': 52,
            'intermediate_size': 17920,
            'max_length': 2048,
            'vocab_size': 32000,
            'rms_norm': True,
            'rope': True,
            'approximate_size_gb': 60.0
        },
        'llama-65b': {
            'hidden_size': 8192,
            'num_layers': 80,
            'num_heads': 64,
            'intermediate_size': 22016,
            'max_length': 2048,
            'vocab_size': 32000,
            'rms_norm': True,
            'rope': True,
            'approximate_size_gb': 120.0
        },
        'alpaca-7b': {
            'base_model': 'llama-7b',
            'instruction_tuned': True,
            'training_method': 'instruction_following',
            'hidden_size': 4096,
            'num_layers': 32,
            'num_heads': 32,
            'max_length': 2048,
            'vocab_size': 32000,
            'rms_norm': True,
            'rope': True,
            'approximate_size_gb': 13.5
        },
        'alpaca-13b': {
            'base_model': 'llama-13b',
            'instruction_tuned': True,
            'training_method': 'instruction_following',
            'hidden_size': 5120,
            'num_layers': 40,
            'num_heads': 40,
            'max_length': 2048,
            'vocab_size': 32000,
            'rms_norm': True,
            'rope': True,
            'approximate_size_gb': 26.0
        },
        'vicuna-7b': {
            'base_model': 'llama-7b',
            'conversation_tuned': True,
            'training_method': 'conversation_optimization',
            'hidden_size': 4096,
            'num_layers': 32,
            'num_heads': 32,
            'max_length': 2048,
            'vocab_size': 32000,
            'rms_norm': True,
            'rope': True,
            'approximate_size_gb': 13.5
        },
        'vicuna-13b': {
            'base_model': 'llama-13b',
            'conversation_tuned': True,
            'training_method': 'conversation_optimization',
            'hidden_size': 5120,
            'num_layers': 40,
            'num_heads': 40,
            'max_length': 2048,
            'vocab_size': 32000,
            'rms_norm': True,
            'rope': True,
            'approximate_size_gb': 26.0
        }
    }

    # Instruction patterns for analysis
    INSTRUCTION_PATTERNS = {
        'direct_instruction': [
            'please', 'can you', 'could you', 'would you',
            'explain', 'describe', 'tell me', 'help me'
        ],
        'question_pattern': [
            'what', 'how', 'why', 'when', 'where', 'who', 'which'
        ],
        'task_pattern': [
            'write', 'create', 'generate', 'analyze', 'summarize',
            'translate', 'code', 'solve', 'calculate'
        ],
        'conversational': [
            'hi', 'hello', 'thanks', 'thank you', 'bye', 'goodbye'
        ]
    }

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize LLaMA model handler."""
        super().__init__(model_name, config)
        self.rms_norm_hooks = {}
        self.memory_tracker = MemoryTracker()
        self.instruction_analyzer = InstructionAnalyzer()

    def _get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for LLaMA model variant."""
        # Normalize model name
        normalized_name = self._normalize_model_name(model_name)

        if normalized_name not in self.LLAMA_CONFIGS:
            logger.warning(f"Unknown LLaMA variant: {model_name}, using llama-7b config")
            normalized_name = 'llama-7b'

        variant_config = self.LLAMA_CONFIGS[normalized_name]

        return ModelConfig(
            model_name=model_name,
            d_model=variant_config['hidden_size'],
            num_layers=variant_config['num_layers'],
            num_heads=variant_config['num_heads'],
            max_length=variant_config['max_length'],
            vocab_size=variant_config.get('vocab_size'),
            architecture_type="autoregressive",
            special_features={
                'autoregressive': True,
                'rms_norm': variant_config.get('rms_norm', True),
                'rope': variant_config.get('rope', True),
                'instruction_tuned': variant_config.get('instruction_tuned', False),
                'conversation_tuned': variant_config.get('conversation_tuned', False),
                'training_method': variant_config.get('training_method', 'standard'),
                'base_model': variant_config.get('base_model'),
                'intermediate_size': variant_config.get('intermediate_size'),
                'approximate_size_gb': variant_config.get('approximate_size_gb', 13.5)
            }
        )

    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name for config lookup."""
        # Handle common model naming variations
        model_name = model_name.lower()

        # Map common variations to standard names
        name_mappings = {
            'llama_7b': 'llama-7b',
            'llama_13b': 'llama-13b',
            'llama_30b': 'llama-30b',
            'llama_65b': 'llama-65b',
            'alpaca_7b': 'alpaca-7b',
            'alpaca_13b': 'alpaca-13b',
            'vicuna_7b': 'vicuna-7b',
            'vicuna_13b': 'vicuna-13b',
        }

        # Handle HuggingFace model names
        if any(org in model_name for org in ['huggyllama/', 'meta-llama/', 'chavinlo/', 'lmsys/']):
            # Extract model variant from path
            for variant in self.LLAMA_CONFIGS.keys():
                if variant.replace('-', '') in model_name.replace('-', '').replace('_', ''):
                    return variant

        return name_mappings.get(model_name, model_name)

    def _load_tokenizer(self, model_name_hf: str, trust_remote_code: bool):
        try:
            self.tokenizer = LlamaTokenizer.from_pretrained(
                model_name_hf,
                trust_remote_code=trust_remote_code
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_hf,
                trust_remote_code=trust_remote_code
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model_from_pretrained(self, model_name_hf: str, model_kwargs: Dict[str, Any]):
        try:
            self.model = LlamaForCausalLM.from_pretrained(
                model_name_hf,
                **model_kwargs
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_hf,
                **model_kwargs
            )

    def load_model(self,
                   max_memory_gb: Optional[float] = None,
                   device_map: Optional[str] = "auto",
                   torch_dtype: torch.dtype = torch.float16,
                   trust_remote_code: bool = False,
                   use_gradient_checkpointing: bool = True,
                   **kwargs) -> bool:
        """Load LLaMA model with memory optimization."""
        try:
            logger.info(f"Loading LLaMA model: {self.model_name}")

            model_size_gb = self.model_config.special_features.get('approximate_size_gb', 13.5)
            available_memory = self._get_available_memory_gb()

            if max_memory_gb is None:
                max_memory_gb = available_memory * 0.8

            logger.info(f"Model size: {model_size_gb:.1f}GB, Available: {available_memory:.1f}GB, Max: {max_memory_gb:.1f}GB")

            if model_size_gb > max_memory_gb:
                logger.info("Model size exceeds available memory, using CPU offloading")
                device_map = self._create_memory_optimized_device_map(max_memory_gb, model_size_gb)

            model_name_hf = self.model_config.special_features.get('model_name_hf', self.model_name)
            self._load_tokenizer(model_name_hf, trust_remote_code)

            model_kwargs = {
                'torch_dtype': torch_dtype,
                'trust_remote_code': trust_remote_code,
                'low_cpu_mem_usage': True,
                **kwargs
            }

            if device_map:
                model_kwargs['device_map'] = device_map

            self._load_model_from_pretrained(model_name_hf, model_kwargs)

            if use_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()

            if not device_map or device_map == "cpu":
                self.model.to(self.device)

            self.model.eval()
            self.is_loaded = True

            self.memory_tracker.record_baseline()

            logger.info(f"Successfully loaded LLaMA model: {self.model_name}")
            logger.info(f"Memory usage: {self.memory_tracker.get_current_usage()}")

            return True

        except Exception as e:
            logger.error(f"Failed to load LLaMA model {self.model_name}: {str(e)}")
            return False

    def _get_available_memory_gb(self) -> float:
        """Get available GPU memory in GB."""
        if torch.cuda.is_available():
            # Get GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return gpu_memory
        else:
            # Get system RAM
            ram_memory = psutil.virtual_memory().total / (1024**3)
            return ram_memory * 0.5  # Conservative estimate for model loading

    def _create_memory_optimized_device_map(self, max_memory_gb: float, model_size_gb: float) -> Dict[str, str]:
        """Create device map for memory-constrained loading."""
        # Simple strategy: put as much as possible on GPU, rest on CPU
        gpu_layers = int((max_memory_gb / model_size_gb) * self.model_config.num_layers)

        device_map = {}

        # Place embedding layers on GPU
        device_map["model.embed_tokens"] = 0

        # Place first N layers on GPU
        for i in range(min(gpu_layers, self.model_config.num_layers)):
            device_map[f"model.layers.{i}"] = 0

        # Place remaining layers on CPU
        for i in range(gpu_layers, self.model_config.num_layers):
            device_map[f"model.layers.{i}"] = "cpu"

        # Place final layers strategically
        device_map["model.norm"] = 0 if gpu_layers > 0 else "cpu"
        device_map["lm_head"] = 0 if gpu_layers > 0 else "cpu"

        logger.info(f"Created device map: {gpu_layers}/{self.model_config.num_layers} layers on GPU")
        return device_map

    def extract_activations(
        self,
        input_text: str,
        target_text: Optional[str] = None,
        layer_indices: Optional[List[int]] = None,
        return_attention: bool = True,
        max_new_tokens: int = 50,
        analyze_instructions: bool = True
    ) -> LlamaActivationResult:
        """
        Extract activations from LLaMA model with autoregressive analysis.

        Args:
            input_text: Input text for processing
            target_text: Optional target text for teacher forcing
            layer_indices: Specific layers to analyze
            return_attention: Whether to return attention weights
            max_new_tokens: Maximum tokens to generate
            analyze_instructions: Whether to analyze instruction-following patterns

        Returns:
            LlamaActivationResult with comprehensive activation data
        """
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded first")

        self._validate_input(input_text)

        # Detect instruction patterns
        instruction_type = None
        if analyze_instructions:
            instruction_type = self.instruction_analyzer.detect_instruction_type(input_text)

        # Prepare inputs
        inputs = self._prepare_inputs(input_text)

        # Storage for activations and analysis
        layer_activations = {}
        attention_weights = {}
        hidden_states = []
        rms_norm_stats = {}

        # Register hooks for activation extraction
        hooks = self._register_llama_hooks(
            layer_activations,
            hidden_states,
            rms_norm_stats,
            attention_weights,
            layer_indices,
            return_attention
        )

        # Track memory usage
        self.memory_tracker.record_before_forward()

        try:
            with torch.no_grad():
                if target_text:
                    # Teacher forcing mode
                    target_inputs = self._prepare_inputs(target_text)
                    combined_ids = torch.cat([inputs['input_ids'], target_inputs['input_ids']], dim=1)

                    outputs = self.model(
                        input_ids=combined_ids,
                        attention_mask=torch.ones_like(combined_ids),
                        output_attentions=return_attention,
                        output_hidden_states=True,
                        return_dict=True
                    )
                else:
                    # Generation mode
                    outputs = self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs.get('attention_mask'),
                        max_new_tokens=max_new_tokens,
                        output_attentions=return_attention,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

        finally:
            # Clean up hooks
            for hook in hooks.values():
                hook.remove()

        # Record memory usage after forward pass
        self.memory_tracker.record_after_forward()

        # Analyze instruction compliance if applicable
        instruction_compliance_score = None
        instruction_attention_patterns = None

        if analyze_instructions and instruction_type:
            instruction_compliance_score = self.instruction_analyzer.calculate_compliance_score(
                input_text,
                outputs,
                attention_weights
            )

            instruction_attention_patterns = self.instruction_analyzer.extract_instruction_patterns(
                attention_weights,
                input_text
            )

        # Create result
        result = LlamaActivationResult(
            layer_activations=layer_activations,
            attention_weights=attention_weights,
            hidden_states={'layers': hidden_states},
            metadata={
                'model_name': self.model_name,
                'instruction_type': instruction_type,
                'num_layers': len(hidden_states),
                'input_length': inputs['input_ids'].size(1),
                'architecture': 'autoregressive',
                'rms_norm': True,
                'rope': True,
                'memory_optimized': self.memory_tracker.is_optimized()
            },
            input_ids=inputs['input_ids'],
            input_text=input_text,
            rms_norm_stats=rms_norm_stats,
            instruction_attention_patterns=instruction_attention_patterns,
            conversation_state=self._extract_conversation_state(input_text, outputs),
            memory_usage=self.memory_tracker.get_summary(),
            instruction_compliance_score=instruction_compliance_score
        )

        return result

    def _register_layer_hooks(
        self,
        activations_dict: Dict[str, torch.Tensor],
        hidden_states_list: List[torch.Tensor],
        attention_weights: Dict[str, torch.Tensor],
        layer_indices: Optional[List[int]] = None,
        return_attention: bool = True
    ) -> Dict[str, Any]:
        hooks = {}

        def layer_hook(name, layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_state = output[0]
                    if return_attention and len(output) > 1:
                        attn_weights = output[1]
                        if attn_weights is not None:
                            attention_weights[f"layer_{layer_idx}_attention"] = attn_weights.detach().cpu()
                else:
                    hidden_state = output

                activations_dict[f"layer_{layer_idx}"] = hidden_state.detach().cpu()

                if len(hidden_states_list) <= layer_idx:
                    hidden_states_list.extend([None] * (layer_idx - len(hidden_states_list) + 1))
                hidden_states_list[layer_idx] = hidden_state.detach().cpu()

            return hook

        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
            target_indices = layer_indices if layer_indices else range(len(layers))

            for i in target_indices:
                if i < len(layers):
                    hook_handle = layers[i].register_forward_hook(layer_hook(f"layer_{i}", i))
                    hooks[f"layer_{i}"] = hook_handle
        else:
            logger.warning("Could not find model layers for hook registration")

        return hooks

    def _register_rms_norm_hooks(
        self,
        rms_norm_stats: Dict[str, torch.Tensor],
        layer_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        hooks = {}

        def rms_norm_hook(name):
            def hook(module, input, output):
                if len(input) > 0 and input[0] is not None:
                    input_tensor = input[0]
                    output_tensor = output

                    rms_norm_stats[name] = {
                        'input_rms': torch.sqrt(torch.mean(input_tensor**2, dim=-1)).detach().cpu(),
                        'output_rms': torch.sqrt(torch.mean(output_tensor**2, dim=-1)).detach().cpu(),
                        'scaling_factor': torch.sqrt(torch.mean(output_tensor**2, dim=-1) / torch.mean(input_tensor**2, dim=-1)).detach().cpu()
                    }
            return hook

        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
            target_indices = layer_indices if layer_indices else range(len(layers))

            for i in target_indices:
                if i < len(layers):
                    if hasattr(layers[i], 'input_layernorm'):
                        rms_hook = layers[i].input_layernorm.register_forward_hook(
                            rms_norm_hook(f"layer_{i}_input_norm")
                        )
                        hooks[f"layer_{i}_input_norm"] = rms_hook

                    if hasattr(layers[i], 'post_attention_layernorm'):
                        rms_hook = layers[i].post_attention_layernorm.register_forward_hook(
                            rms_norm_hook(f"layer_{i}_post_attn_norm")
                        )
                        hooks[f"layer_{i}_post_attn_norm"] = rms_hook
        return hooks

    def _register_llama_hooks(
        self,
        activations_dict: Dict[str, torch.Tensor],
        hidden_states_list: List[torch.Tensor],
        rms_norm_stats: Dict[str, torch.Tensor],
        attention_weights: Dict[str, torch.Tensor],
        layer_indices: Optional[List[int]] = None,
        return_attention: bool = True
    ) -> Dict[str, Any]:
        """Register hooks for LLaMA-specific analysis."""
        hooks = {}
        hooks.update(self._register_layer_hooks(activations_dict, hidden_states_list, attention_weights, layer_indices, return_attention))
        hooks.update(self._register_rms_norm_hooks(rms_norm_stats, layer_indices))
        return hooks

    def _extract_conversation_state(self, input_text: str, outputs) -> Dict[str, Any]:
        """Extract conversation state for multi-turn analysis."""
        # Simple conversation state extraction
        conversation_state = {
            'turns': input_text.count('\n') + 1,
            'has_greeting': any(greeting in input_text.lower() for greeting in ['hello', 'hi', 'hey']),
            'has_question': '?' in input_text,
            'has_instruction': any(inst in input_text.lower() for inst in ['please', 'can you', 'help']),
            'length': len(input_text.split()),
            'sentiment': 'neutral'  # Placeholder for sentiment analysis
        }

        return conversation_state

    def analyze_rms_normalization(
        self,
        input_text: str,
        comparison_inputs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze RMS normalization patterns across layers.

        Args:
            input_text: Primary input for analysis
            comparison_inputs: Additional inputs for comparison analysis

        Returns:
            Comprehensive RMS normalization analysis
        """
        # Extract activations with RMS norm tracking
        result = self.extract_activations(input_text, return_attention=False)

        analysis = {
            'input_text': input_text,
            'rms_norm_analysis': {},
            'layer_stability': {},
            'normalization_effectiveness': {}
        }

        # Analyze RMS norm statistics
        for layer_name, stats in result.rms_norm_stats.items():
            layer_analysis = {
                'input_rms_mean': float(stats['input_rms'].mean()),
                'input_rms_std': float(stats['input_rms'].std()),
                'output_rms_mean': float(stats['output_rms'].mean()),
                'output_rms_std': float(stats['output_rms'].std()),
                'scaling_factor_mean': float(stats['scaling_factor'].mean()),
                'scaling_factor_std': float(stats['scaling_factor'].std()),
                'normalization_strength': float(stats['scaling_factor'].mean())
            }

            analysis['rms_norm_analysis'][layer_name] = layer_analysis

        # Calculate layer stability metrics
        if len(result.rms_norm_stats) > 1:
            scaling_factors = []
            for stats in result.rms_norm_stats.values():
                scaling_factors.append(stats['scaling_factor'].mean())

            scaling_factors = torch.stack(scaling_factors)
            analysis['layer_stability'] = {
                'variance_across_layers': float(scaling_factors.var()),
                'stability_trend': float(torch.diff(scaling_factors).mean()),
                'max_scaling_layer': int(scaling_factors.argmax()),
                'min_scaling_layer': int(scaling_factors.argmin())
            }

        # Compare with additional inputs if provided
        if comparison_inputs:
            comparison_analyses = []
            for comp_input in comparison_inputs:
                comp_result = self.extract_activations(comp_input, return_attention=False)
                comparison_analyses.append(comp_result.rms_norm_stats)

            analysis['normalization_effectiveness'] = self._compare_rms_normalization(
                result.rms_norm_stats,
                comparison_analyses
            )

        return analysis

    def _compare_rms_normalization(
        self,
        primary_stats: Dict[str, torch.Tensor],
        comparison_stats: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Any]:
        """Compare RMS normalization across different inputs."""
        comparison_results = {
            'consistency_score': 0.0,
            'layer_wise_consistency': {},
            'input_sensitivity': {}
        }

        # Calculate consistency across inputs
        consistency_scores = []

        for layer_name in primary_stats.keys():
            if all(layer_name in comp_stats for comp_stats in comparison_stats):
                primary_scaling = primary_stats[layer_name]['scaling_factor'].mean()
                comp_scalings = [comp_stats[layer_name]['scaling_factor'].mean() for comp_stats in comparison_stats]

                # Calculate coefficient of variation
                all_scalings = torch.tensor([primary_scaling] + comp_scalings)
                cv = all_scalings.std() / all_scalings.mean()
                consistency_scores.append(1.0 / (1.0 + cv))  # Higher is more consistent

                comparison_results['layer_wise_consistency'][layer_name] = {
                    'coefficient_of_variation': float(cv),
                    'consistency_score': float(1.0 / (1.0 + cv)),
                    'mean_scaling': float(all_scalings.mean()),
                    'scaling_range': float(all_scalings.max() - all_scalings.min())
                }

        if consistency_scores:
            comparison_results['consistency_score'] = float(torch.tensor(consistency_scores).mean())

        return comparison_results


class MemoryTracker:
    """Track memory usage during model operations."""

    def __init__(self):
        self.baseline_memory = 0
        self.before_forward = 0
        self.after_forward = 0
        self.peak_memory = 0

    def record_baseline(self):
        """Record baseline memory usage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.baseline_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
        else:
            self.baseline_memory = psutil.Process().memory_info().rss / (1024**3)

    def record_before_forward(self):
        """Record memory before forward pass."""
        if torch.cuda.is_available():
            self.before_forward = torch.cuda.memory_allocated() / (1024**3)
        else:
            self.before_forward = psutil.Process().memory_info().rss / (1024**3)

    def record_after_forward(self):
        """Record memory after forward pass."""
        if torch.cuda.is_available():
            self.after_forward = torch.cuda.memory_allocated() / (1024**3)
            self.peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
        else:
            self.after_forward = psutil.Process().memory_info().rss / (1024**3)

    def get_current_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated() / (1024**3)
            cached = torch.cuda.memory_reserved() / (1024**3)
        else:
            current = psutil.Process().memory_info().rss / (1024**3)
            cached = 0.0

        return {
            'current_gb': current,
            'cached_gb': cached,
            'baseline_gb': self.baseline_memory
        }

    def get_summary(self) -> Dict[str, float]:
        """Get memory usage summary."""
        return {
            'baseline_gb': self.baseline_memory,
            'before_forward_gb': self.before_forward,
            'after_forward_gb': self.after_forward,
            'peak_gb': self.peak_memory,
            'forward_increase_gb': self.after_forward - self.before_forward
        }

    def is_optimized(self) -> bool:
        """Check if memory optimization is being used."""
        return self.peak_memory > 0 and self.peak_memory < 20.0  # Heuristic


class InstructionAnalyzer:
    """Analyze instruction-following behavior in LLaMA models."""

    def detect_instruction_type(self, text: str) -> Optional[str]:
        """Detect the type of instruction in the input text."""
        text_lower = text.lower()

        # Check for different instruction patterns
        if any(pattern in text_lower for pattern in ['please', 'can you', 'could you', 'would you']):
            return 'polite_request'
        elif any(pattern in text_lower for pattern in ['what', 'how', 'why', 'when', 'where']):
            return 'question'
        elif any(pattern in text_lower for pattern in ['write', 'create', 'generate', 'make']):
            return 'creation_task'
        elif any(pattern in text_lower for pattern in ['explain', 'describe', 'tell me about']):
            return 'explanation_request'
        elif any(pattern in text_lower for pattern in ['solve', 'calculate', 'compute']):
            return 'problem_solving'
        elif any(pattern in text_lower for pattern in ['translate', 'convert', 'transform']):
            return 'transformation_task'
        else:
            return 'general'

    def calculate_compliance_score(
        self,
        input_text: str,
        outputs,
        attention_weights: Dict[str, torch.Tensor]
    ) -> float:
        """Calculate instruction compliance score based on attention patterns."""
        # Simplified compliance score calculation
        # In practice, this would be more sophisticated

        # Check if model attended to instruction keywords
        instruction_keywords = ['please', 'can', 'you', 'help', 'explain', 'write', 'create']
        input_tokens = input_text.lower().split()

        keyword_count = sum(1 for word in input_tokens if word in instruction_keywords)
        total_words = len(input_tokens)

        # Simple heuristic: higher keyword ratio suggests higher instruction compliance
        base_score = keyword_count / max(total_words, 1)

        # Adjust based on attention patterns (simplified)
        if attention_weights:
            # In a real implementation, we would analyze attention to instruction tokens
            attention_adjustment = 0.1  # Placeholder
            base_score += attention_adjustment

        return min(1.0, base_score)

    def extract_instruction_patterns(
        self,
        attention_weights: Dict[str, torch.Tensor],
        input_text: str
    ) -> Dict[str, Any]:
        """Extract instruction-specific attention patterns."""
        patterns = {
            'instruction_attention_strength': 0.0,
            'context_attention_strength': 0.0,
            'attention_distribution': {},
            'layer_wise_instruction_focus': {}
        }

        # Simplified pattern extraction
        # In practice, this would involve detailed attention analysis

        for layer_name, attention in attention_weights.items():
            if attention.dim() >= 3:
                # Average attention across heads and batch
                avg_attention = attention.mean(dim=(0, 1)) if attention.dim() == 4 else attention.mean(dim=0)

                # Calculate attention distribution metrics
                patterns['layer_wise_instruction_focus'][layer_name] = {
                    'max_attention': float(avg_attention.max()),
                    'attention_entropy': float(self._calculate_attention_entropy(avg_attention)),
                    'attention_concentration': float(avg_attention.std())
                }

        return patterns

    def _calculate_attention_entropy(self, attention_matrix: torch.Tensor) -> float:
        """Calculate entropy of attention distribution."""
        flat_attention = attention_matrix.flatten()
        flat_attention = flat_attention / flat_attention.sum()

        epsilon = 1e-10
        entropy = -(flat_attention * torch.log(flat_attention + epsilon)).sum()

        return entropy.item()


# Register LLaMA handler with the factory
from .base_model_handler import ModelFactory
ModelFactory.register_handler('llama', LlamaModelHandler)
