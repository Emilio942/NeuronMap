"""
TransformerLens Adapter for NeuronMap
====================================

Adapter for integrating TransformerLens models with NeuronMap's
interpretability framework, enabling neuron hooking and analysis.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import logging
from pathlib import Path
import time
from dataclasses import dataclass

# Optional imports with fallbacks
try:
    import transformer_lens as tl
    from transformer_lens import HookedTransformer
    TL_AVAILABLE = True
except ImportError:
    TL_AVAILABLE = False
    logging.warning("TransformerLens not available, adapter functionality disabled")

from ...core.plugin_interface import MechanisticPluginBase, ToolExecutionResult

logger = logging.getLogger(__name__)

@dataclass
class HookResult:
    """Result from a TransformerLens hook."""
    layer_name: str
    activation: torch.Tensor
    hook_point: str
    metadata: Dict[str, Any]

@dataclass
class AnalysisResult:
    """Result from TransformerLens analysis."""
    model_name: str
    hook_results: List[HookResult]
    residual_stream_data: Dict[str, torch.Tensor]
    attention_patterns: Dict[str, torch.Tensor]
    analysis_metadata: Dict[str, Any]

class TransformerLensAdapter(MechanisticPluginBase):
    """
    Adapter for integrating TransformerLens models with NeuronMap.
    
    Provides unified interface for neuron hooking, activation extraction,
    and mechanistic interpretability analysis using TransformerLens.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(tool_id="transformerlens_adapter", config=config)
        
        self.version = "1.0.0"
        self.description = "TransformerLens adapter for mechanistic interpretability"
        
        # Configuration
        self.model_name = config.get('model_name', 'gpt2-small') if config else 'gpt2-small'
        self.device = config.get('device', 'cpu') if config else 'cpu'
        self.hook_points = config.get('hook_points', []) if config else []
        self.extract_residual_stream = config.get('extract_residual_stream', True) if config else True
        self.extract_attention = config.get('extract_attention', True) if config else True
        self.cache_activations = config.get('cache_activations', True) if config else True
        
        # Internal state
        self.model: Optional[HookedTransformer] = None
        self.hook_handles = []
        self.activation_cache = {}
        
        logger.info(f"Initialized TransformerLens adapter for {self.model_name}")
    
    def initialize(self) -> bool:
        """Initialize the TransformerLens adapter."""
        try:
            if not TL_AVAILABLE:
                logger.error("TransformerLens not available - cannot initialize adapter")
                return False
            
            # Load model
            logger.info(f"Loading TransformerLens model: {self.model_name}")
            self.model = HookedTransformer.from_pretrained(
                self.model_name,
                device=self.device
            )
            
            # Set up default hook points if none specified
            if not self.hook_points:
                self.hook_points = self._get_default_hook_points()
            
            logger.info(f"Model loaded with {len(self.hook_points)} hook points")
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TransformerLens adapter: {e}")
            return False
    
    def execute(self, input_text: Union[str, List[str]], 
                analysis_type: str = "full",
                custom_hooks: Optional[List[str]] = None,
                **kwargs) -> ToolExecutionResult:
        """
        Execute TransformerLens analysis.
        
        Args:
            input_text: Text input(s) to analyze
            analysis_type: Type of analysis ('activations', 'attention', 'residual', 'full')
            custom_hooks: Optional custom hook points to use
            
        Returns:
            ToolExecutionResult with analysis results
        """
        start_time = time.time()
        
        try:
            if not self.initialized or self.model is None:
                raise RuntimeError("TransformerLens adapter not initialized")
            
            # Prepare inputs
            if isinstance(input_text, str):
                input_text = [input_text]
            
            # Tokenize inputs
            tokens = self.model.to_tokens(input_text)
            
            # Set up hooks
            hook_points = custom_hooks if custom_hooks else self.hook_points
            self._setup_hooks(hook_points, analysis_type)
            
            # Run forward pass with hooks
            with torch.no_grad():
                if analysis_type == "full" or analysis_type == "attention":
                    # Run with attention analysis
                    logits, cache = self.model.run_with_cache(tokens)
                else:
                    # Simple forward pass
                    logits = self.model(tokens)
                    cache = None
            
            # Collect results
            hook_results = self._collect_hook_results()
            
            # Extract residual stream data if requested
            residual_stream_data = {}
            if self.extract_residual_stream and cache is not None:
                residual_stream_data = self._extract_residual_stream(cache, tokens)
            
            # Extract attention patterns if requested
            attention_patterns = {}
            if self.extract_attention and cache is not None:
                attention_patterns = self._extract_attention_patterns(cache, tokens)
            
            # Generate analysis metadata
            analysis_metadata = self._generate_analysis_metadata(
                tokens, logits, analysis_type
            )
            
            # Create analysis result
            analysis_result = AnalysisResult(
                model_name=self.model_name,
                hook_results=hook_results,
                residual_stream_data=residual_stream_data,
                attention_patterns=attention_patterns,
                analysis_metadata=analysis_metadata
            )
            
            # Convert to NeuronMap format
            outputs = self._convert_to_neuronmap_format(analysis_result)
            
            # Clean up hooks
            self._cleanup_hooks()
            
            execution_time = time.time() - start_time
            
            return ToolExecutionResult(
                tool_id=self.tool_id,
                success=True,
                execution_time=execution_time,
                outputs=outputs,
                metadata=self.get_metadata(),
                errors=[],
                warnings=[],
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
        except Exception as e:
            logger.error(f"TransformerLens analysis failed: {e}")
            self._cleanup_hooks()
            execution_time = time.time() - start_time
            
            return ToolExecutionResult(
                tool_id=self.tool_id,
                success=False,
                execution_time=execution_time,
                outputs={},
                metadata=self.get_metadata(),
                errors=[str(e)],
                warnings=[],
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
    
    def _get_default_hook_points(self) -> List[str]:
        """Get default hook points for the model."""
        if self.model is None:
            return []
        
        hook_points = []
        
        # Add standard hook points
        for layer_idx in range(self.model.cfg.n_layers):
            # Attention hooks
            hook_points.append(f"blocks.{layer_idx}.attn.hook_q")
            hook_points.append(f"blocks.{layer_idx}.attn.hook_k")
            hook_points.append(f"blocks.{layer_idx}.attn.hook_v")
            hook_points.append(f"blocks.{layer_idx}.attn.hook_pattern")
            hook_points.append(f"blocks.{layer_idx}.attn.hook_z")
            
            # MLP hooks
            hook_points.append(f"blocks.{layer_idx}.mlp.hook_pre")
            hook_points.append(f"blocks.{layer_idx}.mlp.hook_post")
            
            # Residual stream hooks
            hook_points.append(f"blocks.{layer_idx}.hook_resid_pre")
            hook_points.append(f"blocks.{layer_idx}.hook_resid_post")
        
        # Output hooks
        hook_points.append("ln_final.hook_normalized")
        
        return hook_points
    
    def _setup_hooks(self, hook_points: List[str], analysis_type: str):
        """Set up activation hooks."""
        self.activation_cache.clear()
        
        def make_hook_fn(hook_name: str):
            def hook_fn(activation: torch.Tensor, hook):
                if self.cache_activations:
                    self.activation_cache[hook_name] = activation.clone()
                return activation
            return hook_fn
        
        # Add hooks
        for hook_name in hook_points:
            try:
                if hasattr(self.model, 'add_hook'):
                    handle = self.model.add_hook(hook_name, make_hook_fn(hook_name))
                    self.hook_handles.append(handle)
            except Exception as e:
                logger.warning(f"Failed to add hook {hook_name}: {e}")
    
    def _collect_hook_results(self) -> List[HookResult]:
        """Collect results from activated hooks."""
        results = []
        
        for hook_name, activation in self.activation_cache.items():
            result = HookResult(
                layer_name=hook_name,
                activation=activation,
                hook_point=hook_name,
                metadata={
                    'shape': list(activation.shape),
                    'dtype': str(activation.dtype),
                    'device': str(activation.device),
                    'requires_grad': activation.requires_grad
                }
            )
            results.append(result)
        
        return results
    
    def _extract_residual_stream(self, cache: Dict[str, torch.Tensor], 
                                tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract residual stream data from cache."""
        residual_data = {}
        
        # Get residual stream activations
        for key, activation in cache.items():
            if 'resid' in key:
                # Convert to numpy for NeuronMap compatibility
                residual_data[key] = activation.detach().cpu()
        
        return residual_data
    
    def _extract_attention_patterns(self, cache: Dict[str, torch.Tensor],
                                   tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract attention patterns from cache."""
        attention_data = {}
        
        # Get attention patterns
        for key, activation in cache.items():
            if 'attn' in key and 'pattern' in key:
                attention_data[key] = activation.detach().cpu()
        
        return attention_data
    
    def _generate_analysis_metadata(self, tokens: torch.Tensor, 
                                   logits: torch.Tensor,
                                   analysis_type: str) -> Dict[str, Any]:
        """Generate metadata for the analysis."""
        return {
            'model_config': {
                'model_name': self.model_name,
                'n_layers': self.model.cfg.n_layers if self.model else 0,
                'n_heads': self.model.cfg.n_heads if self.model else 0,
                'd_model': self.model.cfg.d_model if self.model else 0,
                'd_head': self.model.cfg.d_head if self.model else 0,
                'vocab_size': self.model.cfg.d_vocab if self.model else 0
            },
            'input_info': {
                'batch_size': tokens.shape[0],
                'seq_length': tokens.shape[1],
                'token_shape': list(tokens.shape)
            },
            'output_info': {
                'logits_shape': list(logits.shape),
                'output_vocab_size': logits.shape[-1]
            },
            'analysis_type': analysis_type,
            'num_hooks_activated': len(self.activation_cache),
            'device': self.device
        }
    
    def _convert_to_neuronmap_format(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """Convert TransformerLens results to NeuronMap format."""
        
        # Convert hook results
        neuron_activations = {}
        for hook_result in analysis_result.hook_results:
            # Convert tensor to numpy for NeuronMap compatibility
            activation_np = hook_result.activation.detach().cpu().numpy()
            
            neuron_activations[hook_result.layer_name] = {
                'activations': activation_np.tolist(),  # JSON serializable
                'shape': list(hook_result.activation.shape),
                'hook_point': hook_result.hook_point,
                'metadata': hook_result.metadata
            }
        
        # Convert residual stream data
        residual_stream_np = {}
        for key, tensor in analysis_result.residual_stream_data.items():
            residual_stream_np[key] = {
                'data': tensor.numpy().tolist(),
                'shape': list(tensor.shape)
            }
        
        # Convert attention patterns
        attention_patterns_np = {}
        for key, tensor in analysis_result.attention_patterns.items():
            attention_patterns_np[key] = {
                'data': tensor.numpy().tolist(),
                'shape': list(tensor.shape)
            }
        
        # Compute summary statistics
        summary_stats = self._compute_summary_statistics(analysis_result)
        
        return {
            'model_name': analysis_result.model_name,
            'neuron_activations': neuron_activations,
            'residual_stream_data': residual_stream_np,
            'attention_patterns': attention_patterns_np,
            'analysis_metadata': analysis_result.analysis_metadata,
            'summary_statistics': summary_stats,
            'neuronmap_compatibility': {
                'format_version': '1.0',
                'data_type': 'transformer_lens_analysis',
                'hook_count': len(analysis_result.hook_results),
                'layer_count': len(set(hr.layer_name.split('.')[1] 
                                     for hr in analysis_result.hook_results
                                     if 'blocks.' in hr.layer_name))
            }
        }
    
    def _compute_summary_statistics(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """Compute summary statistics for the analysis."""
        stats = {
            'total_activations': len(analysis_result.hook_results),
            'layer_statistics': {},
            'activation_ranges': {}
        }
        
        # Per-layer statistics
        layer_stats = {}
        for hook_result in analysis_result.hook_results:
            layer_name = hook_result.layer_name
            activation = hook_result.activation
            
            if layer_name not in layer_stats:
                layer_stats[layer_name] = {
                    'mean': float(torch.mean(activation)),
                    'std': float(torch.std(activation)),
                    'min': float(torch.min(activation)),
                    'max': float(torch.max(activation)),
                    'shape': list(activation.shape),
                    'activation_count': activation.numel()
                }
        
        stats['layer_statistics'] = layer_stats
        
        # Overall activation ranges
        all_activations = torch.cat([hr.activation.flatten() 
                                   for hr in analysis_result.hook_results])
        stats['activation_ranges'] = {
            'global_min': float(torch.min(all_activations)),
            'global_max': float(torch.max(all_activations)),
            'global_mean': float(torch.mean(all_activations)),
            'global_std': float(torch.std(all_activations)),
            'total_neurons': int(len(all_activations))
        }
        
        return stats
    
    def _cleanup_hooks(self):
        """Clean up registered hooks."""
        for handle in self.hook_handles:
            try:
                if hasattr(handle, 'remove'):
                    handle.remove()
            except Exception as e:
                logger.warning(f"Failed to remove hook: {e}")
        
        self.hook_handles.clear()
        
        if not self.cache_activations:
            self.activation_cache.clear()
    
    def analyze_model(self, input_data: Any, analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """Interface method for MechanisticPluginBase."""
        result = self.execute(
            input_text=input_data,
            analysis_type=analysis_config.get('analysis_type', 'full'),
            custom_hooks=analysis_config.get('custom_hooks')
        )
        return result.outputs if result.success else {}
    
    def extract_mechanisms(self, model_data: Any) -> Dict[str, Any]:
        """Extract mechanistic information from model data."""
        if not isinstance(model_data, dict) or 'neuron_activations' not in model_data:
            return {}
        
        mechanisms = {
            'attention_mechanisms': self._analyze_attention_mechanisms(model_data),
            'mlp_mechanisms': self._analyze_mlp_mechanisms(model_data),
            'residual_stream_analysis': self._analyze_residual_stream(model_data)
        }
        
        return mechanisms
    
    def _analyze_attention_mechanisms(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze attention mechanisms from model data."""
        attention_analysis = {}
        
        for layer_name, activation_data in model_data['neuron_activations'].items():
            if 'attn' in layer_name and 'pattern' in layer_name:
                activations = np.array(activation_data['activations'])
                
                attention_analysis[layer_name] = {
                    'attention_entropy': float(np.mean(-np.sum(activations * np.log(activations + 1e-10), axis=-1))),
                    'max_attention': float(np.max(activations)),
                    'attention_sparsity': float(np.mean(activations > 0.1))
                }
        
        return attention_analysis
    
    def _analyze_mlp_mechanisms(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze MLP mechanisms from model data."""
        mlp_analysis = {}
        
        for layer_name, activation_data in model_data['neuron_activations'].items():
            if 'mlp' in layer_name:
                activations = np.array(activation_data['activations'])
                
                mlp_analysis[layer_name] = {
                    'activation_sparsity': float(np.mean(activations > 0)),
                    'mean_activation': float(np.mean(activations)),
                    'activation_variance': float(np.var(activations))
                }
        
        return mlp_analysis
    
    def _analyze_residual_stream(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze residual stream from model data."""
        residual_analysis = {}
        
        if 'residual_stream_data' in model_data:
            for stream_name, stream_data in model_data['residual_stream_data'].items():
                data = np.array(stream_data['data'])
                
                residual_analysis[stream_name] = {
                    'stream_norm': float(np.linalg.norm(data)),
                    'stream_mean': float(np.mean(data)),
                    'stream_std': float(np.std(data))
                }
        
        return residual_analysis
    
    def validate_output(self, output: Any) -> bool:
        """Validate TransformerLens adapter output."""
        if not isinstance(output, dict):
            return False
        
        required_keys = ['model_name', 'neuron_activations', 'analysis_metadata']
        for key in required_keys:
            if key not in output:
                logger.error(f"Missing required output key: {key}")
                return False
        
        return True

def create_transformerlens_adapter(config: Optional[Dict[str, Any]] = None) -> TransformerLensAdapter:
    """Factory function to create TransformerLens adapter."""
    return TransformerLensAdapter(config=config)
