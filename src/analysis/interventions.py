"""
Model Intervention System - Core implementation for model surgery and path analysis.
This module implements the foundational infrastructure for intervening in neural network
forward passes, enabling ablation studies, path patching, and causal tracing.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from contextlib import contextmanager
logger = logging.getLogger(__name__)
class InterventionType(Enum):
    """Types of interventions supported by the system."""
    ABLATION = "ablation"         # Set activations to zero
    NOISE = "noise"               # Add noise to activations
    MEAN = "mean"                 # Replace with mean activation
    PATCHING = "patching"         # Replace with cached activations
    CUSTOM = "custom"             # User-defined modification function
@dataclass
class InterventionSpec:
    """Specification for a single intervention."""
    layer_name: str
    intervention_type: InterventionType
    target_indices: Optional[List[int]] = None  # Specific neurons/heads to target
    intervention_value: Optional[float] = None   # For ablation/mean interventions
    custom_function: Optional[Callable] = None   # For custom interventions
    patch_source: Optional[str] = None           # Source cache key for patching
class ModifiableHookManager:
    """
    Advanced hook manager that supports modifiable forward hooks.
    This is the core implementation of B1: Modifizierbare Forward-Hooks.
    Unlike standard PyTorch hooks that only observe, these hooks can modify
    the forward pass by returning modified tensors.
    """
    def __init__(self):
        self.hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self.interventions: Dict[str, InterventionSpec] = {}
        self.activation_cache: Dict[str, torch.Tensor] = {}
        self._hook_outputs: Dict[str, torch.Tensor] = {}
    def register_modifiable_hook(
        self,
        module: nn.Module,
        layer_name: str,
        intervention_spec: Optional[InterventionSpec] = None
    ) -> torch.utils.hooks.RemovableHandle:
        """
        Register a forward hook that can modify activations.
        Args:
            module: PyTorch module to hook
            layer_name: Unique identifier for this layer
            intervention_spec: Optional intervention to apply
        Returns:
            Handle for removing the hook
        """
        if intervention_spec:
            self.interventions[layer_name] = intervention_spec
        def modifiable_forward_hook(module, input, output):
            """
            Forward hook that can modify the output tensor.
            This is the key innovation: instead of just observing the output,
            we can return a modified version that becomes the actual output
            for the next layer.
            """
            # Handle different output types (tensor, tuple, etc.)
            if isinstance(output, tuple):
                original_tensor = output[0]
                other_outputs = output[1:]
            else:
                original_tensor = output
                other_outputs = None
            # Store original activation for analysis
            self.activation_cache[layer_name] = original_tensor.detach().clone()
            # Apply intervention if specified
            if layer_name in self.interventions:
                modified_tensor = self._apply_intervention(
                    original_tensor,
                    self.interventions[layer_name]
                )
                # Log intervention for debugging
                logger.debug(
                    f"Applied {self.interventions[layer_name].intervention_type.value} intervention to {layer_name}"
                )
                # Return modified output in correct format
                if other_outputs is not None:
                    return (modified_tensor,) + other_outputs
                else:
                    return modified_tensor
            # No intervention - return original
            return output
        # Register the hook
        handle = module.register_forward_hook(modifiable_forward_hook)
        self.hooks[layer_name] = handle
        logger.info(f"Registered modifiable hook for layer: {layer_name}")
        return handle
    def _apply_ablation(self, tensor: torch.Tensor, spec: InterventionSpec) -> torch.Tensor:
        modified_tensor = tensor.clone()
        if spec.target_indices:
            if len(modified_tensor.shape) >= 2:
                for idx in spec.target_indices:
                    if idx < modified_tensor.shape[-1]:
                        modified_tensor[..., idx] = 0.0
            else:
                logger.warning(f"Cannot ablate specific indices in 1D tensor")
                modified_tensor.fill_(0.0)
        else:
            modified_tensor.fill_(0.0)
        return modified_tensor

    def _apply_noise(self, tensor: torch.Tensor, spec: InterventionSpec) -> torch.Tensor:
        modified_tensor = tensor.clone()
        noise_std = spec.intervention_value or 0.1
        noise = torch.randn_like(modified_tensor) * noise_std
        modified_tensor += noise
        return modified_tensor

    def _apply_mean(self, tensor: torch.Tensor, spec: InterventionSpec) -> torch.Tensor:
        modified_tensor = tensor.clone()
        if spec.target_indices:
            mean_val = tensor.mean()
            for idx in spec.target_indices:
                if idx < modified_tensor.shape[-1]:
                    modified_tensor[..., idx] = mean_val
        else:
            modified_tensor.fill_(tensor.mean())
        return modified_tensor

    def _apply_patching(self, tensor: torch.Tensor, spec: InterventionSpec) -> torch.Tensor:
        modified_tensor = tensor.clone()
        if spec.patch_source and spec.patch_source in self.activation_cache:
            patch_tensor = self.activation_cache[spec.patch_source]
            if patch_tensor.shape == modified_tensor.shape:
                if spec.target_indices:
                    for idx in spec.target_indices:
                        if idx < modified_tensor.shape[-1]:
                            modified_tensor[..., idx] = patch_tensor[..., idx]
                else:
                    modified_tensor = patch_tensor.clone()
            else:
                logger.warning(
                    f"Shape mismatch in patching: {patch_tensor.shape} vs {modified_tensor.shape}"
                )
        return modified_tensor

    def _apply_custom(self, tensor: torch.Tensor, spec: InterventionSpec) -> torch.Tensor:
        modified_tensor = tensor.clone()
        if spec.custom_function:
            modified_tensor = spec.custom_function(modified_tensor)
        return modified_tensor

    def _apply_intervention(
        self, 
        tensor: torch.Tensor, 
        spec: InterventionSpec
    ) -> torch.Tensor:
        """Apply the specified intervention to a tensor."""
        if spec.intervention_type == InterventionType.ABLATION:
            return self._apply_ablation(tensor, spec)
        elif spec.intervention_type == InterventionType.NOISE:
            return self._apply_noise(tensor, spec)
        elif spec.intervention_type == InterventionType.MEAN:
            return self._apply_mean(tensor, spec)
        elif spec.intervention_type == InterventionType.PATCHING:
            return self._apply_patching(tensor, spec)
        elif spec.intervention_type == InterventionType.CUSTOM:
            return self._apply_custom(tensor, spec)
        return tensor
    def add_intervention(self, layer_name: str, intervention_spec: InterventionSpec):
        """Add or update an intervention for a specific layer."""
        self.interventions[layer_name] = intervention_spec
        logger.info(f"Added {intervention_spec.intervention_type.value} intervention for {layer_name}")
    def remove_intervention(self, layer_name: str):
        """Remove intervention for a specific layer."""
        if layer_name in self.interventions:
            del self.interventions[layer_name]
            logger.info(f"Removed intervention for {layer_name}")
    def clear_interventions(self):
        """Clear all interventions."""
        self.interventions.clear()
        logger.info("Cleared all interventions")
    def remove_all_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()
        logger.info("Removed all hooks")
    def get_cached_activations(self) -> Dict[str, torch.Tensor]:
        """Get all cached activations."""
        return self.activation_cache.copy()
    def clear_cache(self):
        """Clear activation cache."""
        self.activation_cache.clear()
@contextmanager
def intervention_context(
    model: nn.Module,
    intervention_specs: List[Tuple[str, InterventionSpec]]
):
    """
    Context manager for safely applying interventions.
    Usage:
        with intervention_context(model, [(layer_name, intervention_spec)]):
            output = model(input_tensor)
    """
    hook_manager = ModifiableHookManager()
    try:
        # Register all hooks with interventions
        for layer_name, spec in intervention_specs:
            module = dict(model.named_modules())[layer_name]
            hook_manager.register_modifiable_hook(module, layer_name, spec)
        yield hook_manager
    finally:
        # Always clean up hooks
        hook_manager.remove_all_hooks()
# Core intervention functions (B3: Core-Funktion für Ablation)
def run_with_ablation(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_name: str,
    neuron_indices: Optional[List[int]] = None,
    return_activations: bool = False
) -> Dict[str, Any]:
    """
    Run model with specific neurons ablated (zeroed out).
    This implements B3: Core-Funktion für Ablation from the task list.
    Args:
        model: PyTorch model
        input_tensor: Input to process
        layer_name: Layer to ablate
        neuron_indices: Specific neurons to ablate (None = all)
        return_activations: Whether to return intermediate activations
    Returns:
        Dict containing output and optionally activations
    """
    ablation_spec = InterventionSpec(
        layer_name=layer_name,
        intervention_type=InterventionType.ABLATION,
        target_indices=neuron_indices
    )
    with intervention_context(model, [(layer_name, ablation_spec)]) as hook_manager:
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        result = {
            'output': output,
            'ablated_layer': layer_name,
            'ablated_neurons': neuron_indices
        }
        if return_activations:
            result['activations'] = hook_manager.get_cached_activations()
        return result
def run_with_patching(
    model: nn.Module,
    clean_input: torch.Tensor,
    corrupted_input: torch.Tensor,
    patch_specs: List[Tuple[str, Optional[List[int]]]]
) -> Dict[str, Any]:
    """
    Run path patching experiment.
    This implements B4: Core-Funktion für Path Patching from the task list.
    Args:
        model: PyTorch model
        clean_input: "Clean" input tensor
        corrupted_input: "Corrupted" input tensor
        patch_specs: List of (layer_name, neuron_indices) to patch
    Returns:
        Results of path patching experiment
    """
    model.eval()
    # Step 1: Get clean activations
    clean_cache = {}
    def cache_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                clean_cache[name] = output[0].detach().clone()
            else:
                clean_cache[name] = output.detach().clone()
        return hook
    # Register caching hooks
    hooks = []
    for layer_name, _ in patch_specs:
        module = dict(model.named_modules())[layer_name]
        hooks.append(module.register_forward_hook(cache_hook(layer_name)))
    # Run clean forward pass
    with torch.no_grad():
        clean_output = model(clean_input)
    # Remove caching hooks
    for hook in hooks:
        hook.remove()
    # Step 2: Run corrupted pass with patching
    intervention_specs = []
    for layer_name, neuron_indices in patch_specs:
        patch_spec = InterventionSpec(
            layer_name=layer_name,
            intervention_type=InterventionType.PATCHING,
            target_indices=neuron_indices,
            patch_source=layer_name
        )
        intervention_specs.append((layer_name, patch_spec))
    with intervention_context(model, intervention_specs) as hook_manager:
        # Pre-populate cache with clean activations
        hook_manager.activation_cache.update(clean_cache)
        with torch.no_grad():
            patched_output = model(corrupted_input)
    # Step 3: Run pure corrupted for comparison
    with torch.no_grad():
        corrupted_output = model(corrupted_input)
    return {
        'clean_output': clean_output,
        'corrupted_output': corrupted_output,
        'patched_output': patched_output,
        'patch_specs': patch_specs,
        'clean_activations': clean_cache
    }
def calculate_causal_effect(
    clean_output: torch.Tensor,
    corrupted_output: torch.Tensor,
    patched_output: torch.Tensor,
    metric: str = "logit_diff"
) -> float:
    """
    Calculate causal effect of intervention.
    This implements B5: Kausale Effekt-Analyse from the task list.
    Args:
        clean_output: Output from clean run
        corrupted_output: Output from corrupted run
        patched_output: Output from patched run
        metric: Metric to use ("logit_diff", "probability", "entropy")
    Returns:
        Causal effect score
    """
    if metric == "logit_diff":
        # Simple logit difference
        clean_logits = clean_output.logits if hasattr(clean_output, 'logits') else clean_output
        corrupted_logits = corrupted_output.logits if hasattr(corrupted_output, 'logits') else corrupted_output
        patched_logits = patched_output.logits if hasattr(patched_output, 'logits') else patched_output
        # Causal effect = (patched - corrupted) / (clean - corrupted)
        numerator = torch.norm(patched_logits - corrupted_logits)
        denominator = torch.norm(clean_logits - corrupted_logits)
        return (numerator / (denominator + 1e-8)).item()
    elif metric == "probability":
        # Probability difference on most likely token
        clean_probs = torch.softmax(clean_output.logits if hasattr(clean_output, 'logits') else clean_output, dim=-1)
        corrupted_probs = torch.softmax(corrupted_output.logits if hasattr(corrupted_output, 'logits') else corrupted_output, dim=-1)
        patched_probs = torch.softmax(patched_output.logits if hasattr(patched_output, 'logits') else patched_output, dim=-1)
        # Get most likely token from clean run
        top_token = torch.argmax(clean_probs, dim=-1)
        clean_prob = clean_probs.gather(-1, top_token.unsqueeze(-1))
        corrupted_prob = corrupted_probs.gather(-1, top_token.unsqueeze(-1))
        patched_prob = patched_probs.gather(-1, top_token.unsqueeze(-1))
        numerator = patched_prob - corrupted_prob
        denominator = clean_prob - corrupted_prob
        return (numerator / (denominator + 1e-8)).mean().item()
    else:
        raise ValueError(f"Unknown metric: {metric}")
