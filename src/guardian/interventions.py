"""
Guardian Interventions
======================

Primitives for modifying activation tensors (noise, steering, ablation).
"""

import torch
from typing import Optional

class InterventionManager:
    """
    Executes interventions on tensors.
    """
    def __init__(self):
        pass

    def inject_noise(self, tensor: torch.Tensor, std: float = 0.1) -> torch.Tensor:
        """
        Inject Gaussian noise into the activation tensor.
        Used to break out of repetitive loops or low-entropy states.
        """
        noise = torch.randn_like(tensor) * std
        return tensor + noise

    def apply_steering_vector(self, tensor: torch.Tensor, vector: Optional[torch.Tensor] = None, coeff: float = 1.0) -> torch.Tensor:
        """
        Add a steering vector to the activation.
        Used to guide the model towards a specific concept or style.
        """
        if vector is None:
            # If no vector provided, we can't steer. 
            # Fallback: maybe dampen the activation slightly?
            return tensor
            
        # Ensure vector is on the same device
        if vector.device != tensor.device:
            vector = vector.to(tensor.device)
            
        return tensor + (vector * coeff)

    def ablate_neurons(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Zero out specific neurons based on a mask.
        """
        if mask.device != tensor.device:
            mask = mask.to(tensor.device)
            
        return tensor * mask
        return tensor

    def scale_logits(self, tensor: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Scale logits to adjust sampling temperature.
        Higher temperature (>1.0) = flatter distribution (more random).
        Lower temperature (<1.0) = sharper distribution (more deterministic).
        """
        if temperature == 1.0:
            return tensor
        return tensor / temperature
