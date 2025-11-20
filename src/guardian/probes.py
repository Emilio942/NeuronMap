"""
Guardian Probes
===============

Sensors for analyzing hidden layer activations in real-time.
Designed for zero-copy GPU execution to minimize latency.
"""

import torch
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class ProbeManager:
    """
    Manages analysis probes (entropy, sparsity, etc.).
    All calculations are performed on the device where the tensor resides.
    """
    def __init__(self):
        pass

    def calculate_entropy(self, tensor: torch.Tensor, epsilon: float = 1e-8) -> float:
        """
        Calculate Shannon entropy of the activation tensor.
        
        Strategy:
        1. Apply Softmax to treat activations as a probability distribution.
        2. Compute H(x) = -sum(p(x) * log(p(x)))
        
        Args:
            tensor: Input activation tensor (batch_size, hidden_dim) or (hidden_dim,)
            epsilon: Small constant for numerical stability
            
        Returns:
            Entropy value (float)
        """
        # Ensure we are working with probabilities
        # If tensor is 2D (batch), we compute mean entropy across batch
        if tensor.dim() > 1:
            probs = torch.softmax(tensor, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)
            return entropy.mean().item()
        else:
            probs = torch.softmax(tensor, dim=0)
            entropy = -torch.sum(probs * torch.log(probs + epsilon))
            return entropy.item()

    def calculate_l2_norm(self, tensor: torch.Tensor) -> float:
        """
        Calculate L2 norm (Euclidean magnitude) of the activations.
        High norm might indicate instability or outliers.
        """
        if tensor.dim() > 1:
            return torch.norm(tensor, p=2, dim=-1).mean().item()
        return torch.norm(tensor, p=2).item()

    def calculate_sparsity(self, tensor: torch.Tensor, threshold: float = 1e-3) -> float:
        """
        Calculate sparsity (fraction of neurons with near-zero activation).
        """
        # Count elements below threshold
        is_zero = torch.abs(tensor) < threshold
        sparsity = is_zero.float().mean()
        return sparsity.item()

    def detect_collapse(self, tensor: torch.Tensor, variance_threshold: float = 1e-4) -> bool:
        """
        Detect if the representation has collapsed (extremely low variance).
        This happens when all neurons output nearly the same value.
        """
        var = torch.var(tensor)
        return var.item() < variance_threshold

    def get_full_report(self, tensor: torch.Tensor) -> Dict[str, float]:
        """
        Run all probes and return a dictionary of metrics.
        """
        # Detach once to ensure we don't track gradients for metrics
        # (Though usually the input tensor is already detached in the hook if intended)
        t = tensor.detach()
        
        return {
            "entropy": self.calculate_entropy(t),
            "l2_norm": self.calculate_l2_norm(t),
            "sparsity": self.calculate_sparsity(t),
            "collapsed": 1.0 if self.detect_collapse(t) else 0.0
        }


class LatentProjector:
    """
    Projects high-dimensional activations into a lower-dimensional latent space
    for efficient analysis by the Guardian.
    """
    def __init__(self, input_dim: int, target_dim: int, device: torch.device):
        self.projection = torch.nn.Linear(input_dim, target_dim, bias=False).to(device)
        # Initialize with random orthogonal matrix for better preservation of geometry
        torch.nn.init.orthogonal_(self.projection.weight)
        self.device = device

    def project(self, tensor: torch.Tensor) -> torch.Tensor:
        """Project tensor to latent space."""
        if tensor.device != self.device:
            tensor = tensor.to(self.device)
        return self.projection(tensor)
