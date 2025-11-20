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
        """
        try:
            # Ensure we are working with probabilities
            if tensor.dim() > 1:
                # Use Categorical for robust entropy calculation
                # logits=tensor handles softmax internally and stably
                dist = torch.distributions.Categorical(logits=tensor)
                entropy = dist.entropy()
                return entropy.mean().item()
            else:
                dist = torch.distributions.Categorical(logits=tensor)
                return dist.entropy().item()
        except Exception as e:
            logger.warning(f"Entropy calculation failed: {e}")
            return 0.0


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

    def calculate_confidence(self, tensor: torch.Tensor) -> float:
        """
        Estimate confidence based on maximum probability.
        High confidence = 1.0, Low confidence = 0.0.
        """
        if tensor.dim() > 1:
            probs = torch.softmax(tensor, dim=-1)
            max_prob, _ = torch.max(probs, dim=-1)
            return max_prob.mean().item()
        else:
            probs = torch.softmax(tensor, dim=0)
            return torch.max(probs).item()

    def analyze_entropy_trend(self, entropy_history: list[float], window_size: int = 5) -> float:
        """
        Analyze the trend of entropy over the last N steps.
        Returns the slope of the linear regression line.
        Positive slope -> Increasing uncertainty.
        Negative slope -> Increasing confidence.
        """
        if len(entropy_history) < 2:
            return 0.0
            
        recent = entropy_history[-window_size:]
        n = len(recent)
        if n < 2:
            return 0.0
            
        # Simple linear regression slope
        # We use torch for calculation to keep it efficient if we were on GPU, 
        # though history is likely a list of floats on CPU.
        try:
            x = torch.arange(n, dtype=torch.float32)
            y = torch.tensor(recent, dtype=torch.float32)
            
            x_mean = x.mean()
            y_mean = y.mean()
            
            numerator = ((x - x_mean) * (y - y_mean)).sum()
            denominator = ((x - x_mean) ** 2).sum()
            
            if denominator == 0:
                return 0.0
                
            slope = numerator / denominator
            return slope.item()
        except Exception:
            return 0.0

    def get_full_report(self, tensor: torch.Tensor) -> Dict[str, float]:
        """
        Run all probes and return a dictionary of metrics.
        """
        # Detach once to ensure we don't track gradients for metrics
        # (Though usually the input tensor is already detached in the hook if intended)
        t = tensor.detach()
        
        return {
            "entropy": self.calculate_entropy(t),
            "confidence": self.calculate_confidence(t),
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
