"""
Guardian Policies
=================

Decision logic for interventions based on probe metrics.
Implements FlowRL concepts to balance creativity (entropy) and coherence.
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BasePolicy(ABC):
    """Abstract base class for Guardian policies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def decide(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Decide on an action based on metrics.
        Returns a dictionary describing the action (e.g., {'action': 'inject_noise', 'params': {...}}).
        """
        pass

class SimpleThresholdPolicy(BasePolicy):
    """
    A simple policy based on static thresholds for entropy.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.entropy_min = config.get('entropy_min', 0.5)
        self.entropy_max = config.get('entropy_max', 2.5)
        self.noise_std = config.get('noise_std', 0.1)
        self.steering_coeff = config.get('steering_coeff', 1.0)

    def decide(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        entropy = metrics.get('entropy', 0.0)
        
        # Case 1: Entropy too low -> Model is stuck/repetitive -> Inject Noise
        if entropy < self.entropy_min:
            logger.debug(f"Entropy {entropy:.4f} < {self.entropy_min}: Triggering Noise Injection")
            return {
                'action': 'inject_noise',
                'params': {'std': self.noise_std}
            }
            
        # Case 2: Entropy too high -> Model is chaotic -> Apply Steering/Dampening
        elif entropy > self.entropy_max:
            logger.debug(f"Entropy {entropy:.4f} > {self.entropy_max}: Triggering Steering")
            return {
                'action': 'apply_steering',
                'params': {'coeff': self.steering_coeff}
            }
            
        # Case 3: Flow State -> No intervention
        return {'action': 'none'}

class PolicyManager:
    """
    Factory and manager for policies.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Currently defaulting to SimpleThresholdPolicy
        # In the future, we could load RL agents here
        self.active_policy = SimpleThresholdPolicy(config)

    def decide(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Make an intervention decision based on metrics.
        """
        return self.active_policy.decide(metrics)
