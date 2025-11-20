"""
Guardian Policies
=================

Decision logic for interventions based on probe metrics.
Implements FlowRL concepts to balance creativity (entropy) and coherence.
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from .swireasoning import SwiReasoningTrace, ThinkingBlock, ReasoningSwitch
import time

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

class SwiReasoningPolicy(BasePolicy):
    """
    Policy that implements the SwiReasoning logic:
    Dynamically switches between 'explicit' (normal generation) and 'latent' (internal thought)
    modes based on entropy trends.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.trace = SwiReasoningTrace(problem_id=f"session_{int(time.time())}")
        self.current_mode = "explicit" # Start in explicit mode
        self.entropy_history = []
        self.history_window = config.get('history_window', 5)
        self.switch_threshold = config.get('switch_threshold', 0.1) # Slope threshold
        self.entropy_min = config.get('entropy_min', 0.5)
        self.entropy_max = config.get('entropy_max', 1.5)
        
        # Start first block
        self.current_block = ThinkingBlock(
            block_id=0,
            block_type="explicit",
            start_token_idx=0
        )
        self.trace.add_block(self.current_block)
        self.token_counter = 0

    def decide(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        self.token_counter += 1
        entropy = metrics.get('entropy', 0.0)
        self.entropy_history.append(entropy)
        
        # Update current block stats
        self.current_block.entropy_values.append(entropy)
        
        # Calculate trend (simple slope)
        slope = 0.0
        if len(self.entropy_history) >= 2:
            recent = self.entropy_history[-self.history_window:]
            if len(recent) > 1:
                # Simple rise/run
                slope = (recent[-1] - recent[0]) / len(recent)

        action = {'action': 'none'}
        
        # Logic:
        # 1. Explicit -> Latent: If confidence is high (entropy low) and stable
        if self.current_mode == "explicit":
            # Only switch if we have enough history to be sure it's stable
            if len(self.entropy_history) >= self.history_window:
                if entropy < self.entropy_min and abs(slope) < 0.05:
                    # Switch to Latent
                    self._switch_mode("latent", "high_confidence")
                
        # 2. Latent -> Explicit: If uncertainty rises (entropy increases)
        elif self.current_mode == "latent":
            if slope > self.switch_threshold or entropy > self.entropy_max:
                # Switch to Explicit
                self._switch_mode("explicit", "uncertainty_rise")

        # Apply continuous intervention based on mode
        if self.current_mode == "latent":
            # Latent Mode: Increase temperature to encourage diverse "thinking"
            action = {
                'action': 'scale_logits',
                'params': {'temperature': 1.5} 
            }
        else:
            # Explicit Mode: Decrease temperature for focused "speaking"
            action = {
                'action': 'scale_logits',
                'params': {'temperature': 0.7}
            }

        # Add trace info to action for visualization
        action['swireasoning_trace'] = {
            'mode': self.current_mode,
            'block_id': self.current_block.block_id,
            'entropy_trend': slope
        }
        
        return action

    def _switch_mode(self, new_mode: str, reason: str):
        # Close current block
        self.current_block.end_token_idx = self.token_counter
        
        # Create switch event
        switch = ReasoningSwitch(
            switch_id=len(self.trace.switches),
            from_block_id=self.current_block.block_id,
            to_block_id=self.current_block.block_id + 1,
            reason=reason
        )
        self.trace.add_switch(switch)
        
        # Start new block
        self.current_mode = new_mode
        self.current_block = ThinkingBlock(
            block_id=self.current_block.block_id + 1,
            block_type=new_mode,
            start_token_idx=self.token_counter
        )
        self.trace.add_block(self.current_block)
        logger.info(f"SwiReasoning: Switched to {new_mode} ({reason})")

class PolicyManager:
    """
    Factory and manager for policies.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        policy_type = config.get('policy_type', 'simple')
        
        if policy_type == 'swireasoning':
            logger.info("Initializing SwiReasoning Policy")
            self.active_policy = SwiReasoningPolicy(config)
        else:
            logger.info("Initializing Simple Threshold Policy")
            self.active_policy = SimpleThresholdPolicy(config)

    def decide(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Make an intervention decision based on metrics.
        """
        return self.active_policy.decide(metrics)
