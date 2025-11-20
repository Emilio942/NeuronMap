"""
Guardian Engine
===============

Core engine that orchestrates the interaction between the main model and the
guardian components (probes, policies, interventions).
"""

import torch
import logging
from typing import Dict, Any, Optional, List

from .probes import ProbeManager
from .policies import PolicyManager
from .interventions import InterventionManager

logger = logging.getLogger(__name__)

class GuardianEngine:
    """
    Main controller for the Guardian Network.
    Orchestrates the Introspection -> Analysis -> Intervention loop.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.mode = config.get('mode', 'monitoring')
        self.intervention_layers = set(config.get('intervention_layers', []))
        
        # Initialize components
        self.probes = ProbeManager()
        self.policies = PolicyManager(config)
        self.interventions = InterventionManager()
        
        # Optional: Load secondary guardian model if configured
        self.guardian_model = None
        if config.get('guardian_model_path'):
            self._load_guardian_model(config['guardian_model_path'])
            
        # State tracking for visualization
        self.last_metrics: Dict[str, float] = {}
        self.last_action: str = "none"
            
        logger.info(f"Guardian Engine initialized (Mode: {self.mode}, Enabled: {self.enabled})")

    def _load_guardian_model(self, path: str):
        """Load the secondary guardian network."""
        try:
            # Placeholder for loading a specific model architecture (e.g., SAE)
            # self.guardian_model = torch.load(path)
            logger.info(f"Loaded guardian model from {path}")
        except Exception as e:
            logger.error(f"Failed to load guardian model: {e}")

    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration at runtime."""
        self.config.update(new_config)
        self.enabled = self.config.get('enabled', False)
        self.mode = self.config.get('mode', 'monitoring')
        self.intervention_layers = set(self.config.get('intervention_layers', []))
        
        # Update policy config
        if hasattr(self.policies, 'active_policy'):
            # Re-initialize policy manager with new config
            # We create a new PolicyManager to handle the logic of choosing the right policy class
            from .policies import PolicyManager
            self.policies = PolicyManager(self.config)
            
        logger.info(f"Guardian Engine config updated (Mode: {self.mode}, Enabled: {self.enabled})")

    def process_activation(self, layer_idx: int, activation: torch.Tensor) -> torch.Tensor:
        """
        Process a single layer activation through the Guardian pipeline.
        
        Args:
            layer_idx: Index of the current layer
            activation: The activation tensor from the main model
            
        Returns:
            The (potentially modified) activation tensor
        """
        if not self.enabled:
            return activation

        # 1. Check if we should intervene at this layer
        if layer_idx not in self.intervention_layers and not self.config.get('monitor_all_layers', False):
            return activation

        # 2. Introspection (Probes)
        # Calculate metrics on the raw activation
        metrics = self.probes.get_full_report(activation)
        self.last_metrics = metrics
        self.last_action = "none"
        self.last_decision = {}
        
        # Log metrics if in monitoring mode
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Layer {layer_idx} Metrics: {metrics}")

        # 3. Analysis & Decision (Policy)
        if self.mode == 'intervention':
            decision = self.policies.decide(metrics)
            action = decision.get('action')
            self.last_action = action
            self.last_decision = decision
            
            # 4. Intervention (Action)
            if action == 'inject_noise':
                params = decision.get('params', {})
                return self.interventions.inject_noise(activation, **params)
            elif action == 'apply_steering':
                params = decision.get('params', {})
                # Assuming we have a steering vector available, or we generate one
                # For now, we might just dampen or do nothing if no vector is provided
                return self.interventions.apply_steering_vector(activation, **params)
            elif action == 'scale_logits':
                params = decision.get('params', {})
                return self.interventions.scale_logits(activation, **params)
        
        return activation
