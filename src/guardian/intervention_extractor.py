"""
Intervention Extractor
======================

Extends ActivationExtractor to support read-write access to activations
via the Guardian Engine.
"""

import torch
import logging
from typing import Optional, Any, Tuple, Union
from src.analysis.activation_extractor import ActivationExtractor

logger = logging.getLogger(__name__)

class InterventionExtractor(ActivationExtractor):
    """
    Extractor that allows modifying activations during the forward pass.
    """
    def __init__(self, guardian_engine, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.guardian_engine = guardian_engine
        self.layer_idx = kwargs.get('layer_idx', 0) # Need to know which layer this is

    def _intervention_hook(self, module, input_hook, output_hook) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Hook that passes activations to the Guardian Engine and returns the modified output.
        """
        # 1. Identify the tensor in the output (handle tuples for Transformers)
        if isinstance(output_hook, torch.Tensor):
            output_tensor = output_hook
            is_tuple = False
        elif isinstance(output_hook, tuple) and len(output_hook) > 0 and isinstance(output_hook[0], torch.Tensor):
            output_tensor = output_hook[0]
            is_tuple = True
        else:
            # Unknown format, skip intervention
            return output_hook

        # 2. Pass to Guardian Engine (Synchronous, Blocking)
        # We do NOT detach here if we want gradients to flow back (though usually for inference we don't need gradients)
        # But for intervention, we need to return a tensor that is part of the graph if we were training,
        # or just a modified tensor for inference.
        
        # The Guardian Engine expects a tensor and returns a tensor.
        try:
            modified_tensor = self.guardian_engine.process_activation(self.layer_idx, output_tensor)
        except Exception as e:
            logger.error(f"Guardian intervention failed: {e}")
            modified_tensor = output_tensor

        # 3. Reconstruct output
        if is_tuple:
            # Create a new tuple with the modified tensor as the first element
            return (modified_tensor,) + output_hook[1:]
        else:
            return modified_tensor

    def register_intervention_hook(self, layer_module, layer_idx: int):
        """Register the intervention hook on a specific layer."""
        self.layer_idx = layer_idx
        return layer_module.register_forward_hook(self._intervention_hook)
