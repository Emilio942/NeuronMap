"""
Parameter Sparsity Gate (PSG) Detector
======================================

This module implements detection of Parameter Sparsity Gates (PSGs).
A PSG is defined as a structural component (neuron, attention head, or sub-layer)
that exhibits high sparsity in its parameters or activations, effectively acting
as a gate that only opens for specific, rare features.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class PSGNode:
    """Represents a detected Parameter Sparsity Gate."""
    id: str
    layer: int
    component_type: str  # 'neuron', 'head', 'mlp'
    index: int  # Index within the layer (e.g., neuron index)
    
    # Metrics
    weight_sparsity: float = 0.0  # % of weights close to zero
    activation_sparsity: float = 0.0  # % of inputs where activation is zero/low
    gating_strength: float = 0.0  # Derived metric: how 'gating' the behavior is
    
    # Reaction Data (filled during reaction analysis)
    mean_activation: float = 0.0
    max_activation: float = 0.0
    reaction_profile: List[float] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)

class PSGDetector:
    """Detects Parameter Sparsity Gates in neural networks."""

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    def detect_weight_sparsity(self, threshold: float = 1e-3) -> List[PSGNode]:
        """
        Detects PSGs based on weight sparsity in Linear layers.
        Returns a list of potential PSGNodes.
        """
        psgs = []
        logger.info("Scanning model weights for sparsity...")
        
        layer_idx = 0
        for name, module in self.model.named_modules():
            # Focus on MLP layers (Linear) usually found in Transformers
            if isinstance(module, torch.nn.Linear):
                weights = module.weight.detach()
                
                # Calculate sparsity per neuron (row-wise for output neurons)
                # Shape: [out_features, in_features]
                is_sparse = (torch.abs(weights) < threshold).float()
                sparsity_per_neuron = torch.mean(is_sparse, dim=1).cpu().numpy()
                
                # Identify highly sparse neurons
                for i, sparsity in enumerate(sparsity_per_neuron):
                    if sparsity > 0.5: # Initial filter
                        psgs.append(PSGNode(
                            id=f"{name}_neuron_{i}",
                            layer=layer_idx, # Approximate layer index
                            component_type="neuron",
                            index=i,
                            weight_sparsity=float(sparsity),
                            metadata={"module_name": name}
                        ))
                
                # Increment layer count roughly per block
                if "proj" in name or "fc" in name: 
                    layer_idx += 1
                    
        logger.info(f"Found {len(psgs)} potential PSGs based on weight sparsity.")
        return psgs

    def analyze_activation_sparsity(self, 
                                  texts: List[str], 
                                  candidates: List[PSGNode],
                                  activation_threshold: float = 0.0) -> List[PSGNode]:
        """
        Analyzes activation sparsity for candidate PSGs.
        Updates the candidates with activation metrics.
        """
        if not candidates:
            return []
            
        logger.info(f"Analyzing activation sparsity for {len(candidates)} candidates...")
        
        # Group candidates by module for efficient hooking
        candidates_by_module = {}
        for psg in candidates:
            mod_name = psg.metadata["module_name"]
            if mod_name not in candidates_by_module:
                candidates_by_module[mod_name] = []
            candidates_by_module[mod_name].append(psg)
            
        # Store activations: {psg_id: [act1, act2, ...]}
        activations_store = {psg.id: [] for psg in candidates}
        
        # Hook function
        def get_hook(module_name):
            def hook(model, input, output):
                # Output shape: [batch, seq_len, hidden_dim] or [batch, hidden_dim]
                # We care about the specific neurons
                if isinstance(output, tuple):
                    output = output[0]
                
                # Flatten batch and seq dimensions
                acts = output.detach().reshape(-1, output.shape[-1])
                
                for psg in candidates_by_module.get(module_name, []):
                    # Extract specific neuron activation
                    neuron_acts = acts[:, psg.index].cpu().numpy()
                    activations_store[psg.id].extend(neuron_acts)
            return hook

        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if name in candidates_by_module:
                hooks.append(module.register_forward_hook(get_hook(name)))
                
        # Run inference
        batch_size = 4
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i:i+batch_size]
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                self.model(**inputs)
                
        # Remove hooks
        for h in hooks:
            h.remove()
            
        # Calculate metrics
        confirmed_psgs = []
        for psg in candidates:
            acts = np.array(activations_store[psg.id])
            if len(acts) == 0:
                continue
            
            # Handle NaNs
            acts = np.nan_to_num(acts, nan=0.0)
                
            # Sparsity: % of activations below threshold
            sparsity = np.mean(acts <= activation_threshold)
            
            # Gating Strength: Combination of sparsity and max activation
            # A good gate is rarely active (high sparsity) but strong when active
            max_act = np.max(acts) if len(acts) > 0 else 0
            # Ensure max_act is non-negative for log1p, though GeLU can be slightly negative
            safe_max_act = max(0.0, float(max_act))
            gating_strength = sparsity * (1.0 + np.log1p(safe_max_act))
            
            psg.activation_sparsity = float(sparsity)
            psg.gating_strength = float(gating_strength)
            psg.mean_activation = float(np.mean(acts))
            psg.max_activation = float(max_act)
            psg.reaction_profile = acts.tolist()[:100] # Store sample for viz
            
            confirmed_psgs.append(psg)
            
        return confirmed_psgs

    def detect(self, texts: List[str], weight_threshold: float = 1e-2) -> List[PSGNode]:
        """Full detection pipeline."""
        candidates = self.detect_weight_sparsity(threshold=weight_threshold)
        # If too many candidates, take top N by weight sparsity to save time
        candidates = sorted(candidates, key=lambda x: x.weight_sparsity, reverse=True)[:500]
        
        results = self.analyze_activation_sparsity(texts, candidates)
        return results
