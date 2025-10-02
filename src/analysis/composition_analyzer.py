"""
Attention Head Composition Analysis Module

This module implements analysis of attention head composition patterns
and connections between attention heads across layers.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .circuits import (
    NeuralCircuit, CircuitComponent, CircuitConnection, ComponentType,
    create_attention_head_component
)

logger = logging.getLogger(__name__)


@dataclass
class HeadCompositionResult:
    """Result of attention head composition analysis."""
    source_layer: int
    source_head: int
    target_layer: int
    target_head: int
    composition_strength: float
    ov_composition_score: float
    qk_composition_score: float
    residual_contribution: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_layer': self.source_layer,
            'source_head': self.source_head,
            'target_layer': self.target_layer,
            'target_head': self.target_head,
            'composition_strength': self.composition_strength,
            'ov_composition_score': self.ov_composition_score,
            'qk_composition_score': self.qk_composition_score,
            'residual_contribution': self.residual_contribution,
            'metadata': self.metadata
        }


class AttentionHeadCompositionAnalyzer:
    """
    Analyzer for attention head composition patterns.
    
    This class implements methods to analyze how attention heads compose
    with each other, particularly focusing on the mathematical composition
    of their W_OV (output-value) matrices.
    """
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        # Extract model architecture info
        self.num_layers = model.config.n_layer
        self.num_heads = model.config.n_head
        self.d_model = model.config.n_embd
        self.d_head = self.d_model // self.num_heads
        
        logger.info(f"Initialized composition analyzer for model with "
                   f"{self.num_layers} layers, {self.num_heads} heads, d_model={self.d_model}")
    
    def extract_attention_weights(self) -> Dict[str, torch.Tensor]:
        """
        Extract attention weight matrices from the model.
        
        Returns:
            Dictionary mapping weight names to tensors
        """
        weights = {}
        
        for layer_idx in range(self.num_layers):
            layer = self.model.transformer.h[layer_idx]
            attn = layer.attn
            
            # Extract weight matrices
            # Note: GPT-2 uses c_attn for combined Q,K,V and c_proj for output
            c_attn_weight = attn.c_attn.weight  # Shape: [d_model, 3*d_model]
            c_proj_weight = attn.c_proj.weight  # Shape: [d_model, d_model]
            
            # Split c_attn into Q, K, V components
            qkv_weights = c_attn_weight.view(self.d_model, 3, self.d_model)
            q_weight = qkv_weights[:, 0, :]  # Query weights
            k_weight = qkv_weights[:, 1, :]  # Key weights  
            v_weight = qkv_weights[:, 2, :]  # Value weights
            
            # Reshape to per-head format
            q_heads = q_weight.view(self.d_model, self.num_heads, self.d_head)
            k_heads = k_weight.view(self.d_model, self.num_heads, self.d_head)
            v_heads = v_weight.view(self.d_model, self.num_heads, self.d_head)
            o_heads = c_proj_weight.view(self.num_heads, self.d_head, self.d_model)
            
            # Store per-head weights
            for head_idx in range(self.num_heads):
                prefix = f"layer_{layer_idx}_head_{head_idx}"
                weights[f"{prefix}_W_Q"] = q_heads[:, head_idx, :]  # [d_model, d_head]
                weights[f"{prefix}_W_K"] = k_heads[:, head_idx, :]  # [d_model, d_head]
                weights[f"{prefix}_W_V"] = v_heads[:, head_idx, :]  # [d_model, d_head]
                weights[f"{prefix}_W_O"] = o_heads[head_idx, :, :]  # [d_head, d_model]
        
        return weights
    
    def compute_ov_composition(self, source_layer: int, source_head: int,
                             target_layer: int, target_head: int,
                             weights: Dict[str, torch.Tensor]) -> float:
        """
        Compute the composition score between two attention heads via OV matrices.
        
        The composition W_OV1 @ W_OV2 represents how the output of head 1
        can be processed by head 2.
        """
        if target_layer <= source_layer:
            return 0.0  # Only consider forward connections
        
        # Get OV matrices
        source_prefix = f"layer_{source_layer}_head_{source_head}"
        target_prefix = f"layer_{target_layer}_head_{target_head}"
        
        source_v = weights[f"{source_prefix}_W_V"]  # [d_model, d_head]
        source_o = weights[f"{source_prefix}_W_O"]  # [d_head, d_model]
        
        target_v = weights[f"{target_prefix}_W_V"]  # [d_model, d_head]
        target_o = weights[f"{target_prefix}_W_O"]  # [d_head, d_model]
        
        # Compute OV matrices: W_OV = W_O @ W_V
        source_ov = torch.mm(source_o, source_v)  # [d_model, d_model]
        target_ov = torch.mm(target_o, target_v)   # [d_model, d_model]
        
        # Compute composition: how much does source_ov affect target_ov
        composition_matrix = torch.mm(target_ov, source_ov)  # [d_model, d_model]
        
        # Measure strength of composition using Frobenius norm
        composition_strength = torch.norm(composition_matrix, p='fro').item()
        
        # Normalize by the norms of individual OV matrices
        source_norm = torch.norm(source_ov, p='fro').item()
        target_norm = torch.norm(target_ov, p='fro').item()
        
        if source_norm > 0 and target_norm > 0:
            normalized_strength = composition_strength / (source_norm * target_norm)
        else:
            normalized_strength = 0.0
        
        return normalized_strength
    
    def compute_qk_composition(self, source_layer: int, source_head: int,
                             target_layer: int, target_head: int,
                             weights: Dict[str, torch.Tensor]) -> float:
        """
        Compute QK composition score between attention heads.
        
        This measures how the output of the source head affects the
        attention patterns of the target head.
        """
        if target_layer <= source_layer:
            return 0.0
        
        source_prefix = f"layer_{source_layer}_head_{source_head}"
        target_prefix = f"layer_{target_layer}_head_{target_head}"
        
        source_o = weights[f"{source_prefix}_W_O"]  # [d_head, d_model]
        target_q = weights[f"{target_prefix}_W_Q"]  # [d_model, d_head]
        target_k = weights[f"{target_prefix}_W_K"]  # [d_model, d_head]
        
        # How much does source output affect target queries and keys
        q_effect = torch.mm(target_q.T, source_o.T)  # [d_head, d_head]
        k_effect = torch.mm(target_k.T, source_o.T)  # [d_head, d_head]
        
        # Measure the strength of these effects
        q_strength = torch.norm(q_effect, p='fro').item()
        k_strength = torch.norm(k_effect, p='fro').item()
        
        # Normalize
        source_norm = torch.norm(source_o, p='fro').item()
        target_q_norm = torch.norm(target_q, p='fro').item()
        target_k_norm = torch.norm(target_k, p='fro').item()
        
        if source_norm > 0 and target_q_norm > 0 and target_k_norm > 0:
            normalized_q = q_strength / (source_norm * target_q_norm)
            normalized_k = k_strength / (source_norm * target_k_norm)
            return (normalized_q + normalized_k) / 2
        else:
            return 0.0
    
    def analyze_residual_contribution(self, layer: int, head: int,
                                    weights: Dict[str, torch.Tensor]) -> float:
        """
        Analyze how much this head contributes to the residual stream.
        
        This is measured by the norm of the OV matrix.
        """
        prefix = f"layer_{layer}_head_{head}"
        
        v_weight = weights[f"{prefix}_W_V"]  # [d_model, d_head]
        o_weight = weights[f"{prefix}_W_O"]  # [d_head, d_model]
        
        # Compute OV matrix
        ov_matrix = torch.mm(o_weight, v_weight)  # [d_model, d_model]
        
        # Measure contribution strength
        contribution = torch.norm(ov_matrix, p='fro').item()
        
        # Normalize by model dimension
        normalized_contribution = contribution / self.d_model
        
        return normalized_contribution
    
    def compute_all_head_compositions(self, max_layer_gap: int = 3) -> List[HeadCompositionResult]:
        """
        Compute composition scores between all pairs of attention heads.
        
        Args:
            max_layer_gap: Maximum number of layers between heads to consider
            
        Returns:
            List of composition results
        """
        logger.info("Computing all head compositions...")
        
        # Extract weights
        weights = self.extract_attention_weights()
        
        results = []
        total_pairs = 0
        
        for source_layer in range(self.num_layers):
            for source_head in range(self.num_heads):
                # Compute residual contribution for this head
                residual_contrib = self.analyze_residual_contribution(
                    source_layer, source_head, weights
                )
                
                for target_layer in range(source_layer + 1,
                                        min(source_layer + max_layer_gap + 1, self.num_layers)):
                    for target_head in range(self.num_heads):
                        # Compute composition scores
                        ov_score = self.compute_ov_composition(
                            source_layer, source_head, target_layer, target_head, weights
                        )
                        
                        qk_score = self.compute_qk_composition(
                            source_layer, source_head, target_layer, target_head, weights
                        )
                        
                        # Overall composition strength
                        composition_strength = (ov_score + qk_score) / 2
                        
                        result = HeadCompositionResult(
                            source_layer=source_layer,
                            source_head=source_head,
                            target_layer=target_layer,
                            target_head=target_head,
                            composition_strength=composition_strength,
                            ov_composition_score=ov_score,
                            qk_composition_score=qk_score,
                            residual_contribution=residual_contrib,
                            metadata={
                                'layer_gap': target_layer - source_layer,
                                'max_layer_gap': max_layer_gap
                            }
                        )
                        
                        results.append(result)
                        total_pairs += 1
        
        logger.info(f"Computed compositions for {total_pairs} head pairs")
        return results
    
    def find_strongest_compositions(self, compositions: List[HeadCompositionResult],
                                  top_k: int = 20) -> List[HeadCompositionResult]:
        """Find the strongest composition connections."""
        sorted_compositions = sorted(compositions,
                                   key=lambda x: x.composition_strength,
                                   reverse=True)
        return sorted_compositions[:top_k]
    
    def create_composition_circuit(self, compositions: List[HeadCompositionResult],
                                 model_name: str, threshold: float = 0.1) -> NeuralCircuit:
        """
        Create a neural circuit from composition analysis results.
        
        Args:
            compositions: List of composition results
            model_name: Name of the model
            threshold: Minimum composition strength to include in circuit
            
        Returns:
            Neural circuit representing head compositions
        """
        circuit = NeuralCircuit(
            circuit_id=f"composition_circuit_{model_name}",
            model_name=model_name,
            description="Circuit showing attention head composition patterns"
        )
        
        # Track which components we've added
        added_components = set()
        
        # Add components and connections for strong compositions
        for comp in compositions:
            if comp.composition_strength >= threshold:
                # Add source component
                source_id = f"{model_name}_attention_head_L{comp.source_layer}_P{comp.source_head}"
                if source_id not in added_components:
                    source_comp = create_attention_head_component(
                        model_name, comp.source_layer, comp.source_head,
                        metadata={
                            'residual_contribution': comp.residual_contribution,
                            'type': 'attention_head'
                        }
                    )
                    circuit.add_component(source_comp)
                    added_components.add(source_id)
                
                # Add target component
                target_id = f"{model_name}_attention_head_L{comp.target_layer}_P{comp.target_head}"
                if target_id not in added_components:
                    target_comp = create_attention_head_component(
                        model_name, comp.target_layer, comp.target_head,
                        metadata={'type': 'attention_head'}
                    )
                    circuit.add_component(target_comp)
                    added_components.add(target_id)
                
                # Add connection
                source_comp = circuit.get_component(source_id)
                target_comp = circuit.get_component(target_id)
                
                if source_comp and target_comp:
                    connection = CircuitConnection(
                        source=source_comp,
                        target=target_comp,
                        weight=comp.composition_strength,
                        connection_type="head_composition",
                        metadata={
                            'ov_score': comp.ov_composition_score,
                            'qk_score': comp.qk_composition_score,
                            'layer_gap': comp.target_layer - comp.source_layer
                        }
                    )
                    circuit.add_connection(connection)
        
        logger.info(f"Created composition circuit with {len(circuit._components)} components "
                   f"and {len(circuit._connections)} connections")
        
        return circuit
    
    def analyze_composition_patterns(self, compositions: List[HeadCompositionResult]) -> Dict[str, Any]:
        """
        Analyze patterns in head compositions.
        
        Returns:
            Dictionary with analysis results
        """
        if not compositions:
            return {}
        
        # Group by layer gap
        by_layer_gap = {}
        for comp in compositions:
            gap = comp.target_layer - comp.source_layer
            if gap not in by_layer_gap:
                by_layer_gap[gap] = []
            by_layer_gap[gap].append(comp.composition_strength)
        
        # Calculate statistics
        all_strengths = [comp.composition_strength for comp in compositions]
        ov_scores = [comp.ov_composition_score for comp in compositions]
        qk_scores = [comp.qk_composition_score for comp in compositions]
        
        analysis = {
            'total_compositions': len(compositions),
            'strength_statistics': {
                'mean': np.mean(all_strengths),
                'std': np.std(all_strengths),
                'min': np.min(all_strengths),
                'max': np.max(all_strengths),
                'percentiles': {
                    '50': np.percentile(all_strengths, 50),
                    '90': np.percentile(all_strengths, 90),
                    '95': np.percentile(all_strengths, 95),
                    '99': np.percentile(all_strengths, 99)
                }
            },
            'ov_score_statistics': {
                'mean': np.mean(ov_scores),
                'std': np.std(ov_scores),
                'max': np.max(ov_scores)
            },
            'qk_score_statistics': {
                'mean': np.mean(qk_scores),
                'std': np.std(qk_scores),
                'max': np.max(qk_scores)
            },
            'layer_gap_analysis': {}
        }
        
        # Analyze by layer gap
        for gap, strengths in by_layer_gap.items():
            analysis['layer_gap_analysis'][gap] = {
                'count': len(strengths),
                'mean_strength': np.mean(strengths),
                'max_strength': np.max(strengths),
                'strong_connections': len([s for s in strengths if s > 0.1])
            }
        
        return analysis


def save_composition_analysis(compositions: List[HeadCompositionResult], 
                            analysis: Dict[str, Any],
                            filepath: Path) -> None:
    """Save composition analysis results to JSON file."""
    results = {
        'compositions': [comp.to_dict() for comp in compositions],
        'analysis': analysis,
        'metadata': {
            'total_compositions': len(compositions),
            'analysis_timestamp': str(Path().cwd())
        }
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Composition analysis saved to {filepath}")


if __name__ == "__main__":
    print("Attention Head Composition Analyzer loaded!")
    print("Use with a transformer model to analyze head compositions.")
