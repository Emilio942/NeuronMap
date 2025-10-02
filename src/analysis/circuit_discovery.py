"""
Circuit Discovery System for NeuronMap
=====================================

This module implements advanced circuit discovery techniques for understanding
functional relationships and information flow in neural networks.

Implements tasks B1-B6 from "Die Entdeckung von Circuits" feature block:
- B1: Attention Head Composition Analysis
- B2: Neuron-to-Head Connection Analysis  
- B3: Graph/Circuit Data Structure
- B4: Induction Head Scanner
- B5: Copying/Saliency Head Scanner
- B6: Circuit Verification with Causal Tools
"""

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class CircuitType(Enum):
    """Types of circuits that can be discovered."""
    INDUCTION = "induction"
    COPYING = "copying"
    SALIENCY = "saliency"
    COMPOSITION = "composition"
    CUSTOM = "custom"


class NodeType(Enum):
    """Types of nodes in circuit graphs."""
    ATTENTION_HEAD = "attention_head"
    MLP_NEURON = "mlp_neuron"
    EMBEDDING = "embedding"
    OUTPUT = "output"


@dataclass
class CircuitNode:
    """Represents a node in a neural circuit."""
    node_id: str
    node_type: NodeType
    layer: int
    position: int  # Head index or neuron index
    layer_name: str
    activation_pattern: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.node_id:
            self.node_id = f"{self.layer_name}_{self.position}"


@dataclass 
class CircuitEdge:
    """Represents a connection between circuit nodes."""
    source: str  # Source node ID
    target: str  # Target node ID
    weight: float
    connection_type: str
    evidence: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Circuit:
    """Represents a discovered functional circuit."""
    circuit_id: str
    circuit_type: CircuitType
    nodes: List[CircuitNode]
    edges: List[CircuitEdge]
    function: str
    confidence: float
    verification_results: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert circuit to NetworkX graph."""
        G = nx.DiGraph()
        
        # Add nodes
        for node in self.nodes:
            G.add_node(
                node.node_id,
                node_type=node.node_type.value,
                layer=node.layer,
                position=node.position,
                layer_name=node.layer_name,
                **node.metadata
            )
        
        # Add edges
        for edge in self.edges:
            G.add_edge(
                edge.source,
                edge.target,
                weight=edge.weight,
                connection_type=edge.connection_type,
                evidence=edge.evidence,
                **edge.metadata
            )
        
        return G
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert circuit to dictionary for JSON serialization."""
        return {
            'circuit_id': self.circuit_id,
            'circuit_type': self.circuit_type.value,
            'function': self.function,
            'confidence': self.confidence,
            'nodes': [
                {
                    'node_id': node.node_id,
                    'node_type': node.node_type.value,
                    'layer': node.layer,
                    'position': node.position,
                    'layer_name': node.layer_name,
                    'metadata': node.metadata
                }
                for node in self.nodes
            ],
            'edges': [
                {
                    'source': edge.source,
                    'target': edge.target,
                    'weight': edge.weight,
                    'connection_type': edge.connection_type,
                    'evidence': edge.evidence,
                    'metadata': edge.metadata
                }
                for edge in self.edges
            ],
            'verification_results': self.verification_results,
            'metadata': self.metadata
        }


class CircuitAnalyzer:
    """Main class for discovering and analyzing neural circuits."""
    
    def __init__(self, model: nn.Module, model_name: str, device: torch.device = None):
        """Initialize circuit analyzer."""
        self.model = model
        self.model_name = model_name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract model architecture info
        self.layer_info = self._extract_layer_info()
        self.attention_layers = self._find_attention_layers()
        self.mlp_layers = self._find_mlp_layers()
        
        logger.info(f"CircuitAnalyzer initialized for {model_name}")
        logger.info(f"Found {len(self.attention_layers)} attention layers, {len(self.mlp_layers)} MLP layers")
    
    def _extract_layer_info(self) -> Dict[str, Any]:
        """Extract information about model layers."""
        layer_info = {
            'total_layers': 0,
            'layer_names': [],
            'layer_types': {}
        }
        
        for name, module in self.model.named_modules():
            layer_info['layer_names'].append(name)
            layer_info['layer_types'][name] = type(module).__name__
        
        # Count transformer layers (common pattern)
        transformer_layers = [name for name in layer_info['layer_names'] 
                            if any(pattern in name for pattern in ['h.', 'layer.', 'encoder.layer.'])]
        layer_info['total_layers'] = len(set(['.'.join(name.split('.')[:3]) for name in transformer_layers]))
        
        return layer_info
    
    def _find_attention_layers(self) -> List[str]:
        """Find all attention layers in the model."""
        attention_layers = []
        for name, module in self.model.named_modules():
            if any(attn_pattern in name.lower() for attn_pattern in ['attn', 'attention']):
                # Skip dropout and projection layers, focus on main attention
                if not any(skip in name.lower() for skip in ['dropout', 'proj', 'c_attn', 'c_proj']):
                    attention_layers.append(name)
        return attention_layers
    
    def _find_mlp_layers(self) -> List[str]:
        """Find all MLP layers in the model."""
        mlp_layers = []
        for name, module in self.model.named_modules():
            if any(mlp_pattern in name.lower() for mlp_pattern in ['mlp', 'feed_forward', 'ffn']):
                # Skip individual components, focus on main MLP modules
                if not any(skip in name.lower() for skip in ['dropout', 'act', 'c_fc', 'c_proj']):
                    mlp_layers.append(name)
        return mlp_layers
    
    # B1: Attention Head Composition Analysis
    def analyze_attention_composition(
        self,
        input_ids: torch.Tensor,
        layer_pairs: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze composition of attention heads across layers.
        
        Implements B1: Analyse der Attention Head Komposition
        """
        logger.info("Starting attention head composition analysis")
        
        if layer_pairs is None:
            # Analyze consecutive layers
            max_layer = self.layer_info['total_layers']
            layer_pairs = [(i, i+1) for i in range(max_layer-1)]
        
        composition_results = {}
        
        # Get attention weights for all layers
        attention_weights = self._extract_attention_weights(input_ids)
        
        for early_layer, late_layer in layer_pairs:
            if f"layer_{early_layer}" not in attention_weights or f"layer_{late_layer}" not in attention_weights:
                continue
                
            early_attn = attention_weights[f"layer_{early_layer}"]  # [batch, heads, seq, seq]
            late_attn = attention_weights[f"layer_{late_layer}"]
            
            # Compute composition matrices (simplified OV circuit analysis)
            composition_scores = self._compute_composition_scores(early_attn, late_attn)
            
            composition_results[f"layer_{early_layer}_to_{late_layer}"] = {
                'layer_pair': (early_layer, late_layer),
                'composition_scores': composition_scores,
                'max_composition': float(composition_scores.max()),
                'mean_composition': float(composition_scores.mean())
            }
        
        logger.info(f"Completed composition analysis for {len(layer_pairs)} layer pairs")
        return composition_results
    
    def _extract_attention_weights(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract attention weights from all layers."""
        attention_weights = {}
        
        def attention_hook(name):
            def hook(module, input, output):
                if hasattr(output, 'attentions') and output.attentions is not None:
                    attention_weights[name] = output.attentions.detach()
                elif isinstance(output, tuple) and len(output) > 1:
                    # Some models return (output, attention_weights)
                    if output[1] is not None:
                        attention_weights[name] = output[1].detach()
            return hook
        
        # Register hooks
        hooks = []
        for i, layer_name in enumerate(self.attention_layers):
            layer_key = f"layer_{i}"
            hook = self._get_module_by_name(layer_name).register_forward_hook(attention_hook(layer_key))
            hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_ids)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_weights
    
    def _compute_composition_scores(self, early_attn: torch.Tensor, late_attn: torch.Tensor) -> torch.Tensor:
        """Compute composition scores between attention layers."""
        # Simplified composition analysis
        # In practice, this would involve OV matrix multiplication
        batch_size, num_heads, seq_len, _ = early_attn.shape
        
        # Compute attention pattern similarity (simplified)
        early_patterns = early_attn.mean(dim=0)  # Average over batch
        late_patterns = late_attn.mean(dim=0)
        
        # Compute correlation between attention patterns
        composition_scores = torch.zeros(num_heads, num_heads)
        
        for early_head in range(num_heads):
            for late_head in range(num_heads):
                early_pattern = early_patterns[early_head].flatten()
                late_pattern = late_patterns[late_head].flatten()
                
                # Compute correlation
                correlation = torch.corrcoef(torch.stack([early_pattern, late_pattern]))[0, 1]
                composition_scores[early_head, late_head] = correlation if not torch.isnan(correlation) else 0.0
        
        return composition_scores
    
    # B2: Neuron-to-Head Connection Analysis  
    def analyze_neuron_to_head_connections(
        self,
        input_ids: torch.Tensor,
        target_layers: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Analyze connections from MLP neurons to attention heads.
        
        Implements B2: Analyse der Neuron-zu-Head Verbindung
        """
        logger.info("Starting neuron-to-head connection analysis")
        
        if target_layers is None:
            target_layers = list(range(min(6, self.layer_info['total_layers'])))
        
        connection_results = {}
        
        # Extract activations from MLP and attention layers
        mlp_activations = self._extract_mlp_activations(input_ids, target_layers)
        attention_activations = self._extract_attention_activations(input_ids, target_layers)
        
        for layer in target_layers:
            mlp_key = f"mlp_{layer}"
            attn_key = f"attn_{layer+1}"  # Next layer attention
            
            if mlp_key not in mlp_activations or attn_key not in attention_activations:
                continue
            
            mlp_acts = mlp_activations[mlp_key]  # [batch, seq, hidden]
            attn_acts = attention_activations[attn_key]  # [batch, heads, seq, hidden]
            
            # Compute influence scores
            influence_scores = self._compute_neuron_head_influence(mlp_acts, attn_acts)
            
            connection_results[f"layer_{layer}_to_{layer+1}"] = {
                'source_layer': layer,
                'target_layer': layer + 1,
                'influence_scores': influence_scores,
                'top_connections': self._find_top_connections(influence_scores, top_k=10)
            }
        
        logger.info(f"Completed neuron-to-head analysis for {len(target_layers)} layers")
        return connection_results
    
    def _extract_mlp_activations(self, input_ids: torch.Tensor, layers: List[int]) -> Dict[str, torch.Tensor]:
        """Extract MLP activations."""
        activations = {}
        
        def mlp_hook(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        hooks = []
        for layer in layers:
            for layer_name in self.mlp_layers:
                if f".{layer}." in layer_name or f"h.{layer}" in layer_name:
                    hook = self._get_module_by_name(layer_name).register_forward_hook(mlp_hook(f"mlp_{layer}"))
                    hooks.append(hook)
                    break
        
        with torch.no_grad():
            _ = self.model(input_ids)
        
        for hook in hooks:
            hook.remove()
            
        return activations
    
    def _extract_attention_activations(self, input_ids: torch.Tensor, layers: List[int]) -> Dict[str, torch.Tensor]:
        """Extract attention activations (not weights, but the attended values)."""
        activations = {}
        
        def attn_hook(name):
            def hook(module, input, output):
                # Get the attended values (output of attention)
                if isinstance(output, tuple):
                    activations[name] = output[0].detach()
                else:
                    activations[name] = output.detach()
            return hook
        
        hooks = []
        for layer in layers:
            for layer_name in self.attention_layers:
                if f".{layer}." in layer_name or f"h.{layer}" in layer_name:
                    hook = self._get_module_by_name(layer_name).register_forward_hook(attn_hook(f"attn_{layer}"))
                    hooks.append(hook)
                    break
        
        with torch.no_grad():
            _ = self.model(input_ids)
        
        for hook in hooks:
            hook.remove()
            
        return activations
    
    def _compute_neuron_head_influence(self, mlp_acts: torch.Tensor, attn_acts: torch.Tensor) -> np.ndarray:
        """Compute influence scores between MLP neurons and attention heads."""
        # Simplified influence computation using correlation
        batch_size, seq_len, mlp_dim = mlp_acts.shape
        
        if len(attn_acts.shape) == 4:  # [batch, heads, seq, hidden]
            batch_size, num_heads, seq_len_attn, attn_dim = attn_acts.shape
        else:  # [batch, seq, hidden]
            batch_size, seq_len_attn, attn_dim = attn_acts.shape
            num_heads = 1
            attn_acts = attn_acts.unsqueeze(1)
        
        # Average over batch and sequence dimensions
        mlp_mean = mlp_acts.mean(dim=(0, 1))  # [mlp_dim]
        attn_mean = attn_acts.mean(dim=(0, 2))  # [heads, attn_dim]
        
        # Compute correlation matrix
        influence_matrix = np.zeros((mlp_dim, num_heads))
        
        for neuron_idx in range(min(mlp_dim, 1000)):  # Limit for computational efficiency
            for head_idx in range(num_heads):
                # Simple correlation between neuron activation and head output
                neuron_act = mlp_acts[:, :, neuron_idx].flatten().cpu().numpy()
                head_act = attn_acts[:, head_idx, :, :].flatten().cpu().numpy()
                
                if len(head_act) > len(neuron_act):
                    head_act = head_act[:len(neuron_act)]
                elif len(neuron_act) > len(head_act):
                    neuron_act = neuron_act[:len(head_act)]
                
                correlation = np.corrcoef(neuron_act, head_act)[0, 1]
                influence_matrix[neuron_idx, head_idx] = correlation if not np.isnan(correlation) else 0.0
        
        return influence_matrix
    
    def _find_top_connections(self, influence_scores: np.ndarray, top_k: int = 10) -> List[Tuple[int, int, float]]:
        """Find top neuron-to-head connections."""
        flat_indices = np.argsort(np.abs(influence_scores).flatten())[-top_k:]
        top_connections = []
        
        for flat_idx in reversed(flat_indices):
            neuron_idx, head_idx = np.unravel_index(flat_idx, influence_scores.shape)
            score = influence_scores[neuron_idx, head_idx]
            top_connections.append((int(neuron_idx), int(head_idx), float(score)))
        
        return top_connections
    
    # B4: Induction Head Scanner
    def scan_induction_heads(
        self,
        input_ids: torch.Tensor,
        min_confidence: float = 0.7
    ) -> List[Circuit]:
        """
        Scan for induction heads - heads that attend to previous instances of current tokens.
        
        Implements B4: "Scanner" für Induction Heads
        """
        logger.info("Scanning for induction heads")
        
        # Create sequences with repeated patterns for induction detection
        test_sequences = self._create_induction_test_sequences(input_ids)
        induction_circuits = []
        
        for seq_idx, test_seq in enumerate(test_sequences):
            attention_weights = self._extract_attention_weights(test_seq)
            
            for layer_name, attn_weights in attention_weights.items():
                if attn_weights is None:
                    continue
                
                # Analyze each attention head
                batch_size, num_heads, seq_len, _ = attn_weights.shape
                
                for head_idx in range(num_heads):
                    head_attn = attn_weights[0, head_idx]  # First batch item
                    
                    induction_score = self._compute_induction_score(head_attn, test_seq)
                    
                    if induction_score > min_confidence:
                        # Create induction circuit
                        circuit = self._create_induction_circuit(
                            layer_name, head_idx, induction_score, test_seq
                        )
                        induction_circuits.append(circuit)
        
        logger.info(f"Found {len(induction_circuits)} induction head circuits")
        return induction_circuits
    
    def _create_induction_test_sequences(self, base_input: torch.Tensor) -> List[torch.Tensor]:
        """Create test sequences for induction head detection."""
        # Create sequences with repeated patterns
        batch_size, seq_len = base_input.shape
        test_sequences = []
        
        # Pattern 1: ABC...ABC (direct repetition)
        if seq_len >= 8:
            pattern_len = min(4, seq_len // 2)
            repeated_seq = torch.cat([
                base_input[:, :pattern_len],
                base_input[:, :pattern_len]
            ], dim=1)
            test_sequences.append(repeated_seq)
        
        # Pattern 2: Random repeated elements
        if seq_len >= 6:
            random_seq = torch.randint_like(base_input[:, :6], low=1, high=1000)
            repeated_pattern = torch.cat([
                random_seq[:, :3],
                random_seq[:, :3]
            ], dim=1)
            test_sequences.append(repeated_pattern)
        
        return test_sequences
    
    def _compute_induction_score(self, attention_matrix: torch.Tensor, input_seq: torch.Tensor) -> float:
        """Compute induction score for an attention head."""
        seq_len = attention_matrix.shape[0]
        if seq_len < 4:
            return 0.0
        
        induction_evidence = 0.0
        pattern_len = seq_len // 2
        
        # Check if head attends to previous instances of current token
        for pos in range(pattern_len, seq_len):
            current_token = input_seq[0, pos].item()
            
            # Find previous instances of this token
            prev_positions = []
            for prev_pos in range(pos):
                if input_seq[0, prev_pos].item() == current_token:
                    prev_positions.append(prev_pos)
            
            if prev_positions:
                # Check attention to previous instances
                prev_attention = sum(attention_matrix[pos, prev_pos].item() for prev_pos in prev_positions)
                induction_evidence += prev_attention
        
        # Normalize by sequence length
        return induction_evidence / max(1, seq_len - pattern_len)
    
    def _create_induction_circuit(
        self,
        layer_name: str,
        head_idx: int,
        confidence: float,
        test_seq: torch.Tensor
    ) -> Circuit:
        """Create a circuit object for an induction head."""
        layer_num = self._extract_layer_number(layer_name)
        
        # Create nodes
        head_node = CircuitNode(
            node_id=f"{layer_name}_head_{head_idx}",
            node_type=NodeType.ATTENTION_HEAD,
            layer=layer_num,
            position=head_idx,
            layer_name=layer_name,
            metadata={'head_type': 'induction', 'test_sequence_length': test_seq.shape[1]}
        )
        
        # Create circuit
        circuit = Circuit(
            circuit_id=f"induction_{layer_name}_h{head_idx}",
            circuit_type=CircuitType.INDUCTION,
            nodes=[head_node],
            edges=[],  # Induction heads are primarily single-node circuits
            function=f"Induction head in {layer_name} at position {head_idx}",
            confidence=confidence,
            metadata={
                'detection_method': 'pattern_repetition',
                'test_sequences': 1
            }
        )
        
        return circuit
    
    # B5: Copying/Saliency Head Scanner
    def scan_copying_heads(
        self,
        input_ids: torch.Tensor,
        min_confidence: float = 0.6
    ) -> List[Circuit]:
        """
        Scan for copying/saliency heads - heads that primarily copy from important positions.
        
        Implements B5: "Scanner" für Copying/Saliency Heads  
        """
        logger.info("Scanning for copying/saliency heads")
        
        attention_weights = self._extract_attention_weights(input_ids)
        copying_circuits = []
        
        for layer_name, attn_weights in attention_weights.items():
            if attn_weights is None:
                continue
                
            batch_size, num_heads, seq_len, _ = attn_weights.shape
            
            for head_idx in range(num_heads):
                head_attn = attn_weights[0, head_idx]  # First batch item
                
                copying_score = self._compute_copying_score(head_attn)
                
                if copying_score > min_confidence:
                    circuit = self._create_copying_circuit(
                        layer_name, head_idx, copying_score, head_attn
                    )
                    copying_circuits.append(circuit)
        
        logger.info(f"Found {len(copying_circuits)} copying/saliency head circuits")
        return copying_circuits
    
    def _compute_copying_score(self, attention_matrix: torch.Tensor) -> float:
        """Compute copying score for an attention head."""
        seq_len = attention_matrix.shape[0]
        if seq_len < 2:
            return 0.0
        
        # Check for strong attention to specific positions (especially first/last tokens)
        first_token_attention = attention_matrix[:, 0].mean().item()
        last_token_attention = attention_matrix[:, -1].mean().item()
        
        # Check for diagonal attention (position to itself)
        diagonal_attention = torch.diagonal(attention_matrix).mean().item()
        
        # Check concentration (entropy-based measure)
        attention_entropy = self._compute_attention_entropy(attention_matrix)
        max_entropy = np.log(seq_len)
        concentration_score = 1.0 - (attention_entropy / max_entropy) if max_entropy > 0 else 0.0
        
        # Combine scores
        copying_score = max(
            first_token_attention * 0.4 + concentration_score * 0.6,
            last_token_attention * 0.4 + concentration_score * 0.6,
            diagonal_attention * 0.3 + concentration_score * 0.7
        )
        
        return copying_score
    
    def _compute_attention_entropy(self, attention_matrix: torch.Tensor) -> float:
        """Compute entropy of attention distribution."""
        # Average attention distribution across all positions
        avg_attention = attention_matrix.mean(dim=0)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        avg_attention = avg_attention + epsilon
        avg_attention = avg_attention / avg_attention.sum()
        
        # Compute entropy
        entropy = -(avg_attention * torch.log(avg_attention)).sum().item()
        return entropy
    
    def _create_copying_circuit(
        self,
        layer_name: str,
        head_idx: int,
        confidence: float,
        attention_pattern: torch.Tensor
    ) -> Circuit:
        """Create a circuit object for a copying head."""
        layer_num = self._extract_layer_number(layer_name)
        
        # Determine copying type based on attention pattern
        seq_len = attention_pattern.shape[0]
        first_attn = attention_pattern[:, 0].mean().item()
        last_attn = attention_pattern[:, -1].mean().item()
        diag_attn = torch.diagonal(attention_pattern).mean().item()
        
        if first_attn > max(last_attn, diag_attn):
            copying_type = "first_token_copying"
        elif last_attn > max(first_attn, diag_attn):
            copying_type = "last_token_copying"  
        else:
            copying_type = "positional_copying"
        
        # Create nodes
        head_node = CircuitNode(
            node_id=f"{layer_name}_head_{head_idx}",
            node_type=NodeType.ATTENTION_HEAD,
            layer=layer_num,
            position=head_idx,
            layer_name=layer_name,
            metadata={
                'head_type': 'copying',
                'copying_type': copying_type,
                'attention_concentration': confidence
            }
        
        # Create circuit
        circuit = Circuit(
            circuit_id=f"copying_{layer_name}_h{head_idx}",
            circuit_type=CircuitType.COPYING,
            nodes=[head_node],
            edges=[],
            function=f"Copying head ({copying_type}) in {layer_name} at position {head_idx}",
            confidence=confidence,
            metadata={
                'detection_method': 'attention_analysis',
                'copying_type': copying_type
            }
        )
        
        return circuit
    
    def _extract_layer_number(self, layer_name: str) -> int:
        """Extract layer number from layer name."""
        # Handle different naming conventions
        import re
        
        # Pattern: transformer.h.N or transformer.layer.N or layer.N
        patterns = [
            r'\.h\.(\d+)',
            r'\.layer\.(\d+)',
            r'layer_(\d+)',
            r'layer\.(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, layer_name)
            if match:
                return int(match.group(1))
        
        # Fallback: extract any number from the string
        numbers = re.findall(r'\d+', layer_name)
        if numbers:
            return int(numbers[0])
        
        return 0
    
    # B6: Circuit Verification with Causal Tools
    def verify_circuit(
        self,
        circuit: Circuit,
        test_input: torch.Tensor,
        verification_method: str = "ablation"
    ) -> Dict[str, Any]:
        """
        Verify circuit function using causal intervention tools.
        
        Implements B6: Circuit-Verifizierung mit Kausal-Tools
        """
        logger.info(f"Verifying circuit {circuit.circuit_id}")
        
        from .interventions import run_with_ablation, InterventionSpec, InterventionType
        
        verification_results = {
            'circuit_id': circuit.circuit_id,
            'verification_method': verification_method,
            'baseline_performance': None,
            'intervened_performance': None,
            'effect_size': None,
            'verification_score': None
        }
        
        try:
            # Get baseline performance
            with torch.no_grad():
                baseline_output = self.model(test_input)
            
            verification_results['baseline_performance'] = self._compute_output_metrics(baseline_output)
            
            # Perform interventions on circuit components
            intervention_effects = []
            
            for node in circuit.nodes:
                if verification_method == "ablation":
                    # Ablate this component
                    intervention_spec = InterventionSpec(
                        layer_name=node.layer_name,
                        intervention_type=InterventionType.ABLATION,
                        target_indices=[node.position] if node.node_type == NodeType.ATTENTION_HEAD else None
                    )
                    
                    ablation_result = run_with_ablation(
                        model=self.model,
                        input_tensor=test_input,
                        layer_name=node.layer_name,
                        neuron_indices=[node.position] if node.node_type == NodeType.MLP_NEURON else None
                    )
                    
                    intervened_output = ablation_result['output']
                    intervened_performance = self._compute_output_metrics(intervened_output)
                    
                    # Compute effect size
                    effect_size = self._compute_effect_size(
                        verification_results['baseline_performance'],
                        intervened_performance
                    )
                    
                    intervention_effects.append({
                        'node_id': node.node_id,
                        'effect_size': effect_size,
                        'performance_change': intervened_performance
                    })
            
            # Aggregate results
            avg_effect_size = np.mean([effect['effect_size'] for effect in intervention_effects])
            verification_results['effect_size'] = avg_effect_size
            verification_results['intervention_effects'] = intervention_effects
            
            # Compute verification score
            verification_score = self._compute_verification_score(circuit, intervention_effects)
            verification_results['verification_score'] = verification_score
            
            # Update circuit with verification results
            circuit.verification_results = verification_results
            
        except Exception as e:
            logger.error(f"Circuit verification failed: {e}")
            verification_results['error'] = str(e)
            verification_results['verification_score'] = 0.0
        
        return verification_results
    
    def _compute_output_metrics(self, output) -> Dict[str, float]:
        """Compute metrics from model output."""
        if hasattr(output, 'logits'):
            logits = output.logits
        else:
            logits = output
        
        # Compute basic metrics
        max_prob = torch.softmax(logits, dim=-1).max().item()
        entropy = -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean().item()
        
        return {
            'max_probability': max_prob,
            'entropy': entropy,
            'logit_magnitude': logits.abs().mean().item()
        }
    
    def _compute_effect_size(self, baseline: Dict[str, float], intervened: Dict[str, float]) -> float:
        """Compute effect size between baseline and intervened performance."""
        if 'max_probability' in baseline and 'max_probability' in intervened:
            return abs(baseline['max_probability'] - intervened['max_probability'])
        elif 'entropy' in baseline and 'entropy' in intervened:
            return abs(baseline['entropy'] - intervened['entropy'])
        else:
            return 0.0
    
    def _compute_verification_score(self, circuit: Circuit, intervention_effects: List[Dict]) -> float:
        """Compute overall verification score for the circuit."""
        if not intervention_effects:
            return 0.0
        
        # For specialized circuits, expect larger effects when ablated
        effect_sizes = [effect['effect_size'] for effect in intervention_effects]
        avg_effect = np.mean(effect_sizes)
        
        # Score based on circuit type expectations
        if circuit.circuit_type in [CircuitType.INDUCTION, CircuitType.COPYING]:
            # These should show moderate to strong effects when ablated
            if avg_effect > 0.1:
                return min(1.0, avg_effect * 2.0)  # Scale effect to [0, 1]
            else:
                return avg_effect * 5.0  # Smaller effects get lower scores
        else:
            # Generic circuits
            return min(1.0, avg_effect * 3.0)
    
    # B3: Graph/Circuit Data Structure Support Methods
    def save_circuit(self, circuit: Circuit, filepath: Union[str, Path]) -> None:
        """Save circuit to file."""
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(circuit.to_dict(), f, indent=2)
        elif filepath.suffix in ['.pkl', '.pickle']:
            with open(filepath, 'wb') as f:
                pickle.dump(circuit, f)
        elif filepath.suffix == '.graphml':
            G = circuit.to_networkx()
            nx.write_graphml(G, filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Circuit saved to {filepath}")
    
    def load_circuit(self, filepath: Union[str, Path]) -> Circuit:
        """Load circuit from file."""
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            return self._circuit_from_dict(data)
        elif filepath.suffix in ['.pkl', '.pickle']:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format for loading: {filepath.suffix}")
    
    def _circuit_from_dict(self, data: Dict[str, Any]) -> Circuit:
        """Reconstruct circuit from dictionary."""
        nodes = [
            CircuitNode(
                node_id=node_data['node_id'],
                node_type=NodeType(node_data['node_type']),
                layer=node_data['layer'],
                position=node_data['position'],
                layer_name=node_data['layer_name'],
                metadata=node_data.get('metadata', {})
            )
            for node_data in data['nodes']
        ]
        
        edges = [
            CircuitEdge(
                source=edge_data['source'],
                target=edge_data['target'],
                weight=edge_data['weight'],
                connection_type=edge_data['connection_type'],
                evidence=edge_data.get('evidence', {}),
                metadata=edge_data.get('metadata', {})
            )
            for edge_data in data['edges']
        ]
        
        return Circuit(
            circuit_id=data['circuit_id'],
            circuit_type=CircuitType(data['circuit_type']),
            nodes=nodes,
            edges=edges,
            function=data['function'],
            confidence=data['confidence'],
            verification_results=data.get('verification_results'),
            metadata=data.get('metadata', {})
        )
    
    def discover_all_circuits(
        self,
        input_ids: torch.Tensor,
        circuit_types: Optional[List[CircuitType]] = None,
        min_confidence: float = 0.6
    ) -> List[Circuit]:
        """Discover all types of circuits in the model."""
        if circuit_types is None:
            circuit_types = [CircuitType.INDUCTION, CircuitType.COPYING]
        
        all_circuits = []
        
        if CircuitType.INDUCTION in circuit_types:
            induction_circuits = self.scan_induction_heads(input_ids, min_confidence)
            all_circuits.extend(induction_circuits)
        
        if CircuitType.COPYING in circuit_types:
            copying_circuits = self.scan_copying_heads(input_ids, min_confidence)
            all_circuits.extend(copying_circuits)
        
        logger.info(f"Discovered {len(all_circuits)} total circuits")
        return all_circuits
    
    def _get_module_by_name(self, name: str) -> nn.Module:
        """Get module by name."""
        module = self.model
        for attr in name.split('.'):
            module = getattr(module, attr)
        return module
