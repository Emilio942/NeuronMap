"""Conceptual Analysis Module for Neural Network Interpretability

This module implements cutting-edge techniques for understanding neural networks
at a conceptual and mechanistic level, including cross-model analysis, causal
tracing, circuit analysis, and world model understanding.

Author: NeuronMap Team
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
import warnings
from scipy import stats
from scipy.spatial.distance import pdist, squareform

# Optional dependencies with fallbacks
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")

try:
    from sklearn.decomposition import PCA, NMF
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available for some analyses")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX not available for circuit analysis")

import json
from datetime import datetime

from ..utils.error_handling import log_error

logger = logging.getLogger(__name__)


@dataclass
class ConceptVector:
    """Represents a concept as a vector in activation space."""
    name: str
    vector: np.ndarray
    layer: str
    confidence: float
    metadata: Dict[str, Any] = None
    model_name: str = None  # For test compatibility

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Circuit:
    """Represents a computational circuit in a neural network."""
    name: str
    nodes: List[Tuple[str, int]] = None  # (layer, neuron_idx)
    edges: List[Tuple[Tuple[str, int], Tuple[str, int], float]] = None  # (from, to, weight)
    function: str = None
    confidence: float = None
    components: List[str] = None  # For test compatibility
    connections: List[Tuple[str, str, float]] = None  # Alternative to edges for test compatibility
    evidence_strength: float = None  # Alternative to confidence for test compatibility
    metadata: Dict[str, Any] = None  # For test compatibility

    def __post_init__(self):
        if self.components is None:
            self.components = [self.name] if self.name else []
        if self.connections is None and self.edges:
            # Convert edges to connections format
            self.connections = [(f"{edge[0][0]}_{edge[0][1]}", f"{edge[1][0]}_{edge[1][1]}", edge[2]) for edge in self.edges]
        if self.evidence_strength is None and self.confidence is not None:
            self.evidence_strength = self.confidence
        if self.confidence is None and self.evidence_strength is not None:
            self.confidence = self.evidence_strength
        if self.metadata is None:
            self.metadata = {}
        if self.nodes is None:
            self.nodes = []
        if self.edges is None:
            self.edges = []


@dataclass
class KnowledgeTransferResult:
    """Results from knowledge transfer analysis."""
    source_model: str
    target_model: str
    transfer_score: float
    transferred_concepts: List[str] = None
    layer_mapping: Dict[str, str] = None
    transfer_map: Dict[str, float] = None  # For test compatibility
    preserved_concepts: List[str] = None
    lost_concepts: List[str] = None
    emergent_concepts: List[str] = None

    def __post_init__(self):
        if self.transferred_concepts is None:
            self.transferred_concepts = []
        if self.layer_mapping is None:
            self.layer_mapping = {}
        if self.transfer_map is None:
            self.transfer_map = {concept: self.transfer_score for concept in self.transferred_concepts}
        if self.preserved_concepts is None:
            self.preserved_concepts = []
        if self.lost_concepts is None:
            self.lost_concepts = []
        if self.emergent_concepts is None:
            self.emergent_concepts = []


class ConceptualAnalyzer:
    """Advanced conceptual analysis for neural network interpretability."""

    def __init__(self, device: str = "auto", config: dict = None):
        """Initialize the conceptual analyzer.

        Args:
            device: Device to use for computation ('auto', 'cpu', 'cuda') or config dict
            config: Configuration dict (for test compatibility)
        """
        # Handle test compatibility - config dict passed as device
        if isinstance(device, dict):
            config = device
            device = config.get('device', 'auto')
        elif config is None:
            config = {}

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Store config for test compatibility
        self.config = config

        # Extract config values with defaults for test compatibility
        self.concept_threshold = config.get('concept_threshold', 0.5)
        self.circuit_threshold = config.get('circuit_threshold', 0.5)
        self.causal_threshold = config.get('causal_threshold', 0.5)

        # Initialize storage containers for downstream analyses
        self.concepts: Dict[str, ConceptVector] = {}
        self.circuits: Dict[str, Circuit] = {}
        self.world_models: Dict[str, Dict[str, Any]] = {}

        logger.info(f"ConceptualAnalyzer initialized with device: {self.device}")

    @log_error("concept_extraction")
    def extract_concepts(
        self,
        activations: Dict[str, np.ndarray],
        labels: List[str] = None,
        method: str = "pca",
        n_concepts: int = 10,
        **kwargs
    ) -> Dict[str, Union[List[ConceptVector], ConceptVector]]:
        """Extract concepts from neural network activations.

        Args:
            activations: Dictionary mapping layer names to activation arrays
            labels: Optional labels for supervised concept extraction
            method: Extraction method ('pca', 'nmf', 'ica')
            n_concepts: Number of concepts to extract per layer
            **kwargs: Additional arguments for the extraction method

        Returns:
            Dictionary mapping layer names to lists of ConceptVector objects,
            or concept names to ConceptVector objects if labels provided
        """
        # Handle test compatibility - if labels is a string, it's actually the method
        if isinstance(labels, str):
            method = labels
            labels = None

        logger.info(f"Extracting concepts using {method} method")

        # Reset stored concepts for this extraction run
        self.concepts = {}

        concepts_by_layer: Dict[str, List[ConceptVector]] = {}
        flattened_concepts: Dict[str, ConceptVector] = {}

        for layer_name, layer_activations in activations.items():
            if layer_activations.size == 0:
                logger.warning(f"Empty activations for layer {layer_name}")
                continue

            layer_concepts = []

            if method == "pca":
                layer_concepts = self._extract_pca_concepts(
                    layer_activations, layer_name, n_concepts, **kwargs
                )
            elif method == "nmf":
                layer_concepts = self._extract_nmf_concepts(
                    layer_activations, layer_name, n_concepts, **kwargs
                )
            elif method == "ica":
                layer_concepts = self._extract_ica_concepts(
                    layer_activations, layer_name, n_concepts, **kwargs
                )
            else:
                raise ValueError(f"Unknown concept extraction method: {method}")

            renamed_layer_concepts: List[ConceptVector] = []
            for idx, concept in enumerate(layer_concepts):
                original_name = concept.name
                if method == "nmf":
                    concept_name = f"{layer_name}_component_{idx}"
                elif method == "ica":
                    concept_name = f"{layer_name}_ica_{idx}"
                else:
                    concept_name = f"{layer_name}_concept_{idx}"

                concept.metadata.setdefault("original_name", original_name)
                concept.name = concept_name
                self.concepts[concept_name] = concept
                flattened_concepts[concept_name] = concept
                renamed_layer_concepts.append(concept)

            concepts_by_layer[layer_name] = renamed_layer_concepts
            logger.debug(f"Extracted {len(renamed_layer_concepts)} concepts from {layer_name}")

        # If labels are provided, return flattened dict for test compatibility
        if labels is not None:
            return flattened_concepts

        return concepts_by_layer

    def _extract_pca_concepts(
        self,
        activations: np.ndarray,
        layer_name: str,
        n_concepts: int,
        **kwargs
    ) -> List[ConceptVector]:
        """Extract concepts using Principal Component Analysis."""
        pca = PCA(n_components=min(n_concepts, activations.shape[1]))
        pca.fit(activations)

        concepts = []
        for i, (component, variance_ratio) in enumerate(
            zip(pca.components_, pca.explained_variance_ratio_)
        ):
            concept = ConceptVector(
                name=f"PCA_concept_{i}",
                vector=component,
                layer=layer_name,
                confidence=float(variance_ratio),
                metadata={
                    "method": "pca",
                    "explained_variance_ratio": float(variance_ratio),
                    "component_index": i
                }
            )
            concepts.append(concept)

        return concepts

    def _extract_nmf_concepts(
        self,
        activations: np.ndarray,
        layer_name: str,
        n_concepts: int,
        **kwargs
    ) -> List[ConceptVector]:
        """Extract concepts using Non-negative Matrix Factorization."""
        # Ensure non-negative activations for NMF
        activations_nn = np.maximum(activations, 0)

        nmf = NMF(n_components=min(n_concepts, activations.shape[1]), random_state=42)
        nmf.fit(activations_nn)

        concepts = []
        for i, component in enumerate(nmf.components_):
            # Calculate confidence based on reconstruction error
            reconstruction = nmf.transform(activations_nn) @ nmf.components_
            error = np.mean((activations_nn - reconstruction) ** 2)
            confidence = 1.0 / (1.0 + error)

            concept = ConceptVector(
                name=f"NMF_concept_{i}",
                vector=component,
                layer=layer_name,
                confidence=float(confidence),
                metadata={
                    "method": "nmf",
                    "reconstruction_error": float(error),
                    "component_index": i
                }
            )
            concepts.append(concept)

        return concepts

    def _extract_ica_concepts(
        self,
        activations: np.ndarray,
        layer_name: str,
        n_concepts: int,
        **kwargs
    ) -> List[ConceptVector]:
        """Extract concepts using Independent Component Analysis."""
        try:
            from sklearn.decomposition import FastICA
        except ImportError:
            logger.warning("FastICA not available, falling back to PCA")
            return self._extract_pca_concepts(activations, layer_name, n_concepts, **kwargs)

        ica = FastICA(n_components=min(n_concepts, activations.shape[1]), random_state=42)
        ica.fit(activations)

        concepts = []
        for i, component in enumerate(ica.components_):
            # Calculate confidence based on independence (kurtosis)
            projected = activations @ component
            kurtosis = stats.kurtosis(projected)
            confidence = min(abs(kurtosis) / 10.0, 1.0)  # Normalize to [0, 1]

            concept = ConceptVector(
                name=f"ICA_concept_{i}",
                vector=component,
                layer=layer_name,
                confidence=float(confidence),
                metadata={
                    "method": "ica",
                    "kurtosis": float(kurtosis),
                    "component_index": i
                }
            )
            concepts.append(concept)

        return concepts

    @log_error("concept_algebra")
    def concept_algebra(
        self,
        concept1_or_concepts: Union[str, Dict[str, List[ConceptVector]]],
        concept2_or_operations: Union[str, List[Tuple[str, str, str, str]]] = None,
        operation: str = None
    ) -> Union[ConceptVector, Dict[str, ConceptVector]]:
        """Perform algebraic operations on concepts.

        Args:
            concept1_or_concepts: Concept name or dictionary of concepts by layer
            concept2_or_operations: Concept name or list of operations
            operation: Operation type ('add', 'subtract', 'multiply', 'normalize')

        Returns:
            Single ConceptVector or dictionary of resulting concepts
        """
        # Handle test signature: concept_algebra(name1, name2, operation)
        if isinstance(concept1_or_concepts, str) and isinstance(concept2_or_operations, str) and operation:
            return self._concept_algebra_simple(concept1_or_concepts, concept2_or_operations, operation)

        # Handle original signature: concept_algebra(concepts, operations)
        return self._concept_algebra_batch(concept1_or_concepts, concept2_or_operations)

    def _concept_algebra_simple(
        self,
        concept1_name: str,
        concept2_name: str,
        operation: str
    ) -> ConceptVector:
        """Simple concept algebra for test compatibility."""
        logger.info(f"Performing {operation} operation on {concept1_name} and {concept2_name}")

        if concept1_name not in self.concepts:
            raise ValueError(f"Concept {concept1_name} not found")
        if concept2_name not in self.concepts:
            raise ValueError(f"Concept {concept2_name} not found")

        concept1 = self.concepts[concept1_name]
        concept2 = self.concepts[concept2_name]

        # Perform operation
        if operation == "add":
            result_vector = concept1.vector + concept2.vector
        elif operation == "subtract":
            result_vector = concept1.vector - concept2.vector
        elif operation == "multiply":
            result_vector = concept1.vector * concept2.vector
        elif operation == "average":
            result_vector = (concept1.vector + concept2.vector) / 2.0
        elif operation == "normalize":
            result_vector = concept1.vector / (np.linalg.norm(concept1.vector) + 1e-8)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Create result concept
        result = ConceptVector(
            name=f"{concept1_name}_{operation}_{concept2_name}",
            vector=result_vector,
            layer=concept1.layer,
            confidence=min(concept1.confidence, concept2.confidence),
            metadata={
                'operation': operation,
                'operands': [concept1_name, concept2_name]
            }
        )

        return result

    def _concept_algebra_batch(
        self,
        concepts: Dict[str, List[ConceptVector]],
        operations: List[Tuple[str, str, str, str]]
    ) -> Dict[str, ConceptVector]:
        """Perform algebraic operations on concepts.

        Args:
            concepts: Dictionary of concepts by layer
            operations: List of operations as (operation, concept1, concept2, result_name)
                       Operations: 'add', 'subtract', 'multiply', 'normalize'

        Returns:
            Dictionary of resulting concepts
        """
        logger.info(f"Performing {len(operations)} concept algebra operations")

        # Flatten concepts for easy lookup
        concept_lookup = {}
        for layer_concepts in concepts.values():
            for concept in layer_concepts:
                concept_lookup[concept.name] = concept

        results = {}

        for operation, concept1_name, concept2_name, result_name in operations:
            if concept1_name not in concept_lookup:
                logger.warning(f"Concept {concept1_name} not found")
                continue

            concept1 = concept_lookup[concept1_name]

            if operation == "normalize":
                # Single concept operation
                normalized_vector = concept1.vector / np.linalg.norm(concept1.vector)
                result = ConceptVector(
                    name=result_name,
                    vector=normalized_vector,
                    layer=concept1.layer,
                    confidence=concept1.confidence,
                    metadata={
                        "operation": "normalize",
                        "source": concept1_name,
                        **concept1.metadata
                    }
                )
                results[result_name] = result
                continue

            if concept2_name not in concept_lookup:
                logger.warning(f"Concept {concept2_name} not found")
                continue

            concept2 = concept_lookup[concept2_name]

            # Ensure concepts are from the same layer and have same dimensionality
            if concept1.layer != concept2.layer:
                logger.warning(f"Cannot perform {operation} on concepts from different layers")
                continue

            if concept1.vector.shape != concept2.vector.shape:
                logger.warning(f"Cannot perform {operation} on concepts with different shapes")
                continue

            if operation == "add":
                result_vector = concept1.vector + concept2.vector
            elif operation == "subtract":
                result_vector = concept1.vector - concept2.vector
            elif operation == "multiply":
                result_vector = concept1.vector * concept2.vector
            else:
                logger.warning(f"Unknown operation: {operation}")
                continue

            # Calculate confidence as average
            result_confidence = (concept1.confidence + concept2.confidence) / 2

            result = ConceptVector(
                name=result_name,
                vector=result_vector,
                layer=concept1.layer,
                confidence=result_confidence,
                metadata={
                    "operation": operation,
                    "source_concepts": [concept1_name, concept2_name],
                    "source_metadata": [concept1.metadata, concept2.metadata]
                }
            )

            results[result_name] = result

        logger.info(f"Generated {len(results)} concept algebra results")
        return results

    @log_error("analysis_operation")
    def discover_circuits(
        self,
        activations: Union[nn.Module, Dict[str, np.ndarray]],
        labels: Union[Dict[str, np.ndarray], List[str]] = None,
        task_name: Union[Dict[str, List[ConceptVector]], str] = None,
        threshold: float = 0.1
    ) -> Union[List[Circuit], Dict[str, Any]]:
        """Discover computational circuits in the neural network.

        Args:
            activations: The neural network model or layer activations dict
            labels: Layer activations dict or labels list
            task_name: Extracted concepts dict or task name string
            threshold: Minimum connection strength to include in circuit

        Returns:
            List of discovered Circuit objects or dict for test compatibility
        """
        # Handle both old signature (model, activations, concepts, threshold)
        # and new test signature (activations, labels, task_name)
        if isinstance(activations, nn.Module):
            # Old signature: (model, activations, concepts, threshold)
            model = activations
            activations = labels
            concepts = task_name
            return self._discover_circuits_old(model, activations, concepts, threshold)
        else:
            # New test signature: (activations, labels, task_name)
            return self._discover_circuits_new(activations, labels, task_name, threshold)

    def _add_circuit_nodes(self, G: nx.DiGraph, concepts: Dict[str, Any]):
        """Helper to add nodes to the circuit graph."""
        for layer_name, layer_concepts in concepts.items():
            if isinstance(layer_concepts, list):
                for i, concept in enumerate(layer_concepts):
                    node_id = f"{layer_name}_{i}"
                    G.add_node(node_id,
                              layer=layer_name,
                              concept=concept.name,
                              confidence=concept.confidence)
            else:
                concept = layer_concepts
                node_id = layer_name
                actual_layer = layer_name.split('_concept_')[0] if '_concept_' in layer_name else layer_name
                G.add_node(node_id,
                          layer=actual_layer,
                          concept=concept.name,
                          confidence=concept.confidence)

    def _add_circuit_edges(self, G: nx.DiGraph, activations: Dict[str, np.ndarray], concepts: Dict[str, Any], threshold: float):
        """Helper to add edges to the circuit graph based on correlations."""
        layer_groups = {}
        for concept_key, concept_value in concepts.items():
            if isinstance(concept_value, list):
                layer_groups[concept_key] = concept_value
            else:
                actual_layer = concept_key.split('_concept_')[0] if '_concept_' in concept_key else concept_key
                if actual_layer not in layer_groups:
                    layer_groups[actual_layer] = []
                layer_groups[actual_layer].append(concept_value)

        layer_names = list(layer_groups.keys())
        for i in range(len(layer_names) - 1):
            current_layer = layer_names[i]
            next_layer = layer_names[i + 1]

            if (current_layer in activations and
                next_layer in activations):

                current_acts = activations[current_layer]
                next_acts = activations[next_layer]

                for j, concept1 in enumerate(layer_groups[current_layer]):
                    for k, concept2 in enumerate(layer_groups[next_layer]):
                        if isinstance(concepts.get(f"{current_layer}_concept_{j}"), ConceptVector):
                            node1 = f"{current_layer}_concept_{j}"
                            node2 = f"{next_layer}_concept_{k}"
                        else:
                            node1 = f"{current_layer}_{j}"
                            node2 = f"{next_layer}_{k}"

                        if (j < current_acts.shape[1] and
                            k < next_acts.shape[1]):
                            correlation = np.corrcoef(
                                current_acts[:, j],
                                next_acts[:, k]
                            )[0, 1]

                            if (not np.isnan(correlation) and
                                abs(correlation) > threshold):
                                G.add_edge(node1, node2, weight=correlation)

    def _extract_circuits_from_graph(self, G: nx.DiGraph) -> List[Circuit]:
        """Helper to extract Circuit objects from a NetworkX graph."""
        circuits = []
        for component in nx.weakly_connected_components(G):
            if len(component) > 1:
                nodes = []
                edges = []

                for node in component:
                    layer = G.nodes[node]['layer']
                    nodes.append((layer, int(node.split('_')[-1])))

                for edge in G.edges(component):
                    if edge[0] in component and edge[1] in component:
                        weight = G.edges[edge]['weight']
                        from_layer = G.nodes[edge[0]]['layer']
                        to_layer = G.nodes[edge[1]]['layer']
                        from_idx = int(edge[0].split('_')[-1])
                        to_idx = int(edge[1].split('_')[-1])

                        edges.append(
                            ((from_layer, from_idx), (to_layer, to_idx), weight)
                        )

                circuit = Circuit(
                    name=f"circuit_{len(circuits)}",
                    nodes=nodes,
                    edges=edges,
                    function="unknown",
                    confidence=0.8,
                    components=[f"component_{len(circuits)}"]
                )
                circuits.append(circuit)
        return circuits

    def _discover_circuits_old(
        self,
        model: nn.Module,
        activations: Dict[str, np.ndarray],
        concepts: Dict[str, List[ConceptVector]],
        threshold: float = 0.1
    ) -> List[Circuit]:
        """Original circuit discovery implementation."""
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available, cannot perform circuit discovery")
            return []

        logger.info("Discovering computational circuits")

        G = nx.DiGraph()
        self._add_circuit_nodes(G, concepts)
        self._add_circuit_edges(G, activations, concepts, threshold)
        circuits = self._extract_circuits_from_graph(G)

        logger.info(f"Discovered {len(circuits)} circuits")
        return circuits

    def _discover_circuits_new(
        self,
        activations: Dict[str, np.ndarray],
        labels: List[str],
        task_name: str,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Test-compatible circuit discovery implementation."""
        logger.info(f"Discovering circuits for task: {task_name}")

        # Extract concepts first
        concepts = self.extract_concepts(activations, labels, method='pca')

        # Use old implementation with extracted concepts to get circuits
        if NETWORKX_AVAILABLE:
            circuits_list = self._discover_circuits_old(None, activations, concepts, threshold)

            # Convert to test-expected format: dict with circuit objects as values
            circuits_dict = {}
            for i, circuit in enumerate(circuits_list):
                # Add test-expected attributes if missing
                if not hasattr(circuit, 'connections'):
                    circuit.connections = circuit.edges  # Use edges as connections
                if not hasattr(circuit, 'evidence_strength'):
                    circuit.evidence_strength = circuit.confidence

                circuits_dict[f"circuit_{i}"] = circuit

            return circuits_dict
        else:
            # Return empty dict if NetworkX not available
            return {}

    @log_error("analysis_operation")
    def causal_tracing(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        target_layer: str,
        target_neurons: Optional[List[int]] = None,
        intervention_type: str = "noise"
    ) -> Dict[str, float]:
        """Perform causal tracing to understand information flow.

        Args:
            model: The neural network model
            input_data: Input data tensor
            target_layer: Layer to trace causality to
            target_neurons: Specific neurons to trace (None for all)
            intervention_type: Type of intervention ('noise', 'zero', 'mean')

        Returns:
            Dictionary mapping source layers to causal influence scores
        """
        logger.info(f"Performing causal tracing to {target_layer}")

        model.eval()
        causal_scores = {}

        # Get baseline activations
        with torch.no_grad():
            baseline_activations = self._get_activations(model, input_data, target_layer)

        if target_neurons is None:
            target_neurons = list(range(baseline_activations.shape[-1]))

        # Test intervention at each layer
        layer_names = [name for name, _ in model.named_modules()
                      if isinstance(_, (nn.Linear, nn.Conv2d, nn.MultiheadAttention))]

        for intervention_layer in layer_names:
            if intervention_layer == target_layer:
                continue

            # Apply intervention
            intervened_activations = self._apply_intervention(
                model, input_data, intervention_layer, target_layer, intervention_type
            )

            # Calculate causal effect
            effect = self._calculate_causal_effect(
                baseline_activations, intervened_activations, target_neurons
            )

            causal_scores[intervention_layer] = float(effect)

        # Normalize scores
        max_score = max(causal_scores.values()) if causal_scores else 1.0
        causal_scores = {k: v / max_score for k, v in causal_scores.items()}

        logger.info(f"Computed causal scores for {len(causal_scores)} layers")
        return causal_scores

    def trace_causal_effects(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        target_layer: str,
        target_neurons: Optional[List[int]] = None,
        min_effect: Optional[float] = None,
        intervention_type: str = "noise"
    ) -> Dict[str, Any]:
        """Convenience wrapper that returns structured causal tracing results.

        Args:
            model: Neural network model under analysis.
            input_data: Input tensor used for probing.
            target_layer: Target layer whose activations we monitor.
            target_neurons: Optional list of neuron indices to focus on.
            min_effect: Minimum normalized effect size to retain in the
                ``significant_effects`` mapping. Defaults to the configured
                ``causal_threshold`` when not specified.
            intervention_type: Intervention strategy passed through to
                :meth:`causal_tracing`.

        Returns:
            Dictionary containing full effect scores, a filtered subset that
            meets the minimum effect requirement, and convenience metadata.
        """

        scores = self.causal_tracing(
            model,
            input_data,
            target_layer,
            target_neurons,
            intervention_type,
        )

        threshold = self.causal_threshold if min_effect is None else min_effect
        significant = {
            layer: value
            for layer, value in scores.items()
            if value >= threshold
        }

        summary = {
            "effects": scores,
            "significant_effects": significant,
            "max_effect_layer": max(scores, key=scores.get) if scores else None,
            "threshold": threshold,
        }

        return summary

    def _get_activations(self, model: nn.Module, input_data: torch.Tensor, target_layer: str) -> torch.Tensor:
        """Get activations from a specific layer."""
        activations = {}

        def hook_fn(module, input, output):
            activations[target_layer] = output.detach()

        # Register hook
        for name, module in model.named_modules():
            if name == target_layer:
                handle = module.register_forward_hook(hook_fn)
                break
        else:
            raise ValueError(f"Layer {target_layer} not found in model")

        # Forward pass
        with torch.no_grad():
            _ = model(input_data)

        # Remove hook
        handle.remove()

        return activations[target_layer]

    def _apply_intervention(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        intervention_layer: str,
        target_layer: str,
        intervention_type: str
    ) -> torch.Tensor:
        """Apply intervention to a specific layer and get target activations."""
        activations = {}

        def intervention_hook(module, input, output):
            if intervention_type == "noise":
                noise = torch.randn_like(output) * 0.1
                return output + noise
            elif intervention_type == "zero":
                return torch.zeros_like(output)
            elif intervention_type == "mean":
                return torch.full_like(output, output.mean())
            else:
                return output

        def target_hook(module, input, output):
            activations[target_layer] = output.detach()

        # Register hooks
        intervention_handle = None
        target_handle = None

        for name, module in model.named_modules():
            if name == intervention_layer:
                intervention_handle = module.register_forward_hook(intervention_hook)
            elif name == target_layer:
                target_handle = module.register_forward_hook(target_hook)

        if intervention_handle is None:
            raise ValueError(f"Intervention layer {intervention_layer} not found")
        if target_handle is None:
            raise ValueError(f"Target layer {target_layer} not found")

        # Forward pass with intervention
        with torch.no_grad():
            _ = model(input_data)

        # Remove hooks
        intervention_handle.remove()
        target_handle.remove()

        return activations[target_layer]

    def _calculate_causal_effect(
        self,
        baseline: torch.Tensor,
        intervened: torch.Tensor,
        target_neurons: List[int]
    ) -> float:
        """Calculate the causal effect of intervention."""
        baseline_mean = baseline[:, target_neurons].mean()
        intervened_mean = intervened[:, target_neurons].mean()

        effect = abs(baseline_mean - intervened_mean)
        return effect.item()

    @log_error("analysis_operation")
    def world_model_analysis(
        self,
        model: nn.Module,
        input_sequences: torch.Tensor,
        prediction_tasks: List[str] = None
    ) -> Dict[str, Any]:
        """Analyze the world model learned by the neural network.

        Args:
            model: The neural network model
            input_sequences: Sequential input data
            prediction_tasks: List of prediction tasks to test

        Returns:
            Dictionary containing world model analysis results
        """
        logger.info("Analyzing learned world model")

        if prediction_tasks is None:
            prediction_tasks = ["next_token", "object_permanence", "causality"]

        results = {
            "prediction_accuracy": {},
            "temporal_coherence": 0.0,
            "causal_understanding": 0.0,
            "object_tracking": 0.0
        }

        model.eval()

        # Test next token prediction
        if "next_token" in prediction_tasks:
            results["prediction_accuracy"]["next_token"] = self._test_next_token_prediction(
                model, input_sequences
            )

        # Test object permanence
        if "object_permanence" in prediction_tasks:
            results["object_tracking"] = self._test_object_permanence(model, input_sequences)

        # Test causal understanding
        if "causality" in prediction_tasks:
            results["causal_understanding"] = self._test_causal_understanding(
                model, input_sequences
            )

        # Test temporal coherence
        results["temporal_coherence"] = self._test_temporal_coherence(model, input_sequences)

        logger.info("World model analysis completed")
        return results

    def _test_next_token_prediction(self, model: nn.Module, sequences: torch.Tensor) -> float:
        """Test next token prediction accuracy."""
        correct = 0
        total = 0

        with torch.no_grad():
            for seq in sequences:
                if len(seq) < 2:
                    continue

                # Use all but last token as input
                input_seq = seq[:-1].unsqueeze(0)
                target = seq[-1]

                try:
                    output = model(input_seq)
                    if hasattr(output, 'logits'):
                        logits = output.logits
                    else:
                        logits = output

                    predicted = torch.argmax(logits[0, -1])
                    if predicted == target:
                        correct += 1
                    total += 1
                except Exception as e:
                    logger.debug(f"Error in next token prediction: {e}")
                    continue

        return correct / total if total > 0 else 0.0

    def _test_object_permanence(self, model: nn.Module, sequences: torch.Tensor) -> float:
        """Test object permanence understanding."""
        # Simplified test - check if model maintains object representations
        # when objects are temporarily occluded

        coherence_scores = []

        with torch.no_grad():
            for seq in sequences:
                if len(seq) < 10:  # Need sufficient sequence length
                    continue

                try:
                    # Get activations for full sequence
                    full_activations = self._get_sequence_activations(model, seq)

                    # Get activations for sequence with middle portion removed (occlusion)
                    mid_start = len(seq) // 3
                    mid_end = 2 * len(seq) // 3
                    occluded_seq = torch.cat([seq[:mid_start], seq[mid_end:]])
                    occluded_activations = self._get_sequence_activations(model, occluded_seq)

                    # Compare representations before and after occlusion
                    before_repr = full_activations[mid_start - 1]
                    after_repr = occluded_activations[mid_start - 1] if mid_start - 1 < len(occluded_activations) else occluded_activations[-1]

                    # Calculate similarity
                    similarity = torch.cosine_similarity(before_repr, after_repr, dim=0)
                    coherence_scores.append(similarity.item())

                except Exception as e:
                    logger.debug(f"Error in object permanence test: {e}")
                    continue

        return np.mean(coherence_scores) if coherence_scores else 0.0

    def _test_causal_understanding(self, model: nn.Module, sequences: torch.Tensor) -> float:
        """Test causal understanding by measuring intervention effects."""
        causal_scores = []

        with torch.no_grad():
            for seq in sequences:
                if len(seq) < 5:
                    continue

                try:
                    # Original prediction
                    original_output = model(seq.unsqueeze(0))
                    if hasattr(original_output, 'logits'):
                        original_logits = original_output.logits
                    else:
                        original_logits = original_output

                    # Intervene at random position
                    intervention_pos = np.random.randint(1, len(seq) - 1)
                    intervened_seq = seq.clone()
                    intervened_seq[intervention_pos] = torch.randint_like(
                        intervened_seq[intervention_pos], 0, 1000
                    )

                    # Prediction after intervention
                    intervened_output = model(intervened_seq.unsqueeze(0))
                    if hasattr(intervened_output, 'logits'):
                        intervened_logits = intervened_output.logits
                    else:
                        intervened_logits = intervened_output

                    # Measure prediction change
                    logit_diff = torch.abs(original_logits - intervened_logits).mean()
                    causal_scores.append(logit_diff.item())

                except Exception as e:
                    logger.debug(f"Error in causal understanding test: {e}")
                    continue

        # Higher difference indicates better causal understanding
        return np.mean(causal_scores) if causal_scores else 0.0

    def _test_temporal_coherence(self, model: nn.Module, sequences: torch.Tensor) -> float:
        """Test temporal coherence of representations."""
        coherence_scores = []

        with torch.no_grad():
            for seq in sequences:
                if len(seq) < 3:
                    continue

                try:
                    activations = self._get_sequence_activations(model, seq)

                    # Calculate coherence between adjacent time steps
                    for i in range(len(activations) - 1):
                        curr_repr = activations[i]
                        next_repr = activations[i + 1]

                        # Measure similarity
                        similarity = torch.cosine_similarity(curr_repr, next_repr, dim=0)
                        coherence_scores.append(similarity.item())

                except Exception as e:
                    logger.debug(f"Error in temporal coherence test: {e}")
                    continue

        return np.mean(coherence_scores) if coherence_scores else 0.0

    def _get_sequence_activations(self, model: nn.Module, sequence: torch.Tensor) -> List[torch.Tensor]:
        """Get activations for each position in a sequence."""


        # Get activations from the last layer before output
        target_layer = None
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                target_layer = name
                break

        if target_layer is None:
            # Fallback: use model output
            output = model(sequence.unsqueeze(0))
            if hasattr(output, 'last_hidden_state'):
                return [output.last_hidden_state[0, i] for i in range(output.last_hidden_state.shape[1])]
            else:
                return [output[0, i] if output.dim() > 1 else output for i in range(output.shape[0] if output.dim() > 0 else 1)]

        # Get activations from target layer
        layer_activations = []

        def hook_fn(module, input, output):  # pragma: no cover - simple hook
            layer_activations.append(output.detach())

        # Register hook
        handle = None
        for name, module in model.named_modules():
            if name == target_layer:
                handle = module.register_forward_hook(hook_fn)
                break

        # Forward pass
        _ = model(sequence.unsqueeze(0))

        # Remove hook
        if handle is not None:
            handle.remove()

        if layer_activations:
            # Extract activations for each sequence position
            batch_activations = layer_activations[0]
            if batch_activations.dim() >= 2:
                return [batch_activations[0, i] for i in range(batch_activations.shape[1])]
            else:
                return [batch_activations[0]]

        return []

    @log_error("analysis_operation")
    def cross_model_rsa(
        self,
        model_activations_or_model1: Union[Dict[str, Dict[str, np.ndarray]], nn.Module],
        stimuli_or_model2: Union[List[str], nn.Module] = None,
        input_data: torch.Tensor = None,
        layers1: List[str] = None,
        layers2: List[str] = None
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """Perform representational similarity analysis between two models.

        Args:
            model_activations_or_model1: Either model activations dict or first model
            stimuli_or_model2: Either stimuli list or second model
            input_data: Input data for both models (when using models)
            layers1: Layer names from model1 to analyze (when using models)
            layers2: Layer names from model2 to analyze (when using models)

        Returns:
            RSA results as array or dict
        """
        # Handle test signature: cross_model_rsa(model_activations, stimuli)
        if isinstance(model_activations_or_model1, dict) and isinstance(stimuli_or_model2, list):
            return self._cross_model_rsa_test(model_activations_or_model1, stimuli_or_model2)

        # Handle original signature: cross_model_rsa(model1, model2, input_data, layers1, layers2)
        return self._cross_model_rsa_original(
            model_activations_or_model1, stimuli_or_model2, input_data, layers1, layers2
        )

    def _compute_similarity_matrices(self, model_activations: Dict[str, Dict[str, np.ndarray]], stimuli: List[str]) -> Dict[str, Any]:
        """Compute similarity matrices for each model."""
        similarity_matrices = {}
        for model_name, activations in model_activations.items():
            similarity_matrices[model_name] = {}
            for layer_name, layer_acts in activations.items():
                if layer_acts.shape[0] == len(stimuli):
                    rdm = pdist(layer_acts, metric='correlation')
                    rdm_matrix = squareform(rdm)
                    similarity_matrices[model_name][layer_name] = {
                        'rdm': rdm_matrix,
                        'stimuli': stimuli,
                        'layer': layer_name,
                        'model': model_name
                    }
        return similarity_matrices

    def _compare_rsa_models(self, model_activations: Dict[str, Dict[str, np.ndarray]], stimuli: List[str], model_names: List[str]) -> Dict[str, Any]:
        """Compare models based on their RSA matrices."""
        model_comparisons = {}
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                comparison_key = f"{model1}_vs_{model2}"
                model_comparisons[comparison_key] = {}
                for layer1, acts1 in model_activations[model1].items():
                    for layer2, acts2 in model_activations[model2].items():
                        if acts1.shape == acts2.shape:
                            rdm1 = pdist(acts1, metric='correlation')
                            rdm2 = pdist(acts2, metric='correlation')
                            if len(rdm1) > 1 and len(rdm2) > 1:
                                correlation = np.corrcoef(rdm1, rdm2)[0, 1]
                                if not np.isnan(correlation):
                                    model_comparisons[comparison_key][f"{layer1}_vs_{layer2}"] = correlation
        return model_comparisons

    def _analyze_hierarchical_alignment(self, model_activations: Dict[str, Dict[str, np.ndarray]], model_names: List[str]) -> Dict[str, Any]:
        """Analyze hierarchical alignment across models."""
        hierarchical_alignment = {}
        layer_names = set()
        for model_acts in model_activations.values():
            layer_names.update(model_acts.keys())

        for layer in layer_names:
            alignment_scores = []
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names[i+1:], i+1):
                    if (layer in model_activations[model1] and
                        layer in model_activations[model2]):
                        acts1 = model_activations[model1][layer]
                        acts2 = model_activations[model2][layer]
                        if acts1.shape == acts2.shape:
                            corr = np.corrcoef(acts1.flatten(), acts2.flatten())[0, 1]
                            if not np.isnan(corr):
                                alignment_scores.append(abs(corr))

            if alignment_scores:
                hierarchical_alignment[layer] = {
                    'mean_alignment': np.mean(alignment_scores),
                    'std_alignment': np.std(alignment_scores),
                    'scores': alignment_scores
                }
        return hierarchical_alignment

    def _cross_model_rsa_test(
        self,
        model_activations: Dict[str, Dict[str, np.ndarray]],
        stimuli: List[str]
    ) -> Dict[str, Any]:
        """Test-compatible RSA implementation."""
        logger.info("Performing cross-model RSA analysis")

        results = {
            'similarity_matrices': {},
            'model_comparisons': {},
            'hierarchical_alignment': {},
            'stimuli': stimuli
        }

        model_names = list(model_activations.keys())

        results['similarity_matrices'] = self._compute_similarity_matrices(model_activations, stimuli)

        if len(model_names) > 1:
            results['model_comparisons'] = self._compare_rsa_models(model_activations, stimuli, model_names)
            results['hierarchical_alignment'] = self._analyze_hierarchical_alignment(model_activations, model_names)

        return results

    def _cross_model_rsa_original(
        self,
        model1: nn.Module,
        model2: nn.Module,
        input_data: torch.Tensor,
        layers1: List[str],
        layers2: List[str]
    ) -> np.ndarray:
        """Perform representational similarity analysis between two models.

        Args:
            model1: First neural network model
            model2: Second neural network model
            input_data: Input data for both models
            layers1: Layer names from model1 to analyze
            layers2: Layer names from model2 to analyze

        Returns:
            RSA matrix showing similarities between model layers
        """
        logger.info("Performing cross-model representational similarity analysis")

        # Get activations from both models
        activations1 = self._get_multi_layer_activations(model1, input_data, layers1)
        activations2 = self._get_multi_layer_activations(model2, input_data, layers2)

        # Compute RSA matrix
        rsa_matrix = np.zeros((len(layers1), len(layers2)))

        for i, layer1 in enumerate(layers1):
            for j, layer2 in enumerate(layers2):
                if layer1 in activations1 and layer2 in activations2:
                    acts1 = activations1[layer1].cpu().numpy()
                    acts2 = activations2[layer2].cpu().numpy()

                    # Flatten activations for each example
                    acts1_flat = acts1.reshape(acts1.shape[0], -1)
                    acts2_flat = acts2.reshape(acts2.shape[0], -1)

                    # Compute representational dissimilarity matrices
                    rdm1 = pdist(acts1_flat, metric='correlation')
                    rdm2 = pdist(acts2_flat, metric='correlation')

                    # Compute Spearman correlation between RDMs
                    rsa_score, _ = stats.spearmanr(rdm1, rdm2)
                    rsa_matrix[i, j] = rsa_score if not np.isnan(rsa_score) else 0.0

        logger.info(f"Computed RSA matrix of shape {rsa_matrix.shape}")
        return rsa_matrix

    def _get_multi_layer_activations(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        layer_names: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Get activations from multiple layers."""
        activations = {}

        def make_hook(layer_name):
            def hook_fn(module, input, output):
                activations[layer_name] = output.detach()
            return hook_fn

        # Register hooks
        handles = []
        for name, module in model.named_modules():
            if name in layer_names:
                handle = module.register_forward_hook(make_hook(name))
                handles.append(handle)

        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(input_data)

        # Remove hooks
        for handle in handles:
            handle.remove()

        return activations

    @log_error("analysis_operation")
    def analyze_knowledge_transfer(
        self,
        source_activations: Dict[str, np.ndarray],
        target_activations: Dict[str, np.ndarray],
        source_model: str,
        target_model: str
    ) -> KnowledgeTransferResult:
        """Analyze knowledge transfer between models.

        Args:
            source_activations: Activations from source model
            target_activations: Activations from target model
            source_model: Source model name
            target_model: Target model name

        Returns:
            KnowledgeTransferResult object
        """
        logger.info(f"Analyzing knowledge transfer from {source_model} to {target_model}")

        transfer_scores = {}
        layer_mapping = {}
        transferred_concepts = []

        # Compute transfer scores for each layer pair
        for source_layer, source_acts in source_activations.items():
            best_correlation = 0
            best_target_layer = None

            for target_layer, target_acts in target_activations.items():
                if source_acts.shape == target_acts.shape:
                    # Compute correlation as transfer score
                    correlation = np.corrcoef(
                        source_acts.flatten(),
                        target_acts.flatten()
                    )[0, 1]
                    if not np.isnan(correlation) and abs(correlation) > best_correlation:
                        best_correlation = abs(correlation)
                        best_target_layer = target_layer

            if best_target_layer:
                transfer_scores[f"{source_layer}->{best_target_layer}"] = best_correlation
                layer_mapping[source_layer] = best_target_layer
                if best_correlation > 0.5:  # Threshold for transferred concepts
                    transferred_concepts.append(source_layer)

        # Compute overall transfer efficiency
        overall_transfer_score = np.mean(list(transfer_scores.values())) if transfer_scores else 0.0

        # Analyze preserved, lost, and emergent concepts
        preserved_concepts = [concept for concept in transferred_concepts if transfer_scores.get(f"{concept}->{layer_mapping.get(concept, '')}", 0) > 0.7]
        lost_concepts = [layer for layer in source_activations.keys() if layer not in transferred_concepts]
        emergent_concepts = [layer for layer in target_activations.keys() if layer not in layer_mapping.values()]

        return KnowledgeTransferResult(
            source_model=source_model,
            target_model=target_model,
            transfer_score=overall_transfer_score,
            transferred_concepts=transferred_concepts,
            layer_mapping=layer_mapping,
            transfer_map=transfer_scores,
            preserved_concepts=preserved_concepts,
            lost_concepts=lost_concepts,
            emergent_concepts=emergent_concepts
        )

    def _analyze_spatial_representations(self, layer_acts: np.ndarray, stimuli_metadata: List[Dict[str, Any]]) -> Optional[float]:
        spatial_positions = [meta.get('position', [0, 0]) for meta in stimuli_metadata]
        if all(len(pos) == 2 for pos in spatial_positions):
            spatial_positions = np.array(spatial_positions)
            spatial_corr = []
            for dim in range(min(layer_acts.shape[1], 10)):
                corr_x = np.corrcoef(spatial_positions[:, 0], layer_acts[:, dim])[0, 1]
                corr_y = np.corrcoef(spatial_positions[:, 1], layer_acts[:, dim])[0, 1]
                if not (np.isnan(corr_x) or np.isnan(corr_y)):
                    spatial_corr.append(max(abs(corr_x), abs(corr_y)))
            if spatial_corr:
                return np.mean(spatial_corr)
        return None

    def _analyze_temporal_representations(self, layer_acts: np.ndarray, stimuli_metadata: List[Dict[str, Any]]) -> Optional[float]:
        temporal_sequences = [meta.get('sequence_id', meta.get('time', i)) for i, meta in enumerate(stimuli_metadata)]
        if len(set(temporal_sequences)) > 1:
            temporal_corr = []
            for dim in range(min(layer_acts.shape[1], 10)):
                corr_t = np.corrcoef(temporal_sequences, layer_acts[:, dim])[0, 1]
                if not np.isnan(corr_t):
                    temporal_corr.append(abs(corr_t))
            if temporal_corr:
                return np.mean(temporal_corr)
        return 0.1  # Default if no temporal correlation found

    def _analyze_object_representations(self, layer_acts: np.ndarray, stimuli_metadata: List[Dict[str, Any]]) -> Optional[float]:
        objects = [meta.get('object', 'unknown') for meta in stimuli_metadata]
        unique_objects = list(set(objects))
        if len(unique_objects) > 1:
            object_consistency = []
            for obj in unique_objects:
                obj_indices = [i for i, o in enumerate(objects) if o == obj]
                if len(obj_indices) > 1:
                    obj_acts = layer_acts[obj_indices]
                    consistency = np.mean([
                        np.corrcoef(obj_acts[i], obj_acts[j])[0, 1]
                        for i in range(len(obj_acts))
                        for j in range(i+1, len(obj_acts))
                        if not np.isnan(np.corrcoef(obj_acts[i], obj_acts[j])[0, 1])
                    ])
                    if not np.isnan(consistency):
                        object_consistency.append(consistency)
            if object_consistency:
                return np.mean(object_consistency)
        return None

    def _analyze_relational_representations(self, layer_acts: np.ndarray, stimuli_metadata: List[Dict[str, Any]]) -> Optional[float]:
        relational_consistency = []
        for i in range(len(stimuli_metadata)):
            for j in range(i+1, len(stimuli_metadata)):
                meta_i, meta_j = stimuli_metadata[i], stimuli_metadata[j]
                pos_i, pos_j = meta_i.get('position', [0, 0]), meta_j.get('position', [0, 0])
                obj_i, obj_j = meta_i.get('object', 'unknown'), meta_j.get('object', 'unknown')

                spatial_dist = np.sqrt(sum((pi - pj) ** 2 for pi, pj in zip(pos_i, pos_j)))
                act_dist = np.linalg.norm(layer_acts[i] - layer_acts[j])

                same_object = obj_i == obj_j
                if same_object:
                    relational_consistency.append(1.0 / (1.0 + abs(spatial_dist - act_dist)))
                else:
                    relational_consistency.append(spatial_dist / (1.0 + act_dist))

        if relational_consistency:
            return np.mean(relational_consistency)
        return 0.5  # Default value if no relations found

    def _analyze_causal_representations(self, layer_acts: np.ndarray) -> Optional[float]:
        """Estimate causal sensitivity based on temporal activation differences."""

        if layer_acts.ndim < 2 or layer_acts.shape[0] < 2:
            return 0.0

        diffs = np.diff(layer_acts, axis=0)
        magnitude = np.mean(np.abs(diffs))
        if np.isnan(magnitude):
            return None

        score = 1.0 / (1.0 + magnitude)
        return float(np.clip(score, 0.0, 1.0))

    def analyze_world_model(
        self,
        activations: Dict[str, np.ndarray],
        stimuli_metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze world model representations in neural networks."""
        logger.info("Analyzing learned world model")

        results = {
            'spatial_representations': {},
            'temporal_representations': {},
            'object_representations': {},
            'relational_representations': {},
            'causal_representations': {},
            'consistency_scores': {},
            'world_model_quality': 0.0
        }

        for layer_name, layer_acts in activations.items():
            if len(stimuli_metadata) == layer_acts.shape[0]:
                spatial_score = self._analyze_spatial_representations(layer_acts, stimuli_metadata)
                if spatial_score is not None:
                    results['spatial_representations'][layer_name] = spatial_score

                temporal_score = self._analyze_temporal_representations(layer_acts, stimuli_metadata)
                if temporal_score is not None:
                    results['temporal_representations'][layer_name] = temporal_score

                object_score = self._analyze_object_representations(layer_acts, stimuli_metadata)
                if object_score is not None:
                    results['object_representations'][layer_name] = object_score

                relational_score = self._analyze_relational_representations(layer_acts, stimuli_metadata)
                if relational_score is not None:
                    results['relational_representations'][layer_name] = relational_score

                causal_score = self._analyze_causal_representations(layer_acts)
                if causal_score is not None:
                    results['causal_representations'][layer_name] = causal_score

        all_scores = []
        all_scores.extend(results['spatial_representations'].values())
        all_scores.extend(results['temporal_representations'].values())
        all_scores.extend(results['object_representations'].values())
        all_scores.extend(results['relational_representations'].values())
        all_scores.extend(results['causal_representations'].values())
        if all_scores:
            results['world_model_quality'] = np.mean(all_scores)

        self.world_models['latest'] = results
        return results

    def _calculate_concept_confidence(
        self,
        concept_vector: np.ndarray,
        all_vectors: np.ndarray,
        positive_mask: np.ndarray
    ) -> float:
        """Estimate confidence that a vector captures a concept."""

        if concept_vector.size == 0 or all_vectors.size == 0:
            return 0.0

        concept_norm = np.linalg.norm(concept_vector)
        if concept_norm == 0:
            return 0.0

        normalized_concept = concept_vector / concept_norm
        normalized_vectors = all_vectors / (np.linalg.norm(all_vectors, axis=1, keepdims=True) + 1e-8)
        similarities = normalized_vectors @ normalized_concept

        positives = similarities[positive_mask]
        negatives = similarities[~positive_mask] if (~positive_mask).any() else np.array([0.0])

        pos_mean = float(np.mean(positives)) if positives.size else 0.0
        neg_mean = float(np.mean(negatives)) if negatives.size else 0.0

        confidence = (pos_mean - neg_mean + 1.0) / 2.0
        return float(np.clip(confidence, 0.0, 1.0))

    def _calculate_representation_similarity(
        self,
        activations_a: np.ndarray,
        activations_b: np.ndarray
    ) -> float:
        """Compute a normalized similarity score between two activation sets."""

        if activations_a.shape != activations_b.shape or activations_a.size == 0:
            return 0.0

        flat_a = activations_a.reshape(activations_a.shape[0], -1)
        flat_b = activations_b.reshape(activations_b.shape[0], -1)

        similarities = []
        for vec_a, vec_b in zip(flat_a, flat_b):
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a == 0 or norm_b == 0:
                continue
            cosine = np.dot(vec_a, vec_b) / (norm_a * norm_b)
            similarities.append(cosine)

        if not similarities:
            return 0.0

        mean_cosine = float(np.clip(np.mean(similarities), -1.0, 1.0))
        return (mean_cosine + 1.0) / 2.0

    @log_error("analysis_operation")
    def save_analysis_results(self, output_path: str) -> None:
        """Save analysis results to file.

        Args:
            output_path: Path to save results
        """
        logger.info(f"Saving analysis results to {output_path}")

        results = {
            'concepts': self.concepts,
            'circuits': self.circuits,
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'config': self.config
        }

        # Convert non-serializable objects
        serializable_results = self._make_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Analysis results saved to {output_path}")

    @log_error("analysis_operation")
    def load_analysis_results(self, input_path: str) -> Dict[str, Any]:
        """Load analysis results from file.

        Args:
            input_path: Path to load results from

        Returns:
            Loaded analysis results
        """
        logger.info(f"Loading analysis results from {input_path}")

        with open(input_path, 'r') as f:
            results = json.load(f)

        # Restore concepts
        if 'concepts' in results:
            self.concepts = {}
            for concept_name, concept_data in results['concepts'].items():
                if isinstance(concept_data, dict):
                    concept = ConceptVector(
                        name=concept_data.get('name', concept_name),
                        vector=np.array(concept_data.get('vector', [])),
                        layer=concept_data.get('layer', ''),
                        confidence=concept_data.get('confidence', 0.0),
                        model_name=concept_data.get('model_name', ''),
                        metadata=concept_data.get('metadata', {})
                    )
                    self.concepts[concept_name] = concept

        # Restore circuits
        if 'circuits' in results:
            self.circuits = {}
            for circuit_name, circuit_data in results['circuits'].items():
                if isinstance(circuit_data, dict):
                    circuit = Circuit(
                        name=circuit_data.get('name', circuit_name),
                        components=circuit_data.get('components', []),
                        connections=circuit_data.get('connections', []),
                        function=circuit_data.get('function', ''),
                        evidence_strength=circuit_data.get('evidence_strength', 0.0),
                        metadata=circuit_data.get('metadata', {})
                    )
                    self.circuits[circuit_name] = circuit

        logger.info(f"Analysis results loaded from {input_path}")
        return results

    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (ConceptVector, Circuit, KnowledgeTransferResult)):
            # Convert dataclass to dict first, then make serializable
            obj_dict = asdict(obj) if hasattr(obj, '__dataclass_fields__') else vars(obj)
            return self._make_serializable(obj_dict)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            # Try to convert to string as fallback
            return str(obj)


# Factory function for backward compatibility
def create_conceptual_analyzer(config: dict = None, **kwargs) -> ConceptualAnalyzer:
    """
    Create and return a ConceptualAnalyzer instance.

    Args:
        config: Configuration dictionary
        **kwargs: Additional arguments to pass to ConceptualAnalyzer constructor

    Returns:
        ConceptualAnalyzer instance
    """
    if config is None:
        config = {}

    config.update(kwargs)
    return ConceptualAnalyzer(config)
