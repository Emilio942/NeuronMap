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
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
import warnings
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE

# Optional dependencies with fallbacks
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")

try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import accuracy_score, r2_score
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

from pathlib import Path
import pickle
import json
from datetime import datetime

from ..utils.error_handling import handle_errors, NeuronMapError
from ..utils.monitoring import log_performance

logger = logging.getLogger(__name__)


@dataclass
class ConceptVector:
    """Represents a concept as a vector in activation space."""
    name: str
    vector: np.ndarray
    layer: str
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class Circuit:
    """Represents a computational circuit in a neural network."""
    name: str
    nodes: List[Tuple[str, int]]  # (layer, neuron_idx)
    edges: List[Tuple[Tuple[str, int], Tuple[str, int], float]]  # (from, to, weight)
    function: str
    confidence: float


@dataclass
class KnowledgeTransferResult:
    """Results from knowledge transfer analysis."""
    source_model: str
    target_model: str
    transfer_score: float
    transferred_concepts: List[str]
    layer_mapping: Dict[str, str]


class ConceptualAnalyzer:
    """Advanced conceptual analysis for neural network interpretability."""

    def __init__(self, device: str = "auto"):
        """Initialize the conceptual analyzer.

        Args:
            device: Device to use for computation ('auto', 'cpu', 'cuda')
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"ConceptualAnalyzer initialized with device: {self.device}")

    @handle_errors
    @log_performance
    def extract_concepts(
        self,
        activations: Dict[str, np.ndarray],
        method: str = "pca",
        n_concepts: int = 10,
        **kwargs
    ) -> Dict[str, List[ConceptVector]]:
        """Extract concepts from neural network activations.

        Args:
            activations: Dictionary mapping layer names to activation arrays
            method: Extraction method ('pca', 'nmf', 'ica')
            n_concepts: Number of concepts to extract per layer
            **kwargs: Additional arguments for the extraction method

        Returns:
            Dictionary mapping layer names to lists of ConceptVector objects
        """
        logger.info(f"Extracting concepts using {method} method")

        concepts = {}

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

            concepts[layer_name] = layer_concepts
            logger.debug(f"Extracted {len(layer_concepts)} concepts from {layer_name}")

        return concepts

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

    @handle_errors
    @log_performance
    def concept_algebra(
        self,
        concepts: Dict[str, List[ConceptVector]],
        operations: List[Tuple[str, str, str, str]]  # (op, concept1, concept2, result_name)
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

    @handle_errors
    @log_performance
    def discover_circuits(
        self,
        model: nn.Module,
        activations: Dict[str, np.ndarray],
        concepts: Dict[str, List[ConceptVector]],
        threshold: float = 0.1
    ) -> List[Circuit]:
        """Discover computational circuits in the neural network.

        Args:
            model: The neural network model
            activations: Layer activations
            concepts: Extracted concepts
            threshold: Minimum connection strength to include in circuit

        Returns:
            List of discovered Circuit objects
        """
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available, cannot perform circuit discovery")
            return []

        logger.info("Discovering computational circuits")

        circuits = []

        # Build connectivity graph
        G = nx.DiGraph()

        # Add nodes for each concept
        for layer_name, layer_concepts in concepts.items():
            for i, concept in enumerate(layer_concepts):
                node_id = f"{layer_name}_{i}"
                G.add_node(node_id,
                          layer=layer_name,
                          concept=concept.name,
                          confidence=concept.confidence)

        # Add edges based on activation correlation
        layer_names = list(activations.keys())
        for i in range(len(layer_names) - 1):
            current_layer = layer_names[i]
            next_layer = layer_names[i + 1]

            current_acts = activations[current_layer]
            next_acts = activations[next_layer]

            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(current_acts.T, next_acts.T)
            current_size = current_acts.shape[1]

            # Extract cross-correlation block
            cross_corr = correlation_matrix[:current_size, current_size:]

            # Add edges above threshold
            for src_idx in range(cross_corr.shape[0]):
                for dst_idx in range(cross_corr.shape[1]):
                    corr_strength = abs(cross_corr[src_idx, dst_idx])
                    if corr_strength > threshold:
                        src_node = f"{current_layer}_{src_idx}"
                        dst_node = f"{next_layer}_{dst_idx}"
                        if G.has_node(src_node) and G.has_node(dst_node):
                            G.add_edge(src_node, dst_node, weight=corr_strength)

        # Find strongly connected components as circuits
        try:
            components = list(nx.weakly_connected_components(G))

            for i, component in enumerate(components):
                if len(component) > 1:  # Only consider multi-node circuits
                    subgraph = G.subgraph(component)

                    # Extract circuit information
                    nodes = []
                    edges = []

                    for node in component:
                        layer = G.nodes[node]['layer']
                        # Extract neuron index from node ID
                        neuron_idx = int(node.split('_')[-1])
                        nodes.append((layer, neuron_idx))

                    for edge in subgraph.edges(data=True):
                        src_layer = G.nodes[edge[0]]['layer']
                        src_idx = int(edge[0].split('_')[-1])
                        dst_layer = G.nodes[edge[1]]['layer']
                        dst_idx = int(edge[1].split('_')[-1])
                        weight = edge[2]['weight']

                        edges.append(((src_layer, src_idx), (dst_layer, dst_idx), weight))

                    # Calculate circuit confidence as average edge weight
                    avg_weight = np.mean([edge[2] for edge in edges])

                    circuit = Circuit(
                        name=f"circuit_{i}",
                        nodes=nodes,
                        edges=edges,
                        function="unknown",  # Could be inferred from concept analysis
                        confidence=float(avg_weight)
                    )

                    circuits.append(circuit)

        except Exception as e:
            logger.warning(f"Error in circuit discovery: {e}")

        logger.info(f"Discovered {len(circuits)} circuits")
        return circuits

    @handle_errors
    @log_performance
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

    @handle_errors
    @log_performance
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
        activations = []

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

        def hook_fn(module, input, output):
            layer_activations.append(output.detach())

        # Register hook
        for name, module in model.named_modules():
            if name == target_layer:
                handle = module.register_forward_hook(hook_fn)
                break

        # Forward pass
        _ = model(sequence.unsqueeze(0))

        # Remove hook
        handle.remove()

        if layer_activations:
            # Extract activations for each sequence position
            batch_activations = layer_activations[0]
            if batch_activations.dim() >= 2:
                return [batch_activations[0, i] for i in range(batch_activations.shape[1])]
            else:
                return [batch_activations[0]]

        return []

    @handle_errors
    @log_performance
    def cross_model_rsa(
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

    @handle_errors
    @log_performance
    def knowledge_transfer_analysis(
        self,
        source_model: nn.Module,
        target_model: nn.Module,
        transfer_data: torch.Tensor,
        task_data: torch.Tensor = None
    ) -> KnowledgeTransferResult:
        """Analyze knowledge transfer between models.

        Args:
            source_model: Source model (pre-trained)
            target_model: Target model (fine-tuned or transferred)
            transfer_data: Data used for transfer/fine-tuning
            task_data: Task-specific evaluation data

        Returns:
            KnowledgeTransferResult object with transfer analysis
        """
        logger.info("Analyzing knowledge transfer between models")

        # Get layer mappings (assume similar architectures for now)
        source_layers = [name for name, _ in source_model.named_modules()
                        if isinstance(_, (nn.Linear, nn.Conv2d))]
        target_layers = [name for name, _ in target_model.named_modules()
                        if isinstance(_, (nn.Linear, nn.Conv2d))]

        layer_mapping = {}
        for i, (src_layer, tgt_layer) in enumerate(zip(source_layers, target_layers)):
            layer_mapping[src_layer] = tgt_layer

        # Perform RSA between corresponding layers
        if len(source_layers) > 0 and len(target_layers) > 0:
            rsa_matrix = self.cross_model_rsa(
                source_model, target_model, transfer_data,
                source_layers[:min(5, len(source_layers))],  # Limit to avoid memory issues
                target_layers[:min(5, len(target_layers))]
            )

            # Calculate transfer score as average RSA
            transfer_score = np.mean(np.diag(rsa_matrix[:min(len(source_layers), len(target_layers))]))
        else:
            transfer_score = 0.0

        # Identify transferred concepts (simplified)
        transferred_concepts = []
        if transfer_score > 0.5:  # Threshold for successful transfer
            transferred_concepts = [f"concept_{i}" for i in range(min(10, int(transfer_score * 20)))]

        result = KnowledgeTransferResult(
            source_model=source_model.__class__.__name__,
            target_model=target_model.__class__.__name__,
            transfer_score=float(transfer_score),
            transferred_concepts=transferred_concepts,
            layer_mapping=layer_mapping
        )

        logger.info(f"Knowledge transfer analysis completed. Score: {transfer_score:.3f}")
        return result

    def save_analysis(self, results: Dict[str, Any], output_path: Path) -> None:
        """Save analysis results to file.

        Args:
            results: Analysis results dictionary
            output_path: Path to save results
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert non-serializable objects
        serializable_results = self._make_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Analysis results saved to {output_path}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (ConceptVector, Circuit, KnowledgeTransferResult)):
            return asdict(obj) if hasattr(obj, '__dataclass_fields__') else str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
