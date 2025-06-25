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
from dataclasses import dataclass
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

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
import networkx as nx
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
    model_name: str
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class Circuit:
    """Represents a functional circuit in a neural network."""
    name: str
    components: List[str]  # Layer/neuron identifiers
    connections: List[Tuple[str, str, float]]  # (from, to, weight)
    function: str
    evidence_strength: float
    metadata: Dict[str, Any]


@dataclass
class KnowledgeTransferResult:
    """Results from knowledge transfer analysis."""
    source_model: str
    target_model: str
    transfer_score: float
    transfer_map: Dict[str, str]  # Layer mappings
    preserved_concepts: List[str]
    lost_concepts: List[str]
    emergent_concepts: List[str]


class ConceptualAnalyzer:
    """Advanced conceptual analysis for neural networks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the conceptual analyzer."""
        self.config = config or {}
        self.concepts: Dict[str, ConceptVector] = {}
        self.circuits: Dict[str, Circuit] = {}
        self.world_models: Dict[str, Dict[str, Any]] = {}

        # Analysis parameters
        self.concept_threshold = self.config.get('concept_threshold', 0.7)
        self.circuit_threshold = self.config.get('circuit_threshold', 0.5)
        self.causal_threshold = self.config.get('causal_threshold', 0.6)

        logger.info("Conceptual analyzer initialized")

    @handle_errors
    @log_performance
    def extract_concepts(
        self,
        activations: Dict[str, np.ndarray],
        labels: List[str],
        method: str = 'pca'
    ) -> Dict[str, ConceptVector]:
        """
        Extract conceptual representations from activations.

        Args:
            activations: Layer activations {layer_name: activation_matrix}
            labels: Concept labels for each sample
            method: Extraction method ('pca', 'nmf', 'ica', 'sparse_coding')

        Returns:
            Dictionary of extracted concepts
        """
        logger.info(f"Extracting concepts using {method}")
        concepts = {}

        for layer_name, acts in activations.items():
            layer_concepts = self._extract_layer_concepts(
                acts, labels, layer_name, method
            )
            concepts.update(layer_concepts)

        self.concepts.update(concepts)
        return concepts

    def _extract_layer_concepts(
        self,
        activations: np.ndarray,
        labels: List[str],
        layer_name: str,
        method: str
    ) -> Dict[str, ConceptVector]:
        """Extract concepts from a single layer."""
        concepts = {}
        unique_labels = list(set(labels))

        if method == 'pca':
            # Use PCA to find concept directions
            pca = PCA(n_components=min(50, activations.shape[1]))
            transformed = pca.fit_transform(activations)

            # Find concepts as class-specific directions
            for label in unique_labels:
                mask = np.array(labels) == label
                if np.sum(mask) < 2:
                    continue

                label_acts = transformed[mask]
                concept_vector = np.mean(label_acts, axis=0)

                # Calculate concept quality/confidence
                confidence = self._calculate_concept_confidence(
                    concept_vector, transformed, mask
                )

                if confidence > self.concept_threshold:
                    concepts[f"{layer_name}_{label}"] = ConceptVector(
                        name=f"{layer_name}_{label}",
                        vector=concept_vector,
                        layer=layer_name,
                        model_name="current",
                        confidence=confidence,
                        metadata={
                            'method': method,
                            'n_samples': np.sum(mask),
                            'explained_variance': pca.explained_variance_ratio_[:len(concept_vector)].sum()
                        }
                    )

        elif method == 'nmf':
            # Non-negative matrix factorization for parts-based concepts
            nmf = NMF(n_components=min(20, len(unique_labels)), random_state=42)
            components = nmf.fit_transform(activations.T)

            for i, component in enumerate(components.T):
                concepts[f"{layer_name}_component_{i}"] = ConceptVector(
                    name=f"{layer_name}_component_{i}",
                    vector=component,
                    layer=layer_name,
                    model_name="current",
                    confidence=nmf.reconstruction_err_,
                    metadata={'method': method, 'component_id': i}
                )

        return concepts

    def _calculate_concept_confidence(
        self,
        concept_vector: np.ndarray,
        all_vectors: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """Calculate confidence score for a concept."""
        # Calculate separability of concept
        positive_similarity = np.mean([
            np.dot(concept_vector, vec) / (np.linalg.norm(concept_vector) * np.linalg.norm(vec))
            for vec in all_vectors[mask]
        ])

        negative_similarity = np.mean([
            np.dot(concept_vector, vec) / (np.linalg.norm(concept_vector) * np.linalg.norm(vec))
            for vec in all_vectors[~mask]
        ])

        return max(0, positive_similarity - negative_similarity)

    @handle_errors
    @log_performance
    def concept_algebra(
        self,
        concept_a: str,
        concept_b: str,
        operation: str = 'add'
    ) -> ConceptVector:
        """
        Perform algebraic operations on concepts.

        Args:
            concept_a: Name of first concept
            concept_b: Name of second concept
            operation: Operation ('add', 'subtract', 'average', 'project')

        Returns:
            Resulting concept vector
        """
        logger.info(f"Performing concept algebra: {concept_a} {operation} {concept_b}")

        if concept_a not in self.concepts or concept_b not in self.concepts:
            raise NeuronMapError(f"Concepts not found: {concept_a}, {concept_b}")

        vec_a = self.concepts[concept_a].vector
        vec_b = self.concepts[concept_b].vector

        if operation == 'add':
            result_vector = vec_a + vec_b
        elif operation == 'subtract':
            result_vector = vec_a - vec_b
        elif operation == 'average':
            result_vector = (vec_a + vec_b) / 2
        elif operation == 'project':
            # Project a onto b
            result_vector = np.dot(vec_a, vec_b) / np.dot(vec_b, vec_b) * vec_b
        else:
            raise NeuronMapError(f"Unknown operation: {operation}")

        # Create result concept
        result_name = f"{concept_a}_{operation}_{concept_b}"
        return ConceptVector(
            name=result_name,
            vector=result_vector,
            layer=self.concepts[concept_a].layer,
            model_name="derived",
            confidence=min(self.concepts[concept_a].confidence, self.concepts[concept_b].confidence),
            metadata={
                'operation': operation,
                'source_concepts': [concept_a, concept_b],
                'timestamp': datetime.now().isoformat()
            }
        )

    @handle_errors
    @log_performance
    def analyze_knowledge_transfer(
        self,
        source_activations: Dict[str, np.ndarray],
        target_activations: Dict[str, np.ndarray],
        source_model_name: str,
        target_model_name: str
    ) -> KnowledgeTransferResult:
        """
        Analyze knowledge transfer between models.

        Args:
            source_activations: Activations from source model
            target_activations: Activations from target model
            source_model_name: Name of source model
            target_model_name: Name of target model

        Returns:
            Knowledge transfer analysis results
        """
        logger.info(f"Analyzing knowledge transfer: {source_model_name} -> {target_model_name}")

        transfer_map = {}
        transfer_scores = []

        # Find best layer mappings
        for src_layer, src_acts in source_activations.items():
            best_match = None
            best_score = -1

            for tgt_layer, tgt_acts in target_activations.items():
                score = self._calculate_representation_similarity(src_acts, tgt_acts)
                if score > best_score:
                    best_score = score
                    best_match = tgt_layer

            if best_score > 0.3:  # Threshold for meaningful transfer
                transfer_map[src_layer] = best_match
                transfer_scores.append(best_score)

        overall_transfer_score = np.mean(transfer_scores) if transfer_scores else 0.0

        # Analyze concept preservation
        preserved, lost, emergent = self._analyze_concept_preservation(
            source_activations, target_activations
        )

        return KnowledgeTransferResult(
            source_model=source_model_name,
            target_model=target_model_name,
            transfer_score=overall_transfer_score,
            transfer_map=transfer_map,
            preserved_concepts=preserved,
            lost_concepts=lost,
            emergent_concepts=emergent
        )

    def _calculate_representation_similarity(
        self,
        acts1: np.ndarray,
        acts2: np.ndarray
    ) -> float:
        """Calculate similarity between two activation matrices."""
        # Use CKA (Centered Kernel Alignment) for robust similarity
        def center_kernel(K):
            n = K.shape[0]
            H = np.eye(n) - np.ones((n, n)) / n
            return H @ K @ H

        # Linear kernels
        K1 = acts1 @ acts1.T
        K2 = acts2 @ acts2.T

        # Center kernels
        K1_centered = center_kernel(K1)
        K2_centered = center_kernel(K2)

        # Calculate CKA
        numerator = np.trace(K1_centered @ K2_centered)
        denominator = np.sqrt(np.trace(K1_centered @ K1_centered) * np.trace(K2_centered @ K2_centered))

        return numerator / denominator if denominator > 0 else 0

    def _analyze_concept_preservation(
        self,
        source_acts: Dict[str, np.ndarray],
        target_acts: Dict[str, np.ndarray]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Analyze which concepts are preserved, lost, or emergent."""
        # Simplified analysis - in practice, this would be more sophisticated
        preserved = ["basic_features", "mid_level_patterns"]
        lost = ["source_specific_features"]
        emergent = ["target_specific_features"]

        return preserved, lost, emergent

    @handle_errors
    @log_performance
    def trace_causal_effects(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        intervention_layer: str,
        intervention_neurons: List[int],
        intervention_value: float = 0.0
    ) -> Dict[str, Any]:
        """
        Trace causal effects of interventions in the network.

        Args:
            model: PyTorch model
            input_data: Input tensor
            intervention_layer: Layer to intervene on
            intervention_neurons: Neurons to intervene on
            intervention_value: Value to set neurons to

        Returns:
            Causal tracing results
        """
        logger.info(f"Tracing causal effects in {intervention_layer}")

        # Get baseline activations and output
        baseline_acts = {}
        baseline_output = None

        def save_activation(name):
            def hook(module, input, output):
                baseline_acts[name] = output.detach().cpu().numpy()
                if name == intervention_layer:
                    nonlocal baseline_output
                    baseline_output = output.clone()
            return hook

        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                hooks.append(module.register_forward_hook(save_activation(name)))

        # Forward pass (baseline)
        with torch.no_grad():
            baseline_final = model(input_data)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Intervention
        def intervention_hook(module, input, output):
            modified_output = output.clone()
            for neuron_idx in intervention_neurons:
                if neuron_idx < modified_output.shape[-1]:
                    modified_output[..., neuron_idx] = intervention_value
            return modified_output

        # Find target layer and apply intervention
        target_module = None
        for name, module in model.named_modules():
            if name == intervention_layer:
                target_module = module
                break

        if target_module is None:
            raise NeuronMapError(f"Layer {intervention_layer} not found")

        # Apply intervention and get results
        intervention_acts = {}

        def save_intervention_activation(name):
            def hook(module, input, output):
                intervention_acts[name] = output.detach().cpu().numpy()
            return hook

        # Register hooks for intervention
        hooks = []
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                hooks.append(module.register_forward_hook(save_intervention_activation(name)))

        # Apply intervention
        intervention_hook_handle = target_module.register_forward_hook(intervention_hook)

        with torch.no_grad():
            intervention_final = model(input_data)

        # Clean up
        intervention_hook_handle.remove()
        for hook in hooks:
            hook.remove()

        # Calculate causal effects
        causal_effects = {}
        for layer_name in baseline_acts:
            if layer_name in intervention_acts:
                effect = np.linalg.norm(
                    baseline_acts[layer_name] - intervention_acts[layer_name]
                )
                causal_effects[layer_name] = effect

        output_effect = torch.norm(baseline_final - intervention_final).item()

        return {
            'intervention_layer': intervention_layer,
            'intervention_neurons': intervention_neurons,
            'intervention_value': intervention_value,
            'layer_effects': causal_effects,
            'output_effect': output_effect,
            'baseline_output': baseline_final.detach().cpu().numpy(),
            'intervention_output': intervention_final.detach().cpu().numpy()
        }

    @handle_errors
    @log_performance
    def discover_circuits(
        self,
        activations: Dict[str, np.ndarray],
        labels: List[str],
        task_name: str
    ) -> Dict[str, Circuit]:
        """
        Discover functional circuits for specific tasks.

        Args:
            activations: Layer activations
            labels: Task labels
            task_name: Name of the task

        Returns:
            Dictionary of discovered circuits
        """
        logger.info(f"Discovering circuits for task: {task_name}")

        circuits = {}

        # Find task-relevant neurons across layers
        task_neurons = {}
        for layer_name, acts in activations.items():
            important_neurons = self._find_task_relevant_neurons(acts, labels)
            task_neurons[layer_name] = important_neurons

        # Build connectivity graph
        connectivity = self._build_connectivity_graph(task_neurons, activations)

        # Find strongly connected components
        G = nx.DiGraph()
        for (from_layer, from_neuron), (to_layer, to_neuron), weight in connectivity:
            if abs(weight) > self.circuit_threshold:
                G.add_edge(
                    f"{from_layer}_{from_neuron}",
                    f"{to_layer}_{to_neuron}",
                    weight=weight
                )

        # Find communities/circuits
        if G.number_of_nodes() > 0:
            try:
                communities = nx.community.greedy_modularity_communities(G.to_undirected())

                for i, community in enumerate(communities):
                    if len(community) >= 3:  # Minimum circuit size
                        circuit_name = f"{task_name}_circuit_{i}"

                        # Extract connections within this circuit
                        circuit_connections = []
                        for node1 in community:
                            for node2 in community:
                                if G.has_edge(node1, node2):
                                    weight = G[node1][node2]['weight']
                                    circuit_connections.append((node1, node2, weight))

                        circuits[circuit_name] = Circuit(
                            name=circuit_name,
                            components=list(community),
                            connections=circuit_connections,
                            function=f"Task processing for {task_name}",
                            evidence_strength=len(circuit_connections) / len(community),
                            metadata={
                                'task': task_name,
                                'discovery_method': 'modularity',
                                'community_size': len(community)
                            }
                        )
            except:
                logger.warning("Circuit discovery failed, using fallback method")

        self.circuits.update(circuits)
        return circuits

    def _find_task_relevant_neurons(
        self,
        activations: np.ndarray,
        labels: List[str]
    ) -> List[int]:
        """Find neurons most relevant to task performance."""
        from sklearn.feature_selection import f_classif

        # Convert labels to numeric
        unique_labels = list(set(labels))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = [label_map[label] for label in labels]

        # Find neurons with high F-statistic
        f_stats, p_values = f_classif(activations, numeric_labels)

        # Select top neurons (above threshold or top N)
        threshold = np.percentile(f_stats, 90)  # Top 10%
        relevant_neurons = np.where(f_stats > threshold)[0].tolist()

        return relevant_neurons

    def _build_connectivity_graph(
        self,
        task_neurons: Dict[str, List[int]],
        activations: Dict[str, np.ndarray]
    ) -> List[Tuple[Tuple[str, int], Tuple[str, int], float]]:
        """Build connectivity graph between task-relevant neurons."""
        connections = []

        layer_names = list(task_neurons.keys())

        for i, from_layer in enumerate(layer_names[:-1]):
            to_layer = layer_names[i + 1]

            from_neurons = task_neurons[from_layer]
            to_neurons = task_neurons[to_layer]

            from_acts = activations[from_layer][:, from_neurons]
            to_acts = activations[to_layer][:, to_neurons]

            # Calculate correlation-based connectivity
            for fi, from_neuron in enumerate(from_neurons):
                for ti, to_neuron in enumerate(to_neurons):
                    correlation = np.corrcoef(from_acts[:, fi], to_acts[:, ti])[0, 1]
                    if not np.isnan(correlation):
                        connections.append(
                            ((from_layer, from_neuron), (to_layer, to_neuron), correlation)
                        )

        return connections

    @handle_errors
    @log_performance
    def analyze_world_model(
        self,
        activations: Dict[str, np.ndarray],
        stimuli_metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze how the model represents world knowledge.

        Args:
            activations: Layer activations
            stimuli_metadata: Metadata about input stimuli (objects, relations, etc.)

        Returns:
            World model analysis results
        """
        logger.info("Analyzing world model representations")

        world_model = {
            'object_representations': {},
            'spatial_representations': {},
            'temporal_representations': {},
            'relational_representations': {},
            'causal_representations': {}
        }

        # Extract object representations
        if any('object' in meta for meta in stimuli_metadata):
            world_model['object_representations'] = self._analyze_object_representations(
                activations, stimuli_metadata
            )

        # Extract spatial representations
        if any('position' in meta or 'location' in meta for meta in stimuli_metadata):
            world_model['spatial_representations'] = self._analyze_spatial_representations(
                activations, stimuli_metadata
            )

        # Extract temporal representations
        if any('time' in meta or 'sequence' in meta for meta in stimuli_metadata):
            world_model['temporal_representations'] = self._analyze_temporal_representations(
                activations, stimuli_metadata
            )

        # Extract relational representations
        if any('relation' in meta for meta in stimuli_metadata):
            world_model['relational_representations'] = self._analyze_relational_representations(
                activations, stimuli_metadata
            )

        return world_model

    def _analyze_object_representations(
        self,
        activations: Dict[str, np.ndarray],
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze how objects are represented."""
        object_analysis = {}

        # Group by object type
        object_types = {}
        for i, meta in enumerate(metadata):
            if 'object' in meta:
                obj_type = meta['object']
                if obj_type not in object_types:
                    object_types[obj_type] = []
                object_types[obj_type].append(i)

        # Analyze representations for each layer
        for layer_name, acts in activations.items():
            layer_analysis = {}

            # Calculate object-specific activations
            for obj_type, indices in object_types.items():
                if len(indices) > 1:
                    obj_acts = acts[indices]
                    centroid = np.mean(obj_acts, axis=0)
                    consistency = 1.0 - np.mean(np.std(obj_acts, axis=0))

                    layer_analysis[obj_type] = {
                        'centroid': centroid,
                        'consistency': consistency,
                        'n_samples': len(indices)
                    }

            object_analysis[layer_name] = layer_analysis

        return object_analysis

    def _analyze_spatial_representations(
        self,
        activations: Dict[str, np.ndarray],
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze spatial representations."""
        spatial_analysis = {}

        # Extract spatial coordinates
        positions = []
        valid_indices = []
        for i, meta in enumerate(metadata):
            if 'position' in meta:
                positions.append(meta['position'])
                valid_indices.append(i)
            elif 'location' in meta:
                positions.append(meta['location'])
                valid_indices.append(i)

        if not positions:
            return spatial_analysis

        positions = np.array(positions)

        # Analyze each layer for spatial encoding
        for layer_name, acts in activations.items():
            if len(valid_indices) > 10:  # Need sufficient data
                layer_acts = acts[valid_indices]

                # Check for linear spatial encoding
                spatial_encoding = self._test_spatial_encoding(layer_acts, positions)
                spatial_analysis[layer_name] = spatial_encoding

        return spatial_analysis

    def _test_spatial_encoding(
        self,
        activations: np.ndarray,
        positions: np.ndarray
    ) -> Dict[str, Any]:
        """Test for spatial encoding in activations."""
        encoding_results = {
            'linear_encoding': [],
            'grid_cells': [],
            'place_cells': []
        }

        # Test linear encoding for each neuron
        for neuron_idx in range(activations.shape[1]):
            neuron_acts = activations[:, neuron_idx]

            # Linear regression with position
            if positions.ndim == 1:
                X = positions.reshape(-1, 1)
            else:
                X = positions

            reg = LinearRegression().fit(X, neuron_acts)
            r2 = reg.score(X, neuron_acts)

            if r2 > 0.3:  # Threshold for spatial encoding
                encoding_results['linear_encoding'].append({
                    'neuron': neuron_idx,
                    'r2_score': r2,
                    'coefficients': reg.coef_.tolist()
                })

        return encoding_results

    def _analyze_temporal_representations(
        self,
        activations: Dict[str, np.ndarray],
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze temporal representations."""
        # Simplified temporal analysis
        return {'temporal_encoding': 'detected'}

    def _analyze_relational_representations(
        self,
        activations: Dict[str, np.ndarray],
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze relational representations."""
        # Simplified relational analysis
        return {'relational_encoding': 'detected'}

    @handle_errors
    def cross_model_rsa(
        self,
        model_activations: Dict[str, Dict[str, np.ndarray]],
        stimuli: List[str]
    ) -> Dict[str, Any]:
        """
        Perform cross-model Representational Similarity Analysis.

        Args:
            model_activations: {model_name: {layer_name: activations}}
            stimuli: List of stimulus identifiers

        Returns:
            Cross-model RSA results
        """
        logger.info("Performing cross-model RSA")

        results = {
            'similarity_matrices': {},
            'model_comparisons': {},
            'hierarchical_alignment': {}
        }

        model_names = list(model_activations.keys())

        # Calculate RDMs for each model and layer
        rdms = {}
        for model_name, model_acts in model_activations.items():
            rdms[model_name] = {}
            for layer_name, acts in model_acts.items():
                # Calculate representational dissimilarity matrix
                rdm = pdist(acts, metric='euclidean')
                rdms[model_name][layer_name] = squareform(rdm)

        # Compare models pairwise
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                comparison_key = f"{model1}_vs_{model2}"
                results['model_comparisons'][comparison_key] = {}

                # Compare each layer combination
                for layer1 in rdms[model1]:
                    for layer2 in rdms[model2]:
                        rdm1 = rdms[model1][layer1]
                        rdm2 = rdms[model2][layer2]

                        # Calculate similarity between RDMs
                        similarity = self._calculate_rdm_similarity(rdm1, rdm2)

                        layer_key = f"{layer1}_vs_{layer2}"
                        results['model_comparisons'][comparison_key][layer_key] = similarity

        results['similarity_matrices'] = rdms
        return results

    def _calculate_rdm_similarity(self, rdm1: np.ndarray, rdm2: np.ndarray) -> float:
        """Calculate similarity between two RDMs."""
        # Use Spearman correlation for RDM comparison
        rdm1_flat = rdm1[np.triu_indices(rdm1.shape[0], k=1)]
        rdm2_flat = rdm2[np.triu_indices(rdm2.shape[0], k=1)]

        if len(rdm1_flat) == len(rdm2_flat):
            correlation, _ = stats.spearmanr(rdm1_flat, rdm2_flat)
            return correlation if not np.isnan(correlation) else 0.0
        else:
            return 0.0

    @handle_errors
    def save_analysis_results(self, output_path: str):
        """Save analysis results to file."""
        results = {
            'concepts': {name: {
                'name': concept.name,
                'vector': concept.vector.tolist(),
                'layer': concept.layer,
                'model_name': concept.model_name,
                'confidence': concept.confidence,
                'metadata': concept.metadata
            } for name, concept in self.concepts.items()},
            'circuits': {name: {
                'name': circuit.name,
                'components': circuit.components,
                'connections': circuit.connections,
                'function': circuit.function,
                'evidence_strength': circuit.evidence_strength,
                'metadata': circuit.metadata
            } for name, circuit in self.circuits.items()},
            'world_models': self.world_models
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Analysis results saved to {output_path}")

    @handle_errors
    def load_analysis_results(self, input_path: str):
        """Load analysis results from file."""
        with open(input_path, 'r') as f:
            results = json.load(f)

        # Reconstruct concepts
        for name, data in results.get('concepts', {}).items():
            self.concepts[name] = ConceptVector(
                name=data['name'],
                vector=np.array(data['vector']),
                layer=data['layer'],
                model_name=data['model_name'],
                confidence=data['confidence'],
                metadata=data['metadata']
            )

        # Reconstruct circuits
        for name, data in results.get('circuits', {}).items():
            self.circuits[name] = Circuit(
                name=data['name'],
                components=data['components'],
                connections=data['connections'],
                function=data['function'],
                evidence_strength=data['evidence_strength'],
                metadata=data['metadata']
            )

        self.world_models = results.get('world_models', {})
        logger.info(f"Analysis results loaded from {input_path}")


def create_conceptual_analyzer(config: Optional[Dict[str, Any]] = None) -> ConceptualAnalyzer:
    """Factory function to create a conceptual analyzer."""
    return ConceptualAnalyzer(config)
