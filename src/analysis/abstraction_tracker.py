"""
Abstraction Tracking Engine

This module implements tools for tracking how concepts and abstractions evolve
through the layers of a transformer model, using techniques like Concept 
Activation Vectors (CAVs) and layer-wise representation analysis.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .model_integration import ModelManager

# Import config from utils
try:
    from ..utils.config import AnalysisConfig
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.config import AnalysisConfig

logger = logging.getLogger(__name__)


@dataclass
class ConceptVector:
    """Represents a concept vector for abstraction tracking."""
    concept_name: str
    vector: np.ndarray
    layer_index: int
    confidence: float

    # Metadata
    creation_method: str  # 'manual', 'learned', 'interpolated'
    source_examples: List[str] = field(default_factory=list)
    validation_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'concept_name': self.concept_name,
            'vector': self.vector.tolist(),
            'layer_index': self.layer_index,
            'confidence': float(self.confidence),
            'creation_method': self.creation_method,
            'source_examples': self.source_examples,
            'validation_score': float(self.validation_score) if self.validation_score else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConceptVector':
        """Create from dictionary."""
        data['vector'] = np.array(data['vector'])
        return cls(**data)


@dataclass
class AbstractionTrajectory:
    """Represents how a concept evolves through model layers."""
    concept_name: str
    input_text: str
    token_index: int
    layer_similarities: List[float]  # Similarity to concept at each layer
    layer_activations: List[np.ndarray]  # Raw activations at each layer

    # Analysis results
    peak_layer: int = -1
    peak_similarity: float = 0.0
    emergence_layer: int = -1  # First layer where concept becomes detectable
    saturation_layer: int = -1  # Layer where concept stabilizes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'concept_name': self.concept_name,
            'input_text': self.input_text,
            'token_index': self.token_index,
            'layer_similarities': self.layer_similarities,
            'layer_activations': [act.tolist() for act in self.layer_activations],
            'peak_layer': self.peak_layer,
            'peak_similarity': float(self.peak_similarity),
            'emergence_layer': self.emergence_layer,
            'saturation_layer': self.saturation_layer
        }


@dataclass
class AbstractionAnalysisResult:
    """Complete abstraction analysis results."""
    model_name: str
    trajectories: List[AbstractionTrajectory]
    concept_vectors: Dict[str, List[ConceptVector]]  # concept_name -> list of layer vectors

    # Global analysis
    layer_abstraction_scores: List[float]  # Overall abstraction level per layer
    concept_emergence_patterns: Dict[str, List[int]]  # concept_name -> emergence layers

    # Visualization data
    dimensionality_reduction: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'trajectories': [t.to_dict() for t in self.trajectories],
            'concept_vectors': {
                name: [cv.to_dict() for cv in vectors]
                for name, vectors in self.concept_vectors.items()
            },
            'layer_abstraction_scores': self.layer_abstraction_scores,
            'concept_emergence_patterns': self.concept_emergence_patterns,
            'dimensionality_reduction': self.dimensionality_reduction
        }


class ConceptVectorBuilder:
    """Builds concept vectors from examples."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.device = next(model_manager.model.parameters()).device

    def build_concept_vector_from_examples(
        self,
        concept_name: str,
        positive_examples: List[str],
        negative_examples: List[str],
        layer_index: int,
        method: str = 'difference'
    ) -> ConceptVector:
        """
        Build a concept vector from positive and negative examples.
    
        Args:
            concept_name: Name of the concept
            positive_examples: Examples that contain the concept
            negative_examples: Examples that don't contain the concept
            layer_index: Layer to extract activations from
            method: Method for combining examples ('difference', 'classification')
        
        Returns:
            Concept vector
        """
        logger.info(f"Building concept vector for '{concept_name}' at layer {layer_index}")
    
        # Get activations for examples
        pos_activations = self._get_activations_for_texts(positive_examples, layer_index)
        neg_activations = self._get_activations_for_texts(negative_examples, layer_index)
    
        if method == 'difference':
            # Simple difference method
            pos_mean = np.mean(pos_activations, axis=0)
            neg_mean = np.mean(neg_activations, axis=0)
            concept_vector = pos_mean - neg_mean
        
            # Normalize
            concept_vector = concept_vector / np.linalg.norm(concept_vector)
        
            # Compute confidence (separation between positive and negative)
            pos_similarities = [cosine_similarity([concept_vector], [act])[0, 0] for act in pos_activations]
            neg_similarities = [cosine_similarity([concept_vector], [act])[0, 0] for act in neg_activations]
        
            confidence = np.mean(pos_similarities) - np.mean(neg_similarities)
    
        elif method == 'classification':
            # Use a simple linear classifier
            from sklearn.linear_model import LogisticRegression
        
            X = np.vstack([pos_activations, neg_activations])
            y = np.concatenate([np.ones(len(pos_activations)), np.zeros(len(neg_activations))])
        
            classifier = LogisticRegression()
            classifier.fit(X, y)
        
            concept_vector = classifier.coef_[0]
            concept_vector = concept_vector / np.linalg.norm(concept_vector)
            confidence = classifier.score(X, y)
    
        else:
            raise ValueError(f"Unknown method: {method}")
    
        return ConceptVector(
            concept_name=concept_name,
            vector=concept_vector,
            layer_index=layer_index,
            confidence=confidence,
            creation_method=method,
            source_examples=positive_examples + negative_examples
        )

    def _get_activations_for_texts(self, texts: List[str], layer_index: int) -> np.ndarray:
        """Get activations for a list of texts at a specific layer."""
        all_activations = []
    
        # Set up hook
        activations_cache = {}
    
        def activation_hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations_cache['layer_activations'] = output.detach()
    
        # Register hook
        model = self.model_manager.model
        if hasattr(model, 'transformer'):
            hook_module = model.transformer.h[layer_index]
        elif hasattr(model, 'encoder'):
            hook_module = model.encoder.layer[layer_index]
        else:
            raise ValueError("Unknown model architecture")
    
        hook = hook_module.register_forward_hook(activation_hook)
    
        try:
            for text in texts:
                # Tokenize
                inputs = self.model_manager.model.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
            
                # Forward pass
                with torch.no_grad():
                    _ = model(**inputs)
            
                if 'layer_activations' in activations_cache:
                    # Use mean pooling over sequence dimension
                    activations = activations_cache['layer_activations']
                    pooled = torch.mean(activations, dim=1)  # [batch, hidden] -> [hidden]
                    all_activations.append(pooled.cpu().numpy())
            
                activations_cache.clear()
    
        finally:
            hook.remove()
    
        return np.vstack(all_activations)

    def interpolate_concept_across_layers(
        self,
        concept_vector: ConceptVector,
        target_layers: List[int],
        interpolation_texts: List[str]
    ) -> List[ConceptVector]:
        """
        Interpolate a concept vector across multiple layers.
    
        Args:
            concept_vector: Base concept vector
            target_layers: Layers to interpolate to
            interpolation_texts: Texts to use for interpolation
        
        Returns:
            List of concept vectors for target layers
        """
        logger.info(f"Interpolating concept '{concept_vector.concept_name}' across layers")
    
        interpolated_vectors = []
    
        for layer_idx in target_layers:
            if layer_idx == concept_vector.layer_index:
                # Use original vector
                interpolated_vectors.append(concept_vector)
            else:
                # Get activations at target layer
                activations = self._get_activations_for_texts(interpolation_texts, layer_idx)
            
                # Project original concept vector to this layer's space
                # This is a simplified approach - more sophisticated methods could be used
                mean_activation = np.mean(activations, axis=0)
            
                # Simple linear interpolation (could be improved)
                if activations.shape[1] == len(concept_vector.vector):
                    # Same dimensionality
                    interpolated_vector = concept_vector.vector.copy()
                else:
                    # Different dimensionality - use PCA or padding
                    if activations.shape[1] > len(concept_vector.vector):
                        # Pad with zeros
                        interpolated_vector = np.pad(
                            concept_vector.vector, 
                            (0, activations.shape[1] - len(concept_vector.vector))
                        )
                    else:
                        # Truncate
                        interpolated_vector = concept_vector.vector[:activations.shape[1]]
            
                # Normalize
                interpolated_vector = interpolated_vector / np.linalg.norm(interpolated_vector)
            
                interpolated_cv = ConceptVector(
                    concept_name=concept_vector.concept_name,
                    vector=interpolated_vector,
                    layer_index=layer_idx,
                    confidence=concept_vector.confidence * 0.8,  # Reduce confidence for interpolated
                    creation_method='interpolated',
                    source_examples=concept_vector.source_examples
                )
            
                interpolated_vectors.append(interpolated_cv)
    
        return interpolated_vectors


class AbstractionTracker:
    """Main abstraction tracking engine."""

    def __init__(self, model_manager: ModelManager, config: Optional[AnalysisConfig] = None):
        self.model_manager = model_manager
        self.config = config or AnalysisConfig()
        self.device = next(model_manager.model.parameters()).device
    
        self.concept_builder = ConceptVectorBuilder(model_manager)
        self.concept_vectors = {}  # concept_name -> List[ConceptVector]

    def add_concept_vector(self, concept_vector: ConceptVector):
        """Add a concept vector to the tracker."""
        concept_name = concept_vector.concept_name
        if concept_name not in self.concept_vectors:
            self.concept_vectors[concept_name] = []
    
        self.concept_vectors[concept_name].append(concept_vector)
        logger.info(f"Added concept vector for '{concept_name}' at layer {concept_vector.layer_index}")

    def build_and_add_concept(
        self,
        concept_name: str,
        positive_examples: List[str],
        negative_examples: List[str],
        layers: Optional[List[int]] = None
    ):
        """Build concept vectors and add them to the tracker."""
        if layers is None:
            # Use middle layers by default
            num_layers = self.model_manager.get_model_info()['num_layers']
            layers = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]
    
        for layer_idx in layers:
            concept_vector = self.concept_builder.build_concept_vector_from_examples(
                concept_name, positive_examples, negative_examples, layer_idx
            )
            self.add_concept_vector(concept_vector)

    def track_abstraction_trajectory(
        self,
        input_text: str,
        token_indices: Optional[List[int]] = None, # Changed to list of indices
        layers: Optional[List[int]] = None
    ) -> List[AbstractionTrajectory]:
        """
        Track how concepts evolve through layers for a given input.
    
        Args:
            input_text: Input text to analyze
            token_indices: Specific token indices to track (default: all tokens)
            layers: Layers to analyze (default: all layers)
        
        Returns:
            List of abstraction trajectories for each concept and each tracked token
        """
        logger.info(f"Tracking abstraction trajectory for: '{input_text[:50]}...'")
    
        if layers is None:
            num_layers = self.model_manager.get_model_info()['layer_count']
            layers = list(range(num_layers))
    
        # Get activations at all layers for all tokens
        layer_activations_all_tokens, tokens_list = self._get_activations_all_layers(input_text, layers)

        if not tokens_list:
            logger.warning(f"No tokens found for input text: {input_text}")
            return []

        if token_indices is None:
            token_indices = list(range(len(tokens_list))) # Track all tokens by default

        trajectories = []
    
        for token_idx in token_indices:
            if token_idx >= len(tokens_list):
                logger.warning(f"Token index {token_idx} out of bounds for text: {input_text}")
                continue

            for concept_name, concept_vectors in self.concept_vectors.items():
                # Create a trajectory for this concept and token
                trajectory = AbstractionTrajectory(
                    concept_name=concept_name,
                    input_text=input_text,
                    token_index=token_idx,
                    layer_similarities=[],
                    layer_activations=[]
                )
            
                # Compute similarities at each layer
                for layer_idx in layers:
                    if layer_idx < len(layer_activations_all_tokens):
                        # Get activation for the specific token at this layer
                        activation = layer_activations_all_tokens[layer_idx][token_idx]
                        trajectory.layer_activations.append(activation)
                    
                        # Find concept vector for this layer
                        concept_vector = self._find_concept_vector_for_layer(concept_vectors, layer_idx)
                    
                        if concept_vector is not None:
                            # Compute cosine similarity
                            similarity = cosine_similarity([activation], [concept_vector.vector])[0, 0]
                            trajectory.layer_similarities.append(similarity)
                        else:
                            trajectory.layer_similarities.append(0.0)
                    else:
                        trajectory.layer_similarities.append(0.0)
                        # Placeholder for missing layers, use the actual hidden size if possible
                        hidden_size = self.model_manager.get_model_info()['hidden_size'] if 'hidden_size' in self.model_manager.get_model_info() else 768
                        trajectory.layer_activations.append(np.zeros(hidden_size))
            
                # Analyze trajectory
                self._analyze_trajectory(trajectory)
                trajectories.append(trajectory)
    
        return trajectories

    def _get_activations_all_layers(
        self,
        input_text: str,
        layers: List[int]
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Get activations at all specified layers and corresponding tokens.
        
        Returns:
            Tuple of (List of activations [seq_len, hidden_size] for each layer, List of tokens)
        """
        activations_cache = {}
        hooks = []
    
        def make_hook(layer_name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                activations_cache[layer_name] = output.detach().cpu().numpy()
            return hook
    
        # Register hooks for all layers
        model = self.model_manager.model
        for layer_idx in layers:
            if hasattr(model, 'transformer'):
                hook_module = model.transformer.h[layer_idx]
            elif hasattr(model, 'encoder'):
                hook_module = model.encoder.layer[layer_idx]
            else:
                continue
        
            hook = hook_module.register_forward_hook(make_hook(f'layer_{layer_idx}'))
            hooks.append(hook)
    
        tokens_list = []
        try:
            # Tokenize and forward pass
            inputs = self.model_manager.model.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
        
            with torch.no_grad():
                _ = model(**inputs)

            # Get tokens from the input
            tokens_list = self.model_manager.model.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
            # Extract activations in order
            layer_activations = []
            for layer_idx in layers:
                layer_name = f'layer_{layer_idx}'
                if layer_name in activations_cache:
                    # Get activations for batch=0: [batch, seq, hidden] -> [seq, hidden]
                    activations = activations_cache[layer_name][0]
                    layer_activations.append(activations)
                else:
                    # Placeholder for missing layers
                    seq_len = inputs['input_ids'].shape[1]
                    model_info = self.model_manager.get_model_info(self.model_manager.model_name)
                    hidden_size = model_info.get('hidden_size', 768)
                    layer_activations.append(np.zeros((seq_len, hidden_size)))
    
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
    
        return layer_activations, tokens_list

    def _find_concept_vector_for_layer(
        self,
        concept_vectors: List[ConceptVector],
        layer_idx: int
    ) -> Optional[ConceptVector]:
        """Find the concept vector for a specific layer."""
        # Exact match first
        for cv in concept_vectors:
            if cv.layer_index == layer_idx:
                return cv
    
        # If no exact match, find the closest layer
        if concept_vectors:
            closest_cv = min(concept_vectors, key=lambda cv: abs(cv.layer_index - layer_idx))
            return closest_cv
    
        return None

    def _analyze_trajectory(self, trajectory: AbstractionTrajectory):
        """Analyze a trajectory to find key properties."""
        similarities = trajectory.layer_similarities
    
        if not similarities:
            return
    
        # Find peak
        trajectory.peak_layer = int(np.argmax(similarities))
        trajectory.peak_similarity = float(max(similarities))
    
        # Find emergence layer (first significant activation)
        emergence_threshold = 0.1
        for i, sim in enumerate(similarities):
            if sim > emergence_threshold:
                trajectory.emergence_layer = i
                break
    
        # Find saturation layer (where change becomes minimal)
        if len(similarities) > 3:
            diffs = np.diff(similarities)
            saturation_threshold = 0.01
        
            for i in range(len(diffs) - 2):
                if all(abs(d) < saturation_threshold for d in diffs[i:i+3]):
                    trajectory.saturation_layer = i + 1
                    break

    def analyze_abstractions(
        self,
        input_texts: List[str],
        token_indices: Optional[List[int]] = None,
        layers: Optional[List[int]] = None
    ) -> AbstractionAnalysisResult:
        """
        Perform complete abstraction analysis.
    
        Args:
            input_texts: List of texts to analyze
            token_indices: Specific token indices to track (default: all tokens)
            layers: Layers to analyze (default: all layers)
        
        Returns:
            Complete abstraction analysis results
        """
        logger.info(f"Analyzing abstractions for {len(input_texts)} texts")
    
        all_trajectories = []
    
        # Track trajectories for all inputs
        for text in tqdm(input_texts, desc="Tracking trajectories"):
            trajectories = self.track_abstraction_trajectory(text, token_indices=token_indices, layers=layers)
            all_trajectories.extend(trajectories)
    
        # Compute global analysis
        layer_abstraction_scores = self._compute_layer_abstraction_scores(all_trajectories, layers)
        concept_emergence_patterns = self._analyze_emergence_patterns(all_trajectories)
    
        # Create result
        result = AbstractionAnalysisResult(
            model_name=self.model_manager.model_name,
            trajectories=all_trajectories,
            concept_vectors=self.concept_vectors,
            layer_abstraction_scores=layer_abstraction_scores,
            concept_emergence_patterns=concept_emergence_patterns
        )
    
        logger.info("Abstraction analysis complete")
        return result

    def _compute_layer_abstraction_scores(
        self,
        trajectories: List[AbstractionTrajectory],
        layers: Optional[List[int]]
    ) -> List[float]:
        """Compute overall abstraction scores for each layer."""
        if layers is None:
            layers = list(range(12))  # Default
    
        layer_scores = []
    
        for layer_idx in layers:
            # Average similarity across all concepts and trajectories at this layer
            similarities = []
        
            for trajectory in trajectories:
                if layer_idx < len(trajectory.layer_similarities):
                    similarities.append(trajectory.layer_similarities[layer_idx])
        
            if similarities:
                layer_scores.append(float(np.mean(similarities)))
            else:
                layer_scores.append(0.0)
    
        return layer_scores

    def _analyze_emergence_patterns(
        self,
        trajectories: List[AbstractionTrajectory]
    ) -> Dict[str, List[int]]:
        """Analyze emergence patterns for each concept."""
        patterns = {}
    
        for trajectory in trajectories:
            concept_name = trajectory.concept_name
            if concept_name not in patterns:
                patterns[concept_name] = []
        
            if trajectory.emergence_layer >= 0:
                patterns[concept_name].append(trajectory.emergence_layer)
    
        return patterns

    def visualize_trajectory(
        self,
        trajectory: AbstractionTrajectory,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize a single abstraction trajectory."""
        fig, ax = plt.subplots(figsize=(10, 6))
    
        layers = list(range(len(trajectory.layer_similarities)))
        similarities = trajectory.layer_similarities
    
        ax.plot(layers, similarities, marker='o', linewidth=2, markersize=6)
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Concept Similarity')
        ax.set_title(f'Abstraction Trajectory: {trajectory.concept_name}')
        ax.grid(True, alpha=0.3)
    
        # Mark key points
        if trajectory.emergence_layer >= 0:
            ax.axvline(trajectory.emergence_layer, color='green', linestyle='--', 
                      label=f'Emergence (Layer {trajectory.emergence_layer})')
    
        if trajectory.peak_layer >= 0:
            ax.axvline(trajectory.peak_layer, color='red', linestyle='--',
                      label=f'Peak (Layer {trajectory.peak_layer})')
    
        if trajectory.saturation_layer >= 0:
            ax.axvline(trajectory.saturation_layer, color='blue', linestyle='--',
                      label=f'Saturation (Layer {trajectory.saturation_layer})')
    
        ax.legend()
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trajectory plot saved: {save_path}")
    
        return fig

    def visualize_all_trajectories(
        self,
        trajectories: List[AbstractionTrajectory],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize all trajectories together."""
        fig, ax = plt.subplots(figsize=(12, 8))
    
        # Group by concept
        concept_trajectories = {}
        for traj in trajectories:
            if traj.concept_name not in concept_trajectories:
                concept_trajectories[traj.concept_name] = []
            concept_trajectories[traj.concept_name].append(traj)
    
        colors = plt.cm.tab10(np.linspace(0, 1, len(concept_trajectories)))
    
        for i, (concept_name, trajs) in enumerate(concept_trajectories.items()):
            # Average trajectories for this concept
            max_len = max(len(t.layer_similarities) for t in trajs)
            avg_similarities = []
        
            for layer_idx in range(max_len):
                layer_sims = [t.layer_similarities[layer_idx] 
                             for t in trajs if layer_idx < len(t.layer_similarities)]
                if layer_sims:
                    avg_similarities.append(np.mean(layer_sims))
                else:
                    avg_similarities.append(0.0)
        
            layers = list(range(len(avg_similarities)))
            ax.plot(layers, avg_similarities, marker='o', linewidth=2, 
                   label=concept_name, color=colors[i])
    
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Average Concept Similarity')
        ax.set_title('Abstraction Trajectories by Concept')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"All trajectories plot saved: {save_path}")
    
        return fig


# Utility functions
def create_basic_concepts(model_manager: ModelManager) -> Dict[str, List[ConceptVector]]:
    """Create basic concept vectors for common concepts."""
    tracker = AbstractionTracker(model_manager)

    # Define basic concepts with examples
    concept_examples = {
        'sentiment_positive': {
            'positive': ["I love this!", "Great job!", "Wonderful experience", "Amazing work"],
            'negative': ["I hate this", "Terrible job", "Awful experience", "Poor work"]
        },
        'sentiment_negative': {
            'positive': ["I hate this", "Terrible job", "Awful experience", "Poor work"],
            'negative': ["I love this!", "Great job!", "Wonderful experience", "Amazing work"]
        },
        'temporal_past': {
            'positive': ["Yesterday I went", "Last week we saw", "Previously he was", "Earlier she said"],
            'negative': ["Tomorrow I will go", "Next week we see", "Later he will be", "Soon she will say"]
        },
        'temporal_future': {
            'positive': ["Tomorrow I will go", "Next week we see", "Later he will be", "Soon she will say"],
            'negative': ["Yesterday I went", "Last week we saw", "Previously he was", "Earlier she said"]
        }
    }

    concepts = {}

    for concept_name, examples in concept_examples.items():
        tracker.build_and_add_concept(
            concept_name,
            examples['positive'],
            examples['negative']
        )

    return tracker.concept_vectors


def analyze_model_abstractions(
    model_manager: ModelManager,
    input_texts: List[str],
    custom_concepts: Optional[Dict[str, Dict[str, List[str]]]] = None,
    token_indices: Optional[List[int]] = None
) -> AbstractionAnalysisResult:
    """Convenience function for complete abstraction analysis."""
    tracker = AbstractionTracker(model_manager)

    # Add basic concepts
    basic_concepts = create_basic_concepts(model_manager)
    for concept_name, vectors in basic_concepts.items():
        for vector in vectors:
            tracker.add_concept_vector(vector)

    # Add custom concepts if provided
    if custom_concepts:
        for concept_name, examples in custom_concepts.items():
            tracker.build_and_add_concept(
                concept_name,
                examples['positive'],
                examples['negative']
            )

    return tracker.analyze_abstractions(input_texts, token_indices=token_indices)
