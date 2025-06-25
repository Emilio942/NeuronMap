"""Advanced interpretability methods for neural network analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from pathlib import Path
import json
import h5py
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings

try:
    from ..utils.config import get_config
    from ..utils.error_handling import with_retry, safe_execute
    from ..utils.monitoring import check_gpu_memory
except ImportError:
    # Fallback for development
    get_config = lambda: type('Config', (), {'get_experiment_config': lambda x: {}})()
    handle_errors = lambda func: func
    check_gpu_memory = lambda: {'available_mb': 1000}


logger = logging.getLogger(__name__)


@dataclass
class ConceptActivationVector:
    """Represents a Concept Activation Vector (CAV)."""
    concept_name: str
    layer_name: str
    vector: np.ndarray
    accuracy: float
    importance_scores: np.ndarray
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'concept_name': self.concept_name,
            'layer_name': self.layer_name,
            'vector': self.vector.tolist(),
            'accuracy': float(self.accuracy),
            'importance_scores': self.importance_scores.tolist(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConceptActivationVector':
        """Create from dictionary."""
        return cls(
            concept_name=data['concept_name'],
            layer_name=data['layer_name'],
            vector=np.array(data['vector']),
            accuracy=data['accuracy'],
            importance_scores=np.array(data['importance_scores']),
            metadata=data['metadata']
        )


@dataclass
class SaliencyResult:
    """Results from saliency analysis."""
    input_tokens: List[str]
    saliency_scores: np.ndarray
    layer_name: str
    method: str
    target_neuron: Optional[int]
    metadata: Dict[str, Any]


class ConceptActivationVectorAnalyzer:
    """Implement Concept Activation Vector (CAV) analysis."""

    def __init__(self, config_name: str = "default"):
        """Initialize CAV analyzer.

        Args:
            config_name: Name of experiment configuration.
        """
        try:
            self.config = get_config()
            self.experiment_config = self.config.get_experiment_config(config_name)
        except:
            self.config = None
            self.experiment_config = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def load_model(self, model_name: str = None):
        """Load model for CAV analysis.

        Args:
            model_name: Name of model to load.
        """
        if model_name is None:
            model_name = self.experiment_config.get('model_name', 'gpt2')

        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Loaded model: {model_name}")

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def extract_activations_for_concepts(self, concept_examples: Dict[str, List[str]],
                                       layer_name: str,
                                       max_length: int = 512) -> Dict[str, np.ndarray]:
        """Extract activations for concept examples.

        Args:
            concept_examples: Dictionary mapping concept names to example texts.
            layer_name: Name of layer to extract from.
            max_length: Maximum sequence length.

        Returns:
            Dictionary mapping concept names to activation arrays.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        concept_activations = {}

        for concept_name, examples in concept_examples.items():
            logger.info(f"Extracting activations for concept: {concept_name}")
            activations = []

            for example in examples:
                try:
                    # Tokenize
                    inputs = self.tokenizer(
                        example,
                        return_tensors="pt",
                        max_length=max_length,
                        truncation=True,
                        padding=True
                    ).to(self.device)

                    # Extract activations
                    with torch.no_grad():
                        outputs = self.model(**inputs, output_hidden_states=True)

                        # Find layer index
                        layer_idx = self._get_layer_index(layer_name)
                        if layer_idx is None:
                            raise ValueError(f"Layer {layer_name} not found")

                        # Get layer activations (mean pooled)
                        layer_activations = outputs.hidden_states[layer_idx]
                        pooled_activations = layer_activations.mean(dim=1).squeeze()

                        activations.append(pooled_activations.cpu().numpy())

                except Exception as e:
                    logger.warning(f"Failed to process example '{example[:50]}...': {e}")
                    continue

            if activations:
                concept_activations[concept_name] = np.stack(activations)
                logger.info(f"Extracted {len(activations)} activations for {concept_name}")
            else:
                logger.warning(f"No activations extracted for concept: {concept_name}")

        return concept_activations

    def train_concept_vector(self, concept_activations: Dict[str, np.ndarray],
                           target_concept: str,
                           negative_concepts: List[str] = None) -> ConceptActivationVector:
        """Train a Concept Activation Vector.

        Args:
            concept_activations: Dictionary of concept activations.
            target_concept: Name of target concept.
            negative_concepts: List of negative concept names (default: all others).

        Returns:
            Trained ConceptActivationVector.
        """
        if target_concept not in concept_activations:
            raise ValueError(f"Target concept '{target_concept}' not found in activations")

        # Prepare positive examples
        positive_activations = concept_activations[target_concept]

        # Prepare negative examples
        if negative_concepts is None:
            negative_concepts = [name for name in concept_activations.keys()
                               if name != target_concept]

        negative_activations = []
        for neg_concept in negative_concepts:
            if neg_concept in concept_activations:
                negative_activations.append(concept_activations[neg_concept])

        if not negative_activations:
            raise ValueError("No negative examples available")

        negative_activations = np.vstack(negative_activations)

        # Prepare training data
        X = np.vstack([positive_activations, negative_activations])
        y = np.hstack([
            np.ones(len(positive_activations)),
            np.zeros(len(negative_activations))
        ])

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train linear classifier
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(X_scaled, y)

        # Get predictions and accuracy
        y_pred = classifier.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)

        # Extract CAV (normalized weight vector)
        cav_vector = classifier.coef_[0]
        cav_vector = cav_vector / np.linalg.norm(cav_vector)

        # Calculate feature importance
        importance_scores = np.abs(cav_vector)

        logger.info(f"Trained CAV for '{target_concept}' with accuracy: {accuracy:.3f}")

        return ConceptActivationVector(
            concept_name=target_concept,
            layer_name="unknown",  # Will be set by caller
            vector=cav_vector,
            accuracy=accuracy,
            importance_scores=importance_scores,
            metadata={
                'negative_concepts': negative_concepts,
                'n_positive': len(positive_activations),
                'n_negative': len(negative_activations),
                'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist()
            }
        )

    def calculate_tcav_score(self, cav: ConceptActivationVector,
                           test_activations: np.ndarray,
                           test_gradients: np.ndarray = None) -> float:
        """Calculate Testing with Concept Activation Vectors (TCAV) score.

        Args:
            cav: Trained Concept Activation Vector.
            test_activations: Test activations to analyze.
            test_gradients: Gradients w.r.t. activations (optional).

        Returns:
            TCAV score (proportion of positive directional derivatives).
        """
        if test_gradients is None:
            # If no gradients provided, use unit gradients (simplified TCAV)
            test_gradients = np.ones_like(test_activations)

        # Calculate directional derivatives
        directional_derivatives = np.sum(test_gradients * cav.vector, axis=1)

        # TCAV score is proportion of positive derivatives
        tcav_score = np.mean(directional_derivatives > 0)

        return float(tcav_score)

    def _get_layer_index(self, layer_name: str) -> Optional[int]:
        """Get layer index from layer name.

        Args:
            layer_name: Name of layer.

        Returns:
            Layer index or None if not found.
        """
        if hasattr(self.model, 'config'):
            num_layers = getattr(self.model.config, 'num_hidden_layers', 12)

            # Try to parse layer index from name
            if 'layer' in layer_name or 'block' in layer_name:
                import re
                match = re.search(r'(\d+)', layer_name)
                if match:
                    layer_idx = int(match.group(1))
                    if 0 <= layer_idx < num_layers:
                        return layer_idx + 1  # +1 for embedding layer

        return None


class SaliencyAnalyzer:
    """Implement various saliency analysis methods."""

    def __init__(self, config_name: str = "default"):
        """Initialize saliency analyzer.

        Args:
            config_name: Name of experiment configuration.
        """
        try:
            self.config = get_config()
            self.experiment_config = self.config.get_experiment_config(config_name)
        except:
            self.config = None
            self.experiment_config = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def load_model(self, model_name: str = None):
        """Load model for saliency analysis.

        Args:
            model_name: Name of model to load.
        """
        if model_name is None:
            model_name = self.experiment_config.get('model_name', 'gpt2')

        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Loaded model for saliency analysis: {model_name}")

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def compute_gradient_saliency(self, text: str, target_layer: str,
                                target_neuron: int = None) -> SaliencyResult:
        """Compute gradient-based saliency.

        Args:
            text: Input text to analyze.
            target_layer: Layer to compute saliency for.
            target_neuron: Specific neuron to target (optional).

        Returns:
            SaliencyResult with gradient saliency scores.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        input_ids = inputs['input_ids']

        # Get tokens for interpretation
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Enable gradients for embeddings
        embeddings = self.model.get_input_embeddings()(input_ids)
        embeddings.requires_grad_(True)

        # Forward pass
        outputs = self.model(inputs_embeds=embeddings, output_hidden_states=True)

        # Get target layer activations
        layer_idx = self._get_layer_index(target_layer)
        if layer_idx is None:
            raise ValueError(f"Layer {target_layer} not found")

        target_activations = outputs.hidden_states[layer_idx]

        # Choose target (specific neuron or mean activation)
        if target_neuron is not None:
            target = target_activations[0, :, target_neuron].mean()
        else:
            target = target_activations.mean()

        # Compute gradients
        target.backward()

        # Get gradient magnitudes
        gradients = embeddings.grad
        saliency_scores = gradients.norm(dim=-1).squeeze().detach().cpu().numpy()

        logger.info(f"Computed gradient saliency for '{text[:50]}...'")

        return SaliencyResult(
            input_tokens=tokens,
            saliency_scores=saliency_scores,
            layer_name=target_layer,
            method="gradient",
            target_neuron=target_neuron,
            metadata={'text': text}
        )

    def compute_integrated_gradients(self, text: str, target_layer: str,
                                   target_neuron: int = None,
                                   steps: int = 50) -> SaliencyResult:
        """Compute Integrated Gradients saliency.

        Args:
            text: Input text to analyze.
            target_layer: Layer to compute saliency for.
            target_neuron: Specific neuron to target (optional).
            steps: Number of integration steps.

        Returns:
            SaliencyResult with integrated gradients scores.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        input_ids = inputs['input_ids']
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Get baseline (all pad tokens)
        baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)

        # Get embeddings
        input_embeddings = self.model.get_input_embeddings()(input_ids)
        baseline_embeddings = self.model.get_input_embeddings()(baseline_ids)

        # Compute integrated gradients
        integrated_gradients = torch.zeros_like(input_embeddings)

        for step in range(steps):
            # Interpolate between baseline and input
            alpha = step / steps
            interpolated = baseline_embeddings + alpha * (input_embeddings - baseline_embeddings)
            interpolated.requires_grad_(True)

            # Forward pass
            outputs = self.model(inputs_embeds=interpolated, output_hidden_states=True)

            # Get target
            layer_idx = self._get_layer_index(target_layer)
            target_activations = outputs.hidden_states[layer_idx]

            if target_neuron is not None:
                target = target_activations[0, :, target_neuron].mean()
            else:
                target = target_activations.mean()

            # Backward pass
            target.backward()

            # Accumulate gradients
            integrated_gradients += interpolated.grad / steps

        # Final integrated gradients
        final_gradients = integrated_gradients * (input_embeddings - baseline_embeddings)
        saliency_scores = final_gradients.norm(dim=-1).squeeze().detach().cpu().numpy()

        logger.info(f"Computed integrated gradients for '{text[:50]}...'")

        return SaliencyResult(
            input_tokens=tokens,
            saliency_scores=saliency_scores,
            layer_name=target_layer,
            method="integrated_gradients",
            target_neuron=target_neuron,
            metadata={'text': text, 'steps': steps}
        )

    def _get_layer_index(self, layer_name: str) -> Optional[int]:
        """Get layer index from layer name."""
        if hasattr(self.model, 'config'):
            num_layers = getattr(self.model.config, 'num_hidden_layers', 12)

            # Try to parse layer index from name
            if 'layer' in layer_name or 'block' in layer_name:
                import re
                match = re.search(r'(\d+)', layer_name)
                if match:
                    layer_idx = int(match.group(1))
                    if 0 <= layer_idx < num_layers:
                        return layer_idx + 1  # +1 for embedding layer

        return None


class ActivationMaximizationAnalyzer:
    """Implement activation maximization for feature visualization."""

    def __init__(self, config_name: str = "default"):
        """Initialize activation maximization analyzer.

        Args:
            config_name: Name of experiment configuration.
        """
        try:
            self.config = get_config()
            self.experiment_config = self.config.get_experiment_config(config_name)
        except:
            self.config = None
            self.experiment_config = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def load_model(self, model_name: str = None):
        """Load model for activation maximization.

        Args:
            model_name: Name of model to load.
        """
        if model_name is None:
            model_name = self.experiment_config.get('model_name', 'gpt2')

        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Loaded model for activation maximization: {model_name}")

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def find_maximizing_inputs(self, target_layer: str, target_neuron: int,
                             candidate_texts: List[str],
                             top_k: int = 10) -> List[Tuple[str, float]]:
        """Find inputs that maximally activate a target neuron.

        Args:
            target_layer: Layer containing target neuron.
            target_neuron: Index of target neuron.
            candidate_texts: List of candidate input texts.
            top_k: Number of top activating inputs to return.

        Returns:
            List of (text, activation_score) tuples, sorted by score.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        activations_scores = []

        for text in candidate_texts:
            try:
                # Tokenize
                inputs = self.tokenizer(
                    text, return_tensors="pt",
                    max_length=512, truncation=True, padding=True
                ).to(self.device)

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)

                    # Get target layer activations
                    layer_idx = self._get_layer_index(target_layer)
                    if layer_idx is None:
                        continue

                    layer_activations = outputs.hidden_states[layer_idx]
                    neuron_activation = layer_activations[0, :, target_neuron].mean().item()

                    activations_scores.append((text, neuron_activation))

            except Exception as e:
                logger.warning(f"Failed to process text '{text[:50]}...': {e}")
                continue

        # Sort by activation score and return top-k
        activations_scores.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Found {len(activations_scores)} valid activations for neuron {target_neuron}")

        return activations_scores[:top_k]

    def _get_layer_index(self, layer_name: str) -> Optional[int]:
        """Get layer index from layer name."""
        if hasattr(self.model, 'config'):
            num_layers = getattr(self.model.config, 'num_hidden_layers', 12)

            if 'layer' in layer_name or 'block' in layer_name:
                import re
                match = re.search(r'(\d+)', layer_name)
                if match:
                    layer_idx = int(match.group(1))
                    if 0 <= layer_idx < num_layers:
                        return layer_idx + 1

        return None


class InterpretabilityPipeline:
    """Complete interpretability analysis pipeline."""

    def __init__(self, config_name: str = "default"):
        """Initialize interpretability pipeline.

        Args:
            config_name: Name of experiment configuration.
        """
        self.config_name = config_name
        self.cav_analyzer = ConceptActivationVectorAnalyzer(config_name)
        self.saliency_analyzer = SaliencyAnalyzer(config_name)
        self.activation_maximizer = ActivationMaximizationAnalyzer(config_name)

    def run_full_interpretability_analysis(self,
                                         model_name: str,
                                         concept_examples: Dict[str, List[str]],
                                         test_texts: List[str],
                                         target_layer: str,
                                         output_dir: str = "data/outputs/interpretability") -> Dict[str, Any]:
        """Run complete interpretability analysis.

        Args:
            model_name: Name of model to analyze.
            concept_examples: Dictionary of concept examples for CAV training.
            test_texts: List of texts for saliency and activation analysis.
            target_layer: Layer to analyze.
            output_dir: Output directory for results.

        Returns:
            Dictionary with analysis results.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {
            'model_name': model_name,
            'target_layer': target_layer,
            'cavs': {},
            'saliency_results': [],
            'activation_maximization': {},
            'metadata': {}
        }

        # Load models
        logger.info("Loading models for interpretability analysis...")
        self.cav_analyzer.load_model(model_name)
        self.saliency_analyzer.load_model(model_name)
        self.activation_maximizer.load_model(model_name)

        # 1. CAV Analysis
        logger.info("Performing CAV analysis...")
        concept_activations = self.cav_analyzer.extract_activations_for_concepts(
            concept_examples, target_layer
        )

        for concept_name in concept_examples.keys():
            if concept_name in concept_activations:
                cav = self.cav_analyzer.train_concept_vector(
                    concept_activations, concept_name
                )
                cav.layer_name = target_layer
                results['cavs'][concept_name] = cav.to_dict()

        # 2. Saliency Analysis
        logger.info("Performing saliency analysis...")
        for i, text in enumerate(test_texts[:10]):  # Limit for performance
            # Gradient saliency
            grad_saliency = self.saliency_analyzer.compute_gradient_saliency(
                text, target_layer
            )
            results['saliency_results'].append({
                'text_index': i,
                'method': 'gradient',
                'tokens': grad_saliency.input_tokens,
                'scores': grad_saliency.saliency_scores.tolist()
            })

            # Integrated gradients (for first few texts due to computational cost)
            if i < 3:
                ig_saliency = self.saliency_analyzer.compute_integrated_gradients(
                    text, target_layer, steps=20
                )
                results['saliency_results'].append({
                    'text_index': i,
                    'method': 'integrated_gradients',
                    'tokens': ig_saliency.input_tokens,
                    'scores': ig_saliency.saliency_scores.tolist()
                })

        # 3. Activation Maximization
        logger.info("Performing activation maximization...")
        # Find top activating inputs for first few neurons
        for neuron_idx in range(min(5, 100)):  # Analyze first 5 neurons
            top_activating = self.activation_maximizer.find_maximizing_inputs(
                target_layer, neuron_idx, test_texts, top_k=5
            )
            results['activation_maximization'][f'neuron_{neuron_idx}'] = [
                {'text': text, 'activation': float(score)}
                for text, score in top_activating
            ]

        # Save results
        results_file = output_path / f"interpretability_results_{target_layer.replace('.', '_')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Interpretability analysis completed. Results saved to {results_file}")

        return results


class InterpretabilityAnalyzer:
    """Advanced interpretability analysis for neural network activations."""

    def __init__(self, config_name: str = "default"):
        """Initialize interpretability analyzer.

        Args:
            config_name: Configuration name to use.
        """
        self.config_name = config_name
        self.logger = logging.getLogger(__name__)

    def compute_concept_activation_vectors(self, activations: np.ndarray,
                                         labels: np.ndarray,
                                         concept_name: str = "concept") -> Dict[str, Any]:
        """Compute Concept Activation Vectors (CAVs) for interpretability.

        Args:
            activations: Activation data matrix [samples, features].
            labels: Binary labels for concept classification.
            concept_name: Name of the concept being analyzed.

        Returns:
            Dictionary containing CAV results.
        """
        try:
            # Train linear classifier to separate concept
            scaler = StandardScaler()
            scaled_activations = scaler.fit_transform(activations)

            classifier = LogisticRegression(random_state=42, max_iter=1000)
            classifier.fit(scaled_activations, labels)

            # Get predictions and accuracy
            predictions = classifier.predict(scaled_activations)
            accuracy = accuracy_score(labels, predictions)

            # CAV vector is the normal to the decision boundary
            cav_vector = classifier.coef_[0]

            # Compute feature importance
            feature_importance = np.abs(cav_vector)
            feature_rankings = np.argsort(feature_importance)[::-1]

            return {
                'cav_vector': cav_vector.tolist(),
                'accuracy': float(accuracy),
                'feature_importance': feature_importance.tolist(),
                'feature_rankings': feature_rankings.tolist(),
                'concept_name': concept_name,
                'n_samples': len(activations),
                'n_features': activations.shape[1]
            }

        except Exception as e:
            self.logger.error(f"CAV computation failed: {e}")
            raise

    def compute_saliency_maps(self, activations: np.ndarray,
                            gradients: np.ndarray,
                            method: str = "gradient") -> Dict[str, Any]:
        """Compute saliency maps for activation interpretability.

        Args:
            activations: Activation data matrix [samples, features].
            gradients: Gradient data matrix [samples, features].
            method: Saliency computation method ('gradient', 'integrated', 'smooth').

        Returns:
            Dictionary containing saliency results.
        """
        try:
            if method == "gradient":
                # Simple gradient-based saliency
                saliency_scores = np.abs(gradients)
            elif method == "integrated":
                # Simplified integrated gradients
                saliency_scores = np.abs(gradients * activations)
            elif method == "smooth":
                # Smooth gradients approximation
                noise_std = 0.1
                n_samples = 10
                smooth_gradients = []

                for _ in range(n_samples):
                    noise = np.random.normal(0, noise_std, activations.shape)
                    noisy_activations = activations + noise
                    # In practice, would recompute gradients with noisy inputs
                    smooth_gradients.append(gradients)

                saliency_scores = np.abs(np.mean(smooth_gradients, axis=0))
            else:
                raise ValueError(f"Unsupported saliency method: {method}")

            # Get top features per sample
            top_features = []
            for i in range(saliency_scores.shape[0]):
                top_indices = np.argsort(saliency_scores[i])[-10:][::-1]
                top_features.append({
                    'sample_idx': i,
                    'top_feature_indices': top_indices.tolist(),
                    'top_saliency_scores': saliency_scores[i][top_indices].tolist()
                })

            return {
                'saliency_scores': saliency_scores,
                'top_features': top_features,
                'method': method,
                'mean_saliency': float(np.mean(saliency_scores)),
                'max_saliency': float(np.max(saliency_scores))
            }

        except Exception as e:
            self.logger.error(f"Saliency computation failed: {e}")
            raise

    def analyze_feature_importance(self, activations: np.ndarray,
                                 labels: np.ndarray,
                                 method: str = "permutation") -> Dict[str, Any]:
        """Analyze feature importance for activation patterns.

        Args:
            activations: Activation data matrix [samples, features].
            labels: Target labels for supervised importance analysis.
            method: Importance computation method ('permutation', 'correlation', 'mutual_info').

        Returns:
            Dictionary containing feature importance results.
        """
        try:
            n_features = activations.shape[1]

            if method == "permutation":
                # Simplified permutation importance
                baseline_score = self._compute_baseline_score(activations, labels)
                feature_importance = np.zeros(n_features)

                for feature_idx in range(min(n_features, 100)):  # Limit for performance
                    # Permute feature and measure score drop
                    permuted_activations = activations.copy()
                    np.random.shuffle(permuted_activations[:, feature_idx])

                    permuted_score = self._compute_baseline_score(permuted_activations, labels)
                    feature_importance[feature_idx] = baseline_score - permuted_score

            elif method == "correlation":
                # Correlation-based importance
                feature_importance = np.abs([
                    np.corrcoef(activations[:, i], labels)[0, 1]
                    if not np.isnan(np.corrcoef(activations[:, i], labels)[0, 1]) else 0.0
                    for i in range(n_features)
                ])

            elif method == "mutual_info":
                # Simplified mutual information
                from sklearn.feature_selection import mutual_info_classif
                feature_importance = mutual_info_classif(activations, labels)

            else:
                raise ValueError(f"Unsupported importance method: {method}")

            # Get top features
            feature_rankings = np.argsort(feature_importance)[::-1]
            top_features = [
                {
                    'feature_idx': int(idx),
                    'importance_score': float(feature_importance[idx])
                }
                for idx in feature_rankings[:20]
            ]

            return {
                'feature_importance': feature_importance.tolist(),
                'feature_rankings': feature_rankings.tolist(),
                'top_features': top_features,
                'method': method,
                'mean_importance': float(np.mean(feature_importance)),
                'std_importance': float(np.std(feature_importance))
            }

        except Exception as e:
            self.logger.error(f"Feature importance analysis failed: {e}")
            raise

    def activation_maximization(self, activation_function: Callable,
                              input_shape: Tuple[int, ...],
                              target_neuron: int,
                              iterations: int = 100,
                              learning_rate: float = 0.01) -> Dict[str, Any]:
        """Perform activation maximization to understand neuron preferences.

        Args:
            activation_function: Function that computes activations given input.
            input_shape: Shape of the input space.
            target_neuron: Index of neuron to maximize.
            iterations: Number of optimization iterations.
            learning_rate: Learning rate for optimization.

        Returns:
            Dictionary containing optimization results.
        """
        try:
            # Initialize random input
            optimized_input = np.random.randn(*input_shape) * 0.1
            activation_trajectory = []

            for iteration in range(iterations):
                # Compute current activation
                current_activation = activation_function(optimized_input[np.newaxis, :])[0]
                activation_trajectory.append(float(current_activation))

                # Compute numerical gradient
                epsilon = 1e-5
                gradients = np.zeros_like(optimized_input)

                for i in range(min(len(optimized_input), 100)):  # Limit for performance
                    # Positive perturbation
                    perturbed_input = optimized_input.copy()
                    perturbed_input[i] += epsilon
                    pos_activation = activation_function(perturbed_input[np.newaxis, :])[0]

                    # Negative perturbation
                    perturbed_input[i] -= 2 * epsilon
                    neg_activation = activation_function(perturbed_input[np.newaxis, :])[0]

                    # Numerical gradient
                    gradients[i] = (pos_activation - neg_activation) / (2 * epsilon)

                # Update input using gradient ascent
                optimized_input += learning_rate * gradients

                # Optional: Add regularization
                optimized_input *= 0.999  # L2 regularization

            final_activation = activation_function(optimized_input[np.newaxis, :])[0]

            return {
                'optimized_input': optimized_input.tolist(),
                'activation_trajectory': activation_trajectory,
                'final_activation': float(final_activation),
                'iterations': iterations,
                'target_neuron': target_neuron,
                'improvement': float(final_activation - activation_trajectory[0])
            }

        except Exception as e:
            self.logger.error(f"Activation maximization failed: {e}")
            raise

    def _compute_baseline_score(self, activations: np.ndarray, labels: np.ndarray) -> float:
        """Compute baseline classification score."""
        try:
            # Simple logistic regression score
            classifier = LogisticRegression(random_state=42, max_iter=1000)
            classifier.fit(activations, labels)
            return classifier.score(activations, labels)
        except Exception:
            # Fallback to correlation if classification fails
            return np.abs(np.corrcoef(np.mean(activations, axis=1), labels)[0, 1])
