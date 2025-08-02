"""
Automated Concept Extraction (ACE) for NeuronMap
===============================================

Automated discovery and naming of semantic concepts in neural network
representations using TF-IDF and CNN kernel pooling approaches.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from pathlib import Path
import time
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import scipy.stats as stats

from ...core.plugin_interface import ConceptPluginBase, ToolExecutionResult

logger = logging.getLogger(__name__)

@dataclass
class ConceptInfo:
    """Information about an extracted concept."""
    concept_id: str
    name: str
    description: str
    activation_pattern: np.ndarray
    examples: List[Any]
    importance_score: float
    coherence_score: float
    cluster_size: int

class ACEConceptExtractor(ConceptPluginBase):
    """
    Automated Concept Extraction using TF-IDF and CNN kernel analysis.
    
    Implements concept discovery through clustering of activation patterns
    and automatic concept naming based on input characteristics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(tool_id="ace_concepts", config=config)
        
        self.version = "1.0.0"
        self.description = "Automated Concept Extraction with TF-IDF and CNN kernel pooling for concept isolation"
        
        # Configuration parameters
        self.n_concepts = config.get('n_concepts', 10) if config else 10
        self.min_cluster_size = config.get('min_cluster_size', 5) if config else 5
        self.clustering_method = config.get('clustering_method', 'kmeans') if config else 'kmeans'
        self.concept_threshold = config.get('concept_threshold', 0.1) if config else 0.1
        self.use_pca = config.get('use_pca', True) if config else True
        self.pca_components = config.get('pca_components', 50) if config else 50
        
        # Internal components
        self.clusterer = None
        self.pca_reducer = None
        self.tfidf_vectorizer = None
        
        logger.info(f"Initialized ACE concept extractor (method: {self.clustering_method})")
    
    def initialize(self) -> bool:
        """Initialize the ACE concept extractor."""
        try:
            # Initialize clustering algorithm
            if self.clustering_method == 'kmeans':
                self.clusterer = KMeans(n_clusters=self.n_concepts, random_state=42)
            elif self.clustering_method == 'dbscan':
                self.clusterer = DBSCAN(eps=0.5, min_samples=self.min_cluster_size)
            else:
                raise ValueError(f"Unknown clustering method: {self.clustering_method}")
            
            # Initialize PCA if requested
            if self.use_pca:
                self.pca_reducer = PCA(n_components=self.pca_components, random_state=42)
            
            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ACE extractor: {e}")
            return False
    
    def execute(self, model: nn.Module, dataset: Any,
                layer_name: Optional[str] = None,
                input_texts: Optional[List[str]] = None,
                **kwargs) -> ToolExecutionResult:
        """
        Execute ACE concept extraction.
        
        Args:
            model: PyTorch model to analyze
            dataset: Dataset to extract concepts from
            layer_name: Specific layer to analyze
            input_texts: Text inputs corresponding to dataset (for concept naming)
            
        Returns:
            ToolExecutionResult with extracted concepts and metadata
        """
        start_time = time.time()
        
        try:
            # Extract activations from the model
            activations, input_examples = self._extract_activations(
                model, dataset, layer_name
            )
            
            # Use provided texts or generate descriptions
            if input_texts is None:
                input_texts = self._generate_input_descriptions(input_examples)
            
            # Perform concept extraction
            concepts = self._extract_concepts(activations, input_texts, input_examples)
            
            # Compute concept metrics
            concept_metrics = self._compute_concept_metrics(concepts, activations)
            
            # Generate concept importance ranking
            concept_ranking = self._rank_concepts(concepts)
            
            # Prepare outputs
            outputs = {
                'concepts': {c.concept_id: self._serialize_concept(c) for c in concepts},
                'concept_scores': {c.concept_id: c.importance_score for c in concepts},
                'concept_examples': {c.concept_id: c.examples[:5] for c in concepts},  # Top 5 examples
                'concept_metrics': concept_metrics,
                'concept_ranking': concept_ranking,
                'extraction_metadata': {
                    'n_concepts_extracted': len(concepts),
                    'layer_analyzed': layer_name,
                    'clustering_method': self.clustering_method,
                    'total_activations': activations.shape[0]
                }
            }
            
            execution_time = time.time() - start_time
            
            return ToolExecutionResult(
                tool_id=self.tool_id,
                success=True,
                execution_time=execution_time,
                outputs=outputs,
                metadata=self.get_metadata(),
                errors=[],
                warnings=[],
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            
        except Exception as e:
            logger.error(f"ACE execution failed: {e}")
            execution_time = time.time() - start_time
            
            return ToolExecutionResult(
                tool_id=self.tool_id,
                success=False,
                execution_time=execution_time,
                outputs={},
                metadata=self.get_metadata(),
                errors=[str(e)],
                warnings=[],
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
            )
    
    def _extract_activations(self, model: nn.Module, dataset: Any,
                            layer_name: Optional[str] = None) -> Tuple[np.ndarray, List[Any]]:
        """Extract activations from the specified layer."""
        model.eval()
        activations_list = []
        input_examples = []
        
        # Register hook to capture activations
        activation_hook = None
        captured_activations = []
        
        def hook_fn(module, input, output):
            captured_activations.append(output.detach().cpu())
        
        # Find target layer and register hook
        if layer_name:
            target_layer = self._find_layer_by_name(model, layer_name)
            if target_layer is None:
                raise ValueError(f"Layer {layer_name} not found in model")
            activation_hook = target_layer.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                for i, batch in enumerate(dataset):
                    if i >= 100:  # Limit to 100 batches for efficiency
                        break
                    
                    # Handle different batch formats
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0]
                        input_examples.extend(batch[0] if len(batch) > 0 else [])
                    else:
                        inputs = batch
                        input_examples.append(batch)
                    
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Collect activations
                    if layer_name and captured_activations:
                        batch_activations = captured_activations[-1]
                        # Flatten spatial dimensions but keep batch dimension
                        if batch_activations.dim() > 2:
                            batch_activations = batch_activations.flatten(start_dim=1)
                        activations_list.append(batch_activations.numpy())
                    else:
                        # Use final layer output if no specific layer
                        if outputs.dim() > 2:
                            outputs = outputs.flatten(start_dim=1)
                        activations_list.append(outputs.cpu().numpy())
            
        finally:
            if activation_hook:
                activation_hook.remove()
        
        # Concatenate all activations
        if activations_list:
            activations = np.concatenate(activations_list, axis=0)
        else:
            raise ValueError("No activations were extracted")
        
        return activations, input_examples
    
    def _find_layer_by_name(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Find layer by name in the model."""
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        return None
    
    def _generate_input_descriptions(self, input_examples: List[Any]) -> List[str]:
        """Generate text descriptions for input examples."""
        descriptions = []
        
        for example in input_examples:
            if isinstance(example, str):
                descriptions.append(example)
            elif isinstance(example, torch.Tensor):
                # Generate basic description for tensor
                shape_desc = f"tensor_shape_{example.shape}"
                descriptions.append(shape_desc)
            else:
                descriptions.append(str(type(example).__name__))
        
        return descriptions
    
    def _extract_concepts(self, activations: np.ndarray, input_texts: List[str],
                         input_examples: List[Any]) -> List[ConceptInfo]:
        """Extract concepts from activations using clustering and TF-IDF."""
        
        # Reduce dimensionality if requested
        if self.use_pca and activations.shape[1] > self.pca_components:
            logger.info(f"Reducing dimensionality from {activations.shape[1]} to {self.pca_components}")
            activations_reduced = self.pca_reducer.fit_transform(activations)
        else:
            activations_reduced = activations
        
        # Perform clustering
        cluster_labels = self.clusterer.fit_predict(activations_reduced)
        
        # Handle DBSCAN noise points
        if self.clustering_method == 'dbscan':
            unique_labels = set(cluster_labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)  # Remove noise points
            n_clusters = len(unique_labels)
        else:
            n_clusters = self.n_concepts
        
        logger.info(f"Found {n_clusters} clusters")
        
        # Extract concepts for each cluster
        concepts = []
        for cluster_id in range(n_clusters):
            if self.clustering_method == 'dbscan':
                cluster_mask = cluster_labels == list(unique_labels)[cluster_id]
            else:
                cluster_mask = cluster_labels == cluster_id
            
            if np.sum(cluster_mask) < self.min_cluster_size:
                continue
            
            concept = self._create_concept_from_cluster(
                cluster_id, cluster_mask, activations, input_texts, input_examples
            )
            concepts.append(concept)
        
        return concepts
    
    def _create_concept_from_cluster(self, cluster_id: int, cluster_mask: np.ndarray,
                                    activations: np.ndarray, input_texts: List[str],
                                    input_examples: List[Any]) -> ConceptInfo:
        """Create a concept from a cluster of activations."""
        
        # Extract cluster activations
        cluster_activations = activations[cluster_mask]
        cluster_texts = [input_texts[i] for i in np.where(cluster_mask)[0] if i < len(input_texts)]
        cluster_examples = [input_examples[i] for i in np.where(cluster_mask)[0] if i < len(input_examples)]
        
        # Compute representative activation pattern
        activation_pattern = np.mean(cluster_activations, axis=0)
        
        # Generate concept name using TF-IDF
        concept_name = self._generate_concept_name(cluster_texts, cluster_id)
        
        # Compute concept scores
        importance_score = self._compute_importance_score(cluster_activations, activation_pattern)
        coherence_score = self._compute_coherence_score(cluster_activations)
        
        # Generate description
        description = self._generate_concept_description(
            concept_name, cluster_texts, importance_score, coherence_score
        )
        
        return ConceptInfo(
            concept_id=f"concept_{cluster_id}",
            name=concept_name,
            description=description,
            activation_pattern=activation_pattern,
            examples=cluster_examples[:10],  # Keep top 10 examples
            importance_score=importance_score,
            coherence_score=coherence_score,
            cluster_size=len(cluster_activations)
        )
    
    def _generate_concept_name(self, cluster_texts: List[str], cluster_id: int) -> str:
        """Generate concept name using TF-IDF analysis."""
        if not cluster_texts:
            return f"concept_{cluster_id}"
        
        try:
            # Fit TF-IDF on cluster texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(cluster_texts)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_tfidf = tfidf_matrix.mean(axis=0).A1
            
            # Get top terms
            top_indices = mean_tfidf.argsort()[-3:][::-1]  # Top 3 terms
            top_terms = [feature_names[i] for i in top_indices if mean_tfidf[i] > 0.01]
            
            if top_terms:
                # Create concept name from top terms
                concept_name = "_".join(top_terms[:2])  # Use top 2 terms
                return concept_name.replace(" ", "_")
            else:
                return f"concept_{cluster_id}"
                
        except Exception as e:
            logger.warning(f"TF-IDF naming failed: {e}")
            return f"concept_{cluster_id}"
    
    def _compute_importance_score(self, cluster_activations: np.ndarray,
                                 activation_pattern: np.ndarray) -> float:
        """Compute importance score for the concept."""
        # Based on activation magnitude and consistency
        magnitude_score = np.linalg.norm(activation_pattern)
        consistency_score = 1.0 - np.std(cluster_activations, axis=0).mean()
        
        # Combine scores
        importance = 0.7 * magnitude_score + 0.3 * max(0, consistency_score)
        return float(importance)
    
    def _compute_coherence_score(self, cluster_activations: np.ndarray) -> float:
        """Compute coherence score for the concept cluster."""
        # Based on intra-cluster similarity
        if len(cluster_activations) < 2:
            return 1.0
        
        # Compute pairwise cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(cluster_activations)
        
        # Average similarity (excluding diagonal)
        n = similarities.shape[0]
        total_similarity = similarities.sum() - n  # Subtract diagonal
        avg_similarity = total_similarity / (n * (n - 1)) if n > 1 else 1.0
        
        return float(max(0, avg_similarity))
    
    def _generate_concept_description(self, concept_name: str, cluster_texts: List[str],
                                     importance_score: float, coherence_score: float) -> str:
        """Generate human-readable description for the concept."""
        
        description_parts = [f"Concept '{concept_name}'"]
        
        if cluster_texts:
            unique_patterns = set(cluster_texts[:10])  # Look at first 10 unique patterns
            if len(unique_patterns) > 1:
                description_parts.append(f"appears in {len(unique_patterns)} different contexts")
            else:
                description_parts.append("has consistent activation pattern")
        
        description_parts.append(f"with importance score {importance_score:.3f}")
        description_parts.append(f"and coherence score {coherence_score:.3f}")
        
        return " ".join(description_parts)
    
    def _compute_concept_metrics(self, concepts: List[ConceptInfo],
                                activations: np.ndarray) -> Dict[str, Any]:
        """Compute overall metrics for the extracted concepts."""
        if not concepts:
            return {}
        
        # Compute silhouette score if we have enough data
        try:
            if len(concepts) > 1 and activations.shape[0] > len(concepts):
                # Create cluster labels for silhouette score
                cluster_labels = np.zeros(activations.shape[0])
                current_idx = 0
                
                for i, concept in enumerate(concepts):
                    cluster_size = concept.cluster_size
                    cluster_labels[current_idx:current_idx + cluster_size] = i
                    current_idx += cluster_size
                
                if current_idx <= activations.shape[0]:
                    sil_score = silhouette_score(
                        activations[:current_idx], 
                        cluster_labels[:current_idx]
                    )
                else:
                    sil_score = 0.0
            else:
                sil_score = 0.0
        except Exception:
            sil_score = 0.0
        
        # Compute other metrics
        importance_scores = [c.importance_score for c in concepts]
        coherence_scores = [c.coherence_score for c in concepts]
        cluster_sizes = [c.cluster_size for c in concepts]
        
        metrics = {
            'silhouette_score': float(sil_score),
            'avg_importance': float(np.mean(importance_scores)),
            'avg_coherence': float(np.mean(coherence_scores)),
            'importance_std': float(np.std(importance_scores)),
            'coherence_std': float(np.std(coherence_scores)),
            'cluster_size_distribution': {
                'mean': float(np.mean(cluster_sizes)),
                'std': float(np.std(cluster_sizes)),
                'min': int(np.min(cluster_sizes)),
                'max': int(np.max(cluster_sizes))
            }
        }
        
        return metrics
    
    def _rank_concepts(self, concepts: List[ConceptInfo]) -> List[Dict[str, Any]]:
        """Rank concepts by importance and coherence."""
        # Combine importance and coherence for ranking
        ranked_concepts = []
        
        for concept in concepts:
            combined_score = 0.7 * concept.importance_score + 0.3 * concept.coherence_score
            
            ranked_concepts.append({
                'concept_id': concept.concept_id,
                'name': concept.name,
                'combined_score': combined_score,
                'importance_score': concept.importance_score,
                'coherence_score': concept.coherence_score,
                'cluster_size': concept.cluster_size
            })
        
        # Sort by combined score
        ranked_concepts.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return ranked_concepts
    
    def _serialize_concept(self, concept: ConceptInfo) -> Dict[str, Any]:
        """Serialize ConceptInfo to dictionary."""
        return {
            'concept_id': concept.concept_id,
            'name': concept.name,
            'description': concept.description,
            'activation_pattern': concept.activation_pattern.tolist(),
            'examples': concept.examples,
            'importance_score': concept.importance_score,
            'coherence_score': concept.coherence_score,
            'cluster_size': concept.cluster_size
        }
    
    def extract_concepts(self, model: Any, dataset: Any, **kwargs) -> Dict[str, Any]:
        """Interface method for ConceptPluginBase."""
        result = self.execute(model, dataset, **kwargs)
        return result.outputs if result.success else {}
    
    def validate_output(self, output: Any) -> bool:
        """Validate that the output contains required concept data."""
        if not isinstance(output, dict):
            return False
        
        required_keys = ['concepts', 'concept_scores']
        for key in required_keys:
            if key not in output:
                logger.error(f"Missing required output key: {key}")
                return False
        
        # Check that concepts are properly formatted
        concepts = output['concepts']
        if not isinstance(concepts, dict):
            logger.error("Concepts must be a dictionary")
            return False
        
        # Check that each concept has required fields
        for concept_id, concept_info in concepts.items():
            required_concept_keys = ['name', 'importance_score', 'coherence_score']
            for key in required_concept_keys:
                if key not in concept_info:
                    logger.error(f"Missing required concept key {key} for concept {concept_id}")
                    return False
        
        # Check that we have at least one concept
        if not concepts:
            logger.warning("No concepts were extracted")
            return False
        
        return True

def create_ace_extractor(config: Optional[Dict[str, Any]] = None) -> ACEConceptExtractor:
    """Factory function to create ACE concept extractor."""
    return ACEConceptExtractor(config=config)
