"""
SAE Feature Analysis Module

This module provides tools for analyzing features learned by Sparse Autoencoders,
including feature extraction, max activating examples, and feature interpretation.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

from .sae_training import SparseAutoencoder, SAEConfig, load_sae_model
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
class FeatureActivation:
    """Single feature activation example."""
    feature_id: int
    activation_value: float
    text: str
    token_index: int
    token: str
    context_before: str
    context_after: str
    layer_index: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'feature_id': self.feature_id,
            'activation_value': float(self.activation_value),
            'text': self.text,
            'token_index': self.token_index,
            'token': self.token,
            'context_before': self.context_before,
            'context_after': self.context_after,
            'layer_index': self.layer_index
        }


@dataclass
class FeatureAnalysis:
    """Analysis results for a single SAE feature."""
    feature_id: int
    feature_name: Optional[str] = None
    description: Optional[str] = None
    
    # Activation statistics
    mean_activation: float = 0.0
    std_activation: float = 0.0
    max_activation: float = 0.0
    sparsity: float = 0.0  # Fraction of inputs where feature is active
    
    # Top activating examples
    max_activating_examples: List[FeatureActivation] = field(default_factory=list)
    
    # Feature interpretability
    common_tokens: List[Tuple[str, int]] = field(default_factory=list)  # (token, count)
    common_contexts: List[Tuple[str, int]] = field(default_factory=list)  # (context, count)
    
    # Feature relationships
    correlated_features: List[Tuple[int, float]] = field(default_factory=list)  # (feature_id, correlation)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'feature_id': self.feature_id,
            'feature_name': self.feature_name,
            'description': self.description,
            'mean_activation': float(self.mean_activation),
            'std_activation': float(self.std_activation),
            'max_activation': float(self.max_activation),
            'sparsity': float(self.sparsity),
            'max_activating_examples': [ex.to_dict() for ex in self.max_activating_examples],
            'common_tokens': self.common_tokens,
            'common_contexts': self.common_contexts,
            'correlated_features': [(int(fid), float(corr)) for fid, corr in self.correlated_features]
        }


@dataclass
class SAEFeatureAnalysisResult:
    """Complete SAE feature analysis results."""
    model_name: str
    layer_index: int
    sae_config: SAEConfig
    feature_analyses: List[FeatureAnalysis]
    global_statistics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'layer_index': self.layer_index,
            'sae_config': self.sae_config.to_dict(),
            'feature_analyses': [fa.to_dict() for fa in self.feature_analyses],
            'global_statistics': self.global_statistics
        }


class SAEFeatureExtractor:
    """Extract and analyze features from trained SAE models."""
    
    def __init__(self, sae_model: SparseAutoencoder, sae_config: SAEConfig, device: Optional[torch.device] = None):
        self.sae_model = sae_model
        self.sae_config = sae_config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.sae_model.to(self.device)
        self.sae_model.eval()
    
    def extract_features(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Extract sparse features from input activations.
        
        Args:
            activations: Input activations [batch_size, input_dim]
            
        Returns:
            Sparse features [batch_size, hidden_dim]
        """
        activations = activations.to(self.device)
        
        with torch.no_grad():
            features = self.sae_model.encode(activations)
        
        return features
    
    def get_reconstruction(self, features: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input from features.
        
        Args:
            features: Sparse features [batch_size, hidden_dim]
            
        Returns:
            Reconstructed activations [batch_size, input_dim]
        """
        features = features.to(self.device)
        
        with torch.no_grad():
            reconstruction = self.sae_model.decode(features)
        
        return reconstruction
    
    def compute_feature_statistics(self, features: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Compute statistics for all features.
        
        Args:
            features: Feature activations [num_samples, num_features]
            
        Returns:
            Dictionary of statistics
        """
        features_np = features.cpu().numpy()
        
        # Basic statistics
        mean_activations = np.mean(features_np, axis=0)
        std_activations = np.std(features_np, axis=0)
        max_activations = np.max(features_np, axis=0)
        
        # Sparsity (fraction of samples where feature is active)
        threshold = 1e-6
        sparsity = np.mean(features_np > threshold, axis=0)
        
        return {
            'mean_activations': mean_activations,
            'std_activations': std_activations,
            'max_activations': max_activations,
            'sparsity': sparsity
        }
    
    def find_top_activating_examples(
        self,
        features: torch.Tensor,
        flat_texts_and_tokens: List[Tuple[str, List[str], int]],
        feature_id: int,
        top_k: int = 10
    ) -> List[FeatureActivation]:
        """
        Find examples that maximally activate a specific feature.
        
        Args:
            features: Feature activations [num_samples, num_features]
            flat_texts_and_tokens: List of (original_text, token_list, token_index) tuples
            feature_id: ID of the feature to analyze
            top_k: Number of top examples to return
            
        Returns:
            List of top activating examples
        """
        # Get activations for the specific feature
        feature_activations = features[:, feature_id].cpu().numpy()
        
        # Find top-k activating indices
        top_indices = np.argsort(feature_activations)[-top_k:][::-1]
        
        examples = []
        for idx in top_indices:
            idx = int(idx)
            activation_value = feature_activations[idx]
            
            if idx < len(flat_texts_and_tokens):
                original_text, token_list, token_index = flat_texts_and_tokens[idx]
                token = token_list[token_index] if token_list and 0 <= token_index < len(token_list) else ""
                
                # Create context
                context_size = 5
                context_before = " ".join(token_list[max(0, token_index-context_size):token_index])
                context_after = " ".join(token_list[token_index+1:token_index+1+context_size])
                
                example = FeatureActivation(
                    feature_id=feature_id,
                    activation_value=activation_value,
                    text=original_text,
                    token_index=token_index,
                    token=token,
                    context_before=context_before,
                    context_after=context_after,
                    layer_index=self.sae_config.layer
                )
                examples.append(example)
        
        return examples
    
    def analyze_feature_interpretability(
        self,
        examples: List[FeatureActivation],
        min_frequency: int = 2
    ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """
        Analyze interpretability of a feature based on its activating examples.
        
        Args:
            examples: List of activating examples
            min_frequency: Minimum frequency for tokens/contexts to be included
            
        Returns:
            Tuple of (common_tokens, common_contexts)
        """
        # Count tokens
        token_counts = defaultdict(int)
        context_counts = defaultdict(int)
        
        for example in examples:
            # Count the specific token
            if example.token:
                token_counts[example.token] += 1
            
            # Count context patterns
            if example.context_before:
                # Look for patterns in context
                context_words = example.context_before.split()
                if context_words:
                    # Use last word before as context pattern
                    context_counts[context_words[-1]] += 1
        
        # Filter by minimum frequency and sort
        common_tokens = [(token, count) for token, count in token_counts.items() 
                        if count >= min_frequency]
        common_tokens.sort(key=lambda x: x[1], reverse=True)
        
        common_contexts = [(context, count) for context, count in context_counts.items() 
                          if count >= min_frequency]
        common_contexts.sort(key=lambda x: x[1], reverse=True)
        
        return common_tokens, common_contexts
    
    def compute_feature_correlations(
        self,
        features: torch.Tensor,
        feature_id: int,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Compute correlations between a feature and all other features.
        
        Args:
            features: Feature activations [num_samples, num_features]
            feature_id: ID of the feature to analyze
            top_k: Number of top correlated features to return
            
        Returns:
            List of (feature_id, correlation) tuples
        """
        features_np = features.cpu().numpy()
        target_feature = features_np[:, feature_id]
        
        correlations = []
        for i in range(features.shape[1]):
            if i != feature_id:
                correlation = np.corrcoef(target_feature, features_np[:, i])[0, 1]
                if not np.isnan(correlation):
                    correlations.append((i, correlation))
        
        # Sort by absolute correlation value
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        return correlations[:top_k]


class MaxActivatingExamplesFinder:
    """Find max activating examples for SAE features."""
    
    def __init__(self, model_manager: ModelManager, sae_extractor: SAEFeatureExtractor):
        self.model_manager = model_manager
        self.sae_extractor = sae_extractor
        self.device = sae_extractor.device
    
    def find_max_activating_examples(
        self,
        texts: List[str],
        feature_ids: Optional[List[int]] = None,
        top_k: int = 10,
        batch_size: int = 8
    ) -> Tuple[Dict[int, List[FeatureActivation]], torch.Tensor]:
        """
        Find max activating examples for specified features.
        
        Args:
            texts: List of texts to analyze
            feature_ids: Specific feature IDs to analyze (default: all)
            top_k: Number of top examples per feature
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (Dictionary mapping feature_id to list of max activating examples, combined_features_tensor)
        """
        logger.info(f"Finding max activating examples for {len(texts)} texts")
        
        # Load the model and get the target module for activations
        adapter = self.model_manager.load_model(self.sae_extractor.sae_config.model_name)
        model = adapter.model
        
        target_module = None
        layer_idx = self.sae_extractor.sae_config.layer
        component = self.sae_extractor.sae_config.component

        if hasattr(model, 'transformer'):  # GPT-style models
            if component == 'residual':
                target_module = model.transformer.h[layer_idx]
            elif component == 'mlp':
                target_module = model.transformer.h[layer_idx].mlp
            elif component == 'attention':
                target_module = model.transformer.h[layer_idx].attn
        elif hasattr(model, 'encoder'):  # BERT-style models
            if component == 'residual':
                target_module = model.encoder.layer[layer_idx]
            elif component == 'mlp':
                target_module = model.encoder.layer[layer_idx].intermediate
            elif component == 'attention':
                target_module = model.encoder.layer[layer_idx].attention.output
        
        if target_module is None:
            raise ValueError(f"Could not find target module for {self.sae_extractor.sae_config.model_name}, layer {layer_idx}, component {component}")

        all_activations = []
        all_tokens_flat = [] # Store flattened tokens for context reconstruction
        
        def activation_hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            all_activations.append(output.detach().cpu())
        
        hook = target_module.register_forward_hook(activation_hook)
        
        try:
            for i in tqdm(range(0, len(texts), batch_size), desc="Processing texts for activations"):
                batch_texts = texts[i:i + batch_size]
                
                inputs = adapter.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.sae_extractor.sae_config.sequence_length
                ).to(self.device)
                
                with torch.no_grad():
                    _ = model(**inputs)
                
                # Store tokens for context reconstruction
                for text in batch_texts:
                    tokenized_input = adapter.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.sae_extractor.sae_config.sequence_length)
                    tokens = adapter.tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'][0])
                    all_tokens_flat.extend(tokens)

                # Clear cache for next batch
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        finally:
            hook.remove()
        
        if not all_activations:
            raise ValueError("No activations collected.")

        combined_activations = torch.cat(all_activations, dim=0).view(-1, self.sae_extractor.sae_config.input_dim)
        logger.info(f"Collected {combined_activations.shape[0]} activations for feature extraction.")

        # Extract features from collected activations
        combined_features = self.sae_extractor.extract_features(combined_activations.to(self.device)).cpu()
        logger.info(f"Extracted {combined_features.shape[0]} features.")
        
        # Determine which features to analyze
        if feature_ids is None:
            feature_ids = list(range(combined_features.shape[1]))
        
        # Find max activating examples for each feature
        results = {}
        
        # Create a flat list of (text, token_list) pairs corresponding to each activation
        # This is crucial for correctly reconstructing context for each activation point
        flat_texts_and_tokens = []
        for text in texts:
            tokenized_input = adapter.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.sae_extractor.sae_config.sequence_length)
            tokens = adapter.tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'][0])
            for i in range(len(tokens)):
                flat_texts_and_tokens.append((text, tokens, i))

        for feature_id in tqdm(feature_ids, desc="Finding max examples"):
            examples = self.sae_extractor.find_top_activating_examples(
                combined_features,
                flat_texts_and_tokens, # Pass the flat list of (text, tokens, token_index) for context
                feature_id,
                top_k
            )
            results[feature_id] = examples
        
        logger.info(f"Found max activating examples for {len(results)} features")
        return results, combined_features
    
    def analyze_all_features(
        self,
        texts: List[str],
        feature_ids: Optional[List[int]] = None,
        top_k_examples: int = 10,
        top_k_correlations: int = 10
    ) -> SAEFeatureAnalysisResult:
        """
        Perform complete analysis of SAE features.
        
        Args:
            texts: List of texts to analyze
            feature_ids: Specific feature IDs to analyze (default: all)
            top_k_examples: Number of top examples per feature
            top_k_correlations: Number of top correlations per feature
            
        Returns:
            Complete feature analysis results
        """
        logger.info("Starting complete SAE feature analysis")
        
        # Find max activating examples and get combined features
        max_examples, combined_features = self.find_max_activating_examples(
            texts, feature_ids, top_k_examples
        )
        
        # Compute feature statistics
        feature_stats = self.sae_extractor.compute_feature_statistics(combined_features)
        
        # Analyze each feature
        feature_analyses = []
        
        for feature_id in max_examples.keys():
            examples = max_examples[feature_id]
            
            # Analyze interpretability
            common_tokens, common_contexts = self.sae_extractor.analyze_feature_interpretability(examples)
            
            # Compute correlations
            correlations = self.sae_extractor.compute_feature_correlations(
                combined_features, feature_id, top_k_correlations
            )
            
            # Create feature analysis
            analysis = FeatureAnalysis(
                feature_id=feature_id,
                mean_activation=float(feature_stats['mean_activations'][feature_id]),
                std_activation=float(feature_stats['std_activations'][feature_id]),
                max_activation=float(feature_stats['max_activations'][feature_id]),
                sparsity=float(feature_stats['sparsity'][feature_id]),
                max_activating_examples=examples,
                common_tokens=common_tokens,
                common_contexts=common_contexts,
                correlated_features=correlations
            )
            
            feature_analyses.append(analysis)
        
        # Global statistics
        global_stats = {
            'num_features': len(feature_analyses),
            'mean_sparsity': float(np.mean([fa.sparsity for fa in feature_analyses])),
            'mean_max_activation': float(np.mean([fa.max_activation for fa in feature_analyses])),
            'num_texts_analyzed': len(texts)
        }
        
        result = SAEFeatureAnalysisResult(
            model_name=self.sae_extractor.sae_config.model_name,
            layer_index=self.sae_extractor.sae_config.layer,
            sae_config=self.sae_extractor.sae_config,
            feature_analyses=feature_analyses,
            global_statistics=global_stats
        )
        
        logger.info("SAE feature analysis complete")
        return result


# Utility functions
def load_sae_and_create_extractor(sae_model_path: str, device: Optional[torch.device] = None) -> SAEFeatureExtractor:
    """Load SAE model and create feature extractor."""
    sae_model, sae_config = load_sae_model(sae_model_path, device)
    return SAEFeatureExtractor(sae_model, sae_config, device)


def analyze_sae_features(
    model_manager: ModelManager,
    sae_model_path: str,
    texts: List[str],
    feature_ids: Optional[List[int]] = None,
    top_k_examples: int = 10
) -> SAEFeatureAnalysisResult:
    """Convenience function for complete SAE feature analysis."""
    # Load SAE
    extractor = load_sae_and_create_extractor(sae_model_path)
    
    # Create analyzer
    analyzer = MaxActivatingExamplesFinder(model_manager, extractor)
    
    # Perform analysis
    return analyzer.analyze_all_features(
        texts, feature_ids, top_k_examples
    )


def save_feature_analysis(result: SAEFeatureAnalysisResult, output_path: str):
    """Save feature analysis results to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    logger.info(f"Feature analysis saved to {output_path}")


def load_feature_analysis(input_path: str) -> SAEFeatureAnalysisResult:
    """Load feature analysis results from file."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Reconstruct the result (simplified)
    # In a full implementation, you'd need proper deserialization
    return SAEFeatureAnalysisResult(
        model_name=data['model_name'],
        layer_index=data['layer_index'],
        sae_config=SAEConfig.from_dict(data['sae_config']),
        feature_analyses=[],  # Would need proper reconstruction
        global_statistics=data['global_statistics']
    )
