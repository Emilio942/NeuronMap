"""
Semantic Labeling for NeuronMap
==============================

LLM-based automatic semantic labeling of neuron clusters and activations.
Provides human-interpretable descriptions for neural network components.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from pathlib import Path
import time
import json
from dataclasses import dataclass

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI library not available")

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available")

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from ...core.plugin_interface import AttributionPluginBase, ToolExecutionResult

logger = logging.getLogger(__name__)

@dataclass
class SemanticLabel:
    """Semantic label for a cluster or activation pattern."""
    label: str
    confidence: float
    description: str
    keywords: List[str]
    cluster_size: int
    representative_examples: List[str]

class SemanticLabeler(AttributionPluginBase):
    """
    LLM-based semantic labeling for neural network analysis.
    
    Automatically generates human-interpretable labels for neuron clusters,
    activation patterns, and feature representations using language models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(tool_id="llm_auto_labeling", config=config)
        
        self.version = "1.0.0"
        self.description = "LLM-based automatic semantic labeling of neuron clusters"
        
        # Configuration parameters
        self.use_openai = config.get('use_openai', False) if config else False
        self.openai_model = config.get('openai_model', 'gpt-3.5-turbo') if config else 'gpt-3.5-turbo'
        self.local_model = config.get('local_model', 'distilbert-base-uncased') if config else 'distilbert-base-uncased'
        self.max_clusters = config.get('max_clusters', 20) if config else 20
        self.min_cluster_size = config.get('min_cluster_size', 5) if config else 5
        self.confidence_threshold = config.get('confidence_threshold', 0.7) if config else 0.7
        
        # Model components
        self.tokenizer = None
        self.text_model = None
        self.text_pipeline = None
        
        logger.info(f"Initialized semantic labeler (OpenAI: {self.use_openai})")
    
    def initialize(self) -> bool:
        """Initialize the semantic labeler."""
        try:
            if self.use_openai and OPENAI_AVAILABLE:
                # Initialize OpenAI client
                logger.info("Using OpenAI for semantic labeling")
                # OpenAI API key should be set in environment
            elif TRANSFORMERS_AVAILABLE:
                # Initialize local transformer models
                logger.info(f"Using local model {self.local_model} for semantic labeling")
                self.tokenizer = AutoTokenizer.from_pretrained(self.local_model)
                self.text_model = AutoModel.from_pretrained(self.local_model)
                self.text_pipeline = pipeline("text-classification", 
                                             model=self.local_model, 
                                             return_all_scores=True)
            else:
                logger.warning("No suitable language model available, using rule-based labeling")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic labeler: {e}")
            return False
    
    def execute(self, cluster_activations: Dict[str, torch.Tensor],
                cluster_metadata: Optional[Dict[str, Any]] = None,
                input_examples: Optional[List[str]] = None,
                **kwargs) -> ToolExecutionResult:
        """
        Execute semantic labeling on cluster data.
        
        Args:
            cluster_activations: Dictionary mapping cluster IDs to activation tensors
            cluster_metadata: Optional metadata about clusters
            input_examples: Optional example inputs that produced the activations
            
        Returns:
            ToolExecutionResult with semantic labels and confidence scores
        """
        start_time = time.time()
        
        try:
            # Process clusters and generate labels
            semantic_labels = {}
            confidence_scores = {}
            
            for cluster_id, activations in cluster_activations.items():
                label_result = self._generate_cluster_label(
                    cluster_id, activations, cluster_metadata, input_examples
                )
                
                semantic_labels[cluster_id] = label_result
                confidence_scores[cluster_id] = label_result.confidence
            
            # Compute global metrics
            avg_confidence = np.mean(list(confidence_scores.values()))
            high_confidence_labels = sum(1 for conf in confidence_scores.values() 
                                       if conf >= self.confidence_threshold)
            
            # Generate summary
            summary = self._generate_labeling_summary(semantic_labels, confidence_scores)
            
            # Prepare outputs
            outputs = {
                'semantic_labels': {k: self._serialize_label(v) for k, v in semantic_labels.items()},
                'confidence_scores': confidence_scores,
                'summary': summary,
                'metrics': {
                    'total_clusters': len(cluster_activations),
                    'avg_confidence': avg_confidence,
                    'high_confidence_count': high_confidence_labels,
                    'labeling_coverage': len(semantic_labels) / len(cluster_activations)
                },
                'method_config': {
                    'use_openai': self.use_openai,
                    'model': self.openai_model if self.use_openai else self.local_model,
                    'confidence_threshold': self.confidence_threshold
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
            logger.error(f"Semantic labeling execution failed: {e}")
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
    
    def _generate_cluster_label(self, cluster_id: str, activations: torch.Tensor,
                               metadata: Optional[Dict[str, Any]] = None,
                               input_examples: Optional[List[str]] = None) -> SemanticLabel:
        """Generate semantic label for a single cluster."""
        
        # Extract cluster statistics
        cluster_stats = self._compute_cluster_statistics(activations)
        
        # Generate descriptive features
        features = self._extract_cluster_features(activations, cluster_stats)
        
        # Generate label using available method
        if self.use_openai and OPENAI_AVAILABLE:
            label_info = self._generate_openai_label(cluster_id, features, input_examples)
        elif TRANSFORMERS_AVAILABLE:
            label_info = self._generate_transformer_label(cluster_id, features, input_examples)
        else:
            label_info = self._generate_rule_based_label(cluster_id, features)
        
        return SemanticLabel(
            label=label_info['label'],
            confidence=label_info['confidence'],
            description=label_info['description'],
            keywords=label_info['keywords'],
            cluster_size=activations.shape[0],
            representative_examples=label_info.get('examples', [])
        )
    
    def _compute_cluster_statistics(self, activations: torch.Tensor) -> Dict[str, float]:
        """Compute statistical properties of cluster activations."""
        with torch.no_grad():
            stats = {
                'mean_activation': float(activations.mean()),
                'std_activation': float(activations.std()),
                'max_activation': float(activations.max()),
                'min_activation': float(activations.min()),
                'sparsity': float((activations == 0).sum() / activations.numel()),
                'l2_norm': float(torch.norm(activations)),
                'cluster_size': activations.shape[0]
            }
        
        return stats
    
    def _extract_cluster_features(self, activations: torch.Tensor, 
                                 stats: Dict[str, float]) -> Dict[str, Any]:
        """Extract interpretable features from cluster activations."""
        features = {}
        
        # Activation pattern features
        features['activation_level'] = self._categorize_activation_level(stats['mean_activation'])
        features['variability'] = self._categorize_variability(stats['std_activation'])
        features['sparsity_level'] = self._categorize_sparsity(stats['sparsity'])
        
        # Spatial features (if applicable)
        if activations.dim() > 2:
            features['spatial_pattern'] = self._analyze_spatial_pattern(activations)
        
        # Temporal features (if applicable)
        if 'sequence' in str(activations.device) or activations.shape[1] > 100:
            features['temporal_pattern'] = self._analyze_temporal_pattern(activations)
        
        # Frequency features
        features['dominant_frequencies'] = self._analyze_frequency_content(activations)
        
        return features
    
    def _categorize_activation_level(self, mean_activation: float) -> str:
        """Categorize the overall activation level."""
        if mean_activation > 1.0:
            return "high"
        elif mean_activation > 0.1:
            return "medium"
        elif mean_activation > 0.01:
            return "low"
        else:
            return "very_low"
    
    def _categorize_variability(self, std_activation: float) -> str:
        """Categorize the activation variability."""
        if std_activation > 1.0:
            return "high_variance"
        elif std_activation > 0.5:
            return "medium_variance"
        else:
            return "low_variance"
    
    def _categorize_sparsity(self, sparsity: float) -> str:
        """Categorize the sparsity level."""
        if sparsity > 0.8:
            return "very_sparse"
        elif sparsity > 0.5:
            return "sparse"
        elif sparsity > 0.2:
            return "moderately_dense"
        else:
            return "dense"
    
    def _analyze_spatial_pattern(self, activations: torch.Tensor) -> str:
        """Analyze spatial patterns in activations."""
        # Simplified spatial analysis
        if activations.dim() >= 3:
            # Check for edge-like patterns
            diff_h = torch.diff(activations, dim=-2).abs().mean()
            diff_w = torch.diff(activations, dim=-1).abs().mean()
            
            if diff_h > diff_w * 1.5:
                return "horizontal_edges"
            elif diff_w > diff_h * 1.5:
                return "vertical_edges"
            elif diff_h > 0.1 or diff_w > 0.1:
                return "edge_detection"
            else:
                return "smooth_regions"
        
        return "unknown"
    
    def _analyze_temporal_pattern(self, activations: torch.Tensor) -> str:
        """Analyze temporal patterns in activations."""
        # Simplified temporal analysis
        if activations.shape[1] > 10:
            # Check for sequential patterns
            autocorr = torch.corrcoef(activations.mean(dim=0))
            if torch.isnan(autocorr).any():
                return "irregular"
            
            avg_autocorr = autocorr.mean()
            if avg_autocorr > 0.7:
                return "highly_correlated"
            elif avg_autocorr > 0.3:
                return "moderately_correlated"
            else:
                return "weakly_correlated"
        
        return "unknown"
    
    def _analyze_frequency_content(self, activations: torch.Tensor) -> List[str]:
        """Analyze frequency content of activations."""
        # Simplified frequency analysis
        features = []
        
        # Check for periodic patterns
        mean_signal = activations.mean(dim=0)
        if mean_signal.numel() > 20:
            # Simple frequency analysis using variance
            high_freq_variance = torch.diff(mean_signal, n=2).var()
            low_freq_variance = mean_signal.var()
            
            if high_freq_variance > low_freq_variance * 0.5:
                features.append("high_frequency")
            else:
                features.append("low_frequency")
        
        return features
    
    def _generate_openai_label(self, cluster_id: str, features: Dict[str, Any],
                              input_examples: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate label using OpenAI API."""
        try:
            # Create prompt for OpenAI
            prompt = self._create_labeling_prompt(cluster_id, features, input_examples)
            
            # Call OpenAI API (simplified - you'd need actual API key and proper error handling)
            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content
            return self._parse_llm_response(result_text)
            
        except Exception as e:
            logger.error(f"OpenAI labeling failed: {e}")
            return self._generate_rule_based_label(cluster_id, features)
    
    def _generate_transformer_label(self, cluster_id: str, features: Dict[str, Any],
                                   input_examples: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate label using local transformer model."""
        try:
            # Create descriptive text from features
            feature_text = self._features_to_text(features)
            
            # Use text classification pipeline (simplified approach)
            if self.text_pipeline:
                # This is a simplified example - you'd need a trained model for this task
                results = self.text_pipeline(feature_text)
                
                if results and len(results[0]) > 0:
                    best_result = max(results[0], key=lambda x: x['score'])
                    return {
                        'label': best_result['label'],
                        'confidence': best_result['score'],
                        'description': f"Transformer-based classification: {feature_text}",
                        'keywords': self._extract_keywords_from_features(features)
                    }
            
            # Fallback to rule-based
            return self._generate_rule_based_label(cluster_id, features)
            
        except Exception as e:
            logger.error(f"Transformer labeling failed: {e}")
            return self._generate_rule_based_label(cluster_id, features)
    
    def _generate_rule_based_label(self, cluster_id: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate label using rule-based approach."""
        # Simple rule-based labeling
        label_parts = []
        keywords = []
        
        # Activation level
        activation_level = features.get('activation_level', 'unknown')
        label_parts.append(activation_level)
        keywords.append(activation_level)
        
        # Sparsity
        sparsity_level = features.get('sparsity_level', 'unknown')
        if sparsity_level != 'unknown':
            label_parts.append(sparsity_level)
            keywords.append(sparsity_level)
        
        # Spatial pattern
        spatial_pattern = features.get('spatial_pattern', '')
        if spatial_pattern:
            label_parts.append(spatial_pattern)
            keywords.append(spatial_pattern)
        
        # Create label
        if label_parts:
            label = "_".join(label_parts)
        else:
            label = f"cluster_{cluster_id}"
        
        return {
            'label': label,
            'confidence': 0.6,  # Medium confidence for rule-based
            'description': f"Rule-based classification based on activation patterns",
            'keywords': keywords
        }
    
    def _create_labeling_prompt(self, cluster_id: str, features: Dict[str, Any],
                               input_examples: Optional[List[str]] = None) -> str:
        """Create prompt for LLM labeling."""
        prompt = f"""
        Analyze the following neural network activation cluster and provide a semantic label:
        
        Cluster ID: {cluster_id}
        Activation Features:
        - Activation Level: {features.get('activation_level', 'unknown')}
        - Variability: {features.get('variability', 'unknown')}
        - Sparsity: {features.get('sparsity_level', 'unknown')}
        - Spatial Pattern: {features.get('spatial_pattern', 'none')}
        - Temporal Pattern: {features.get('temporal_pattern', 'none')}
        
        Please provide:
        1. A concise semantic label (2-4 words)
        2. A brief description of what this cluster might represent
        3. Keywords describing its function
        4. Confidence score (0-1)
        
        Format your response as JSON:
        {{
            "label": "semantic_label",
            "description": "brief description",
            "keywords": ["keyword1", "keyword2"],
            "confidence": 0.8
        }}
        """
        
        if input_examples:
            prompt += f"\n\nExample inputs that activated this cluster:\n"
            for example in input_examples[:3]:  # Limit to 3 examples
                prompt += f"- {example}\n"
        
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        try:
            # Try to parse as JSON
            result = json.loads(response_text)
            
            # Validate required fields
            if 'label' not in result:
                result['label'] = 'unknown'
            if 'confidence' not in result:
                result['confidence'] = 0.5
            if 'description' not in result:
                result['description'] = 'LLM-generated label'
            if 'keywords' not in result:
                result['keywords'] = []
            
            return result
            
        except json.JSONDecodeError:
            # Fallback parsing
            lines = response_text.strip().split('\n')
            label = lines[0] if lines else 'unknown'
            
            return {
                'label': label,
                'confidence': 0.7,
                'description': response_text,
                'keywords': label.split('_')
            }
    
    def _features_to_text(self, features: Dict[str, Any]) -> str:
        """Convert features to descriptive text."""
        text_parts = []
        
        for key, value in features.items():
            if isinstance(value, str):
                text_parts.append(f"{key}: {value}")
            elif isinstance(value, list):
                text_parts.append(f"{key}: {', '.join(map(str, value))}")
        
        return "; ".join(text_parts)
    
    def _extract_keywords_from_features(self, features: Dict[str, Any]) -> List[str]:
        """Extract keywords from features."""
        keywords = []
        
        for key, value in features.items():
            if isinstance(value, str) and value != 'unknown':
                keywords.append(value)
            elif isinstance(value, list):
                keywords.extend([str(v) for v in value])
        
        return list(set(keywords))
    
    def _serialize_label(self, label: SemanticLabel) -> Dict[str, Any]:
        """Serialize SemanticLabel to dictionary."""
        return {
            'label': label.label,
            'confidence': label.confidence,
            'description': label.description,
            'keywords': label.keywords,
            'cluster_size': label.cluster_size,
            'representative_examples': label.representative_examples
        }
    
    def _generate_labeling_summary(self, semantic_labels: Dict[str, SemanticLabel],
                                  confidence_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate summary of labeling results."""
        
        # Collect all labels and keywords
        all_labels = [label.label for label in semantic_labels.values()]
        all_keywords = []
        for label in semantic_labels.values():
            all_keywords.extend(label.keywords)
        
        # Compute summary statistics
        summary = {
            'total_clusters_labeled': len(semantic_labels),
            'unique_labels': len(set(all_labels)),
            'most_common_labels': self._get_most_common(all_labels, 5),
            'most_common_keywords': self._get_most_common(all_keywords, 10),
            'confidence_distribution': {
                'mean': np.mean(list(confidence_scores.values())),
                'std': np.std(list(confidence_scores.values())),
                'min': min(confidence_scores.values()),
                'max': max(confidence_scores.values())
            }
        }
        
        return summary
    
    def _get_most_common(self, items: List[str], top_k: int) -> List[Tuple[str, int]]:
        """Get most common items with counts."""
        from collections import Counter
        counter = Counter(items)
        return counter.most_common(top_k)
    
    def compute_attributions(self, model: Any, inputs: Any, **kwargs) -> Dict[str, Any]:
        """Interface method for AttributionPluginBase."""
        # Convert inputs to cluster format if needed
        if isinstance(inputs, torch.Tensor):
            cluster_activations = {'main_cluster': inputs}
        else:
            cluster_activations = inputs
        
        result = self.execute(cluster_activations, **kwargs)
        return result.outputs if result.success else {}
    
    def validate_output(self, output: Any) -> bool:
        """Validate that the output contains required semantic labeling data."""
        if not isinstance(output, dict):
            return False
        
        required_keys = ['semantic_labels', 'confidence_scores']
        for key in required_keys:
            if key not in output:
                logger.error(f"Missing required output key: {key}")
                return False
        
        # Check that semantic labels are properly formatted
        semantic_labels = output['semantic_labels']
        if not isinstance(semantic_labels, dict):
            logger.error("Semantic labels must be a dictionary")
            return False
        
        # Check that each label has required fields
        for cluster_id, label_info in semantic_labels.items():
            required_label_keys = ['label', 'confidence', 'description']
            for key in required_label_keys:
                if key not in label_info:
                    logger.error(f"Missing required label key {key} for cluster {cluster_id}")
                    return False
        
        return True

def create_semantic_labeler(config: Optional[Dict[str, Any]] = None) -> SemanticLabeler:
    """Factory function to create semantic labeler."""
    return SemanticLabeler(config=config)
