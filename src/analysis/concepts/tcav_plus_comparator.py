"""
TCAV++ Concept Comparator for NeuronMap
======================================

Advanced concept comparison using TCAV++ methodology with
CKA (Centered Kernel Alignment) and Cosine similarity metrics.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from pathlib import Path
import time
from dataclasses import dataclass

# Optional imports with fallbacks
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available, using basic implementations")

try:
    import scipy.stats as stats
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available, using numpy alternatives")

from ...core.plugin_interface import ConceptPluginBase, ToolExecutionResult

logger = logging.getLogger(__name__)

@dataclass
class ConceptComparison:
    """Result of concept comparison analysis."""
    concept_a_id: str
    concept_b_id: str
    cka_similarity: float
    cosine_similarity: float
    tcav_score: float
    compatibility_score: float
    directional_agreement: float
    concept_overlap: Dict[str, Any]

class TCAVPlusComparator(ConceptPluginBase):
    """
    TCAV++ Concept Comparator with advanced similarity metrics.
    
    Implements concept compatibility analysis using:
    - CKA (Centered Kernel Alignment)
    - Cosine similarity
    - TCAV directional derivatives
    - Concept overlap analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(tool_id="tcav_plus_comparator", config=config)
        
        self.version = "1.0.0"
        self.description = "TCAV++ concept comparison with CKA and cosine metrics"
        
        # Configuration
        self.similarity_threshold = config.get('similarity_threshold', 0.7) if config else 0.7
        self.cka_regularization = config.get('cka_regularization', 1e-6) if config else 1e-6
        self.concept_activation_threshold = config.get('concept_activation_threshold', 0.5) if config else 0.5
        self.directional_samples = config.get('directional_samples', 100) if config else 100
        self.normalize_activations = config.get('normalize_activations', True) if config else True
        
        logger.info("Initialized TCAV++ concept comparator")
    
    def initialize(self) -> bool:
        """Initialize the TCAV++ comparator."""
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("Scikit-learn not available - using basic implementations")
            
            if not SCIPY_AVAILABLE:
                logger.warning("SciPy not available - limited statistical analysis")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TCAV++ comparator: {e}")
            return False
    
    def execute(self, 
                concept_a: Dict[str, Any],
                concept_b: Dict[str, Any],
                model_activations: Optional[Dict[str, torch.Tensor]] = None,
                **kwargs) -> ToolExecutionResult:
        """
        Execute concept comparison analysis.
        
        Args:
            concept_a: First concept data (activations, CAVs, metadata)
            concept_b: Second concept data (activations, CAVs, metadata)
            model_activations: Optional model activations for TCAV analysis
            
        Returns:
            ToolExecutionResult with comparison analysis
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            self._validate_concept_inputs(concept_a, concept_b)
            
            # Extract concept activations and CAVs
            activations_a = self._extract_concept_activations(concept_a)
            activations_b = self._extract_concept_activations(concept_b)
            
            cav_a = concept_a.get('cav_vector')
            cav_b = concept_b.get('cav_vector')
            
            # Compute CKA similarity
            cka_similarity = self._compute_cka_similarity(activations_a, activations_b)
            
            # Compute cosine similarity
            cosine_sim = self._compute_cosine_similarity(activations_a, activations_b, cav_a, cav_b)
            
            # Compute TCAV scores if model activations provided
            tcav_score = 0.0
            directional_agreement = 0.0
            if model_activations is not None and cav_a is not None and cav_b is not None:
                tcav_score = self._compute_tcav_similarity(cav_a, cav_b, model_activations)
                directional_agreement = self._compute_directional_agreement(cav_a, cav_b)
            
            # Analyze concept overlap
            concept_overlap = self._analyze_concept_overlap(concept_a, concept_b, activations_a, activations_b)
            
            # Compute overall compatibility score
            compatibility_score = self._compute_compatibility_score(
                cka_similarity, cosine_sim, tcav_score, directional_agreement
            )
            
            # Create comparison result
            comparison = ConceptComparison(
                concept_a_id=concept_a.get('id', 'concept_a'),
                concept_b_id=concept_b.get('id', 'concept_b'),
                cka_similarity=cka_similarity,
                cosine_similarity=cosine_sim,
                tcav_score=tcav_score,
                compatibility_score=compatibility_score,
                directional_agreement=directional_agreement,
                concept_overlap=concept_overlap
            )
            
            # Generate detailed analysis
            detailed_analysis = self._generate_detailed_analysis(comparison, concept_a, concept_b)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(comparison)
            
            # Prepare outputs
            outputs = {
                'comparison_summary': {
                    'concept_a_id': comparison.concept_a_id,
                    'concept_b_id': comparison.concept_b_id,
                    'compatibility_score': compatibility_score,
                    'is_compatible': compatibility_score > self.similarity_threshold
                },
                'similarity_metrics': {
                    'cka_similarity': cka_similarity,
                    'cosine_similarity': cosine_sim,
                    'tcav_score': tcav_score,
                    'directional_agreement': directional_agreement
                },
                'concept_overlap_analysis': concept_overlap,
                'detailed_analysis': detailed_analysis,
                'recommendations': recommendations,
                'method_metadata': {
                    'similarity_threshold': self.similarity_threshold,
                    'cka_regularization': self.cka_regularization,
                    'normalization_applied': self.normalize_activations,
                    'concept_a_samples': activations_a.shape[0] if isinstance(activations_a, np.ndarray) else 0,
                    'concept_b_samples': activations_b.shape[0] if isinstance(activations_b, np.ndarray) else 0
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
            logger.error(f"TCAV++ comparison failed: {e}")
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
    
    def _validate_concept_inputs(self, concept_a: Dict[str, Any], concept_b: Dict[str, Any]):
        """Validate concept input data."""
        required_keys = ['activations']
        
        for concept, name in [(concept_a, 'concept_a'), (concept_b, 'concept_b')]:
            for key in required_keys:
                if key not in concept:
                    raise ValueError(f"Missing required key '{key}' in {name}")
    
    def _extract_concept_activations(self, concept: Dict[str, Any]) -> np.ndarray:
        """Extract and preprocess concept activations."""
        activations = concept['activations']
        
        # Convert to numpy if needed
        if torch.is_tensor(activations):
            activations = activations.detach().cpu().numpy()
        elif not isinstance(activations, np.ndarray):
            activations = np.array(activations)
        
        # Normalize if requested
        if self.normalize_activations:
            if SKLEARN_AVAILABLE:
                scaler = StandardScaler()
                activations = scaler.fit_transform(activations)
            else:
                # Basic z-score normalization
                mean = np.mean(activations, axis=0)
                std = np.std(activations, axis=0)
                activations = (activations - mean) / (std + 1e-8)
        
        return activations
    
    def _compute_cka_similarity(self, activations_a: np.ndarray, activations_b: np.ndarray) -> float:
        """Compute Centered Kernel Alignment (CKA) similarity."""
        try:
            # Compute Gram matrices
            gram_a = np.dot(activations_a, activations_a.T)
            gram_b = np.dot(activations_b, activations_b.T)
            
            # Center the Gram matrices
            n_a, n_b = gram_a.shape[0], gram_b.shape[0]
            
            # For different sized matrices, take the minimum
            min_size = min(n_a, n_b)
            gram_a = gram_a[:min_size, :min_size]
            gram_b = gram_b[:min_size, :min_size]
            
            # Centering matrix
            h = np.eye(min_size) - np.ones((min_size, min_size)) / min_size
            
            # Centered Gram matrices
            centered_gram_a = h @ gram_a @ h
            centered_gram_b = h @ gram_b @ h
            
            # Compute CKA
            numerator = np.trace(centered_gram_a @ centered_gram_b)
            denominator = np.sqrt(np.trace(centered_gram_a @ centered_gram_a) * 
                                 np.trace(centered_gram_b @ centered_gram_b))
            
            if denominator == 0:
                return 0.0
            
            cka = numerator / denominator
            return float(np.clip(cka, 0, 1))
            
        except Exception as e:
            logger.warning(f"CKA computation failed: {e}")
            return 0.0
    
    def _compute_cosine_similarity(self, activations_a: np.ndarray, activations_b: np.ndarray,
                                  cav_a: Optional[np.ndarray] = None, 
                                  cav_b: Optional[np.ndarray] = None) -> float:
        """Compute cosine similarity between concepts."""
        try:
            # If CAVs are available, use them
            if cav_a is not None and cav_b is not None:
                if torch.is_tensor(cav_a):
                    cav_a = cav_a.detach().cpu().numpy()
                if torch.is_tensor(cav_b):
                    cav_b = cav_b.detach().cpu().numpy()
                
                # Compute cosine similarity between CAVs
                if SKLEARN_AVAILABLE:
                    similarity = cosine_similarity(cav_a.reshape(1, -1), cav_b.reshape(1, -1))[0, 0]
                else:
                    # Manual cosine similarity
                    dot_product = np.dot(cav_a.flatten(), cav_b.flatten())
                    norm_a = np.linalg.norm(cav_a.flatten())
                    norm_b = np.linalg.norm(cav_b.flatten())
                    similarity = dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
                
                return float(np.clip(similarity, -1, 1))
            
            # Otherwise, use activation centroids
            centroid_a = np.mean(activations_a, axis=0)
            centroid_b = np.mean(activations_b, axis=0)
            
            if SKLEARN_AVAILABLE:
                similarity = cosine_similarity(centroid_a.reshape(1, -1), centroid_b.reshape(1, -1))[0, 0]
            else:
                dot_product = np.dot(centroid_a, centroid_b)
                norm_a = np.linalg.norm(centroid_a)
                norm_b = np.linalg.norm(centroid_b)
                similarity = dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
            
            return float(np.clip(similarity, -1, 1))
            
        except Exception as e:
            logger.warning(f"Cosine similarity computation failed: {e}")
            return 0.0
    
    def _compute_tcav_similarity(self, cav_a: np.ndarray, cav_b: np.ndarray,
                                model_activations: Dict[str, torch.Tensor]) -> float:
        """Compute TCAV-based similarity score."""
        try:
            tcav_scores = []
            
            for layer_name, activations in model_activations.items():
                if torch.is_tensor(activations):
                    activations = activations.detach().cpu().numpy()
                
                # Flatten activations for dot product
                flat_activations = activations.reshape(activations.shape[0], -1)
                
                # Ensure CAV dimensions match
                if flat_activations.shape[1] != len(cav_a.flatten()):
                    continue
                
                # Compute TCAV scores for both concepts
                tcav_a = np.mean(flat_activations @ cav_a.flatten() > self.concept_activation_threshold)
                tcav_b = np.mean(flat_activations @ cav_b.flatten() > self.concept_activation_threshold)
                
                # Similarity of TCAV scores
                tcav_similarity = 1 - abs(tcav_a - tcav_b)
                tcav_scores.append(tcav_similarity)
            
            return float(np.mean(tcav_scores)) if tcav_scores else 0.0
            
        except Exception as e:
            logger.warning(f"TCAV similarity computation failed: {e}")
            return 0.0
    
    def _compute_directional_agreement(self, cav_a: np.ndarray, cav_b: np.ndarray) -> float:
        """Compute directional agreement between CAV vectors."""
        try:
            if torch.is_tensor(cav_a):
                cav_a = cav_a.detach().cpu().numpy()
            if torch.is_tensor(cav_b):
                cav_b = cav_b.detach().cpu().numpy()
            
            # Flatten vectors
            vec_a = cav_a.flatten()
            vec_b = cav_b.flatten()
            
            # Compute cosine of angle between vectors
            cosine_angle = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
            
            # Convert to agreement score (0 to 1)
            agreement = (cosine_angle + 1) / 2  # Map from [-1, 1] to [0, 1]
            
            return float(agreement)
            
        except Exception as e:
            logger.warning(f"Directional agreement computation failed: {e}")
            return 0.0
    
    def _analyze_concept_overlap(self, concept_a: Dict[str, Any], concept_b: Dict[str, Any],
                               activations_a: np.ndarray, activations_b: np.ndarray) -> Dict[str, Any]:
        """Analyze overlap between concepts."""
        
        overlap_analysis = {}
        
        try:
            # Activation space overlap
            # Find highly activated samples for each concept
            activation_threshold = np.percentile(np.concatenate([activations_a.flatten(), 
                                                               activations_b.flatten()]), 75)
            
            highly_active_a = np.mean(activations_a > activation_threshold, axis=1)
            highly_active_b = np.mean(activations_b > activation_threshold, axis=1)
            
            overlap_analysis['activation_overlap'] = {
                'concept_a_active_ratio': float(np.mean(highly_active_a > 0.5)),
                'concept_b_active_ratio': float(np.mean(highly_active_b > 0.5)),
                'threshold_used': float(activation_threshold)
            }
            
            # Statistical overlap
            if SCIPY_AVAILABLE:
                # Compute KS test for distribution overlap
                ks_stat, ks_p_value = stats.ks_2samp(activations_a.flatten(), activations_b.flatten())
                overlap_analysis['statistical_overlap'] = {
                    'ks_statistic': float(ks_stat),
                    'ks_p_value': float(ks_p_value),
                    'distributions_similar': ks_p_value > 0.05
                }
            
            # Feature importance overlap
            feature_importance_a = np.std(activations_a, axis=0)
            feature_importance_b = np.std(activations_b, axis=0)
            
            # Compute overlap in important features
            important_features_a = feature_importance_a > np.percentile(feature_importance_a, 75)
            important_features_b = feature_importance_b > np.percentile(feature_importance_b, 75)
            
            feature_overlap = np.mean(important_features_a & important_features_b)
            
            overlap_analysis['feature_overlap'] = {
                'important_feature_overlap': float(feature_overlap),
                'concept_a_important_features': int(np.sum(important_features_a)),
                'concept_b_important_features': int(np.sum(important_features_b))
            }
            
        except Exception as e:
            logger.warning(f"Concept overlap analysis failed: {e}")
            overlap_analysis['error'] = str(e)
        
        return overlap_analysis
    
    def _compute_compatibility_score(self, cka_sim: float, cosine_sim: float, 
                                   tcav_score: float, directional_agreement: float) -> float:
        """Compute overall compatibility score."""
        
        # Weighted combination of metrics
        weights = {
            'cka': 0.3,
            'cosine': 0.3,
            'tcav': 0.2,
            'directional': 0.2
        }
        
        # Handle missing TCAV scores
        if tcav_score == 0.0 and directional_agreement == 0.0:
            # Redistribute weights when TCAV metrics unavailable
            weights = {'cka': 0.5, 'cosine': 0.5, 'tcav': 0.0, 'directional': 0.0}
        
        compatibility = (weights['cka'] * cka_sim + 
                        weights['cosine'] * abs(cosine_sim) +  # Take absolute value for cosine
                        weights['tcav'] * tcav_score +
                        weights['directional'] * directional_agreement)
        
        return float(np.clip(compatibility, 0, 1))
    
    def _generate_detailed_analysis(self, comparison: ConceptComparison,
                                   concept_a: Dict[str, Any], concept_b: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed analysis of concept comparison."""
        
        analysis = {
            'compatibility_assessment': self._assess_compatibility(comparison),
            'similarity_breakdown': {
                'cka_analysis': self._interpret_cka_score(comparison.cka_similarity),
                'cosine_analysis': self._interpret_cosine_score(comparison.cosine_similarity),
                'tcav_analysis': self._interpret_tcav_score(comparison.tcav_score),
                'directional_analysis': self._interpret_directional_score(comparison.directional_agreement)
            },
            'concept_properties': {
                'concept_a': self._extract_concept_properties(concept_a),
                'concept_b': self._extract_concept_properties(concept_b)
            }
        }
        
        return analysis
    
    def _assess_compatibility(self, comparison: ConceptComparison) -> str:
        """Assess overall compatibility between concepts."""
        score = comparison.compatibility_score
        
        if score >= 0.8:
            return "Highly compatible concepts - strong alignment across metrics"
        elif score >= 0.6:
            return "Moderately compatible concepts - good alignment with some differences"
        elif score >= 0.4:
            return "Partially compatible concepts - mixed alignment signals"
        elif score >= 0.2:
            return "Low compatibility - concepts appear quite different"
        else:
            return "Very low compatibility - concepts are likely orthogonal or opposing"
    
    def _interpret_cka_score(self, cka_score: float) -> str:
        """Interpret CKA similarity score."""
        if cka_score >= 0.8:
            return f"Very high representational similarity (CKA={cka_score:.3f})"
        elif cka_score >= 0.6:
            return f"High representational similarity (CKA={cka_score:.3f})"
        elif cka_score >= 0.4:
            return f"Moderate representational similarity (CKA={cka_score:.3f})"
        elif cka_score >= 0.2:
            return f"Low representational similarity (CKA={cka_score:.3f})"
        else:
            return f"Very low representational similarity (CKA={cka_score:.3f})"
    
    def _interpret_cosine_score(self, cosine_score: float) -> str:
        """Interpret cosine similarity score."""
        abs_score = abs(cosine_score)
        direction = "same" if cosine_score >= 0 else "opposite"
        
        if abs_score >= 0.8:
            return f"Very high directional alignment in {direction} direction (cosine={cosine_score:.3f})"
        elif abs_score >= 0.6:
            return f"High directional alignment in {direction} direction (cosine={cosine_score:.3f})"
        elif abs_score >= 0.4:
            return f"Moderate directional alignment in {direction} direction (cosine={cosine_score:.3f})"
        else:
            return f"Low directional alignment (cosine={cosine_score:.3f})"
    
    def _interpret_tcav_score(self, tcav_score: float) -> str:
        """Interpret TCAV similarity score."""
        if tcav_score == 0.0:
            return "TCAV analysis not available (requires model activations)"
        elif tcav_score >= 0.8:
            return f"Very similar TCAV activation patterns (score={tcav_score:.3f})"
        elif tcav_score >= 0.6:
            return f"Similar TCAV activation patterns (score={tcav_score:.3f})"
        elif tcav_score >= 0.4:
            return f"Moderately similar TCAV patterns (score={tcav_score:.3f})"
        else:
            return f"Different TCAV activation patterns (score={tcav_score:.3f})"
    
    def _interpret_directional_score(self, directional_score: float) -> str:
        """Interpret directional agreement score."""
        if directional_score == 0.0:
            return "Directional analysis not available (requires CAV vectors)"
        elif directional_score >= 0.8:
            return f"Very high directional agreement (score={directional_score:.3f})"
        elif directional_score >= 0.6:
            return f"High directional agreement (score={directional_score:.3f})"
        elif directional_score >= 0.4:
            return f"Moderate directional agreement (score={directional_score:.3f})"
        else:
            return f"Low directional agreement (score={directional_score:.3f})"
    
    def _extract_concept_properties(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Extract properties of a concept."""
        properties = {
            'id': concept.get('id', 'unknown'),
            'description': concept.get('description', 'No description'),
            'num_samples': len(concept.get('activations', [])),
            'has_cav': concept.get('cav_vector') is not None,
            'activation_statistics': {}
        }
        
        try:
            activations = concept.get('activations')
            if activations is not None:
                if torch.is_tensor(activations):
                    activations = activations.detach().cpu().numpy()
                elif not isinstance(activations, np.ndarray):
                    activations = np.array(activations)
                
                properties['activation_statistics'] = {
                    'mean': float(np.mean(activations)),
                    'std': float(np.std(activations)),
                    'min': float(np.min(activations)),
                    'max': float(np.max(activations)),
                    'sparsity': float(np.mean(activations == 0))
                }
        except Exception as e:
            logger.warning(f"Failed to extract activation statistics: {e}")
        
        return properties
    
    def _generate_recommendations(self, comparison: ConceptComparison) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []
        
        # Compatibility-based recommendations
        if comparison.compatibility_score >= 0.7:
            recommendations.append("Concepts are highly compatible - consider merging or treating as related")
        elif comparison.compatibility_score <= 0.3:
            recommendations.append("Concepts are quite different - investigate why they diverge")
        
        # CKA-based recommendations
        if comparison.cka_similarity < 0.3:
            recommendations.append("Low CKA similarity suggests different representational spaces")
        
        # Cosine-based recommendations  
        if abs(comparison.cosine_similarity) > 0.8:
            if comparison.cosine_similarity < 0:
                recommendations.append("High negative cosine similarity suggests opposing concepts")
            else:
                recommendations.append("High positive cosine similarity suggests aligned concepts")
        
        # TCAV-based recommendations
        if comparison.tcav_score > 0 and comparison.tcav_score < 0.4:
            recommendations.append("Different TCAV patterns suggest concepts activate differently in model")
        
        # Directional recommendations
        if comparison.directional_agreement < 0.4 and comparison.directional_agreement > 0:
            recommendations.append("Low directional agreement - CAV vectors point in different directions")
        
        return recommendations
    
    def compare_concepts(self, concepts: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Interface method for ConceptPluginBase."""
        if len(concepts) < 2:
            return {'error': 'Need at least 2 concepts for comparison'}
        
        result = self.execute(concepts[0], concepts[1], **kwargs)
        return result.outputs if result.success else {}
    
    def validate_output(self, output: Any) -> bool:
        """Validate TCAV++ comparison output."""
        if not isinstance(output, dict):
            return False
        
        required_keys = ['comparison_summary', 'similarity_metrics']
        for key in required_keys:
            if key not in output:
                logger.error(f"Missing required output key: {key}")
                return False
        
        # Validate similarity scores are in valid ranges
        similarity_metrics = output['similarity_metrics']
        for metric, value in similarity_metrics.items():
            if not isinstance(value, (int, float)) or np.isnan(value):
                logger.error(f"Invalid similarity metric {metric}: {value}")
                return False
        
        return True

def create_tcav_plus_comparator(config: Optional[Dict[str, Any]] = None) -> TCAVPlusComparator:
    """Factory function to create TCAV++ comparator."""
    return TCAVPlusComparator(config=config)
