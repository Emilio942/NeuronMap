"""Ethics, Fairness and Bias Analysis Module.

This module implements bias detection and fairness analysis for neural network activations:
- Bias detection in neural activations
- Fairness metrics for model analysis
- Model cards generation
- Audit trails for transparency
"""

import logging
import numpy as np
import torch
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import warnings
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BiasMetrics:
    """Metrics for bias assessment."""
    demographic_parity: float
    equalized_odds: float
    equal_opportunity: float
    statistical_parity: float
    bias_score: float
    affected_groups: List[str]
    confidence_interval: Tuple[float, float]


@dataclass
class FairnessResult:
    """Results from fairness analysis."""
    overall_bias_score: float
    group_bias_scores: Dict[str, float]
    fairness_metrics: BiasMetrics
    recommendations: List[str]
    warnings: List[str]


@dataclass
class ModelCard:
    """Model card for transparency and documentation."""
    model_name: str
    model_version: str
    created_date: str
    description: str
    intended_use: str
    limitations: List[str]
    bias_assessment: Dict[str, Any]
    performance_metrics: Dict[str, float]
    training_data: Dict[str, Any]
    evaluation_data: Dict[str, Any]
    ethical_considerations: List[str]
    recommendations: List[str]


class BiasDetector(ABC):
    """Abstract base class for bias detection methods."""

    @abstractmethod
    def detect_bias(self, activations: np.ndarray,
                    groups: np.ndarray,
                    labels: Optional[np.ndarray] = None) -> BiasMetrics:
        """Detect bias in neural activations."""
        pass


class ActivationBiasDetector(BiasDetector):
    """Detects bias in neural network activations."""

    def __init__(self, threshold: float = 0.1):
        """Initialize bias detector.

        Args:
            threshold: Threshold for bias detection (0-1)
        """
        self.threshold = threshold

    def detect_bias(self, activations: np.ndarray,
                    groups: np.ndarray,
                    labels: Optional[np.ndarray] = None) -> BiasMetrics:
        """Detect bias in neural activations.

        Args:
            activations: Neural activations (samples x features)
            groups: Group assignments for each sample
            labels: Optional labels for supervised bias detection

        Returns:
            BiasMetrics object with bias assessment
        """
        logger.info("Analyzing activation patterns for bias...")

        # Calculate group-wise activation statistics
        unique_groups = np.unique(groups)
        group_stats = {}

        for group in unique_groups:
            group_mask = groups == group
            group_activations = activations[group_mask]

            group_stats[group] = {
                'mean': np.mean(group_activations, axis=0),
                'std': np.std(group_activations, axis=0),
                'count': np.sum(group_mask)
            }

        # Calculate bias metrics
        bias_scores = self._calculate_bias_scores(group_stats, unique_groups)
        fairness_metrics = self._calculate_fairness_metrics(
            activations, groups, labels
        )

        # Identify affected groups
        affected_groups = [
            group for group, score in bias_scores.items()
            if score > self.threshold
        ]

        # Calculate overall bias score
        overall_bias = np.mean(list(bias_scores.values()))

        # Calculate confidence interval (bootstrap)
        ci_lower, ci_upper = self._bootstrap_confidence_interval(
            activations, groups, labels
        )

        return BiasMetrics(
            demographic_parity=fairness_metrics['demographic_parity'],
            equalized_odds=fairness_metrics['equalized_odds'],
            equal_opportunity=fairness_metrics['equal_opportunity'],
            statistical_parity=fairness_metrics['statistical_parity'],
            bias_score=overall_bias,
            affected_groups=affected_groups,
            confidence_interval=(ci_lower, ci_upper)
        )

    def _calculate_bias_scores(self, group_stats: Dict,
                               groups: np.ndarray) -> Dict[str, float]:
        """Calculate bias scores between groups."""
        bias_scores = {}

        # Calculate pairwise differences between group means
        group_list = list(group_stats.keys())

        for i, group1 in enumerate(group_list):
            max_diff = 0

            for j, group2 in enumerate(group_list):
                if i != j:
                    mean1 = group_stats[group1]['mean']
                    mean2 = group_stats[group2]['mean']

                    # Calculate normalized difference
                    diff = np.abs(mean1 - mean2)
                    norm_diff = np.mean(diff / (np.abs(mean1) + np.abs(mean2) + 1e-8))

                    max_diff = max(max_diff, norm_diff)

            bias_scores[group1] = max_diff

        return bias_scores

    def _calculate_fairness_metrics(self, activations: np.ndarray,
                                    groups: np.ndarray,
                                    labels: Optional[np.ndarray]) -> Dict[str, float]:
        """Calculate fairness metrics."""
        metrics = {}

        # Demographic Parity (equal representation in high-activation regions)
        high_activation_mask = np.mean(
            activations,
            axis=1) > np.median(
            np.mean(
                activations,
                axis=1))
        group_representation = {}

        for group in np.unique(groups):
            group_mask = groups == group
            group_high_activation = np.sum(high_activation_mask & group_mask)
            group_total = np.sum(group_mask)
            group_representation[group] = group_high_activation / \
                group_total if group_total > 0 else 0

        # Calculate demographic parity as standard deviation of representation rates
        metrics['demographic_parity'] = np.std(list(group_representation.values()))

        # Statistical Parity (similar to demographic parity)
        metrics['statistical_parity'] = metrics['demographic_parity']

        if labels is not None:
            # Equalized Odds (equal TPR and FPR across groups)
            metrics['equalized_odds'] = self._calculate_equalized_odds(
                activations, groups, labels
            )

            # Equal Opportunity (equal TPR across groups)
            metrics['equal_opportunity'] = self._calculate_equal_opportunity(
                activations, groups, labels
            )
        else:
            # Use activation-based proxies
            metrics['equalized_odds'] = metrics['demographic_parity']
            metrics['equal_opportunity'] = metrics['demographic_parity']

        return metrics

    def _calculate_equalized_odds(self, activations: np.ndarray,
                                  groups: np.ndarray,
                                  labels: np.ndarray) -> float:
        """Calculate equalized odds metric."""
        # Use activation threshold as proxy for model predictions
        predictions = np.mean(
            activations,
            axis=1) > np.median(
            np.mean(
                activations,
                axis=1))

        group_tpr = {}
        group_fpr = {}

        for group in np.unique(groups):
            group_mask = groups == group
            group_labels = labels[group_mask]
            group_preds = predictions[group_mask]

            if len(group_labels) == 0:
                continue

            # Calculate TPR and FPR
            tp = np.sum((group_labels == 1) & (group_preds == 1))
            fn = np.sum((group_labels == 1) & (group_preds == 0))
            fp = np.sum((group_labels == 0) & (group_preds == 1))
            tn = np.sum((group_labels == 0) & (group_preds == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            group_tpr[group] = tpr
            group_fpr[group] = fpr

        # Calculate difference in TPR and FPR across groups
        tpr_diff = np.std(list(group_tpr.values())) if group_tpr else 0
        fpr_diff = np.std(list(group_fpr.values())) if group_fpr else 0

        return (tpr_diff + fpr_diff) / 2

    def _calculate_equal_opportunity(self, activations: np.ndarray,
                                     groups: np.ndarray,
                                     labels: np.ndarray) -> float:
        """Calculate equal opportunity metric."""
        predictions = np.mean(
            activations,
            axis=1) > np.median(
            np.mean(
                activations,
                axis=1))

        group_tpr = {}

        for group in np.unique(groups):
            group_mask = groups == group
            group_labels = labels[group_mask]
            group_preds = predictions[group_mask]

            if len(group_labels) == 0:
                continue

            # Calculate TPR
            tp = np.sum((group_labels == 1) & (group_preds == 1))
            fn = np.sum((group_labels == 1) & (group_preds == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            group_tpr[group] = tpr

        # Calculate difference in TPR across groups
        return np.std(list(group_tpr.values())) if group_tpr else 0

    def _bootstrap_confidence_interval(self, activations: np.ndarray,
                                       groups: np.ndarray,
                                       labels: Optional[np.ndarray],
                                       n_bootstrap: int = 100) -> Tuple[float, float]:
        """Calculate confidence interval using bootstrap."""
        bootstrap_scores = []

        for _ in range(n_bootstrap):
            # Sample with replacement
            n_samples = len(activations)
            indices = np.random.choice(n_samples, n_samples, replace=True)

            boot_activations = activations[indices]
            boot_groups = groups[indices]
            boot_labels = labels[indices] if labels is not None else None

            # Calculate bias metrics for bootstrap sample
            try:
                bias_result = self.detect_bias(
                    boot_activations, boot_groups, boot_labels)
                bootstrap_scores.append(bias_result.bias_score)
            except BaseException:
                continue

        if bootstrap_scores:
            return np.percentile(bootstrap_scores, [5, 95])
        else:
            return 0.0, 0.0


class FairnessAnalyzer:
    """Comprehensive fairness analysis for neural networks."""

    def __init__(self, bias_detector: Optional[BiasDetector] = None):
        """Initialize fairness analyzer.

        Args:
            bias_detector: Custom bias detector (defaults to ActivationBiasDetector)
        """
        self.bias_detector = bias_detector or ActivationBiasDetector()

    def analyze_fairness(self, model, tokenizer, texts: List[str],
                         groups: List[str],
                         labels: Optional[List[int]] = None,
                         layers: Optional[List[str]] = None) -> FairnessResult:
        """Comprehensive fairness analysis.

        Args:
            model: Neural network model
            tokenizer: Model tokenizer
            texts: Input texts
            groups: Group assignments for each text
            labels: Optional labels for supervised analysis
            layers: Layers to analyze (defaults to all)

        Returns:
            FairnessResult with comprehensive fairness assessment
        """
        logger.info("Starting comprehensive fairness analysis...")

        # Extract activations
        activations = self._extract_activations(
            model, tokenizer, texts, layers
        )

        # Convert inputs to numpy arrays
        groups_array = np.array(groups)
        labels_array = np.array(labels) if labels else None

        # Detect bias across layers
        layer_bias_scores = {}
        overall_metrics = None

        for layer_name, layer_activations in activations.items():
            logger.info(f"Analyzing bias in layer: {layer_name}")

            bias_metrics = self.bias_detector.detect_bias(
                layer_activations, groups_array, labels_array
            )

            layer_bias_scores[layer_name] = bias_metrics.bias_score

            # Use the worst layer for overall metrics
            if (overall_metrics is None or
                    bias_metrics.bias_score > overall_metrics.bias_score):
                overall_metrics = bias_metrics

        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_metrics, layer_bias_scores
        )

        # Generate warnings
        warnings_list = self._generate_warnings(
            overall_metrics, layer_bias_scores
        )

        return FairnessResult(
            overall_bias_score=overall_metrics.bias_score,
            group_bias_scores=layer_bias_scores,
            fairness_metrics=overall_metrics,
            recommendations=recommendations,
            warnings=warnings_list
        )

    def _extract_activations(self,
                             model,
                             tokenizer,
                             texts: List[str],
                             layers: Optional[List[str]] = None) -> Dict[str,
                                                                         np.ndarray]:
        """Extract activations from specified layers."""
        try:
            from ..analysis.activation_extractor import ActivationExtractor

            # extractor = ActivationExtractor(model, tokenizer)

            # Get default layers if none specified
            if layers is None:
                if hasattr(model.config, 'num_hidden_layers'):
                    num_layers = model.config.num_hidden_layers
                    layers = [f"layer_{i}" for i in range(min(3, num_layers))]
                else:
                    layers = ["layer_0"]

            activations = {}

            for layer in layers:
                layer_activations = []

                for text in texts:
                    # Tokenize
                    inputs = tokenizer(text, return_tensors='pt',
                                       padding=True, truncation=True, max_length=512)

                    # Extract activations
                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True)

                        # Get layer index
                        if layer.startswith('layer_'):
                            layer_idx = int(layer.split('_')[1])
                            if layer_idx < len(outputs.hidden_states):
                                hidden_state = outputs.hidden_states[layer_idx]
                                # Average over sequence length
                                activation = hidden_state.mean(dim=1).cpu().numpy()
                                layer_activations.append(activation[0])

                if layer_activations:
                    activations[layer] = np.array(layer_activations)

            return activations

        except ImportError:
            logger.warning(
                "Could not import ActivationExtractor, using mock activations")
            # Return mock activations for testing
            return {
                'layer_0': np.random.randn(len(texts), 768),
                'layer_1': np.random.randn(len(texts), 768)
            }

    def _generate_recommendations(self, metrics: BiasMetrics,
                                  layer_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on bias analysis."""
        recommendations = []

        if metrics.bias_score > 0.3:
            recommendations.append(
                "High bias detected. Consider retraining with more balanced data."
            )

        if metrics.demographic_parity > 0.2:
            recommendations.append(
                "Poor demographic parity. Review data collection and sampling methods."
            )

        if metrics.affected_groups:
            recommendations.append(
                f"Special attention needed for groups: {
                    ', '.join(
                        metrics.affected_groups)}")

        # Layer-specific recommendations
        worst_layer = max(layer_scores.items(), key=lambda x: x[1])
        if worst_layer[1] > 0.4:
            recommendations.append(
                f"Layer {
                    worst_layer[0]} shows highest bias. Consider layer-specific interventions.")

        if not recommendations:
            recommendations.append("Model shows acceptable fairness levels.")

        return recommendations

    def _generate_warnings(self, metrics: BiasMetrics,
                           layer_scores: Dict[str, float]) -> List[str]:
        """Generate warnings based on bias analysis."""
        warnings_list = []

        if metrics.bias_score > 0.5:
            warnings_list.append(
                "CRITICAL: Very high bias detected - model may be unsuitable for deployment")

        if metrics.confidence_interval[1] - metrics.confidence_interval[0] > 0.3:
            warnings_list.append(
                "High uncertainty in bias estimates - collect more data")

        if len(metrics.affected_groups) > 0:
            warnings_list.append(
                f"Bias detected against groups: {
                    ', '.join(
                        metrics.affected_groups)}")

        return warnings_list


class ModelCardGenerator:
    """Generate model cards for transparency and documentation."""

    def __init__(self):
        self.card_template = {
            'model_name': '',
            'model_version': '1.0.0',
            'created_date': datetime.now().isoformat(),
            'description': '',
            'intended_use': '',
            'limitations': [],
            'bias_assessment': {},
            'performance_metrics': {},
            'training_data': {},
            'evaluation_data': {},
            'ethical_considerations': [],
            'recommendations': []
        }

    def generate_card(self, model_name: str,
                      fairness_result: FairnessResult,
                      performance_metrics: Optional[Dict[str, float]] = None,
                      **kwargs) -> ModelCard:
        """Generate a model card with bias assessment.

        Args:
            model_name: Name of the model
            fairness_result: Results from fairness analysis
            performance_metrics: Model performance metrics
            **kwargs: Additional card fields

        Returns:
            ModelCard object
        """
        card_data = self.card_template.copy()
        card_data['model_name'] = model_name

        # Add fairness assessment
        card_data['bias_assessment'] = {
            'overall_bias_score': fairness_result.overall_bias_score,
            'fairness_metrics': asdict(fairness_result.fairness_metrics),
            'analysis_date': datetime.now().isoformat(),
            'methodology': 'Activation-based bias detection'
        }

        # Add performance metrics
        if performance_metrics:
            card_data['performance_metrics'] = performance_metrics

        # Add ethical considerations
        card_data['ethical_considerations'] = [
            'Bias assessment conducted using activation analysis',
            'Results should be interpreted in context of use case',
            'Regular monitoring recommended for production deployment'
        ]

        # Add recommendations from fairness analysis
        card_data['recommendations'] = fairness_result.recommendations

        # Add any additional fields
        card_data.update(kwargs)

        return ModelCard(**card_data)

    def save_card(self, card: ModelCard, output_path: Path):
        """Save model card to file.

        Args:
            card: ModelCard object
            output_path: Path to save the card
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(asdict(card), f, indent=2)

        logger.info(f"Model card saved to {output_path}")


class AuditTrail:
    """Audit trail for tracking analysis steps and decisions."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log = []

    def log_step(self, step_name: str, description: str,
                 data: Optional[Dict[str, Any]] = None):
        """Log an analysis step.

        Args:
            step_name: Name of the analysis step
            description: Description of what was done
            data: Optional data associated with the step
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'description': description,
            'data': data or {}
        }

        self.audit_log.append(entry)
        logger.info(f"Audit: {step_name} - {description}")

    def save_audit_log(self, filename: str = "audit_trail.json"):
        """Save audit trail to file."""
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(self.audit_log, f, indent=2)

        logger.info(f"Audit trail saved to {output_path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of audit trail."""
        return {
            'total_steps': len(self.audit_log),
            'start_time': self.audit_log[0]['timestamp'] if self.audit_log else None,
            'end_time': self.audit_log[-1]['timestamp'] if self.audit_log else None,
            'steps': [entry['step'] for entry in self.audit_log]
        }


def main():
    """Example usage of ethics and bias analysis."""
    # Example texts and groups
    texts = [
        "The software engineer solved the problem",
        "The nurse helped the patient",
        "The CEO made the decision",
        "The teacher explained the concept"
    ]

    groups = ["male", "female", "male", "female"]
    labels = [1, 1, 1, 1]  # All positive examples

    # Initialize components
    analyzer = FairnessAnalyzer()
    card_generator = ModelCardGenerator()
    audit_trail = AuditTrail(Path("./audit_logs"))

    # Start audit trail
    audit_trail.log_step("initialization", "Started bias analysis", {
        "num_texts": len(texts),
        "num_groups": len(set(groups))
    })

    print("Ethics and Bias Analysis Module")
    print("This module would normally require a real model and tokenizer")
    print("Running with mock data for demonstration...")

    # Mock analysis (would normally use real model)
    mock_result = FairnessResult(
        overall_bias_score=0.15,
        group_bias_scores={"layer_0": 0.12, "layer_1": 0.18},
        fairness_metrics=BiasMetrics(
            demographic_parity=0.1,
            equalized_odds=0.08,
            equal_opportunity=0.09,
            statistical_parity=0.1,
            bias_score=0.15,
            affected_groups=[],
            confidence_interval=(0.05, 0.25)
        ),
        recommendations=["Model shows acceptable fairness levels."],
        warnings=[]
    )

    audit_trail.log_step("bias_analysis", "Completed fairness analysis", {
        "bias_score": mock_result.overall_bias_score
    })

    # Generate model card
    card = card_generator.generate_card(
        model_name="example_model",
        fairness_result=mock_result,
        description="Example model for demonstration"
    )

    audit_trail.log_step("model_card", "Generated model card")

    # Save results
    card_generator.save_card(card, Path("./model_card.json"))
    audit_trail.save_audit_log()

    print(f"Bias analysis complete. Overall bias score: {
          mock_result.overall_bias_score:.3f}")
    print(f"Recommendations: {mock_result.recommendations}")


if __name__ == "__main__":
    main()
