"""Quality management system for NeuronMap analysis."""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)


@dataclass
class QualityMetric:
    """Represents a quality metric with validation rules."""
    name: str
    description: str
    threshold: float
    operator: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
    weight: float = 1.0
    category: str = "general"

    def validate(self, value: float) -> bool:
        """Validate if value meets the quality threshold."""
        if self.operator == 'gt':
            return value > self.threshold
        elif self.operator == 'gte':
            return value >= self.threshold
        elif self.operator == 'lt':
            return value < self.threshold
        elif self.operator == 'lte':
            return value <= self.threshold
        elif self.operator == 'eq':
            return abs(value - self.threshold) < 1e-6
        else:
            raise ValueError(f"Unknown operator: {self.operator}")


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    timestamp: datetime
    overall_score: float
    passed_metrics: List[str]
    failed_metrics: List[str]
    metric_values: Dict[str, float]
    recommendations: List[str]
    warnings: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_score': self.overall_score,
            'passed_metrics': self.passed_metrics,
            'failed_metrics': self.failed_metrics,
            'metric_values': self.metric_values,
            'recommendations': self.recommendations,
            'warnings': self.warnings,
            'metadata': self.metadata
        }


class QualityManager:
    """
    Manages quality assessment for neural network analysis.

    This class provides comprehensive quality monitoring, validation,
    and reporting capabilities for NeuronMap analysis results.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize QualityManager.

        Args:
            config: Configuration dictionary with quality settings
        """
        self.config = config or {}
        self.metrics: Dict[str, QualityMetric] = {}
        self.history: List[QualityReport] = []
        self.active_monitors: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

        # Initialize default metrics
        self._initialize_default_metrics()

        logger.info("QualityManager initialized")

    def _initialize_default_metrics(self):
        """Initialize default quality metrics."""
        default_metrics = [
            QualityMetric(
                name="activation_variance",
                description="Minimum activation variance for meaningful analysis",
                threshold=0.01,
                operator="gte",
                category="activation"
            ),
            QualityMetric(
                name="gradient_norm",
                description="Maximum gradient norm to detect exploding gradients",
                threshold=10.0,
                operator="lte",
                category="training"
            ),
            QualityMetric(
                name="attention_entropy",
                description="Minimum attention entropy for interpretability",
                threshold=0.1,
                operator="gte",
                category="attention"
            ),
            QualityMetric(
                name="layer_utilization",
                description="Minimum layer utilization percentage",
                threshold=0.05,
                operator="gte",
                category="architecture"
            ),
            QualityMetric(
                name="computation_time",
                description="Maximum computation time in seconds",
                threshold=300.0,
                operator="lte",
                category="performance"
            )
        ]

        for metric in default_metrics:
            self.add_metric(metric)

    def add_metric(self, metric: QualityMetric):
        """Add a quality metric to the manager."""
        with self._lock:
            self.metrics[metric.name] = metric
            logger.debug(f"Added quality metric: {metric.name}")

    def remove_metric(self, metric_name: str):
        """Remove a quality metric."""
        with self._lock:
            if metric_name in self.metrics:
                del self.metrics[metric_name]
                logger.debug(f"Removed quality metric: {metric_name}")

    def assess_quality(self,
                      data: Dict[str, Any],
                      custom_metrics: Optional[List[QualityMetric]] = None) -> QualityReport:
        """
        Assess the quality of analysis results.

        Args:
            data: Analysis data to assess
            custom_metrics: Additional metrics for this assessment

        Returns:
            QualityReport with assessment results
        """
        start_time = time.time()

        # Combine default and custom metrics
        all_metrics = dict(self.metrics)
        if custom_metrics:
            for metric in custom_metrics:
                all_metrics[metric.name] = metric

        passed_metrics = []
        failed_metrics = []
        metric_values = {}
        warnings_list = []
        recommendations = []

        # Evaluate each metric
        for metric_name, metric in all_metrics.items():
            try:
                value = self._extract_metric_value(data, metric_name)
                metric_values[metric_name] = value

                if metric.validate(value):
                    passed_metrics.append(metric_name)
                else:
                    failed_metrics.append(metric_name)
                    recommendations.append(
                        f"Improve {metric_name}: current={value:.4f}, "
                        f"threshold={metric.threshold} ({metric.operator})"
                    )

            except Exception as e:
                logger.warning(f"Failed to evaluate metric {metric_name}: {e}")
                warnings_list.append(f"Metric evaluation failed: {metric_name}")

        # Calculate overall score
        if all_metrics:
            weights = sum(metric.weight for metric in all_metrics.values())
            weighted_score = sum(
                metric.weight for name, metric in all_metrics.items()
                if name in passed_metrics
            )
            overall_score = weighted_score / weights if weights > 0 else 0.0
        else:
            overall_score = 1.0

        # Create report
        report = QualityReport(
            timestamp=datetime.now(),
            overall_score=overall_score,
            passed_metrics=passed_metrics,
            failed_metrics=failed_metrics,
            metric_values=metric_values,
            recommendations=recommendations,
            warnings=warnings_list,
            metadata={
                'assessment_duration': time.time() - start_time,
                'total_metrics': len(all_metrics),
                'data_keys': list(data.keys())
            }
        )

        # Store in history
        with self._lock:
            self.history.append(report)
            # Keep only last 100 reports
            if len(self.history) > 100:
                self.history = self.history[-100:]

        logger.info(f"Quality assessment completed: score={overall_score:.3f}")
        return report

    def _extract_metric_value(self, data: Dict[str, Any], metric_name: str) -> float:
        """Extract metric value from analysis data."""
        # Default extraction logic - can be overridden
        if metric_name == "activation_variance":
            if 'activations' in data:
                activations = np.array(data['activations'])
                return float(np.var(activations))
            return 0.0

        elif metric_name == "gradient_norm":
            if 'gradients' in data:
                gradients = np.array(data['gradients'])
                return float(np.linalg.norm(gradients))
            return 0.0

        elif metric_name == "attention_entropy":
            if 'attention_weights' in data:
                attention = np.array(data['attention_weights'])
                # Calculate entropy
                attention = attention + 1e-10  # Avoid log(0)
                entropy = -np.sum(attention * np.log(attention), axis=-1)
                return float(np.mean(entropy))
            return 0.0

        elif metric_name == "layer_utilization":
            if 'layer_activations' in data:
                activations = np.array(data['layer_activations'])
                # Calculate percentage of active neurons
                active_ratio = np.mean(np.abs(activations) > 1e-6)
                return float(active_ratio)
            return 0.0

        elif metric_name == "computation_time":
            return data.get('computation_time', 0.0)

        # Try direct access
        if metric_name in data:
            return float(data[metric_name])

        # Try nested access
        for key, value in data.items():
            if isinstance(value, dict) and metric_name in value:
                return float(value[metric_name])

        raise ValueError(f"Metric {metric_name} not found in data")

    def get_quality_trends(self,
                          metric_name: Optional[str] = None,
                          time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get quality trends over time.

        Args:
            metric_name: Specific metric to analyze (None for overall score)
            time_window: Time window to consider (None for all history)

        Returns:
            Dictionary with trend analysis
        """
        with self._lock:
            history = list(self.history)

        if not history:
            return {'trend': 'no_data', 'values': [], 'timestamps': []}

        # Filter by time window
        if time_window:
            cutoff = datetime.now() - time_window
            history = [r for r in history if r.timestamp >= cutoff]

        if not history:
            return {'trend': 'no_data', 'values': [], 'timestamps': []}

        # Extract values and timestamps
        if metric_name:
            values = [r.metric_values.get(metric_name, 0.0) for r in history]
        else:
            values = [r.overall_score for r in history]

        timestamps = [r.timestamp for r in history]

        # Calculate trend
        if len(values) < 2:
            trend = 'insufficient_data'
        else:
            # Simple linear trend
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            if slope > 0.01:
                trend = 'improving'
            elif slope < -0.01:
                trend = 'degrading'
            else:
                trend = 'stable'

        return {
            'trend': trend,
            'values': values,
            'timestamps': [t.isoformat() for t in timestamps],
            'latest_value': values[-1] if values else 0.0,
            'average_value': np.mean(values) if values else 0.0,
            'min_value': np.min(values) if values else 0.0,
            'max_value': np.max(values) if values else 0.0
        }

    def generate_recommendations(self, report: QualityReport) -> List[str]:
        """Generate actionable recommendations based on quality report."""
        recommendations = list(report.recommendations)

        # Add general recommendations based on patterns
        if report.overall_score < 0.5:
            recommendations.insert(0,
                "Overall quality is low. Consider reviewing data preprocessing and model configuration.")

        if len(report.failed_metrics) > len(report.passed_metrics):
            recommendations.append(
                "More metrics are failing than passing. Consider adjusting quality thresholds or improving analysis pipeline.")

        # Add specific recommendations for common failure patterns
        if 'activation_variance' in report.failed_metrics:
            recommendations.append(
                "Low activation variance detected. Check for dead neurons or inappropriate input scaling.")

        if 'gradient_norm' in report.failed_metrics:
            recommendations.append(
                "High gradient norm detected. Consider gradient clipping or learning rate reduction.")

        return recommendations

    def export_history(self, filepath: Union[str, Path]) -> None:
        """Export quality history to JSON file."""
        filepath = Path(filepath)

        with self._lock:
            history_data = [report.to_dict() for report in self.history]

        with open(filepath, 'w') as f:
            json.dump({
                'export_timestamp': datetime.now().isoformat(),
                'total_reports': len(history_data),
                'history': history_data
            }, f, indent=2)

        logger.info(f"Quality history exported to {filepath}")

    def import_history(self, filepath: Union[str, Path]) -> None:
        """Import quality history from JSON file."""
        filepath = Path(filepath)

        with open(filepath, 'r') as f:
            data = json.load(f)

        imported_reports = []
        for report_data in data.get('history', []):
            report = QualityReport(
                timestamp=datetime.fromisoformat(report_data['timestamp']),
                overall_score=report_data['overall_score'],
                passed_metrics=report_data['passed_metrics'],
                failed_metrics=report_data['failed_metrics'],
                metric_values=report_data['metric_values'],
                recommendations=report_data['recommendations'],
                warnings=report_data['warnings'],
                metadata=report_data.get('metadata', {})
            )
            imported_reports.append(report)

        with self._lock:
            self.history.extend(imported_reports)
            # Keep only last 100 reports
            if len(self.history) > 100:
                self.history = self.history[-100:]

        logger.info(f"Imported {len(imported_reports)} quality reports from {filepath}")

    def start_continuous_monitoring(self,
                                  data_source: Callable[[], Dict[str, Any]],
                                  interval: float = 60.0,
                                  name: str = "default") -> None:
        """Start continuous quality monitoring."""
        def monitor():
            while name in self.active_monitors:
                try:
                    data = data_source()
                    report = self.assess_quality(data)

                    # Log significant quality changes
                    if report.overall_score < 0.3:
                        logger.warning(f"Low quality detected: {report.overall_score:.3f}")

                    time.sleep(interval)

                except Exception as e:
                    logger.error(f"Error in continuous monitoring: {e}")
                    time.sleep(interval)

        if name in self.active_monitors:
            self.stop_continuous_monitoring(name)

        thread = threading.Thread(target=monitor, daemon=True)
        self.active_monitors[name] = thread
        thread.start()

        logger.info(f"Started continuous monitoring: {name}")

    def stop_continuous_monitoring(self, name: str = "default") -> None:
        """Stop continuous quality monitoring."""
        if name in self.active_monitors:
            del self.active_monitors[name]
            logger.info(f"Stopped continuous monitoring: {name}")

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for all quality assessments."""
        with self._lock:
            if not self.history:
                return {'status': 'no_data'}

            scores = [r.overall_score for r in self.history]
            recent_scores = scores[-10:] if len(scores) >= 10 else scores

            return {
                'total_assessments': len(self.history),
                'average_score': np.mean(scores),
                'recent_average_score': np.mean(recent_scores),
                'best_score': np.max(scores),
                'worst_score': np.min(scores),
                'score_std': np.std(scores),
                'assessment_timespan': (
                    (self.history[-1].timestamp - self.history[0].timestamp).total_seconds() / 3600
                    if len(self.history) > 1 else 0
                ),
                'active_monitors': list(self.active_monitors.keys())
            }


# Convenience functions
def create_quality_manager(config: Optional[Dict[str, Any]] = None) -> QualityManager:
    """Create a QualityManager instance with default configuration."""
    return QualityManager(config)


def assess_activation_quality(activations: np.ndarray,
                            layer_name: str = "unknown") -> QualityReport:
    """Quick quality assessment for activation data."""
    manager = QualityManager()

    data = {
        'activations': activations,
        'layer_name': layer_name,
        'activation_variance': np.var(activations),
        'layer_utilization': np.mean(np.abs(activations) > 1e-6)
    }

    return manager.assess_quality(data)


def assess_attention_quality(attention_weights: np.ndarray) -> QualityReport:
    """Quick quality assessment for attention weights."""
    manager = QualityManager()

    # Calculate attention entropy
    attention_safe = attention_weights + 1e-10
    entropy = -np.sum(attention_safe * np.log(attention_safe), axis=-1)

    data = {
        'attention_weights': attention_weights,
        'attention_entropy': np.mean(entropy),
        'attention_variance': np.var(attention_weights)
    }

    return manager.assess_quality(data)
