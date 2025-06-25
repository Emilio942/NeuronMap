"""Performance Metrics Collection and Analysis System for NeuronMap.

This module implements comprehensive performance analytics with detailed metric collection,
trend analysis, and performance regression detection according to roadmap section 2.3.
"""

import time
import threading
import logging
import json
import sqlite3
import statistics
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
from contextlib import contextmanager
import pickle
import os
from pathlib import Path

from .error_handling import NeuronMapException, ResourceError
from .robust_decorators import robust_execution

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    THROUGHPUT = "throughput"
    ACCURACY = "accuracy"
    ERROR_RATE = "error_rate"
    RESOURCE_UTILIZATION = "resource_utilization"


class AggregationType(Enum):
    """Types of metric aggregation."""
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    P95 = "p95"
    P99 = "p99"
    SUM = "sum"
    COUNT = "count"


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    operation: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'value': self.value,
            'metric_type': self.metric_type.value,
            'timestamp': self.timestamp,
            'operation': self.operation,
            'metadata': self.metadata,
            'tags': self.tags
        }


@dataclass
class AggregatedMetric:
    """Aggregated performance metric over a time period."""
    name: str
    operation: str
    aggregation_type: AggregationType
    value: float
    count: int
    start_time: float
    end_time: float
    percentiles: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class PerformanceTrend:
    """Performance trend analysis result."""
    metric_name: str
    operation: str
    trend_direction: str  # 'improving', 'degrading', 'stable'
    trend_strength: float  # 0-1, strength of trend
    change_percent: float
    regression_detected: bool
    confidence: float  # 0-1, confidence in trend analysis
    time_window_hours: float
    data_points: int


class PerformanceProfiler:
    """Context manager for profiling operations."""

    def __init__(self, collector: 'PerformanceCollector', operation_name: str,
                 metadata: Optional[Dict[str, Any]] = None,
                 tags: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.operation_name = operation_name
        self.metadata = metadata or {}
        self.tags = tags or {}
        self.start_time = None
        self.start_memory = None

    def __enter__(self):
        """Start profiling."""
        self.start_time = time.perf_counter()
        try:
            import psutil
            process = psutil.Process()
            self.start_memory = process.memory_info().rss
        except ImportError:
            self.start_memory = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End profiling and record metrics."""
        end_time = time.perf_counter()
        duration = end_time - self.start_time

        # Record execution time
        self.collector.record_metric(
            name=f"{self.operation_name}_duration",
            value=duration,
            metric_type=MetricType.EXECUTION_TIME,
            operation=self.operation_name,
            metadata=self.metadata,
            tags=self.tags
        )

        # Record memory usage if available
        if self.start_memory is not None:
            try:
                import psutil
                process = psutil.Process()
                end_memory = process.memory_info().rss
                memory_delta = end_memory - self.start_memory

                self.collector.record_metric(
                    name=f"{self.operation_name}_memory_delta",
                    value=memory_delta,
                    metric_type=MetricType.MEMORY_USAGE,
                    operation=self.operation_name,
                    metadata=self.metadata,
                    tags=self.tags
                )
            except ImportError:
                pass

        # Record error if exception occurred
        if exc_type is not None:
            self.collector.record_metric(
                name=f"{self.operation_name}_error",
                value=1.0,
                metric_type=MetricType.ERROR_RATE,
                operation=self.operation_name,
                metadata={**self.metadata, 'error_type': exc_type.__name__},
                tags=self.tags
            )
        else:
            self.collector.record_metric(
                name=f"{self.operation_name}_success",
                value=1.0,
                metric_type=MetricType.ERROR_RATE,
                operation=self.operation_name,
                metadata=self.metadata,
                tags=self.tags
            )


class MetricsStorage:
    """Storage backend for performance metrics."""

    def __init__(self, storage_path: str = "metrics.db"):
        self.storage_path = storage_path
        self.connection = None
        self._initialize_database()

    def _initialize_database(self):
        """Initialize SQLite database for metrics storage."""
        self.connection = sqlite3.connect(self.storage_path, check_same_thread=False)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                metric_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                operation TEXT NOT NULL,
                metadata TEXT,
                tags TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)
        """)

        self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_operation ON metrics(operation)
        """)

        self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name)
        """)

        self.connection.commit()

    def store_metric(self, metric: PerformanceMetric):
        """Store a performance metric."""
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO metrics (name, value, metric_type, timestamp, operation, metadata, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            metric.name,
            metric.value,
            metric.metric_type.value,
            metric.timestamp,
            metric.operation,
            json.dumps(metric.metadata),
            json.dumps(metric.tags)
        ))
        self.connection.commit()

    def get_metrics(self,
                   operation: Optional[str] = None,
                   metric_name: Optional[str] = None,
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None,
                   limit: Optional[int] = None) -> List[PerformanceMetric]:
        """Retrieve metrics with optional filtering."""
        query = "SELECT name, value, metric_type, timestamp, operation, metadata, tags FROM metrics WHERE 1=1"
        params = []

        if operation:
            query += " AND operation = ?"
            params.append(operation)

        if metric_name:
            query += " AND name = ?"
            params.append(metric_name)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = self.connection.cursor()
        cursor.execute(query, params)

        metrics = []
        for row in cursor.fetchall():
            name, value, metric_type, timestamp, operation, metadata_json, tags_json = row

            metadata = json.loads(metadata_json) if metadata_json else {}
            tags = json.loads(tags_json) if tags_json else {}

            metrics.append(PerformanceMetric(
                name=name,
                value=value,
                metric_type=MetricType(metric_type),
                timestamp=timestamp,
                operation=operation,
                metadata=metadata,
                tags=tags
            ))

        return metrics

    def cleanup_old_metrics(self, days_to_keep: int = 30):
        """Clean up metrics older than specified days."""
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)

        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff_time,))
        deleted_count = cursor.rowcount
        self.connection.commit()

        logger.info(f"Cleaned up {deleted_count} old metrics (older than {days_to_keep} days)")
        return deleted_count

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()


class TrendAnalyzer:
    """Analyzes performance trends and detects regressions."""

    def __init__(self, min_data_points: int = 10, regression_threshold: float = 0.2):
        self.min_data_points = min_data_points
        self.regression_threshold = regression_threshold  # 20% degradation threshold

    def analyze_trend(self, metrics: List[PerformanceMetric],
                     time_window_hours: float = 24.0) -> Optional[PerformanceTrend]:
        """Analyze performance trend for a list of metrics."""
        if len(metrics) < self.min_data_points:
            return None

        # Sort metrics by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)

        # Filter to time window
        current_time = time.time()
        cutoff_time = current_time - (time_window_hours * 3600)
        recent_metrics = [m for m in sorted_metrics if m.timestamp >= cutoff_time]

        if len(recent_metrics) < self.min_data_points:
            return None

        # Extract values and timestamps
        values = [m.value for m in recent_metrics]
        timestamps = [m.timestamp for m in recent_metrics]

        # Calculate linear regression
        trend_direction, trend_strength, change_percent = self._calculate_linear_trend(
            timestamps, values
        )

        # Detect regression
        regression_detected = self._detect_regression(values, trend_direction, change_percent)

        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(values, timestamps)

        return PerformanceTrend(
            metric_name=recent_metrics[0].name,
            operation=recent_metrics[0].operation,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            change_percent=change_percent,
            regression_detected=regression_detected,
            confidence=confidence,
            time_window_hours=time_window_hours,
            data_points=len(recent_metrics)
        )

    def _calculate_linear_trend(self, timestamps: List[float],
                               values: List[float]) -> Tuple[str, float, float]:
        """Calculate linear trend using least squares regression."""
        n = len(timestamps)

        # Normalize timestamps to start from 0
        min_time = min(timestamps)
        x = [(t - min_time) for t in timestamps]
        y = values

        # Calculate slope using least squares
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable", 0.0, 0.0

        slope = numerator / denominator

        # Calculate trend strength (R-squared)
        y_pred = [y_mean + slope * (x[i] - x_mean) for i in range(n)]
        ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Calculate percentage change
        if y_mean > 0:
            total_change = (values[-1] - values[0]) / values[0] * 100
        else:
            total_change = 0

        # Determine trend direction
        if abs(slope) < 1e-6:  # Very small slope
            trend_direction = "stable"
        elif slope > 0:
            # For metrics like execution time, increasing is degrading
            # For metrics like throughput, increasing is improving
            # We'll assume execution time/error rate increasing is bad
            if any(keyword in values[0] for keyword in ['time', 'duration', 'error', 'latency']):
                trend_direction = "degrading"
            else:
                trend_direction = "improving"
        else:
            if any(keyword in values[0] for keyword in ['time', 'duration', 'error', 'latency']):
                trend_direction = "improving"
            else:
                trend_direction = "degrading"

        return trend_direction, abs(r_squared), total_change

    def _detect_regression(self, values: List[float],
                          trend_direction: str, change_percent: float) -> bool:
        """Detect if there's a performance regression."""
        if trend_direction == "degrading" and abs(change_percent) > (self.regression_threshold * 100):
            return True

        # Also check for sudden spikes in recent data
        if len(values) >= 5:
            recent_avg = statistics.mean(values[-3:])
            baseline_avg = statistics.mean(values[:-3])

            if baseline_avg > 0:
                spike_percent = (recent_avg - baseline_avg) / baseline_avg * 100
                if spike_percent > (self.regression_threshold * 100):
                    return True

        return False

    def _calculate_confidence(self, values: List[float], timestamps: List[float]) -> float:
        """Calculate confidence in trend analysis."""
        confidence = 0.5  # Base confidence

        # More data points increase confidence
        data_point_factor = min(len(values) / 50, 1.0) * 0.3
        confidence += data_point_factor

        # Regular intervals increase confidence
        if len(timestamps) > 1:
            intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            interval_variance = statistics.variance(intervals) if len(intervals) > 1 else 0
            regular_interval_factor = max(0, (1 - interval_variance / statistics.mean(intervals)) * 0.2)
            confidence += regular_interval_factor

        # Low variance in values decreases confidence (might be synthetic data)
        if len(values) > 1:
            value_variance = statistics.variance(values)
            value_mean = statistics.mean(values)
            if value_mean > 0:
                cv = (value_variance ** 0.5) / value_mean  # Coefficient of variation
                variance_factor = min(cv, 0.5) * 0.2  # Some variance is good
                confidence += variance_factor

        return min(confidence, 1.0)


class PerformanceCollector:
    """Main performance metrics collection system."""

    def __init__(self,
                 storage_path: str = "performance_metrics.db",
                 max_memory_metrics: int = 10000,
                 auto_aggregation_interval: float = 300.0):  # 5 minutes
        """Initialize performance collector.

        Args:
            storage_path: Path to SQLite database for persistent storage
            max_memory_metrics: Maximum metrics to keep in memory
            auto_aggregation_interval: Interval for automatic aggregation in seconds
        """
        self.storage = MetricsStorage(storage_path)
        self.max_memory_metrics = max_memory_metrics
        self.auto_aggregation_interval = auto_aggregation_interval

        # In-memory metrics for fast access
        self.memory_metrics: deque = deque(maxlen=max_memory_metrics)
        self.trend_analyzer = TrendAnalyzer()

        # Threading for background tasks
        self.aggregation_thread: Optional[threading.Thread] = None
        self.stop_aggregation = threading.Event()
        self.lock = threading.Lock()

        # Aggregation cache
        self.aggregation_cache: Dict[str, List[AggregatedMetric]] = defaultdict(list)

        logger.info("Performance collector initialized")

    def record_metric(self,
                     name: str,
                     value: float,
                     metric_type: MetricType,
                     operation: str,
                     metadata: Optional[Dict[str, Any]] = None,
                     tags: Optional[Dict[str, str]] = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=time.time(),
            operation=operation,
            metadata=metadata or {},
            tags=tags or {}
        )

        # Store in memory
        with self.lock:
            self.memory_metrics.append(metric)

        # Store persistently
        try:
            self.storage.store_metric(metric)
        except Exception as e:
            logger.error(f"Error storing metric: {e}")

    @contextmanager
    def measure_operation(self, operation_name: str,
                         metadata: Optional[Dict[str, Any]] = None,
                         tags: Optional[Dict[str, str]] = None):
        """Context manager to measure operation performance."""
        profiler = PerformanceProfiler(self, operation_name, metadata, tags)
        with profiler:
            yield profiler

    def get_recent_metrics(self,
                          operation: Optional[str] = None,
                          metric_name: Optional[str] = None,
                          hours: float = 1.0) -> List[PerformanceMetric]:
        """Get recent metrics from memory and storage."""
        start_time = time.time() - (hours * 3600)

        # Try memory first
        with self.lock:
            memory_results = []
            for metric in self.memory_metrics:
                if metric.timestamp >= start_time:
                    if operation is None or metric.operation == operation:
                        if metric_name is None or metric.name == metric_name:
                            memory_results.append(metric)

        # If we need more data, query storage
        if len(memory_results) < 100:  # Arbitrary threshold
            storage_results = self.storage.get_metrics(
                operation=operation,
                metric_name=metric_name,
                start_time=start_time,
                limit=1000
            )

            # Combine and deduplicate
            all_metrics = memory_results + storage_results
            seen_timestamps = set()
            unique_metrics = []

            for metric in sorted(all_metrics, key=lambda m: m.timestamp, reverse=True):
                if metric.timestamp not in seen_timestamps:
                    seen_timestamps.add(metric.timestamp)
                    unique_metrics.append(metric)

            return unique_metrics

        return memory_results

    def aggregate_metrics(self,
                         operation: str,
                         metric_name: str,
                         aggregation_type: AggregationType,
                         time_window_hours: float = 1.0) -> Optional[AggregatedMetric]:
        """Aggregate metrics over a time window."""
        end_time = time.time()
        start_time = end_time - (time_window_hours * 3600)

        metrics = self.storage.get_metrics(
            operation=operation,
            metric_name=metric_name,
            start_time=start_time,
            end_time=end_time
        )

        if not metrics:
            return None

        values = [m.value for m in metrics]

        if aggregation_type == AggregationType.MEAN:
            aggregated_value = statistics.mean(values)
        elif aggregation_type == AggregationType.MEDIAN:
            aggregated_value = statistics.median(values)
        elif aggregation_type == AggregationType.MIN:
            aggregated_value = min(values)
        elif aggregation_type == AggregationType.MAX:
            aggregated_value = max(values)
        elif aggregation_type == AggregationType.SUM:
            aggregated_value = sum(values)
        elif aggregation_type == AggregationType.COUNT:
            aggregated_value = len(values)
        elif aggregation_type == AggregationType.P95:
            aggregated_value = self._calculate_percentile(values, 95)
        elif aggregation_type == AggregationType.P99:
            aggregated_value = self._calculate_percentile(values, 99)
        else:
            aggregated_value = statistics.mean(values)

        # Calculate percentiles
        percentiles = {
            'p50': self._calculate_percentile(values, 50),
            'p95': self._calculate_percentile(values, 95),
            'p99': self._calculate_percentile(values, 99)
        }

        return AggregatedMetric(
            name=metric_name,
            operation=operation,
            aggregation_type=aggregation_type,
            value=aggregated_value,
            count=len(values),
            start_time=start_time,
            end_time=end_time,
            percentiles=percentiles
        )

    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (percentile / 100)
        f = int(k)
        c = k - f

        if f + 1 < len(sorted_values):
            return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
        else:
            return sorted_values[f]

    def analyze_performance_trends(self,
                                  operation: Optional[str] = None,
                                  time_window_hours: float = 24.0) -> List[PerformanceTrend]:
        """Analyze performance trends for operations."""
        trends = []

        # Get unique operation/metric combinations
        recent_time = time.time() - (time_window_hours * 3600)
        all_metrics = self.storage.get_metrics(
            operation=operation,
            start_time=recent_time
        )

        # Group by operation and metric name
        grouped_metrics = defaultdict(list)
        for metric in all_metrics:
            key = f"{metric.operation}_{metric.name}"
            grouped_metrics[key].append(metric)

        # Analyze trends for each group
        for key, metrics in grouped_metrics.items():
            trend = self.trend_analyzer.analyze_trend(metrics, time_window_hours)
            if trend:
                trends.append(trend)

        return trends

    def get_performance_summary(self, hours: float = 24.0) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        end_time = time.time()
        start_time = end_time - (hours * 3600)

        # Get all metrics in time window
        metrics = self.storage.get_metrics(start_time=start_time, end_time=end_time)

        if not metrics:
            return {
                'summary': 'No metrics available',
                'time_window_hours': hours,
                'total_metrics': 0
            }

        # Group by operation
        operation_stats = defaultdict(lambda: {
            'total_metrics': 0,
            'operations': set(),
            'avg_duration': 0,
            'error_rate': 0,
            'total_errors': 0,
            'total_operations': 0
        })

        for metric in metrics:
            op_stats = operation_stats[metric.operation]
            op_stats['total_metrics'] += 1
            op_stats['operations'].add(metric.name)

            if metric.metric_type == MetricType.EXECUTION_TIME:
                if 'durations' not in op_stats:
                    op_stats['durations'] = []
                op_stats['durations'].append(metric.value)

            elif metric.metric_type == MetricType.ERROR_RATE:
                if 'error' in metric.name:
                    op_stats['total_errors'] += metric.value
                op_stats['total_operations'] += metric.value

        # Calculate aggregated stats
        summary = {}
        for operation, stats in operation_stats.items():
            if 'durations' in stats:
                stats['avg_duration'] = statistics.mean(stats['durations'])
                stats['p95_duration'] = self._calculate_percentile(stats['durations'], 95)

            if stats['total_operations'] > 0:
                stats['error_rate'] = stats['total_errors'] / stats['total_operations']

            stats['operations'] = list(stats['operations'])
            summary[operation] = stats

        # Analyze trends
        trends = self.analyze_performance_trends(time_window_hours=hours)
        regression_count = sum(1 for trend in trends if trend.regression_detected)

        return {
            'summary': f"Performance analysis for {hours} hours",
            'time_window_hours': hours,
            'total_metrics': len(metrics),
            'unique_operations': len(operation_stats),
            'operation_stats': summary,
            'trends_analyzed': len(trends),
            'regressions_detected': regression_count,
            'trends': [trend.__dict__ for trend in trends]
        }

    def start_auto_aggregation(self):
        """Start automatic metric aggregation in background."""
        if self.aggregation_thread and self.aggregation_thread.is_alive():
            logger.warning("Auto-aggregation already started")
            return

        self.stop_aggregation.clear()
        self.aggregation_thread = threading.Thread(target=self._aggregation_loop, daemon=True)
        self.aggregation_thread.start()
        logger.info("Auto-aggregation started")

    def stop_auto_aggregation(self):
        """Stop automatic aggregation."""
        self.stop_aggregation.set()
        if self.aggregation_thread:
            self.aggregation_thread.join(timeout=5.0)
        logger.info("Auto-aggregation stopped")

    def _aggregation_loop(self):
        """Background aggregation loop."""
        while not self.stop_aggregation.is_set():
            try:
                # Perform periodic aggregation and cleanup
                self.storage.cleanup_old_metrics(days_to_keep=30)

                # Wait for next cycle
                self.stop_aggregation.wait(self.auto_aggregation_interval)

            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")
                self.stop_aggregation.wait(self.auto_aggregation_interval)

    def export_metrics(self, filename: str, format: str = 'json', hours: float = 24.0):
        """Export metrics to file."""
        start_time = time.time() - (hours * 3600)
        metrics = self.storage.get_metrics(start_time=start_time)

        if format.lower() == 'json':
            data = [metric.to_dict() for metric in metrics]
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported {len(metrics)} metrics to {filename}")

    def close(self):
        """Close collector and cleanup resources."""
        self.stop_auto_aggregation()
        self.storage.close()


# Global collector instance
performance_collector = PerformanceCollector()

def record_metric(name: str, value: float, metric_type: MetricType,
                 operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Record a performance metric."""
    performance_collector.record_metric(name, value, metric_type, operation, metadata)

def measure_operation(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager to measure operation performance."""
    return performance_collector.measure_operation(operation_name, metadata)

def get_performance_summary(hours: float = 24.0):
    """Get performance summary."""
    return performance_collector.get_performance_summary(hours)

def analyze_trends(operation: Optional[str] = None, hours: float = 24.0):
    """Analyze performance trends."""
    return performance_collector.analyze_performance_trends(operation, hours)
