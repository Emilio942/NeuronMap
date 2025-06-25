"""Comprehensive System Resource Monitoring for NeuronMap.

This module provides comprehensive system resource monitoring with real-time metrics,
threshold-based alerting, and automatic resource optimization according to roadmap section 2.3.
"""

import time
import threading
import logging
import psutil
import platform
import socket
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import statistics
from contextlib import contextmanager
import warnings

from .error_handling import NeuronMapException, ResourceError
from .robust_decorators import robust_execution

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert levels for resource monitoring."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class CPUMetrics:
    """CPU utilization metrics."""
    usage_percent: float
    usage_per_core: List[float]
    load_average: Tuple[float, float, float]  # 1min, 5min, 15min
    frequency_current: float  # MHz
    frequency_max: float      # MHz
    context_switches: int
    interrupts: int
    temperature: Optional[float] = None


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""
    total: int           # bytes
    available: int       # bytes
    used: int           # bytes
    free: int           # bytes
    percent: float
    swap_total: int     # bytes
    swap_used: int      # bytes
    swap_free: int      # bytes
    swap_percent: float
    buffers: int        # bytes (Linux)
    cached: int         # bytes (Linux)

    @property
    def total_gb(self) -> float:
        return self.total / (1024**3)

    @property
    def used_gb(self) -> float:
        return self.used / (1024**3)

    @property
    def available_gb(self) -> float:
        return self.available / (1024**3)


@dataclass
class DiskMetrics:
    """Disk usage and I/O metrics."""
    total_space: int     # bytes
    used_space: int      # bytes
    free_space: int      # bytes
    usage_percent: float
    read_bytes: int
    write_bytes: int
    read_count: int
    write_count: int
    read_time: int       # ms
    write_time: int      # ms
    read_speed: float    # bytes/sec
    write_speed: float   # bytes/sec

    @property
    def total_gb(self) -> float:
        return self.total_space / (1024**3)

    @property
    def free_gb(self) -> float:
        return self.free_space / (1024**3)


@dataclass
class NetworkMetrics:
    """Network I/O metrics."""
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int
    errin: int
    errout: int
    dropin: int
    dropout: int
    send_speed: float    # bytes/sec
    recv_speed: float    # bytes/sec
    connections_active: int
    connections_listening: int


@dataclass
class ProcessMetrics:
    """Process-specific metrics."""
    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    memory_rss: int      # bytes
    memory_vms: int      # bytes
    num_threads: int
    num_fds: int         # file descriptors
    create_time: float
    status: str


@dataclass
class SystemMetrics:
    """Complete system metrics snapshot."""
    timestamp: float
    cpu: CPUMetrics
    memory: MemoryMetrics
    disk: DiskMetrics
    network: NetworkMetrics
    processes: List[ProcessMetrics] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ResourceThresholds:
    """Configurable thresholds for resource monitoring."""
    cpu_warning: float = 80.0      # %
    cpu_critical: float = 95.0     # %
    memory_warning: float = 80.0   # %
    memory_critical: float = 95.0  # %
    disk_warning: float = 85.0     # %
    disk_critical: float = 95.0    # %
    swap_warning: float = 50.0     # %
    swap_critical: float = 80.0    # %


class ResourceOptimizer:
    """Provides resource optimization recommendations."""

    def __init__(self):
        self.recommendations_cache = {}

    def analyze_cpu_usage(self, metrics: CPUMetrics, history: List[SystemMetrics]) -> List[str]:
        """Analyze CPU usage and provide optimization recommendations."""
        recommendations = []

        if metrics.usage_percent > 80:
            recommendations.append(
                "High CPU usage detected. Consider reducing batch size or enabling multiprocessing."
            )

        # Check for uneven core utilization
        if len(metrics.usage_per_core) > 1:
            core_variance = statistics.variance(metrics.usage_per_core)
            if core_variance > 400:  # High variance in core usage
                recommendations.append(
                    "Uneven CPU core utilization. Consider using ThreadPoolExecutor for better distribution."
                )

        # Check load average trends
        if len(history) > 5:
            recent_loads = [m.cpu.load_average[0] for m in history[-5:]]
            if all(load > psutil.cpu_count() * 0.8 for load in recent_loads):
                recommendations.append(
                    "Sustained high load average. Consider scaling horizontally or reducing concurrent operations."
                )

        return recommendations

    def analyze_memory_usage(self, metrics: MemoryMetrics, history: List[SystemMetrics]) -> List[str]:
        """Analyze memory usage and provide optimization recommendations."""
        recommendations = []

        if metrics.percent > 80:
            recommendations.append(
                "High memory usage detected. Consider reducing model size or implementing memory-efficient processing."
            )

        if metrics.swap_percent > 20:
            recommendations.append(
                "Swap usage detected. This will significantly slow down operations. Consider adding more RAM."
            )

        # Check for memory leaks
        if len(history) > 10:
            recent_usage = [m.memory.percent for m in history[-10:]]
            if all(usage > recent_usage[i-1] for i, usage in enumerate(recent_usage[1:], 1)):
                recommendations.append(
                    "Potential memory leak detected. Memory usage is continuously increasing."
                )

        return recommendations

    def analyze_disk_usage(self, metrics: DiskMetrics, history: List[SystemMetrics]) -> List[str]:
        """Analyze disk usage and provide optimization recommendations."""
        recommendations = []

        if metrics.usage_percent > 85:
            recommendations.append(
                "High disk usage detected. Consider cleaning up temporary files or adding storage."
            )

        # Check I/O patterns
        if metrics.read_speed + metrics.write_speed > 100 * 1024 * 1024:  # 100 MB/s
            recommendations.append(
                "High disk I/O detected. Consider using faster storage (SSD) or optimizing data access patterns."
            )

        return recommendations


class SystemResourceMonitor:
    """Comprehensive system resource monitoring with alerting and optimization."""

    def __init__(self,
                 thresholds: Optional[ResourceThresholds] = None,
                 history_size: int = 1000,
                 collection_interval: float = 1.0):
        """Initialize system resource monitor.

        Args:
            thresholds: Resource usage thresholds for alerting
            history_size: Number of metrics snapshots to keep in history
            collection_interval: Interval between metric collections in seconds
        """
        self.thresholds = thresholds or ResourceThresholds()
        self.history_size = history_size
        self.collection_interval = collection_interval

        self.metrics_history: deque = deque(maxlen=history_size)
        self.alert_callbacks: List[Callable] = []
        self.optimizer = ResourceOptimizer()

        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        self.lock = threading.Lock()

        # Cache for expensive operations
        self._last_disk_io = None
        self._last_network_io = None
        self._disk_io_cache_time = 0
        self._network_io_cache_time = 0

        logger.info("System resource monitor initialized")

    def add_alert_callback(self, callback: Callable[[AlertLevel, str, Dict[str, Any]], None]):
        """Add callback for resource alerts.

        Args:
            callback: Function to call when alerts are triggered
                     Signature: callback(level, message, context)
        """
        self.alert_callbacks.append(callback)

    def _add_to_history(self, metrics: Dict[str, Any]):
        """Add metrics to history."""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()

            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # Disk metrics
            disk_usage = psutil.disk_usage('/')

            # GPU metrics (if available)
            gpu_info = self._get_gpu_info()

            # Process metrics
            process_info = self._get_process_info()

            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'frequency': cpu_freq.current if cpu_freq else None
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used,
                    'percent': memory.percent
                },
                'swap': {
                    'total': swap.total,
                    'used': swap.used,
                    'percent': swap.percent
                },
                'disk': {
                    'total': disk_usage.total,
                    'used': disk_usage.used,
                    'free': disk_usage.free,
                    'percent': (disk_usage.used / disk_usage.total) * 100
                },
                'gpu': gpu_info,
                'process': process_info
            }

            return metrics

        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return self._get_fallback_metrics()

    def _get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get GPU information if available."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_info = []

                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    memory_total = props.total_memory

                    gpu_info.append({
                        'index': i,
                        'name': props.name,
                        'memory_allocated': memory_allocated,
                        'memory_reserved': memory_reserved,
                        'memory_total': memory_total,
                        'memory_percent': (memory_allocated / memory_total) * 100,
                        'compute_capability': f"{props.major}.{props.minor}"
                    })

                return {
                    'available': True,
                    'count': gpu_count,
                    'devices': gpu_info
                }

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error getting GPU info: {e}")

        return {
            'available': False,
            'count': 0,
            'devices': []
        }

    def _get_process_info(self) -> Dict[str, Any]:
        """Get current process information."""
        try:
            process = psutil.Process()

            return {
                'pid': process.pid,
                'cpu_percent': process.cpu_percent(),
                'memory_info': process.memory_info()._asdict(),
                'memory_percent': process.memory_percent(),
                'num_threads': process.num_threads(),
                'create_time': process.create_time()
            }

        except Exception as e:
            logger.error(f"Error getting process info: {e}")
            return {}

    def _get_fallback_metrics(self) -> Dict[str, Any]:
        """Get minimal fallback metrics."""
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu': {'percent': 0, 'count': 1},
            'memory': {'total': 0, 'available': 0, 'used': 0, 'percent': 0},
            'disk': {'total': 0, 'used': 0, 'free': 0, 'percent': 0},
            'gpu': {'available': False, 'count': 0, 'devices': []},
            'process': {}
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics_history:
            return self.get_current_metrics()

        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements

        # Calculate averages
        avg_cpu = sum(m['cpu']['percent'] for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m['memory']['percent'] for m in recent_metrics) / len(recent_metrics)

        # Determine performance status
        performance_status = 'good'
        if avg_cpu > 80 or avg_memory > 80:
            performance_status = 'critical'
        elif avg_cpu > 60 or avg_memory > 60:
            performance_status = 'warning'

        current = self.get_current_metrics()

        return {
            **current,
            'performance': {
                'status': performance_status,
                'avg_cpu_10min': avg_cpu,
                'avg_memory_10min': avg_memory,
                'history_size': len(self.metrics_history)
            }
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health assessment."""
        metrics = self.get_performance_summary()

        health_issues = []
        recommendations = []

        # Check CPU
        if metrics['cpu']['percent'] > 90:
            health_issues.append("High CPU usage detected")
            recommendations.append("Consider reducing model complexity or batch size")

        # Check Memory
        if metrics['memory']['percent'] > 90:
            health_issues.append("High memory usage detected")
            recommendations.append("Consider using smaller models or reducing batch size")

        # Check Disk
        if metrics['disk']['percent'] > 90:
            health_issues.append("Low disk space")
            recommendations.append("Clean up old analysis results or increase storage")

        # Check GPU
        if metrics['gpu']['available']:
            for gpu in metrics['gpu']['devices']:
                if gpu['memory_percent'] > 90:
                    health_issues.append(f"High GPU memory usage on {gpu['name']}")
                    recommendations.append("Consider reducing model size or batch size")

        overall_health = 'healthy'
        if health_issues:
            overall_health = 'warning' if len(health_issues) <= 2 else 'critical'

        return {
            'overall_health': overall_health,
            'issues': health_issues,
            'recommendations': recommendations,
            'metrics': metrics,
            'last_updated': datetime.now().isoformat()
        }

    def format_bytes(self, bytes_value: int) -> str:
        """Format bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"

    def get_formatted_metrics(self) -> Dict[str, Any]:
        """Get metrics with formatted values for display."""
        metrics = self.get_current_metrics()

        formatted = {
            'cpu_usage': f"{metrics['cpu']['percent']:.1f}%",
            'memory_usage': f"{metrics['memory']['percent']:.1f}%",
            'memory_used': self.format_bytes(metrics['memory']['used']),
            'memory_total': self.format_bytes(metrics['memory']['total']),
            'disk_usage': f"{metrics['disk']['percent']:.1f}%",
            'disk_free': self.format_bytes(metrics['disk']['free']),
            'disk_total': self.format_bytes(metrics['disk']['total'])
        }

        if metrics['gpu']['available']:
            gpu_info = []
            for gpu in metrics['gpu']['devices']:
                gpu_info.append({
                    'name': gpu['name'],
                    'memory_usage': f"{gpu['memory_percent']:.1f}%",
                    'memory_used': self.format_bytes(gpu['memory_allocated']),
                    'memory_total': self.format_bytes(gpu['memory_total'])
                })
            formatted['gpu_devices'] = gpu_info
        else:
            formatted['gpu_devices'] = []

        return formatted

# Global monitor instance
system_monitor = SystemResourceMonitor()

def start_system_monitoring():
    """Start system monitoring."""
    system_monitor.start_monitoring()

def stop_system_monitoring():
    """Stop system monitoring."""
    system_monitor.stop_monitoring()

def get_system_status():
    """Get current system status."""
    return system_monitor.get_performance_summary()

def get_system_health():
    """Get system health assessment."""
    return system_monitor.get_system_health()
