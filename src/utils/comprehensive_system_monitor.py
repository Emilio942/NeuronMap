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
        """Add callback for resource alerts."""
        self.alert_callbacks.append(callback)

    @robust_execution(max_retries=3, exceptions_to_catch=(psutil.Error,))
    def collect_cpu_metrics(self) -> CPUMetrics:
        """Collect CPU metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)

            # CPU frequency
            freq = psutil.cpu_freq()
            freq_current = freq.current if freq else 0
            freq_max = freq.max if freq else 0

            # CPU stats
            cpu_stats = psutil.cpu_stats()

            # Temperature (if available)
            temperature = None
            try:
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps:
                    temperature = temps['coretemp'][0].current
            except (AttributeError, KeyError, IndexError):
                pass

            return CPUMetrics(
                usage_percent=cpu_percent,
                usage_per_core=cpu_per_core,
                load_average=load_avg,
                frequency_current=freq_current,
                frequency_max=freq_max,
                context_switches=cpu_stats.ctx_switches,
                interrupts=cpu_stats.interrupts,
                temperature=temperature
            )

        except Exception as e:
            logger.error(f"Error collecting CPU metrics: {e}")
            raise ResourceError(f"Failed to collect CPU metrics: {e}")

    @robust_execution(max_retries=3, exceptions_to_catch=(psutil.Error,))
    def collect_memory_metrics(self) -> MemoryMetrics:
        """Collect memory metrics."""
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()

            return MemoryMetrics(
                total=mem.total,
                available=mem.available,
                used=mem.used,
                free=mem.free,
                percent=mem.percent,
                swap_total=swap.total,
                swap_used=swap.used,
                swap_free=swap.free,
                swap_percent=swap.percent,
                buffers=getattr(mem, 'buffers', 0),
                cached=getattr(mem, 'cached', 0)
            )

        except Exception as e:
            logger.error(f"Error collecting memory metrics: {e}")
            raise ResourceError(f"Failed to collect memory metrics: {e}")

    @robust_execution(max_retries=3, exceptions_to_catch=(psutil.Error,))
    def collect_disk_metrics(self, path: str = '/') -> DiskMetrics:
        """Collect disk metrics."""
        try:
            # Disk usage
            usage = psutil.disk_usage(path)

            # Disk I/O with caching
            current_time = time.time()
            if (self._last_disk_io is None or
                current_time - self._disk_io_cache_time > self.collection_interval):

                io_counters = psutil.disk_io_counters()
                if io_counters:
                    if self._last_disk_io:
                        time_delta = current_time - self._disk_io_cache_time
                        read_speed = (io_counters.read_bytes - self._last_disk_io.read_bytes) / time_delta
                        write_speed = (io_counters.write_bytes - self._last_disk_io.write_bytes) / time_delta
                    else:
                        read_speed = write_speed = 0

                    self._last_disk_io = io_counters
                    self._disk_io_cache_time = current_time
                else:
                    io_counters = type('DiskIO', (), {
                        'read_bytes': 0, 'write_bytes': 0,
                        'read_count': 0, 'write_count': 0,
                        'read_time': 0, 'write_time': 0
                    })()
                    read_speed = write_speed = 0
            else:
                io_counters = self._last_disk_io
                read_speed = write_speed = 0

            return DiskMetrics(
                total_space=usage.total,
                used_space=usage.used,
                free_space=usage.free,
                usage_percent=(usage.used / usage.total * 100),
                read_bytes=io_counters.read_bytes,
                write_bytes=io_counters.write_bytes,
                read_count=io_counters.read_count,
                write_count=io_counters.write_count,
                read_time=io_counters.read_time,
                write_time=io_counters.write_time,
                read_speed=read_speed,
                write_speed=write_speed
            )

        except Exception as e:
            logger.error(f"Error collecting disk metrics: {e}")
            raise ResourceError(f"Failed to collect disk metrics: {e}")

    @robust_execution(max_retries=3, exceptions_to_catch=(psutil.Error,))
    def collect_network_metrics(self) -> NetworkMetrics:
        """Collect network metrics."""
        try:
            # Network I/O with caching
            current_time = time.time()
            if (self._last_network_io is None or
                current_time - self._network_io_cache_time > self.collection_interval):

                io_counters = psutil.net_io_counters()
                if self._last_network_io:
                    time_delta = current_time - self._network_io_cache_time
                    send_speed = (io_counters.bytes_sent - self._last_network_io.bytes_sent) / time_delta
                    recv_speed = (io_counters.bytes_recv - self._last_network_io.bytes_recv) / time_delta
                else:
                    send_speed = recv_speed = 0

                self._last_network_io = io_counters
                self._network_io_cache_time = current_time
            else:
                io_counters = self._last_network_io
                send_speed = recv_speed = 0

            # Connection counts
            connections = psutil.net_connections()
            active_connections = len([c for c in connections if c.status == 'ESTABLISHED'])
            listening_connections = len([c for c in connections if c.status == 'LISTEN'])

            return NetworkMetrics(
                bytes_sent=io_counters.bytes_sent,
                bytes_recv=io_counters.bytes_recv,
                packets_sent=io_counters.packets_sent,
                packets_recv=io_counters.packets_recv,
                errin=io_counters.errin,
                errout=io_counters.errout,
                dropin=io_counters.dropin,
                dropout=io_counters.dropout,
                send_speed=send_speed,
                recv_speed=recv_speed,
                connections_active=active_connections,
                connections_listening=listening_connections
            )

        except Exception as e:
            logger.error(f"Error collecting network metrics: {e}")
            raise ResourceError(f"Failed to collect network metrics: {e}")

    def collect_process_metrics(self, limit: int = 10) -> List[ProcessMetrics]:
        """Collect metrics for top processes by CPU usage."""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent',
                                           'memory_info', 'num_threads', 'create_time', 'status']):
                try:
                    pinfo = proc.info
                    memory_info = pinfo.get('memory_info')

                    # Get file descriptor count (Unix only)
                    num_fds = 0
                    try:
                        if hasattr(proc, 'num_fds'):
                            num_fds = proc.num_fds()
                    except (psutil.AccessDenied, psutil.NoSuchProcess):
                        pass

                    processes.append(ProcessMetrics(
                        pid=pinfo['pid'],
                        name=pinfo['name'],
                        cpu_percent=pinfo['cpu_percent'] or 0,
                        memory_percent=pinfo['memory_percent'] or 0,
                        memory_rss=memory_info.rss if memory_info else 0,
                        memory_vms=memory_info.vms if memory_info else 0,
                        num_threads=pinfo['num_threads'] or 0,
                        num_fds=num_fds,
                        create_time=pinfo['create_time'] or 0,
                        status=pinfo['status'] or 'unknown'
                    ))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort by CPU usage and return top processes
            processes.sort(key=lambda p: p.cpu_percent, reverse=True)
            return processes[:limit]

        except Exception as e:
            logger.error(f"Error collecting process metrics: {e}")
            return []

    def check_thresholds(self, metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """Check resource usage against thresholds and generate alerts."""
        alerts = []

        # CPU alerts
        if metrics.cpu.usage_percent >= self.thresholds.cpu_critical:
            alerts.append({
                'level': AlertLevel.CRITICAL,
                'resource': 'cpu',
                'message': f'Critical CPU usage: {metrics.cpu.usage_percent:.1f}%',
                'value': metrics.cpu.usage_percent,
                'threshold': self.thresholds.cpu_critical
            })
        elif metrics.cpu.usage_percent >= self.thresholds.cpu_warning:
            alerts.append({
                'level': AlertLevel.WARNING,
                'resource': 'cpu',
                'message': f'High CPU usage: {metrics.cpu.usage_percent:.1f}%',
                'value': metrics.cpu.usage_percent,
                'threshold': self.thresholds.cpu_warning
            })

        # Memory alerts
        if metrics.memory.percent >= self.thresholds.memory_critical:
            alerts.append({
                'level': AlertLevel.CRITICAL,
                'resource': 'memory',
                'message': f'Critical memory usage: {metrics.memory.percent:.1f}%',
                'value': metrics.memory.percent,
                'threshold': self.thresholds.memory_critical
            })
        elif metrics.memory.percent >= self.thresholds.memory_warning:
            alerts.append({
                'level': AlertLevel.WARNING,
                'resource': 'memory',
                'message': f'High memory usage: {metrics.memory.percent:.1f}%',
                'value': metrics.memory.percent,
                'threshold': self.thresholds.memory_warning
            })

        # Swap alerts
        if metrics.memory.swap_percent >= self.thresholds.swap_critical:
            alerts.append({
                'level': AlertLevel.CRITICAL,
                'resource': 'swap',
                'message': f'Critical swap usage: {metrics.memory.swap_percent:.1f}%',
                'value': metrics.memory.swap_percent,
                'threshold': self.thresholds.swap_critical
            })
        elif metrics.memory.swap_percent >= self.thresholds.swap_warning:
            alerts.append({
                'level': AlertLevel.WARNING,
                'resource': 'swap',
                'message': f'High swap usage: {metrics.memory.swap_percent:.1f}%',
                'value': metrics.memory.swap_percent,
                'threshold': self.thresholds.swap_warning
            })

        # Disk alerts
        if metrics.disk.usage_percent >= self.thresholds.disk_critical:
            alerts.append({
                'level': AlertLevel.CRITICAL,
                'resource': 'disk',
                'message': f'Critical disk usage: {metrics.disk.usage_percent:.1f}%',
                'value': metrics.disk.usage_percent,
                'threshold': self.thresholds.disk_critical
            })
        elif metrics.disk.usage_percent >= self.thresholds.disk_warning:
            alerts.append({
                'level': AlertLevel.WARNING,
                'resource': 'disk',
                'message': f'High disk usage: {metrics.disk.usage_percent:.1f}%',
                'value': metrics.disk.usage_percent,
                'threshold': self.thresholds.disk_warning
            })

        return alerts

    def get_system_metrics(self) -> SystemMetrics:
        """Collect complete system metrics snapshot."""
        timestamp = time.time()

        cpu_metrics = self.collect_cpu_metrics()
        memory_metrics = self.collect_memory_metrics()
        disk_metrics = self.collect_disk_metrics()
        network_metrics = self.collect_network_metrics()
        process_metrics = self.collect_process_metrics()

        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu=cpu_metrics,
            memory=memory_metrics,
            disk=disk_metrics,
            network=network_metrics,
            processes=process_metrics
        )

        # Check thresholds and generate alerts
        alerts = self.check_thresholds(metrics)
        metrics.alerts = alerts

        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert['level'], alert['message'], alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")

        return metrics

    def start_monitoring(self):
        """Start continuous monitoring in background thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already started")
            return

        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("System monitoring started")

    def stop_monitoring_gracefully(self):
        """Stop monitoring gracefully."""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("System monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                metrics = self.get_system_metrics()

                with self.lock:
                    self.metrics_history.append(metrics)

                self.stop_monitoring.wait(self.collection_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self.stop_monitoring.wait(self.collection_interval)

    def get_metrics_history(self, last_n: Optional[int] = None) -> List[SystemMetrics]:
        """Get metrics history."""
        with self.lock:
            history = list(self.metrics_history)

        if last_n is not None:
            return history[-last_n:]
        return history

    def get_optimization_recommendations(self) -> Dict[str, List[str]]:
        """Get optimization recommendations based on current metrics and history."""
        if not self.metrics_history:
            return {}

        current_metrics = self.metrics_history[-1]
        history = list(self.metrics_history)

        recommendations = {
            'cpu': self.optimizer.analyze_cpu_usage(current_metrics.cpu, history),
            'memory': self.optimizer.analyze_memory_usage(current_metrics.memory, history),
            'disk': self.optimizer.analyze_disk_usage(current_metrics.disk, history)
        }

        return {k: v for k, v in recommendations.items() if v}

    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager to monitor a specific operation."""
        start_metrics = self.get_system_metrics()
        start_time = time.time()

        logger.info(f"Starting monitoring for operation: {operation_name}")

        try:
            yield start_metrics
        finally:
            end_metrics = self.get_system_metrics()
            end_time = time.time()

            duration = end_time - start_time
            cpu_delta = end_metrics.cpu.usage_percent - start_metrics.cpu.usage_percent
            memory_delta = end_metrics.memory.used - start_metrics.memory.used

            logger.info(
                f"Operation '{operation_name}' completed in {duration:.2f}s. "
                f"CPU delta: {cpu_delta:+.1f}%, Memory delta: {memory_delta/1024**2:+.1f}MB"
            )

    def export_metrics(self, filename: str, format: str = 'json'):
        """Export metrics history to file."""
        history = self.get_metrics_history()

        if format.lower() == 'json':
            data = [metrics.to_dict() for metrics in history]
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported {len(history)} metrics snapshots to {filename}")


# Utility functions
def get_system_info() -> Dict[str, Any]:
    """Get basic system information."""
    return {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'architecture': platform.architecture(),
        'hostname': socket.gethostname(),
        'cpu_count': psutil.cpu_count(),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'memory_total': psutil.virtual_memory().total,
        'boot_time': psutil.boot_time(),
        'python_version': platform.python_version()
    }


def format_bytes(bytes_value: int) -> str:
    """Format bytes as human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_metrics_summary(metrics: SystemMetrics) -> str:
    """Format metrics as human-readable summary."""
    return f"""System Metrics Summary ({time.ctime(metrics.timestamp)}):
CPU: {metrics.cpu.usage_percent:.1f}% (Load: {metrics.cpu.load_average[0]:.2f})
Memory: {metrics.memory.percent:.1f}% ({format_bytes(metrics.memory.used)}/{format_bytes(metrics.memory.total)})
Disk: {metrics.disk.usage_percent:.1f}% ({format_bytes(metrics.disk.free_space)} free)
Network: ↑{format_bytes(metrics.network.send_speed)}/s ↓{format_bytes(metrics.network.recv_speed)}/s
Alerts: {len(metrics.alerts)} active"""


# Global monitor instance
system_monitor = SystemResourceMonitor()

def start_system_monitoring():
    """Start system monitoring."""
    system_monitor.start_monitoring()

def stop_system_monitoring():
    """Stop system monitoring."""
    system_monitor.stop_monitoring_gracefully()

def get_system_status():
    """Get current system status."""
    return system_monitor.get_system_metrics()

def get_system_health():
    """Get system health assessment."""
    return system_monitor.get_optimization_recommendations()
