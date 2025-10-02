"""Monitoring and health check utilities for NeuronMap."""

import psutil
import torch
import requests
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json


logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    disk_used_gb: float
    disk_total_gb: float
    disk_percent: float
    gpu_available: bool
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None


@dataclass
class ProcessMetrics:
    """Process-specific metrics."""
    timestamp: float
    process_id: int
    cpu_percent: float
    memory_used_mb: float
    memory_percent: float
    num_threads: int
    status: str


class SystemMonitor:
    """Monitor system resources and performance."""

    def __init__(self, log_interval: int = 60):
        """Initialize system monitor.

        Args:
            log_interval: Interval in seconds between metric logging.
        """
        self.log_interval = log_interval
        self.metrics_history: List[SystemMetrics] = []
        self.process_metrics_history: List[ProcessMetrics] = []

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics.

        Returns:
            SystemMetrics object with current system state.
        """
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # GPU metrics (if available)
        gpu_available = torch.cuda.is_available()
        gpu_memory_used = None
        gpu_memory_total = None
        gpu_memory_percent = None
        gpu_utilization = None

        if gpu_available:
            try:
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_total = torch.cuda.max_memory_allocated() / 1024**3  # GB
                if gpu_memory_total > 0:
                    gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100

                # Try to get GPU utilization using nvidia-ml-py if available
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization = utilization.gpu
                except ImportError:
                    pass  # pynvml not available
                except Exception:
                    pass  # Other GPU monitoring error

            except Exception as e:
                logger.debug(f"Error getting GPU metrics: {e}")

        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_used_gb=memory.used / 1024**3,
            memory_total_gb=memory.total / 1024**3,
            memory_percent=memory.percent,
            disk_used_gb=disk.used / 1024**3,
            disk_total_gb=disk.total / 1024**3,
            disk_percent=(disk.used / disk.total) * 100,
            gpu_available=gpu_available,
            gpu_memory_used_gb=gpu_memory_used,
            gpu_memory_total_gb=gpu_memory_total,
            gpu_memory_percent=gpu_memory_percent,
            gpu_utilization_percent=gpu_utilization
        )

    def get_process_metrics(self, pid: Optional[int] = None) -> ProcessMetrics:
        """Get current process metrics.

        Args:
            pid: Process ID. If None, uses current process.

        Returns:
            ProcessMetrics object with current process state.
        """
        try:
            if pid is None:
                process = psutil.Process()
            else:
                process = psutil.Process(pid)

            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            return ProcessMetrics(
                timestamp=time.time(),
                process_id=process.pid,
                cpu_percent=cpu_percent,
                memory_used_mb=memory_info.rss / 1024**2,
                memory_percent=memory_percent,
                num_threads=process.num_threads(),
                status=process.status()
            )

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.error(f"Error getting process metrics: {e}")
            raise

    def check_resource_availability(self,
                                  min_memory_gb: float = 2.0,
                                  min_disk_gb: float = 1.0,
                                  max_cpu_percent: float = 90.0) -> Dict[str, Any]:
        """Check if system has sufficient resources.

        Args:
            min_memory_gb: Minimum required free memory in GB.
            min_disk_gb: Minimum required free disk space in GB.
            max_cpu_percent: Maximum acceptable CPU usage percentage.

        Returns:
            Dictionary with resource check results.
        """
        metrics = self.get_system_metrics()

        memory_available_gb = metrics.memory_total_gb - metrics.memory_used_gb
        disk_available_gb = metrics.disk_total_gb - metrics.disk_used_gb

        checks = {
            'sufficient_memory': memory_available_gb >= min_memory_gb,
            'sufficient_disk': disk_available_gb >= min_disk_gb,
            'acceptable_cpu_load': metrics.cpu_percent <= max_cpu_percent,
            'memory_available_gb': memory_available_gb,
            'disk_available_gb': disk_available_gb,
            'cpu_percent': metrics.cpu_percent,
            'recommendations': []
        }

        # Generate recommendations
        if not checks['sufficient_memory']:
            checks['recommendations'].append(
                f"Free up memory: {memory_available_gb:.1f}GB available, "
                f"{min_memory_gb:.1f}GB required"
            )

        if not checks['sufficient_disk']:
            checks['recommendations'].append(
                f"Free up disk space: {disk_available_gb:.1f}GB available, "
                f"{min_disk_gb:.1f}GB required"
            )

        if not checks['acceptable_cpu_load']:
            checks['recommendations'].append(
                f"High CPU usage: {metrics.cpu_percent:.1f}%, consider reducing load"
            )

        checks['all_checks_passed'] = all([
            checks['sufficient_memory'],
            checks['sufficient_disk'],
            checks['acceptable_cpu_load']
        ])

        return checks

    def log_metrics(self, metrics: SystemMetrics):
        """Log system metrics.

        Args:
            metrics: SystemMetrics to log.
        """
        self.metrics_history.append(metrics)

        logger.info(f"System Metrics - CPU: {metrics.cpu_percent:.1f}%, "
                   f"Memory: {metrics.memory_percent:.1f}% "
                   f"({metrics.memory_used_gb:.1f}/{metrics.memory_total_gb:.1f}GB), "
                   f"Disk: {metrics.disk_percent:.1f}%")

        if metrics.gpu_available and metrics.gpu_memory_percent is not None:
            logger.info(f"GPU Memory: {metrics.gpu_memory_percent:.1f}% "
                       f"({metrics.gpu_memory_used_gb:.1f}GB)")

        # Keep only recent history (last 1000 entries)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

    def save_metrics_to_file(self, filepath: str):
        """Save metrics history to file.

        Args:
            filepath: Path to save metrics file.
        """
        try:
            metrics_data = [asdict(metric) for metric in self.metrics_history]

            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)

            logger.info(f"Metrics saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving metrics to {filepath}: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Alias for get_system_metrics that returns a dict for test compatibility."""
        metrics = self.get_system_metrics()
        return {
            'timestamp': metrics.timestamp,
            'cpu_usage': metrics.cpu_percent,  # Renamed for test compatibility
            'cpu_percent': metrics.cpu_percent,
            'memory_used_gb': metrics.memory_used_gb,
            'memory_total_gb': metrics.memory_total_gb,
            'memory_percent': metrics.memory_percent,
            'disk_used_gb': metrics.disk_used_gb,
            'disk_total_gb': metrics.disk_total_gb,
            'disk_percent': metrics.disk_percent,
            'gpu_available': metrics.gpu_available,
            'gpu_memory_used_gb': metrics.gpu_memory_used_gb,
            'gpu_memory_total_gb': metrics.gpu_memory_total_gb,
            'gpu_memory_percent': metrics.gpu_memory_percent,
            'gpu_utilization_percent': metrics.gpu_utilization_percent
        }


class HealthChecker:
    """Health check utilities for NeuronMap components."""

    def __init__(self):
        """Initialize health checker."""
        self.last_check_results: Dict[str, Any] = {}

    def check_ollama_health(self, host: str = "http://localhost:11434",
                           timeout: int = 10) -> Dict[str, Any]:
        """Check if Ollama is healthy and responsive.

        Args:
            host: Ollama host URL.
            timeout: Request timeout in seconds.

        Returns:
            Dictionary with health check results.
        """
        result = {
            'healthy': False,
            'reachable': False,
            'models_available': False,
            'response_time_ms': None,
            'error': None,
            'models': []
        }

        try:
            start_time = time.time()

            # Check if Ollama is reachable
            response = requests.get(f"{host}/api/tags", timeout=timeout)
            response_time = (time.time() - start_time) * 1000
            result['response_time_ms'] = response_time

            if response.status_code == 200:
                result['reachable'] = True

                # Check available models
                try:
                    models_data = response.json()
                    if 'models' in models_data:
                        result['models'] = [model['name'] for model in models_data['models']]
                        result['models_available'] = len(result['models']) > 0
                        result['healthy'] = True

                except json.JSONDecodeError:
                    result['error'] = "Invalid JSON response from Ollama"

            else:
                result['error'] = f"HTTP {response.status_code}: {response.text}"

        except requests.ConnectionError:
            result['error'] = f"Cannot connect to Ollama at {host}"
        except requests.Timeout:
            result['error'] = f"Timeout connecting to Ollama (>{timeout}s)"
        except Exception as e:
            result['error'] = f"Unexpected error: {str(e)}"

        self.last_check_results['ollama'] = result
        return result


class PerformanceMonitor:
    """Lightweight performance timing helper."""

    class _TimingContext:
        def __init__(self, monitor: "PerformanceMonitor", name: str):
            self._monitor = monitor
            self._name = name
            self._start: float | None = None

        def __enter__(self):
            self._start = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            end = time.perf_counter()
            duration = end - (self._start or end)
            self._monitor._record_duration(self._name, duration)
            # Never suppress exceptions
            return False

    def __init__(self):
        self._metrics: Dict[str, Dict[str, float]] = {}

    def time_operation(self, name: str) -> "PerformanceMonitor._TimingContext":
        """Return a context manager for measuring an operation."""
        return PerformanceMonitor._TimingContext(self, name)

    def _record_duration(self, name: str, duration: float):
        data = self._metrics.setdefault(name, {
            'count': 0,
            'total_duration': 0.0,
            'average_duration': 0.0,
            'last_duration': 0.0,
            'duration': 0.0
        })
        data['count'] += 1
        data['total_duration'] += duration
        data['last_duration'] = duration
        data['duration'] = duration
        data['average_duration'] = data['total_duration'] / data['count']

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Return aggregated timing metrics."""
        return {name: dict(values) for name, values in self._metrics.items()}

    def reset(self):
        """Clear recorded metrics."""
        self._metrics.clear()


class ResourceMonitor:
    """Monitor system resource usage for the current process."""

    def __init__(self):
        try:
            self._process = psutil.Process()
        except Exception:
            self._process = None
        self._peak_memory_mb = 0.0

    def get_memory_usage(self) -> Dict[str, float]:
        """Return current and peak resident memory in MB."""
        if self._process is None:
            return {
                'current_mb': 0.0,
                'peak_mb': self._peak_memory_mb
            }

        mem_info = self._process.memory_info()
        current_mb = mem_info.rss / 1024 / 1024
        self._peak_memory_mb = max(self._peak_memory_mb, current_mb)
        return {
            'current_mb': current_mb,
            'peak_mb': self._peak_memory_mb
        }

    def get_gpu_usage(self) -> Dict[str, float]:
        """Return basic GPU utilisation stats if CUDA is available."""
        if not torch.cuda.is_available():
            return {}

        usage: Dict[str, float] = {
            'gpu_memory_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024
        }

        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            usage['gpu_memory_total_mb'] = total_memory
        except Exception:
            pass

        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            usage['gpu_utilization'] = float(util.gpu)
            pynvml.nvmlShutdown()
        except Exception:
            # Best effort only
            pass

        return usage

    def reset(self):
        """Reset tracked peak metrics."""
        self._peak_memory_mb = 0.0

    def check_model_availability(self, model_name: str,
                                host: str = "http://localhost:11434") -> Dict[str, Any]:
        """Check if a specific model is available in Ollama.

        Args:
            model_name: Name of the model to check.
            host: Ollama host URL.

        Returns:
            Dictionary with model availability results.
        """
        result = {
            'available': False,
            'model_name': model_name,
            'error': None,
            'size_gb': None,
            'modified_at': None
        }

        try:
            # First check Ollama health
            ollama_health = self.check_ollama_health(host)

            if not ollama_health['healthy']:
                result['error'] = f"Ollama not healthy: {ollama_health['error']}"
                return result

            # Check if specific model is available
            if model_name in ollama_health['models']:
                result['available'] = True

                # Try to get model details
                try:
                    response = requests.post(
                        f"{host}/api/show",
                        json={'name': model_name},
                        timeout=10
                    )

                    if response.status_code == 200:
                        model_info = response.json()
                        if 'details' in model_info:
                            details = model_info['details']
                            if 'parameter_size' in details:
                                # Estimate size (rough approximation)
                                param_size = details.get('parameter_size', '0')
                                if param_size.endswith('B'):
                                    result['size_gb'] = float(param_size[:-1]) * 2  # Rough estimate

                            result['modified_at'] = model_info.get('modified_at')

                except Exception as e:
                    logger.debug(f"Could not get model details: {e}")

            else:
                result['error'] = f"Model '{model_name}' not found in available models: {ollama_health['models']}"

        except Exception as e:
            result['error'] = f"Error checking model availability: {str(e)}"

        return result

    def check_pytorch_gpu(self) -> Dict[str, Any]:
        """Check PyTorch GPU availability and configuration.

        Returns:
            Dictionary with GPU check results.
        """
        result = {
            'cuda_available': False,
            'device_count': 0,
            'current_device': None,
            'devices': [],
            'error': None
        }

        try:
            result['cuda_available'] = torch.cuda.is_available()

            if result['cuda_available']:
                result['device_count'] = torch.cuda.device_count()
                result['current_device'] = torch.cuda.current_device()

                # Get device information
                for i in range(result['device_count']):
                    try:
                        device_props = torch.cuda.get_device_properties(i)
                        device_info = {
                            'id': i,
                            'name': device_props.name,
                            'memory_total_gb': device_props.total_memory / 1024**3,
                            'memory_allocated_gb': torch.cuda.memory_allocated(i) / 1024**3,
                            'memory_reserved_gb': torch.cuda.memory_reserved(i) / 1024**3,
                            'compute_capability': f"{device_props.major}.{device_props.minor}"
                        }
                        result['devices'].append(device_info)

                    except Exception as e:
                        logger.debug(f"Error getting info for GPU {i}: {e}")

        except Exception as e:
            result['error'] = f"Error checking PyTorch GPU: {str(e)}"

        self.last_check_results['pytorch_gpu'] = result
        return result

    def run_comprehensive_health_check(self,
                                     ollama_host: str = "http://localhost:11434",
                                     model_name: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive health check of all components.

        Args:
            ollama_host: Ollama host URL.
            model_name: Specific model to check (optional).

        Returns:
            Dictionary with comprehensive health check results.
        """
        logger.info("Running comprehensive health check...")

        results = {
            'timestamp': time.time(),
            'overall_healthy': True,
            'checks': {},
            'recommendations': []
        }

        # System resources
        monitor = SystemMonitor()
        resource_check = monitor.check_resource_availability()
        results['checks']['system_resources'] = resource_check

        if not resource_check['all_checks_passed']:
            results['overall_healthy'] = False
            results['recommendations'].extend(resource_check['recommendations'])

        # Ollama health
        ollama_check = self.check_ollama_health(ollama_host)
        results['checks']['ollama'] = ollama_check

        if not ollama_check['healthy']:
            results['overall_healthy'] = False
            results['recommendations'].append(f"Ollama issue: {ollama_check['error']}")

        # Specific model check
        if model_name:
            model_check = self.check_model_availability(model_name, ollama_host)
            results['checks']['model'] = model_check

            if not model_check['available']:
                results['overall_healthy'] = False
                results['recommendations'].append(
                    f"Model issue: {model_check['error']}"
                )

        # GPU check
        gpu_check = self.check_pytorch_gpu()
        results['checks']['gpu'] = gpu_check

        if gpu_check['error']:
            results['recommendations'].append(f"GPU issue: {gpu_check['error']}")

        # Log results
        if results['overall_healthy']:
            logger.info("✓ All health checks passed")
        else:
            logger.warning("⚠ Some health checks failed")
            for rec in results['recommendations']:
                logger.warning(f"  - {rec}")

        return results


def main():
    """Command line interface for monitoring and health checks."""
    import argparse

    parser = argparse.ArgumentParser(description="NeuronMap monitoring and health checks")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Health check command
    health_parser = subparsers.add_parser("health", help="Run health checks")
    health_parser.add_argument("--ollama-host", default="http://localhost:11434",
                              help="Ollama host URL")
    health_parser.add_argument("--model", help="Specific model to check")

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor system resources")
    monitor_parser.add_argument("--interval", type=int, default=60,
                               help="Monitoring interval in seconds")
    monitor_parser.add_argument("--duration", type=int, default=600,
                               help="Monitoring duration in seconds")
    monitor_parser.add_argument("--output", help="Output file for metrics")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if args.command == "health":
        checker = HealthChecker()
        results = checker.run_comprehensive_health_check(
            ollama_host=args.ollama_host,
            model_name=args.model
        )

        print("\n=== Health Check Results ===")
        print(f"Overall Status: {'✓ HEALTHY' if results['overall_healthy'] else '⚠ ISSUES FOUND'}")

        if results['recommendations']:
            print("\nRecommendations:")
            for rec in results['recommendations']:
                print(f"  - {rec}")

        print(f"\nDetailed results saved to log")

    elif args.command == "monitor":
        monitor = SystemMonitor(args.interval)

        print(f"Monitoring system for {args.duration} seconds...")
        start_time = time.time()

        while time.time() - start_time < args.duration:
            metrics = monitor.get_system_metrics()
            monitor.log_metrics(metrics)
            time.sleep(args.interval)

        if args.output:
            monitor.save_metrics_to_file(args.output)
            print(f"Metrics saved to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
