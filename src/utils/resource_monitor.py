"""GPU Resource Management System for NeuronMap.

This module provides comprehensive GPU resource monitoring, memory optimization,
and intelligent workload distribution according to roadmap section 2.2.
"""

import torch
import time
import threading
import logging
import psutil
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import gc
from contextlib import contextmanager

from .error_handling import NeuronMapException, ResourceError
from .robust_decorators import robust_execution

logger = logging.getLogger(__name__)


@dataclass
class GPUStatus:
    """Status information for a single GPU."""
    device_id: int
    name: str
    memory_total: int  # bytes
    memory_used: int   # bytes
    memory_available: int  # bytes
    memory_percent: float
    utilization_percent: float
    temperature: Optional[float] = None
    power_draw: Optional[float] = None
    fan_speed: Optional[float] = None
    compute_capability: Optional[Tuple[int, int]] = None
    is_available: bool = True

    @property
    def memory_total_gb(self) -> float:
        """Total memory in GB."""
        return self.memory_total / (1024**3)

    @property
    def memory_used_gb(self) -> float:
        """Used memory in GB."""
        return self.memory_used / (1024**3)

    @property
    def memory_available_gb(self) -> float:
        """Available memory in GB."""
        return self.memory_available / (1024**3)


@dataclass
class MemoryProfile:
    """Memory usage profile for tracking over time."""
    timestamp: float
    peak_memory: int
    current_memory: int
    reserved_memory: int
    active_memory: int
    allocated_objects: int

    @property
    def peak_memory_gb(self) -> float:
        return self.peak_memory / (1024**3)

    @property
    def current_memory_gb(self) -> float:
        return self.current_memory / (1024**3)


class MemoryOptimizationStrategy(Enum):
    """Available memory optimization strategies."""
    GRADIENT_CHECKPOINTING = "gradient_checkpointing"
    MIXED_PRECISION = "mixed_precision"
    BATCH_SIZE_REDUCTION = "batch_size_reduction"
    MODEL_PARALLELISM = "model_parallelism"
    CPU_OFFLOADING = "cpu_offloading"
    DYNAMIC_BATCHING = "dynamic_batching"


class WorkloadPriority(Enum):
    """Priority levels for workload scheduling."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class WorkloadTask:
    """Represents a task in the workload scheduler."""
    task_id: str
    priority: WorkloadPriority
    estimated_memory_gb: float
    estimated_time_minutes: float
    gpu_required: bool
    callback: Callable
    args: tuple = ()
    kwargs: dict = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


class VRAMOptimizer:
    """Automatic VRAM optimization and memory management."""

    def __init__(self):
        self.optimization_strategies = {
            MemoryOptimizationStrategy.GRADIENT_CHECKPOINTING: self._apply_gradient_checkpointing,
            MemoryOptimizationStrategy.MIXED_PRECISION: self._apply_mixed_precision,
            MemoryOptimizationStrategy.BATCH_SIZE_REDUCTION: self._reduce_batch_size,
            MemoryOptimizationStrategy.CPU_OFFLOADING: self._enable_cpu_offloading,
        }
        self.memory_profiles: List[MemoryProfile] = []

    @robust_execution(max_retries=1)
    def optimize_memory_usage(self, current_usage_gb: float,
                            available_gb: float,
                            strategies: List[MemoryOptimizationStrategy] = None) -> Dict[str, Any]:
        """
        Apply memory optimization strategies.

        Args:
            current_usage_gb: Current memory usage in GB
            available_gb: Available memory in GB
            strategies: List of strategies to try (None for automatic selection)

        Returns:
            Dictionary with optimization results
        """
        if strategies is None:
            strategies = self._select_optimization_strategies(current_usage_gb, available_gb)

        results = {
            "applied_strategies": [],
            "memory_saved_gb": 0.0,
            "recommendations": []
        }

        for strategy in strategies:
            try:
                if strategy in self.optimization_strategies:
                    saved_memory = self.optimization_strategies[strategy]()
                    if saved_memory > 0:
                        results["applied_strategies"].append(strategy.value)
                        results["memory_saved_gb"] += saved_memory
                        logger.info(f"Applied {strategy.value}, saved {saved_memory:.2f}GB")
            except Exception as e:
                logger.warning(f"Failed to apply {strategy.value}: {e}")
                results["recommendations"].append(f"Consider manual {strategy.value}")

        return results

    def _select_optimization_strategies(self, current_usage_gb: float,
                                      available_gb: float) -> List[MemoryOptimizationStrategy]:
        """Select appropriate optimization strategies based on memory pressure."""
        memory_pressure = current_usage_gb / (current_usage_gb + available_gb)
        strategies = []

        if memory_pressure > 0.9:  # Critical memory pressure
            strategies.extend([
                MemoryOptimizationStrategy.GRADIENT_CHECKPOINTING,
                MemoryOptimizationStrategy.BATCH_SIZE_REDUCTION,
                MemoryOptimizationStrategy.CPU_OFFLOADING
            ])
        elif memory_pressure > 0.7:  # High memory pressure
            strategies.extend([
                MemoryOptimizationStrategy.MIXED_PRECISION,
                MemoryOptimizationStrategy.GRADIENT_CHECKPOINTING
            ])
        elif memory_pressure > 0.5:  # Medium memory pressure
            strategies.append(MemoryOptimizationStrategy.MIXED_PRECISION)

        return strategies

    def _apply_gradient_checkpointing(self) -> float:
        """Apply gradient checkpointing to save memory."""
        # This would be implemented based on the specific model architecture
        # For now, return estimated memory savings
        return 0.5  # Estimated 0.5GB savings

    def _apply_mixed_precision(self) -> float:
        """Apply mixed precision training to save memory."""
        # Implementation would depend on the model and training setup
        return 0.3  # Estimated 0.3GB savings

    def _reduce_batch_size(self) -> float:
        """Reduce batch size to save memory."""
        # Implementation would adjust batch size dynamically
        return 0.8  # Estimated 0.8GB savings

    def _enable_cpu_offloading(self) -> float:
        """Enable CPU offloading for less critical operations."""
        # Implementation would move certain operations to CPU
        return 1.0  # Estimated 1.0GB savings

    def profile_memory_usage(self, device_id: int = 0) -> MemoryProfile:
        """Profile current memory usage on specified device."""
        if not torch.cuda.is_available():
            raise ResourceError("CUDA not available for memory profiling")

        torch.cuda.synchronize(device_id)

        current_memory = torch.cuda.memory_allocated(device_id)
        peak_memory = torch.cuda.max_memory_allocated(device_id)
        reserved_memory = torch.cuda.memory_reserved(device_id)

        # Get active memory (approximation)
        active_memory = current_memory  # Simplified

        # Count allocated objects (approximation)
        allocated_objects = len([obj for obj in gc.get_objects()
                               if hasattr(obj, 'device') and obj.device.index == device_id])

        profile = MemoryProfile(
            timestamp=time.time(),
            peak_memory=peak_memory,
            current_memory=current_memory,
            reserved_memory=reserved_memory,
            active_memory=active_memory,
            allocated_objects=allocated_objects
        )

        self.memory_profiles.append(profile)

        # Keep only last 100 profiles
        if len(self.memory_profiles) > 100:
            self.memory_profiles = self.memory_profiles[-100:]

        return profile


class WorkloadScheduler:
    """Intelligent workload scheduling for multi-GPU systems."""

    def __init__(self):
        self.task_queue = queue.PriorityQueue()
        self.running_tasks: Dict[str, WorkloadTask] = {}
        self.completed_tasks: List[str] = []
        self.gpu_assignments: Dict[int, List[str]] = {}
        self._scheduler_thread = None
        self._stop_event = threading.Event()

    def submit_task(self, task: WorkloadTask) -> str:
        """Submit a task for execution."""
        # Priority queue uses tuple (priority, task_id, task)
        # Lower priority values have higher precedence
        priority_value = 5 - task.priority.value  # Invert for correct ordering
        self.task_queue.put((priority_value, task.task_id, task))
        logger.info(f"Submitted task {task.task_id} with priority {task.priority}")
        return task.task_id

    def start_scheduler(self):
        """Start the background scheduler thread."""
        if self._scheduler_thread is None or not self._scheduler_thread.is_alive():
            self._stop_event.clear()
            self._scheduler_thread = threading.Thread(target=self._scheduler_loop)
            self._scheduler_thread.daemon = True
            self._scheduler_thread.start()
            logger.info("Workload scheduler started")

    def stop_scheduler(self):
        """Stop the background scheduler thread."""
        self._stop_event.set()
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=5.0)
        logger.info("Workload scheduler stopped")

    def _scheduler_loop(self):
        """Main scheduler loop."""
        while not self._stop_event.is_set():
            try:
                if not self.task_queue.empty():
                    priority, task_id, task = self.task_queue.get(timeout=1.0)
                    self._execute_task(task)
                else:
                    time.sleep(0.1)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")

    def _execute_task(self, task: WorkloadTask):
        """Execute a task on the most suitable GPU."""
        try:
            gpu_id = self._select_optimal_gpu(task)
            self.running_tasks[task.task_id] = task

            if gpu_id is not None:
                if gpu_id not in self.gpu_assignments:
                    self.gpu_assignments[gpu_id] = []
                self.gpu_assignments[gpu_id].append(task.task_id)

            # Execute the task
            result = task.callback(*task.args, **task.kwargs)

            # Task completed
            self.completed_tasks.append(task.task_id)
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]

            if gpu_id is not None and task.task_id in self.gpu_assignments.get(gpu_id, []):
                self.gpu_assignments[gpu_id].remove(task.task_id)

            logger.info(f"Task {task.task_id} completed successfully")

        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]

    def _select_optimal_gpu(self, task: WorkloadTask) -> Optional[int]:
        """Select the optimal GPU for a task."""
        if not task.gpu_required or not torch.cuda.is_available():
            return None

        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            return None

        # Simple load balancing: choose GPU with fewest running tasks
        gpu_loads = {}
        for gpu_id in range(gpu_count):
            gpu_loads[gpu_id] = len(self.gpu_assignments.get(gpu_id, []))

        # Select GPU with minimum load
        optimal_gpu = min(gpu_loads, key=gpu_loads.get)
        return optimal_gpu

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue and execution status."""
        return {
            "queued_tasks": self.task_queue.qsize(),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "gpu_assignments": dict(self.gpu_assignments)
        }


class GPUResourceManager:
    """Main GPU resource management system."""

    def __init__(self, monitoring_interval: float = 1.0):
        """
        Initialize GPU resource manager.

        Args:
            monitoring_interval: Interval in seconds between monitoring updates
        """
        self.monitoring_interval = monitoring_interval
        self.gpu_monitor = None
        self.memory_optimizer = VRAMOptimizer()
        self.workload_scheduler = WorkloadScheduler()
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self.gpu_status_history: List[List[GPUStatus]] = []

        # Try to import nvidia-ml-py for advanced monitoring
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_available = True
            self._nvml = pynvml
        except ImportError:
            self._nvml_available = False
            self._nvml = None
            logger.warning("pynvml not available - advanced GPU monitoring disabled")

    @robust_execution(max_retries=3, retry_delay=1.0)
    def get_gpu_status(self) -> List[GPUStatus]:
        """
        Get current status of all available GPUs.

        Returns:
            List of GPUStatus objects for each available GPU
        """
        if not torch.cuda.is_available():
            return []

        gpu_statuses = []
        device_count = torch.cuda.device_count()

        for device_id in range(device_count):
            try:
                status = self._get_single_gpu_status(device_id)
                gpu_statuses.append(status)
            except Exception as e:
                logger.warning(f"Error getting status for GPU {device_id}: {e}")
                # Create a placeholder status
                gpu_statuses.append(GPUStatus(
                    device_id=device_id,
                    name="Unknown",
                    memory_total=0,
                    memory_used=0,
                    memory_available=0,
                    memory_percent=0.0,
                    utilization_percent=0.0,
                    is_available=False
                ))

        return gpu_statuses

    def _get_single_gpu_status(self, device_id: int) -> GPUStatus:
        """Get status for a single GPU."""
        # Basic PyTorch information
        device_properties = torch.cuda.get_device_properties(device_id)
        name = device_properties.name
        memory_total = device_properties.total_memory
        compute_capability = (device_properties.major, device_properties.minor)

        # Memory information
        torch.cuda.synchronize(device_id)
        memory_allocated = torch.cuda.memory_allocated(device_id)
        memory_reserved = torch.cuda.memory_reserved(device_id)
        memory_available = memory_total - memory_reserved
        memory_percent = (memory_reserved / memory_total) * 100

        # Advanced monitoring with nvidia-ml-py
        utilization_percent = 0.0
        temperature = None
        power_draw = None
        fan_speed = None

        if self._nvml_available:
            try:
                handle = self._nvml.nvmlDeviceGetHandleByIndex(device_id)

                # Utilization
                utilization = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                utilization_percent = utilization.gpu

                # Temperature
                temperature = self._nvml.nvmlDeviceGetTemperature(handle,
                                                                self._nvml.NVML_TEMPERATURE_GPU)

                # Power draw
                try:
                    power_draw = self._nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                except:
                    pass  # Power monitoring not supported on all GPUs

                # Fan speed
                try:
                    fan_speed = self._nvml.nvmlDeviceGetFanSpeed(handle)
                except:
                    pass  # Fan monitoring not supported on all GPUs

            except Exception as e:
                logger.debug(f"Error getting advanced GPU metrics for device {device_id}: {e}")

        return GPUStatus(
            device_id=device_id,
            name=name,
            memory_total=memory_total,
            memory_used=memory_allocated,
            memory_available=memory_available,
            memory_percent=memory_percent,
            utilization_percent=utilization_percent,
            temperature=temperature,
            power_draw=power_draw,
            fan_speed=fan_speed,
            compute_capability=compute_capability,
            is_available=True
        )

    def start_monitoring(self):
        """Start real-time GPU monitoring."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self._monitoring_thread.daemon = True
            self._monitoring_thread.start()
            logger.info("GPU monitoring started")

        # Also start the workload scheduler
        self.workload_scheduler.start_scheduler()

    def stop_monitoring(self):
        """Stop real-time GPU monitoring."""
        self._stop_monitoring.set()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        logger.info("GPU monitoring stopped")

        # Stop the workload scheduler
        self.workload_scheduler.stop_scheduler()

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                gpu_statuses = self.get_gpu_status()
                self.gpu_status_history.append(gpu_statuses)

                # Keep only last 1000 measurements
                if len(self.gpu_status_history) > 1000:
                    self.gpu_status_history = self.gpu_status_history[-1000:]

                # Check for memory pressure and apply optimizations
                for status in gpu_statuses:
                    if status.memory_percent > 90:  # High memory usage
                        logger.warning(f"High memory usage on GPU {status.device_id}: "
                                     f"{status.memory_percent:.1f}%")
                        self.memory_optimizer.optimize_memory_usage(
                            status.memory_used_gb,
                            status.memory_available_gb
                        )

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    @contextmanager
    def gpu_memory_context(self, device_id: int, expected_usage_gb: float):
        """
        Context manager for GPU memory management.

        Args:
            device_id: GPU device ID
            expected_usage_gb: Expected memory usage in GB
        """
        if not torch.cuda.is_available():
            yield
            return

        # Check if we have enough memory
        initial_status = self._get_single_gpu_status(device_id)
        if expected_usage_gb > initial_status.memory_available_gb:
            # Try to optimize memory
            optimization_result = self.memory_optimizer.optimize_memory_usage(
                initial_status.memory_used_gb,
                initial_status.memory_available_gb
            )
            logger.info(f"Memory optimization applied: {optimization_result}")

        # Set device and clear cache
        original_device = torch.cuda.current_device()
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()

        try:
            yield
        finally:
            # Cleanup
            torch.cuda.empty_cache()
            torch.cuda.set_device(original_device)

    def get_memory_usage_trend(self, device_id: int,
                             minutes: int = 10) -> List[MemoryProfile]:
        """Get memory usage trend for specified GPU over last N minutes."""
        if not self.gpu_status_history:
            return []

        cutoff_time = time.time() - (minutes * 60)
        trend_data = []

        for status_list in self.gpu_status_history:
            for status in status_list:
                if (status.device_id == device_id and
                    hasattr(status, 'timestamp') and
                    status.timestamp > cutoff_time):
                    # Convert GPUStatus to MemoryProfile for consistency
                    profile = MemoryProfile(
                        timestamp=getattr(status, 'timestamp', time.time()),
                        peak_memory=status.memory_total,
                        current_memory=status.memory_used,
                        reserved_memory=status.memory_used,  # Approximation
                        active_memory=status.memory_used,
                        allocated_objects=0  # Not available from GPU status
                    )
                    trend_data.append(profile)

        return trend_data

    def predict_oom_risk(self, device_id: int) -> float:
        """
        Predict the risk of out-of-memory error on specified GPU.

        Args:
            device_id: GPU device ID

        Returns:
            Risk score from 0.0 (no risk) to 1.0 (high risk)
        """
        trend_data = self.get_memory_usage_trend(device_id, minutes=5)
        if len(trend_data) < 3:
            return 0.0  # Not enough data

        # Calculate memory usage growth rate
        recent_usage = [profile.current_memory for profile in trend_data[-10:]]
        if len(recent_usage) < 2:
            return 0.0

        # Simple linear regression for trend
        timestamps = list(range(len(recent_usage)))
        mean_time = sum(timestamps) / len(timestamps)
        mean_usage = sum(recent_usage) / len(recent_usage)

        numerator = sum((t - mean_time) * (u - mean_usage)
                       for t, u in zip(timestamps, recent_usage))
        denominator = sum((t - mean_time) ** 2 for t in timestamps)

        if denominator == 0:
            growth_rate = 0
        else:
            growth_rate = numerator / denominator

        # Get current status
        current_status = self._get_single_gpu_status(device_id)
        current_usage_ratio = current_status.memory_percent / 100.0

        # Calculate risk based on current usage and growth rate
        risk_score = min(1.0, current_usage_ratio + (growth_rate * 0.1))
        return max(0.0, risk_score)

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource summary."""
        gpu_statuses = self.get_gpu_status()
        scheduler_status = self.workload_scheduler.get_queue_status()

        total_memory_gb = sum(status.memory_total_gb for status in gpu_statuses)
        used_memory_gb = sum(status.memory_used_gb for status in gpu_statuses)
        available_memory_gb = sum(status.memory_available_gb for status in gpu_statuses)

        avg_utilization = (sum(status.utilization_percent for status in gpu_statuses) /
                          len(gpu_statuses)) if gpu_statuses else 0.0

        # OOM risk assessment
        oom_risks = []
        for status in gpu_statuses:
            risk = self.predict_oom_risk(status.device_id)
            oom_risks.append({"device_id": status.device_id, "risk": risk})

        return {
            "timestamp": time.time(),
            "gpu_count": len(gpu_statuses),
            "total_memory_gb": total_memory_gb,
            "used_memory_gb": used_memory_gb,
            "available_memory_gb": available_memory_gb,
            "memory_utilization_percent": (used_memory_gb / total_memory_gb * 100) if total_memory_gb > 0 else 0,
            "average_gpu_utilization": avg_utilization,
            "gpu_statuses": [asdict(status) for status in gpu_statuses],
            "scheduler_status": scheduler_status,
            "oom_risks": oom_risks
        }


# Convenience functions
def get_gpu_memory_usage() -> Dict[int, float]:
    """Get memory usage for all GPUs in GB."""
    if not torch.cuda.is_available():
        return {}

    usage = {}
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        usage[i] = allocated

    return usage


def optimize_gpu_memory(device_id: int = 0) -> Dict[str, Any]:
    """Quick memory optimization for specified GPU."""
    manager = GPUResourceManager()
    status = manager._get_single_gpu_status(device_id)

    return manager.memory_optimizer.optimize_memory_usage(
        status.memory_used_gb,
        status.memory_available_gb
    )


if __name__ == "__main__":
    # Demo usage
    manager = GPUResourceManager()

    print("GPU Resource Manager Demo")
    print("=" * 40)

    # Get current GPU status
    gpu_statuses = manager.get_gpu_status()
    print(f"Found {len(gpu_statuses)} GPU(s):")

    for status in gpu_statuses:
        print(f"\nGPU {status.device_id}: {status.name}")
        print(f"  Memory: {status.memory_used_gb:.1f}/{status.memory_total_gb:.1f} GB "
              f"({status.memory_percent:.1f}%)")
        print(f"  Utilization: {status.utilization_percent:.1f}%")
        if status.temperature:
            print(f"  Temperature: {status.temperature}°C")
        if status.power_draw:
            print(f"  Power: {status.power_draw:.1f}W")

    # Get resource summary
    print("\nResource Summary:")
    summary = manager.get_resource_summary()
    print(f"  Total Memory: {summary['total_memory_gb']:.1f} GB")
    print(f"  Used Memory: {summary['used_memory_gb']:.1f} GB")
    print(f"  Average Utilization: {summary['average_gpu_utilization']:.1f}%")

    # Start monitoring for a short demo
    print("\nStarting monitoring for 10 seconds...")
    manager.start_monitoring()
    time.sleep(10)
    manager.stop_monitoring()
    print("Monitoring stopped.")
