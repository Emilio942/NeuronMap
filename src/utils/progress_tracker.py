"""Advanced Progress Tracking and ETA Estimation System for NeuronMap.

This module implements comprehensive progress tracking with intelligent ETA calculation,
real-time updates, and multi-level progress visualization according to roadmap section 2.3.
"""

import time
import threading
import logging
import math
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import statistics
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ProgressState(Enum):
    """Progress tracking states."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class ProgressMetrics:
    """Metrics for progress tracking."""
    items_per_second: float = 0.0
    tokens_per_second: float = 0.0
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    memory_usage_gb: float = 0.0
    estimated_time_remaining: float = 0.0


@dataclass
class ProgressSnapshot:
    """Snapshot of progress at a specific time."""
    timestamp: float
    completed_items: int
    total_items: int
    progress_percent: float
    current_step_info: Dict[str, Any] = field(default_factory=dict)
    metrics: ProgressMetrics = field(default_factory=ProgressMetrics)


class ETACalculator:
    """Intelligent ETA calculation with multiple estimation methods."""

    def __init__(self, history_size: int = 50):
        self.history_size = history_size
        self.completion_history: deque = deque(maxlen=history_size)
        self.time_history: deque = deque(maxlen=history_size)

    def update_progress(self, completed: int, timestamp: float):
        """Update progress history for ETA calculation."""
        self.completion_history.append(completed)
        self.time_history.append(timestamp)

    def calculate_eta(self, completed: int, total: int, start_time: float) -> float:
        """
        Calculate estimated time to completion using multiple methods.

        Args:
            completed: Number of completed items
            total: Total number of items
            start_time: Start timestamp

        Returns:
            Estimated seconds to completion
        """
        if completed >= total:
            return 0.0

        remaining = total - completed
        current_time = time.time()

        # Method 1: Simple linear extrapolation
        elapsed = current_time - start_time
        if completed > 0:
            simple_eta = (elapsed / completed) * remaining
        else:
            simple_eta = float('inf')

        # Method 2: Moving average of recent progress
        moving_avg_eta = self._calculate_moving_average_eta(remaining)

        # Method 3: Weighted combination based on data availability
        if len(self.completion_history) < 10:
            # Use simple method for small datasets
            return simple_eta
        else:
            # Combine methods with weighted average
            weight_simple = 0.3
            weight_moving = 0.7
            combined_eta = (weight_simple * simple_eta +
                          weight_moving * moving_avg_eta)
            return combined_eta

    def _calculate_moving_average_eta(self, remaining: int) -> float:
        """Calculate ETA based on moving average of recent progress."""
        if len(self.completion_history) < 2:
            return float('inf')

        # Calculate completion rates for recent history
        rates = []
        for i in range(1, len(self.completion_history)):
            time_delta = self.time_history[i] - self.time_history[i-1]
            completion_delta = self.completion_history[i] - self.completion_history[i-1]
            if time_delta > 0:
                rate = completion_delta / time_delta
                rates.append(rate)

        if not rates:
            return float('inf')

        # Use recent rates with exponential weighting
        weighted_rate = 0.0
        total_weight = 0.0
        for i, rate in enumerate(rates):
            weight = math.exp(-0.1 * (len(rates) - i - 1))  # More weight to recent rates
            weighted_rate += rate * weight
            total_weight += weight

        if total_weight > 0:
            avg_rate = weighted_rate / total_weight
            if avg_rate > 0:
                return remaining / avg_rate

        return float('inf')


class ProgressReporter:
    """Multi-level progress reporting with real-time updates."""

    def __init__(self, update_callback: Optional[Callable] = None):
        self.update_callback = update_callback
        self.last_report_time = 0.0
        self.min_report_interval = 0.1  # Minimum 100ms between reports

    def report_progress(self, progress: float, eta: float,
                       step_info: Dict[str, Any] = None,
                       metrics: ProgressMetrics = None):
        """
        Report progress update with throttling to avoid spam.

        Args:
            progress: Progress as fraction (0.0 to 1.0)
            eta: Estimated time to completion in seconds
            step_info: Additional information about current step
            metrics: Performance metrics
        """
        current_time = time.time()

        # Throttle updates to avoid performance impact
        if current_time - self.last_report_time < self.min_report_interval:
            return

        self.last_report_time = current_time

        # Format progress information
        progress_percent = progress * 100
        eta_formatted = self._format_eta(eta)

        report = {
            'progress_percent': progress_percent,
            'eta_seconds': eta,
            'eta_formatted': eta_formatted,
            'timestamp': current_time,
            'step_info': step_info or {},
            'metrics': metrics or ProgressMetrics()
        }

        # Send to callback if provided
        if self.update_callback:
            try:
                self.update_callback(report)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

        # Log progress periodically
        if progress_percent % 10 < 1:  # Log every ~10%
            logger.info(f"Progress: {progress_percent:.1f}% (ETA: {eta_formatted})")

    def _format_eta(self, eta_seconds: float) -> str:
        """Format ETA in human-readable format."""
        if eta_seconds == float('inf') or eta_seconds < 0:
            return "Unknown"

        if eta_seconds < 60:
            return f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            minutes = eta_seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = eta_seconds / 3600
            return f"{hours:.1f}h"


class ProgressTracker:
    """Main progress tracking system with multi-level support."""

    def __init__(self, total_steps: int, operation_name: str,
                 update_callback: Optional[Callable] = None,
                 enable_metrics: bool = True):
        """
        Initialize progress tracker.

        Args:
            total_steps: Total number of steps to complete
            operation_name: Name of the operation being tracked
            update_callback: Optional callback for progress updates
            enable_metrics: Whether to collect performance metrics
        """
        self.total_steps = total_steps
        self.operation_name = operation_name
        self.completed_steps = 0
        self.start_time = time.time()
        self.state = ProgressState.NOT_STARTED
        self.enable_metrics = enable_metrics

        # Components
        self.eta_calculator = ETACalculator()
        self.progress_reporter = ProgressReporter(update_callback)

        # Multi-level tracking
        self.sub_trackers: Dict[str, 'ProgressTracker'] = {}
        self.parent_tracker: Optional['ProgressTracker'] = None

        # Metrics collection
        self.metrics_history: List[ProgressSnapshot] = []
        self.current_metrics = ProgressMetrics()

        # Cancellation support
        self._cancelled = threading.Event()
        self._lock = threading.Lock()

        logger.info(f"Progress tracker initialized for '{operation_name}' ({total_steps} steps)")

    def start(self):
        """Start progress tracking."""
        with self._lock:
            if self.state != ProgressState.NOT_STARTED:
                logger.warning(f"Progress tracker already started for '{self.operation_name}'")
                return

            self.state = ProgressState.RUNNING
            self.start_time = time.time()
            logger.info(f"Started tracking progress for '{self.operation_name}'")

    def update(self, steps_completed: int = 1,
               current_step_info: Dict[str, Any] = None,
               force_report: bool = False):
        """
        Update progress by specified number of steps.

        Args:
            steps_completed: Number of steps completed in this update
            current_step_info: Information about current processing step
            force_report: Force progress report even if throttled
        """
        if self._cancelled.is_set():
            raise InterruptedError("Operation was cancelled")

        with self._lock:
            if self.state != ProgressState.RUNNING:
                return

            self.completed_steps = min(self.completed_steps + steps_completed, self.total_steps)
            current_time = time.time()

            # Update ETA calculator
            self.eta_calculator.update_progress(self.completed_steps, current_time)

            # Calculate progress and ETA
            progress = self.completed_steps / self.total_steps if self.total_steps > 0 else 1.0
            eta = self.eta_calculator.calculate_eta(
                self.completed_steps, self.total_steps, self.start_time
            )

            # Collect metrics if enabled
            if self.enable_metrics:
                self._update_metrics(current_step_info)

            # Create progress snapshot
            snapshot = ProgressSnapshot(
                timestamp=current_time,
                completed_items=self.completed_steps,
                total_items=self.total_steps,
                progress_percent=progress * 100,
                current_step_info=current_step_info or {},
                metrics=self.current_metrics
            )
            self.metrics_history.append(snapshot)

            # Keep history size manageable
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-500:]

            # Report progress
            if force_report or True:  # Always try to report (throttling is in reporter)
                self.progress_reporter.report_progress(
                    progress, eta, current_step_info, self.current_metrics
                )

            # Update parent tracker if we have one
            if self.parent_tracker:
                self.parent_tracker._update_from_subtask(self.operation_name, progress)

            # Check if completed
            if self.completed_steps >= self.total_steps:
                self.complete()

    def _update_metrics(self, step_info: Dict[str, Any] = None):
        """Update performance metrics."""
        try:
            import psutil

            # System metrics
            self.current_metrics.cpu_utilization = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            self.current_metrics.memory_usage_gb = memory.used / (1024**3)

            # GPU metrics (if available)
            try:
                import torch
                if torch.cuda.is_available():
                    self.current_metrics.gpu_utilization = self._get_gpu_utilization()
            except ImportError:
                pass

            # Throughput metrics
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.current_metrics.items_per_second = self.completed_steps / elapsed

                # Calculate tokens per second if available
                if step_info and 'tokens_processed' in step_info:
                    total_tokens = step_info['tokens_processed']
                    self.current_metrics.tokens_per_second = total_tokens / elapsed

        except Exception as e:
            logger.debug(f"Error updating metrics: {e}")

    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu
        except:
            return 0.0

    def add_subtask(self, name: str, total_steps: int) -> 'ProgressTracker':
        """
        Add a subtask tracker for multi-level progress tracking.

        Args:
            name: Name of the subtask
            total_steps: Total steps for the subtask

        Returns:
            ProgressTracker instance for the subtask
        """
        subtask = ProgressTracker(
            total_steps=total_steps,
            operation_name=f"{self.operation_name}.{name}",
            enable_metrics=False  # Avoid duplicate metrics collection
        )
        subtask.parent_tracker = self
        self.sub_trackers[name] = subtask

        logger.debug(f"Added subtask '{name}' with {total_steps} steps")
        return subtask

    def _update_from_subtask(self, subtask_name: str, subtask_progress: float):
        """Update progress based on subtask completion."""
        # This could implement weighted progress updates from subtasks
        # For now, we just log the subtask progress
        logger.debug(f"Subtask '{subtask_name}' progress: {subtask_progress:.1%}")

    def pause(self):
        """Pause progress tracking."""
        with self._lock:
            if self.state == ProgressState.RUNNING:
                self.state = ProgressState.PAUSED
                logger.info(f"Paused progress tracking for '{self.operation_name}'")

    def resume(self):
        """Resume progress tracking."""
        with self._lock:
            if self.state == ProgressState.PAUSED:
                self.state = ProgressState.RUNNING
                logger.info(f"Resumed progress tracking for '{self.operation_name}'")

    def cancel(self):
        """Cancel the operation gracefully."""
        with self._lock:
            self.state = ProgressState.CANCELLED
            self._cancelled.set()
            logger.info(f"Cancelled progress tracking for '{self.operation_name}'")

    def complete(self):
        """Mark operation as completed."""
        with self._lock:
            if self.state in [ProgressState.COMPLETED, ProgressState.CANCELLED]:
                return

            self.state = ProgressState.COMPLETED
            elapsed = time.time() - self.start_time

            logger.info(f"Completed '{self.operation_name}' in {elapsed:.1f}s "
                       f"({self.current_metrics.items_per_second:.1f} items/s)")

            # Final progress report
            self.progress_reporter.report_progress(
                1.0, 0.0, {'status': 'completed'}, self.current_metrics
            )

    def get_status(self) -> Dict[str, Any]:
        """Get current status information."""
        with self._lock:
            elapsed = time.time() - self.start_time
            progress = self.completed_steps / self.total_steps if self.total_steps > 0 else 1.0
            eta = self.eta_calculator.calculate_eta(
                self.completed_steps, self.total_steps, self.start_time
            )

            return {
                'operation_name': self.operation_name,
                'state': self.state.value,
                'progress_percent': progress * 100,
                'completed_steps': self.completed_steps,
                'total_steps': self.total_steps,
                'elapsed_seconds': elapsed,
                'eta_seconds': eta,
                'eta_formatted': self.progress_reporter._format_eta(eta),
                'metrics': self.current_metrics,
                'subtasks': {name: tracker.get_status()
                           for name, tracker in self.sub_trackers.items()}
            }

    @contextmanager
    def track_operation(self, operation_name: str, total_items: int):
        """Context manager for tracking a specific operation."""
        subtask = self.add_subtask(operation_name, total_items)
        subtask.start()
        try:
            yield subtask
        except Exception as e:
            subtask.state = ProgressState.ERROR
            logger.error(f"Operation '{operation_name}' failed: {e}")
            raise
        finally:
            if subtask.state == ProgressState.RUNNING:
                subtask.complete()


class MultiLevelProgressTracker:
    """Multi-level hierarchical progress tracker for complex operations."""

    def __init__(self, operation_name: str):
        """Initialize multi-level progress tracker.

        Args:
            operation_name: Name of the overall operation
        """
        self.operation_name = operation_name
        self.levels: Dict[str, Dict[str, Any]] = {}
        self.level_order: List[str] = []
        self.start_time = time.time()
        self.lock = threading.Lock()

        logger.info(f"Multi-level progress tracker initialized for: {operation_name}")

    def add_level(self, level_name: str, total_steps: int, description: str = ""):
        """Add a new level to the tracker.

        Args:
            level_name: Unique name for this level
            total_steps: Total number of steps for this level
            description: Human-readable description
        """
        with self.lock:
            self.levels[level_name] = {
                'total_steps': total_steps,
                'completed_steps': 0,
                'description': description,
                'start_time': None,
                'current_step_info': {}
            }

            if level_name not in self.level_order:
                self.level_order.append(level_name)

            logger.debug(f"Added level '{level_name}' with {total_steps} steps")

    def update_level(self, level_name: str, steps_completed: int,
                    current_step_info: Optional[Dict[str, Any]] = None):
        """Update progress for a specific level.

        Args:
            level_name: Name of the level to update
            steps_completed: Number of steps completed in this update
            current_step_info: Optional information about current step
        """
        with self.lock:
            if level_name not in self.levels:
                raise ValueError(f"Level '{level_name}' not found")

            level = self.levels[level_name]

            # Start tracking this level if not already started
            if level['start_time'] is None:
                level['start_time'] = time.time()

            level['completed_steps'] += steps_completed
            level['completed_steps'] = min(level['completed_steps'], level['total_steps'])

            if current_step_info:
                level['current_step_info'] = current_step_info

    def reset_level(self, level_name: str):
        """Reset a level's progress (useful for loops).

        Args:
            level_name: Name of the level to reset
        """
        with self.lock:
            if level_name in self.levels:
                self.levels[level_name]['completed_steps'] = 0
                self.levels[level_name]['start_time'] = None
                self.levels[level_name]['current_step_info'] = {}

    def get_level_status(self, level_name: str) -> Optional[Dict[str, Any]]:
        """Get status for a specific level.

        Args:
            level_name: Name of the level

        Returns:
            Dictionary with level status information
        """
        with self.lock:
            return self._get_level_status_unlocked(level_name)

    def _get_level_status_unlocked(self, level_name: str) -> Optional[Dict[str, Any]]:
        """Get status for a specific level without acquiring lock.

        Args:
            level_name: Name of the level

        Returns:
            Dictionary with level status information
        """
        if level_name not in self.levels:
            return None

        level = self.levels[level_name]

        if level['total_steps'] == 0:
            progress_percent = 100.0
        else:
            progress_percent = (level['completed_steps'] / level['total_steps']) * 100

        return {
            'level_name': level_name,
            'description': level['description'],
            'total_steps': level['total_steps'],
            'completed_steps': level['completed_steps'],
            'progress_percent': progress_percent,
            'current_step_info': level['current_step_info'],
            'is_complete': level['completed_steps'] >= level['total_steps']
        }

    def get_overall_status(self) -> Dict[str, Any]:
        """Get overall status across all levels.

        Returns:
            Dictionary with overall progress information
        """
        with self.lock:
            if not self.levels:
                return {
                    'operation_name': self.operation_name,
                    'overall_progress_percent': 0.0,
                    'overall_eta_seconds': 0.0,
                    'levels': [],
                    'elapsed_time': time.time() - self.start_time
                }

            # Calculate overall progress as weighted average
            total_weight = sum(level['total_steps'] for level in self.levels.values())
            if total_weight == 0:
                overall_progress = 100.0
            else:
                weighted_progress = sum(
                    (level['completed_steps'] / level['total_steps']) * level['total_steps']
                    for level in self.levels.values()
                    if level['total_steps'] > 0
                )
                overall_progress = (weighted_progress / total_weight) * 100

            # Calculate ETA based on current progress rate
            elapsed_time = time.time() - self.start_time
            if overall_progress > 0 and elapsed_time > 0:
                time_per_percent = elapsed_time / overall_progress
                remaining_percent = 100 - overall_progress
                eta_seconds = time_per_percent * remaining_percent
            else:
                eta_seconds = 0.0

            # Get status for all levels
            level_statuses = []
            for level_name in self.level_order:
                status = self._get_level_status_unlocked(level_name)
                if status:
                    level_statuses.append(status)

            return {
                'operation_name': self.operation_name,
                'overall_progress_percent': overall_progress,
                'overall_eta_seconds': eta_seconds,
                'elapsed_time': elapsed_time,
                'levels': level_statuses,
                'total_levels': len(self.levels),
                'completed_levels': sum(1 for level in self.levels.values()
                                      if level['completed_steps'] >= level['total_steps'])
            }

    def print_status(self, show_all_levels: bool = True):
        """Print current status to console.

        Args:
            show_all_levels: Whether to show details for all levels
        """
        status = self.get_overall_status()

        print(f"\n{self.operation_name}: {status['overall_progress_percent']:.1f}% complete")
        print(f"ETA: {status['overall_eta_seconds']:.1f}s, Elapsed: {status['elapsed_time']:.1f}s")
        print(f"Levels: {status['completed_levels']}/{status['total_levels']} complete")

        if show_all_levels:
            for level_status in status['levels']:
                print(f"  {level_status['level_name']}: {level_status['progress_percent']:.1f}% "
                      f"({level_status['completed_steps']}/{level_status['total_steps']})")

    def is_complete(self) -> bool:
        """Check if all levels are complete.

        Returns:
            True if all levels are complete
        """
        with self.lock:
            return all(
                level['completed_steps'] >= level['total_steps']
                for level in self.levels.values()
            )


# Convenience functions for easy usage
def track_progress(total_steps: int, operation_name: str = "Operation",
                  update_callback: Optional[Callable] = None) -> ProgressTracker:
    """Create and start a progress tracker."""
    tracker = ProgressTracker(total_steps, operation_name, update_callback)
    tracker.start()
    return tracker


@contextmanager
def progress_context(total_steps: int, operation_name: str = "Operation",
                    update_callback: Optional[Callable] = None):
    """Context manager for automatic progress tracking."""
    tracker = track_progress(total_steps, operation_name, update_callback)
    try:
        yield tracker
    except Exception as e:
        tracker.state = ProgressState.ERROR
        logger.error(f"Operation '{operation_name}' failed: {e}")
        raise
    finally:
        if tracker.state == ProgressState.RUNNING:
            tracker.complete()


# Demo function for multi-level tracking
def demo_multi_level_tracking():
    """Demonstrate multi-level progress tracking."""
    print("Multi-Level Progress Tracking Demo")
    print("==================================")

    tracker = MultiLevelProgressTracker("Neural Network Training")

    # Add levels for different aspects of training
    tracker.add_level("epochs", 3, "Training Epochs")
    tracker.add_level("batches", 100, "Batches per Epoch")
    tracker.add_level("samples", 1000, "Samples per Batch")

    for epoch in range(3):
        tracker.update_level("epochs", 1, {"current_epoch": epoch + 1})

        for batch in range(100):
            tracker.update_level("batches", 1, {"current_batch": batch + 1})

            for sample in range(1000):
                tracker.update_level("samples", 1)

                if sample % 250 == 0:  # Print status periodically
                    tracker.print_status(show_all_levels=False)

                time.sleep(0.0001)  # Simulate processing

            # Reset samples for next batch
            tracker.reset_level("samples")

        # Reset batches for next epoch
        tracker.reset_level("batches")

    print("\nTraining completed!")
    final_status = tracker.get_overall_status()
    print(f"Total time: {final_status['elapsed_time']:.2f}s")


if __name__ == "__main__":
    # Run demonstration
    print("Progress Tracker Demo")
    demo_multi_level_tracking()
