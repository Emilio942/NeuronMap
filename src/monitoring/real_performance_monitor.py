#!/usr/bin/env python3
"""
📊 NeuronMap REAL Performance Monitor v2.0
Echtes System Performance Monitoring mit Prometheus Integration

ERSETZT: MockPerformanceMonitor
NEUE FEATURES:
- Echte System Resource Monitoring (CPU, Memory, Disk, Network)
- Prometheus Metrics Export
- Real-time Performance Alerts
- Task Performance Tracking mit echten Benchmarks
- Production-Ready Health Checks
"""

import psutil
import time
import threading
import logging
import sqlite3
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import sys
import os

# Prometheus client imports mit fallback
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Setup Advanced Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RealSystemStats:
    """ECHTE System Performance Statistics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    memory_used_gb: float
    disk_usage_percent: float
    disk_read_mb_per_sec: float
    disk_write_mb_per_sec: float
    network_sent_mb_per_sec: float
    network_recv_mb_per_sec: float
    load_average_1min: float
    processes_count: int
    threads_count: int
    gpu_count: int = 0
    gpu_memory_used_percent: float = 0.0
    temperature_celsius: Optional[float] = None

@dataclass
class RealTaskStats:
    """ECHTE Task Performance Statistics"""
    task_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    peak_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    total_disk_read_mb: float = 0.0
    total_disk_write_mb: float = 0.0
    status: str = "running"  # running, completed, failed
    error_message: Optional[str] = None
    progress_percent: float = 0.0
    eta_seconds: Optional[float] = None

class RealPrometheusExporter:
    """
    📈 ECHTER Prometheus Metrics Exporter
    
    Exportiert echte System- und Performance-Metriken
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and PROMETHEUS_AVAILABLE
        
        if self.enabled:
            self.registry = CollectorRegistry()
            
            # System metrics
            self.cpu_usage = Gauge('neuronmap_cpu_usage_percent', 
                                 'CPU usage percentage', registry=self.registry)
            self.memory_usage = Gauge('neuronmap_memory_usage_percent', 
                                    'Memory usage percentage', registry=self.registry)
            self.disk_usage = Gauge('neuronmap_disk_usage_percent', 
                                  'Disk usage percentage', registry=self.registry)
            
            # Performance metrics
            self.task_duration = Histogram('neuronmap_task_duration_seconds', 
                                         'Task execution duration', 
                                         buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0],
                                         registry=self.registry)
            self.task_memory_peak = Gauge('neuronmap_task_memory_peak_mb', 
                                        'Peak memory usage per task', registry=self.registry)
            
            # Error counters
            self.errors_total = Counter('neuronmap_errors_total', 
                                      'Total number of errors', 
                                      labelnames=['error_type'], registry=self.registry)
            
            logger.info("✅ Prometheus exporter initialized")
        else:
            logger.warning("⚠️ Prometheus not available - install with: pip install prometheus-client")
    
    def update_system_metrics(self, stats: RealSystemStats):
        """Update Prometheus system metrics"""
        if not self.enabled:
            return
        
        self.cpu_usage.set(stats.cpu_percent)
        self.memory_usage.set(stats.memory_percent)
        self.disk_usage.set(stats.disk_usage_percent)
    
    def record_task_completion(self, task: RealTaskStats):
        """Record task completion metrics"""
        if not self.enabled or not task.duration_seconds:
            return
        
        self.task_duration.observe(task.duration_seconds)
        self.task_memory_peak.set(task.peak_memory_mb)
        
        if task.status == "failed":
            self.errors_total.labels(error_type="task_failure").inc()
    
    def get_metrics_output(self) -> str:
        """Get Prometheus metrics output"""
        if not self.enabled:
            return "# Prometheus not available\n"
        
        return generate_latest(self.registry).decode('utf-8')

class RealPerformanceMonitor:
    """
    🚀 ECHTER Performance Monitor mit echten System-Metriken
    
    Features:
    - Echte System Resource Monitoring
    - Task Performance Tracking
    - Real-time Alerting
    - Prometheus Integration
    - Historical Data Storage
    - Performance Predictions
    """
    
    def __init__(self, 
                 monitoring_interval: float = 1.0,
                 data_retention_days: int = 30,
                 enable_prometheus: bool = True,
                 storage_dir: str = "./monitoring"):
        
        self.monitoring_interval = monitoring_interval
        self.data_retention_days = data_retention_days
        self.storage_dir = Path(storage_dir)
        
        # Initialize storage
        self._init_storage()
        
        # Initialize Prometheus exporter
        self.prometheus = RealPrometheusExporter(enable_prometheus)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Task tracking
        self.active_tasks = {}
        self.task_history = []
        self._task_lock = threading.Lock()
        
        # Performance baselines
        self.baseline_stats = None
        self.performance_alerts = []
        
        # Initialize baseline disk I/O counters
        self._last_disk_io = psutil.disk_io_counters()
        self._last_network_io = psutil.net_io_counters()
        self._last_check_time = time.time()
        
        logger.info(f"🚀 Real Performance Monitor initialized: "
                   f"interval={monitoring_interval}s, "
                   f"retention={data_retention_days}d, "
                   f"prometheus={'✅' if self.prometheus.enabled else '❌'}")
    
    def _init_storage(self):
        """Initialize SQLite storage for performance data"""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_dir / 'performance_metrics.db'
        
        with sqlite3.connect(str(self.db_path)) as conn:
            # System stats table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_stats (
                    timestamp REAL PRIMARY KEY,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_available_gb REAL,
                    memory_used_gb REAL,
                    disk_usage_percent REAL,
                    disk_read_mb_per_sec REAL,
                    disk_write_mb_per_sec REAL,
                    network_sent_mb_per_sec REAL,
                    network_recv_mb_per_sec REAL,
                    load_average_1min REAL,
                    processes_count INTEGER,
                    threads_count INTEGER,
                    gpu_count INTEGER,
                    gpu_memory_used_percent REAL,
                    temperature_celsius REAL
                )
            ''')
            
            # Task stats table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS task_stats (
                    task_id TEXT PRIMARY KEY,
                    name TEXT,
                    start_time REAL,
                    end_time REAL,
                    duration_seconds REAL,
                    peak_memory_mb REAL,
                    peak_cpu_percent REAL,
                    total_disk_read_mb REAL,
                    total_disk_write_mb REAL,
                    status TEXT,
                    error_message TEXT,
                    progress_percent REAL,
                    eta_seconds REAL
                )
            ''')
            
            conn.commit()
        
        logger.info(f"💾 Performance database initialized: {self.db_path}")
    
    def start_monitoring(self):
        """Start real-time system monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("📊 Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time system monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("⏹️ Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect real system stats
                stats = self._collect_system_stats()
                
                # Store in database
                self._store_system_stats(stats)
                
                # Update Prometheus metrics
                self.prometheus.update_system_metrics(stats)
                
                # Check for performance alerts
                self._check_performance_alerts(stats)
                
                # Update baseline if needed
                self._update_baseline(stats)
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)  # Brief pause before retry
    
    def _collect_system_stats(self) -> RealSystemStats:
        """Collect ECHTE system performance statistics"""
        current_time = time.time()
        
        # CPU stats
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory stats
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        memory_used_gb = memory.used / (1024**3)
        
        # Disk stats
        disk_usage = psutil.disk_usage('/')
        disk_usage_percent = disk_usage.percent
        
        # Disk I/O rates
        current_disk_io = psutil.disk_io_counters()
        time_delta = current_time - self._last_check_time
        
        if self._last_disk_io and time_delta > 0:
            disk_read_mb_per_sec = (current_disk_io.read_bytes - self._last_disk_io.read_bytes) / (1024**2) / time_delta
            disk_write_mb_per_sec = (current_disk_io.write_bytes - self._last_disk_io.write_bytes) / (1024**2) / time_delta
        else:
            disk_read_mb_per_sec = 0.0
            disk_write_mb_per_sec = 0.0
        
        # Network I/O rates
        current_network_io = psutil.net_io_counters()
        
        if self._last_network_io and time_delta > 0:
            network_sent_mb_per_sec = (current_network_io.bytes_sent - self._last_network_io.bytes_sent) / (1024**2) / time_delta
            network_recv_mb_per_sec = (current_network_io.bytes_recv - self._last_network_io.bytes_recv) / (1024**2) / time_delta
        else:
            network_sent_mb_per_sec = 0.0
            network_recv_mb_per_sec = 0.0
        
        # Process stats
        processes_count = len(psutil.pids())
        
        # Thread count (sum across all processes)
        threads_count = 0
        try:
            for proc in psutil.process_iter(['num_threads']):
                threads_count += proc.info['num_threads'] or 0
        except:
            threads_count = 0
        
        # Load average (Linux/Unix only)
        try:
            load_average_1min = os.getloadavg()[0]
        except (OSError, AttributeError):
            load_average_1min = 0.0
        
        # GPU stats (if available)
        gpu_count = 0
        gpu_memory_used_percent = 0.0
        temperature_celsius = None
        
        try:
            # Try to get GPU stats from real GPU optimizer
            from .real_gpu_optimizer import get_real_gpu_optimizer
            gpu_optimizer = get_real_gpu_optimizer()
            gpu_stats = gpu_optimizer.get_real_gpu_stats()
            
            if gpu_stats:
                gpu_count = len(gpu_stats)
                gpu_memory_used_percent = sum(gpu.utilization_percent for gpu in gpu_stats) / len(gpu_stats)
                
                # Get average temperature
                temps = [gpu.temperature_celsius for gpu in gpu_stats if gpu.temperature_celsius]
                if temps:
                    temperature_celsius = sum(temps) / len(temps)
        except:
            pass  # GPU stats not available
        
        # Update last check values
        self._last_disk_io = current_disk_io
        self._last_network_io = current_network_io
        self._last_check_time = current_time
        
        return RealSystemStats(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            memory_used_gb=memory_used_gb,
            disk_usage_percent=disk_usage_percent,
            disk_read_mb_per_sec=disk_read_mb_per_sec,
            disk_write_mb_per_sec=disk_write_mb_per_sec,
            network_sent_mb_per_sec=network_sent_mb_per_sec,
            network_recv_mb_per_sec=network_recv_mb_per_sec,
            load_average_1min=load_average_1min,
            processes_count=processes_count,
            threads_count=threads_count,
            gpu_count=gpu_count,
            gpu_memory_used_percent=gpu_memory_used_percent,
            temperature_celsius=temperature_celsius
        )
    
    def _store_system_stats(self, stats: RealSystemStats):
        """Store system stats in database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT INTO system_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    stats.timestamp, stats.cpu_percent, stats.memory_percent,
                    stats.memory_available_gb, stats.memory_used_gb, stats.disk_usage_percent,
                    stats.disk_read_mb_per_sec, stats.disk_write_mb_per_sec,
                    stats.network_sent_mb_per_sec, stats.network_recv_mb_per_sec,
                    stats.load_average_1min, stats.processes_count, stats.threads_count,
                    stats.gpu_count, stats.gpu_memory_used_percent, stats.temperature_celsius
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to store system stats: {e}")
    
    def start_task_monitoring(self, task_id: str, task_name: str) -> RealTaskStats:
        """
        🎯 Start monitoring a specific task
        
        Args:
            task_id: Unique task identifier
            task_name: Human-readable task name
            
        Returns:
            Task stats object for tracking
        """
        with self._task_lock:
            task_stats = RealTaskStats(
                task_id=task_id,
                name=task_name,
                start_time=time.time(),
                status="running"
            )
            
            self.active_tasks[task_id] = task_stats
            logger.info(f"📊 Started monitoring task: {task_name} ({task_id})")
            
            return task_stats
    
    def update_task_progress(self, task_id: str, 
                           progress_percent: float,
                           eta_seconds: Optional[float] = None):
        """Update task progress and ETA"""
        with self._task_lock:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.progress_percent = progress_percent
                task.eta_seconds = eta_seconds
                
                # Update peak resource usage
                try:
                    # Get current process stats
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / (1024**2)
                    cpu_percent = process.cpu_percent()
                    
                    task.peak_memory_mb = max(task.peak_memory_mb, memory_mb)
                    task.peak_cpu_percent = max(task.peak_cpu_percent, cpu_percent)
                    
                    # Update disk I/O (approximate)
                    io_counters = process.io_counters()
                    task.total_disk_read_mb = io_counters.read_bytes / (1024**2)
                    task.total_disk_write_mb = io_counters.write_bytes / (1024**2)
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass  # Process stats not available
    
    def complete_task_monitoring(self, task_id: str, 
                               success: bool = True,
                               error_message: Optional[str] = None):
        """
        ✅ Complete task monitoring and record final stats
        
        Args:
            task_id: Task identifier
            success: Whether task completed successfully
            error_message: Error message if task failed
        """
        with self._task_lock:
            if task_id not in self.active_tasks:
                logger.warning(f"Task {task_id} not found in active tasks")
                return
            
            task = self.active_tasks[task_id]
            task.end_time = time.time()
            task.duration_seconds = task.end_time - task.start_time
            task.status = "completed" if success else "failed"
            task.error_message = error_message
            task.progress_percent = 100.0 if success else task.progress_percent
            
            # Move to history
            self.task_history.append(task)
            del self.active_tasks[task_id]
            
            # Store in database
            self._store_task_stats(task)
            
            # Update Prometheus metrics
            self.prometheus.record_task_completion(task)
            
            status_emoji = "✅" if success else "❌"
            logger.info(f"{status_emoji} Task completed: {task.name} "
                       f"({task.duration_seconds:.2f}s, "
                       f"peak memory: {task.peak_memory_mb:.1f}MB)")
    
    def _store_task_stats(self, task: RealTaskStats):
        """Store task stats in database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO task_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    task.task_id, task.name, task.start_time, task.end_time,
                    task.duration_seconds, task.peak_memory_mb, task.peak_cpu_percent,
                    task.total_disk_read_mb, task.total_disk_write_mb, task.status,
                    task.error_message, task.progress_percent, task.eta_seconds
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to store task stats: {e}")
    
    def _check_performance_alerts(self, stats: RealSystemStats):
        """Check for performance alerts and warnings"""
        alerts = []
        
        # High CPU usage
        if stats.cpu_percent > 90:
            alerts.append(f"🔴 HIGH CPU: {stats.cpu_percent:.1f}%")
        
        # High memory usage
        if stats.memory_percent > 90:
            alerts.append(f"🔴 HIGH MEMORY: {stats.memory_percent:.1f}%")
        
        # Low disk space
        if stats.disk_usage_percent > 95:
            alerts.append(f"🔴 DISK FULL: {stats.disk_usage_percent:.1f}%")
        
        # High disk I/O
        if stats.disk_read_mb_per_sec > 100 or stats.disk_write_mb_per_sec > 100:
            alerts.append(f"🟡 HIGH DISK I/O: R={stats.disk_read_mb_per_sec:.1f}MB/s "
                         f"W={stats.disk_write_mb_per_sec:.1f}MB/s")
        
        # High load average
        if stats.load_average_1min > psutil.cpu_count() * 2:
            alerts.append(f"🟡 HIGH LOAD: {stats.load_average_1min:.1f}")
        
        # GPU temperature warnings
        if stats.temperature_celsius and stats.temperature_celsius > 85:
            alerts.append(f"🔥 GPU HOT: {stats.temperature_celsius:.1f}°C")
        
        # Log new alerts
        for alert in alerts:
            if alert not in self.performance_alerts:
                logger.warning(f"PERFORMANCE ALERT: {alert}")
                self.performance_alerts.append(alert)
        
        # Remove resolved alerts
        self.performance_alerts = [alert for alert in self.performance_alerts 
                                  if alert in alerts]
    
    def _update_baseline(self, stats: RealSystemStats):
        """Update performance baseline for comparison"""
        if self.baseline_stats is None:
            self.baseline_stats = stats
        else:
            # Simple moving average for baseline
            alpha = 0.01  # Low alpha for stable baseline
            self.baseline_stats.cpu_percent = (
                alpha * stats.cpu_percent + (1 - alpha) * self.baseline_stats.cpu_percent
            )
            self.baseline_stats.memory_percent = (
                alpha * stats.memory_percent + (1 - alpha) * self.baseline_stats.memory_percent
            )
    
    def get_current_stats(self) -> RealSystemStats:
        """Get current system performance statistics"""
        return self._collect_system_stats()
    
    def get_historical_stats(self, hours: int = 24) -> List[RealSystemStats]:
        """Get historical system stats from database"""
        cutoff_time = time.time() - (hours * 3600)
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute('''
                    SELECT * FROM system_stats 
                    WHERE timestamp > ? 
                    ORDER BY timestamp
                ''', (cutoff_time,))
                
                stats = []
                for row in cursor.fetchall():
                    stats.append(RealSystemStats(*row))
                
                return stats
        except Exception as e:
            logger.warning(f"Failed to get historical stats: {e}")
            return []
    
    def get_task_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive task performance report"""
        with self._task_lock:
            active_count = len(self.active_tasks)
            completed_tasks = [t for t in self.task_history if t.status == "completed"]
            failed_tasks = [t for t in self.task_history if t.status == "failed"]
            
            # Calculate statistics
            if completed_tasks:
                avg_duration = sum(t.duration_seconds for t in completed_tasks) / len(completed_tasks)
                avg_memory = sum(t.peak_memory_mb for t in completed_tasks) / len(completed_tasks)
                total_tasks = len(completed_tasks) + len(failed_tasks)
                success_rate = len(completed_tasks) / total_tasks if total_tasks > 0 else 0.0
            else:
                avg_duration = 0.0
                avg_memory = 0.0
                success_rate = 0.0
            
            return {
                "active_tasks": active_count,
                "completed_tasks": len(completed_tasks),
                "failed_tasks": len(failed_tasks),
                "success_rate": success_rate,
                "avg_duration_seconds": avg_duration,
                "avg_peak_memory_mb": avg_memory,
                "current_alerts": self.performance_alerts.copy(),
                "monitoring_active": self.monitoring_active
            }
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        current_stats = self.get_current_stats()
        
        # Health scoring
        health_score = 100
        
        if current_stats.cpu_percent > 80:
            health_score -= 20
        if current_stats.memory_percent > 80:
            health_score -= 20
        if current_stats.disk_usage_percent > 90:
            health_score -= 30
        if current_stats.load_average_1min > psutil.cpu_count() * 1.5:
            health_score -= 15
        if current_stats.temperature_celsius and current_stats.temperature_celsius > 80:
            health_score -= 15
        
        health_status = "excellent" if health_score >= 90 else \
                       "good" if health_score >= 70 else \
                       "warning" if health_score >= 50 else "critical"
        
        return {
            "health_score": max(0, health_score),
            "health_status": health_status,
            "current_stats": asdict(current_stats),
            "baseline_stats": asdict(self.baseline_stats) if self.baseline_stats else None,
            "active_alerts": self.performance_alerts.copy(),
            "prometheus_enabled": self.prometheus.enabled,
            "uptime_hours": (time.time() - current_stats.timestamp) / 3600 if hasattr(self, 'start_time') else 0
        }
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics output"""
        return self.prometheus.get_metrics_output()
    
    def cleanup_old_data(self):
        """Clean up old performance data"""
        cutoff_time = time.time() - (self.data_retention_days * 24 * 3600)
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Clean system stats
                cursor = conn.execute('DELETE FROM system_stats WHERE timestamp < ?', (cutoff_time,))
                system_deleted = cursor.rowcount
                
                # Clean task stats
                cursor = conn.execute('DELETE FROM task_stats WHERE start_time < ?', (cutoff_time,))
                task_deleted = cursor.rowcount
                
                conn.commit()
                
                if system_deleted > 0 or task_deleted > 0:
                    logger.info(f"🧹 Cleaned up old data: {system_deleted} system stats, "
                               f"{task_deleted} task stats")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

# Global monitor instance
_real_performance_monitor = None

def get_real_performance_monitor() -> RealPerformanceMonitor:
    """Get global real performance monitor instance"""
    global _real_performance_monitor
    if _real_performance_monitor is None:
        _real_performance_monitor = RealPerformanceMonitor()
    return _real_performance_monitor

def start_real_monitoring():
    """Quick function to start real performance monitoring"""
    monitor = get_real_performance_monitor()
    monitor.start_monitoring()

def get_real_system_stats() -> RealSystemStats:
    """Quick function to get current real system stats"""
    monitor = get_real_performance_monitor()
    return monitor.get_current_stats()

def monitor_task(task_id: str, task_name: str) -> RealTaskStats:
    """Quick function to start task monitoring"""
    monitor = get_real_performance_monitor()
    return monitor.start_task_monitoring(task_id, task_name)

def complete_task(task_id: str, success: bool = True, error: Optional[str] = None):
    """Quick function to complete task monitoring"""
    monitor = get_real_performance_monitor()
    monitor.complete_task_monitoring(task_id, success, error)

def get_real_health_report() -> Dict[str, Any]:
    """Quick function to get system health report"""
    monitor = get_real_performance_monitor()
    return monitor.get_system_health_report()

if __name__ == "__main__":
    # Demo der ECHTEN Performance Monitoring
    print("🚀 NeuronMap REAL Performance Monitor Demo")
    print("=" * 50)
    
    # Initialize real monitor
    monitor = RealPerformanceMonitor(
        monitoring_interval=2.0,  # Fast for demo
        enable_prometheus=True
    )
    
    # Start monitoring
    monitor.start_monitoring()
    print("📊 Real-time monitoring started...")
    
    # Show current system stats
    print("\n📈 Current System Stats:")
    current_stats = monitor.get_current_stats()
    
    print(f"  CPU: {current_stats.cpu_percent:.1f}%")
    print(f"  Memory: {current_stats.memory_percent:.1f}% "
          f"({current_stats.memory_used_gb:.1f}/{current_stats.memory_used_gb + current_stats.memory_available_gb:.1f}GB)")
    print(f"  Disk: {current_stats.disk_usage_percent:.1f}%")
    print(f"  Load Average: {current_stats.load_average_1min:.2f}")
    print(f"  Processes: {current_stats.processes_count}")
    print(f"  Threads: {current_stats.threads_count}")
    
    if current_stats.gpu_count > 0:
        print(f"  GPUs: {current_stats.gpu_count} "
              f"({current_stats.gpu_memory_used_percent:.1f}% memory used)")
        if current_stats.temperature_celsius:
            print(f"  GPU Temperature: {current_stats.temperature_celsius:.1f}°C")
    
    # Test task monitoring
    print("\n🎯 Testing Task Monitoring:")
    
    # Start monitoring a demo task
    task_stats = monitor.start_task_monitoring("demo_task_001", "Performance Demo Task")
    
    # Simulate task progress
    for progress in [25, 50, 75, 100]:
        time.sleep(1)
        eta = (100 - progress) / 25 * 1.0  # Estimate remaining time
        monitor.update_task_progress("demo_task_001", progress, eta if progress < 100 else None)
        print(f"  Progress: {progress}% (ETA: {eta:.1f}s)")
    
    # Complete task
    monitor.complete_task_monitoring("demo_task_001", success=True)
    
    # Test error case
    error_task = monitor.start_task_monitoring("error_task_002", "Error Demo Task")
    time.sleep(0.5)
    monitor.complete_task_monitoring("error_task_002", success=False, 
                                    error_message="Simulated error for demo")
    
    # Show performance reports
    print("\n📊 Performance Reports:")
    
    # Task performance
    task_report = monitor.get_task_performance_report()
    print(f"  Active Tasks: {task_report['active_tasks']}")
    print(f"  Completed: {task_report['completed_tasks']}")
    print(f"  Failed: {task_report['failed_tasks']}")
    print(f"  Success Rate: {task_report['success_rate']:.1%}")
    print(f"  Avg Duration: {task_report['avg_duration_seconds']:.2f}s")
    print(f"  Avg Memory: {task_report['avg_peak_memory_mb']:.1f}MB")
    
    # System health
    health_report = monitor.get_system_health_report()
    print(f"\n🏥 System Health Report:")
    print(f"  Health Score: {health_report['health_score']}/100")
    print(f"  Status: {health_report['health_status'].upper()}")
    print(f"  Prometheus: {'✅' if health_report['prometheus_enabled'] else '❌'}")
    
    # Show active alerts
    if health_report['active_alerts']:
        print(f"  Active Alerts:")
        for alert in health_report['active_alerts']:
            print(f"    {alert}")
    else:
        print(f"  No active alerts ✅")
    
    # Show Prometheus metrics sample
    if monitor.prometheus.enabled:
        print(f"\n📈 Prometheus Metrics Sample:")
        metrics = monitor.get_prometheus_metrics()
        # Show first few lines
        for line in metrics.split('\n')[:10]:
            if line.strip() and not line.startswith('#'):
                print(f"  {line}")
        print("  ... (truncated)")
    
    # Stop monitoring
    print(f"\n⏹️ Stopping monitoring...")
    monitor.stop_monitoring()
    
    print(f"\n✅ Real Performance Monitor Demo completed!")
