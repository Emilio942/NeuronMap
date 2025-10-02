#!/usr/bin/env python3
"""
🎯 NeuronMap Real-Time Monitoring System v2.0
Advanced Performance Dashboard & Health Monitoring

Features:
- Real-time ETA prediction for all operations
- Live performance dashboards
- Health checks for external services
- Automated bottleneck detection
"""

import time
import threading
import psutil
import requests
import subprocess
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import numpy as np
from collections import defaultdict, deque
import sqlite3

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemHealth:
    """System health status"""
    component: str
    status: str  # "healthy", "warning", "critical", "unknown"
    response_time_ms: Optional[float]
    last_check: datetime
    details: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    gpu_stats: List[Dict[str, Any]]
    active_tasks: int
    system_load: float

@dataclass
class TaskProgress:
    """Progress tracking for long-running tasks"""
    task_id: str
    task_name: str
    start_time: datetime
    estimated_duration_seconds: Optional[float]
    progress_percent: float
    current_step: str
    steps_completed: int
    total_steps: int
    throughput_per_second: Optional[float]
    eta: Optional[datetime]

class PerformanceMonitor:
    """
    🎯 Real-Time Performance Monitoring System
    
    Tracks system performance, predicts ETAs, and provides
    real-time dashboards for the NeuronMap platform.
    """
    
    def __init__(self, db_path: str = "performance_metrics.db"):
        self.db_path = db_path
        self.performance_history = deque(maxlen=3600)  # Keep 1 hour of data
        self.active_tasks = {}
        self.health_status = {}
        self.monitoring_active = False
        self.alert_thresholds = {
            "cpu_percent": 90,
            "memory_percent": 85,
            "disk_percent": 95,
            "response_time_ms": 5000,
            "gpu_temperature": 85
        }
        
        # Initialize database
        self._init_database()
        
        # Start monitoring
        self.start_monitoring()
        
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    cpu_percent REAL,
                    memory_percent REAL,
                    disk_usage_percent REAL,
                    gpu_stats TEXT,
                    system_load REAL,
                    active_tasks INTEGER
                )
            """)
            
            # Task progress table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS task_progress (
                    task_id TEXT PRIMARY KEY,
                    task_name TEXT,
                    start_time DATETIME,
                    end_time DATETIME,
                    duration_seconds REAL,
                    progress_percent REAL,
                    status TEXT
                )
            """)
            
            # Health checks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS health_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT,
                    timestamp DATETIME,
                    status TEXT,
                    response_time_ms REAL,
                    details TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("✅ Performance monitoring database initialized")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def start_monitoring(self):
        """Start real-time monitoring threads"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        
        # Performance monitoring thread
        perf_thread = threading.Thread(target=self._performance_monitor_loop, daemon=True)
        perf_thread.start()
        
        # Health monitoring thread
        health_thread = threading.Thread(target=self._health_monitor_loop, daemon=True)
        health_thread.start()
        
        logger.info("🎯 Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring threads"""
        self.monitoring_active = False
        logger.info("⏸️ Monitoring stopped")
    
    def _performance_monitor_loop(self):
        """Main performance monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                network = psutil.net_io_counters()
                load = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
                
                # GPU stats
                gpu_stats = self._get_gpu_stats()
                
                # Create snapshot
                snapshot = PerformanceSnapshot(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    disk_usage_percent=disk.percent,
                    network_io={"bytes_sent": network.bytes_sent, "bytes_recv": network.bytes_recv},
                    gpu_stats=gpu_stats,
                    active_tasks=len(self.active_tasks),
                    system_load=load
                )
                
                # Store in memory
                self.performance_history.append(snapshot)
                
                # Store in database (every 10th measurement to reduce I/O)
                if len(self.performance_history) % 10 == 0:
                    self._store_performance_metric(snapshot)
                
                # Check for alerts
                self._check_performance_alerts(snapshot)
                
                time.sleep(1)  # 1-second interval
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(5)
    
    def _health_monitor_loop(self):
        """Health monitoring loop for external services"""
        while self.monitoring_active:
            try:
                # Check various services
                services_to_check = [
                    ("ollama", "http://localhost:11434/api/version"),
                    ("huggingface_hub", "https://huggingface.co/api/models"),
                    ("local_server", "http://localhost:8000/health"),
                ]
                
                for service_name, url in services_to_check:
                    self._check_service_health(service_name, url)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(60)
    
    def _get_gpu_stats(self) -> List[Dict[str, Any]]:
        """Get GPU statistics"""
        gpu_stats = []
        
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        props = torch.cuda.get_device_properties(i)
                        allocated = torch.cuda.memory_allocated(i)
                        reserved = torch.cuda.memory_reserved(i)
                        total = props.total_memory
                        
                        # Try to get temperature
                        temp = None
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        except:
                            pass
                        
                        gpu_stats.append({
                            "device_id": i,
                            "name": props.name,
                            "memory_allocated_gb": allocated / 1e9,
                            "memory_reserved_gb": reserved / 1e9,
                            "memory_total_gb": total / 1e9,
                            "memory_percent": (reserved / total) * 100,
                            "temperature": temp
                        })
        except ImportError:
            pass  # PyTorch not available
        except Exception as e:
            logger.warning(f"GPU stats collection failed: {e}")
        
        return gpu_stats
    
    def _check_service_health(self, service_name: str, url: str):
        """Check health of external service"""
        start_time = time.time()
        health = SystemHealth(
            component=service_name,
            status="unknown",
            response_time_ms=None,
            last_check=datetime.now(),
            details={}
        )
        
        try:
            response = requests.get(url, timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            health.response_time_ms = response_time
            health.details = {"status_code": response.status_code}
            
            if response.status_code == 200:
                if response_time < self.alert_thresholds["response_time_ms"]:
                    health.status = "healthy"
                else:
                    health.status = "warning"
                    health.details["warning"] = f"Slow response: {response_time:.0f}ms"
            else:
                health.status = "warning"
                health.details["warning"] = f"HTTP {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            health.status = "critical"
            health.error_message = "Connection refused"
        except requests.exceptions.Timeout:
            health.status = "warning"
            health.error_message = "Request timeout"
        except Exception as e:
            health.status = "critical"
            health.error_message = str(e)
        
        self.health_status[service_name] = health
        self._store_health_check(health)
    
    def _check_performance_alerts(self, snapshot: PerformanceSnapshot):
        """Check for performance alerts"""
        alerts = []
        
        if snapshot.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append(f"⚠️ High CPU usage: {snapshot.cpu_percent:.1f}%")
        
        if snapshot.memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append(f"⚠️ High memory usage: {snapshot.memory_percent:.1f}%")
        
        if snapshot.disk_usage_percent > self.alert_thresholds["disk_percent"]:
            alerts.append(f"⚠️ High disk usage: {snapshot.disk_usage_percent:.1f}%")
        
        for gpu in snapshot.gpu_stats:
            if gpu.get("temperature", 0) > self.alert_thresholds["gpu_temperature"]:
                alerts.append(f"🔥 GPU {gpu['device_id']} overheating: {gpu['temperature']}°C")
        
        if alerts:
            for alert in alerts:
                logger.warning(alert)
    
    def _store_performance_metric(self, snapshot: PerformanceSnapshot):
        """Store performance metric in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO performance_metrics 
                (timestamp, cpu_percent, memory_percent, disk_usage_percent, 
                 gpu_stats, system_load, active_tasks)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.timestamp,
                snapshot.cpu_percent,
                snapshot.memory_percent,
                snapshot.disk_usage_percent,
                json.dumps(snapshot.gpu_stats),
                snapshot.system_load,
                snapshot.active_tasks
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store performance metric: {e}")
    
    def _store_health_check(self, health: SystemHealth):
        """Store health check result in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO health_checks 
                (component, timestamp, status, response_time_ms, details)
                VALUES (?, ?, ?, ?, ?)
            """, (
                health.component,
                health.last_check,
                health.status,
                health.response_time_ms,
                json.dumps(health.details)
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store health check: {e}")
    
    def start_task(self, task_id: str, task_name: str, estimated_duration: Optional[float] = None) -> TaskProgress:
        """Start tracking a new task"""
        task = TaskProgress(
            task_id=task_id,
            task_name=task_name,
            start_time=datetime.now(),
            estimated_duration_seconds=estimated_duration,
            progress_percent=0.0,
            current_step="Starting...",
            steps_completed=0,
            total_steps=1,
            throughput_per_second=None,
            eta=None
        )
        
        self.active_tasks[task_id] = task
        logger.info(f"📋 Task started: {task_name} ({task_id})")
        return task
    
    def update_task_progress(self, task_id: str, progress_percent: float, 
                           current_step: str = None, steps_completed: int = None,
                           total_steps: int = None):
        """Update progress for an active task"""
        if task_id not in self.active_tasks:
            logger.warning(f"Task {task_id} not found")
            return
        
        task = self.active_tasks[task_id]
        task.progress_percent = min(100.0, max(0.0, progress_percent))
        
        if current_step:
            task.current_step = current_step
        if steps_completed is not None:
            task.steps_completed = steps_completed
        if total_steps is not None:
            task.total_steps = total_steps
        
        # Calculate ETA
        elapsed = (datetime.now() - task.start_time).total_seconds()
        if task.progress_percent > 0:
            estimated_total = elapsed / (task.progress_percent / 100)
            remaining = estimated_total - elapsed
            task.eta = datetime.now() + timedelta(seconds=remaining)
            task.throughput_per_second = task.steps_completed / elapsed if elapsed > 0 else None
    
    def complete_task(self, task_id: str, success: bool = True):
        """Mark task as completed"""
        if task_id not in self.active_tasks:
            return
        
        task = self.active_tasks[task_id]
        task.progress_percent = 100.0
        
        # Store in database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            duration = (datetime.now() - task.start_time).total_seconds()
            
            cursor.execute("""
                INSERT INTO task_progress 
                (task_id, task_name, start_time, end_time, duration_seconds, 
                 progress_percent, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                task.task_id,
                task.task_name,
                task.start_time,
                datetime.now(),
                duration,
                task.progress_percent,
                "completed" if success else "failed"
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store task completion: {e}")
        
        # Remove from active tasks
        del self.active_tasks[task_id]
        logger.info(f"✅ Task completed: {task.task_name} ({task_id})")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data"""
        current_snapshot = self.performance_history[-1] if self.performance_history else None
        
        # Calculate trends
        recent_snapshots = list(self.performance_history)[-60:]  # Last minute
        
        cpu_trend = np.mean([s.cpu_percent for s in recent_snapshots]) if recent_snapshots else 0
        memory_trend = np.mean([s.memory_percent for s in recent_snapshots]) if recent_snapshots else 0
        
        return {
            "current_performance": asdict(current_snapshot) if current_snapshot else None,
            "trends": {
                "cpu_avg_1min": round(cpu_trend, 1),
                "memory_avg_1min": round(memory_trend, 1),
                "samples_count": len(recent_snapshots)
            },
            "active_tasks": [asdict(task) for task in self.active_tasks.values()],
            "health_status": {name: asdict(health) for name, health in self.health_status.items()},
            "alerts_active": self._get_active_alerts(),
            "monitoring_since": min(s.timestamp for s in self.performance_history) if self.performance_history else None
        }
    
    def _get_active_alerts(self) -> List[str]:
        """Get currently active alerts"""
        alerts = []
        
        if not self.performance_history:
            return alerts
        
        latest = self.performance_history[-1]
        
        if latest.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append(f"High CPU: {latest.cpu_percent:.1f}%")
        
        if latest.memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append(f"High Memory: {latest.memory_percent:.1f}%")
        
        for service, health in self.health_status.items():
            if health.status in ["warning", "critical"]:
                alerts.append(f"{service}: {health.status}")
        
        return alerts

# Global monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def track_task(task_id: str, task_name: str, estimated_duration: Optional[float] = None) -> TaskProgress:
    """Quick function to start task tracking"""
    monitor = get_performance_monitor()
    return monitor.start_task(task_id, task_name, estimated_duration)

def update_progress(task_id: str, progress: float, step: str = None):
    """Quick function to update task progress"""
    monitor = get_performance_monitor()
    monitor.update_task_progress(task_id, progress, step)

def complete_task(task_id: str, success: bool = True):
    """Quick function to complete task"""
    monitor = get_performance_monitor()
    monitor.complete_task(task_id, success)

def get_dashboard_data() -> Dict[str, Any]:
    """Quick function to get dashboard data"""
    monitor = get_performance_monitor()
    return monitor.get_dashboard_data()

if __name__ == "__main__":
    # Demo the performance monitor
    print("🎯 NeuronMap Performance Monitor Demo")
    print("=" * 50)
    
    # Initialize monitor
    monitor = PerformanceMonitor()
    
    # Simulate a task
    task = monitor.start_task("demo_task", "Demo Circuit Discovery", estimated_duration=30)
    
    # Simulate progress updates
    for i in range(11):
        progress = i * 10
        monitor.update_task_progress("demo_task", progress, f"Processing step {i+1}/10")
        print(f"📊 Progress: {progress}% - ETA: {task.eta}")
        time.sleep(2)
    
    monitor.complete_task("demo_task", success=True)
    
    # Show dashboard data
    dashboard = monitor.get_dashboard_data()
    print(f"\n📈 Dashboard Summary:")
    print(f"  CPU: {dashboard['current_performance']['cpu_percent']:.1f}%")
    print(f"  Memory: {dashboard['current_performance']['memory_percent']:.1f}%")
    print(f"  Active Tasks: {len(dashboard['active_tasks'])}")
    print(f"  Health Services: {len(dashboard['health_status'])}")
    
    if dashboard['alerts_active']:
        print(f"\n⚠️ Active Alerts:")
        for alert in dashboard['alerts_active']:
            print(f"  {alert}")
    else:
        print(f"\n✅ No active alerts")
    
    print("\n✅ Performance Monitor Demo completed!")
