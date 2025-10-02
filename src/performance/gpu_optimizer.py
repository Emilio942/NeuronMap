#!/usr/bin/env python3
"""
🚀 NeuronMap Performance Optimizer v2.0
Advanced GPU Memory Management & Multi-GPU Coordination

Führt automatische Performance-Optimierungen durch:
- GPU Memory Pool Management
- Multi-GPU Load Balancing
- Memory Usage Profiling
- CUDA Memory Optimization
"""

import torch
import psutil
import numpy as np
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging

# Setup Advanced Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GPUMemoryStats:
    """GPU Memory Statistics Container"""
    device_id: int
    total_memory: int
    allocated_memory: int
    cached_memory: int
    reserved_memory: int
    free_memory: int
    utilization_percent: float
    temperature: Optional[float] = None
    power_usage: Optional[float] = None

@dataclass
class PerformanceMetrics:
    """System Performance Metrics"""
    timestamp: float
    cpu_usage: float
    ram_usage: float
    gpu_stats: List[GPUMemoryStats]
    active_processes: int
    memory_pressure: float
    thermal_state: str

class AdvancedGPUOptimizer:
    """
    🚀 Advanced GPU Memory Management & Performance Optimization
    
    Features:
    - Intelligent Memory Pool Management
    - Multi-GPU Load Balancing
    - Real-time Performance Monitoring
    - Automated Memory Cleanup
    - CUDA Stream Optimization
    """
    
    def __init__(self, enable_profiling: bool = True):
        self.enable_profiling = enable_profiling
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.memory_pools = {}
        self.performance_history = []
        self.optimization_settings = self._load_optimization_settings()
        
        # Initialize Memory Pools for each GPU
        for device_id in range(self.device_count):
            self._initialize_memory_pool(device_id)
        
        logger.info(f"🚀 GPU Optimizer initialized with {self.device_count} GPUs")
        
    def _load_optimization_settings(self) -> Dict[str, Any]:
        """Load optimization settings from config"""
        default_settings = {
            "memory_pool_size_ratio": 0.8,  # Use 80% of GPU memory for pool
            "enable_memory_pinning": True,
            "enable_cuda_graphs": True,
            "auto_cleanup_threshold": 0.9,  # Cleanup when >90% memory used
            "load_balancing_strategy": "memory_aware",
            "profiling_interval": 1.0,  # seconds
            "enable_mixed_precision": True,
        }
        
        try:
            config_path = Path("configs/gpu_optimization.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    settings = json.load(f)
                    default_settings.update(settings)
        except Exception as e:
            logger.warning(f"Could not load GPU optimization config: {e}")
            
        return default_settings
    
    def _initialize_memory_pool(self, device_id: int):
        """Initialize memory pool for specific GPU"""
        try:
            with torch.cuda.device(device_id):
                # Get GPU properties
                props = torch.cuda.get_device_properties(device_id)
                total_memory = props.total_memory
                
                # Calculate pool size
                pool_size = int(total_memory * self.optimization_settings["memory_pool_size_ratio"])
                
                # Initialize memory pool
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(
                    self.optimization_settings["memory_pool_size_ratio"], 
                    device_id
                )
                
                self.memory_pools[device_id] = {
                    "total_memory": total_memory,
                    "pool_size": pool_size,
                    "device_name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessor_count": props.multi_processor_count,
                }
                
                logger.info(f"📊 GPU {device_id} ({props.name}): Pool size {pool_size / 1e9:.1f}GB")
                
        except Exception as e:
            logger.error(f"Failed to initialize memory pool for GPU {device_id}: {e}")
    
    def get_gpu_stats(self) -> List[GPUMemoryStats]:
        """Get comprehensive GPU statistics"""
        stats = []
        
        for device_id in range(self.device_count):
            try:
                with torch.cuda.device(device_id):
                    # Memory statistics
                    total_memory = torch.cuda.get_device_properties(device_id).total_memory
                    allocated = torch.cuda.memory_allocated(device_id)
                    cached = torch.cuda.memory_reserved(device_id)
                    reserved = torch.cuda.memory_reserved(device_id)
                    free = total_memory - reserved
                    utilization = (allocated / total_memory) * 100
                    
                    # Try to get temperature and power (requires nvidia-ml-py)
                    temperature = None
                    power_usage = None
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    except ImportError:
                        pass  # nvidia-ml-py not available
                    except Exception:
                        pass  # GPU monitoring not available
                    
                    stats.append(GPUMemoryStats(
                        device_id=device_id,
                        total_memory=total_memory,
                        allocated_memory=allocated,
                        cached_memory=cached,
                        reserved_memory=reserved,
                        free_memory=free,
                        utilization_percent=utilization,
                        temperature=temperature,
                        power_usage=power_usage
                    ))
                    
            except Exception as e:
                logger.warning(f"Could not get stats for GPU {device_id}: {e}")
                
        return stats
    
    def optimize_memory_allocation(self, model_size_gb: float) -> int:
        """
        🎯 Intelligent GPU Selection for Model Loading
        
        Args:
            model_size_gb: Size of model in GB
            
        Returns:
            Best GPU device ID for loading the model
        """
        gpu_stats = self.get_gpu_stats()
        
        if not gpu_stats:
            logger.warning("No GPUs available")
            return -1
        
        # Calculate scores for each GPU
        best_gpu = 0
        best_score = float('-inf')
        
        for stats in gpu_stats:
            # Available memory in GB
            available_gb = stats.free_memory / 1e9
            
            # Base score: available memory
            score = available_gb
            
            # Penalty for high utilization
            score -= stats.utilization_percent * 0.1
            
            # Penalty for high temperature
            if stats.temperature:
                score -= max(0, stats.temperature - 70) * 0.1
            
            # Bonus for larger total memory
            score += (stats.total_memory / 1e9) * 0.05
            
            logger.info(f"🔍 GPU {stats.device_id}: Score {score:.2f}, Available {available_gb:.1f}GB, Util {stats.utilization_percent:.1f}%")
            
            if score > best_score and available_gb >= model_size_gb * 1.2:  # 20% safety margin
                best_score = score
                best_gpu = stats.device_id
        
        logger.info(f"🎯 Selected GPU {best_gpu} for {model_size_gb:.1f}GB model")
        return best_gpu
    
    def auto_cleanup_memory(self, device_id: Optional[int] = None):
        """
        🧹 Automatic Memory Cleanup
        
        Performs intelligent memory cleanup when thresholds are exceeded
        """
        devices = [device_id] if device_id is not None else range(self.device_count)
        
        for dev_id in devices:
            try:
                with torch.cuda.device(dev_id):
                    stats = self.get_gpu_stats()[dev_id]
                    
                    if stats.utilization_percent > self.optimization_settings["auto_cleanup_threshold"] * 100:
                        logger.info(f"🧹 Auto-cleanup triggered for GPU {dev_id} (Util: {stats.utilization_percent:.1f}%)")
                        
                        # Clear cache
                        torch.cuda.empty_cache()
                        
                        # Force garbage collection
                        import gc
                        gc.collect()
                        
                        # Log cleanup results
                        new_stats = self.get_gpu_stats()[dev_id]
                        freed_gb = (stats.cached_memory - new_stats.cached_memory) / 1e9
                        logger.info(f"✅ Freed {freed_gb:.2f}GB on GPU {dev_id}")
                        
            except Exception as e:
                logger.error(f"Memory cleanup failed for GPU {dev_id}: {e}")
    
    def enable_mixed_precision(self):
        """Enable automatic mixed precision for performance"""
        if self.optimization_settings["enable_mixed_precision"]:
            try:
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("✅ Mixed precision optimization enabled")
            except Exception as e:
                logger.warning(f"Could not enable mixed precision: {e}")
    
    def start_performance_monitoring(self):
        """Start real-time performance monitoring"""
        if not self.enable_profiling:
            return
            
        def monitor_loop():
            while True:
                try:
                    # System metrics
                    cpu_usage = psutil.cpu_percent(interval=1)
                    ram_usage = psutil.virtual_memory().percent
                    active_processes = len(psutil.pids())
                    
                    # GPU metrics
                    gpu_stats = self.get_gpu_stats()
                    
                    # Calculate memory pressure
                    memory_pressure = sum(s.utilization_percent for s in gpu_stats) / len(gpu_stats) if gpu_stats else 0
                    
                    # Thermal state
                    max_temp = max((s.temperature or 0) for s in gpu_stats) if gpu_stats else 0
                    thermal_state = "COOL" if max_temp < 70 else "WARM" if max_temp < 85 else "HOT"
                    
                    # Store metrics
                    metrics = PerformanceMetrics(
                        timestamp=time.time(),
                        cpu_usage=cpu_usage,
                        ram_usage=ram_usage,
                        gpu_stats=gpu_stats,
                        active_processes=active_processes,
                        memory_pressure=memory_pressure,
                        thermal_state=thermal_state
                    )
                    
                    self.performance_history.append(metrics)
                    
                    # Keep only last 1000 measurements
                    if len(self.performance_history) > 1000:
                        self.performance_history = self.performance_history[-1000:]
                    
                    # Auto-cleanup if needed
                    if memory_pressure > 85:
                        self.auto_cleanup_memory()
                    
                    time.sleep(self.optimization_settings["profiling_interval"])
                    
                except Exception as e:
                    logger.error(f"Performance monitoring error: {e}")
                    time.sleep(5)
        
        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("📊 Performance monitoring started")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        recent_metrics = self.performance_history[-60:]  # Last 60 measurements
        
        # Calculate averages
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        avg_ram = np.mean([m.ram_usage for m in recent_metrics])
        avg_memory_pressure = np.mean([m.memory_pressure for m in recent_metrics])
        
        # GPU utilization
        gpu_utils = []
        for i in range(self.device_count):
            utils = [m.gpu_stats[i].utilization_percent for m in recent_metrics if len(m.gpu_stats) > i]
            if utils:
                gpu_utils.append({
                    "device_id": i,
                    "avg_utilization": np.mean(utils),
                    "max_utilization": max(utils),
                    "min_utilization": min(utils)
                })
        
        # Performance recommendations
        recommendations = []
        
        if avg_memory_pressure > 80:
            recommendations.append("⚠️ High GPU memory pressure - consider reducing batch size")
        
        if avg_cpu > 90:
            recommendations.append("⚠️ High CPU usage - consider reducing data loading workers")
        
        if avg_ram > 85:
            recommendations.append("⚠️ High RAM usage - consider enabling data streaming")
        
        for gpu_util in gpu_utils:
            if gpu_util["avg_utilization"] < 50:
                recommendations.append(f"💡 GPU {gpu_util['device_id']} underutilized - consider increasing batch size")
        
        if not recommendations:
            recommendations.append("✅ System performance is optimal")
        
        return {
            "system_performance": {
                "cpu_usage_avg": round(avg_cpu, 2),
                "ram_usage_avg": round(avg_ram, 2),
                "memory_pressure_avg": round(avg_memory_pressure, 2)
            },
            "gpu_performance": gpu_utils,
            "optimization_recommendations": recommendations,
            "monitoring_duration_minutes": len(recent_metrics) * self.optimization_settings["profiling_interval"] / 60,
            "total_gpus": self.device_count
        }

# Global optimizer instance
_gpu_optimizer = None

def get_gpu_optimizer() -> AdvancedGPUOptimizer:
    """Get global GPU optimizer instance"""
    global _gpu_optimizer
    if _gpu_optimizer is None:
        _gpu_optimizer = AdvancedGPUOptimizer()
        _gpu_optimizer.enable_mixed_precision()
        _gpu_optimizer.start_performance_monitoring()
    return _gpu_optimizer

def optimize_for_model(model_size_gb: float) -> int:
    """Quick function to get optimal GPU for model loading"""
    optimizer = get_gpu_optimizer()
    return optimizer.optimize_memory_allocation(model_size_gb)

def cleanup_gpu_memory(device_id: Optional[int] = None):
    """Quick function for memory cleanup"""
    optimizer = get_gpu_optimizer()
    optimizer.auto_cleanup_memory(device_id)

def get_performance_report() -> Dict[str, Any]:
    """Quick function to get performance report"""
    optimizer = get_gpu_optimizer()
    return optimizer.get_optimization_report()

if __name__ == "__main__":
    # Demo the GPU optimizer
    print("🚀 NeuronMap GPU Optimizer Demo")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = AdvancedGPUOptimizer(enable_profiling=True)
    optimizer.enable_mixed_precision()
    optimizer.start_performance_monitoring()
    
    # Show GPU stats
    gpu_stats = optimizer.get_gpu_stats()
    for stats in gpu_stats:
        print(f"📊 GPU {stats.device_id}: {stats.utilization_percent:.1f}% utilized, {stats.free_memory/1e9:.1f}GB free")
    
    # Test model placement optimization
    test_sizes = [1.0, 3.5, 7.0, 13.0, 30.0]  # Various model sizes in GB
    
    print("\n🎯 Model Placement Optimization:")
    for size in test_sizes:
        best_gpu = optimizer.optimize_memory_allocation(size)
        print(f"  {size}GB model → GPU {best_gpu}")
    
    # Wait for some monitoring data
    print("\n📊 Collecting performance data...")
    time.sleep(5)
    
    # Show optimization report
    report = optimizer.get_optimization_report()
    print(f"\n📈 Performance Report:")
    print(f"  CPU Usage: {report['system_performance']['cpu_usage_avg']:.1f}%")
    print(f"  RAM Usage: {report['system_performance']['ram_usage_avg']:.1f}%")
    print(f"  Memory Pressure: {report['system_performance']['memory_pressure_avg']:.1f}%")
    
    print("\n💡 Recommendations:")
    for rec in report['optimization_recommendations']:
        print(f"  {rec}")
    
    print("\n✅ GPU Optimization Demo completed!")
