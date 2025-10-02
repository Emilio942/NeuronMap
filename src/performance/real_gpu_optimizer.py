#!/usr/bin/env python3
"""
🚀 NeuronMap REAL GPU Optimizer v2.0
Echte GPU Memory Management mit PyTorch & CUDA Integration

ERSETZT: MockGPUOptimizer
NEUE FEATURES:
- Echte CUDA Memory Allocation
- PyTorch Tensor Management
- Reale GPU Monitoring mit nvidia-ml-py
- Production-Ready Error Handling
"""

import torch
import psutil
import time
import threading
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import subprocess
import sys

# Setup Advanced Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RealGPUStats:
    """ECHTE GPU Statistics mit Hardware-Integration"""
    device_id: int
    name: str
    total_memory_gb: float
    allocated_memory_gb: float
    cached_memory_gb: float
    free_memory_gb: float
    utilization_percent: float
    temperature_celsius: Optional[float] = None
    power_usage_watts: Optional[float] = None
    compute_capability: Tuple[int, int] = (0, 0)

class RealGPUOptimizer:
    """
    🎯 ECHTER GPU Optimizer mit PyTorch Integration
    
    Features:
    - Echte CUDA Memory Management
    - Hardware Temperature Monitoring
    - PyTorch Tensor Optimization
    - Production-Ready Error Recovery
    """
    
    def __init__(self, enable_monitoring: bool = True):
        self.enable_monitoring = enable_monitoring
        self.device_count = 0
        self.cuda_available = False
        self.nvidia_ml_available = False
        
        # Initialize CUDA
        self._initialize_cuda()
        
        # Initialize NVIDIA-ML for advanced monitoring
        self._initialize_nvidia_ml()
        
        # Performance tracking
        self.optimization_stats = {
            "allocations": 0,
            "deallocations": 0,
            "cache_clears": 0,
            "memory_errors": 0,
            "temperature_warnings": 0
        }
        
        # Start monitoring if enabled
        if self.enable_monitoring and self.cuda_available:
            self._start_monitoring_thread()
        
        logger.info(f"🚀 Real GPU Optimizer initialized: {self.device_count} CUDA devices")
    
    def _initialize_cuda(self):
        """Initialize CUDA and detect available GPUs"""
        try:
            if torch.cuda.is_available():
                self.cuda_available = True
                self.device_count = torch.cuda.device_count()
                
                # Test CUDA functionality
                test_tensor = torch.randn(100, 100).cuda()
                _ = test_tensor @ test_tensor.T
                del test_tensor
                torch.cuda.empty_cache()
                
                logger.info(f"✅ CUDA initialized: {self.device_count} GPUs available")
                
                # Log GPU details
                for i in range(self.device_count):
                    props = torch.cuda.get_device_properties(i)
                    logger.info(f"  GPU {i}: {props.name} ({props.total_memory/1e9:.1f}GB)")
                    
            else:
                logger.warning("❌ CUDA not available - GPU optimization disabled")
                
        except Exception as e:
            logger.error(f"CUDA initialization failed: {e}")
            self.cuda_available = False
    
    def _initialize_nvidia_ml(self):
        """Initialize NVIDIA Management Library for advanced monitoring"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvidia_ml_available = True
            self.pynvml = pynvml
            logger.info("✅ NVIDIA-ML initialized for advanced GPU monitoring")
            
        except ImportError:
            logger.warning("⚠️ pynvml not available - install with: pip install nvidia-ml-py")
            self.nvidia_ml_available = False
        except Exception as e:
            logger.warning(f"NVIDIA-ML initialization failed: {e}")
            self.nvidia_ml_available = False
    
    def get_real_gpu_stats(self) -> List[RealGPUStats]:
        """Get ECHTE GPU statistics mit Hardware-Integration"""
        if not self.cuda_available:
            return []
        
        stats = []
        
        for device_id in range(self.device_count):
            try:
                # PyTorch memory statistics
                with torch.cuda.device(device_id):
                    props = torch.cuda.get_device_properties(device_id)
                    total_memory = props.total_memory
                    allocated = torch.cuda.memory_allocated(device_id)
                    cached = torch.cuda.memory_reserved(device_id)
                    free = total_memory - cached
                    
                    utilization = (allocated / total_memory) * 100
                
                # Advanced monitoring mit NVIDIA-ML
                temperature = None
                power_usage = None
                
                if self.nvidia_ml_available:
                    try:
                        handle = self.pynvml.nvmlDeviceGetHandleByIndex(device_id)
                        temperature = self.pynvml.nvmlDeviceGetTemperature(
                            handle, self.pynvml.NVML_TEMPERATURE_GPU)
                        power_usage = self.pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    except Exception as e:
                        logger.debug(f"Advanced monitoring failed for GPU {device_id}: {e}")
                
                stats.append(RealGPUStats(
                    device_id=device_id,
                    name=props.name,
                    total_memory_gb=total_memory / 1e9,
                    allocated_memory_gb=allocated / 1e9,
                    cached_memory_gb=cached / 1e9,
                    free_memory_gb=free / 1e9,
                    utilization_percent=utilization,
                    temperature_celsius=temperature,
                    power_usage_watts=power_usage,
                    compute_capability=(props.major, props.minor)
                ))
                
            except Exception as e:
                logger.error(f"Failed to get stats for GPU {device_id}: {e}")
        
        return stats
    
    def optimize_model_placement(self, model_size_gb: float, 
                               memory_overhead: float = 0.2) -> int:
        """
        🎯 ECHTE GPU-Auswahl für Model Loading
        
        Args:
            model_size_gb: Model size in GB
            memory_overhead: Additional memory overhead (20% default)
            
        Returns:
            Best GPU device ID or -1 if no suitable GPU
        """
        if not self.cuda_available:
            logger.warning("CUDA not available for model placement")
            return -1
        
        gpu_stats = self.get_real_gpu_stats()
        if not gpu_stats:
            return -1
        
        required_memory = model_size_gb * (1 + memory_overhead)
        
        # Find best GPU based on multiple criteria
        best_gpu = -1
        best_score = float('-inf')
        
        for gpu in gpu_stats:
            # Check if GPU has enough memory
            if gpu.free_memory_gb < required_memory:
                continue
            
            # Calculate placement score
            score = gpu.free_memory_gb
            
            # Penalty for high utilization
            score -= gpu.utilization_percent * 0.01
            
            # Penalty for high temperature
            if gpu.temperature_celsius:
                temp_penalty = max(0, gpu.temperature_celsius - 70) * 0.1
                score -= temp_penalty
                
                if gpu.temperature_celsius > 85:
                    self.optimization_stats["temperature_warnings"] += 1
                    logger.warning(f"🔥 GPU {gpu.device_id} running hot: {gpu.temperature_celsius}°C")
            
            # Bonus for better compute capability
            compute_score = gpu.compute_capability[0] + gpu.compute_capability[1] * 0.1
            score += compute_score
            
            if score > best_score:
                best_score = score
                best_gpu = gpu.device_id
        
        if best_gpu >= 0:
            self.optimization_stats["allocations"] += 1
            logger.info(f"🎯 Selected GPU {best_gpu} for {model_size_gb:.1f}GB model "
                       f"(score: {best_score:.2f}, free: {gpu_stats[best_gpu].free_memory_gb:.1f}GB)")
        else:
            logger.error(f"❌ No suitable GPU found for {model_size_gb:.1f}GB model")
            
        return best_gpu
    
    def allocate_tensor_optimized(self, shape: Tuple[int, ...], 
                                dtype: torch.dtype = torch.float32,
                                device_id: Optional[int] = None) -> torch.Tensor:
        """
        ⚡ Optimierte Tensor-Allokation mit Memory Management
        
        Args:
            shape: Tensor shape
            dtype: Data type
            device_id: Specific GPU or auto-select
            
        Returns:
            Allocated tensor on optimal GPU
        """
        if not self.cuda_available:
            # Fallback to CPU
            logger.warning("CUDA not available, allocating on CPU")
            return torch.zeros(shape, dtype=dtype)
        
        # Calculate memory requirement
        element_size = torch.tensor([], dtype=dtype).element_size()
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        required_gb = (total_elements * element_size) / 1e9
        
        # Auto-select GPU if not specified
        if device_id is None:
            device_id = self.optimize_model_placement(required_gb)
            if device_id < 0:
                # Fallback to CPU if no GPU suitable
                logger.warning(f"No GPU available for {required_gb:.2f}GB tensor, using CPU")
                return torch.zeros(shape, dtype=dtype)
        
        try:
            with torch.cuda.device(device_id):
                # Pre-allocation memory check
                available_memory = torch.cuda.get_device_properties(device_id).total_memory
                allocated_memory = torch.cuda.memory_allocated(device_id)
                required_bytes = total_elements * element_size
                
                if (allocated_memory + required_bytes) > available_memory * 0.9:
                    # Try cleanup first
                    self.cleanup_gpu_memory(device_id)
                
                # Allocate tensor
                tensor = torch.zeros(shape, dtype=dtype, device=device_id)
                logger.debug(f"✅ Allocated {required_gb:.2f}GB tensor on GPU {device_id}")
                return tensor
                
        except RuntimeError as e:
            self.optimization_stats["memory_errors"] += 1
            logger.error(f"Memory allocation failed on GPU {device_id}: {e}")
            
            # Try cleanup and retry
            self.cleanup_gpu_memory(device_id)
            try:
                with torch.cuda.device(device_id):
                    tensor = torch.zeros(shape, dtype=dtype, device=device_id)
                    logger.info(f"✅ Allocation succeeded after cleanup on GPU {device_id}")
                    return tensor
            except RuntimeError:
                # Final fallback to CPU
                logger.warning("GPU allocation failed, falling back to CPU")
                return torch.zeros(shape, dtype=dtype)
    
    def cleanup_gpu_memory(self, device_id: Optional[int] = None):
        """
        🧹 ECHTE GPU Memory Cleanup
        
        Args:
            device_id: Specific GPU or all GPUs if None
        """
        if not self.cuda_available:
            return
        
        devices = [device_id] if device_id is not None else range(self.device_count)
        
        for dev_id in devices:
            try:
                with torch.cuda.device(dev_id):
                    # Get memory stats before cleanup
                    before_allocated = torch.cuda.memory_allocated(dev_id)
                    before_cached = torch.cuda.memory_reserved(dev_id)
                    
                    # Clear cache
                    torch.cuda.empty_cache()
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
                    
                    # Get memory stats after cleanup
                    after_allocated = torch.cuda.memory_allocated(dev_id)
                    after_cached = torch.cuda.memory_reserved(dev_id)
                    
                    freed_allocated = (before_allocated - after_allocated) / 1e9
                    freed_cached = (before_cached - after_cached) / 1e9
                    
                    if freed_allocated > 0.1 or freed_cached > 0.1:
                        logger.info(f"🧹 GPU {dev_id} cleanup: "
                                   f"freed {freed_allocated:.2f}GB allocated, "
                                   f"{freed_cached:.2f}GB cached")
                    
                    self.optimization_stats["cache_clears"] += 1
                    
            except Exception as e:
                logger.error(f"Cleanup failed for GPU {dev_id}: {e}")
    
    def _start_monitoring_thread(self):
        """Start background monitoring thread"""
        def monitor_loop():
            while True:
                try:
                    stats = self.get_real_gpu_stats()
                    
                    for gpu in stats:
                        # Check for thermal throttling
                        if gpu.temperature_celsius and gpu.temperature_celsius > 85:
                            logger.warning(f"🔥 GPU {gpu.device_id} thermal warning: "
                                         f"{gpu.temperature_celsius}°C")
                        
                        # Check for memory pressure
                        if gpu.utilization_percent > 90:
                            logger.warning(f"💾 GPU {gpu.device_id} memory pressure: "
                                         f"{gpu.utilization_percent:.1f}%")
                    
                    time.sleep(30)  # Monitor every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("📊 GPU monitoring thread started")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        gpu_stats = self.get_real_gpu_stats()
        
        return {
            "cuda_available": self.cuda_available,
            "nvidia_ml_available": self.nvidia_ml_available,
            "total_gpus": self.device_count,
            "gpu_details": [
                {
                    "device_id": gpu.device_id,
                    "name": gpu.name,
                    "memory_total_gb": gpu.total_memory_gb,
                    "memory_free_gb": gpu.free_memory_gb,
                    "utilization_percent": gpu.utilization_percent,
                    "temperature_celsius": gpu.temperature_celsius,
                    "compute_capability": f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
                }
                for gpu in gpu_stats
            ],
            "optimization_stats": self.optimization_stats.copy(),
            "recommendations": self._generate_recommendations(gpu_stats)
        }
    
    def _generate_recommendations(self, gpu_stats: List[RealGPUStats]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if not gpu_stats:
            recommendations.append("❌ No GPUs available - consider using CPU-only mode")
            return recommendations
        
        # Check for underutilized GPUs
        underutilized = [gpu for gpu in gpu_stats if gpu.utilization_percent < 20]
        if underutilized:
            recommendations.append(f"💡 {len(underutilized)} GPU(s) underutilized - consider larger batch sizes")
        
        # Check for memory pressure
        high_usage = [gpu for gpu in gpu_stats if gpu.utilization_percent > 85]
        if high_usage:
            recommendations.append(f"⚠️ {len(high_usage)} GPU(s) under memory pressure - consider cleanup")
        
        # Check for thermal issues
        hot_gpus = [gpu for gpu in gpu_stats 
                   if gpu.temperature_celsius and gpu.temperature_celsius > 80]
        if hot_gpus:
            recommendations.append(f"🔥 {len(hot_gpus)} GPU(s) running hot - check cooling")
        
        # Check for compute capability
        old_gpus = [gpu for gpu in gpu_stats if gpu.compute_capability[0] < 7]
        if old_gpus:
            recommendations.append(f"🔧 {len(old_gpus)} GPU(s) with older compute capability - consider upgrade")
        
        if not recommendations:
            recommendations.append("✅ All GPUs operating optimally")
        
        return recommendations

# Global optimizer instance
_real_gpu_optimizer = None

def get_real_gpu_optimizer() -> RealGPUOptimizer:
    """Get global real GPU optimizer instance"""
    global _real_gpu_optimizer
    if _real_gpu_optimizer is None:
        _real_gpu_optimizer = RealGPUOptimizer()
    return _real_gpu_optimizer

def optimize_for_real_model(model_size_gb: float) -> int:
    """Quick function to get optimal GPU for model loading"""
    optimizer = get_real_gpu_optimizer()
    return optimizer.optimize_model_placement(model_size_gb)

def allocate_optimized_tensor(shape: Tuple[int, ...], 
                            dtype: torch.dtype = torch.float32,
                            device_id: Optional[int] = None) -> torch.Tensor:
    """Quick function for optimized tensor allocation"""
    optimizer = get_real_gpu_optimizer()
    return optimizer.allocate_tensor_optimized(shape, dtype, device_id)

def cleanup_real_gpu_memory(device_id: Optional[int] = None):
    """Quick function for real GPU memory cleanup"""
    optimizer = get_real_gpu_optimizer()
    optimizer.cleanup_gpu_memory(device_id)

def get_real_gpu_report() -> Dict[str, Any]:
    """Quick function to get real GPU optimization report"""
    optimizer = get_real_gpu_optimizer()
    return optimizer.get_optimization_report()

if __name__ == "__main__":
    # Demo der ECHTEN GPU Optimization
    print("🚀 NeuronMap REAL GPU Optimizer Demo")
    print("=" * 50)
    
    # Initialize real optimizer
    optimizer = RealGPUOptimizer(enable_monitoring=True)
    
    # Show real GPU stats
    gpu_stats = optimizer.get_real_gpu_stats()
    
    if gpu_stats:
        print("\n📊 ECHTE GPU Statistics:")
        for gpu in gpu_stats:
            print(f"  GPU {gpu.device_id} ({gpu.name}):")
            print(f"    Memory: {gpu.allocated_memory_gb:.1f}/{gpu.total_memory_gb:.1f}GB "
                  f"({gpu.utilization_percent:.1f}% used)")
            if gpu.temperature_celsius:
                print(f"    Temperature: {gpu.temperature_celsius}°C")
            if gpu.power_usage_watts:
                print(f"    Power: {gpu.power_usage_watts:.0f}W")
            print(f"    Compute: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
        
        # Test real model placement optimization
        print("\n🎯 ECHTE Model Placement Tests:")
        test_sizes = [1.0, 3.5, 7.0, 13.0]  # GB
        
        for size in test_sizes:
            best_gpu = optimizer.optimize_model_placement(size)
            status = f"GPU {best_gpu}" if best_gpu >= 0 else "No suitable GPU"
            print(f"  {size}GB model → {status}")
        
        # Test real tensor allocation
        print("\n⚡ ECHTE Tensor Allocation Test:")
        try:
            # Allocate a real tensor
            test_tensor = optimizer.allocate_tensor_optimized((1000, 1000), torch.float32)
            print(f"  ✅ Allocated {test_tensor.numel() * 4 / 1e6:.1f}MB tensor on {test_tensor.device}")
            
            # Clean up
            del test_tensor
            optimizer.cleanup_gpu_memory()
            print("  🧹 Cleanup completed")
            
        except Exception as e:
            print(f"  ❌ Tensor allocation failed: {e}")
        
        # Show optimization report
        report = optimizer.get_optimization_report()
        print(f"\n📈 Optimization Report:")
        print(f"  Total Allocations: {report['optimization_stats']['allocations']}")
        print(f"  Memory Cleanups: {report['optimization_stats']['cache_clears']}")
        print(f"  Memory Errors: {report['optimization_stats']['memory_errors']}")
        
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  {rec}")
            
    else:
        print("❌ No CUDA GPUs available")
        print("💡 This demo requires NVIDIA GPUs with CUDA support")
    
    print("\n✅ Real GPU Optimizer Demo completed!")
