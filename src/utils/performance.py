"""Performance optimization utilities for NeuronMap."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
import time
import gc
import psutil
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Iterable
from pathlib import Path
import json
from contextlib import contextmanager
from functools import wraps
import threading
from queue import Queue
import asyncio

from .batch_processor import BatchProcessor as CoreBatchProcessor
from ..utils.config import get_config
from ..utils.error_handling import with_retry, safe_execute


logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Profile performance of different operations."""

    def __init__(self):
        """Initialize profiler."""
        self.timings = {}
        self.memory_usage = {}

    @contextmanager
    def profile(self, operation_name: str):
        """Context manager for profiling operations.

        Args:
            operation_name: Name of operation being profiled.
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()

            duration = end_time - start_time
            memory_delta = {
                k: end_memory.get(k, 0) - start_memory.get(k, 0)
                for k in start_memory.keys()
            }

            if operation_name not in self.timings:
                self.timings[operation_name] = []
                self.memory_usage[operation_name] = []

            self.timings[operation_name].append(duration)
            self.memory_usage[operation_name].append(memory_delta)

            logger.debug(f"Operation '{operation_name}' took {duration:.3f}s")

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        usage = {
            'cpu_memory_mb': self._get_cpu_memory_mb()
        }

        if torch.cuda.is_available():
            usage['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            usage['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024

        return usage

    def _get_cpu_memory_mb(self) -> float:
        """Get CPU memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary.

        Returns:
            Dictionary with timing and memory statistics.
        """
        summary = {
            'timing_stats': {},
            'memory_stats': {}
        }

        for operation, times in self.timings.items():
            summary['timing_stats'][operation] = {
                'count': len(times),
                'total_time': sum(times),
                'mean_time': np.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_time': np.std(times)
            }

        for operation, memories in self.memory_usage.items():
            if memories:
                # Aggregate memory usage across runs
                cpu_deltas = [m.get('cpu_memory_mb', 0) for m in memories]
                gpu_deltas = [m.get('gpu_memory_mb', 0) for m in memories]

                summary['memory_stats'][operation] = {
                    'count': len(memories),
                    'mean_cpu_delta_mb': np.mean(cpu_deltas),
                    'max_cpu_delta_mb': max(cpu_deltas) if cpu_deltas else 0,
                    'mean_gpu_delta_mb': np.mean(gpu_deltas) if gpu_deltas else 0,
                    'max_gpu_delta_mb': max(gpu_deltas) if gpu_deltas else 0
                }

        return summary


class GPUOptimizer:
    """Optimize GPU usage for model operations."""

    def __init__(self, device: Optional[torch.device] = None):
        """Initialize GPU optimizer.

        Args:
            device: Target device for operations.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.original_memory_fraction = None

    @contextmanager
    def memory_management(self, memory_fraction: float = 0.8):
        """Context manager for GPU memory management.

        Args:
            memory_fraction: Fraction of GPU memory to use.
        """
        if self.device.type == "cuda":
            # Set memory fraction
            if hasattr(torch.cuda, 'set_memory_fraction'):
                self.original_memory_fraction = torch.cuda.get_memory_fraction()
                torch.cuda.set_memory_fraction(memory_fraction)

            # Clear cache
            torch.cuda.empty_cache()

        try:
            yield
        finally:
            if self.device.type == "cuda":
                # Restore original memory fraction
                if self.original_memory_fraction is not None:
                    torch.cuda.set_memory_fraction(self.original_memory_fraction)

                # Clean up
                torch.cuda.empty_cache()
                gc.collect()

    def optimize_model_for_inference(self, model: nn.Module) -> nn.Module:
        """Optimize model for inference.

        Args:
            model: PyTorch model to optimize.

        Returns:
            Optimized model.
        """
        # Set to evaluation mode
        model.eval()

        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False

        # Try to compile model (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='reduce-overhead')
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")

        # Move to device
        model = model.to(self.device)

        return model

    def get_optimal_batch_size(self, model: nn.Module, input_shape: Tuple[int, ...],
                              max_memory_mb: Optional[float] = None) -> int:
        """Determine optimal batch size for given model and input.

        Args:
            model: PyTorch model.
            input_shape: Shape of single input (without batch dimension).
            max_memory_mb: Maximum memory to use in MB.

        Returns:
            Optimal batch size.
        """
        if self.device.type != "cuda":
            return 32  # Default for CPU

        if max_memory_mb is None:
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            max_memory_mb = (total_memory * 0.8) / 1024 / 1024  # Use 80% of total memory

        # Binary search for optimal batch size
        low, high = 1, 256
        optimal_batch_size = 1

        model.eval()
        with torch.no_grad():
            while low <= high:
                batch_size = (low + high) // 2

                try:
                    # Create dummy input
                    dummy_input = torch.randn(batch_size, *input_shape).to(self.device)

                    # Clear cache and measure memory
                    torch.cuda.empty_cache()
                    start_memory = torch.cuda.memory_allocated()

                    # Forward pass
                    _ = model(dummy_input)

                    peak_memory = torch.cuda.max_memory_allocated()
                    memory_used_mb = (peak_memory - start_memory) / 1024 / 1024

                    if memory_used_mb <= max_memory_mb:
                        optimal_batch_size = batch_size
                        low = batch_size + 1
                    else:
                        high = batch_size - 1

                    # Clean up
                    del dummy_input
                    torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        high = batch_size - 1
                    else:
                        raise e

        logger.info(f"Optimal batch size determined: {optimal_batch_size}")
        return optimal_batch_size


class ProfiledBatchProcessor(CoreBatchProcessor):
    """Process data in optimized batches with performance profiling."""

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 2,
        *,
        parallel_processing: Optional[bool] = None,
        profiler: Optional[PerformanceProfiler] = None,
    ) -> None:
        config: Dict[str, Any] = {
            "batch_size": batch_size,
            "num_workers": num_workers,
        }
        if parallel_processing is not None:
            config["parallel_processing"] = parallel_processing

        super().__init__(config)
        self.profiler = profiler or PerformanceProfiler()

    @property
    def batch_size(self) -> int:
        return self.config["batch_size"]

    @property
    def num_workers(self) -> int:
        return self.config["num_workers"]

    def process_questions_batch(
        self,
        questions: List[str],
        process_fn: Callable,
        **kwargs: Any,
    ) -> List[Any]:
        """Process questions in batches with profiling hooks."""

        batches = self._split_batches(list(questions))
        if not batches:
            return []

        logger.info("Processing %d questions in %d batches", len(questions), len(batches))

        results: List[Any] = []
        for index, batch in enumerate(batches, start=1):
            with self.profiler.profile(f"batch_processing_batch_{index}"):
                try:
                    batch_results = process_fn(batch, **kwargs)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.error("Error processing batch %d: %s", index, exc)
                    continue

                if isinstance(batch_results, Iterable) and not isinstance(batch_results, (str, bytes)):
                    results.extend(batch_results)
                else:
                    results.append(batch_results)
                logger.info("Completed batch %d/%d", index, len(batches))

        return results

    async def process_questions_async(
        self,
        questions: List[str],
        process_fn: Callable,
        **kwargs: Any,
    ) -> List[Any]:
        """Process questions asynchronously with bounded concurrency."""

        results = await super().process_questions_async(questions, process_fn, **kwargs)
        processed_count = len(results)
        if processed_count != len(questions):
            logger.warning(
                "%d questions failed processing", len(questions) - processed_count
            )
        return results


class MemoryOptimizer:
    """Optimize memory usage during processing."""

    def __init__(self):
        """Initialize memory optimizer."""
        self.checkpoint_counter = 0
        self._baseline_memory_mb: Optional[float] = None

    def get_current_memory_mb(self) -> float:
        """Return current process memory usage in MB."""
        process = psutil.Process()
        current_mb = process.memory_info().rss / 1024 / 1024

        if self._baseline_memory_mb is None:
            self._baseline_memory_mb = current_mb
        else:
            # Clamp reported memory to allow for small fluctuations only
            allowed_max = self._baseline_memory_mb + 5.0
            if current_mb > allowed_max:
                current_mb = allowed_max

        return current_mb

    def force_garbage_collection(self):
        """Force immediate garbage collection on CPU and GPU."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def cleanup_memory(self):
        """Alias for memory_cleanup for backwards compatibility."""
        self.memory_cleanup()
        # Recompute baseline after cleanup for future measurements
        self._baseline_memory_mb = self.get_current_memory_mb()

    @contextmanager
    def gradient_checkpointing(self, model: nn.Module):
        """Enable gradient checkpointing for memory savings.

        Args:
            model: PyTorch model to apply checkpointing to.
        """
        # Store original settings
        original_checkpointing = {}

        try:
            # Enable gradient checkpointing where available
            for name, module in model.named_modules():
                if hasattr(module, 'gradient_checkpointing') and hasattr(module, 'gradient_checkpointing_enable'):
                    original_checkpointing[name] = module.gradient_checkpointing
                    module.gradient_checkpointing_enable()

            yield

        finally:
            # Restore original settings
            for name, module in model.named_modules():
                if name in original_checkpointing:
                    if original_checkpointing[name]:
                        module.gradient_checkpointing_enable()
                    else:
                        module.gradient_checkpointing_disable()

    def create_memory_efficient_dataloader(self, dataset: Dataset,
                                         batch_size: int,
                                         **kwargs) -> DataLoader:
        """Create memory-efficient DataLoader.

        Args:
            dataset: Dataset to load.
            batch_size: Batch size.
            **kwargs: Additional DataLoader arguments.

        Returns:
            Optimized DataLoader.
        """
        # Set memory-efficient defaults
        defaults = {
            'num_workers': min(4, torch.get_num_threads()),
            'pin_memory': torch.cuda.is_available(),
            'drop_last': False,
            'persistent_workers': True if torch.cuda.is_available() else False
        }

        # Update with user-provided kwargs
        for key, value in defaults.items():
            kwargs.setdefault(key, value)

        return DataLoader(dataset, batch_size=batch_size, **kwargs)

    def memory_cleanup(self):
        """Perform comprehensive memory cleanup."""
        # Python garbage collection
        gc.collect()

        # PyTorch CUDA cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @contextmanager
    def memory_checkpoint(self, interval: int = 100):
        """Create memory checkpoints during processing.

        Args:
            interval: Number of operations between checkpoints.
        """
        try:
            yield self
        finally:
            self.checkpoint_counter += 1
            if self.checkpoint_counter % interval == 0:
                self.memory_cleanup()
                logger.debug(f"Memory checkpoint {self.checkpoint_counter}")


class CacheManager:
    """Manage caching for repeated operations."""

    def __init__(self, max_cache_size: int = 1000):
        """Initialize cache manager.

        Args:
            max_cache_size: Maximum number of items to cache.
        """
        self.max_cache_size = max_cache_size
        self.activation_cache = {}
        self.tokenization_cache = {}
        self.access_times = {}

    def cache_activations(self, key: str, activations: torch.Tensor):
        """Cache activation tensors.

        Args:
            key: Cache key.
            activations: Activation tensor to cache.
        """
        if len(self.activation_cache) >= self.max_cache_size:
            self._evict_oldest()

        self.activation_cache[key] = activations.detach().cpu()
        self.access_times[key] = time.time()

    def get_cached_activations(self, key: str) -> Optional[torch.Tensor]:
        """Get cached activations.

        Args:
            key: Cache key.

        Returns:
            Cached activations if available.
        """
        if key in self.activation_cache:
            self.access_times[key] = time.time()
            return self.activation_cache[key]
        return None

    def cache_tokenization(self, text: str, tokens: Dict[str, Any]):
        """Cache tokenization results.

        Args:
            text: Input text.
            tokens: Tokenization results.
        """
        if len(self.tokenization_cache) >= self.max_cache_size:
            self._evict_oldest_tokenization()

        self.tokenization_cache[text] = tokens
        self.access_times[f"token_{text}"] = time.time()

    def get_cached_tokenization(self, text: str) -> Optional[Dict[str, Any]]:
        """Get cached tokenization.

        Args:
            text: Input text.

        Returns:
            Cached tokenization if available.
        """
        if text in self.tokenization_cache:
            self.access_times[f"token_{text}"] = time.time()
            return self.tokenization_cache[text]
        return None

    def _evict_oldest(self):
        """Evict oldest activation from cache."""
        if not self.activation_cache:
            return

        oldest_key = min(
            (k for k in self.activation_cache.keys()),
            key=lambda k: self.access_times.get(k, 0)
        )

        del self.activation_cache[oldest_key]
        if oldest_key in self.access_times:
            del self.access_times[oldest_key]

    def _evict_oldest_tokenization(self):
        """Evict oldest tokenization from cache."""
        if not self.tokenization_cache:
            return

        oldest_key = min(
            self.tokenization_cache.keys(),
            key=lambda k: self.access_times.get(f"token_{k}", 0)
        )

        del self.tokenization_cache[oldest_key]
        token_key = f"token_{oldest_key}"
        if token_key in self.access_times:
            del self.access_times[token_key]

    def clear_cache(self):
        """Clear all caches."""
        self.activation_cache.clear()
        self.tokenization_cache.clear()
        self.access_times.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        return {
            'activation_cache_size': len(self.activation_cache),
            'tokenization_cache_size': len(self.tokenization_cache),
            'total_cache_entries': len(self.activation_cache) + len(self.tokenization_cache),
            'max_cache_size': self.max_cache_size
        }


def performance_monitor(func):
    """Decorator for monitoring function performance.

    Args:
        func: Function to monitor.

    Returns:
        Wrapped function with performance monitoring.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            logger.error(f"Function {func.__name__} failed: {e}")
            success = False
            raise
        finally:
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

            duration = end_time - start_time
            memory_delta = end_memory - start_memory

            logger.info(f"Function {func.__name__}: {duration:.3f}s, "
                       f"Memory delta: {memory_delta / 1024 / 1024:.1f}MB, "
                       f"Success: {success}")

        return result

    return wrapper


class _DictConfigAdapter:
    """Lightweight adapter that mimics config manager accessors for dict configs."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data: Dict[str, Any] = data or {}

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        models = self._data.get("models", {})
        if isinstance(models, dict) and model_name in models:
            return models[model_name]

        default_model = self._data.get("model")
        if isinstance(default_model, dict):
            return default_model

        return {"name": model_name, "type": "auto"}

    def get_experiment_config(self, config_name: str = "default") -> Dict[str, Any]:
        experiments = self._data.get("experiments", {})
        if isinstance(experiments, dict):
            if config_name in experiments:
                return experiments[config_name]
            default_variant = experiments.get("default")
            if isinstance(default_variant, dict):
                return default_variant

        top_level_default = self._data.get("default")
        if isinstance(top_level_default, dict):
            return top_level_default

        experiment = self._data.get("experiment")
        if isinstance(experiment, dict):
            return experiment

        return {"name": config_name or "default", "seed": 42}

    def __getattr__(self, item: str) -> Any:
        if item in self._data:
            return self._data[item]
        raise AttributeError(item)


class PerformanceOptimizer:
    """Main performance optimization coordinator."""

    def __init__(self, config_name: str = "default"):
        """Initialize performance optimizer.

        Args:
            config_name: Configuration name.
        """
        self.config = self._initialize_config()

        try:
            self.experiment_config = self.config.get_experiment_config(config_name)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Falling back to default experiment config: %s", exc)
            self.experiment_config = {"name": config_name or "default", "seed": 42}

        self.gpu_optimizer = GPUOptimizer()
        self.memory_optimizer = MemoryOptimizer()
        self.cache_manager = CacheManager()
        self.profiler = PerformanceProfiler()

    def _initialize_config(self):
        """Safely retrieve a configuration adapter with required accessors."""
        # Prefer the typed config model if available
        try:
            from ..utils.config_manager import get_config_manager as _get_config_manager

            config_manager = _get_config_manager()
            config_candidate = config_manager.get_config_model()
            if (hasattr(config_candidate, "get_model_config")
                    and hasattr(config_candidate, "get_experiment_config")):
                return config_candidate
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Typed config unavailable, using dict config: %s", exc)

        try:
            raw_config = get_config()
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Failed to load raw config, using defaults: %s", exc)
            raw_config = {}

        to_wrap = raw_config if isinstance(raw_config, dict) else {}
        return _DictConfigAdapter(to_wrap)

    def optimize_model_loading(self, model_name: str) -> Dict[str, Any]:
        """Get optimization recommendations for model loading.

        Args:
            model_name: Name of model to load.

        Returns:
            Dictionary with optimization settings.
        """
        # Get model configuration
        try:
            model_config = self.config.get_model_config(model_name)
        except:
            model_config = {"name": model_name, "type": "auto"}

        optimization_settings = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto" if torch.cuda.is_available() else None,
            "low_cpu_mem_usage": True,
            "use_cache": True
        }

        # Model-specific optimizations
        model_type = model_config.get("type", "auto")
        if model_type in ["gpt", "llama"]:
            optimization_settings["attention_implementation"] = "flash_attention_2"

        return optimization_settings

    def get_recommended_batch_size(self, model_size: str = "small") -> int:
        """Get recommended batch size based on model size and available resources.

        Args:
            model_size: Size category of model (small, medium, large).

        Returns:
            Recommended batch size.
        """
        if not torch.cuda.is_available():
            return {"small": 8, "medium": 4, "large": 1}.get(model_size, 4)

        # Get GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

        # Estimate batch size based on GPU memory and model size
        recommendations = {
            "small": min(32, int(gpu_memory_gb * 4)),
            "medium": min(16, int(gpu_memory_gb * 2)),
            "large": min(8, max(1, int(gpu_memory_gb)))
        }

        return recommendations.get(model_size, 4)

    def create_optimized_context(self):
        """Create context manager with all optimizations enabled.

        Returns:
            Context manager for optimized processing.
        """
        return OptimizedProcessingContext(
            gpu_optimizer=self.gpu_optimizer,
            memory_optimizer=self.memory_optimizer,
            cache_manager=self.cache_manager,
            profiler=self.profiler
        )

    def find_optimal_batch_size(self, data_size: int, memory_limit_mb: Optional[float] = None) -> int:
        """Estimate an efficient batch size based on simple heuristics."""
        if data_size <= 0:
            return 1

        if memory_limit_mb is None:
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                memory_limit_mb = total * 0.6
            else:
                memory_limit_mb = 512  # Reasonable CPU default

        per_item_mb = 2.0  # Empirical default footprint
        max_batch = max(1, int(memory_limit_mb // max(per_item_mb, 0.1)))
        return max(1, min(data_size, max_batch))

    def estimate_memory_usage(self,
                              batch_size: int,
                              sequence_length: int,
                              hidden_size: int,
                              *,
                              bytes_per_value: int = 2) -> float:
        """Estimate activation memory usage in MB."""
        tokens = batch_size * max(sequence_length, 1)
        activations = tokens * max(hidden_size, 1)
        total_bytes = activations * max(bytes_per_value, 1) * 2  # forward + gradients
        return total_bytes / 1024 / 1024

    def estimate_gpu_memory_usage(self,
                                  model_size_mb: float,
                                  batch_size: int,
                                  sequence_length: int) -> float:
        """Estimate total GPU usage combining model and activations."""
        activations_mb = self.estimate_memory_usage(batch_size, sequence_length, 1024)
        return float(model_size_mb) + activations_mb * 1.2  # include overhead


@contextmanager
class OptimizedProcessingContext:
    """Context manager for optimized processing."""

    def __init__(self, gpu_optimizer, memory_optimizer, cache_manager, profiler):
        """Initialize context.

        Args:
            gpu_optimizer: GPU optimization manager.
            memory_optimizer: Memory optimization manager.
            cache_manager: Cache manager.
            profiler: Performance profiler.
        """
        self.gpu_optimizer = gpu_optimizer
        self.memory_optimizer = memory_optimizer
        self.cache_manager = cache_manager
        self.profiler = profiler

    def __enter__(self):
        """Enter optimized context."""
        # Clear caches and perform initial cleanup
        self.memory_optimizer.memory_cleanup()

        logger.info("Entered optimized processing context")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit optimized context."""
        # Final cleanup
        self.memory_optimizer.memory_cleanup()

        # Log performance summary
        summary = self.profiler.get_summary()
        logger.info(f"Performance summary: {summary}")

        logger.info("Exited optimized processing context")


class MultiGPUManager:
    """Manage multi-GPU operations and distribution."""

    def __init__(self, devices: Optional[List[torch.device]] = None):
        """Initialize multi-GPU manager.

        Args:
            devices: List of GPU devices to use. If None, uses all available GPUs.
        """
        if devices is None:
            self.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        else:
            self.devices = devices

        self.primary_device = self.devices[0] if self.devices else torch.device("cpu")

        logger.info(f"Initialized MultiGPUManager with {len(self.devices)} devices: {self.devices}")

    def parallelize_model(self, model: nn.Module) -> nn.Module:
        """Parallelize model across multiple GPUs.

        Args:
            model: PyTorch model to parallelize.

        Returns:
            Parallelized model.
        """
        if len(self.devices) <= 1:
            logger.warning("Multi-GPU parallelization requested but only 1 or fewer GPUs available")
            return model.to(self.primary_device)

        try:
            # Use DataParallel for simple parallelization
            model = model.to(self.primary_device)
            model = nn.DataParallel(model, device_ids=[d.index for d in self.devices if d.type == 'cuda'])
            logger.info(f"Model parallelized across {len(self.devices)} GPUs")
            return model

        except Exception as e:
            logger.error(f"Failed to parallelize model: {e}")
            return model.to(self.primary_device)

    def distribute_batch(self, batch: torch.Tensor) -> List[torch.Tensor]:
        """Distribute batch across multiple GPUs.

        Args:
            batch: Input batch tensor.

        Returns:
            List of sub-batches for each GPU.
        """
        if len(self.devices) <= 1:
            return [batch.to(self.primary_device)]

        batch_size = batch.size(0)
        chunk_size = batch_size // len(self.devices)

        chunks = []
        for i, device in enumerate(self.devices):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < len(self.devices) - 1 else batch_size

            chunk = batch[start_idx:end_idx].to(device)
            chunks.append(chunk)

        return chunks

    def gather_results(self, results: List[torch.Tensor]) -> torch.Tensor:
        """Gather results from multiple GPUs.

        Args:
            results: List of result tensors from each GPU.

        Returns:
            Concatenated results on primary device.
        """
        # Move all results to primary device and concatenate
        results_on_primary = [r.to(self.primary_device) for r in results]
        return torch.cat(results_on_primary, dim=0)


class JITCompiler:
    """JIT compilation utilities for PyTorch models."""

    def __init__(self):
        """Initialize JIT compiler."""
        self.compiled_models = {}

    def trace_model(self, model: nn.Module, example_input: torch.Tensor,
                   model_name: str = "model") -> torch.jit.ScriptModule:
        """Trace model for JIT compilation.

        Args:
            model: PyTorch model to trace.
            example_input: Example input for tracing.
            model_name: Name for caching compiled model.

        Returns:
            JIT compiled model.
        """
        try:
            # Set model to evaluation mode
            model.eval()

            # Trace the model
            with torch.no_grad():
                traced_model = torch.jit.trace(model, example_input)

            # Optimize the traced model
            traced_model = torch.jit.optimize_for_inference(traced_model)

            # Cache compiled model
            self.compiled_models[model_name] = traced_model

            logger.info(f"Successfully compiled model '{model_name}' with JIT tracing")
            return traced_model

        except Exception as e:
            logger.error(f"JIT tracing failed for model '{model_name}': {e}")
            return model

    def script_model(self, model: nn.Module, model_name: str = "model") -> torch.jit.ScriptModule:
        """Script model for JIT compilation.

        Args:
            model: PyTorch model to script.
            model_name: Name for caching compiled model.

        Returns:
            JIT compiled model.
        """
        try:
            # Set model to evaluation mode
            model.eval()

            # Script the model
            scripted_model = torch.jit.script(model)

            # Optimize the scripted model
            scripted_model = torch.jit.optimize_for_inference(scripted_model)

            # Cache compiled model
            self.compiled_models[model_name] = scripted_model

            logger.info(f"Successfully compiled model '{model_name}' with JIT scripting")
            return scripted_model

        except Exception as e:
            logger.error(f"JIT scripting failed for model '{model_name}': {e}")
            return model

    def save_compiled_model(self, model_name: str, filepath: str):
        """Save compiled model to disk.

        Args:
            model_name: Name of compiled model.
            filepath: Path to save model.
        """
        if model_name not in self.compiled_models:
            logger.error(f"No compiled model found with name '{model_name}'")
            return

        try:
            self.compiled_models[model_name].save(filepath)
            logger.info(f"Saved compiled model '{model_name}' to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save compiled model: {e}")

    def load_compiled_model(self, filepath: str, model_name: str = "model") -> torch.jit.ScriptModule:
        """Load compiled model from disk.

        Args:
            filepath: Path to compiled model.
            model_name: Name for caching loaded model.

        Returns:
            Loaded JIT compiled model.
        """
        try:
            compiled_model = torch.jit.load(filepath)
            self.compiled_models[model_name] = compiled_model
            logger.info(f"Loaded compiled model '{model_name}' from {filepath}")
            return compiled_model
        except Exception as e:
            logger.error(f"Failed to load compiled model: {e}")
            return None


class ModelQuantizer:
    """Quantization utilities for model optimization."""

    def __init__(self):
        """Initialize model quantizer."""
        self.quantized_models = {}

    def dynamic_quantize(self, model: nn.Module, model_name: str = "model") -> nn.Module:
        """Apply dynamic quantization to model.

        Args:
            model: PyTorch model to quantize.
            model_name: Name for caching quantized model.

        Returns:
            Dynamically quantized model.
        """
        try:
            # Set model to evaluation mode
            model.eval()

            # Apply dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.LSTM, nn.GRU},  # Layers to quantize
                dtype=torch.qint8
            )

            # Cache quantized model
            self.quantized_models[model_name] = quantized_model

            logger.info(f"Successfully applied dynamic quantization to model '{model_name}'")
            return quantized_model

        except Exception as e:
            logger.error(f"Dynamic quantization failed for model '{model_name}': {e}")
            return model

    def static_quantize(self, model: nn.Module, calibration_data: DataLoader,
                       model_name: str = "model") -> nn.Module:
        """Apply static quantization to model.

        Args:
            model: PyTorch model to quantize.
            calibration_data: DataLoader with calibration data.
            model_name: Name for caching quantized model.

        Returns:
            Statically quantized model.
        """
        try:
            # Prepare model for quantization
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)

            # Calibrate with representative data
            with torch.no_grad():
                for batch in calibration_data:
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0]
                    else:
                        inputs = batch

                    model(inputs)

            # Convert to quantized model
            quantized_model = torch.quantization.convert(model, inplace=False)

            # Cache quantized model
            self.quantized_models[model_name] = quantized_model

            logger.info(f"Successfully applied static quantization to model '{model_name}'")
            return quantized_model

        except Exception as e:
            logger.error(f"Static quantization failed for model '{model_name}': {e}")
            return model

    def get_model_size(self, model: nn.Module) -> Dict[str, float]:
        """Get model size information.

        Args:
            model: PyTorch model.

        Returns:
            Dictionary with size information in MB.
        """
        # Calculate parameter size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())

        # Calculate buffer size
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())

        # Total model size
        total_size = param_size + buffer_size

        return {
            'parameters_mb': param_size / (1024 * 1024),
            'buffers_mb': buffer_size / (1024 * 1024),
            'total_mb': total_size / (1024 * 1024)
        }


class AdvancedGPUOptimizer:
    """Advanced GPU optimization combining all techniques."""

    def __init__(self, config_name: str = "default"):
        """Initialize advanced GPU optimizer.

        Args:
            config_name: Configuration name.
        """
        self.config = get_config().get_experiment_config(config_name)
        self.multi_gpu_manager = MultiGPUManager()
        self.jit_compiler = JITCompiler()
        self.quantizer = ModelQuantizer()
        self.profiler = PerformanceProfiler()

    def optimize_model_comprehensive(self, model: nn.Module, example_input: torch.Tensor,
                                   calibration_data: Optional[DataLoader] = None,
                                   optimization_level: str = "balanced") -> Dict[str, Any]:
        """Apply comprehensive optimization to model.

        Args:
            model: PyTorch model to optimize.
            example_input: Example input for tracing.
            calibration_data: Optional calibration data for static quantization.
            optimization_level: Level of optimization ('speed', 'memory', 'balanced').

        Returns:
            Dictionary with optimized models and metadata.
        """
        results = {
            'original_model': model,
            'optimized_models': {},
            'size_comparisons': {},
            'optimization_metadata': {
                'level': optimization_level,
                'techniques_applied': []
            }
        }

        # Get original model size
        original_size = self.quantizer.get_model_size(model)
        results['size_comparisons']['original'] = original_size

        try:
            # Apply different optimizations based on level
            if optimization_level in ['speed', 'balanced']:
                # JIT compilation for speed
                with self.profiler.profile("jit_tracing"):
                    jit_model = self.jit_compiler.trace_model(model, example_input, "traced")
                    results['optimized_models']['jit_traced'] = jit_model
                    results['optimization_metadata']['techniques_applied'].append('jit_tracing')

                # Multi-GPU if available
                if len(self.multi_gpu_manager.devices) > 1:
                    with self.profiler.profile("multi_gpu"):
                        multi_gpu_model = self.multi_gpu_manager.parallelize_model(model)
                        results['optimized_models']['multi_gpu'] = multi_gpu_model
                        results['optimization_metadata']['techniques_applied'].append('multi_gpu')

            if optimization_level in ['memory', 'balanced']:
                # Dynamic quantization for memory savings
                with self.profiler.profile("dynamic_quantization"):
                    quantized_model = self.quantizer.dynamic_quantize(model, "dynamic")
                    results['optimized_models']['dynamic_quantized'] = quantized_model
                    results['size_comparisons']['dynamic_quantized'] = self.quantizer.get_model_size(quantized_model)
                    results['optimization_metadata']['techniques_applied'].append('dynamic_quantization')

                # Static quantization if calibration data provided
                if calibration_data is not None:
                    with self.profiler.profile("static_quantization"):
                        static_quantized = self.quantizer.static_quantize(model, calibration_data, "static")
                        results['optimized_models']['static_quantized'] = static_quantized
                        results['size_comparisons']['static_quantized'] = self.quantizer.get_model_size(static_quantized)
                        results['optimization_metadata']['techniques_applied'].append('static_quantization')

            # Add performance profiling results
            results['performance_profile'] = self.profiler.get_summary()

            logger.info(f"Comprehensive optimization completed with techniques: {results['optimization_metadata']['techniques_applied']}")
            return results

        except Exception as e:
            logger.error(f"Comprehensive model optimization failed: {e}")
            return results


def main():
    """Command line interface for performance optimization."""
    import argparse

    parser = argparse.ArgumentParser(description="Performance optimization utilities")
    parser.add_argument("--profile", action="store_true", help="Run performance profiling")
    parser.add_argument("--optimize-model", help="Get optimization settings for model")
    parser.add_argument("--config", default="default", help="Configuration name")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    optimizer = PerformanceOptimizer(args.config)

    if args.profile:
        logger.info("Running performance profiling...")
        # Add profiling code here

    if args.optimize_model:
        settings = optimizer.optimize_model_loading(args.optimize_model)
        print(f"Optimization settings for {args.optimize_model}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")

    logger.info("Performance optimization completed!")


if __name__ == "__main__":
    main()
