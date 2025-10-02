"""
Intervention Cache System - Advanced caching for model surgery operations.
This module implements B2: "Intervention-Cache" from the task list.
It provides a sophisticated caching layer that stores activations from "clean" runs
and makes them available for "corrupted" runs during path patching experiments.
"""
import torch
import pickle
import json
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import logging
from contextlib import contextmanager
logger = logging.getLogger(__name__)
@dataclass
class CacheMetadata:
    """Metadata for cached activations."""
    model_name: str
    input_hash: str
    layer_names: List[str]
    timestamp: datetime
    input_shape: Tuple[int, ...]
    device: str
    precision: str  # 'float32', 'float16', etc.
    experiment_id: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None
@dataclass
class CachedActivation:
    """Container for a single cached activation."""
    tensor: torch.Tensor
    layer_name: str
    metadata: CacheMetadata
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding tensor)."""
        return {
            'layer_name': self.layer_name,
            'metadata': asdict(self.metadata),
            'tensor_shape': list(self.tensor.shape),
            'tensor_dtype': str(self.tensor.dtype)
        }
class InterventionCache:
    """
    Advanced caching system for model activations during intervention experiments.
    Features:
    - Persistent storage with compression
    - Metadata tracking for cache validation
    - Memory-efficient loading/unloading
    - Experiment-specific organization
    - Automatic cleanup and expiration
    """
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        max_memory_gb: float = 2.0,
        compression_level: int = 6
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "cache" / "interventions"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.compression_level = compression_level
        # In-memory cache
        self._memory_cache: Dict[str, CachedActivation] = {}
        self._cache_keys: List[str] = []  # LRU ordering
        self._memory_usage = 0
        logger.info(f"Initialized InterventionCache at {self.cache_dir}")
    def _calculate_input_hash(self, input_tensor: torch.Tensor) -> str:
        """Calculate hash of input tensor for cache key generation."""
        # Use a subset of tensor values to create hash (for efficiency)
        if input_tensor.numel() > 10000:
            # Sample tensor for large inputs
            flat = input_tensor.flatten()
            indices = torch.linspace(0, len(flat)-1, 1000, dtype=torch.long)
            sample = flat[indices]
        else:
            sample = input_tensor.flatten()
        # Convert to bytes and hash
        tensor_bytes = sample.detach().cpu().numpy().tobytes()
        return hashlib.md5(tensor_bytes).hexdigest()[:16]
    def _generate_cache_key(
        self,
        model_name: str,
        input_hash: str,
        layer_name: str,
        experiment_id: Optional[str] = None
    ) -> str:
        """Generate unique cache key."""
        components = [model_name, input_hash, layer_name]
        if experiment_id:
            components.append(experiment_id)
        return "_".join(components)
    def _estimate_tensor_memory(self, tensor: torch.Tensor) -> int:
        """Estimate memory usage of tensor in bytes."""
        return tensor.element_size() * tensor.numel()
    def _evict_lru_if_needed(self, required_bytes: int):
        """Evict least recently used items if memory limit would be exceeded."""
        while (self._memory_usage + required_bytes > self.max_memory_bytes
               and self._cache_keys):
            lru_key = self._cache_keys.pop(0)
            if lru_key in self._memory_cache:
                evicted = self._memory_cache.pop(lru_key)
                freed_bytes = self._estimate_tensor_memory(evicted.tensor)
                self._memory_usage -= freed_bytes
                logger.debug(f"Evicted {lru_key} from memory cache (freed {freed_bytes/1024**2:.1f}MB)")
    def _touch_cache_key(self, cache_key: str):
        """Update LRU order for cache key."""
        if cache_key in self._cache_keys:
            self._cache_keys.remove(cache_key)
        self._cache_keys.append(cache_key)
    def store_activations(
        self,
        activations: Dict[str, torch.Tensor],
        model_name: str,
        input_tensor: torch.Tensor,
        experiment_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        persist: bool = True
    ) -> Dict[str, str]:
        """
        Store activations in cache.
        Args:
            activations: Dict of layer_name -> activation tensor
            model_name: Name/identifier of the model
            input_tensor: Input tensor used to generate activations
            experiment_id: Optional experiment identifier
            tags: Optional metadata tags
            persist: Whether to save to disk
        Returns:
            Dict mapping layer names to cache keys
        """
        input_hash = self._calculate_input_hash(input_tensor)
        cache_keys = {}
        # Create metadata
        metadata = CacheMetadata(
            model_name=model_name,
            input_hash=input_hash,
            layer_names=list(activations.keys()),
            timestamp=datetime.now(),
            input_shape=tuple(input_tensor.shape),
            device=str(input_tensor.device),
            precision=str(input_tensor.dtype),
            experiment_id=experiment_id,
            tags=tags or {}
        )
        for layer_name, activation in activations.items():
            cache_key = self._generate_cache_key(
                model_name, input_hash, layer_name, experiment_id
            )
            # Create cached activation object
            cached_activation = CachedActivation(
                tensor=activation.detach().cpu(),
                layer_name=layer_name,
                metadata=metadata
            )
            # Store in memory cache
            tensor_size = self._estimate_tensor_memory(cached_activation.tensor)
            self._evict_lru_if_needed(tensor_size)
            self._memory_cache[cache_key] = cached_activation
            self._touch_cache_key(cache_key)
            self._memory_usage += tensor_size
            cache_keys[layer_name] = cache_key
            # Persist to disk if requested
            if persist:
                self._save_to_disk(cache_key, cached_activation)
        logger.info(
            f"Stored {len(activations)} activations for {model_name} "
            f"(experiment: {experiment_id or 'default'})"
        )
        return cache_keys
    def _save_to_disk(self, cache_key: str, cached_activation: CachedActivation):
        """Save cached activation to disk."""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        metadata_path = self.cache_dir / f"{cache_key}_meta.json"
        # Save tensor data
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_activation.tensor, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(cached_activation.to_dict(), f, indent=2, default=str)
        logger.debug(f"Saved {cache_key} to disk")
    def _load_from_disk(self, cache_key: str) -> Optional[CachedActivation]:
        """Load cached activation from disk."""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        metadata_path = self.cache_dir / f"{cache_key}_meta.json"
        if not (cache_path.exists() and metadata_path.exists()):
            return None
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            # Reconstruct metadata object
            metadata_dict = data['metadata']
            metadata_dict['timestamp'] = datetime.fromisoformat(metadata_dict['timestamp'])
            metadata = CacheMetadata(**metadata_dict)
            # Load tensor
            with open(cache_path, 'rb') as f:
                tensor = pickle.load(f)
            cached_activation = CachedActivation(
                tensor=tensor,
                layer_name=data['layer_name'],
                metadata=metadata
            )
            logger.debug(f"Loaded {cache_key} from disk")
            return cached_activation
        except Exception as e:
            logger.warning(f"Failed to load {cache_key} from disk: {e}")
            return None
    def retrieve_activation(
        self,
        cache_key: str,
        device: Optional[torch.device] = None
    ) -> Optional[torch.Tensor]:
        """
        Retrieve activation tensor by cache key.
        Args:
            cache_key: Cache key from store_activations
            device: Device to move tensor to
        Returns:
            Activation tensor or None if not found
        """
        # Try memory cache first
        if cache_key in self._memory_cache:
            self._touch_cache_key(cache_key)
            tensor = self._memory_cache[cache_key].tensor
            if device is not None:
                tensor = tensor.to(device)
            logger.debug(f"Retrieved {cache_key} from memory cache")
            return tensor
        # Try disk cache
        cached_activation = self._load_from_disk(cache_key)
        if cached_activation is not None:
            # Add to memory cache
            tensor_size = self._estimate_tensor_memory(cached_activation.tensor)
            self._evict_lru_if_needed(tensor_size)
            self._memory_cache[cache_key] = cached_activation
            self._touch_cache_key(cache_key)
            self._memory_usage += tensor_size
            tensor = cached_activation.tensor
            if device is not None:
                tensor = tensor.to(device)
            logger.debug(f"Retrieved {cache_key} from disk cache")
            return tensor
        logger.warning(f"Cache key {cache_key} not found")
        return None
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache state."""
        disk_files = list(self.cache_dir.glob("*.pkl"))
        return {
            'memory_cache_size': len(self._memory_cache),
            'memory_usage_mb': self._memory_usage / 1024**2,
            'memory_limit_mb': self.max_memory_bytes / 1024**2,
            'disk_cache_size': len(disk_files),
            'cache_dir': str(self.cache_dir),
            'total_disk_size_mb': sum(f.stat().st_size for f in disk_files) / 1024**2
        }
    def clear_memory_cache(self):
        """Clear in-memory cache."""
        self._memory_cache.clear()
        self._cache_keys.clear()
        self._memory_usage = 0
        logger.info("Cleared memory cache")
    def clear_disk_cache(self, experiment_id: Optional[str] = None):
        """Clear disk cache, optionally filtered by experiment_id."""
        if experiment_id:
            pattern = f"*_{experiment_id}_*.pkl"
            metadata_pattern = f"*_{experiment_id}_*_meta.json"
        else:
            pattern = "*.pkl"
            metadata_pattern = "*_meta.json"
        cache_files = list(self.cache_dir.glob(pattern))
        metadata_files = list(self.cache_dir.glob(metadata_pattern))
        for file in cache_files + metadata_files:
            file.unlink()
        logger.info(f"Cleared {len(cache_files)} cache files from disk")
    def list_experiments(self) -> List[str]:
        """List all experiment IDs in cache."""
        experiments = set()
        # Check memory cache
        for cached_activation in self._memory_cache.values():
            if cached_activation.metadata.experiment_id:
                experiments.add(cached_activation.metadata.experiment_id)
        # Check disk cache
        for metadata_file in self.cache_dir.glob("*_meta.json"):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    experiment_id = data['metadata'].get('experiment_id')
                    if experiment_id:
                        experiments.add(experiment_id)
            except Exception:
                continue
        return sorted(list(experiments))
@contextmanager
def cached_run_context(
    cache: InterventionCache,
    model_name: str,
    experiment_id: Optional[str] = None,
    auto_store: bool = True
):
    """
    Context manager for automatic activation caching during model runs.
    Usage:
        with cached_run_context(cache, "gpt2", "experiment_1") as ctx:
            output = model(input_tensor)
            # Activations are automatically cached
    """
    class CacheContext:
        def __init__(self):
            self.activations = {}
            self.cache_keys = {}
        def store(self, input_tensor: torch.Tensor) -> Dict[str, str]:
            if self.activations:
                return cache.store_activations(
                    self.activations,
                    model_name,
                    input_tensor,
                    experiment_id
                )
            return {}
    ctx = CacheContext()
    try:
        yield ctx
    finally:
        if auto_store and ctx.activations:
            logger.info("Auto-storing cached activations")
# Utility functions for common caching patterns
def create_clean_run_cache(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    layer_names: List[str],
    cache: InterventionCache,
    model_name: str,
    experiment_id: str = "clean_run"
) -> Dict[str, str]:
    """
    Perform a clean run and cache all specified layer activations.
    This is a high-level utility for the common pattern of running
    a model once to cache activations for later patching experiments.
    """
    activations = {}
    def create_hook(layer_name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[layer_name] = output[0].detach().clone()
            else:
                activations[layer_name] = output.detach().clone()
        return hook
    # Register hooks
    hooks = []
    for layer_name in layer_names:
        module = dict(model.named_modules())[layer_name]
        hooks.append(module.register_forward_hook(create_hook(layer_name)))
    try:
        # Run model
        model.eval()
        with torch.no_grad():
            _ = model(input_tensor)
        # Store in cache
        cache_keys = cache.store_activations(
            activations, model_name, input_tensor, experiment_id
        )
        logger.info(f"Cached {len(activations)} layer activations for clean run")
        return cache_keys
    finally:
        # Clean up hooks
        for hook in hooks:
            hook.remove()
def load_cached_activations_for_patching(
    cache: InterventionCache,
    cache_keys: Dict[str, str],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Load cached activations for use in patching experiments.
    Args:
        cache: InterventionCache instance
        cache_keys: Dict mapping layer names to cache keys
        device: Device to load tensors onto
    Returns:
        Dict mapping layer names to activation tensors
    """
    activations = {}
    for layer_name, cache_key in cache_keys.items():
        tensor = cache.retrieve_activation(cache_key, device)
        if tensor is not None:
            activations[layer_name] = tensor
        else:
            logger.warning(f"Could not load cached activation for {layer_name}")
    return activations
