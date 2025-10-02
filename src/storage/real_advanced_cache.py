#!/usr/bin/env python3
"""
🔄 NeuronMap REAL Redis Cache System v2.0
Echtes Multi-Level Caching mit Redis & Advanced Compression

ERSETZT: MockAdvancedCache
NEUE FEATURES:
- Echte Redis Integration
- Production-Ready Compression (zlib, lz4, brotli)
- Intelligent Fallback zu Memory/Disk Cache
- Real Performance Monitoring
- Robust Error Handling & Recovery
"""

import redis
import pickle
import zlib
import time
import json
import hashlib
import threading
import logging
import psutil
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta
import sys
import os

# Advanced compression imports mit fallbacks
try:
    import lz4.frame as lz4
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False

# Setup Advanced Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RealCacheStats:
    """ECHTE Cache Performance Statistics"""
    total_gets: int = 0
    total_sets: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    redis_hits: int = 0
    memory_hits: int = 0
    disk_hits: int = 0
    compression_ratio_avg: float = 1.0
    total_bytes_stored: int = 0
    total_bytes_compressed: int = 0
    redis_errors: int = 0
    fallback_activations: int = 0
    
    @property
    def hit_rate(self) -> float:
        if self.total_gets == 0:
            return 0.0
        return self.cache_hits / self.total_gets
    
    @property
    def compression_efficiency(self) -> float:
        if self.total_bytes_stored == 0:
            return 1.0
        return self.total_bytes_compressed / self.total_bytes_stored

class RealCompressionEngine:
    """
    🗜️ ECHTE Compression Engine mit Multiple Algorithms
    
    Unterstützt:
    - zlib (default, universal)
    - lz4 (fast compression/decompression)
    - brotli (best compression ratio)
    """
    
    def __init__(self):
        self.algorithms = {
            'zlib': self._compress_zlib,
            'lz4': self._compress_lz4 if LZ4_AVAILABLE else None,
            'brotli': self._compress_brotli if BROTLI_AVAILABLE else None,
        }
        
        self.decompression = {
            'zlib': self._decompress_zlib,
            'lz4': self._decompress_lz4 if LZ4_AVAILABLE else None,
            'brotli': self._decompress_brotli if BROTLI_AVAILABLE else None,
        }
        
        # Performance tracking
        self.compression_stats = {}
        
        logger.info(f"🗜️ Compression Engine initialized: "
                   f"zlib=✅, lz4={'✅' if LZ4_AVAILABLE else '❌'}, "
                   f"brotli={'✅' if BROTLI_AVAILABLE else '❌'}")
    
    def auto_select_algorithm(self, data_size: int, 
                            priority: str = 'balanced') -> str:
        """
        🎯 Intelligente Algorithm-Auswahl
        
        Args:
            data_size: Size of data in bytes
            priority: 'speed', 'ratio', 'balanced'
        """
        if priority == 'speed':
            if LZ4_AVAILABLE:
                return 'lz4'
            return 'zlib'
        
        elif priority == 'ratio':
            if BROTLI_AVAILABLE and data_size > 1024:  # Brotli für größere Daten
                return 'brotli'
            return 'zlib'
        
        else:  # balanced
            if data_size < 1024:  # Kleine Daten: schnell
                return 'lz4' if LZ4_AVAILABLE else 'zlib'
            elif data_size > 100000:  # Große Daten: beste Ratio
                return 'brotli' if BROTLI_AVAILABLE else 'zlib'
            else:  # Mittlere Daten: zlib
                return 'zlib'
    
    def compress(self, data: bytes, algorithm: Optional[str] = None) -> Tuple[bytes, str, float]:
        """
        Komprimiere Daten mit optimalen Algorithmus
        
        Returns:
            (compressed_data, algorithm_used, compression_ratio)
        """
        if algorithm is None:
            algorithm = self.auto_select_algorithm(len(data))
        
        if algorithm not in self.algorithms or self.algorithms[algorithm] is None:
            logger.warning(f"Algorithm {algorithm} not available, falling back to zlib")
            algorithm = 'zlib'
        
        start_time = time.time()
        compressed_data = self.algorithms[algorithm](data)
        compression_time = time.time() - start_time
        
        ratio = len(data) / len(compressed_data) if compressed_data else 1.0
        
        # Update stats
        if algorithm not in self.compression_stats:
            self.compression_stats[algorithm] = {
                'count': 0, 'total_ratio': 0.0, 'total_time': 0.0
            }
        
        stats = self.compression_stats[algorithm]
        stats['count'] += 1
        stats['total_ratio'] += ratio
        stats['total_time'] += compression_time
        
        return compressed_data, algorithm, ratio
    
    def decompress(self, compressed_data: bytes, algorithm: str) -> bytes:
        """Dekomprimiere Daten"""
        if algorithm not in self.decompression or self.decompression[algorithm] is None:
            raise ValueError(f"Decompression algorithm {algorithm} not available")
        
        return self.decompression[algorithm](compressed_data)
    
    def _compress_zlib(self, data: bytes) -> bytes:
        return zlib.compress(data, level=6)  # Balanced compression
    
    def _decompress_zlib(self, data: bytes) -> bytes:
        return zlib.decompress(data)
    
    def _compress_lz4(self, data: bytes) -> bytes:
        return lz4.compress(data)
    
    def _decompress_lz4(self, data: bytes) -> bytes:
        return lz4.decompress(data)
    
    def _compress_brotli(self, data: bytes) -> bytes:
        return brotli.compress(data, quality=6)  # Balanced quality
    
    def _decompress_brotli(self, data: bytes) -> bytes:
        return brotli.decompress(data)
    
    def get_compression_report(self) -> Dict[str, Any]:
        """Get comprehensive compression performance report"""
        report = {}
        
        for algo, stats in self.compression_stats.items():
            if stats['count'] > 0:
                report[algo] = {
                    'operations': stats['count'],
                    'avg_compression_ratio': stats['total_ratio'] / stats['count'],
                    'avg_time_ms': (stats['total_time'] / stats['count']) * 1000,
                    'total_time_s': stats['total_time']
                }
        
        return report

class RealAdvancedCache:
    """
    🚀 ECHTER Multi-Level Cache mit Redis Integration
    
    Cache Hierarchy:
    1. Memory Cache (L1) - fastest access
    2. Redis Cache (L2) - network storage  
    3. Disk Cache (L3) - persistent fallback
    
    Features:
    - Echte Redis Integration mit Connection Pooling
    - Advanced Compression mit Multiple Algorithms
    - Intelligent Fallback-Strategien
    - Real Performance Monitoring
    - Production-Ready Error Handling
    """
    
    def __init__(self, 
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 memory_limit_mb: int = 512,
                 disk_cache_dir: str = './cache',
                 enable_compression: bool = True):
        
        self.redis_config = {
            'host': redis_host,
            'port': redis_port,
            'db': redis_db
        }
        
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.disk_cache_dir = Path(disk_cache_dir)
        self.enable_compression = enable_compression
        
        # Initialize components
        self._init_redis_connection()
        self._init_memory_cache()
        self._init_disk_cache()
        self._init_compression_engine()
        
        # Performance tracking
        self.stats = RealCacheStats()
        self._stats_lock = threading.Lock()
        
        # Start background cleanup
        self._start_cleanup_thread()
        
        logger.info(f"🚀 Real Advanced Cache initialized: "
                   f"Redis={'✅' if self.redis_available else '❌'}, "
                   f"Memory={memory_limit_mb}MB, "
                   f"Disk={disk_cache_dir}")
    
    def _init_redis_connection(self):
        """Initialize Redis connection mit robust error handling"""
        try:
            # Test Redis connection
            self.redis_client = redis.Redis(
                host=self.redis_config['host'],
                port=self.redis_config['port'],
                db=self.redis_config['db'],
                decode_responses=False,  # We handle bytes directly
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.redis_client.ping()
            self.redis_available = True
            logger.info(f"✅ Redis connected: {self.redis_config['host']}:{self.redis_config['port']}")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            logger.warning("📝 Tip: Start Redis with: sudo systemctl start redis-server")
            self.redis_available = False
            self.redis_client = None
    
    def _init_memory_cache(self):
        """Initialize in-memory cache"""
        self.memory_cache = {}
        self.memory_access_times = {}
        self.memory_sizes = {}
        self.memory_used_bytes = 0
        logger.info(f"💾 Memory cache initialized: {self.memory_limit_bytes/1024/1024:.0f}MB limit")
    
    def _init_disk_cache(self):
        """Initialize persistent disk cache"""
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite index for disk cache
        self.disk_index_path = self.disk_cache_dir / 'cache_index.db'
        
        with sqlite3.connect(str(self.disk_index_path)) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    compressed_size INTEGER NOT NULL,
                    compression_algo TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
            ''')
            conn.commit()
        
        logger.info(f"💿 Disk cache initialized: {self.disk_cache_dir}")
    
    def _init_compression_engine(self):
        """Initialize compression engine"""
        if self.enable_compression:
            self.compression_engine = RealCompressionEngine()
        else:
            self.compression_engine = None
            logger.info("🗜️ Compression disabled")
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data to bytes"""
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize bytes to data"""
        return pickle.loads(data)
    
    def _generate_cache_key(self, key: str) -> str:
        """Generate consistent cache key"""
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        🔍 ECHTER Multi-Level Cache GET
        
        Cache Lookup Order:
        1. Memory Cache (fastest)
        2. Redis Cache (network)
        3. Disk Cache (persistent)
        """
        with self._stats_lock:
            self.stats.total_gets += 1
        
        cache_key = self._generate_cache_key(key)
        
        # Level 1: Memory Cache
        if cache_key in self.memory_cache:
            self.memory_access_times[cache_key] = time.time()
            with self._stats_lock:
                self.stats.cache_hits += 1
                self.stats.memory_hits += 1
            logger.debug(f"💾 Memory cache HIT: {key}")
            return self.memory_cache[cache_key]
        
        # Level 2: Redis Cache
        if self.redis_available:
            try:
                redis_data = self.redis_client.get(f"neuronmap:{cache_key}")
                if redis_data is not None:
                    # Deserialize and decompress if needed
                    data = self._process_retrieved_data(redis_data)
                    
                    # Promote to memory cache
                    self._store_in_memory(cache_key, data)
                    
                    with self._stats_lock:
                        self.stats.cache_hits += 1
                        self.stats.redis_hits += 1
                    logger.debug(f"🔄 Redis cache HIT: {key}")
                    return data
                    
            except Exception as e:
                logger.warning(f"Redis GET error: {e}")
                with self._stats_lock:
                    self.stats.redis_errors += 1
        
        # Level 3: Disk Cache
        disk_data = self._get_from_disk(cache_key)
        if disk_data is not None:
            # Promote to higher cache levels
            self._store_in_memory(cache_key, disk_data)
            if self.redis_available:
                self._store_in_redis(cache_key, disk_data)
            
            with self._stats_lock:
                self.stats.cache_hits += 1
                self.stats.disk_hits += 1
            logger.debug(f"💿 Disk cache HIT: {key}")
            return disk_data
        
        # Cache MISS
        with self._stats_lock:
            self.stats.cache_misses += 1
        logger.debug(f"❌ Cache MISS: {key}")
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        💾 ECHTER Multi-Level Cache SET
        
        Stores data in all available cache levels
        """
        with self._stats_lock:
            self.stats.total_sets += 1
        
        cache_key = self._generate_cache_key(key)
        
        try:
            # Store in memory cache
            self._store_in_memory(cache_key, value)
            
            # Store in Redis cache
            if self.redis_available:
                self._store_in_redis(cache_key, value, ttl)
            
            # Store in disk cache (for persistence)
            self._store_in_disk(cache_key, value)
            
            logger.debug(f"✅ Cache SET complete: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Cache SET failed for {key}: {e}")
            return False
    
    def _store_in_memory(self, cache_key: str, data: Any):
        """Store data in memory cache mit LRU eviction"""
        serialized = self._serialize_data(data)
        data_size = len(serialized)
        
        # Check memory limits and evict if necessary
        while (self.memory_used_bytes + data_size > self.memory_limit_bytes 
               and self.memory_cache):
            self._evict_from_memory()
        
        self.memory_cache[cache_key] = data
        self.memory_access_times[cache_key] = time.time()
        self.memory_sizes[cache_key] = data_size
        self.memory_used_bytes += data_size
    
    def _store_in_redis(self, cache_key: str, data: Any, ttl: Optional[int] = None):
        """Store data in Redis cache mit compression"""
        try:
            serialized = self._serialize_data(data)
            
            # Compress if enabled
            if self.compression_engine:
                compressed, algo, ratio = self.compression_engine.compress(serialized)
                
                # Store compression metadata
                cache_data = {
                    'data': compressed,
                    'compressed': True,
                    'algorithm': algo,
                    'original_size': len(serialized)
                }
                
                with self._stats_lock:
                    self.stats.compression_ratio_avg = (
                        (self.stats.compression_ratio_avg * self.stats.total_sets + ratio) / 
                        (self.stats.total_sets + 1)
                    )
                    self.stats.total_bytes_stored += len(serialized)
                    self.stats.total_bytes_compressed += len(compressed)
                
            else:
                cache_data = {
                    'data': serialized,
                    'compressed': False
                }
            
            # Serialize cache metadata
            final_data = pickle.dumps(cache_data)
            
            # Store in Redis
            redis_key = f"neuronmap:{cache_key}"
            if ttl:
                self.redis_client.setex(redis_key, ttl, final_data)
            else:
                self.redis_client.set(redis_key, final_data)
                
        except Exception as e:
            logger.warning(f"Redis SET error: {e}")
            with self._stats_lock:
                self.stats.redis_errors += 1
    
    def _store_in_disk(self, cache_key: str, data: Any):
        """Store data in disk cache mit compression und indexing"""
        try:
            serialized = self._serialize_data(data)
            
            # Compress if enabled
            compressed_data = serialized
            compression_algo = None
            
            if self.compression_engine:
                compressed_data, compression_algo, _ = self.compression_engine.compress(serialized)
            
            # Generate filename
            filename = f"{cache_key}.cache"
            filepath = self.disk_cache_dir / filename
            
            # Write compressed data
            with open(filepath, 'wb') as f:
                f.write(compressed_data)
            
            # Update index
            with sqlite3.connect(str(self.disk_index_path)) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO cache_entries 
                    (key, filename, size_bytes, compressed_size, compression_algo, accessed_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (cache_key, filename, len(serialized), len(compressed_data), compression_algo))
                conn.commit()
                
        except Exception as e:
            logger.warning(f"Disk cache SET error: {e}")
    
    def _get_from_disk(self, cache_key: str) -> Optional[Any]:
        """Retrieve data from disk cache"""
        try:
            # Query index
            with sqlite3.connect(str(self.disk_index_path)) as conn:
                cursor = conn.execute(
                    'SELECT filename, compression_algo FROM cache_entries WHERE key = ?',
                    (cache_key,)
                )
                result = cursor.fetchone()
                
                if not result:
                    return None
                
                filename, compression_algo = result
                
                # Update access time
                conn.execute(
                    'UPDATE cache_entries SET accessed_at = CURRENT_TIMESTAMP, access_count = access_count + 1 WHERE key = ?',
                    (cache_key,)
                )
                conn.commit()
            
            # Read file
            filepath = self.disk_cache_dir / filename
            if not filepath.exists():
                return None
            
            with open(filepath, 'rb') as f:
                compressed_data = f.read()
            
            # Decompress if needed
            if compression_algo and self.compression_engine:
                serialized = self.compression_engine.decompress(compressed_data, compression_algo)
            else:
                serialized = compressed_data
            
            # Deserialize
            return self._deserialize_data(serialized)
            
        except Exception as e:
            logger.warning(f"Disk cache GET error: {e}")
            return None
    
    def _process_retrieved_data(self, redis_data: bytes) -> Any:
        """Process data retrieved from Redis (decompress if needed)"""
        try:
            cache_data = pickle.loads(redis_data)
            
            if cache_data.get('compressed', False):
                # Decompress data
                algorithm = cache_data['algorithm']
                compressed = cache_data['data']
                
                if self.compression_engine:
                    serialized = self.compression_engine.decompress(compressed, algorithm)
                else:
                    raise ValueError("Compression engine not available for decompression")
            else:
                serialized = cache_data['data']
            
            return self._deserialize_data(serialized)
            
        except Exception as e:
            logger.warning(f"Data processing error: {e}")
            raise
    
    def _evict_from_memory(self):
        """Evict least recently used item from memory cache"""
        if not self.memory_cache:
            return
        
        # Find LRU item
        oldest_key = min(self.memory_access_times.keys(), 
                        key=lambda k: self.memory_access_times[k])
        
        # Remove from memory
        if oldest_key in self.memory_cache:
            self.memory_used_bytes -= self.memory_sizes[oldest_key]
            del self.memory_cache[oldest_key]
            del self.memory_access_times[oldest_key]
            del self.memory_sizes[oldest_key]
            logger.debug(f"🗑️ Evicted from memory: {oldest_key}")
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_loop():
            while True:
                try:
                    self._cleanup_expired_entries()
                    time.sleep(300)  # Cleanup every 5 minutes
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")
                    time.sleep(60)
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
        logger.info("🧹 Cache cleanup thread started")
    
    def _cleanup_expired_entries(self):
        """Clean up expired and old cache entries"""
        try:
            # Cleanup old disk cache entries
            cutoff_time = datetime.now() - timedelta(days=7)
            
            with sqlite3.connect(str(self.disk_index_path)) as conn:
                # Find old entries
                cursor = conn.execute(
                    'SELECT key, filename FROM cache_entries WHERE accessed_at < ?',
                    (cutoff_time,)
                )
                old_entries = cursor.fetchall()
                
                # Delete files and index entries
                for key, filename in old_entries:
                    filepath = self.disk_cache_dir / filename
                    if filepath.exists():
                        filepath.unlink()
                
                # Remove from index
                conn.execute('DELETE FROM cache_entries WHERE accessed_at < ?', (cutoff_time,))
                conn.commit()
                
                if old_entries:
                    logger.info(f"🧹 Cleaned up {len(old_entries)} old cache entries")
                    
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    def clear(self, level: Optional[str] = None):
        """
        🗑️ Clear cache data
        
        Args:
            level: 'memory', 'redis', 'disk', or None for all
        """
        if level is None or level == 'memory':
            self.memory_cache.clear()
            self.memory_access_times.clear()
            self.memory_sizes.clear()
            self.memory_used_bytes = 0
            logger.info("🧹 Memory cache cleared")
        
        if (level is None or level == 'redis') and self.redis_available:
            try:
                # Delete all neuronmap keys
                keys = self.redis_client.keys("neuronmap:*")
                if keys:
                    self.redis_client.delete(*keys)
                logger.info(f"🧹 Redis cache cleared: {len(keys)} keys")
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")
        
        if level is None or level == 'disk':
            try:
                # Clear disk cache files
                for cache_file in self.disk_cache_dir.glob("*.cache"):
                    cache_file.unlink()
                
                # Clear index
                with sqlite3.connect(str(self.disk_index_path)) as conn:
                    conn.execute('DELETE FROM cache_entries')
                    conn.commit()
                
                logger.info("🧹 Disk cache cleared")
            except Exception as e:
                logger.warning(f"Disk clear error: {e}")
    
    def get_cache_report(self) -> Dict[str, Any]:
        """Get comprehensive cache performance report"""
        # System memory info
        memory_info = psutil.virtual_memory()
        
        # Disk cache stats
        disk_stats = self._get_disk_cache_stats()
        
        # Redis stats
        redis_stats = {}
        if self.redis_available:
            try:
                redis_info = self.redis_client.info()
                redis_stats = {
                    'keys': len(self.redis_client.keys("neuronmap:*")),
                    'memory_used_mb': redis_info.get('used_memory', 0) / 1024 / 1024,
                    'connected_clients': redis_info.get('connected_clients', 0)
                }
            except Exception as e:
                redis_stats = {'error': str(e)}
        
        # Compression stats
        compression_stats = {}
        if self.compression_engine:
            compression_stats = self.compression_engine.get_compression_report()
        
        return {
            'cache_stats': asdict(self.stats),
            'memory_cache': {
                'entries': len(self.memory_cache),
                'used_mb': self.memory_used_bytes / 1024 / 1024,
                'limit_mb': self.memory_limit_bytes / 1024 / 1024,
                'utilization_percent': (self.memory_used_bytes / self.memory_limit_bytes) * 100
            },
            'redis_cache': {
                'available': self.redis_available,
                **redis_stats
            },
            'disk_cache': disk_stats,
            'system_memory': {
                'total_gb': memory_info.total / 1024 / 1024 / 1024,
                'available_gb': memory_info.available / 1024 / 1024 / 1024,
                'used_percent': memory_info.percent
            },
            'compression': compression_stats,
            'recommendations': self._generate_cache_recommendations()
        }
    
    def _get_disk_cache_stats(self) -> Dict[str, Any]:
        """Get disk cache statistics"""
        try:
            with sqlite3.connect(str(self.disk_index_path)) as conn:
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as entries,
                        SUM(size_bytes) as total_size,
                        SUM(compressed_size) as compressed_size,
                        AVG(access_count) as avg_access_count
                    FROM cache_entries
                ''')
                stats = cursor.fetchone()
                
                return {
                    'entries': stats[0] or 0,
                    'total_size_mb': (stats[1] or 0) / 1024 / 1024,
                    'compressed_size_mb': (stats[2] or 0) / 1024 / 1024,
                    'compression_ratio': (stats[1] / stats[2]) if stats[2] else 1.0,
                    'avg_access_count': stats[3] or 0
                }
        except Exception as e:
            logger.warning(f"Disk stats error: {e}")
            return {'error': str(e)}
    
    def _generate_cache_recommendations(self) -> List[str]:
        """Generate cache optimization recommendations"""
        recommendations = []
        
        # Hit rate analysis
        if self.stats.hit_rate < 0.5:
            recommendations.append("⚠️ Low cache hit rate - consider increasing cache sizes or TTL")
        elif self.stats.hit_rate > 0.9:
            recommendations.append("✅ Excellent cache hit rate")
        
        # Memory utilization
        memory_util = (self.memory_used_bytes / self.memory_limit_bytes) * 100
        if memory_util > 90:
            recommendations.append("🔴 Memory cache near capacity - consider increasing limit")
        elif memory_util < 20:
            recommendations.append("💡 Memory cache underutilized - could reduce limit")
        
        # Redis availability
        if not self.redis_available:
            recommendations.append("⚠️ Redis unavailable - performance may be degraded")
        
        # Error rate
        if self.stats.redis_errors > self.stats.total_gets * 0.1:
            recommendations.append("🔴 High Redis error rate - check connection")
        
        # Compression efficiency
        if self.compression_engine and self.stats.compression_efficiency < 0.7:
            recommendations.append("💡 Poor compression efficiency - review data types")
        
        if not recommendations:
            recommendations.append("✅ Cache system performing optimally")
        
        return recommendations

# Global cache instance
_real_advanced_cache = None

def get_real_cache() -> RealAdvancedCache:
    """Get global real advanced cache instance"""
    global _real_advanced_cache
    if _real_advanced_cache is None:
        _real_advanced_cache = RealAdvancedCache()
    return _real_advanced_cache

def cache_get(key: str, default: Any = None) -> Any:
    """Quick function for cache GET operations"""
    cache = get_real_cache()
    return cache.get(key, default)

def cache_set(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """Quick function for cache SET operations"""
    cache = get_real_cache()
    return cache.set(key, value, ttl)

def clear_all_caches():
    """Quick function to clear all caches"""
    cache = get_real_cache()
    cache.clear()

def get_cache_performance_report() -> Dict[str, Any]:
    """Quick function to get cache performance report"""
    cache = get_real_cache()
    return cache.get_cache_report()

if __name__ == "__main__":
    # Demo der ECHTEN Advanced Cache
    print("🚀 NeuronMap REAL Advanced Cache Demo")
    print("=" * 50)
    
    # Initialize real cache
    cache = RealAdvancedCache(
        memory_limit_mb=64,  # Small for demo
        enable_compression=True
    )
    
    # Test data
    test_data = {
        "small_data": {"numbers": list(range(100))},
        "medium_data": {"matrix": [[i*j for j in range(50)] for i in range(50)]},
        "text_data": {"content": "This is test content. " * 1000},
        "large_list": list(range(10000))
    }
    
    print("\n💾 Testing REAL Cache Operations:")
    
    # Test SET operations
    for key, data in test_data.items():
        success = cache.set(key, data)
        status = "✅" if success else "❌"
        print(f"  {status} SET {key}: {len(str(data))} characters")
    
    # Test GET operations
    print("\n🔍 Testing Cache Retrieval:")
    for key in test_data.keys():
        start_time = time.time()
        result = cache.get(key)
        retrieval_time = (time.time() - start_time) * 1000
        
        status = "✅" if result is not None else "❌"
        print(f"  {status} GET {key}: {retrieval_time:.2f}ms")
    
    # Test cache levels
    print("\n🏗️ Testing Cache Level Performance:")
    
    # Force memory eviction by storing large data
    large_data = {"huge_list": list(range(100000))}
    cache.set("large_item", large_data)
    
    # Test retrieval from different levels
    for key in ["small_data", "medium_data", "large_item"]:
        start_time = time.time()
        result = cache.get(key)
        retrieval_time = (time.time() - start_time) * 1000
        print(f"  {key}: {retrieval_time:.2f}ms")
    
    # Show comprehensive report
    print("\n📊 REAL Cache Performance Report:")
    report = cache.get_cache_report()
    
    # Cache statistics
    stats = report['cache_stats']
    print(f"  Total Operations: {stats['total_gets']} GETs, {stats['total_sets']} SETs")
    print(f"  Hit Rate: {stats['hit_rate']:.1%}")
    print(f"  Cache Distribution: Memory={stats['memory_hits']}, "
          f"Redis={stats['redis_hits']}, Disk={stats['disk_hits']}")
    
    # Memory cache
    memory = report['memory_cache']
    print(f"  Memory Cache: {memory['entries']} entries, "
          f"{memory['used_mb']:.1f}/{memory['limit_mb']:.1f}MB "
          f"({memory['utilization_percent']:.1f}% used)")
    
    # Redis cache
    redis_info = report['redis_cache']
    if redis_info['available']:
        print(f"  Redis Cache: ✅ Available, {redis_info.get('keys', 0)} keys")
    else:
        print(f"  Redis Cache: ❌ Unavailable")
    
    # Disk cache
    disk = report['disk_cache']
    if 'error' not in disk:
        print(f"  Disk Cache: {disk['entries']} entries, "
              f"{disk['total_size_mb']:.1f}MB "
              f"(compression: {disk['compression_ratio']:.1f}x)")
    
    # Compression performance
    compression = report.get('compression', {})
    if compression:
        print(f"  Compression Performance:")
        for algo, perf in compression.items():
            print(f"    {algo}: {perf['operations']} ops, "
                  f"{perf['avg_compression_ratio']:.1f}x ratio, "
                  f"{perf['avg_time_ms']:.2f}ms avg")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"  {rec}")
    
    # Test cache clearing
    print(f"\n🧹 Testing Cache Clearing:")
    cache.clear('memory')
    print(f"  Memory cache cleared")
    
    print(f"\n✅ Real Advanced Cache Demo completed!")
