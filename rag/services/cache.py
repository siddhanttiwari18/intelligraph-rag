import time
import threading
from typing import Any, Dict, Optional, Tuple

class RAGCache:
    """Thread-safe caching mechanism with Time-To-Live (TTL) support for RAG queries,
    vector lookups, and graph nodes.
    """
    def __init__(self, default_ttl: float = 300.0):
        self.default_ttl = default_ttl
        # Internal structure: {cache_type: {key: (value, expiry_timestamp)}}
        self._cache: Dict[str, Dict[str, Tuple[Any, float]]] = {}
        self._lock = threading.Lock()

    def get(self, cache_type: str, key: str) -> Optional[Any]:
        """Retrieve a value from the cache if it hasn't expired yet."""
        with self._lock:
            if cache_type not in self._cache:
                return None
            
            entry = self._cache[cache_type].get(key)
            if entry is None:
                return None
            
            value, expiry = entry
            if time.time() > expiry:
                # Expired - clean it up
                self._cache[cache_type].pop(key, None)
                return None
                
            return value

    def set(self, cache_type: str, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a value in the cache with an optional custom TTL."""
        ttl_val = ttl if ttl is not None else self.default_ttl
        expiry = time.time() + ttl_val
        
        with self._lock:
            if cache_type not in self._cache:
                self._cache[cache_type] = {}
            self._cache[cache_type][key] = (value, expiry)

    def invalidate(self, cache_type: Optional[str] = None) -> None:
        """Invalidate the cache. 
        If cache_type is specified, only that type is cleared, otherwise all types are cleared.
        """
        with self._lock:
            if cache_type is not None:
                if cache_type in self._cache:
                    self._cache[cache_type].clear()
            else:
                self._cache.clear()

    def clear(self) -> None:
        """Explicitly clear all caches."""
        self.invalidate()

# Singleton instance for platform-wide usage
platform_cache = RAGCache()
