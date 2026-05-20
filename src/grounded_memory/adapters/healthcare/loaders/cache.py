"""
Caching layer for external KB data.

Provides disk-based caching with TTL to reduce API calls.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with TTL."""

    value: Any
    timestamp: float
    ttl_hours: int

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        elapsed_hours = (time.time() - self.timestamp) / 3600
        return elapsed_hours > self.ttl_hours


class FileCache:
    """Simple file-based cache with TTL support."""

    def __init__(self, cache_dir: str = ".cache/grounded_memory", ttl_hours: int = 168):
        """
        Initialize file cache.

        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Default TTL in hours (1 week = 168 hours)
        """
        self.cache_dir = Path(cache_dir)
        self.ttl_hours = ttl_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Initialized FileCache at {self.cache_dir}")

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        # Hash key to avoid filesystem issues with special characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    def get(self, key: str) -> Any | None:
        """
        Retrieve value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        try:
            cache_path = self._get_cache_path(key)

            if not cache_path.exists():
                return None

            with open(cache_path) as f:
                data = json.load(f)

            entry = CacheEntry(**data)

            if entry.is_expired():
                logger.debug(f"Cache expired for key: {key}")
                cache_path.unlink()  # Delete expired file
                return None

            logger.debug(f"Cache hit for key: {key}")
            return entry.value

        except Exception as e:
            logger.warning(f"Error reading cache for {key}: {e}")
            return None

    def put(self, key: str, value: Any, ttl_hours: int | None = None) -> None:
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl_hours: Override default TTL
        """
        try:
            ttl = ttl_hours or self.ttl_hours
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl_hours=ttl,
            )

            cache_path = self._get_cache_path(key)

            # Serialize to dict for JSON
            entry_dict = {
                "value": value,
                "timestamp": entry.timestamp,
                "ttl_hours": entry.ttl_hours,
            }

            with open(cache_path, "w") as f:
                json.dump(entry_dict, f)

            logger.debug(f"Cached key: {key} (TTL: {ttl}h)")

        except Exception as e:
            logger.warning(f"Error writing cache for {key}: {e}")

    def evict_expired(self) -> int:
        """
        Clean up expired cache entries.

        Returns:
            Number of entries evicted
        """
        evicted = 0

        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file) as f:
                        data = json.load(f)

                    entry = CacheEntry(**data)

                    if entry.is_expired():
                        cache_file.unlink()
                        evicted += 1

                except Exception as e:
                    logger.debug(f"Error checking cache file {cache_file}: {e}")

            if evicted > 0:
                logger.info(f"Evicted {evicted} expired cache entries")

        except Exception as e:
            logger.warning(f"Error evicting expired entries: {e}")

        return evicted

    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Cleared all cache entries")
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")

    def stats(self) -> dict:
        """Get cache statistics."""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in cache_files) / 1024  # KB

            return {
                "num_entries": len(cache_files),
                "total_size_kb": round(total_size, 2),
                "cache_dir": str(self.cache_dir),
            }
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            return {}


# Global cache instance
_cache = None


def get_cache(cache_dir: str = ".cache/grounded_memory", ttl_hours: int = 168) -> FileCache:
    """
    Get or create global cache instance.

    Args:
        cache_dir: Cache directory
        ttl_hours: Default TTL in hours

    Returns:
        FileCache instance
    """
    global _cache

    if _cache is None:
        _cache = FileCache(cache_dir=cache_dir, ttl_hours=ttl_hours)

    return _cache


def cache_get(key: str) -> Any | None:
    """Get value from global cache."""
    return get_cache().get(key)


def cache_put(key: str, value: Any, ttl_hours: int | None = None) -> None:
    """Put value in global cache."""
    get_cache().put(key, value, ttl_hours=ttl_hours)


def cache_clear() -> None:
    """Clear global cache."""
    get_cache().clear()


def cache_stats() -> dict:
    """Get global cache statistics."""
    return get_cache().stats()
