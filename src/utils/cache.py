"""Caching utilities for responses and data."""

import json
import hashlib
import time
import pickle
from pathlib import Path
from typing import Any, Optional, Dict
import logging

class CacheManager:
    """Cache manager for storing and retrieving responses."""
    
    def __init__(self, cache_dir: str = "data/cache", ttl_seconds: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get(self, key: Any, cache_type: str = "general") -> Optional[Any]:
        """Get data from cache."""
        try:
            cache_key = hashlib.md5(str(key).encode()).hexdigest()
            cache_subdir = self.cache_dir / cache_type
            cache_subdir.mkdir(exist_ok=True)
            cache_path = cache_subdir / f"{cache_key}.cache"
            
            if not cache_path.exists():
                return None
            
            # Check if cache is expired
            if time.time() - cache_path.stat().st_mtime > self.ttl_seconds:
                cache_path.unlink()
                return None
            
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
                
        except Exception as e:
            self.logger.warning(f"Failed to read from cache: {e}")
            return None
    
    def set(self, key: Any, value: Any, cache_type: str = "general"):
        """Store data in cache."""
        try:
            cache_key = hashlib.md5(str(key).encode()).hexdigest()
            cache_subdir = self.cache_dir / cache_type
            cache_subdir.mkdir(exist_ok=True)
            cache_path = cache_subdir / f"{cache_key}.cache"
            
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
                
        except Exception as e:
            self.logger.warning(f"Failed to write to cache: {e}") 