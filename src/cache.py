"""
Ground Truth Cache Management for Finance Green Agent.

This module provides caching for LLM-generated ground truth answers to ensure
reproducible benchmark results across multiple evaluation runs.
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Literal
from datetime import datetime


class GroundTruthCache:
    """Manages persistent cache for ground truth evaluation data."""
    
    CACHE_VERSION = "1.0.0"
    
    def __init__(self, cache_path: str | Path = "data/ground_truth_cache.json"):
        """
        Initialize cache manager.
        
        Args:
            cache_path: Path to cache file (default: data/ground_truth_cache.json)
        """
        self.cache_path = Path(cache_path)
        self._data: dict[str, Any] = {}
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cache from disk, handling corruption gracefully."""
        if not self.cache_path.exists():
            self._initialize_new_cache()
            return
        
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Validate cache structure
            if not isinstance(cache_data, dict):
                raise ValueError("Invalid cache structure")
            
            # Check version compatibility
            cache_version = cache_data.get("cache_version", "0.0.0")
            if cache_version != self.CACHE_VERSION:
                print(f"⚠️  Cache version mismatch (found {cache_version}, expected {self.CACHE_VERSION})")
                print("   Creating backup and initializing new cache...")
                self._backup_and_reset()
                return
            
            self._data = cache_data
            print(f"✓ Loaded cache with {len(self._data.get('entries', {}))} entries")
            
        except (json.JSONDecodeError, ValueError, IOError) as e:
            print(f"⚠️  Cache file corrupted: {e}")
            print("   Creating backup and initializing new cache...")
            self._backup_and_reset()
    
    def _initialize_new_cache(self) -> None:
        """Initialize a new empty cache."""
        self._data = {
            "cache_version": self.CACHE_VERSION,
            "created_at": datetime.now().isoformat(),
            "model": "deepseek/deepseek-v3.2",  # Default model
            "entries": {}
        }
        self._save_cache()
        print("✓ Initialized new cache")
    
    def _backup_and_reset(self) -> None:
        """Create backup of corrupted cache and reset."""
        if self.cache_path.exists():
            backup_path = self.cache_path.with_suffix('.json.backup')
            self.cache_path.rename(backup_path)
            print(f"   Backup saved to: {backup_path}")
        self._initialize_new_cache()
    
    def _save_cache(self) -> None:
        """Persist cache to disk."""
        # Ensure directory exists
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update metadata
        self._data["last_updated"] = datetime.now().isoformat()
        
        # Write atomically (write to temp file, then rename)
        temp_path = self.cache_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            temp_path.rename(self.cache_path)
        except Exception as e:
            print(f"❌ Failed to save cache: {e}")
            if temp_path.exists():
                temp_path.unlink()
    
    def _generate_cache_key(
        self, 
        cik: str, 
        year: str, 
        task: Literal["risk", "business", "consistency"],
        model: str | None = None
    ) -> str:
        """
        Generate a unique cache key for a ground truth query.
        
        Args:
            cik: Company CIK identifier
            year: Filing year
            task: Task type (risk/business/consistency)
            model: Optional model identifier for versioning
            
        Returns:
            Unique cache key string
        """
        # Use model from cache metadata if not provided
        if model is None:
            model = self._data.get("model", "unknown")
        
        # Create a simple but unique key
        # Format: {cik}_{year}_{task}_{model_hash}
        model_hash = hashlib.md5(model.encode()).hexdigest()[:8]
        return f"{cik}_{year}_{task}_{model_hash}"
    
    def get(
        self, 
        cik: str, 
        year: str, 
        task: Literal["risk", "business", "consistency"]
    ) -> dict[str, Any] | None:
        """
        Retrieve cached ground truth if available.
        
        Args:
            cik: Company CIK identifier
            year: Filing year
            task: Task type
            
        Returns:
            Cached data if found, None otherwise
        """
        cache_key = self._generate_cache_key(cik, year, task)
        entries = self._data.get("entries", {})
        
        if cache_key in entries:
            print(f"✓ Cache HIT: {cache_key}")
            return entries[cache_key]
        
        print(f"⚠ Cache MISS: {cache_key}")
        return None
    
    def set(
        self, 
        cik: str, 
        year: str, 
        task: Literal["risk", "business", "consistency"],
        data: dict[str, Any]
    ) -> None:
        """
        Store ground truth in cache.
        
        Args:
            cik: Company CIK identifier
            year: Filing year
            task: Task type
            data: Ground truth data to cache
        """
        cache_key = self._generate_cache_key(cik, year, task)
        
        # Ensure entries dict exists
        if "entries" not in self._data:
            self._data["entries"] = {}
        
        # Store with metadata
        self._data["entries"][cache_key] = {
            "data": data,
            "cached_at": datetime.now().isoformat(),
            "cik": cik,
            "year": year,
            "task": task
        }
        
        self._save_cache()
        print(f"✓ Cache SAVE: {cache_key}")
    
    def invalidate(
        self, 
        cik: str | None = None, 
        year: str | None = None, 
        task: str | None = None
    ) -> int:
        """
        Invalidate cache entries matching the criteria.
        
        Args:
            cik: Optional CIK filter
            year: Optional year filter
            task: Optional task filter
            
        Returns:
            Number of entries removed
        """
        if "entries" not in self._data:
            return 0
        
        entries = self._data["entries"]
        to_remove = []
        
        for key, entry in entries.items():
            match = True
            if cik and entry.get("cik") != cik:
                match = False
            if year and entry.get("year") != year:
                match = False
            if task and entry.get("task") != task:
                match = False
            
            if match:
                to_remove.append(key)
        
        for key in to_remove:
            del entries[key]
        
        if to_remove:
            self._save_cache()
            print(f"✓ Invalidated {len(to_remove)} cache entries")
        
        return len(to_remove)
    
    def stats(self) -> dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        entries = self._data.get("entries", {})
        
        # Count by task type
        by_task = {}
        for entry in entries.values():
            task = entry.get("task", "unknown")
            by_task[task] = by_task.get(task, 0) + 1
        
        return {
            "version": self._data.get("cache_version"),
            "total_entries": len(entries),
            "by_task": by_task,
            "created_at": self._data.get("created_at"),
            "last_updated": self._data.get("last_updated"),
            "model": self._data.get("model")
        }
