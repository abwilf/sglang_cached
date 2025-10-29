"""
Cache manager for SGLang responses.

Provides in-memory caching with asynchronous file persistence. The cache stores
multiple responses per cache key to support the `n` parameter (number of completions).
"""

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .hashing import generate_cache_key, extract_n_parameter


class CacheManager:
    """
    Manages response cache with in-memory storage and async file persistence.

    The cache is a mapping from cache keys to lists of responses. Each cache key
    represents a unique combination of input and sampling parameters (excluding `n`).
    Multiple responses can be stored per key to handle different `n` values.
    """

    def __init__(self, cache_dir: Optional[str] = None, overwrite: bool = False):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory to store cache file. Defaults to ~/.sglang_cache
            overwrite: If True, remove existing cache file before loading
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.sglang_cache")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "cache.jsonl"

        # In-memory cache: cache_key -> list of responses
        self._cache: Dict[str, List[Dict]] = {}
        self._lock = threading.Lock()

        # Async writer
        self._write_queue: queue.Queue = queue.Queue()
        self._writer_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

        # Stats
        self._hits = 0
        self._misses = 0

        # Remove existing cache if overwrite is requested
        if overwrite and self.cache_file.exists():
            self.cache_file.unlink()
            print(f"Removed existing cache file: {self.cache_file}")

        # Load existing cache and start writer
        self._load_cache()
        self._start_writer()

    def _load_cache(self):
        """Load cache from disk into memory."""
        if not self.cache_file.exists():
            return

        try:
            with open(self.cache_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        cache_key = entry["cache_key"]
                        responses = entry["responses"]
                        self._cache[cache_key] = responses
        except Exception as e:
            print(f"Warning: Failed to load cache from {self.cache_file}: {e}")

    def _start_writer(self):
        """Start the async writer thread."""
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def _writer_loop(self):
        """Background thread that writes cache updates to disk."""
        while not self._shutdown.is_set():
            try:
                # Wait for write requests with timeout to check shutdown flag
                cache_key, responses = self._write_queue.get(timeout=1.0)
                self._write_to_disk(cache_key, responses)
                self._write_queue.task_done()
            except queue.Empty:
                continue

    def _write_to_disk(self, cache_key: str, responses: List[Dict]):
        """
        Write a single cache entry to disk.

        Uses atomic write pattern: write to temp file, then rename.
        """
        try:
            # Read existing cache
            existing_cache = {}
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            existing_cache[entry["cache_key"]] = entry["responses"]

            # Update with new entry
            existing_cache[cache_key] = responses

            # Write atomically
            temp_file = self.cache_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                for key, resps in existing_cache.items():
                    entry = {"cache_key": key, "responses": resps}
                    f.write(json.dumps(entry) + '\n')

            # Atomic rename
            temp_file.replace(self.cache_file)

        except Exception as e:
            print(f"Warning: Failed to write cache to disk: {e}")

    def get(self, request_data: Dict) -> Tuple[List[Dict], int]:
        """
        Get cached responses for a request.

        Implements the key `n` parameter logic:
        - If cache has >= n responses: return first n
        - If cache has < n responses: return all cached + number of additional needed

        Args:
            request_data: The request dictionary

        Returns:
            Tuple of (cached_responses, num_needed)
            - cached_responses: List of cached responses (may be empty)
            - num_needed: Number of additional responses needed from SGLang
        """
        cache_key = generate_cache_key(request_data)
        n = extract_n_parameter(request_data)

        with self._lock:
            cached_responses = self._cache.get(cache_key, [])
            num_cached = len(cached_responses)

            if num_cached >= n:
                # Full cache hit - return a copy
                self._hits += 1
                return cached_responses[:n].copy(), 0
            else:
                # Partial or full miss - return a copy to avoid aliasing issues
                if num_cached > 0:
                    self._hits += 1  # Partial hit
                else:
                    self._misses += 1  # Full miss
                return cached_responses.copy(), n - num_cached

    def put(self, request_data: Dict, new_responses: List[Dict]):
        """
        Add new responses to the cache.

        This appends new responses to the existing cached responses for this key.
        The write to disk happens asynchronously.

        Args:
            request_data: The request dictionary
            new_responses: List of new response dicts to add to cache
        """
        if not new_responses:
            return

        cache_key = generate_cache_key(request_data)

        with self._lock:
            # Append to existing responses
            if cache_key not in self._cache:
                self._cache[cache_key] = []
            self._cache[cache_key].extend(new_responses)

            # Queue async write
            responses_copy = self._cache[cache_key].copy()

        # Non-blocking: queue the write operation
        self._write_queue.put((cache_key, responses_copy))

    def clear(self):
        """Clear all cached responses."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

        # Clear file
        if self.cache_file.exists():
            self.cache_file.unlink()

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            return {
                "num_keys": len(self._cache),
                "total_responses": sum(len(v) for v in self._cache.values()),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0.0,
                "cache_file": str(self.cache_file),
                "pending_writes": self._write_queue.qsize()
            }

    def shutdown(self):
        """Shutdown the cache manager and wait for pending writes."""
        self._shutdown.set()
        if self._writer_thread and self._writer_thread.is_alive():
            # Wait for thread to finish (it checks shutdown flag every 1s)
            self._writer_thread.join(timeout=2.0)
