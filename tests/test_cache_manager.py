"""
Unit tests for CacheManager.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from sglang_cached.cache_manager import CacheManager


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def cache_manager(temp_cache_dir):
    """Create a CacheManager instance."""
    manager = CacheManager(cache_dir=temp_cache_dir)
    yield manager
    manager.shutdown()


class TestCacheBasics:
    """Test basic cache operations."""

    def test_empty_cache(self, cache_manager):
        """Test behavior with empty cache."""
        request = {
            "text": "Hello",
            "sampling_params": {"temperature": 0.8, "n": 1}
        }

        cached, needed = cache_manager.get(request)

        assert cached == []
        assert needed == 1

    def test_full_cache_hit(self, cache_manager):
        """Test full cache hit when enough responses cached."""
        request = {
            "text": "Hello",
            "sampling_params": {"temperature": 0.8, "n": 2}
        }

        # Add responses to cache
        responses = [
            {"text": "response 1", "meta_info": {"id": "1"}},
            {"text": "response 2", "meta_info": {"id": "2"}},
            {"text": "response 3", "meta_info": {"id": "3"}}
        ]
        cache_manager.put(request, responses)

        # Request 2 responses (cache has 3)
        cached, needed = cache_manager.get(request)

        assert len(cached) == 2
        assert needed == 0
        assert cached[0]["text"] == "response 1"
        assert cached[1]["text"] == "response 2"

    def test_partial_cache_hit(self, cache_manager):
        """Test partial cache hit when some responses cached."""
        request = {
            "text": "Hello",
            "sampling_params": {"temperature": 0.8, "n": 5}
        }

        # Add 2 responses to cache
        responses = [
            {"text": "response 1", "meta_info": {}},
            {"text": "response 2", "meta_info": {}}
        ]
        cache_manager.put(request, responses)

        # Request 5 responses (cache has 2)
        cached, needed = cache_manager.get(request)

        assert len(cached) == 2
        assert needed == 3


class TestNParameterLogic:
    """Test the n parameter logic."""

    def test_n_equals_one(self, cache_manager):
        """Test with n=1."""
        request = {
            "text": "Test",
            "sampling_params": {"n": 1}
        }

        # Add response
        cache_manager.put(request, [{"text": "response", "meta_info": {}}])

        # Get with n=1
        cached, needed = cache_manager.get(request)

        assert len(cached) == 1
        assert needed == 0

    def test_increasing_n(self, cache_manager):
        """Test requesting increasing n values."""
        request_base = {
            "text": "Test",
            "sampling_params": {"temperature": 0.8}
        }

        # Request n=1, should need 1
        request1 = {**request_base, "sampling_params": {"temperature": 0.8, "n": 1}}
        cached, needed = cache_manager.get(request1)
        assert needed == 1

        # Add 1 response
        cache_manager.put(request1, [{"text": "r1", "meta_info": {}}])

        # Request n=3, should get 1 and need 2 more
        request3 = {**request_base, "sampling_params": {"temperature": 0.8, "n": 3}}
        cached, needed = cache_manager.get(request3)
        assert len(cached) == 1
        assert needed == 2

        # Add 2 more responses
        cache_manager.put(request3, [{"text": "r2", "meta_info": {}}, {"text": "r3", "meta_info": {}}])

        # Request n=5, should get 3 and need 2 more
        request5 = {**request_base, "sampling_params": {"temperature": 0.8, "n": 5}}
        cached, needed = cache_manager.get(request5)
        assert len(cached) == 3
        assert needed == 2

    def test_append_to_cache(self, cache_manager):
        """Test that put() appends to existing cache."""
        request = {
            "text": "Test",
            "sampling_params": {"n": 1}
        }

        # Add first response
        cache_manager.put(request, [{"text": "r1", "meta_info": {}}])

        # Add second response
        cache_manager.put(request, [{"text": "r2", "meta_info": {}}])

        # Should have both
        cached, needed = cache_manager.get({**request, "sampling_params": {"n": 2}})
        assert len(cached) == 2
        assert cached[0]["text"] == "r1"
        assert cached[1]["text"] == "r2"


class TestPersistence:
    """Test cache persistence to disk."""

    def test_save_and_load(self, temp_cache_dir):
        """Test that cache is saved to disk and can be loaded."""
        request = {
            "text": "Persistent test",
            "sampling_params": {"temperature": 0.8, "n": 1}
        }
        responses = [{"text": "response", "meta_info": {"id": "1"}}]

        # Create cache, add data, shutdown
        cache1 = CacheManager(cache_dir=temp_cache_dir)
        cache1.put(request, responses)
        cache1.shutdown()

        # Wait a bit for async write
        import time
        time.sleep(0.5)

        # Create new cache manager, should load from disk
        cache2 = CacheManager(cache_dir=temp_cache_dir)
        cached, needed = cache2.get(request)

        assert len(cached) == 1
        assert cached[0]["text"] == "response"
        assert needed == 0

        cache2.shutdown()


class TestStats:
    """Test cache statistics."""

    def test_stats_tracking(self, cache_manager):
        """Test that stats are tracked correctly."""
        request = {
            "text": "Test",
            "sampling_params": {"n": 1}
        }

        # Initial stats
        stats = cache_manager.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

        # Cache miss
        cache_manager.get(request)
        stats = cache_manager.get_stats()
        assert stats["misses"] == 1

        # Add to cache
        cache_manager.put(request, [{"text": "r", "meta_info": {}}])

        # Cache hit
        cache_manager.get(request)
        stats = cache_manager.get_stats()
        assert stats["hits"] == 1

    def test_clear_cache(self, cache_manager):
        """Test clearing the cache."""
        request = {"text": "Test", "sampling_params": {"n": 1}}
        cache_manager.put(request, [{"text": "r", "meta_info": {}}])

        # Verify it's there
        cached, _ = cache_manager.get(request)
        assert len(cached) == 1

        # Clear
        cache_manager.clear()

        # Should be empty
        cached, needed = cache_manager.get(request)
        assert len(cached) == 0
        assert needed == 1

        stats = cache_manager.get_stats()
        assert stats["num_keys"] == 0
