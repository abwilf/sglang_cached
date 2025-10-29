"""
Test that model is required and properly used in cache keys.

This test suite verifies that:
1. The model parameter is required in all requests
2. Different models generate different cache keys
3. Same model with same parameters (except n) generates same cache key
4. Responses are properly cached per model
"""

import pytest
import tempfile
from pathlib import Path

from sglang_cached.cache_manager import CacheManager
from sglang_cached.hashing import generate_cache_key, normalize_request


class TestModelRequired:
    """Test that model is required in all requests."""

    def test_normalize_request_without_model_raises_error(self):
        """normalize_request should raise ValueError if model is missing."""
        request_without_model = {
            "text": "Hello world",
            "sampling_params": {"temperature": 0.7}
        }

        with pytest.raises(ValueError) as exc_info:
            normalize_request(request_without_model)

        assert "model" in str(exc_info.value).lower()
        assert "required" in str(exc_info.value).lower()

    def test_generate_cache_key_without_model_raises_error(self):
        """generate_cache_key should raise ValueError if model is missing."""
        request_without_model = {
            "text": "Hello world",
            "sampling_params": {"temperature": 0.7}
        }

        with pytest.raises(ValueError) as exc_info:
            generate_cache_key(request_without_model)

        assert "model" in str(exc_info.value).lower()
        assert "required" in str(exc_info.value).lower()

    def test_cache_manager_get_without_model_raises_error(self):
        """CacheManager.get should raise ValueError if model is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir)

            request_without_model = {
                "text": "Hello world",
                "sampling_params": {"temperature": 0.7}
            }

            with pytest.raises(ValueError) as exc_info:
                cache.get(request_without_model)

            assert "model" in str(exc_info.value).lower()
            assert "required" in str(exc_info.value).lower()

            cache.shutdown()

    def test_cache_manager_put_without_model_raises_error(self):
        """CacheManager.put should raise ValueError if model is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir)

            request_without_model = {
                "text": "Hello world",
                "sampling_params": {"temperature": 0.7}
            }

            with pytest.raises(ValueError) as exc_info:
                cache.put(request_without_model, [{"text": "Hi there"}])

            assert "model" in str(exc_info.value).lower()
            assert "required" in str(exc_info.value).lower()

            cache.shutdown()


class TestModelInCacheKey:
    """Test that model is properly included in cache keys."""

    def test_different_models_different_cache_keys(self):
        """Different models should produce different cache keys."""
        base_request = {
            "text": "What is 2+2?",
            "sampling_params": {"temperature": 0.7, "max_new_tokens": 100}
        }

        models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "llama-2-70b"]
        cache_keys = []

        for model in models:
            request = {**base_request, "model": model}
            key = generate_cache_key(request)
            cache_keys.append(key)

        # All cache keys should be unique
        assert len(cache_keys) == len(set(cache_keys)), (
            f"Expected {len(models)} unique cache keys, but got duplicates. "
            f"Keys: {cache_keys}"
        )

    def test_same_model_same_cache_key(self):
        """Same model with identical parameters should produce same cache key."""
        request1 = {
            "text": "Hello",
            "model": "gpt-4",
            "sampling_params": {"temperature": 0.8, "max_new_tokens": 50}
        }
        request2 = {
            "text": "Hello",
            "model": "gpt-4",
            "sampling_params": {"temperature": 0.8, "max_new_tokens": 50}
        }

        key1 = generate_cache_key(request1)
        key2 = generate_cache_key(request2)

        assert key1 == key2

    def test_same_model_different_params_different_keys(self):
        """Same model but different parameters should produce different cache keys."""
        request1 = {
            "text": "Hello",
            "model": "gpt-4",
            "sampling_params": {"temperature": 0.7}
        }
        request2 = {
            "text": "Hello",
            "model": "gpt-4",
            "sampling_params": {"temperature": 0.9}
        }

        key1 = generate_cache_key(request1)
        key2 = generate_cache_key(request2)

        assert key1 != key2

    def test_n_parameter_excluded_from_cache_key(self):
        """The n parameter should be excluded from cache key."""
        request1 = {
            "text": "Hello",
            "model": "gpt-4",
            "sampling_params": {"temperature": 0.7, "n": 1}
        }
        request2 = {
            "text": "Hello",
            "model": "gpt-4",
            "sampling_params": {"temperature": 0.7, "n": 5}
        }

        key1 = generate_cache_key(request1)
        key2 = generate_cache_key(request2)

        # Keys should be the same since only n differs
        assert key1 == key2


class TestModelBasedCaching:
    """Test that caching works correctly with model parameter."""

    def test_responses_cached_per_model(self):
        """Responses should be cached separately for each model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir)

            # Same input, different models
            request_gpt4 = {
                "text": "What is AI?",
                "model": "gpt-4",
                "sampling_params": {"temperature": 0.7}
            }
            request_claude = {
                "text": "What is AI?",
                "model": "claude-3-opus",
                "sampling_params": {"temperature": 0.7}
            }

            # Store different responses
            cache.put(request_gpt4, [{"text": "GPT-4's answer about AI"}])
            cache.put(request_claude, [{"text": "Claude's answer about AI"}])

            # Retrieve and verify isolation
            cached_gpt4, _ = cache.get(request_gpt4)
            cached_claude, _ = cache.get(request_claude)

            assert len(cached_gpt4) == 1
            assert len(cached_claude) == 1
            assert cached_gpt4[0]["text"] == "GPT-4's answer about AI"
            assert cached_claude[0]["text"] == "Claude's answer about AI"

            cache.shutdown()

    def test_cache_hit_for_same_model(self):
        """Cache should hit when requesting with same model and params."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir)

            request = {
                "text": "Test prompt",
                "model": "gpt-4",
                "sampling_params": {"temperature": 0.5, "max_new_tokens": 100}
            }

            # First request - cache miss
            cached1, needed1 = cache.get(request)
            assert len(cached1) == 0
            assert needed1 == 1

            # Store response
            cache.put(request, [{"text": "Response from GPT-4"}])

            # Second request - cache hit
            cached2, needed2 = cache.get(request)
            assert len(cached2) == 1
            assert needed2 == 0
            assert cached2[0]["text"] == "Response from GPT-4"

            cache.shutdown()

    def test_cache_miss_for_different_model(self):
        """Cache should miss when requesting with different model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir)

            request_gpt4 = {
                "text": "Test prompt",
                "model": "gpt-4",
                "sampling_params": {"temperature": 0.5}
            }
            request_claude = {
                "text": "Test prompt",
                "model": "claude-3-opus",
                "sampling_params": {"temperature": 0.5}
            }

            # Cache response for gpt-4
            cache.put(request_gpt4, [{"text": "Response from GPT-4"}])

            # Request with claude should miss
            cached_claude, needed_claude = cache.get(request_claude)
            assert len(cached_claude) == 0
            assert needed_claude == 1

            cache.shutdown()

    def test_n_parameter_works_with_model(self):
        """Test that n parameter logic works correctly with model field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir)

            # Store 3 responses for gpt-4
            request_base = {
                "text": "Generate a greeting",
                "model": "gpt-4",
                "sampling_params": {"temperature": 0.8}
            }

            cache.put(request_base, [
                {"text": "Hello!"},
                {"text": "Hi there!"},
                {"text": "Greetings!"}
            ])

            # Request n=2 should return 2, need 0
            request_n2 = {
                **request_base,
                "sampling_params": {"temperature": 0.8, "n": 2}
            }
            cached, needed = cache.get(request_n2)
            assert len(cached) == 2
            assert needed == 0

            # Request n=5 should return 3, need 2
            request_n5 = {
                **request_base,
                "sampling_params": {"temperature": 0.8, "n": 5}
            }
            cached, needed = cache.get(request_n5)
            assert len(cached) == 3
            assert needed == 2

            cache.shutdown()


class TestModelPersistence:
    """Test that model-based caching persists correctly."""

    def test_cache_persists_with_multiple_models(self):
        """Cache should persist and reload correctly with multiple models."""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            # First cache instance
            cache1 = CacheManager(cache_dir=tmpdir)

            models = ["gpt-4", "claude-3-opus", "llama-2-70b"]
            for model in models:
                request = {
                    "text": "Test",
                    "model": model,
                    "sampling_params": {"temperature": 0.7}
                }
                cache1.put(request, [{"text": f"Response from {model}"}])

            # Wait for async writes
            time.sleep(0.2)
            cache1.shutdown()

            # Second cache instance - should load from disk
            cache2 = CacheManager(cache_dir=tmpdir)

            # Verify all models' responses persisted
            for model in models:
                request = {
                    "text": "Test",
                    "model": model,
                    "sampling_params": {"temperature": 0.7}
                }
                cached, needed = cache2.get(request)
                assert len(cached) == 1
                assert needed == 0
                assert cached[0]["text"] == f"Response from {model}"

            cache2.shutdown()


class TestEntireResponseCached:
    """Test that entire response (except n parameter) is cached."""

    def test_full_response_cached_including_metadata(self):
        """Entire response dict should be cached, not just text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir)

            request = {
                "text": "Test",
                "model": "gpt-4",
                "sampling_params": {
                    "temperature": 0.7,
                    "max_new_tokens": 100,
                    "top_p": 0.9
                }
            }

            # Store a response with metadata
            full_response = {
                "text": "This is the generated text",
                "tokens": [1, 2, 3, 4, 5],
                "finish_reason": "stop",
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "metadata": {"model_version": "gpt-4-0613"}
            }

            cache.put(request, [full_response])

            # Retrieve and verify all fields are cached
            cached, needed = cache.get(request)
            assert len(cached) == 1
            assert needed == 0

            retrieved = cached[0]
            assert retrieved["text"] == "This is the generated text"
            assert retrieved["tokens"] == [1, 2, 3, 4, 5]
            assert retrieved["finish_reason"] == "stop"
            assert retrieved["prompt_tokens"] == 10
            assert retrieved["completion_tokens"] == 15
            assert retrieved["metadata"]["model_version"] == "gpt-4-0613"

            cache.shutdown()

    def test_all_request_params_except_n_affect_cache(self):
        """All parameters except n should affect the cache key."""
        base_request = {
            "text": "Hello",
            "model": "gpt-4",
        }

        # Different values for various parameters should produce different keys
        params_variations = [
            {"temperature": 0.7},
            {"temperature": 0.8},  # Different temperature
            {"temperature": 0.7, "max_new_tokens": 100},  # Added max_new_tokens
            {"temperature": 0.7, "top_p": 0.9},  # Added top_p
            {"temperature": 0.7, "seed": 42},  # Added seed
        ]

        cache_keys = []
        for params in params_variations:
            request = {**base_request, "sampling_params": params}
            key = generate_cache_key(request)
            cache_keys.append(key)

        # All keys should be unique
        assert len(cache_keys) == len(set(cache_keys)), (
            "Expected all parameter variations to produce unique cache keys"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
