"""
Comprehensive tests for multi-model caching support.

This test suite verifies that the caching system correctly handles multiple models
by ensuring that cached responses are isolated by model name.
"""

import pytest
import tempfile
from pathlib import Path

from sglang_cached.cache_manager import CacheManager
from sglang_cached.hashing import generate_cache_key, normalize_request


class TestMultiModelCacheKeys:
    """Test that different models generate different cache keys."""

    def test_different_models_different_keys(self):
        """Different models should always generate different cache keys."""
        base_request = {
            "text": "What is 2+2?",
            "sampling_params": {"temperature": 0.7, "max_new_tokens": 100}
        }

        models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "llama-2-70b", "mistral-7b"]
        keys = []

        for model in models:
            request = {**base_request, "model": model}
            key = generate_cache_key(request)
            keys.append(key)

        # All keys should be unique
        assert len(keys) == len(set(keys)), "Different models produced duplicate cache keys"

    def test_same_model_same_key(self):
        """Same model with identical parameters should generate same key."""
        request = {
            "text": "Hello world",
            "model": "gpt-4",
            "sampling_params": {"temperature": 0.8}
        }

        key1 = generate_cache_key(request)
        key2 = generate_cache_key(request)
        key3 = generate_cache_key(request)

        assert key1 == key2 == key3

    def test_model_vs_no_model(self):
        """Request with model should have different key than without model."""
        request_with_model = {
            "text": "Test",
            "model": "gpt-4",
            "sampling_params": {"temperature": 0.7}
        }
        request_without_model = {
            "text": "Test",
            "sampling_params": {"temperature": 0.7}
        }

        key1 = generate_cache_key(request_with_model)
        key2 = generate_cache_key(request_without_model)

        assert key1 != key2

    def test_model_with_different_params(self):
        """Same model with different params should have different keys."""
        request1 = {
            "text": "Test",
            "model": "gpt-4",
            "sampling_params": {"temperature": 0.7}
        }
        request2 = {
            "text": "Test",
            "model": "gpt-4",
            "sampling_params": {"temperature": 0.9}
        }

        key1 = generate_cache_key(request1)
        key2 = generate_cache_key(request2)

        assert key1 != key2

    def test_model_with_same_params_different_n(self):
        """Same model, same params, different n should have SAME key (n excluded)."""
        request1 = {
            "text": "Test",
            "model": "gpt-4",
            "sampling_params": {"temperature": 0.7, "n": 1}
        }
        request2 = {
            "text": "Test",
            "model": "gpt-4",
            "sampling_params": {"temperature": 0.7, "n": 5}
        }

        key1 = generate_cache_key(request1)
        key2 = generate_cache_key(request2)

        assert key1 == key2


class TestMultiModelCacheManager:
    """Test CacheManager with multiple models."""

    def test_cache_isolation_by_model(self):
        """Responses should be isolated by model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir)

            # Store responses for different models with same input
            request_gpt4 = {
                "text": "Hello",
                "model": "gpt-4",
                "sampling_params": {"temperature": 0.7}
            }
            request_gpt35 = {
                "text": "Hello",
                "model": "gpt-3.5-turbo",
                "sampling_params": {"temperature": 0.7}
            }

            # Put different responses for each model
            cache.put(request_gpt4, [{"text": "GPT-4 response"}])
            cache.put(request_gpt35, [{"text": "GPT-3.5 response"}])

            # Retrieve and verify isolation
            cached_gpt4, _ = cache.get(request_gpt4)
            cached_gpt35, _ = cache.get(request_gpt35)

            assert len(cached_gpt4) == 1
            assert len(cached_gpt35) == 1
            assert cached_gpt4[0]["text"] == "GPT-4 response"
            assert cached_gpt35[0]["text"] == "GPT-3.5 response"

            cache.shutdown()

    def test_cache_accumulation_per_model(self):
        """Each model should accumulate its own responses."""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir)

            request_gpt4 = {
                "text": "Count to 3",
                "model": "gpt-4",
                "sampling_params": {"temperature": 0.8}
            }
            request_claude = {
                "text": "Count to 3",
                "model": "claude-3-opus",
                "sampling_params": {"temperature": 0.8}
            }

            # Add responses incrementally for each model
            cache.put(request_gpt4, [{"text": "GPT-4 response 1"}, {"text": "GPT-4 response 2"}])
            cache.put(request_claude, [
                {"text": "Claude response 1"},
                {"text": "Claude response 2"},
                {"text": "Claude response 3"}
            ])

            # Give async writer a moment to process
            time.sleep(0.1)

            # Verify each model has its own count
            # Need to request with appropriate n values
            request_gpt4_n2 = {**request_gpt4, "sampling_params": {"temperature": 0.8, "n": 2}}
            request_claude_n3 = {**request_claude, "sampling_params": {"temperature": 0.8, "n": 3}}

            cached_gpt4, _ = cache.get(request_gpt4_n2)
            cached_claude, _ = cache.get(request_claude_n3)

            assert len(cached_gpt4) == 2
            assert len(cached_claude) == 3

            cache.shutdown()

    def test_persistence_with_multiple_models(self):
        """Cache should persist and reload correctly with multiple models."""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            # First cache instance
            cache1 = CacheManager(cache_dir=tmpdir)

            requests = [
                {"text": "Test", "model": "gpt-4", "sampling_params": {"temperature": 0.7}},
                {"text": "Test", "model": "gpt-3.5-turbo", "sampling_params": {"temperature": 0.7}},
                {"text": "Test", "model": "claude-3-opus", "sampling_params": {"temperature": 0.7}},
            ]

            for i, req in enumerate(requests):
                cache1.put(req, [{"text": f"Response from {req['model']}"}])

            # Wait for async writes to complete
            time.sleep(0.2)
            cache1.shutdown()

            # Second cache instance (loads from disk)
            cache2 = CacheManager(cache_dir=tmpdir)

            # Verify all models' responses were persisted
            for req in requests:
                cached, needed = cache2.get(req)
                assert len(cached) == 1
                assert f"Response from {req['model']}" in cached[0]["text"]
                assert needed == 0

            cache2.shutdown()

    def test_n_parameter_with_multiple_models(self):
        """Test n parameter logic works correctly with multiple models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir)

            # Add 3 responses for gpt-4
            request_gpt4 = {
                "text": "Test",
                "model": "gpt-4",
                "sampling_params": {"temperature": 0.7, "n": 3}
            }
            cache.put(request_gpt4, [
                {"text": "GPT-4 response 1"},
                {"text": "GPT-4 response 2"},
                {"text": "GPT-4 response 3"}
            ])

            # Add 2 responses for gpt-3.5
            request_gpt35 = {
                "text": "Test",
                "model": "gpt-3.5-turbo",
                "sampling_params": {"temperature": 0.7, "n": 2}
            }
            cache.put(request_gpt35, [
                {"text": "GPT-3.5 response 1"},
                {"text": "GPT-3.5 response 2"}
            ])

            # Request n=2 from gpt-4 (should return 2, need 0)
            request_gpt4_n2 = {**request_gpt4, "sampling_params": {"temperature": 0.7, "n": 2}}
            cached, needed = cache.get(request_gpt4_n2)
            assert len(cached) == 2
            assert needed == 0

            # Request n=5 from gpt-4 (should return 3, need 2)
            request_gpt4_n5 = {**request_gpt4, "sampling_params": {"temperature": 0.7, "n": 5}}
            cached, needed = cache.get(request_gpt4_n5)
            assert len(cached) == 3
            assert needed == 2

            # Request n=3 from gpt-3.5 (should return 2, need 1)
            request_gpt35_n3 = {**request_gpt35, "sampling_params": {"temperature": 0.7, "n": 3}}
            cached, needed = cache.get(request_gpt35_n3)
            assert len(cached) == 2
            assert needed == 1

            cache.shutdown()


class TestMultiModelNormalization:
    """Test request normalization with models."""

    def test_model_included_in_normalization(self):
        """Model field should be included in normalized request."""
        request = {
            "text": "Test",
            "model": "gpt-4-turbo",
            "sampling_params": {"temperature": 0.5}
        }
        normalized = normalize_request(request)

        assert "model" in normalized
        assert normalized["model"] == "gpt-4-turbo"

    def test_normalization_without_model(self):
        """Normalization should work fine without model field."""
        request = {
            "text": "Test",
            "sampling_params": {"temperature": 0.5}
        }
        normalized = normalize_request(request)

        assert "model" not in normalized
        assert "text" in normalized

    def test_normalization_preserves_model_case(self):
        """Model name case should be preserved."""
        request = {
            "text": "Test",
            "model": "GPT-4-TURBO",
            "sampling_params": {"temperature": 0.5}
        }
        normalized = normalize_request(request)

        assert normalized["model"] == "GPT-4-TURBO"

    def test_chat_format_with_model(self):
        """Chat format requests should include model in normalization."""
        request = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "model": "gpt-4",
            "sampling_params": {"temperature": 0.7}
        }
        normalized = normalize_request(request)

        assert "model" in normalized
        assert normalized["model"] == "gpt-4"
        assert "messages" in normalized


class TestMultiModelStressTest:
    """Stress test with many models and requests."""

    def test_many_models_concurrent(self):
        """Test cache with many different models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir)

            # Create 20 different models
            models = [f"model-{i}" for i in range(20)]
            inputs = ["Test A", "Test B", "Test C"]

            # Store responses for all combinations
            for model in models:
                for input_text in inputs:
                    request = {
                        "text": input_text,
                        "model": model,
                        "sampling_params": {"temperature": 0.7}
                    }
                    cache.put(request, [{"text": f"Response from {model} for {input_text}"}])

            # Verify all 60 combinations are cached correctly
            for model in models:
                for input_text in inputs:
                    request = {
                        "text": input_text,
                        "model": model,
                        "sampling_params": {"temperature": 0.7}
                    }
                    cached, needed = cache.get(request)
                    assert len(cached) == 1
                    assert needed == 0
                    assert cached[0]["text"] == f"Response from {model} for {input_text}"

            stats = cache.get_stats()
            assert stats["num_keys"] == 60  # 20 models * 3 inputs

            cache.shutdown()

    def test_model_name_variations(self):
        """Test that model name variations are treated as different models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir)

            # These should all be treated as different models
            model_variations = [
                "gpt-4",
                "gpt4",
                "GPT-4",
                "gpt-4-turbo",
                "gpt-4-0613",
            ]

            request_base = {
                "text": "Hello",
                "sampling_params": {"temperature": 0.7}
            }

            # Store different responses for each variation
            for model in model_variations:
                request = {**request_base, "model": model}
                cache.put(request, [{"text": f"Response from {model}"}])

            # Verify each variation has its own cache entry
            for model in model_variations:
                request = {**request_base, "model": model}
                cached, needed = cache.get(request)
                assert len(cached) == 1
                assert cached[0]["text"] == f"Response from {model}"

            # Should have 5 different cache keys
            stats = cache.get_stats()
            assert stats["num_keys"] == 5

            cache.shutdown()


class TestMultiModelEdgeCases:
    """Test edge cases in multi-model caching."""

    def test_empty_model_string(self):
        """Empty string model should be treated differently from no model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir)

            request_empty = {
                "text": "Test",
                "model": "",
                "sampling_params": {"temperature": 0.7}
            }
            request_none = {
                "text": "Test",
                "sampling_params": {"temperature": 0.7}
            }

            key1 = generate_cache_key(request_empty)
            key2 = generate_cache_key(request_none)

            assert key1 != key2

            cache.shutdown()

    def test_model_with_special_characters(self):
        """Model names with special characters should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir)

            models = [
                "model-name-with-dashes",
                "model_name_with_underscores",
                "model.name.with.dots",
                "model:name:with:colons",
                "model/name/with/slashes",
            ]

            for model in models:
                request = {
                    "text": "Test",
                    "model": model,
                    "sampling_params": {"temperature": 0.7}
                }
                cache.put(request, [{"text": f"Response for {model}"}])

            for model in models:
                request = {
                    "text": "Test",
                    "model": model,
                    "sampling_params": {"temperature": 0.7}
                }
                cached, needed = cache.get(request)
                assert len(cached) == 1
                assert cached[0]["text"] == f"Response for {model}"

            cache.shutdown()

    def test_unicode_model_names(self):
        """Unicode model names should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CacheManager(cache_dir=tmpdir)

            models = [
                "模型-中文",
                "モデル-日本語",
                "модель-русский",
            ]

            for model in models:
                request = {
                    "text": "Test",
                    "model": model,
                    "sampling_params": {"temperature": 0.7}
                }
                cache.put(request, [{"text": f"Response for {model}"}])

            for model in models:
                request = {
                    "text": "Test",
                    "model": model,
                    "sampling_params": {"temperature": 0.7}
                }
                cached, needed = cache.get(request)
                assert len(cached) == 1

            cache.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
