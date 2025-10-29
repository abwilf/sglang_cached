"""
Unit tests for cache key generation and request normalization.
"""

import pytest
from sglang_cached.hashing import (
    normalize_request,
    generate_cache_key,
    extract_n_parameter
)


class TestNormalization:
    """Test request normalization."""

    def test_normalize_with_text(self):
        """Test normalization with text input."""
        request = {
            "text": "Hello world",
            "model": "test-model",
            "sampling_params": {
                "temperature": 0.8,
                "max_new_tokens": 100,
                "n": 3
            }
        }
        normalized = normalize_request(request)

        # All fields preserved except 'n'
        assert "text" in normalized
        assert normalized["text"] == "Hello world"
        assert "sampling_params" in normalized
        assert "n" not in normalized["sampling_params"]
        assert normalized["sampling_params"]["temperature"] == 0.8

    def test_normalize_with_messages(self):
        """Test normalization with messages (chat format)."""
        request = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "model": "test-model",
            "sampling_params": {
                "temperature": 0.7
            }
        }
        normalized = normalize_request(request)

        assert "messages" in normalized
        assert normalized["messages"] == [{"role": "user", "content": "Hello"}]

    def test_normalize_excludes_n(self):
        """Test that n parameter is excluded from normalized request."""
        request = {
            "text": "Test",
            "model": "test-model",
            "sampling_params": {
                "temperature": 0.5,
                "n": 10,
                "max_new_tokens": 50
            }
        }
        normalized = normalize_request(request)

        assert "n" not in normalized["sampling_params"]
        assert "temperature" in normalized["sampling_params"]
        assert "max_new_tokens" in normalized["sampling_params"]

    def test_normalize_includes_model(self):
        """Test that model parameter is included in normalized request."""
        request = {
            "text": "Test",
            "model": "gpt-4",
            "sampling_params": {
                "temperature": 0.5
            }
        }
        normalized = normalize_request(request)

        assert "model" in normalized
        assert normalized["model"] == "gpt-4"
        assert "text" in normalized
        assert normalized["text"] == "Test"

    def test_normalize_preserves_all_fields(self):
        """Test that normalization preserves all fields except n."""
        request = {
            "text": "Test",
            "model": "gpt-4",
            "seed": 42,
            "stream": False,
            "n": 5,
            "sampling_params": {
                "temperature": 0.5,
                "n": 5
            }
        }
        normalized = normalize_request(request)

        # All fields preserved except 'n'
        assert "text" in normalized
        assert "model" in normalized
        assert "seed" in normalized
        assert "stream" in normalized
        assert "n" not in normalized  # Top-level n excluded
        assert "sampling_params" in normalized
        assert "n" not in normalized["sampling_params"]  # sampling_params n excluded


class TestCacheKey:
    """Test cache key generation."""

    def test_same_request_same_key(self):
        """Same request should generate same cache key."""
        request = {
            "text": "Hello",
            "model": "test-model",
            "sampling_params": {"temperature": 0.8, "max_new_tokens": 100}
        }

        key1 = generate_cache_key(request)
        key2 = generate_cache_key(request)

        assert key1 == key2

    def test_different_n_same_key(self):
        """Different n values should generate same cache key."""
        request1 = {
            "text": "Hello",
            "model": "test-model",
            "sampling_params": {"temperature": 0.8, "n": 1}
        }
        request2 = {
            "text": "Hello",
            "model": "test-model",
            "sampling_params": {"temperature": 0.8, "n": 5}
        }

        key1 = generate_cache_key(request1)
        key2 = generate_cache_key(request2)

        assert key1 == key2

    def test_different_temperature_different_key(self):
        """Different temperature should generate different cache key."""
        request1 = {
            "text": "Hello",
            "model": "test-model",
            "sampling_params": {"temperature": 0.8}
        }
        request2 = {
            "text": "Hello",
            "model": "test-model",
            "sampling_params": {"temperature": 0.9}
        }

        key1 = generate_cache_key(request1)
        key2 = generate_cache_key(request2)

        assert key1 != key2

    def test_different_text_different_key(self):
        """Different input text should generate different cache key."""
        request1 = {
            "text": "Hello",
            "model": "test-model",
            "sampling_params": {"temperature": 0.8}
        }
        request2 = {
            "text": "Hi",
            "model": "test-model",
            "sampling_params": {"temperature": 0.8}
        }

        key1 = generate_cache_key(request1)
        key2 = generate_cache_key(request2)

        assert key1 != key2

    def test_key_is_deterministic(self):
        """Cache key should be deterministic across calls."""
        request = {
            "text": "Test",
            "model": "test-model",
            "sampling_params": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 100
            }
        }

        keys = [generate_cache_key(request) for _ in range(10)]

        # All keys should be identical
        assert len(set(keys)) == 1

    def test_key_is_hex_string(self):
        """Cache key should be a valid hex string."""
        request = {"text": "Test", "model": "test-model"}
        key = generate_cache_key(request)

        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 produces 64 hex chars
        # Should be valid hex
        int(key, 16)

    def test_different_model_different_key(self):
        """Different model should generate different cache key."""
        request1 = {
            "text": "Hello",
            "model": "gpt-4",
            "sampling_params": {"temperature": 0.8}
        }
        request2 = {
            "text": "Hello",
            "model": "gpt-3.5-turbo",
            "sampling_params": {"temperature": 0.8}
        }

        key1 = generate_cache_key(request1)
        key2 = generate_cache_key(request2)

        assert key1 != key2

    def test_same_model_same_key(self):
        """Same model with same parameters should generate same cache key."""
        request1 = {
            "text": "Hello",
            "model": "gpt-4",
            "sampling_params": {"temperature": 0.8}
        }
        request2 = {
            "text": "Hello",
            "model": "gpt-4",
            "sampling_params": {"temperature": 0.8}
        }

        key1 = generate_cache_key(request1)
        key2 = generate_cache_key(request2)

        assert key1 == key2

    def test_different_seed_different_key(self):
        """Different seed should generate different cache key."""
        request1 = {
            "text": "Hello",
            "model": "gpt-4",
            "seed": 42,
            "sampling_params": {"temperature": 0.8}
        }
        request2 = {
            "text": "Hello",
            "model": "gpt-4",
            "seed": 123,
            "sampling_params": {"temperature": 0.8}
        }

        key1 = generate_cache_key(request1)
        key2 = generate_cache_key(request2)

        assert key1 != key2

    def test_different_logprobs_different_key(self):
        """Different logprobs setting should generate different cache key."""
        request1 = {
            "text": "Hello",
            "model": "test-model",
            "logprobs": True,
            "sampling_params": {"temperature": 0.8}
        }
        request2 = {
            "text": "Hello",
            "model": "test-model",
            "logprobs": False,
            "sampling_params": {"temperature": 0.8}
        }

        key1 = generate_cache_key(request1)
        key2 = generate_cache_key(request2)

        assert key1 != key2

    def test_different_response_format_different_key(self):
        """Different response_format should generate different cache key."""
        request1 = {
            "text": "Hello",
            "model": "test-model",
            "response_format": {"type": "json_object"},
            "sampling_params": {"temperature": 0.8}
        }
        request2 = {
            "text": "Hello",
            "model": "test-model",
            "response_format": {"type": "text"},
            "sampling_params": {"temperature": 0.8}
        }

        key1 = generate_cache_key(request1)
        key2 = generate_cache_key(request2)

        assert key1 != key2

    def test_different_tools_different_key(self):
        """Different tools should generate different cache key."""
        request1 = {
            "text": "Hello",
            "model": "test-model",
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "sampling_params": {"temperature": 0.8}
        }
        request2 = {
            "text": "Hello",
            "model": "test-model",
            "tools": [{"type": "function", "function": {"name": "get_time"}}],
            "sampling_params": {"temperature": 0.8}
        }

        key1 = generate_cache_key(request1)
        key2 = generate_cache_key(request2)

        assert key1 != key2

    def test_all_fields_included_except_n(self):
        """Verify that ALL fields are included in cache key except n."""
        request_with_many_fields = {
            "text": "Hello",
            "model": "gpt-4",
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 42,
            "logprobs": True,
            "top_logprobs": 5,
            "stream": False,
            "user": "user123",
            "response_format": {"type": "json_object"},
            "custom_field": "custom_value",
            "n": 5  # This should be excluded
        }

        normalized = normalize_request(request_with_many_fields)

        # All fields should be present except 'n'
        assert "text" in normalized
        assert "model" in normalized
        assert "temperature" in normalized
        assert "top_p" in normalized
        assert "seed" in normalized
        assert "logprobs" in normalized
        assert "top_logprobs" in normalized
        assert "stream" in normalized
        assert "user" in normalized
        assert "response_format" in normalized
        assert "custom_field" in normalized
        assert "n" not in normalized  # This should be excluded

    def test_extra_body_with_lora_path(self):
        """Different LoRA paths should generate different cache keys."""
        request_lora1 = {
            "text": "Hello",
            "model": "gpt-4",
            "temperature": 0.7,
            "extra_body": {"lora_path": "/path/to/lora1"}
        }
        request_lora2 = {
            "text": "Hello",
            "model": "gpt-4",
            "temperature": 0.7,
            "extra_body": {"lora_path": "/path/to/lora2"}
        }

        key1 = generate_cache_key(request_lora1)
        key2 = generate_cache_key(request_lora2)

        # Different LoRA paths should have different keys
        assert key1 != key2

    def test_extra_body_same_lora_different_n(self):
        """Same LoRA with different n should share cache."""
        request_n1 = {
            "text": "Hello",
            "model": "gpt-4",
            "temperature": 0.7,
            "extra_body": {"lora_path": "/path/to/lora1"},
            "n": 1
        }
        request_n5 = {
            "text": "Hello",
            "model": "gpt-4",
            "temperature": 0.7,
            "extra_body": {"lora_path": "/path/to/lora1"},
            "n": 5
        }

        key1 = generate_cache_key(request_n1)
        key2 = generate_cache_key(request_n5)

        # Same LoRA, different n should have same key
        assert key1 == key2


class TestNParameterExtraction:
    """Test extraction of n parameter."""

    def test_extract_from_sampling_params(self):
        """Extract n from sampling_params."""
        request = {
            "text": "Test",
            "model": "test-model",
            "sampling_params": {"n": 5}
        }
        assert extract_n_parameter(request) == 5

    def test_extract_from_top_level(self):
        """Extract n from top level (OpenAI format)."""
        request = {
            "text": "Test",
            "model": "test-model",
            "n": 3
        }
        assert extract_n_parameter(request) == 3

    def test_default_to_one(self):
        """Default to 1 if n not present."""
        request = {"text": "Test", "model": "test-model"}
        assert extract_n_parameter(request) == 1

    def test_sampling_params_takes_precedence(self):
        """sampling_params.n takes precedence over top-level n."""
        request = {
            "text": "Test",
            "model": "test-model",
            "n": 3,
            "sampling_params": {"n": 5}
        }
        assert extract_n_parameter(request) == 5
