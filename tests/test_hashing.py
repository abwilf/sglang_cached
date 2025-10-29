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
            "sampling_params": {
                "temperature": 0.8,
                "max_new_tokens": 100,
                "n": 3
            }
        }
        normalized = normalize_request(request)

        assert "input" in normalized
        assert normalized["input"] == "Hello world"
        assert "sampling_params" in normalized
        assert "n" not in normalized["sampling_params"]
        assert normalized["sampling_params"]["temperature"] == 0.8

    def test_normalize_with_messages(self):
        """Test normalization with messages (chat format)."""
        request = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "sampling_params": {
                "temperature": 0.7
            }
        }
        normalized = normalize_request(request)

        assert normalized["input"] == [{"role": "user", "content": "Hello"}]

    def test_normalize_excludes_n(self):
        """Test that n parameter is excluded from normalized request."""
        request = {
            "text": "Test",
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


class TestCacheKey:
    """Test cache key generation."""

    def test_same_request_same_key(self):
        """Same request should generate same cache key."""
        request = {
            "text": "Hello",
            "sampling_params": {"temperature": 0.8, "max_new_tokens": 100}
        }

        key1 = generate_cache_key(request)
        key2 = generate_cache_key(request)

        assert key1 == key2

    def test_different_n_same_key(self):
        """Different n values should generate same cache key."""
        request1 = {
            "text": "Hello",
            "sampling_params": {"temperature": 0.8, "n": 1}
        }
        request2 = {
            "text": "Hello",
            "sampling_params": {"temperature": 0.8, "n": 5}
        }

        key1 = generate_cache_key(request1)
        key2 = generate_cache_key(request2)

        assert key1 == key2

    def test_different_temperature_different_key(self):
        """Different temperature should generate different cache key."""
        request1 = {
            "text": "Hello",
            "sampling_params": {"temperature": 0.8}
        }
        request2 = {
            "text": "Hello",
            "sampling_params": {"temperature": 0.9}
        }

        key1 = generate_cache_key(request1)
        key2 = generate_cache_key(request2)

        assert key1 != key2

    def test_different_text_different_key(self):
        """Different input text should generate different cache key."""
        request1 = {
            "text": "Hello",
            "sampling_params": {"temperature": 0.8}
        }
        request2 = {
            "text": "Hi",
            "sampling_params": {"temperature": 0.8}
        }

        key1 = generate_cache_key(request1)
        key2 = generate_cache_key(request2)

        assert key1 != key2

    def test_key_is_deterministic(self):
        """Cache key should be deterministic across calls."""
        request = {
            "text": "Test",
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
        request = {"text": "Test"}
        key = generate_cache_key(request)

        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 produces 64 hex chars
        # Should be valid hex
        int(key, 16)


class TestNParameterExtraction:
    """Test extraction of n parameter."""

    def test_extract_from_sampling_params(self):
        """Extract n from sampling_params."""
        request = {
            "text": "Test",
            "sampling_params": {"n": 5}
        }
        assert extract_n_parameter(request) == 5

    def test_extract_from_top_level(self):
        """Extract n from top level (OpenAI format)."""
        request = {
            "text": "Test",
            "n": 3
        }
        assert extract_n_parameter(request) == 3

    def test_default_to_one(self):
        """Default to 1 if n not present."""
        request = {"text": "Test"}
        assert extract_n_parameter(request) == 1

    def test_sampling_params_takes_precedence(self):
        """sampling_params.n takes precedence over top-level n."""
        request = {
            "text": "Test",
            "n": 3,
            "sampling_params": {"n": 5}
        }
        assert extract_n_parameter(request) == 5
