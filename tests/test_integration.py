"""
Integration tests with real SGLang server via HTTP.

These tests require:
1. A running SGLang server at http://127.0.0.1:30000
2. A running cached wrapper server at http://127.0.0.1:30001

Start the servers:
  # Terminal 1: SGLang server
  python -m sglang.launch_server --model-path <model> --port 30000

  # Terminal 2: Wrapper server
  sglang-cached start --sglang-url http://localhost:30000 --port 30001 --cache-path /tmp/test_cache
"""

import pytest
import requests
import tempfile
import shutil
import subprocess
import time
import sys


@pytest.fixture(scope="session")
def sglang_url():
    """SGLang server URL."""
    return "http://127.0.0.1:30000"


@pytest.fixture(scope="session")
def wrapper_url():
    """Wrapper server URL."""
    return "http://127.0.0.1:30001"


@pytest.fixture(scope="session")
def is_sglang_running(sglang_url):
    """Check if SGLang server is running."""
    try:
        response = requests.get(f"{sglang_url}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@pytest.fixture(scope="session")
def is_wrapper_running(wrapper_url):
    """Check if wrapper server is running."""
    try:
        response = requests.get(f"{wrapper_url}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@pytest.fixture(scope="function")
def clear_cache(wrapper_url, is_wrapper_running):
    """Clear cache before each test."""
    if not is_wrapper_running:
        pytest.skip("Wrapper server not running")

    # Clear cache before test
    requests.post(f"{wrapper_url}/cache/clear")
    yield
    # Clear cache after test
    requests.post(f"{wrapper_url}/cache/clear")


class TestIntegrationHTTP:
    """Integration tests via HTTP requests."""

    def test_health_check(self, wrapper_url, is_wrapper_running):
        """Test wrapper server health check."""
        if not is_wrapper_running:
            pytest.skip("Wrapper server not running")

        response = requests.get(f"{wrapper_url}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "sglang_url" in data

    def test_basic_generate_request(self, wrapper_url, clear_cache):
        """Test a basic /generate request."""
        request = {
            "text": "The capital of France is",
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 10
            }
        }

        response = requests.post(f"{wrapper_url}/generate", json=request)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, dict)
        assert "text" in data
        assert len(data["text"]) > 0

    def test_cache_hit(self, wrapper_url, clear_cache):
        """Test that second request uses cache."""
        request = {
            "text": "2 + 2 =",
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 5
            }
        }

        # First request (cache miss)
        response1 = requests.post(f"{wrapper_url}/generate", json=request)
        assert response1.status_code == 200
        data1 = response1.json()

        # Check cache stats
        stats1 = requests.get(f"{wrapper_url}/cache/stats").json()
        initial_hits = stats1["hits"]

        # Second request (cache hit)
        response2 = requests.post(f"{wrapper_url}/generate", json=request)
        assert response2.status_code == 200
        data2 = response2.json()

        # Responses should be identical
        assert data1["text"] == data2["text"]

        # Cache hits should increase
        stats2 = requests.get(f"{wrapper_url}/cache/stats").json()
        assert stats2["hits"] > initial_hits

    def test_n_parameter_caching(self, wrapper_url, clear_cache):
        """Test caching with different n values via HTTP."""
        request_base = {
            "text": "Once upon a time",
            "sampling_params": {
                "temperature": 0.9,
                "max_new_tokens": 20,
                "n": 1
            }
        }

        # Request n=1
        response1 = requests.post(f"{wrapper_url}/generate", json=request_base)
        assert response1.status_code == 200
        data1 = response1.json()
        assert isinstance(data1, dict)

        # Request n=2 (should use cached response + generate 1 more)
        request2 = request_base.copy()
        request2["sampling_params"] = request_base["sampling_params"].copy()
        request2["sampling_params"]["n"] = 2

        response2 = requests.post(f"{wrapper_url}/generate", json=request2)
        assert response2.status_code == 200
        data2 = response2.json()

        assert isinstance(data2, list)
        assert len(data2) == 2
        # First response should match the cached one
        assert data2[0]["text"] == data1["text"]

        # Request n=1 again (should use cache)
        response3 = requests.post(f"{wrapper_url}/generate", json=request_base)
        assert response3.status_code == 200
        data3 = response3.json()
        assert data3["text"] == data1["text"]

    def test_different_params_different_cache(self, wrapper_url, clear_cache):
        """Test that different parameters create different cache entries."""
        prompt = "Tell me about"

        # Low temperature
        request_low = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.1,
                "max_new_tokens": 15
            }
        }
        response_low = requests.post(f"{wrapper_url}/generate", json=request_low)
        assert response_low.status_code == 200

        # High temperature
        request_high = {
            "text": prompt,
            "sampling_params": {
                "temperature": 1.5,
                "max_new_tokens": 15
            }
        }
        response_high = requests.post(f"{wrapper_url}/generate", json=request_high)
        assert response_high.status_code == 200

        # Should have 2 cache entries
        stats = requests.get(f"{wrapper_url}/cache/stats").json()
        assert stats["num_keys"] == 2

    def test_cache_stats_endpoint(self, wrapper_url, clear_cache):
        """Test cache statistics endpoint."""
        # Make a request
        request = {
            "text": "Hello world",
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 5}
        }
        requests.post(f"{wrapper_url}/generate", json=request)

        # Get stats
        response = requests.get(f"{wrapper_url}/cache/stats")
        assert response.status_code == 200

        stats = response.json()
        assert "num_keys" in stats
        assert "total_responses" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert stats["num_keys"] >= 1

    def test_cache_clear_endpoint(self, wrapper_url, is_wrapper_running):
        """Test cache clear endpoint."""
        if not is_wrapper_running:
            pytest.skip("Wrapper server not running")

        # Make a request to populate cache
        request = {"text": "Test", "sampling_params": {"max_new_tokens": 5}}
        requests.post(f"{wrapper_url}/generate", json=request)

        # Verify cache has entries
        stats_before = requests.get(f"{wrapper_url}/cache/stats").json()
        assert stats_before["num_keys"] > 0

        # Clear cache
        response = requests.post(f"{wrapper_url}/cache/clear")
        assert response.status_code == 200
        assert response.json()["status"] == "success"

        # Verify cache is empty
        stats_after = requests.get(f"{wrapper_url}/cache/stats").json()
        assert stats_after["num_keys"] == 0

    def test_cache_info_endpoint(self, wrapper_url, is_wrapper_running):
        """Test cache info endpoint."""
        if not is_wrapper_running:
            pytest.skip("Wrapper server not running")

        response = requests.get(f"{wrapper_url}/cache/info")
        assert response.status_code == 200

        info = response.json()
        assert "cache_dir" in info
        assert "num_keys" in info
        assert "total_responses" in info
        assert "hit_rate" in info
        assert "pending_writes" in info
