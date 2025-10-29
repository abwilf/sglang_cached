"""
Integration tests with real SGLang server.

These tests require a running SGLang server at http://127.0.0.1:30000
"""

import pytest
import requests
import tempfile
import shutil

from sglang_cached import CachedSGLangServer


@pytest.fixture
def sglang_url():
    """SGLang server URL."""
    return "http://127.0.0.1:30000"


@pytest.fixture
def is_server_running(sglang_url):
    """Check if SGLang server is running."""
    try:
        response = requests.get(f"{sglang_url}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def cached_server(sglang_url, is_server_running):
    """Create a cached server instance with fresh cache for each test."""
    if not is_server_running:
        pytest.skip("SGLang server not running")

    # Create a fresh temp directory for each test
    temp_dir = tempfile.mkdtemp()

    server = CachedSGLangServer(
        sglang_url=sglang_url,
        cache_dir=temp_dir,
        verbose=False
    )
    yield server
    server.shutdown()

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestIntegration:
    """Integration tests with real SGLang server."""

    def test_basic_request(self, cached_server):
        """Test a basic request."""
        request = {
            "text": "The capital of France is",
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 10
            }
        }

        response = cached_server.generate(request)

        assert isinstance(response, dict)
        assert "text" in response
        assert "meta_info" in response
        assert len(response["text"]) > 0

    def test_cache_hit(self, cached_server):
        """Test that second request uses cache."""
        request = {
            "text": "2 + 2 =",
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 5
            }
        }

        # First request (cache miss)
        response1 = cached_server.generate(request)
        stats1 = cached_server.get_cache_stats()

        # Second request (cache hit)
        response2 = cached_server.generate(request)
        stats2 = cached_server.get_cache_stats()

        # Responses should be identical
        assert response1["text"] == response2["text"]

        # Cache hits should increase
        assert stats2["hits"] > stats1["hits"]

    def test_n_parameter_caching(self, cached_server):
        """Test caching with different n values."""
        request_base = {
            "text": "Once upon a time",
            "sampling_params": {
                "temperature": 0.9,
                "max_new_tokens": 20
            }
        }

        # Request n=1
        request1 = {**request_base}
        request1["sampling_params"]["n"] = 1
        response1 = cached_server.generate(request1)

        assert isinstance(response1, dict)

        # Request n=2 (should use cached response + generate 1 more)
        request2 = {**request_base}
        request2["sampling_params"]["n"] = 2
        response2 = cached_server.generate(request2)

        assert isinstance(response2, list)
        assert len(response2) == 2
        # First response should match the cached one
        assert response2[0]["text"] == response1["text"]

        # Request n=1 again (should use cache)
        request3 = {**request_base}
        request3["sampling_params"]["n"] = 1
        response3 = cached_server.generate(request3)

        assert response3["text"] == response1["text"]

    def test_different_params_different_cache(self, cached_server):
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
        response_low = cached_server.generate(request_low)

        # High temperature
        request_high = {
            "text": prompt,
            "sampling_params": {
                "temperature": 1.5,
                "max_new_tokens": 15
            }
        }
        response_high = cached_server.generate(request_high)

        # Should have 2 cache entries
        stats = cached_server.get_cache_stats()
        assert stats["num_keys"] == 2

    def test_cache_persistence(self, sglang_url, temp_cache_dir, is_server_running):
        """Test that cache persists across server restarts."""
        if not is_server_running:
            pytest.skip("SGLang server not running")

        request = {
            "text": "Persistence test:",
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 10
            }
        }

        # Create first server, make request, shutdown
        server1 = CachedSGLangServer(
            sglang_url=sglang_url,
            cache_dir=temp_cache_dir,
            verbose=False
        )
        response1 = server1.generate(request)
        server1.shutdown()

        # Wait for async write
        import time
        time.sleep(1.0)

        # Create second server, should load cache
        server2 = CachedSGLangServer(
            sglang_url=sglang_url,
            cache_dir=temp_cache_dir,
            verbose=False
        )

        # Should have cache hit
        stats_before = server2.get_cache_stats()
        response2 = server2.generate(request)
        stats_after = server2.get_cache_stats()

        assert response1["text"] == response2["text"]
        assert stats_after["hits"] > stats_before["hits"]

        server2.shutdown()
