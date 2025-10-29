"""
Comprehensive HTTP server tests.

Tests both SGLang native and OpenAI-compatible APIs.
Requires wrapper server running at http://127.0.0.1:30001
"""

import pytest
import requests


@pytest.fixture(scope="session")
def wrapper_url():
    """Wrapper server URL."""
    return "http://127.0.0.1:30001"


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
    """Clear cache before and after each test."""
    if not is_wrapper_running:
        pytest.skip("Wrapper server not running")

    requests.post(f"{wrapper_url}/cache/clear")
    yield
    requests.post(f"{wrapper_url}/cache/clear")


class TestOpenAICompletionsAPI:
    """Tests for OpenAI-compatible /v1/completions endpoint."""

    def test_basic_completion(self, wrapper_url, clear_cache):
        """Test basic OpenAI completion request."""
        request = {
            "model": "test-model",
            "prompt": "The capital of France is",
            "max_tokens": 10,
            "temperature": 0.0
        }

        response = requests.post(f"{wrapper_url}/v1/completions", json=request)
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "text_completion"
        assert data["model"] == "test-model"
        assert "choices" in data
        assert len(data["choices"]) == 1
        assert "text" in data["choices"][0]
        assert len(data["choices"][0]["text"]) > 0

    def test_completion_with_n_parameter(self, wrapper_url, clear_cache):
        """Test OpenAI completion with n > 1."""
        request = {
            "model": "test-model",
            "prompt": "Once upon a time",
            "max_tokens": 20,
            "temperature": 0.9,
            "n": 3
        }

        response = requests.post(f"{wrapper_url}/v1/completions", json=request)
        assert response.status_code == 200

        data = response.json()
        assert len(data["choices"]) == 3
        assert all("text" in choice for choice in data["choices"])
        assert all("index" in choice for choice in data["choices"])

    def test_completion_caching(self, wrapper_url, clear_cache):
        """Test that OpenAI completions are cached correctly."""
        request = {
            "model": "test-model",
            "prompt": "Hello world",
            "max_tokens": 5,
            "temperature": 0.0
        }

        # First request
        response1 = requests.post(f"{wrapper_url}/v1/completions", json=request)
        assert response1.status_code == 200
        data1 = response1.json()

        # Second request should hit cache
        stats_before = requests.get(f"{wrapper_url}/cache/stats").json()
        response2 = requests.post(f"{wrapper_url}/v1/completions", json=request)
        assert response2.status_code == 200
        data2 = response2.json()
        stats_after = requests.get(f"{wrapper_url}/cache/stats").json()

        # Responses should be identical
        assert data1["choices"][0]["text"] == data2["choices"][0]["text"]
        # Cache hits should increase
        assert stats_after["hits"] > stats_before["hits"]


class TestOpenAIChatCompletionsAPI:
    """Tests for OpenAI-compatible /v1/chat/completions endpoint."""

    def test_basic_chat_completion(self, wrapper_url, clear_cache):
        """Test basic OpenAI chat completion request."""
        request = {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "What is 2+2?"}
            ],
            "max_tokens": 10,
            "temperature": 0.0
        }

        response = requests.post(f"{wrapper_url}/v1/chat/completions", json=request)
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "test-model"
        assert "choices" in data
        assert len(data["choices"]) == 1
        assert "message" in data["choices"][0]
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "content" in data["choices"][0]["message"]

    def test_chat_with_multiple_messages(self, wrapper_url, clear_cache):
        """Test chat completion with conversation history."""
        request = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ],
            "max_tokens": 20,
            "temperature": 0.0
        }

        response = requests.post(f"{wrapper_url}/v1/chat/completions", json=request)
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_chat_with_n_parameter(self, wrapper_url, clear_cache):
        """Test chat completion with n > 1."""
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Tell me a story"}],
            "max_tokens": 30,
            "temperature": 0.8,
            "n": 2
        }

        response = requests.post(f"{wrapper_url}/v1/chat/completions", json=request)
        assert response.status_code == 200

        data = response.json()
        assert len(data["choices"]) == 2
        assert all("message" in choice for choice in data["choices"])
        assert all(choice["message"]["role"] == "assistant" for choice in data["choices"])

    def test_chat_caching(self, wrapper_url, clear_cache):
        """Test that chat completions are cached correctly."""
        request = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "What is Python?"}],
            "max_tokens": 15,
            "temperature": 0.0
        }

        # First request
        response1 = requests.post(f"{wrapper_url}/v1/chat/completions", json=request)
        assert response1.status_code == 200
        data1 = response1.json()

        # Second request should hit cache
        stats_before = requests.get(f"{wrapper_url}/cache/stats").json()
        response2 = requests.post(f"{wrapper_url}/v1/chat/completions", json=request)
        assert response2.status_code == 200
        data2 = response2.json()
        stats_after = requests.get(f"{wrapper_url}/cache/stats").json()

        # Responses should be identical
        assert data1["choices"][0]["message"]["content"] == data2["choices"][0]["message"]["content"]
        # Cache hits should increase
        assert stats_after["hits"] > stats_before["hits"]


class TestCrossAPICompatibility:
    """Test that caching works correctly across different API formats."""

    def test_same_request_different_apis(self, wrapper_url, clear_cache):
        """Test that the same logical request via different APIs uses the same cache."""
        # Make request via SGLang API
        sglang_request = {
            "text": "Hello",
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 5
            }
        }
        response1 = requests.post(f"{wrapper_url}/generate", json=sglang_request)
        assert response1.status_code == 200
        data1 = response1.json()

        # Make equivalent request via OpenAI API
        openai_request = {
            "prompt": "Hello",
            "temperature": 0.0,
            "max_tokens": 5
        }
        stats_before = requests.get(f"{wrapper_url}/cache/stats").json()
        response2 = requests.post(f"{wrapper_url}/v1/completions", json=openai_request)
        assert response2.status_code == 200
        data2 = response2.json()
        stats_after = requests.get(f"{wrapper_url}/cache/stats").json()

        # Should hit cache (same underlying request)
        assert stats_after["hits"] > stats_before["hits"]


class TestErrorHandling:
    """Test error handling in the HTTP server."""

    def test_invalid_json(self, wrapper_url, is_wrapper_running):
        """Test handling of invalid JSON."""
        if not is_wrapper_running:
            pytest.skip("Wrapper server not running")

        response = requests.post(
            f"{wrapper_url}/generate",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        # Should return an error (400 or 422)
        assert response.status_code in [400, 422]

    def test_missing_required_fields(self, wrapper_url, clear_cache):
        """Test handling of missing required fields."""
        # Request without text/prompt
        request = {
            "sampling_params": {"max_new_tokens": 10}
        }

        response = requests.post(f"{wrapper_url}/generate", json=request)
        # Should still work (might use empty text or fail gracefully)
        # The behavior depends on SGLang's handling
        assert response.status_code in [200, 400, 422, 502]


class TestConcurrency:
    """Test concurrent requests to the server."""

    def test_concurrent_requests(self, wrapper_url, clear_cache):
        """Test handling of concurrent requests."""
        import concurrent.futures

        request = {
            "text": "Test concurrent",
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 5}
        }

        def make_request():
            response = requests.post(f"{wrapper_url}/generate", json=request)
            return response.status_code == 200

        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        assert all(results)

        # Should have cache hits
        stats = requests.get(f"{wrapper_url}/cache/stats").json()
        assert stats["hits"] >= 4  # First is miss, rest are hits


class TestParameterVariations:
    """Test various sampling parameter combinations."""

    def test_different_temperatures(self, wrapper_url, clear_cache):
        """Test that different temperatures create different cache entries."""
        base_request = {
            "text": "Test temperature",
            "sampling_params": {"max_new_tokens": 10}
        }

        # Request with temp 0.0
        req1 = base_request.copy()
        req1["sampling_params"] = {**base_request["sampling_params"], "temperature": 0.0}
        requests.post(f"{wrapper_url}/generate", json=req1)

        # Request with temp 1.0
        req2 = base_request.copy()
        req2["sampling_params"] = {**base_request["sampling_params"], "temperature": 1.0}
        requests.post(f"{wrapper_url}/generate", json=req2)

        # Should have 2 different cache entries
        stats = requests.get(f"{wrapper_url}/cache/stats").json()
        assert stats["num_keys"] == 2

    def test_n_parameter_exclusion_from_cache_key(self, wrapper_url, clear_cache):
        """Test that n parameter doesn't affect cache key."""
        base_request = {
            "text": "Test n parameter",
            "sampling_params": {
                "temperature": 0.5,
                "max_new_tokens": 10,
                "n": 1
            }
        }

        # Request with n=1
        requests.post(f"{wrapper_url}/generate", json=base_request)

        # Request with n=2 (should use same cache key)
        req2 = base_request.copy()
        req2["sampling_params"] = base_request["sampling_params"].copy()
        req2["sampling_params"]["n"] = 2
        requests.post(f"{wrapper_url}/generate", json=req2)

        # Should still have only 1 cache key
        stats = requests.get(f"{wrapper_url}/cache/stats").json()
        assert stats["num_keys"] == 1
