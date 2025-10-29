"""
Cached SGLang server implementation.

This module provides a wrapper server that adds response caching to SGLang.
"""

import requests
import time
from typing import Any, Dict, List, Optional, Union

from .cache_manager import CacheManager
from .hashing import extract_n_parameter


class CachedSGLangServer:
    """
    A caching wrapper for SGLang servers.

    This class connects to an existing SGLang server and adds caching capabilities.
    It intercepts requests, checks the cache, and only forwards to SGLang when necessary.
    """

    def __init__(
        self,
        sglang_url: str = "http://127.0.0.1:30000",
        cache_dir: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the cached server.

        Args:
            sglang_url: URL of the underlying SGLang server
            cache_dir: Directory for cache storage (default: ~/.sglang_cache)
            verbose: Whether to print cache statistics
        """
        self.sglang_url = sglang_url.rstrip('/')
        self.cache = CacheManager(cache_dir)
        self.verbose = verbose

    def generate(self, request_data: Dict[str, Any]) -> Union[Dict, List[Dict]]:
        """
        Generate responses with caching.

        Args:
            request_data: Request dictionary for SGLang

        Returns:
            Response from cache or SGLang (dict if n=1, list if n>1)
        """
        n = extract_n_parameter(request_data)

        # Check cache
        cached_responses, num_needed = self.cache.get(request_data)

        if self.verbose:
            num_cached = len(cached_responses)
            if num_needed == 0:
                print(f"✓ Cache hit: {num_cached}/{n} responses from cache")
            elif num_cached > 0:
                print(f"◐ Partial cache hit: {num_cached}/{n} from cache, generating {num_needed} more")
            else:
                print(f"✗ Cache miss: Generating {num_needed} new responses")

        # If we need more responses, call SGLang
        new_responses = []
        if num_needed > 0:
            # Create modified request for SGLang
            sglang_request = request_data.copy()

            # Update n parameter in the request
            if "sampling_params" in sglang_request:
                # Need to create a copy to avoid modifying original
                sglang_request["sampling_params"] = sglang_request["sampling_params"].copy()
                sglang_request["sampling_params"]["n"] = num_needed
            else:
                sglang_request["sampling_params"] = {"n": num_needed}

            # Call SGLang
            response = requests.post(
                f"{self.sglang_url}/generate",
                json=sglang_request
            )
            response.raise_for_status()
            result = response.json()

            # Handle both dict (n=1) and list (n>1) responses
            if isinstance(result, dict):
                new_responses = [result]
            else:
                new_responses = result

            # Update cache asynchronously with original request (not modified one)
            self.cache.put(request_data, new_responses)

        # Merge cached and new responses
        all_responses = cached_responses + new_responses

        # Return in the same format SGLang would (dict if n=1, list otherwise)
        if n == 1:
            return all_responses[0]
        else:
            return all_responses

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self.cache.get_stats()

    def clear_cache(self):
        """Clear all cached responses."""
        self.cache.clear()
        if self.verbose:
            print("✓ Cache cleared")

    def shutdown(self):
        """Shutdown the cache manager."""
        self.cache.shutdown()
