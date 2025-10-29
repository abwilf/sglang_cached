"""
HTTP wrapper server that adds caching to SGLang.

This module provides a FastAPI-based HTTP server that proxies requests to an
underlying SGLang server while adding intelligent response caching.
"""

import sys
from typing import Any, Dict, List, Optional, Union

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .cache_manager import CacheManager
from .hashing import extract_n_parameter


def openai_to_sglang(openai_request: Dict[str, Any], is_chat: bool = False) -> Dict[str, Any]:
    """
    Transform OpenAI API request to SGLang format.

    Args:
        openai_request: OpenAI-formatted request
        is_chat: Whether this is a chat completion (vs text completion)

    Returns:
        SGLang-formatted request
    """
    sglang_request = {}

    # Handle input
    if is_chat:
        # Chat completion - use messages
        sglang_request["text"] = openai_request.get("messages", [])
    else:
        # Text completion - use prompt
        sglang_request["text"] = openai_request.get("prompt", "")

    # Map sampling parameters
    sampling_params = {}

    param_mapping = {
        "max_tokens": "max_new_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
        "n": "n",
        "stop": "stop",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
    }

    for openai_param, sglang_param in param_mapping.items():
        if openai_param in openai_request:
            sampling_params[sglang_param] = openai_request[openai_param]

    if sampling_params:
        sglang_request["sampling_params"] = sampling_params

    return sglang_request


def sglang_to_openai(
    sglang_response: Union[Dict, List[Dict]],
    is_chat: bool = False,
    model: str = "sglang"
) -> Dict[str, Any]:
    """
    Transform SGLang response to OpenAI API format.

    Args:
        sglang_response: SGLang response (dict or list of dicts)
        is_chat: Whether this is a chat completion
        model: Model name to include in response

    Returns:
        OpenAI-formatted response
    """
    import time

    # Normalize to list
    if isinstance(sglang_response, dict):
        responses = [sglang_response]
    else:
        responses = sglang_response

    # Build OpenAI-style response
    choices = []
    for idx, resp in enumerate(responses):
        choice = {
            "index": idx,
            "finish_reason": "stop",  # Could be extracted from SGLang response
        }

        if is_chat:
            # Chat completion format
            choice["message"] = {
                "role": "assistant",
                "content": resp.get("text", "")
            }
        else:
            # Text completion format
            choice["text"] = resp.get("text", "")

        choices.append(choice)

    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion" if is_chat else "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
        "usage": {
            "prompt_tokens": 0,  # Could extract from SGLang if available
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }


class CachedSGLangServer:
    """
    FastAPI-based HTTP server that wraps SGLang with caching.

    This server intercepts HTTP requests, checks the cache, and forwards
    to the underlying SGLang server only when necessary.
    """

    def __init__(
        self,
        sglang_url: str,
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
        self.app = FastAPI(title="SGLang Cached Wrapper")

        # Register routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup all FastAPI routes."""

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "sglang_url": self.sglang_url}

        @self.app.post("/generate")
        async def generate(request: Request):
            """
            SGLang native /generate endpoint with caching.

            Forwards requests to underlying SGLang server, using cache when possible.
            """
            request_data = await request.json()
            return self._handle_generate(request_data)

        @self.app.get("/cache/stats")
        async def cache_stats():
            """Get cache statistics."""
            return self.cache.get_stats()

        @self.app.post("/cache/clear")
        async def cache_clear():
            """Clear all cached responses."""
            self.cache.clear()
            return {"status": "success", "message": "Cache cleared"}

        @self.app.get("/cache/info")
        async def cache_info():
            """Get detailed cache information."""
            stats = self.cache.get_stats()
            return {
                "cache_dir": stats["cache_file"],
                "num_keys": stats["num_keys"],
                "total_responses": stats["total_responses"],
                "hits": stats["hits"],
                "misses": stats["misses"],
                "hit_rate": stats["hit_rate"],
                "pending_writes": stats["pending_writes"]
            }

        @self.app.post("/v1/completions")
        async def openai_completions(request: Request):
            """
            OpenAI-compatible /v1/completions endpoint with caching.

            Converts OpenAI format to SGLang, processes with caching, and converts back.
            """
            openai_request = await request.json()

            # Transform to SGLang format
            sglang_request = openai_to_sglang(openai_request, is_chat=False)

            # Process with caching
            sglang_response = self._handle_generate(sglang_request)

            # Transform back to OpenAI format
            model = openai_request.get("model", "sglang")
            openai_response = sglang_to_openai(sglang_response, is_chat=False, model=model)

            return openai_response

        @self.app.post("/v1/chat/completions")
        async def openai_chat_completions(request: Request):
            """
            OpenAI-compatible /v1/chat/completions endpoint with caching.

            Converts OpenAI format to SGLang, processes with caching, and converts back.
            """
            openai_request = await request.json()

            # Transform to SGLang format
            sglang_request = openai_to_sglang(openai_request, is_chat=True)

            # Process with caching
            sglang_response = self._handle_generate(sglang_request)

            # Transform back to OpenAI format
            model = openai_request.get("model", "sglang")
            openai_response = sglang_to_openai(sglang_response, is_chat=True, model=model)

            return openai_response

    def _handle_generate(self, request_data: Dict[str, Any]) -> Union[Dict, List[Dict]]:
        """
        Handle a generate request with caching logic.

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
            cache_status = "hit" if num_needed == 0 else "partial" if num_cached > 0 else "miss"
            print(f"[Cache {cache_status}] Cached: {num_cached}/{n}, Need: {num_needed}")

        # If we need more responses, call SGLang
        new_responses = []
        if num_needed > 0:
            # Create modified request for SGLang
            sglang_request = request_data.copy()

            # Update n parameter in the request
            if "sampling_params" in sglang_request:
                sglang_request["sampling_params"] = sglang_request["sampling_params"].copy()
                sglang_request["sampling_params"]["n"] = num_needed
            else:
                sglang_request["sampling_params"] = {"n": num_needed}

            # Call SGLang
            try:
                response = requests.post(
                    f"{self.sglang_url}/generate",
                    json=sglang_request,
                    timeout=300  # 5 minute timeout for long generations
                )
                response.raise_for_status()
                result = response.json()

                # Handle both dict (n=1) and list (n>1) responses
                if isinstance(result, dict):
                    new_responses = [result]
                else:
                    new_responses = result

                # Update cache asynchronously with original request
                self.cache.put(request_data, new_responses)

            except requests.exceptions.RequestException as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to connect to SGLang server at {self.sglang_url}: {str(e)}"
                )

        # Merge cached and new responses
        all_responses = cached_responses + new_responses

        # Return in the same format SGLang would (dict if n=1, list otherwise)
        if n == 1:
            return all_responses[0]
        else:
            return all_responses

    def run(self, host: str = "0.0.0.0", port: int = 30001):
        """
        Run the FastAPI server.

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        uvicorn.run(self.app, host=host, port=port)

    def shutdown(self):
        """Shutdown the cache manager."""
        self.cache.shutdown()
