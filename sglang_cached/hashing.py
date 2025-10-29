"""
Cache key generation and request normalization.

This module handles the creation of stable, deterministic cache keys from SGLang requests.
The key insight is that the `n` parameter (number of completions) should NOT be part of
the cache key, since we want to reuse cached responses regardless of how many completions
are requested.
"""

import hashlib
import json
from typing import Any, Dict


def normalize_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a request to canonical form for hashing.

    Extracts the essential parts of a request that should be part of the cache key:
    - Input (text, input_ids, or messages)
    - All sampling parameters EXCEPT `n`

    Args:
        request_data: The raw request dictionary

    Returns:
        Normalized request dictionary with sorted keys
    """
    normalized = {}

    # Extract input (multiple possible formats)
    if "text" in request_data:
        normalized["input"] = request_data["text"]
    elif "input_ids" in request_data:
        normalized["input"] = request_data["input_ids"]
    elif "messages" in request_data:
        normalized["input"] = request_data["messages"]
    elif "prompt" in request_data:  # OpenAI-compatible format
        normalized["input"] = request_data["prompt"]

    # Extract sampling params (excluding 'n')
    sampling_params = request_data.get("sampling_params", {})
    if isinstance(sampling_params, dict):
        # Create a copy without 'n'
        params_for_cache = {k: v for k, v in sampling_params.items() if k != "n"}
        if params_for_cache:
            normalized["sampling_params"] = params_for_cache

    # For OpenAI-compatible API, extract params from top level
    for param in ["temperature", "top_p", "top_k", "max_tokens", "max_new_tokens",
                  "stop", "frequency_penalty", "presence_penalty", "repetition_penalty"]:
        if param in request_data and param not in normalized.get("sampling_params", {}):
            if "sampling_params" not in normalized:
                normalized["sampling_params"] = {}
            normalized["sampling_params"][param] = request_data[param]

    return normalized


def generate_cache_key(request_data: Dict[str, Any]) -> str:
    """
    Generate a deterministic cache key from a request.

    The cache key is a SHA256 hash of the normalized request in JSON format.
    All requests with the same input and sampling parameters (except `n`) will
    produce the same cache key.

    Args:
        request_data: The request dictionary

    Returns:
        A hex string representing the SHA256 hash of the normalized request
    """
    normalized = normalize_request(request_data)

    # Convert to JSON with sorted keys for determinism
    json_str = json.dumps(normalized, sort_keys=True, separators=(',', ':'))

    # Generate SHA256 hash
    hash_obj = hashlib.sha256(json_str.encode('utf-8'))
    return hash_obj.hexdigest()


def extract_n_parameter(request_data: Dict[str, Any]) -> int:
    """
    Extract the `n` parameter from a request.

    Args:
        request_data: The request dictionary

    Returns:
        The value of `n` (defaults to 1 if not present)
    """
    # Check in sampling_params first
    sampling_params = request_data.get("sampling_params", {})
    if isinstance(sampling_params, dict) and "n" in sampling_params:
        return sampling_params["n"]

    # Check at top level (OpenAI-compatible API)
    if "n" in request_data:
        return request_data["n"]

    return 1  # Default value
