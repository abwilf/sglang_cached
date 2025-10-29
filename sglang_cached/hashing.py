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

    Includes ALL fields from the request EXCEPT the `n` parameter.
    This ensures that any parameter that affects the response is part of the cache key,
    including model, temperature, seed, logprobs, response_format, tools, etc.

    The `n` parameter is excluded because we want to reuse cached responses regardless
    of how many completions are requested.

    Args:
        request_data: The raw request dictionary

    Returns:
        Normalized request dictionary with sorted keys and `n` removed
    """
    import copy

    # Deep copy to avoid modifying the original
    normalized = copy.deepcopy(request_data)

    # Remove 'n' from top level (OpenAI-compatible API)
    normalized.pop("n", None)

    # Remove 'n' from sampling_params if present
    if "sampling_params" in normalized and isinstance(normalized["sampling_params"], dict):
        normalized["sampling_params"].pop("n", None)
        # Remove empty sampling_params dict if it becomes empty
        if not normalized["sampling_params"]:
            normalized.pop("sampling_params")

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
