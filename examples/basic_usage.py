#!/usr/bin/env python3
"""
Basic usage example for sglang-cached.

This example demonstrates:
1. Creating a cached server wrapper
2. Making requests with caching
3. Checking cache statistics
4. Using the n parameter intelligently
"""

from sglang_cached import CachedSGLangServer


def main():
    # Create a cached server wrapper
    # Make sure an SGLang server is running at http://127.0.0.1:30000
    server = CachedSGLangServer(
        sglang_url="http://127.0.0.1:30000",
        cache_dir="~/.sglang_cache",
        verbose=True  # Print cache hit statistics
    )

    print("="*60)
    print("Example 1: Basic Caching")
    print("="*60)

    # First request - will be a cache miss
    request = {
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 10
        }
    }

    print("\n--- First request (cache miss) ---")
    response1 = server.generate(request)
    print(f"Response: {response1['text'][:100]}")

    # Second request - should be a cache hit
    print("\n--- Second request (cache hit) ---")
    response2 = server.generate(request)
    print(f"Response: {response2['text'][:100]}")
    print(f"Responses match: {response1['text'] == response2['text']}")

    print("\n" + "="*60)
    print("Example 2: Smart n Parameter Handling")
    print("="*60)

    request = {
        "text": "Write a creative story opening:",
        "sampling_params": {
            "temperature": 0.9,
            "max_new_tokens": 30
        }
    }

    # Generate 1 completion
    print("\n--- Generate n=1 completion ---")
    request["sampling_params"]["n"] = 1
    response1 = server.generate(request)
    print(f"Got: {1 if isinstance(response1, dict) else len(response1)} completion(s)")

    # Generate 3 completions (will reuse the 1 cached + generate 2 more)
    print("\n--- Generate n=3 completions (reuses 1 cached) ---")
    request["sampling_params"]["n"] = 3
    response3 = server.generate(request)
    print(f"Got: {len(response3)} completions")
    print(f"First completion matches cached: {response3[0]['text'] == response1['text']}")

    # Generate 5 completions (will reuse 3 cached + generate 2 more)
    print("\n--- Generate n=5 completions (reuses 3 cached) ---")
    request["sampling_params"]["n"] = 5
    response5 = server.generate(request)
    print(f"Got: {len(response5)} completions")

    print("\n" + "="*60)
    print("Example 3: Cache Statistics")
    print("="*60)

    stats = server.get_cache_stats()
    print(f"\nCache Statistics:")
    print(f"  Unique cache keys: {stats['num_keys']}")
    print(f"  Total cached responses: {stats['total_responses']}")
    print(f"  Cache hits: {stats['hits']}")
    print(f"  Cache misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Cache file: {stats['cache_file']}")

    print("\n" + "="*60)
    print("Example 4: Different Parameters = Different Cache")
    print("="*60)

    base_request = {
        "text": "Explain quantum computing",
        "sampling_params": {
            "max_new_tokens": 20,
            "n": 1
        }
    }

    # Low temperature
    print("\n--- Low temperature (0.1) ---")
    request_low = base_request.copy()
    request_low["sampling_params"] = {**base_request["sampling_params"], "temperature": 0.1}
    response_low = server.generate(request_low)
    print(f"Response: {response_low['text'][:80]}...")

    # High temperature
    print("\n--- High temperature (1.5) ---")
    request_high = base_request.copy()
    request_high["sampling_params"] = {**base_request["sampling_params"], "temperature": 1.5}
    response_high = server.generate(request_high)
    print(f"Response: {response_high['text'][:80]}...")

    print("\nDifferent responses (as expected):", response_low['text'] != response_high['text'])

    # Final stats
    print("\n" + "="*60)
    print("Final Cache Statistics")
    print("="*60)
    stats = server.get_cache_stats()
    print(f"  Total cache keys: {stats['num_keys']}")
    print(f"  Total responses: {stats['total_responses']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")

    # Clean shutdown
    server.shutdown()
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
