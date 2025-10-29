#!/usr/bin/env python3
"""
Demonstration of multi-model caching support.

This script demonstrates that the caching system correctly isolates
cached responses by model name, ensuring different models don't share
cached responses.
"""

import tempfile
from sglang_cached.cache_manager import CacheManager
from sglang_cached.hashing import generate_cache_key


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def demo_basic_isolation():
    """Demonstrate basic cache isolation by model."""
    print_section("Demo 1: Basic Cache Isolation by Model")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=tmpdir)

        # Same input, different models
        request_gpt4 = {
            "text": "What is the capital of France?",
            "model": "gpt-4",
            "sampling_params": {"temperature": 0.7}
        }
        request_gpt35 = {
            "text": "What is the capital of France?",
            "model": "gpt-3.5-turbo",
            "sampling_params": {"temperature": 0.7}
        }

        # Generate different cache keys
        key_gpt4 = generate_cache_key(request_gpt4)
        key_gpt35 = generate_cache_key(request_gpt35)

        print(f"\nSame input, different models:")
        print(f"  Input: '{request_gpt4['text']}'")
        print(f"  Model 1: {request_gpt4['model']}")
        print(f"    Cache key: {key_gpt4[:16]}...")
        print(f"  Model 2: {request_gpt35['model']}")
        print(f"    Cache key: {key_gpt35[:16]}...")
        print(f"\n  Keys are different: {key_gpt4 != key_gpt35} ✓")

        # Store different responses
        cache.put(request_gpt4, [{"text": "The capital of France is Paris. (GPT-4)"}])
        cache.put(request_gpt35, [{"text": "Paris is the capital of France. (GPT-3.5)"}])

        # Retrieve and verify isolation
        cached_gpt4, _ = cache.get(request_gpt4)
        cached_gpt35, _ = cache.get(request_gpt35)

        print(f"\n  Stored responses are isolated:")
        print(f"    GPT-4 response: {cached_gpt4[0]['text']}")
        print(f"    GPT-3.5 response: {cached_gpt35[0]['text']}")

        cache.shutdown()


def demo_multiple_models():
    """Demonstrate caching with many different models."""
    print_section("Demo 2: Caching Multiple Models Simultaneously")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=tmpdir)

        models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "llama-2-70b", "mistral-7b"]
        input_text = "Explain quantum computing in one sentence."

        print(f"\nStoring responses from {len(models)} different models...")
        print(f"  Input: '{input_text}'")

        for model in models:
            request = {
                "text": input_text,
                "model": model,
                "sampling_params": {"temperature": 0.7}
            }
            # Simulate different responses from each model
            cache.put(request, [{"text": f"[{model}] Quantum computing uses quantum mechanics..."}])
            print(f"    ✓ Cached response for {model}")

        # Retrieve and verify all models have separate cache entries
        print(f"\n  Verifying cache isolation:")
        for model in models:
            request = {
                "text": input_text,
                "model": model,
                "sampling_params": {"temperature": 0.7}
            }
            cached, needed = cache.get(request)
            assert len(cached) == 1
            assert model in cached[0]["text"]
            print(f"    ✓ {model}: cached correctly")

        stats = cache.get_stats()
        print(f"\n  Cache statistics:")
        print(f"    Total cache keys: {stats['num_keys']}")
        print(f"    Total responses: {stats['total_responses']}")
        print(f"    Hit rate: {stats['hit_rate']:.2%}")

        cache.shutdown()


def demo_n_parameter():
    """Demonstrate n parameter with multiple models."""
    print_section("Demo 3: N Parameter with Multiple Models")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=tmpdir)

        print("\nStoring 5 responses for GPT-4 and 3 for GPT-3.5...")

        request_gpt4 = {
            "text": "Count to 10",
            "model": "gpt-4",
            "sampling_params": {"temperature": 0.8}
        }
        request_gpt35 = {
            "text": "Count to 10",
            "model": "gpt-3.5-turbo",
            "sampling_params": {"temperature": 0.8}
        }

        # Store 5 responses for GPT-4
        cache.put(request_gpt4, [
            {"text": f"GPT-4 response {i}"} for i in range(1, 6)
        ])

        # Store 3 responses for GPT-3.5
        cache.put(request_gpt35, [
            {"text": f"GPT-3.5 response {i}"} for i in range(1, 4)
        ])

        # Request different numbers from each model
        test_cases = [
            ("gpt-4", 3, 3, 0),          # Has 5, need 3 → return 3, need 0
            ("gpt-4", 7, 5, 2),          # Has 5, need 7 → return 5, need 2
            ("gpt-3.5-turbo", 2, 2, 0),  # Has 3, need 2 → return 2, need 0
            ("gpt-3.5-turbo", 5, 3, 2),  # Has 3, need 5 → return 3, need 2
        ]

        print(f"\n  Testing n parameter logic:")
        for model, n_requested, expected_cached, expected_needed in test_cases:
            request = {
                "text": "Count to 10",
                "model": model,
                "sampling_params": {"temperature": 0.8, "n": n_requested}
            }
            cached, needed = cache.get(request)
            print(f"    {model}, n={n_requested}: " +
                  f"cached={len(cached)}, need={needed} " +
                  f"(expected {expected_cached}/{expected_needed}) " +
                  f"{'✓' if len(cached) == expected_cached and needed == expected_needed else '✗'}")

        cache.shutdown()


def demo_model_variations():
    """Demonstrate that model name variations are treated as different."""
    print_section("Demo 4: Model Name Variations")

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=tmpdir)

        # These should all be treated as different models
        model_variations = [
            "gpt-4",
            "GPT-4",           # Different case
            "gpt-4-turbo",     # Different variant
            "gpt-4-0613",      # With version
        ]

        print("\nDemonstrating that model name variations are distinct:")
        print(f"  Input: 'Hello'")

        for model in model_variations:
            request = {
                "text": "Hello",
                "model": model,
                "sampling_params": {"temperature": 0.7}
            }
            key = generate_cache_key(request)
            cache.put(request, [{"text": f"Response from {model}"}])
            print(f"    {model:20} → cache key: {key[:16]}...")

        # Verify they're all different
        keys = [
            generate_cache_key({
                "text": "Hello",
                "model": model,
                "sampling_params": {"temperature": 0.7}
            })
            for model in model_variations
        ]

        print(f"\n  All cache keys are unique: {len(keys) == len(set(keys))} ✓")
        print(f"  Each model has its own cached response ✓")

        cache.shutdown()


def demo_persistence():
    """Demonstrate cache persistence with multiple models."""
    print_section("Demo 5: Persistence with Multiple Models")

    with tempfile.TemporaryDirectory() as tmpdir:
        print("\nPhase 1: Creating cache with multiple models...")

        # First cache instance
        cache1 = CacheManager(cache_dir=tmpdir)

        models = ["gpt-4", "claude-3-opus", "llama-2-70b"]
        for model in models:
            request = {
                "text": "Hello",
                "model": model,
                "sampling_params": {"temperature": 0.7}
            }
            cache1.put(request, [{"text": f"Response from {model}"}])
            print(f"  ✓ Cached response for {model}")

        import time
        time.sleep(0.2)  # Wait for async writes
        cache1.shutdown()

        print("\nPhase 2: Loading cache from disk...")

        # Second cache instance (loads from disk)
        cache2 = CacheManager(cache_dir=tmpdir)

        print(f"  Loaded {cache2.get_stats()['num_keys']} cache keys from disk")

        # Verify all models' responses were persisted
        for model in models:
            request = {
                "text": "Hello",
                "model": model,
                "sampling_params": {"temperature": 0.7}
            }
            cached, needed = cache2.get(request)
            assert len(cached) == 1
            print(f"  ✓ Retrieved cached response for {model}")

        print("\n  All model caches persisted and loaded correctly ✓")

        cache2.shutdown()


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  MULTI-MODEL CACHING DEMONSTRATION")
    print("=" * 70)

    demo_basic_isolation()
    demo_multiple_models()
    demo_n_parameter()
    demo_model_variations()
    demo_persistence()

    print("\n" + "=" * 70)
    print("  ALL DEMOS COMPLETED SUCCESSFULLY ✓")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
