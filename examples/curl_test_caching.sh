#!/bin/bash
#
# Test caching behavior of SGLang Cached wrapper
#
# Prerequisites:
# 1. Start SGLang server: python -m sglang.launch_server --model-path <model> --port 30000
# 2. Start wrapper server: sglang-cached start --sglang-url http://localhost:30000 --port 30001
#

WRAPPER_URL="http://localhost:30001"

echo "=========================================="
echo "Testing SGLang-Cached Caching Behavior"
echo "Wrapper server: $WRAPPER_URL"
echo "=========================================="

# Clear cache first
echo -e "\n--- Clearing cache ---"
curl -s -X POST "$WRAPPER_URL/cache/clear" | jq .

echo -e "\n--- Request 1: Cache MISS ---"
curl -s -X POST "$WRAPPER_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The capital of France is",
    "sampling_params": {
      "temperature": 0.0,
      "max_new_tokens": 10
    }
  }' | jq '.text'

echo -e "\n--- Request 2: Cache HIT (identical request) ---"
curl -s -X POST "$WRAPPER_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The capital of France is",
    "sampling_params": {
      "temperature": 0.0,
      "max_new_tokens": 10
    }
  }' | jq '.text'

echo -e "\n--- Cache Statistics After 2 Requests ---"
curl -s "$WRAPPER_URL/cache/stats" | jq '{hits, misses, hit_rate, num_keys}'

echo -e "\n--- Request 3: n=3 (should reuse 1 cached, generate 2 more) ---"
curl -s -X POST "$WRAPPER_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The capital of France is",
    "sampling_params": {
      "temperature": 0.0,
      "max_new_tokens": 10,
      "n": 3
    }
  }' | jq 'length'
echo " completions returned"

echo -e "\n--- Request 4: n=2 (should use 2 cached responses) ---"
curl -s -X POST "$WRAPPER_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The capital of France is",
    "sampling_params": {
      "temperature": 0.0,
      "max_new_tokens": 10,
      "n": 2
    }
  }' | jq 'length'
echo " completions returned (all from cache)"

echo -e "\n--- Final Cache Statistics ---"
curl -s "$WRAPPER_URL/cache/stats" | jq '{hits, misses, hit_rate, num_keys, total_responses}'

echo -e "\n--- Testing OpenAI API (should use same cache) ---"
echo "Making OpenAI completion request..."
curl -s -X POST "$WRAPPER_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "temperature": 0.0,
    "max_tokens": 10
  }' | jq '.choices[0].text'

echo -e "\n--- Stats After OpenAI Request (should show cache hit) ---"
curl -s "$WRAPPER_URL/cache/stats" | jq '{hits, misses, hit_rate}'

echo -e "\n=========================================="
echo "Caching test complete!"
echo "=========================================="
