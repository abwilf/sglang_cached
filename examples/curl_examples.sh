#!/bin/bash
# Curl examples for testing SGLang server

echo "=========================================="
echo "Example 1: Basic Generation"
echo "=========================================="
curl -X POST http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The capital of France is",
    "sampling_params": {
      "temperature": 0.0,
      "max_new_tokens": 10
    }
  }' | jq '.'

echo -e "\n=========================================="
echo "Example 2: Generate with Temperature"
echo "=========================================="
curl -X POST http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Once upon a time",
    "sampling_params": {
      "temperature": 0.9,
      "max_new_tokens": 30
    }
  }' | jq '.text'

echo -e "\n=========================================="
echo "Example 3: Multiple Completions (n=3)"
echo "=========================================="
curl -X POST http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Write a haiku about coding:",
    "sampling_params": {
      "temperature": 0.8,
      "max_new_tokens": 30,
      "n": 3
    }
  }' | jq '.[] | .text'

echo -e "\n=========================================="
echo "Example 4: Check Server Health"
echo "=========================================="
curl http://127.0.0.1:30000/health

echo -e "\n\n=========================================="
echo "Example 5: Get Model Info"
echo "=========================================="
curl http://127.0.0.1:30000/get_model_info | jq '.'

echo -e "\n=========================================="
echo "Example 6: Cached Tokens Metric"
echo "=========================================="
echo "First request:"
curl -s -X POST http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Explain quantum computing in simple terms",
    "sampling_params": {
      "temperature": 0.0,
      "max_new_tokens": 50
    }
  }' | jq '.meta_info.cached_tokens'

echo "Second request (same prompt - RadixCache should hit):"
curl -s -X POST http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Explain quantum computing in simple terms",
    "sampling_params": {
      "temperature": 0.0,
      "max_new_tokens": 50
    }
  }' | jq '.meta_info.cached_tokens'
