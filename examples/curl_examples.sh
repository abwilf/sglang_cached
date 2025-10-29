#!/bin/bash
#
# Examples of using SGLang Cached via curl
#
# Prerequisites:
# 1. Start SGLang server: python -m sglang.launch_server --model-path <model> --port 30000
# 2. Start wrapper server: sglang-cached start --sglang-url http://localhost:30000 --port 30001
#

WRAPPER_URL="http://localhost:30001"

echo "================================================"
echo "SGLang Cached - curl Examples"
echo "================================================"

# Health check
echo -e "\n1. Health Check"
curl -s "$WRAPPER_URL/health" | jq .

# SGLang native API - basic request
echo -e "\n2. SGLang Native API - Basic Request"
curl -s -X POST "$WRAPPER_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The capital of France is",
    "sampling_params": {
      "temperature": 0.0,
      "max_new_tokens": 10
    }
  }' | jq .

# SGLang API - with n parameter
echo -e "\n3. SGLang API - Multiple Completions (n=3)"
curl -s -X POST "$WRAPPER_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Once upon a time",
    "sampling_params": {
      "temperature": 0.9,
      "max_new_tokens": 20,
      "n": 3
    }
  }' | jq .

# OpenAI Completions API
echo -e "\n4. OpenAI Completions API"
curl -s -X POST "$WRAPPER_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "prompt": "Write a haiku about coding",
    "max_tokens": 50,
    "temperature": 0.7
  }' | jq .

# OpenAI Chat Completions API
echo -e "\n5. OpenAI Chat Completions API"
curl -s -X POST "$WRAPPER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the meaning of life?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }' | jq .

# OpenAI Chat with multiple completions
echo -e "\n6. OpenAI Chat API - Multiple Completions (n=2)"
curl -s -X POST "$WRAPPER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [
      {"role": "user", "content": "Tell me a short joke"}
    ],
    "max_tokens": 50,
    "temperature": 0.9,
    "n": 2
  }' | jq .

# Cache statistics
echo -e "\n7. Cache Statistics"
curl -s "$WRAPPER_URL/cache/stats" | jq .

# Cache info
echo -e "\n8. Detailed Cache Info"
curl -s "$WRAPPER_URL/cache/info" | jq .

# Clear cache
echo -e "\n9. Clear Cache"
curl -s -X POST "$WRAPPER_URL/cache/clear" | jq .

# Verify cache is cleared
echo -e "\n10. Verify Cache is Empty"
curl -s "$WRAPPER_URL/cache/stats" | jq .

echo -e "\n================================================"
echo "Examples complete!"
echo "================================================"
