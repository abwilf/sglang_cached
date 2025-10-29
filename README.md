# SGLang-Cached

An HTTP wrapper server for [SGLang](https://github.com/sgl-project/sglang) that adds intelligent response caching with support for both SGLang and OpenAI-compatible APIs.

## Features

- **HTTP Proxy Server**: Standalone server that wraps any SGLang instance
- **Dual API Support**: Both SGLang native and OpenAI-compatible endpoints
- **Smart `n` Parameter Handling**: Intelligently reuses cached completions when generating multiple responses
- **Async File Persistence**: In-memory cache with non-blocking disk writes
- **Zero Configuration**: Works out of the box with sensible defaults
- **Language Agnostic**: Use from any language via HTTP (curl, Python requests, Node.js fetch, etc.)

## Installation
```bash
git clone git@github.com:abwilf/sglang_cached.git
cd sglang_cached
pip install -e .
```

## Quick Start

### 1. Start Your SGLang Server

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

### 2. Start the Cached Wrapper Server

```bash
sglang-cached start --sglang-url http://localhost:30000 --port 30001
```

### 3. Make Requests to the cached wrapper server
Want to test quickly? Use these ultra-small models that load in seconds:

#### Terminal 1: Start SGLang Server

```bash
# TinyLlama 1.1B (recommended for testing - ~2GB download)
python -m sglang.launch_server \
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --port 30000

# Or even smaller: Qwen 0.5B (~1GB download)
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --port 30000
```

#### Terminal 2: Start Cached Wrapper

```bash
sglang-cached start \
  --sglang-url http://localhost:30000 \
  --port 30001 \
  --cache-path /tmp/test_cache
```

#### Terminal 3: Test Caching

```bash
# Test 1: Cache miss (will take ~1-2 seconds)
curl -X POST http://localhost:30001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The capital of France is",
    "sampling_params": {"temperature": 0.0, "max_new_tokens": 100}
  }' | jq .

# Test 2: Cache hit (instant! <1ms)
curl -X POST http://localhost:30001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The capital of France is",
    "sampling_params": {"temperature": 0.0, "max_new_tokens": 10}
  }' | jq .

# Test 3: Check cache stats
curl http://localhost:30001/cache/stats | jq .

# Test 4: Test n parameter (reuses 1 cached, generates 2 more)
curl -X POST http://localhost:30001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The capital of France is",
    "sampling_params": {"temperature": 0.0, "max_new_tokens": 10, "n": 3}
  }' | jq 'length'

# Test 5: OpenAI API format
curl -X POST http://localhost:30001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 20
  }' | jq .
```

**Python quick test:**

```python
import requests

base_url = "http://localhost:30001"

# Cache miss
resp1 = requests.post(f"{base_url}/generate", json={
    "text": "2+2=",
    "sampling_params": {"temperature": 0.0, "max_new_tokens": 5}
})
print("First request:", resp1.json()["text"])

# Cache hit (instant!)
resp2 = requests.post(f"{base_url}/generate", json={
    "text": "2+2=",
    "sampling_params": {"temperature": 0.0, "max_new_tokens": 5}
})
print("Second request:", resp2.json()["text"])

# Check stats
stats = requests.get(f"{base_url}/cache/stats").json()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

Look for `[Cache hit]` or `[Cache miss]` messages in Terminal 2 to see caching in action!

## Using with OpenAI Client Libraries

The OpenAI-compatible endpoints work with standard OpenAI client libraries:

**Python:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30001/v1",
    api_key="dummy"  # Not validated, but required by client
)

response = client.chat.completions.create(
    model="llama-2-7b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
print(response.choices[0].message.content)
```

## Available Endpoints

### SGLang Native API

- `POST /generate` - Generate completions (SGLang format)

### OpenAI-Compatible API

- `POST /v1/completions` - Text completions (OpenAI format)
- `POST /v1/chat/completions` - Chat completions (OpenAI format)

### Cache Management

- `GET /cache/stats` - Get cache statistics
- `POST /cache/clear` - Clear all cached responses
- `GET /cache/info` - Detailed cache information
- `GET /health` - Health check

## The `n` Parameter Magic

The `n` parameter controls how many completions to generate. SGLang-Cached handles this intelligently:

```bash
# Request 1: Generate 1 completion (cache miss)
curl -X POST http://localhost:30001/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Once upon a time", "sampling_params": {"temperature": 0.9, "n": 1, "max_new_tokens": 50}}'

# Request 2: Generate 3 completions (reuses 1 cached, generates 2 more)
curl -X POST http://localhost:30001/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Once upon a time", "sampling_params": {"temperature": 0.9, "n": 3, "max_new_tokens": 50}}'

# Request 3: Generate 5 completions (reuses 3 cached, generates 2 more)
curl -X POST http://localhost:30001/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Once upon a time", "sampling_params": {"temperature": 0.9, "n": 5, "max_new_tokens": 50}}'
```

The cache stores multiple completions per unique prompt+parameters combination (excluding `n`), so you can efficiently generate different numbers of completions without re-running inference.

## How It Works

### Architecture

```
┌─────────────────┐
│  Client         │ (any language - curl, Python, Node.js, etc.)
│  (HTTP request) │
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  Cached Wrapper      │ ◄── Check in-memory cache
│  Server (port 30001) │
└────────┬─────────────┘
         │ if cache miss
         ▼
┌──────────────────────┐
│  SGLang Server       │ ◄── Forward request
│  (port 30000)        │
└────────┬─────────────┘
         │ response
         ▼
┌──────────────────────┐
│  Cache Manager       │ ◄── Async disk persistence
└──────────────────────┘
```

### Cache Key Generation

SGLang-Cached generates cache keys from:
- Input text/messages/prompt
- All sampling parameters **except `n`**:
  - `temperature`, `top_p`, `top_k`, `min_p`
  - `frequency_penalty`, `presence_penalty`, `repetition_penalty`
  - `max_new_tokens`, `stop`, etc.

This means two requests with different `n` values but identical other parameters will share the same cache entry.

### Cache Storage

- **In-memory**: Fast dictionary-based cache for instant lookups
- **On-disk**: JSON Lines format for persistence across server restarts
- **Async writes**: File updates happen in a background thread, never blocking requests

## CLI Reference

```bash
sglang-cached start \
  --sglang-url http://localhost:30000  # Required: URL of SGLang server
  --port 30001                          # Optional: Wrapper server port (default: 30001)
  --host 0.0.0.0                        # Optional: Host to bind to (default: 0.0.0.0)
  --cache-path /path/to/cache           # Optional: Cache directory (default: ~/.sglang_cache)
  --quiet                               # Optional: Disable verbose logging
```

## Checking Cache Statistics

```bash
# Get cache stats
curl http://localhost:30001/cache/stats

# Example response:
# {
#   "num_keys": 42,
#   "total_responses": 156,
#   "hits": 89,
#   "misses": 34,
#   "hit_rate": 0.723,
#   "cache_file": "/home/user/.sglang_cache/cache.jsonl",
#   "pending_writes": 0
# }
```

## Performance

SGLang-Cached adds minimal overhead:

- **Cache hit**: < 1ms (in-memory dict lookup)
- **Cache miss**: < 1ms overhead + SGLang inference time
- **Write latency**: 0ms (async background writes)

For repeated experiments or requests, expect **100-1000x speedups** from cache hits!

## Comparison with SGLang's RadixCache

SGLang has a built-in [RadixCache](https://lmsys.org/blog/2024-01-17-sglang/#radixattention-for-automatic-kv-cache-reuse) that caches KV states at the token level. SGLang-Cached complements this by caching **entire responses**:

| Feature | RadixCache | SGLang-Cached |
|---------|-----------|---------------|
| **What it caches** | KV states (internal) | Full responses |
| **Granularity** | Token-level | Response-level |
| **Persistence** | In-memory only | Disk + memory |
| **Use case** | Common prefixes within session | Repeated requests across sessions |
| **Speedup** | 2-5x for prefix reuse | 100-1000x for exact matches |

**Use both together** for maximum performance! RadixCache optimizes within a session, while SGLang-Cached eliminates redundant requests entirely.

## Use Cases

- **Experiment reruns**: Avoid re-generating responses when rerunning notebooks or scripts
- **A/B testing**: Compare prompts without regenerating baselines
- **Development**: Fast iteration during prompt engineering
- **Batch processing**: Dedup requests in large-scale inference jobs
- **Production**: Cache common queries in user-facing applications
- **Multi-language projects**: Access from any programming language via HTTP

## Development

### Running Tests

```bash
# Unit tests (no servers needed)
pytest tests/test_hashing.py tests/test_cache_manager.py

# Integration tests (requires both servers running)
# Terminal 1: Start SGLang server
python -m sglang.launch_server --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 30000

# Terminal 2: Start wrapper server
sglang-cached start --sglang-url http://localhost:30000 --port 30001 --cache-path /tmp/test_cache

# Terminal 3: Run tests
pytest tests/test_integration.py tests/test_http_server.py

# All tests
pytest tests/ -v
```

### Project Structure

```
sglang_cached/
├── __init__.py           # Package exports
├── hashing.py            # Cache key generation
├── cache_manager.py      # Cache storage & persistence
├── server.py             # FastAPI HTTP server
└── cli.py                # Command-line interface

tests/
├── test_hashing.py       # Hash function tests
├── test_cache_manager.py # Cache logic tests
├── test_integration.py   # HTTP integration tests
└── test_http_server.py   # Comprehensive HTTP API tests

examples/
├── curl_examples.sh      # curl usage examples
└── curl_test_caching.sh  # Caching behavior demo
```

## Examples

See the `examples/` directory for:
- `curl_examples.sh` - Comprehensive examples using curl
- `curl_test_caching.sh` - Demonstrates caching behavior

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Acknowledgments

- Built on top of [SGLang](https://github.com/sgl-project/sglang) and [FastAPI](https://fastapi.tiangolo.com/)
- Inspired by the need for fast iteration during LLM research

---

Made for faster LLM inference.
