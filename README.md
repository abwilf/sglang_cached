# SGLang-Cached

An elegant, minimal caching wrapper for [SGLang](https://github.com/sgl-project/sglang) that dramatically speeds up repeated inference requests.

## Features

- **Transparent Caching**: Drop-in wrapper that caches SGLang responses automatically
- **Smart `n` Parameter Handling**: Intelligently reuses cached completions when generating multiple responses
- **Async File Persistence**: In-memory cache with non-blocking disk writes
- **Minimal Code**: Clean, simple implementation with comprehensive tests
- **Zero Configuration**: Works out of the box with sensible defaults

## Installation

```bash
pip install sglang-cached
```

Or install from source:

```bash
git clone https://github.com/yourusername/sglang-cached.git
cd sglang-cached
pip install -e .
```

## Quick Start

### Basic Usage

```python
from sglang_cached import CachedSGLangServer

# Create a cached server wrapper
server = CachedSGLangServer(
    sglang_url="http://127.0.0.1:30000",
    cache_dir="~/.sglang_cache",  # Optional, this is the default
    verbose=True  # Print cache hit statistics
)

# Make requests just like with SGLang
request = {
    "text": "What is the capital of France?",
    "sampling_params": {
        "temperature": 0.8,
        "max_new_tokens": 100
    }
}

# First request: cache miss, calls SGLang
response1 = server.generate(request)

# Second request: cache hit, instant response!
response2 = server.generate(request)

# Check cache statistics
stats = server.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")

# Clean shutdown
server.shutdown()
```

### The `n` Parameter Magic

The `n` parameter controls how many completions to generate. SGLang-Cached handles this intelligently:

```python
# Generate 1 completion
request = {
    "text": "Once upon a time",
    "sampling_params": {"temperature": 0.9, "n": 1, "max_new_tokens": 50}
}
response1 = server.generate(request)  # Cache miss: generates 1

# Generate 3 completions with same parameters
request["sampling_params"]["n"] = 3
response3 = server.generate(request)  # Cache hit: reuses 1, generates 2 more
# Returns list of 3 completions (1 cached + 2 new)

# Generate 5 completions
request["sampling_params"]["n"] = 5
response5 = server.generate(request)  # Cache hit: reuses 3, generates 2 more
# Returns list of 5 completions (3 cached + 2 new)
```

The cache stores multiple completions per unique prompt+parameters combination (excluding `n`), so you can efficiently generate different numbers of completions without re-running inference.

## How It Works

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
- **On-disk**: JSON Lines format for persistence across restarts
- **Async writes**: File updates happen in a background thread, never blocking requests

### Architecture

```
┌─────────────────┐
│  Your Code      │
└────────┬────────┘
         │ request
         ▼
┌─────────────────┐
│ CachedSGLangServer │ ◄── Check in-memory cache
└────────┬────────┘
         │ if needed
         ▼
┌─────────────────┐
│ SGLang Server   │ ◄── Forward partial/full requests
└────────┬────────┘
         │ response
         ▼
┌─────────────────┐
│ Cache Manager   │ ◄── Async disk persistence
└─────────────────┘
```

## CLI Usage

Start an SGLang server with caching:

```bash
# Start a new SGLang server with caching
sglang-cached \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --cache-dir ~/.sglang_cache \
    --port 30000

# Connect to an existing SGLang server
sglang-cached \
    --use-existing-server \
    --sglang-port 30000 \
    --cache-dir ~/.sglang_cache
```

## API Reference

### `CachedSGLangServer`

```python
CachedSGLangServer(
    sglang_url: str = "http://127.0.0.1:30000",
    cache_dir: Optional[str] = None,  # Default: ~/.sglang_cache
    verbose: bool = True
)
```

**Methods:**

- `generate(request_data: Dict) -> Union[Dict, List[Dict]]`: Generate responses with caching
- `get_cache_stats() -> Dict`: Get cache statistics (hits, misses, hit rate, etc.)
- `clear_cache()`: Clear all cached responses
- `shutdown()`: Shutdown cache manager and flush pending writes

### Cache Statistics

```python
stats = server.get_cache_stats()

# Returns:
# {
#     "num_keys": 42,           # Number of unique cache keys
#     "total_responses": 156,   # Total cached responses
#     "hits": 89,               # Cache hits
#     "misses": 34,             # Cache misses
#     "hit_rate": 0.723,        # Hit rate (hits / total)
#     "cache_file": "...",      # Path to cache file
#     "pending_writes": 0       # Queued async writes
# }
```

## Performance

SGLang-Cached adds minimal overhead:

- **Cache hit**: < 1ms (in-memory dict lookup)
- **Cache miss**: < 1ms overhead + SGLang inference time
- **Write latency**: 0ms (async background writes)

For repeated experiments or requests, expect 100-1000x speedups from cache hits!

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

## Configuration

### Cache Directory

Default: `~/.sglang_cache`

```python
# Custom cache directory
server = CachedSGLangServer(cache_dir="/path/to/cache")
```

### Verbosity

```python
# Disable cache statistics printing
server = CachedSGLangServer(verbose=False)
```

### Manual Cache Management

```python
# Clear cache
server.clear_cache()

# Force cache flush (wait for async writes)
server.shutdown()
```

## Development

### Running Tests

```bash
# Unit tests (no SGLang server needed)
pytest tests/test_hashing.py tests/test_cache_manager.py

# Integration tests (requires running SGLang server)
# Start server first:
python -m sglang.launch_server --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 30000

# Then run tests:
pytest tests/test_integration.py

# All tests
pytest tests/ -v
```

### Code Structure

```
sglang_cached/
├── __init__.py           # Package exports
├── hashing.py            # Cache key generation
├── cache_manager.py      # Cache storage & persistence
├── server.py             # Main wrapper class
└── cli.py                # Command-line interface

tests/
├── test_hashing.py       # Hash function tests
├── test_cache_manager.py # Cache logic tests
└── test_integration.py   # End-to-end tests
```

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

- Built on top of [SGLang](https://github.com/sgl-project/sglang)
- Inspired by the need for fast iteration during LLM research

## Citation

If you use SGLang-Cached in your research, please cite:

```bibtex
@software{sglang_cached,
  title = {SGLang-Cached: Response Caching for SGLang},
  author = {SGLang-Cached Contributors},
  year = {2025},
  url = {https://github.com/yourusername/sglang-cached}
}
```

---

Made with \ud83d\udc4d for faster LLM inference.
