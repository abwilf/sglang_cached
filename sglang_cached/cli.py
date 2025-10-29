"""
Command-line interface for sglang-cached.

Starts an HTTP server that wraps an existing SGLang server with caching.
"""

import argparse
import sys

import requests

from .server import CachedSGLangServer


def check_sglang_server(url: str) -> bool:
    """Check if SGLang server is accessible."""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def main():
    """Main entry point for sglang-cached CLI."""
    parser = argparse.ArgumentParser(
        description="HTTP wrapper server that adds caching to SGLang",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Start command
    start_parser = subparsers.add_parser(
        "start",
        help="Start the cached wrapper server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    start_parser.add_argument(
        "--sglang-url",
        type=str,
        required=True,
        help="URL of the existing SGLang server (e.g., http://localhost:30000)"
    )
    start_parser.add_argument(
        "--port",
        type=int,
        default=30001,
        help="Port for the wrapper server"
    )
    start_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the wrapper server to"
    )
    start_parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Directory for cache storage (default: ~/.sglang_cache)"
    )
    start_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose logging"
    )

    args = parser.parse_args()

    if args.command != "start":
        parser.print_help()
        sys.exit(1)

    # Check if SGLang server is accessible
    print(f"Checking SGLang server at {args.sglang_url}...")
    if not check_sglang_server(args.sglang_url):
        print(f"ERROR: Cannot connect to SGLang server at {args.sglang_url}")
        print("Please ensure the SGLang server is running first.")
        print("\nExample: python -m sglang.launch_server --model-path <model> --port 30000")
        sys.exit(1)

    print(f"✓ Connected to SGLang server at {args.sglang_url}\n")

    # Create cached server wrapper
    print("Initializing cache wrapper...")
    cached_server = CachedSGLangServer(
        sglang_url=args.sglang_url,
        cache_dir=args.cache_path,
        verbose=not args.quiet
    )

    print("\n" + "=" * 70)
    print("SGLang Cached Wrapper Server")
    print("=" * 70)
    print(f"Wrapper server:  http://{args.host}:{args.port}")
    print(f"SGLang server:   {args.sglang_url}")
    print(f"Cache directory: {cached_server.cache.cache_dir}")
    print("\nAvailable endpoints:")
    print("  POST   /generate                 - SGLang native API")
    print("  POST   /v1/completions            - OpenAI completions API")
    print("  POST   /v1/chat/completions       - OpenAI chat API")
    print("  GET    /health                    - Health check")
    print("  GET    /cache/stats               - Cache statistics")
    print("  POST   /cache/clear               - Clear cache")
    print("  GET    /cache/info                - Detailed cache info")
    print("\nExample usage:")
    print(f'  curl -X POST http://localhost:{args.port}/generate \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"text": "Hello", "sampling_params": {"max_new_tokens": 10}}\'')
    print("\nPress Ctrl+C to shutdown")
    print("=" * 70 + "\n")

    # Run the server (uvicorn handles SIGINT internally)
    try:
        cached_server.run(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
        cached_server.shutdown()
        print("✓ Cache saved and shutdown complete")
    except Exception as e:
        print(f"\nERROR: Failed to start server: {e}")
        cached_server.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
