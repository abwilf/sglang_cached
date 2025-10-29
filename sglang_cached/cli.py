"""
Command-line interface for sglang-cached.
"""

import argparse
import sys
import subprocess
import time
import requests
from pathlib import Path

from .server import CachedSGLangServer


def wait_for_server(url: str, timeout: int = 60) -> bool:
    """Wait for SGLang server to be ready."""
    print(f"Waiting for SGLang server at {url}...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                print(f"✓ SGLang server is ready at {url}")
                return True
        except requests.exceptions.RequestException:
            time.sleep(1)

    return False


def start_sglang_server(args: argparse.Namespace) -> subprocess.Popen:
    """
    Start the underlying SGLang server.

    Returns:
        The subprocess handle
    """
    # Build SGLang command
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model_path,
        "--host", args.sglang_host,
        "--port", str(args.sglang_port)
    ]

    # Add optional arguments
    if args.tp_size:
        cmd.extend(["--tp-size", str(args.tp_size)])
    if args.dtype:
        cmd.extend(["--dtype", args.dtype])
    if args.disable_radix_cache:
        cmd.append("--disable-radix-cache")

    print(f"Starting SGLang server: {' '.join(cmd)}")

    # Start SGLang server
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    return process


def main():
    """Main entry point for sglang-cached CLI."""
    parser = argparse.ArgumentParser(
        description="SGLang with response caching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Cache options
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for cache storage (default: ~/.sglang_cache)"
    )
    parser.add_argument(
        "--no-cache-stats",
        action="store_true",
        help="Disable printing cache statistics"
    )

    # SGLang server options
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model (required)"
    )
    parser.add_argument(
        "--sglang-host",
        type=str,
        default="127.0.0.1",
        help="Host for underlying SGLang server"
    )
    parser.add_argument(
        "--sglang-port",
        type=int,
        default=30000,
        help="Port for underlying SGLang server"
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=None,
        help="Tensor parallelism size"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Data type (auto, half, float16, bfloat16, float, float32)"
    )
    parser.add_argument(
        "--disable-radix-cache",
        action="store_true",
        help="Disable SGLang's radix cache"
    )

    # Server mode options
    parser.add_argument(
        "--use-existing-server",
        action="store_true",
        help="Connect to existing SGLang server instead of starting one"
    )

    args = parser.parse_args()

    # Build SGLang URL
    sglang_url = f"http://{args.sglang_host}:{args.sglang_port}"

    # Start SGLang server if needed
    sglang_process = None
    if not args.use_existing_server:
        try:
            sglang_process = start_sglang_server(args)
            if not wait_for_server(sglang_url, timeout=120):
                print("✗ Failed to start SGLang server")
                if sglang_process:
                    sglang_process.kill()
                sys.exit(1)
        except Exception as e:
            print(f"✗ Error starting SGLang server: {e}")
            if sglang_process:
                sglang_process.kill()
            sys.exit(1)
    else:
        # Check if server is already running
        try:
            response = requests.get(f"{sglang_url}/health", timeout=2)
            if response.status_code != 200:
                print(f"✗ No SGLang server found at {sglang_url}")
                sys.exit(1)
            print(f"✓ Connected to existing SGLang server at {sglang_url}")
        except requests.exceptions.RequestException:
            print(f"✗ Cannot connect to SGLang server at {sglang_url}")
            sys.exit(1)

    # Create cached server wrapper
    print(f"\n✓ Initializing cache (cache_dir={args.cache_dir or '~/.sglang_cache'})")
    cached_server = CachedSGLangServer(
        sglang_url=sglang_url,
        cache_dir=args.cache_dir,
        verbose=not args.no_cache_stats
    )

    print("\n" + "="*60)
    print("SGLang-Cached is ready!")
    print("="*60)
    print(f"SGLang server: {sglang_url}")
    print(f"Cache directory: {cached_server.cache.cache_dir}")
    print("\nUsage:")
    print("  Use the 'sglang_cached' Python package in your code")
    print("  Example:")
    print("    from sglang_cached import CachedSGLangServer")
    print(f"    server = CachedSGLangServer('{sglang_url}')")
    print("    response = server.generate({'text': 'Hello', 'sampling_params': {'max_new_tokens': 10}})")
    print("\nPress Ctrl+C to shutdown")
    print("="*60 + "\n")

    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        cached_server.shutdown()
        if sglang_process:
            print("Stopping SGLang server...")
            sglang_process.terminate()
            sglang_process.wait(timeout=5)
        print("✓ Shutdown complete")


if __name__ == "__main__":
    main()
