"""
sglang-cached: An elegant caching wrapper for SGLang inference.

This package provides transparent response caching for SGLang servers,
dramatically reducing inference time for repeated or similar requests.
"""

__version__ = "0.1.0"

from .cache_manager import CacheManager
from .server import CachedSGLangServer

__all__ = ["CacheManager", "CachedSGLangServer"]
