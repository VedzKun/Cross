"""
Embedding module for semdiff.
"""

from .cache import EmbeddingDiskCache
from .labse import LaBSEEmbedder

__all__ = ["EmbeddingDiskCache", "LaBSEEmbedder"]
