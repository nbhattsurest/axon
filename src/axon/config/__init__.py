"""Axon configuration — ignore patterns, language detection, and embedding settings."""

from axon.config.embeddings import (
    EmbeddingConfig,
    EmbeddingProvider,
    get_embedding_config,
    load_embedding_config,
    reset_embedding_config,
)
from axon.config.ignore import DEFAULT_IGNORE_PATTERNS, load_gitignore, should_ignore
from axon.config.languages import SUPPORTED_EXTENSIONS, get_language, is_supported

__all__ = [
    "DEFAULT_IGNORE_PATTERNS",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "SUPPORTED_EXTENSIONS",
    "get_embedding_config",
    "get_language",
    "is_supported",
    "load_embedding_config",
    "load_gitignore",
    "reset_embedding_config",
    "should_ignore",
]
