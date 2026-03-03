"""Embedding provider configuration for Axon.

Supports multiple embedding backends:
- FastEmbed (default): Local embedding using fastembed library
- Azure OpenAI via UAIS: Cloud-based embedding using UHG's UAIS authentication

Configuration is done via environment variables:
- AXON_EMBEDDING_PROVIDER: "fastembed" (default) or "azure_openai"
- UAIS_CLIENT_ID: Client ID for UAIS authentication (required for azure_openai)
- UAIS_CLIENT_SECRET: Client secret for UAIS authentication (required for azure_openai)
- AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL (optional, has default)
- AZURE_OPENAI_EMBEDDING_MODEL: Embedding model name (optional, defaults to text-embedding-ada-002)
- AZURE_OPENAI_API_VERSION: API version (optional, defaults to 2024-02-01)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class EmbeddingProvider(Enum):
    """Supported embedding providers."""

    FASTEMBED = "fastembed"
    AZURE_OPENAI = "azure_openai"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for the embedding provider."""

    provider: EmbeddingProvider
    # FastEmbed settings
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    # Azure OpenAI / UAIS settings
    uais_client_id: str | None = None
    uais_client_secret: str | None = None
    azure_openai_endpoint: str = "https://api.uhg.com/api/azureopenai"
    azure_openai_model: str = "text-embedding-ada-002"
    azure_openai_api_version: str = "2024-02-01"
    uais_auth_url: str = "https://api.uhg.com/oauth2/token"
    uais_scope: str = "https://api.uhg.com/.default"


def load_embedding_config() -> EmbeddingConfig:
    """Load embedding configuration from environment variables.

    Returns:
        EmbeddingConfig with settings from environment or defaults.
    """
    provider_str = os.environ.get("AXON_EMBEDDING_PROVIDER", "fastembed").lower()

    try:
        provider = EmbeddingProvider(provider_str)
    except ValueError:
        provider = EmbeddingProvider.FASTEMBED

    return EmbeddingConfig(
        provider=provider,
        fastembed_model=os.environ.get("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        uais_client_id=os.environ.get("UAIS_CLIENT_ID"),
        uais_client_secret=os.environ.get("UAIS_CLIENT_SECRET"),
        azure_openai_endpoint=os.environ.get(
            "AZURE_OPENAI_ENDPOINT", "https://api.uhg.com/api/azureopenai"
        ),
        azure_openai_model=os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        azure_openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        uais_auth_url=os.environ.get("UAIS_AUTH_URL", "https://api.uhg.com/oauth2/token"),
        uais_scope=os.environ.get("UAIS_SCOPE", "https://api.uhg.com/.default"),
    )


# Singleton config instance, lazily loaded
_config: EmbeddingConfig | None = None


def get_embedding_config() -> EmbeddingConfig:
    """Get the current embedding configuration (singleton).

    Returns:
        The loaded embedding configuration.
    """
    global _config
    if _config is None:
        _config = load_embedding_config()
    return _config


def reset_embedding_config() -> None:
    """Reset the configuration singleton (useful for testing)."""
    global _config
    _config = None
