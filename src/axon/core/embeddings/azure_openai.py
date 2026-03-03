"""Azure OpenAI embedding provider using UAIS (UHG AI Studio) authentication.

This module provides embedding functionality via Azure OpenAI, authenticated
through UHG's UAIS OAuth2 flow using client credentials.

Usage:
    1. Set environment variables:
       - AXON_EMBEDDING_PROVIDER=azure_openai
       - UAIS_CLIENT_ID=<your-client-id>
       - UAIS_CLIENT_SECRET=<your-client-secret>

    2. The provider will automatically obtain and refresh OAuth2 tokens
       when making embedding requests.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from axon.config.embeddings import EmbeddingConfig

logger = logging.getLogger(__name__)

# Token cache with expiry tracking
_token_cache: dict[str, tuple[str, float]] = {}  # (token, expiry_time)
_TOKEN_EXPIRY_BUFFER = 60  # Refresh token 60 seconds before expiry


def _get_uais_token(config: "EmbeddingConfig") -> str:
    """Obtain an OAuth2 access token from UAIS.

    Args:
        config: Embedding configuration with UAIS credentials.

    Returns:
        Access token string.

    Raises:
        ValueError: If client credentials are not configured.
        RuntimeError: If token acquisition fails.
    """
    import httpx

    if not config.uais_client_id or not config.uais_client_secret:
        raise ValueError(
            "UAIS credentials not configured. "
            "Set UAIS_CLIENT_ID and UAIS_CLIENT_SECRET environment variables."
        )

    cache_key = f"{config.uais_client_id}:{config.uais_auth_url}"
    now = time.time()

    # Check cache
    if cache_key in _token_cache:
        token, expiry = _token_cache[cache_key]
        if now < expiry - _TOKEN_EXPIRY_BUFFER:
            return token

    # Request new token
    body = {
        "grant_type": "client_credentials",
        "scope": config.uais_scope,
        "client_id": config.uais_client_id,
        "client_secret": config.uais_client_secret,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(config.uais_auth_url, data=body)
            response.raise_for_status()
            data = response.json()

        token = data["access_token"]
        expires_in = data.get("expires_in", 3600)  # Default to 1 hour
        expiry = now + expires_in

        _token_cache[cache_key] = (token, expiry)
        logger.debug("Obtained new UAIS access token (expires in %ds)", expires_in)
        return token

    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to obtain UAIS token: {e}") from e
    except KeyError as e:
        raise RuntimeError(f"Invalid token response: missing {e}") from e


class AzureOpenAIEmbedder:
    """Embedding provider using Azure OpenAI via UAIS authentication."""

    def __init__(self, config: "EmbeddingConfig") -> None:
        """Initialize the Azure OpenAI embedder.

        Args:
            config: Embedding configuration with Azure OpenAI and UAIS settings.
        """
        self.config = config
        self._client = None

    def _get_client(self):
        """Get or create the OpenAI client with current token."""
        from openai import AzureOpenAI

        token = _get_uais_token(self.config)

        # Always create a new client with fresh token
        # (tokens may expire between calls)
        return AzureOpenAI(
            api_key=token,
            api_version=self.config.azure_openai_api_version,
            azure_endpoint=self.config.azure_openai_endpoint,
        )

    def embed(
        self, texts: list[str], batch_size: int = 64
    ) -> Iterator[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to process per API call.

        Yields:
            Embedding vectors as lists of floats.
        """
        client = self._get_client()

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            try:
                response = client.embeddings.create(
                    input=batch,
                    model=self.config.azure_openai_model,
                )

                for embedding_obj in response.data:
                    yield embedding_obj.embedding

            except Exception as e:
                logger.error("Azure OpenAI embedding failed: %s", e)
                raise


@lru_cache(maxsize=4)
def _get_azure_embedder(
    client_id: str | None,
    endpoint: str,
    model: str,
    api_version: str,
) -> AzureOpenAIEmbedder:
    """Get a cached Azure OpenAI embedder instance.

    Args are used as cache key - config is reconstructed inside.
    """
    from axon.config.embeddings import get_embedding_config

    config = get_embedding_config()
    return AzureOpenAIEmbedder(config)


def get_azure_embedder(config: "EmbeddingConfig") -> AzureOpenAIEmbedder:
    """Get an Azure OpenAI embedder for the given configuration.

    Uses caching based on configuration parameters.

    Args:
        config: Embedding configuration.

    Returns:
        AzureOpenAIEmbedder instance.
    """
    return _get_azure_embedder(
        config.uais_client_id,
        config.azure_openai_endpoint,
        config.azure_openai_model,
        config.azure_openai_api_version,
    )


def clear_azure_embedder_cache() -> None:
    """Clear the Azure embedder cache (useful for testing)."""
    _get_azure_embedder.cache_clear()
    _token_cache.clear()
