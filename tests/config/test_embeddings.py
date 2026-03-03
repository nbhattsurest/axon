"""Tests for embedding configuration.

Verifies that:
- Configuration loads correctly from environment variables
- Default values are applied when env vars are missing
- Provider selection works correctly
- Singleton pattern works with reset functionality
"""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch

from axon.config.embeddings import (
    EmbeddingConfig,
    EmbeddingProvider,
    get_embedding_config,
    load_embedding_config,
    reset_embedding_config,
)


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset config before and after each test."""
    reset_embedding_config()
    yield
    reset_embedding_config()


class TestEmbeddingProvider:
    """Tests for the EmbeddingProvider enum."""

    def test_fastembed_value(self) -> None:
        """FastEmbed provider has correct string value."""
        assert EmbeddingProvider.FASTEMBED.value == "fastembed"

    def test_azure_openai_value(self) -> None:
        """Azure OpenAI provider has correct string value."""
        assert EmbeddingProvider.AZURE_OPENAI.value == "azure_openai"


class TestLoadEmbeddingConfig:
    """Tests for load_embedding_config function."""

    def test_default_provider_is_fastembed(self) -> None:
        """Default provider is FastEmbed when no env var is set."""
        with patch.dict(os.environ, {}, clear=True):
            config = load_embedding_config()
        assert config.provider == EmbeddingProvider.FASTEMBED

    def test_azure_openai_provider_from_env(self) -> None:
        """Provider can be set to azure_openai via environment variable."""
        with patch.dict(os.environ, {"AXON_EMBEDDING_PROVIDER": "azure_openai"}, clear=True):
            config = load_embedding_config()
        assert config.provider == EmbeddingProvider.AZURE_OPENAI

    def test_fastembed_provider_from_env(self) -> None:
        """Provider can be explicitly set to fastembed."""
        with patch.dict(os.environ, {"AXON_EMBEDDING_PROVIDER": "fastembed"}, clear=True):
            config = load_embedding_config()
        assert config.provider == EmbeddingProvider.FASTEMBED

    def test_invalid_provider_defaults_to_fastembed(self) -> None:
        """Invalid provider values default to FastEmbed."""
        with patch.dict(os.environ, {"AXON_EMBEDDING_PROVIDER": "invalid"}, clear=True):
            config = load_embedding_config()
        assert config.provider == EmbeddingProvider.FASTEMBED

    def test_case_insensitive_provider(self) -> None:
        """Provider name is case-insensitive."""
        with patch.dict(os.environ, {"AXON_EMBEDDING_PROVIDER": "AZURE_OPENAI"}, clear=True):
            config = load_embedding_config()
        assert config.provider == EmbeddingProvider.AZURE_OPENAI

    def test_default_fastembed_model(self) -> None:
        """Default FastEmbed model is BAAI/bge-small-en-v1.5."""
        with patch.dict(os.environ, {}, clear=True):
            config = load_embedding_config()
        assert config.fastembed_model == "BAAI/bge-small-en-v1.5"

    def test_custom_fastembed_model(self) -> None:
        """FastEmbed model can be customized via environment variable."""
        with patch.dict(os.environ, {"FASTEMBED_MODEL": "BAAI/bge-base-en-v1.5"}, clear=True):
            config = load_embedding_config()
        assert config.fastembed_model == "BAAI/bge-base-en-v1.5"

    def test_uais_credentials_from_env(self) -> None:
        """UAIS credentials are loaded from environment variables."""
        env = {
            "UAIS_CLIENT_ID": "test-client-id",
            "UAIS_CLIENT_SECRET": "test-secret",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_embedding_config()
        assert config.uais_client_id == "test-client-id"
        assert config.uais_client_secret == "test-secret"

    def test_uais_credentials_none_when_not_set(self) -> None:
        """UAIS credentials are None when not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = load_embedding_config()
        assert config.uais_client_id is None
        assert config.uais_client_secret is None

    def test_azure_openai_endpoint_default(self) -> None:
        """Default Azure OpenAI endpoint is UHG API."""
        with patch.dict(os.environ, {}, clear=True):
            config = load_embedding_config()
        assert config.azure_openai_endpoint == "https://api.uhg.com/api/azureopenai"

    def test_azure_openai_endpoint_custom(self) -> None:
        """Azure OpenAI endpoint can be customized."""
        with patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://custom.endpoint"}, clear=True):
            config = load_embedding_config()
        assert config.azure_openai_endpoint == "https://custom.endpoint"

    def test_azure_openai_model_default(self) -> None:
        """Default Azure OpenAI model is text-embedding-ada-002."""
        with patch.dict(os.environ, {}, clear=True):
            config = load_embedding_config()
        assert config.azure_openai_model == "text-embedding-ada-002"

    def test_azure_openai_model_custom(self) -> None:
        """Azure OpenAI model can be customized."""
        with patch.dict(os.environ, {"AZURE_OPENAI_EMBEDDING_MODEL": "text-embedding-3-large"}, clear=True):
            config = load_embedding_config()
        assert config.azure_openai_model == "text-embedding-3-large"


class TestGetEmbeddingConfig:
    """Tests for get_embedding_config singleton."""

    def test_returns_config_instance(self) -> None:
        """Returns an EmbeddingConfig instance."""
        config = get_embedding_config()
        assert isinstance(config, EmbeddingConfig)

    def test_singleton_behavior(self) -> None:
        """Returns the same instance on subsequent calls."""
        config1 = get_embedding_config()
        config2 = get_embedding_config()
        assert config1 is config2

    def test_reset_clears_singleton(self) -> None:
        """Reset allows loading new config."""
        with patch.dict(os.environ, {"AXON_EMBEDDING_PROVIDER": "fastembed"}, clear=True):
            config1 = get_embedding_config()

        reset_embedding_config()

        with patch.dict(os.environ, {"AXON_EMBEDDING_PROVIDER": "azure_openai"}, clear=True):
            config2 = get_embedding_config()

        # After reset, a new config should be loaded
        assert config1 is not config2


class TestEmbeddingConfigDataclass:
    """Tests for EmbeddingConfig dataclass properties."""

    def test_frozen_dataclass(self) -> None:
        """EmbeddingConfig is immutable (frozen)."""
        config = EmbeddingConfig(provider=EmbeddingProvider.FASTEMBED)
        with pytest.raises(AttributeError):
            config.provider = EmbeddingProvider.AZURE_OPENAI  # type: ignore

    def test_default_values(self) -> None:
        """EmbeddingConfig has sensible defaults."""
        config = EmbeddingConfig(provider=EmbeddingProvider.FASTEMBED)
        assert config.fastembed_model == "BAAI/bge-small-en-v1.5"
        assert config.azure_openai_model == "text-embedding-ada-002"
        assert config.azure_openai_api_version == "2024-02-01"
        assert config.uais_auth_url == "https://api.uhg.com/oauth2/token"
        assert config.uais_scope == "https://api.uhg.com/.default"
