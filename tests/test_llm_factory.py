"""Unit tests for LLM provider factory with SOLID design."""

from unittest.mock import Mock

import pytest

from disinfo_relation_checker.classifier import MockLLMProvider
from disinfo_relation_checker.llm_factory import LLMProviderFactory
from disinfo_relation_checker.llm_providers import OllamaProvider
from disinfo_relation_checker.settings import MockLLMConfig, OllamaConfig


class TestLLMProviderFactory:
    """Test LLMProviderFactory following factory pattern."""

    def test_create_mock_provider(self) -> None:
        """Test factory creates MockLLMProvider for mock config."""
        config = MockLLMConfig()
        factory = LLMProviderFactory()

        provider = factory.create_provider(config)

        assert isinstance(provider, MockLLMProvider)

    def test_create_ollama_provider(self) -> None:
        """Test factory creates OllamaProvider for ollama config."""
        config = OllamaConfig(base_url="http://test:1234", model="test-model", timeout=60)
        factory = LLMProviderFactory()

        provider = factory.create_provider(config)

        assert isinstance(provider, OllamaProvider)
        assert provider.base_url == "http://test:1234"
        assert provider.model == "test-model"
        assert provider.timeout == 60

    def test_invalid_config_type(self) -> None:
        """Test factory raises error for unknown config type."""
        # Create a mock config with unknown provider type
        invalid_config = Mock()
        invalid_config.provider_type = "unknown"

        factory = LLMProviderFactory()

        with pytest.raises(ValueError, match="Unknown LLM provider type: unknown"):
            factory.create_provider(invalid_config)

    def test_factory_is_extensible(self) -> None:
        """Test that factory pattern allows easy extension."""
        # This test documents the extensibility pattern
        factory = LLMProviderFactory()

        # Mock future config type
        future_config = Mock()
        future_config.provider_type = "future_provider"

        # Should raise error until implemented
        with pytest.raises(ValueError):
            factory.create_provider(future_config)
