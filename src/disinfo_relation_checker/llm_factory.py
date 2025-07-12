"""LLM provider factory with factory pattern."""

from typing import Protocol

from disinfo_relation_checker.classifier import MockLLMProvider
from disinfo_relation_checker.llm_providers import OllamaProvider
from disinfo_relation_checker.settings import LLMConfig, MockLLMConfig, OllamaConfig


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    def generate(self, prompt: str) -> str:
        """Generate response from LLM."""


class LLMProviderFactory:
    """Factory for creating LLM providers based on configuration."""

    def create_provider(self, config: LLMConfig) -> LLMProvider:
        """Create LLM provider based on configuration type."""
        if isinstance(config, MockLLMConfig):
            return MockLLMProvider()
        if isinstance(config, OllamaConfig):
            return OllamaProvider(
                base_url=config.base_url,
                model=config.model,
                timeout=config.timeout,
            )
        # This should never be reached due to discriminated union
        msg = f"Unknown LLM provider type: {config.provider_type}"
        raise ValueError(msg)  # pragma: no cover
