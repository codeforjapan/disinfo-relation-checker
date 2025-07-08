"""Settings configuration module with pydantic-settings."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, TypeAdapter, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MockLLMConfig(BaseModel):
    """Configuration for Mock LLM provider."""

    provider_type: Literal["mock"] = "mock"


class OllamaConfig(BaseModel):
    """Configuration for Ollama LLM provider."""

    provider_type: Literal["ollama"] = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "gemma3n:e4b"
    timeout: int = 30

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout is positive."""
        if v <= 0:
            msg = "Timeout must be positive"
            raise ValueError(msg)
        return v


# Discriminated union for LLM configuration
LLMConfig = Annotated[MockLLMConfig | OllamaConfig, Field(discriminator="provider_type")]

# Type adapter for validation
LLMConfigAdapter: TypeAdapter[LLMConfig] = TypeAdapter(LLMConfig)


class AppSettings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    llm: LLMConfig = Field(default_factory=MockLLMConfig)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "AppSettings":
        """Load settings from YAML file."""
        if not config_path.exists():
            msg = f"Config file not found: {config_path}"
            raise FileNotFoundError(msg)

        with config_path.open() as f:
            config_data = yaml.safe_load(f)

        return cls.model_validate(config_data)
