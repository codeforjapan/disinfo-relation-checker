"""Unit tests for settings configuration with SOLID design."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from disinfo_relation_checker.settings import (
    AppSettings,
    LLMConfigAdapter,
    MockLLMConfig,
    OllamaConfig,
)


class TestMockLLMConfig:
    """Test MockLLMConfig following single responsibility principle."""

    def test_default_values(self) -> None:
        """Test MockLLMConfig has correct default values."""
        config = MockLLMConfig()
        assert config.provider_type == "mock"

    def test_serialization(self) -> None:
        """Test MockLLMConfig serialization."""
        config = MockLLMConfig()
        data = config.model_dump()
        assert data == {"provider_type": "mock"}


class TestOllamaConfig:
    """Test OllamaConfig following single responsibility principle."""

    def test_default_values(self) -> None:
        """Test OllamaConfig has correct default values."""
        config = OllamaConfig()
        assert config.provider_type == "ollama"
        assert config.base_url == "http://localhost:11434"
        assert config.model == "gemma3n:e4b"
        assert config.timeout == 30

    def test_custom_values(self) -> None:
        """Test OllamaConfig with custom values."""
        config = OllamaConfig(base_url="http://custom:8080", model="llama3.2", timeout=60)
        assert config.provider_type == "ollama"
        assert config.base_url == "http://custom:8080"
        assert config.model == "llama3.2"
        assert config.timeout == 60

    def test_validation(self) -> None:
        """Test OllamaConfig validation."""
        # Valid config
        config = OllamaConfig(timeout=10)
        assert config.timeout == 10

        # Invalid timeout should fail validation
        with pytest.raises(ValueError):
            OllamaConfig(timeout=-1)


class TestLLMConfig:
    """Test LLMConfig discriminated union."""

    def test_mock_provider_discrimination(self) -> None:
        """Test that mock provider is correctly discriminated."""
        data = {"provider_type": "mock"}
        config = LLMConfigAdapter.validate_python(data)
        assert isinstance(config, MockLLMConfig)
        assert config.provider_type == "mock"

    def test_ollama_provider_discrimination(self) -> None:
        """Test that ollama provider is correctly discriminated."""
        data = {"provider_type": "ollama", "base_url": "http://localhost:11434", "model": "llama3.2", "timeout": 45}
        config = LLMConfigAdapter.validate_python(data)
        assert isinstance(config, OllamaConfig)
        assert config.provider_type == "ollama"
        assert config.base_url == "http://localhost:11434"
        assert config.model == "llama3.2"
        assert config.timeout == 45

    def test_invalid_provider_type(self) -> None:
        """Test that invalid provider type raises error."""
        data = {"provider_type": "invalid"}
        with pytest.raises(ValueError):
            LLMConfigAdapter.validate_python(data)


class TestAppSettings:
    """Test AppSettings with dependency injection."""

    def test_default_settings(self) -> None:
        """Test AppSettings with default values."""
        settings = AppSettings()
        assert isinstance(settings.llm, MockLLMConfig)
        assert settings.llm.provider_type == "mock"

    def test_environment_variables(self) -> None:
        """Test AppSettings from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "LLM__PROVIDER_TYPE": "ollama",
                "LLM__BASE_URL": "http://test:1234",
                "LLM__MODEL": "test-model",
                "LLM__TIMEOUT": "90",
            },
        ):
            settings = AppSettings()
            assert isinstance(settings.llm, OllamaConfig)
            assert settings.llm.provider_type == "ollama"
            assert settings.llm.base_url == "http://test:1234"
            assert settings.llm.model == "test-model"
            assert settings.llm.timeout == 90

    def test_config_file_loading(self) -> None:
        """Test AppSettings from YAML config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "llm": {
                    "provider_type": "ollama",
                    "base_url": "http://file-config:5678",
                    "model": "file-model",
                    "timeout": 120,
                }
            }
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            settings = AppSettings.from_yaml(config_path)
            assert isinstance(settings.llm, OllamaConfig)
            assert settings.llm.provider_type == "ollama"
            assert settings.llm.base_url == "http://file-config:5678"
            assert settings.llm.model == "file-model"
            assert settings.llm.timeout == 120
        finally:
            config_path.unlink()

    def test_environment_overrides_defaults(self) -> None:
        """Test that environment variables override defaults."""
        with patch.dict("os.environ", {"LLM__PROVIDER_TYPE": "mock"}):
            settings = AppSettings()
            assert isinstance(settings.llm, MockLLMConfig)
            assert settings.llm.provider_type == "mock"

    def test_invalid_config_file(self) -> None:
        """Test handling of invalid config files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content:")
            config_path = Path(f.name)

        try:
            with pytest.raises(Exception):  # YAML parsing error
                AppSettings.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_missing_config_file(self) -> None:
        """Test handling of missing config files."""
        missing_path = Path("/nonexistent/config.yaml")
        with pytest.raises(FileNotFoundError):
            AppSettings.from_yaml(missing_path)
