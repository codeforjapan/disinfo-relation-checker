"""Unit tests for Ollama provider with SOLID design."""

from unittest.mock import Mock, patch

import pytest

from disinfo_relation_checker.llm_providers import OllamaProvider


class TestOllamaProvider:
    """Test OllamaProvider following single responsibility principle."""

    def test_initialization(self) -> None:
        """Test OllamaProvider initialization with custom config."""
        provider = OllamaProvider(base_url="http://custom:8080", model="custom-model", timeout=90)

        assert provider.base_url == "http://custom:8080"
        assert provider.model == "custom-model"
        assert provider.timeout == 90

    def test_default_initialization(self) -> None:
        """Test OllamaProvider with default values."""
        provider = OllamaProvider()

        assert provider.base_url == "http://localhost:11434"
        assert provider.model == "gemma3n:e4b"
        assert provider.timeout == 30

    @patch("httpx.post")
    def test_successful_generation(self, mock_post: Mock) -> None:
        """Test successful text generation."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Classification: 1\\nConfidence: 0.85"}
        mock_post.return_value = mock_response

        provider = OllamaProvider()
        prompt = "Test prompt"

        result = provider.generate(prompt)

        assert result == "Classification: 1\\nConfidence: 0.85"

        # Verify HTTP request
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["url"] == "http://localhost:11434/api/generate"
        assert call_args[1]["timeout"] == 30

        request_data = call_args[1]["json"]
        assert request_data["model"] == "gemma3n:e4b"
        assert request_data["prompt"] == prompt
        assert request_data["stream"] is False

    @patch("httpx.post")
    def test_http_error_handling(self, mock_post: Mock) -> None:
        """Test handling of HTTP errors."""
        # Mock HTTP error
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        provider = OllamaProvider()

        with pytest.raises(RuntimeError, match="Ollama API error \\(500\\): Internal Server Error"):
            provider.generate("Test prompt")

    @patch("httpx.post")
    def test_connection_error_handling(self, mock_post: Mock) -> None:
        """Test handling of connection errors."""
        # Mock connection error
        import httpx

        mock_post.side_effect = httpx.ConnectError("Connection failed")

        provider = OllamaProvider()

        with pytest.raises(RuntimeError, match="Failed to connect to Ollama"):
            provider.generate("Test prompt")

    @patch("httpx.post")
    def test_timeout_error_handling(self, mock_post: Mock) -> None:
        """Test handling of timeout errors."""
        # Mock timeout error
        import httpx

        mock_post.side_effect = httpx.TimeoutException("Request timeout")

        provider = OllamaProvider()

        with pytest.raises(RuntimeError, match="Ollama request timeout"):
            provider.generate("Test prompt")

    @patch("httpx.post")
    def test_custom_model_and_url(self, mock_post: Mock) -> None:
        """Test provider with custom model and URL."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "test response"}
        mock_post.return_value = mock_response

        provider = OllamaProvider(base_url="http://custom:9999", model="custom-model", timeout=120)

        provider.generate("test")

        # Verify custom settings used
        call_args = mock_post.call_args
        assert call_args[1]["url"] == "http://custom:9999/api/generate"
        assert call_args[1]["timeout"] == 120
        assert call_args[1]["json"]["model"] == "custom-model"

    @patch("httpx.post")
    def test_json_parsing_error(self, mock_post: Mock) -> None:
        """Test handling of JSON parsing errors."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.text = "Invalid JSON response"
        mock_post.return_value = mock_response

        provider = OllamaProvider()

        with pytest.raises(RuntimeError, match="Invalid JSON response from Ollama"):
            provider.generate("Test prompt")

    @patch("httpx.post")
    def test_missing_response_field(self, mock_post: Mock) -> None:
        """Test handling of missing response field in JSON."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"error": "Missing response field"}
        mock_post.return_value = mock_response

        provider = OllamaProvider()

        with pytest.raises(RuntimeError, match="Missing 'response' field in Ollama response"):
            provider.generate("Test prompt")
