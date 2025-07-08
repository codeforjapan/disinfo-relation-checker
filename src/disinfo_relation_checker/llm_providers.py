"""LLM providers with SOLID design principles."""

import httpx


class OllamaProvider:
    """Ollama LLM provider for local model inference."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma3n:e4b",
        timeout: int = 30,
    ) -> None:
        """Initialize Ollama provider with configuration."""
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """Generate response from Ollama API."""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = httpx.post(
                url=url,
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                msg = f"Ollama API error ({response.status_code}): {response.text}"
                raise RuntimeError(msg)

            try:
                response_data = response.json()
            except ValueError as e:
                msg = f"Invalid JSON response from Ollama: {response.text}"
                raise RuntimeError(msg) from e

            if "response" not in response_data:
                msg = "Missing 'response' field in Ollama response"
                raise RuntimeError(msg)

            response_text = response_data["response"]
            return str(response_text)

        except httpx.ConnectError as e:
            msg = f"Failed to connect to Ollama server at {self.base_url}"
            raise RuntimeError(msg) from e
        except httpx.TimeoutException as e:
            msg = "Ollama request timeout"
            raise RuntimeError(msg) from e
