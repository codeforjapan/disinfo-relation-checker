"""Text classification module with SOLID design principles."""

import re
from typing import Protocol


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    def generate(self, prompt: str) -> str:
        """Generate response from LLM."""


class PromptTemplate:
    """Handles prompt formatting and response parsing."""

    def __init__(self, template: str | None = None) -> None:
        """Initialize with optional custom template."""
        self._template = template or self._get_default_template()

    def _get_default_template(self) -> str:
        """Get default classification template."""
        return """Please classify the following text as relevant (1) or not relevant (0) to disinformation analysis.

Text: {text}

Consider if this text contains:
- Claims about political figures or institutions
- Conspiracy theories or unverified claims
- Health misinformation
- Content that could be used to spread disinformation

Respond with:
Classification: [0 or 1]
Confidence: [0.0 to 1.0]

Classification:"""

    def set_template(self, template: str) -> None:
        """Set new prompt template."""
        self._template = template

    def format_classification_prompt(self, text: str) -> str:
        """Format text for classification prompt."""
        return self._template.format(text=text)

    def parse_classification_response(self, response: str) -> tuple[str, float]:
        """Parse LLM response to extract prediction and confidence."""
        # Try to extract classification (0 or 1)
        classification_match = re.search(r"Classification:\s*([01])", response)
        if classification_match:
            prediction = classification_match.group(1)
        # Try alternative format
        elif "1" in response and ("relevant" in response.lower() or "yes" in response.lower()):
            prediction = "1"
        else:
            prediction = "0"  # Default to not relevant

        # Try to extract confidence
        confidence_match = re.search(r"Confidence:\s*(0?\.\d+|1\.0?)", response)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5 if classification_match else 0.0

        return prediction, confidence


class PerformanceMetrics:
    """Calculates performance metrics for classification."""

    def calculate_accuracy(self, predictions: list[str], true_labels: list[str]) -> float:
        """Calculate accuracy score."""
        if len(predictions) != len(true_labels):
            msg = "Predictions and true labels must have same length"
            raise ValueError(msg)

        if not predictions:
            return 0.0

        correct = sum(1 for p, t in zip(predictions, true_labels, strict=False) if p == t)
        return correct / len(predictions)

    def calculate_precision(self, predictions: list[str], true_labels: list[str]) -> float:
        """Calculate precision for positive class (1)."""
        if len(predictions) != len(true_labels):
            msg = "Predictions and true labels must have same length"
            raise ValueError(msg)

        true_positives = sum(1 for p, t in zip(predictions, true_labels, strict=False) if p == "1" and t == "1")
        predicted_positives = sum(1 for p in predictions if p == "1")

        return true_positives / predicted_positives if predicted_positives > 0 else 0.0

    def calculate_recall(self, predictions: list[str], true_labels: list[str]) -> float:
        """Calculate recall for positive class (1)."""
        if len(predictions) != len(true_labels):
            msg = "Predictions and true labels must have same length"
            raise ValueError(msg)

        true_positives = sum(1 for p, t in zip(predictions, true_labels, strict=False) if p == "1" and t == "1")
        actual_positives = sum(1 for t in true_labels if t == "1")

        return true_positives / actual_positives if actual_positives > 0 else 0.0

    def calculate_f1(self, predictions: list[str], true_labels: list[str]) -> float:
        """Calculate F1 score."""
        precision = self.calculate_precision(predictions, true_labels)
        recall = self.calculate_recall(predictions, true_labels)

        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    def generate_report(self, predictions: list[str], true_labels: list[str]) -> dict[str, float]:
        """Generate comprehensive performance report."""
        return {
            "accuracy": self.calculate_accuracy(predictions, true_labels),
            "precision": self.calculate_precision(predictions, true_labels),
            "recall": self.calculate_recall(predictions, true_labels),
            "f1": self.calculate_f1(predictions, true_labels),
        }


class MockLLMProvider:
    """Mock LLM provider for testing and development."""

    def generate(self, prompt: str) -> str:
        """Generate mock response based on simple heuristics."""
        # Simple heuristic: look for keywords that suggest disinformation relevance
        disinformation_keywords = [
            "corrupt",
            "conspiracy",
            "vaccine",
            "government",
            "cover-up",
            "fake",
            "lie",
            "truth",
            "secret",
            "manipulation",
        ]

        text_lower = prompt.lower()
        keyword_count = sum(1 for keyword in disinformation_keywords if keyword in text_lower)

        if keyword_count > 0:
            confidence = min(0.6 + (keyword_count * 0.1), 0.95)
            return f"Classification: 1\nConfidence: {confidence:.2f}"
        return "Classification: 0\nConfidence: 0.8"


class TextClassifier:
    """Main text classifier with dependency injection."""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        prompt_template: PromptTemplate | None = None,
        performance_metrics: PerformanceMetrics | None = None,
    ) -> None:
        """Initialize with dependencies."""
        self._llm_provider = llm_provider or MockLLMProvider()
        self._prompt_template = prompt_template or PromptTemplate()
        self._performance_metrics = performance_metrics or PerformanceMetrics()

    def classify_text(self, text: str) -> tuple[str, float]:
        """Classify single text and return prediction with confidence."""
        prompt = self._prompt_template.format_classification_prompt(text)
        response = self._llm_provider.generate(prompt)
        return self._prompt_template.parse_classification_response(response)

    def classify_batch(self, texts: list[str]) -> list[dict[str, str]]:
        """Classify batch of texts and return results."""
        results = []

        for text in texts:
            prediction, confidence = self.classify_text(text)
            results.append(
                {
                    "text": text,
                    "prediction": prediction,
                    "confidence": f"{confidence:.2f}",
                }
            )

        return results

    def validate(self, labeled_data: list[dict[str, str]]) -> dict[str, float]:
        """Validate classifier performance on labeled data."""
        predictions = []
        true_labels = []

        for item in labeled_data:
            text = item["text"]
            true_label = item["label"]

            prediction, _ = self.classify_text(text)
            predictions.append(prediction)
            true_labels.append(true_label)

        return self._performance_metrics.generate_report(predictions, true_labels)

    def set_prompt_template(self, template: str) -> None:
        """Set new prompt template for classification."""
        self._prompt_template.set_template(template)
