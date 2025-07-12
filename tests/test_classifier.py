"""Unit tests for text classification module with SOLID design."""

from unittest.mock import Mock

from disinfo_relation_checker.classifier import (
    PerformanceMetrics,
    PromptTemplate,
    TextClassifier,
)


class TestPromptTemplate:
    """Test PromptTemplate following single responsibility principle."""

    def test_format_classification_prompt(self) -> None:
        """Test that PromptTemplate formats classification prompts correctly."""
        template = PromptTemplate()

        text = "This politician is corrupt"
        prompt = template.format_classification_prompt(text)

        assert "This politician is corrupt" in prompt
        assert "disinformation" in prompt.lower() or "relevant" in prompt.lower()

    def test_parse_classification_response(self) -> None:
        """Test that PromptTemplate parses LLM responses correctly."""
        template = PromptTemplate()

        # Test valid response
        response = "Classification: 1\nConfidence: 0.95"
        prediction, confidence = template.parse_classification_response(response)

        assert prediction == "1"
        assert confidence == 0.95

    def test_parse_invalid_classification_response(self) -> None:
        """Test that PromptTemplate handles invalid responses gracefully."""
        template = PromptTemplate()

        # Test invalid response
        response = "Invalid response format"
        prediction, confidence = template.parse_classification_response(response)

        assert prediction == "0"  # Default to not relevant
        assert confidence == 0.0  # Zero confidence for invalid responses


class TestPerformanceMetrics:
    """Test PerformanceMetrics following single responsibility principle."""

    def test_calculate_accuracy(self) -> None:
        """Test that PerformanceMetrics calculates accuracy correctly."""
        metrics = PerformanceMetrics()

        predictions = ["1", "0", "1", "0"]
        true_labels = ["1", "0", "1", "1"]

        accuracy = metrics.calculate_accuracy(predictions, true_labels)
        assert accuracy == 0.75  # 3 out of 4 correct

    def test_calculate_precision_recall_f1(self) -> None:
        """Test that PerformanceMetrics calculates precision, recall, and F1."""
        metrics = PerformanceMetrics()

        predictions = ["1", "1", "0", "0"]
        true_labels = ["1", "0", "1", "0"]

        precision = metrics.calculate_precision(predictions, true_labels)
        recall = metrics.calculate_recall(predictions, true_labels)
        f1 = metrics.calculate_f1(predictions, true_labels)

        assert precision == 0.5  # 1 TP / (1 TP + 1 FP)
        assert recall == 0.5  # 1 TP / (1 TP + 1 FN)
        assert f1 == 0.5  # 2 * (0.5 * 0.5) / (0.5 + 0.5)

    def test_generate_performance_report(self) -> None:
        """Test that PerformanceMetrics generates comprehensive reports."""
        metrics = PerformanceMetrics()

        predictions = ["1", "0", "1", "0"]
        true_labels = ["1", "0", "1", "1"]

        report = metrics.generate_report(predictions, true_labels)

        assert "accuracy" in report
        assert "precision" in report
        assert "recall" in report
        assert "f1" in report
        assert report["accuracy"] == 0.75


class TestTextClassifier:
    """Test TextClassifier with dependency injection."""

    def test_classify_single_text(self) -> None:
        """Test that TextClassifier classifies single text correctly."""
        # Mock dependencies
        mock_llm = Mock()
        mock_llm.generate.return_value = "Classification: 1\nConfidence: 0.95"

        mock_template = Mock()
        mock_template.format_classification_prompt.return_value = "Classify this text..."
        mock_template.parse_classification_response.return_value = ("1", 0.95)

        # Create classifier with injected dependencies
        classifier = TextClassifier(
            llm_provider=mock_llm,
            prompt_template=mock_template,
        )

        # Test classification
        text = "This politician is corrupt"
        prediction, confidence = classifier.classify_text(text)

        # Verify interactions
        mock_template.format_classification_prompt.assert_called_once_with(text)
        mock_llm.generate.assert_called_once_with("Classify this text...")
        mock_template.parse_classification_response.assert_called_once_with("Classification: 1\nConfidence: 0.95")

        assert prediction == "1"
        assert confidence == 0.95

    def test_classify_batch_texts(self) -> None:
        """Test that TextClassifier handles batch classification."""
        # Mock dependencies
        mock_llm = Mock()
        mock_llm.generate.side_effect = [
            "Classification: 1\nConfidence: 0.95",
            "Classification: 0\nConfidence: 0.85",
        ]

        mock_template = Mock()
        mock_template.format_classification_prompt.side_effect = [
            "Classify text 1...",
            "Classify text 2...",
        ]
        mock_template.parse_classification_response.side_effect = [
            ("1", 0.95),
            ("0", 0.85),
        ]

        classifier = TextClassifier(
            llm_provider=mock_llm,
            prompt_template=mock_template,
        )

        # Test batch classification
        texts = ["Text 1", "Text 2"]
        results = classifier.classify_batch(texts)

        assert len(results) == 2
        assert results[0]["text"] == "Text 1"
        assert results[0]["prediction"] == "1"
        assert results[0]["confidence"] == "0.95"
        assert results[1]["text"] == "Text 2"
        assert results[1]["prediction"] == "0"
        assert results[1]["confidence"] == "0.85"

    def test_validate_with_labeled_data(self) -> None:
        """Test that TextClassifier validates performance on labeled data."""
        # Mock dependencies
        mock_llm = Mock()
        mock_llm.generate.side_effect = [
            "Classification: 1\nConfidence: 0.95",
            "Classification: 0\nConfidence: 0.85",
        ]

        mock_template = Mock()
        mock_template.format_classification_prompt.side_effect = [
            "Classify text 1...",
            "Classify text 2...",
        ]
        mock_template.parse_classification_response.side_effect = [
            ("1", 0.95),
            ("0", 0.85),
        ]

        mock_metrics = Mock()
        mock_metrics.generate_report.return_value = {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }

        classifier = TextClassifier(
            llm_provider=mock_llm,
            prompt_template=mock_template,
            performance_metrics=mock_metrics,
        )

        # Test validation
        labeled_data = [
            {"text": "Text 1", "label": "1"},
            {"text": "Text 2", "label": "0"},
        ]

        report = classifier.validate(labeled_data)

        # Verify metrics calculation
        mock_metrics.generate_report.assert_called_once_with(["1", "0"], ["1", "0"])
        assert report["accuracy"] == 1.0
