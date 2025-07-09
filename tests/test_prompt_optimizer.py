"""Unit tests for prompt optimization engine with SOLID design."""

from unittest.mock import Mock

from disinfo_relation_checker.prompt_optimizer import (
    OptimizationMetrics,
    PromptCandidate,
    PromptGenerator,
    PromptOptimizer,
)


class TestPromptCandidate:
    """Test PromptCandidate following single responsibility principle."""

    def test_prompt_candidate_creation(self) -> None:
        """Test PromptCandidate creation and properties."""
        candidate = PromptCandidate(template="Classify: {text}", accuracy=0.85, precision=0.90, recall=0.80, f1=0.85)

        assert candidate.template == "Classify: {text}"
        assert candidate.accuracy == 0.85
        assert candidate.precision == 0.90
        assert candidate.recall == 0.80
        assert candidate.f1 == 0.85

    def test_prompt_candidate_comparison(self) -> None:
        """Test PromptCandidate comparison based on F1 score."""
        candidate1 = PromptCandidate(template="Template 1", accuracy=0.8, precision=0.8, recall=0.8, f1=0.8)
        candidate2 = PromptCandidate(template="Template 2", accuracy=0.9, precision=0.9, recall=0.9, f1=0.9)

        assert candidate2 > candidate1
        assert candidate1 < candidate2
        assert candidate1 != candidate2

    def test_prompt_candidate_equality(self) -> None:
        """Test PromptCandidate equality."""
        candidate1 = PromptCandidate(template="Template", accuracy=0.8, precision=0.8, recall=0.8, f1=0.8)
        candidate2 = PromptCandidate(template="Template", accuracy=0.8, precision=0.8, recall=0.8, f1=0.8)

        assert candidate1 == candidate2


class TestPromptGenerator:
    """Test PromptGenerator following single responsibility principle."""

    def test_generate_base_templates(self) -> None:
        """Test generation of base prompt templates."""
        generator = PromptGenerator()

        templates = generator.generate_base_templates()

        assert len(templates) > 0
        assert all("{text}" in template for template in templates)
        assert any("classify" in template.lower() for template in templates)

    def test_generate_variations(self) -> None:
        """Test generation of prompt variations."""
        generator = PromptGenerator()
        base_template = "Classify this text: {text}"

        variations = generator.generate_variations(base_template)

        assert len(variations) > 0
        assert all("{text}" in variation for variation in variations)
        assert base_template in variations  # Should include original

    def test_generate_few_shot_templates(self) -> None:
        """Test generation of few-shot prompt templates."""
        generator = PromptGenerator()
        training_data = [{"text": "Political corruption", "label": "1"}, {"text": "Weather today", "label": "0"}]

        templates = generator.generate_few_shot_templates(training_data)

        assert len(templates) > 0
        assert all("{text}" in template for template in templates)
        assert any("Political corruption" in template for template in templates)


class TestOptimizationMetrics:
    """Test OptimizationMetrics following single responsibility principle."""

    def test_calculate_fitness_score(self) -> None:
        """Test fitness score calculation."""
        metrics = OptimizationMetrics()

        # Test balanced performance
        score = metrics.calculate_fitness_score(0.8, 0.8, 0.8, 0.8)
        assert score == 0.8

        # Test weighted average (F1 gets higher weight)
        score = metrics.calculate_fitness_score(0.7, 0.9, 0.7, 0.8)
        expected = (0.7 * 0.3) + (0.9 * 0.2) + (0.7 * 0.2) + (0.8 * 0.3)
        assert abs(score - expected) < 0.001

    def test_meets_target_accuracy(self) -> None:
        """Test target accuracy checking."""
        metrics = OptimizationMetrics()

        assert metrics.meets_target_accuracy(0.95, 0.9) is True
        assert metrics.meets_target_accuracy(0.85, 0.9) is False
        assert metrics.meets_target_accuracy(0.9, 0.9) is True


class TestPromptOptimizer:
    """Test PromptOptimizer with dependency injection."""

    def test_optimizer_initialization(self) -> None:
        """Test PromptOptimizer initialization with dependencies."""
        mock_generator = Mock()
        mock_strategy = Mock()
        mock_metrics = Mock()
        mock_classifier = Mock()

        optimizer = PromptOptimizer(
            prompt_generator=mock_generator,
            optimization_strategy=mock_strategy,
            optimization_metrics=mock_metrics,
            classifier=mock_classifier,
        )

        assert optimizer._prompt_generator == mock_generator
        assert optimizer._optimization_strategy == mock_strategy
        assert optimizer._optimization_metrics == mock_metrics
        assert optimizer._classifier == mock_classifier

    def test_optimize_prompts_success(self) -> None:
        """Test successful prompt optimization."""
        # Mock dependencies
        mock_generator = Mock()
        mock_generator.generate_base_templates.return_value = ["Template 1: {text}", "Template 2: {text}"]

        mock_strategy = Mock()
        mock_strategy.optimize.return_value = PromptCandidate(
            template="Best Template: {text}", accuracy=0.95, precision=0.95, recall=0.95, f1=0.95
        )

        mock_metrics = Mock()
        mock_metrics.meets_target_accuracy.return_value = True

        mock_classifier = Mock()

        optimizer = PromptOptimizer(
            prompt_generator=mock_generator,
            optimization_strategy=mock_strategy,
            optimization_metrics=mock_metrics,
            classifier=mock_classifier,
        )

        # Test optimization
        training_data = [{"text": "test", "label": "1"}]
        result = optimizer.optimize_prompts(training_data=training_data, target_accuracy=0.9, max_iterations=5)

        # Verify interactions
        mock_generator.generate_base_templates.assert_called_once()
        mock_strategy.optimize.assert_called_once()
        mock_metrics.meets_target_accuracy.assert_called_once_with(0.95, 0.9)

        assert result.template == "Best Template: {text}"
        assert result.accuracy == 0.95

    def test_optimize_prompts_max_iterations(self) -> None:
        """Test optimization stops at max iterations."""
        mock_generator = Mock()
        mock_generator.generate_base_templates.return_value = ["Template: {text}"]

        mock_strategy = Mock()
        mock_strategy.optimize.return_value = PromptCandidate(
            template="Template: {text}", accuracy=0.8, precision=0.8, recall=0.8, f1=0.8
        )

        mock_metrics = Mock()
        mock_metrics.meets_target_accuracy.return_value = False  # Never meets target

        mock_classifier = Mock()

        optimizer = PromptOptimizer(
            prompt_generator=mock_generator,
            optimization_strategy=mock_strategy,
            optimization_metrics=mock_metrics,
            classifier=mock_classifier,
        )

        # Test optimization with max iterations
        training_data = [{"text": "test", "label": "1"}]
        result = optimizer.optimize_prompts(
            training_data=training_data,
            target_accuracy=0.95,  # High target that won't be met
            max_iterations=2,
        )

        # Should call optimize exactly max_iterations times
        assert mock_strategy.optimize.call_count == 2
        assert result.accuracy == 0.8

    def test_evaluate_prompt_template(self) -> None:
        """Test evaluation of a specific prompt template."""
        mock_classifier = Mock()
        mock_classifier.validate.return_value = {"accuracy": 0.85, "precision": 0.90, "recall": 0.80, "f1": 0.85}

        optimizer = PromptOptimizer(classifier=mock_classifier)

        # Test evaluation
        template = "Classify: {text}"
        labeled_data = [{"text": "test", "label": "1"}]

        candidate = optimizer.evaluate_prompt_template(template, labeled_data)

        # Verify classifier was called with updated template
        mock_classifier.validate.assert_called_once_with(labeled_data)

        assert candidate.template == template
        assert candidate.accuracy == 0.85
        assert candidate.precision == 0.90
        assert candidate.recall == 0.80
        assert candidate.f1 == 0.85
