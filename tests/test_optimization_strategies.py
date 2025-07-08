"""Unit tests for optimization strategies with SOLID design."""

from unittest.mock import Mock

import pytest

from disinfo_relation_checker.optimization_strategies import (
    GeneticOptimizationStrategy,
    IterativeOptimizationStrategy,
    OptimizationStrategy,
)
from disinfo_relation_checker.prompt_optimizer import PromptCandidate


class TestOptimizationStrategy:
    """Test OptimizationStrategy interface (ABC)."""

    def test_strategy_is_abstract(self) -> None:
        """Test that OptimizationStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            OptimizationStrategy()  # type: ignore[abstract]


class TestGeneticOptimizationStrategy:
    """Test GeneticOptimizationStrategy following single responsibility principle."""

    def test_strategy_initialization(self) -> None:
        """Test GeneticOptimizationStrategy initialization."""
        strategy = GeneticOptimizationStrategy(population_size=10, mutation_rate=0.1, crossover_rate=0.8)

        assert strategy._population_size == 10
        assert strategy._mutation_rate == 0.1
        assert strategy._crossover_rate == 0.8

    def test_optimize_with_initial_population(self) -> None:
        """Test optimization with initial population generation."""
        mock_prompt_generator = Mock()
        mock_prompt_generator.generate_variations.return_value = [
            "Template 1: {text}",
            "Template 2: {text}",
            "Template 3: {text}",
        ]

        mock_evaluator = Mock()
        mock_evaluator.side_effect = [
            PromptCandidate("Template 1: {text}", 0.7, 0.7, 0.7, 0.7),
            PromptCandidate("Template 2: {text}", 0.8, 0.8, 0.8, 0.8),
            PromptCandidate("Template 3: {text}", 0.9, 0.9, 0.9, 0.9),
        ]

        strategy = GeneticOptimizationStrategy(population_size=3)

        initial_templates = ["Base template: {text}"]
        training_data = [{"text": "test", "label": "1"}]

        result = strategy.optimize(
            initial_templates=initial_templates,
            training_data=training_data,
            evaluator=mock_evaluator,
            prompt_generator=mock_prompt_generator,
            max_generations=1,
        )

        # Should return the best candidate
        assert result.template == "Template 3: {text}"
        assert result.f1 == 0.9
        assert mock_evaluator.call_count == 3

    def test_genetic_selection(self) -> None:
        """Test genetic selection mechanism."""
        strategy = GeneticOptimizationStrategy(population_size=3)

        population = [
            PromptCandidate("Template 1: {text}", 0.5, 0.5, 0.5, 0.5),
            PromptCandidate("Template 2: {text}", 0.8, 0.8, 0.8, 0.8),
            PromptCandidate("Template 3: {text}", 0.9, 0.9, 0.9, 0.9),
        ]

        selected = strategy._select_parents(population, num_parents=2)

        # Should prefer higher fitness candidates
        assert len(selected) == 2
        assert all(candidate.f1 >= 0.8 for candidate in selected)

    def test_genetic_crossover(self) -> None:
        """Test genetic crossover operation."""
        strategy = GeneticOptimizationStrategy()

        parent1 = PromptCandidate("Classify this text: {text}", 0.8, 0.8, 0.8, 0.8)
        parent2 = PromptCandidate("Analyze the content: {text}", 0.7, 0.7, 0.7, 0.7)

        offspring = strategy._crossover(parent1, parent2)

        # Should create new template combining elements
        assert "{text}" in offspring
        assert offspring != parent1.template
        assert offspring != parent2.template

    def test_genetic_mutation(self) -> None:
        """Test genetic mutation operation."""
        strategy = GeneticOptimizationStrategy(mutation_rate=1.0)  # Force mutation

        template = "Classify this text: {text}"
        mutated = strategy._mutate(template)

        # Should create different template but keep {text} placeholder
        assert "{text}" in mutated
        assert len(mutated) > 0


class TestIterativeOptimizationStrategy:
    """Test IterativeOptimizationStrategy following single responsibility principle."""

    def test_strategy_initialization(self) -> None:
        """Test IterativeOptimizationStrategy initialization."""
        strategy = IterativeOptimizationStrategy(improvement_threshold=0.05, refinement_steps=3)

        assert strategy._improvement_threshold == 0.05
        assert strategy._refinement_steps == 3

    def test_optimize_with_iterative_refinement(self) -> None:
        """Test optimization with iterative refinement."""
        mock_prompt_generator = Mock()
        mock_prompt_generator.generate_variations.side_effect = [
            ["Improved template 1: {text}"],
            ["Improved template 2: {text}"],
            ["Improved template 3: {text}"],
        ]

        mock_evaluator = Mock()
        mock_evaluator.side_effect = [
            PromptCandidate("Base template: {text}", 0.7, 0.7, 0.7, 0.7),
            PromptCandidate("Improved template 1: {text}", 0.8, 0.8, 0.8, 0.8),
            PromptCandidate("Improved template 2: {text}", 0.85, 0.85, 0.85, 0.85),
            PromptCandidate("Improved template 3: {text}", 0.87, 0.87, 0.87, 0.87),
        ]

        strategy = IterativeOptimizationStrategy(improvement_threshold=0.01, refinement_steps=3)

        initial_templates = ["Base template: {text}"]
        training_data = [{"text": "test", "label": "1"}]

        result = strategy.optimize(
            initial_templates=initial_templates,
            training_data=training_data,
            evaluator=mock_evaluator,
            prompt_generator=mock_prompt_generator,
            max_iterations=3,
        )

        # Should return the best refined template
        assert result.template == "Improved template 3: {text}"
        assert result.f1 == 0.87

    def test_iterative_early_stopping(self) -> None:
        """Test early stopping when no improvement is found."""
        mock_prompt_generator = Mock()
        mock_prompt_generator.generate_variations.return_value = ["No improvement template: {text}"]

        mock_evaluator = Mock()
        mock_evaluator.side_effect = [
            PromptCandidate("Base template: {text}", 0.8, 0.8, 0.8, 0.8),
            PromptCandidate("No improvement template: {text}", 0.79, 0.79, 0.79, 0.79),
        ]

        strategy = IterativeOptimizationStrategy(improvement_threshold=0.05)

        initial_templates = ["Base template: {text}"]
        training_data = [{"text": "test", "label": "1"}]

        result = strategy.optimize(
            initial_templates=initial_templates,
            training_data=training_data,
            evaluator=mock_evaluator,
            prompt_generator=mock_prompt_generator,
            max_iterations=10,
        )

        # Should stop early and return original best template
        assert result.template == "Base template: {text}"
        assert result.f1 == 0.8
        # Should only evaluate base + 1 refinement attempt
        assert mock_evaluator.call_count == 2

    def test_iterative_refinement_strategy(self) -> None:
        """Test refinement strategy selection."""
        strategy = IterativeOptimizationStrategy()

        current_best = PromptCandidate("Simple: {text}", 0.7, 0.7, 0.7, 0.7)
        refinement_type = strategy._select_refinement_strategy(current_best)

        # Should return a valid refinement strategy
        assert refinement_type in ["clarity", "specificity", "examples", "structure"]

    def test_apply_refinement_clarity(self) -> None:
        """Test clarity-based refinement."""
        strategy = IterativeOptimizationStrategy()

        template = "Classify: {text}"
        refined = strategy._apply_refinement(template, "clarity")

        # Should make template clearer
        assert "{text}" in refined
        assert len(refined) > len(template)
        assert "classify" in refined.lower()

    def test_apply_refinement_specificity(self) -> None:
        """Test specificity-based refinement."""
        strategy = IterativeOptimizationStrategy()

        template = "Analyze: {text}"
        refined = strategy._apply_refinement(template, "specificity")

        # Should make template more specific
        assert "{text}" in refined
        assert len(refined) > len(template)
        assert "disinformation" in refined.lower()

    def test_apply_refinement_examples(self) -> None:
        """Test example-based refinement."""
        strategy = IterativeOptimizationStrategy()

        template = "Classify: {text}"
        refined = strategy._apply_refinement(template, "examples")

        # Should add examples
        assert "{text}" in refined
        assert len(refined) > len(template)
        assert "example" in refined.lower() or ":" in refined

    def test_apply_refinement_structure(self) -> None:
        """Test structure-based refinement."""
        strategy = IterativeOptimizationStrategy()

        template = "Classify: {text}"
        refined = strategy._apply_refinement(template, "structure")

        # Should improve structure
        assert "{text}" in refined
        assert len(refined) > len(template)
