"""Optimization strategies with SOLID design principles."""

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypedDict, Unpack

from .prompt_optimizer import PromptCandidate, PromptGenerator

if TYPE_CHECKING:
    from .prompt_optimizer import PromptEvaluatorProtocol


class OptimizationParams(TypedDict, total=False):
    """Parameters for optimization strategies."""

    max_generations: int
    max_iterations: int


class OptimizationStrategy(ABC):
    """Abstract base class for optimization strategies."""

    @abstractmethod
    def optimize(
        self,
        initial_templates: list[str],
        training_data: list[dict[str, str]],
        evaluator: "PromptEvaluatorProtocol",
        prompt_generator: PromptGenerator,
        **kwargs: Unpack[OptimizationParams],
    ) -> PromptCandidate:
        """Optimize prompts using specific strategy."""
        raise NotImplementedError


class GeneticOptimizationStrategy(OptimizationStrategy):
    """Genetic algorithm optimization strategy."""

    def __init__(
        self,
        population_size: int = 10,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        random_seed: int = 42,
    ) -> None:
        """Initialize genetic optimization strategy."""
        self._population_size = population_size
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._random_seed = random_seed
        random.seed(random_seed)

    def optimize(
        self,
        initial_templates: list[str],
        training_data: list[dict[str, str]],
        evaluator: "PromptEvaluatorProtocol",
        prompt_generator: PromptGenerator,
        **kwargs: Unpack[OptimizationParams],
    ) -> PromptCandidate:
        """Optimize using genetic algorithm."""
        max_generations = kwargs.get("max_generations", 5)

        # Generate initial population
        population = self._generate_initial_population(initial_templates, prompt_generator, training_data, evaluator)

        best_candidate = max(population)

        for _generation in range(max_generations):
            # Selection
            parents = self._select_parents(population, self._population_size // 2)

            # Crossover and mutation
            offspring_templates = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    if random.random() < self._crossover_rate:
                        child1 = self._crossover(parents[i], parents[i + 1])
                        child2 = self._crossover(parents[i + 1], parents[i])
                    else:
                        child1 = parents[i].template
                        child2 = parents[i + 1].template

                    # Mutation
                    if random.random() < self._mutation_rate:
                        child1 = self._mutate(child1)
                    if random.random() < self._mutation_rate:
                        child2 = self._mutate(child2)

                    offspring_templates.extend([child1, child2])

            # Evaluate offspring
            offspring = [
                evaluator(template, training_data) for template in offspring_templates[: self._population_size]
            ]

            # Combine and select next generation
            combined = population + offspring
            population = sorted(combined, reverse=True)[: self._population_size]

            # Update best candidate
            current_best = max(population)
            best_candidate = max(best_candidate, current_best)

        return best_candidate

    def _generate_initial_population(
        self,
        initial_templates: list[str],
        prompt_generator: PromptGenerator,
        training_data: list[dict[str, str]],
        evaluator: "PromptEvaluatorProtocol",
    ) -> list[PromptCandidate]:
        """Generate initial population of prompt candidates."""
        templates = set()

        # Add initial templates
        templates.update(initial_templates)

        # Generate variations
        for template in initial_templates:
            variations = prompt_generator.generate_variations(template)
            templates.update(variations[:3])  # Add first 3 variations

        # Add base templates if we need more
        if len(templates) < self._population_size:
            base_templates = prompt_generator.generate_base_templates()
            templates.update(base_templates)

        # Limit to population size
        template_list = list(templates)[: self._population_size]

        # Evaluate all templates
        return [evaluator(template, training_data) for template in template_list]

    def _select_parents(self, population: list[PromptCandidate], num_parents: int) -> list[PromptCandidate]:
        """Select parents using tournament selection."""
        parents = []
        tournament_size = 3

        for _ in range(num_parents):
            # Tournament selection
            tournament = random.sample(population, min(tournament_size, len(population)))
            winner = max(tournament)
            parents.append(winner)

        return parents

    def _crossover(self, parent1: PromptCandidate, parent2: PromptCandidate) -> str:
        """Perform crossover between two parents."""
        template1 = parent1.template
        template2 = parent2.template

        # Simple word-level crossover
        words1 = template1.split()
        words2 = template2.split()

        if not words1 or not words2:
            return template1

        # Find crossover point
        min_length = min(len(words1), len(words2))
        if min_length <= 2:
            # For very short templates, just combine them
            return f"{words1[0]} {' '.join(words2[1:])}" if len(words2) > 1 else template1

        crossover_point = random.randint(1, min_length - 1)

        # Create offspring
        offspring_words = words1[:crossover_point] + words2[crossover_point:]
        offspring = " ".join(offspring_words)

        # Ensure {text} placeholder is preserved
        if "{text}" not in offspring and "{text}" in template1:
            offspring += " {text}"

        # Ensure offspring is different from parent
        if offspring == template1:
            # Fallback: swap first few words
            if len(words2) > 0:
                offspring_words = words2[:1] + words1[1:]
                offspring = " ".join(offspring_words)

        return offspring

    def _mutate(self, template: str) -> str:
        """Apply mutation to a template."""
        mutations = [
            self._add_instruction_mutation,
            self._replace_word_mutation,
            self._add_context_mutation,
        ]

        mutation_func = random.choice(mutations)
        return mutation_func(template)

    def _add_instruction_mutation(self, template: str) -> str:
        """Add instruction words to template."""
        instructions = ["Please", "Carefully", "Accurately", "Precisely"]
        instruction = random.choice(instructions)

        if instruction.lower() not in template.lower():
            return f"{instruction} {template.lower()}"
        return template

    def _replace_word_mutation(self, template: str) -> str:
        """Replace words with synonyms."""
        replacements = {
            "classify": "categorize",
            "determine": "identify",
            "analyze": "examine",
            "evaluate": "assess",
            "text": "content",
        }

        for old_word, new_word in replacements.items():
            if old_word in template.lower() and new_word not in template.lower():
                return template.replace(old_word, new_word)

        return template

    def _add_context_mutation(self, template: str) -> str:
        """Add context to template."""
        contexts = [
            "for disinformation analysis",
            "related to misinformation",
            "in the context of false information",
        ]

        context = random.choice(contexts)
        if context not in template and ":" in template:
            parts = template.split(":", 1)
            return f"{parts[0]} {context}: {parts[1]}"

        return template


class IterativeOptimizationStrategy(OptimizationStrategy):
    """Iterative refinement optimization strategy."""

    def __init__(self, improvement_threshold: float = 0.01, refinement_steps: int = 5) -> None:
        """Initialize iterative optimization strategy."""
        self._improvement_threshold = improvement_threshold
        self._refinement_steps = refinement_steps

    def optimize(
        self,
        initial_templates: list[str],
        training_data: list[dict[str, str]],
        evaluator: "PromptEvaluatorProtocol",
        prompt_generator: PromptGenerator,
        **kwargs: Unpack[OptimizationParams],
    ) -> PromptCandidate:
        """Optimize using iterative refinement."""
        max_iterations = kwargs.get("max_iterations", 10)

        # Evaluate initial templates
        candidates = [evaluator(template, training_data) for template in initial_templates]

        current_best = max(candidates) if candidates else None
        if not current_best:
            # Fallback to base templates
            base_templates = prompt_generator.generate_base_templates()
            current_best = evaluator(base_templates[0], training_data)

        # Iterative refinement
        for _iteration in range(max_iterations):
            # Generate refinements using prompt generator
            variations = prompt_generator.generate_variations(current_best.template)
            if variations and len(variations) > 1:
                # Use the first variation (skip original template)
                refined_template = variations[0]
            else:
                # Use internal refinement logic as fallback
                refinement_type = self._select_refinement_strategy(current_best)
                refined_template = self._apply_refinement(current_best.template, refinement_type)

            # Evaluate refinement
            refined_candidate = evaluator(refined_template, training_data)

            # Check for improvement
            improvement = refined_candidate.f1 - current_best.f1
            if improvement > self._improvement_threshold:
                current_best = refined_candidate
            else:
                # No significant improvement, try different approach
                break

        return current_best

    def _select_refinement_strategy(self, candidate: PromptCandidate) -> str:
        """Select refinement strategy based on current performance."""
        template = candidate.template

        # Strategy selection based on template characteristics
        if len(template.split()) < 10:
            return "clarity"  # Make short templates clearer
        if "example" not in template.lower():
            return "examples"  # Add examples if none present
        if "disinformation" not in template.lower():
            return "specificity"  # Make more specific
        return "structure"  # Improve structure

    def _apply_refinement(self, template: str, refinement_type: str) -> str:
        """Apply specific refinement to template."""
        if refinement_type == "clarity":
            return self._improve_clarity(template)
        if refinement_type == "specificity":
            return self._add_specificity(template)
        if refinement_type == "examples":
            return self._add_examples(template)
        if refinement_type == "structure":
            return self._improve_structure(template)
        return template

    def _improve_clarity(self, template: str) -> str:
        """Make template clearer and more explicit."""
        if "classify" in template.lower():
            # Handle both capitalized and lowercase versions
            if "Classify" in template:
                return template.replace(
                    "Classify:",
                    "Classify the following text as either related (1) or not related (0) to disinformation:",
                )
            return template.replace(
                "classify",
                "classify the following text as either related (1) or not related (0) to disinformation",
            )
        if "determine" in template.lower():
            return f"Please {template.lower()}"
        return f"Task: {template}"

    def _add_specificity(self, template: str) -> str:
        """Add domain-specific context."""
        specific_terms = ["disinformation", "misinformation", "false information", "propaganda"]

        if not any(term in template.lower() for term in specific_terms):
            return template.replace("{text}", "disinformation-related content: {text}")

        return template

    def _add_examples(self, template: str) -> str:
        """Add examples to template."""
        example_text = """
Examples:
- Political conspiracy theory → 1
- Weather forecast → 0

"""

        if "example" not in template.lower():
            return f"{example_text}{template}"

        return template

    def _improve_structure(self, template: str) -> str:
        """Improve template structure."""
        if "\n" not in template:
            # Add line breaks for better structure
            return template.replace(": {text}", ":\n{text}\n\nClassification:")

        return template
