"""Prompt optimization engine with SOLID design principles."""

from typing import Protocol, TypedDict, Unpack

from pydantic import BaseModel, Field

from .classifier import TextClassifier


class PromptEvaluatorProtocol(Protocol):
    """Protocol for prompt evaluation functions."""

    def __call__(self, template: str, labeled_data: list[dict[str, str]]) -> "PromptCandidate":
        """Evaluate a prompt template on labeled data."""
        ...


class OptimizationParams(TypedDict, total=False):
    """Parameters for optimization strategies."""

    max_generations: int
    max_iterations: int


class PromptCandidate(BaseModel):
    """Represents a prompt candidate with performance metrics."""

    template: str
    accuracy: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)

    def __lt__(self, other: "PromptCandidate") -> bool:
        """Compare candidates based on F1 score."""
        return self.f1 < other.f1

    def __gt__(self, other: "PromptCandidate") -> bool:
        """Compare candidates based on F1 score."""
        return self.f1 > other.f1

    def __eq__(self, other: object) -> bool:
        """Check equality based on all metrics."""
        if not isinstance(other, PromptCandidate):
            return False
        return (
            self.template == other.template
            and self.accuracy == other.accuracy
            and self.precision == other.precision
            and self.recall == other.recall
            and self.f1 == other.f1
        )


class PromptGenerator:
    """Generates prompt templates and variations following SRP."""

    def generate_base_templates(self) -> list[str]:
        """Generate base prompt templates for disinformation classification."""
        return [
            "Classify the following text as related (1) or not related (0) to disinformation: {text}",
            "Is this text relevant to disinformation analysis? Answer 1 for yes, 0 for no: {text}",
            "Analyze this text for disinformation content. Return 1 if relevant, 0 if not: {text}",
            "Does this text contain or discuss disinformation? Reply with 1 (yes) or 0 (no): {text}",
            "Evaluate if this text is about disinformation or misinformation. Output 1 or 0: {text}",
        ]

    def generate_variations(self, base_template: str) -> list[str]:
        """Generate variations of a base template."""
        variations = [base_template]  # Include original

        # Add different instruction styles
        if "classify" in base_template.lower():
            variations.extend(
                [
                    base_template.replace("Classify", "Determine"),
                    base_template.replace("Classify", "Categorize"),
                    base_template.replace("classify", "identify"),
                ],
            )

        # Add different response formats
        if "1 or 0" in base_template:
            variations.append(
                base_template.replace("1 or 0", "true or false").replace("1", "true").replace("0", "false"),
            )

        # Add different phrasings
        variations.extend(
            [
                f"Please {base_template.lower()}",
                f"{base_template}\n\nAnswer:",
                f"Task: {base_template}",
            ],
        )

        return variations

    def generate_few_shot_templates(self, training_data: list[dict[str, str]]) -> list[str]:
        """Generate few-shot prompt templates using training examples."""
        if len(training_data) < 2:
            return self.generate_base_templates()

        # Select representative examples
        positive_examples = [item for item in training_data if item["label"] == "1"][:2]
        negative_examples = [item for item in training_data if item["label"] == "0"][:2]

        if not positive_examples or not negative_examples:
            return self.generate_base_templates()

        templates = []

        # Two-shot template
        example_text = "Examples:\n"
        for example in positive_examples[:1] + negative_examples[:1]:
            example_text += f"Text: {example['text']}\nLabel: {example['label']}\n\n"

        templates.append(f"{example_text}Now classify this text:\nText: {{text}}\nLabel:")

        # Few-shot with context
        context = "Classify text as related to disinformation (1) or not (0).\n\n"
        templates.append(f"{context}{example_text}Text: {{text}}\nLabel:")

        return templates


class OptimizationMetrics:
    """Calculates optimization metrics following SRP."""

    def calculate_fitness_score(self, accuracy: float, precision: float, recall: float, f1: float) -> float:
        """Calculate weighted fitness score for optimization."""
        # Weighted average favoring F1 score
        weights = {"accuracy": 0.3, "precision": 0.2, "recall": 0.2, "f1": 0.3}

        return (
            accuracy * weights["accuracy"]
            + precision * weights["precision"]
            + recall * weights["recall"]
            + f1 * weights["f1"]
        )

    def meets_target_accuracy(self, accuracy: float, target: float) -> bool:
        """Check if accuracy meets target threshold."""
        return accuracy >= target


class OptimizationStrategy(Protocol):
    """Protocol for optimization strategies."""

    def optimize(
        self,
        initial_templates: list[str],
        training_data: list[dict[str, str]],
        evaluator: PromptEvaluatorProtocol,
        prompt_generator: PromptGenerator,
        **kwargs: Unpack[OptimizationParams],
    ) -> PromptCandidate:
        """Optimize prompts using specific strategy."""
        ...


class PromptOptimizer:
    """Main orchestrator for prompt optimization with dependency injection."""

    def __init__(
        self,
        prompt_generator: PromptGenerator | None = None,
        optimization_strategy: OptimizationStrategy | None = None,
        optimization_metrics: OptimizationMetrics | None = None,
        classifier: TextClassifier | None = None,
    ) -> None:
        """Initialize with injected dependencies."""
        self._prompt_generator = prompt_generator or PromptGenerator()
        self._optimization_strategy = optimization_strategy
        self._optimization_metrics = optimization_metrics or OptimizationMetrics()
        self._classifier = classifier

    def optimize_prompts(
        self,
        training_data: list[dict[str, str]],
        target_accuracy: float,
        max_iterations: int = 10,
    ) -> PromptCandidate:
        """Optimize prompts to meet target accuracy."""
        if not self._optimization_strategy:
            msg = "Optimization strategy not provided"
            raise ValueError(msg)

        # Generate initial templates
        initial_templates = self._prompt_generator.generate_base_templates()

        # Run optimization iteratively
        best_candidate = None

        for iteration in range(max_iterations):
            if iteration == 0:
                # First iteration with initial templates
                current_candidate = self._optimization_strategy.optimize(
                    initial_templates=initial_templates,
                    training_data=training_data,
                    evaluator=self.evaluate_prompt_template,
                    prompt_generator=self._prompt_generator,
                    max_iterations=1,
                )
            else:
                # Subsequent iterations with best template
                assert best_candidate is not None  # For mypy
                current_candidate = self._optimization_strategy.optimize(
                    initial_templates=[best_candidate.template],
                    training_data=training_data,
                    evaluator=self.evaluate_prompt_template,
                    prompt_generator=self._prompt_generator,
                    max_iterations=1,
                )

            # Update best candidate
            if best_candidate is None or current_candidate > best_candidate:
                best_candidate = current_candidate

            # Check if target is met
            if self._optimization_metrics.meets_target_accuracy(best_candidate.accuracy, target_accuracy):
                break

        if best_candidate is None:
            msg = "No valid candidate found during optimization"
            raise ValueError(msg)
        return best_candidate

    def evaluate_prompt_template(self, template: str, labeled_data: list[dict[str, str]]) -> PromptCandidate:
        """Evaluate a prompt template on labeled data."""
        if not self._classifier:
            msg = "Classifier not provided"
            raise ValueError(msg)

        # Update classifier with new template
        self._classifier.set_prompt_template(template)

        # Validate on labeled data
        metrics = self._classifier.validate(labeled_data)

        return PromptCandidate(
            template=template,
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
        )
