"""CLI module for disinfo-relation-checker."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter

from disinfo_relation_checker import __version__
from disinfo_relation_checker.ab_testing import ABTestConfig, ABTestRunner, ABTestStatus
from disinfo_relation_checker.classifier import TextClassifier
from disinfo_relation_checker.csv_processor import CsvProcessor
from disinfo_relation_checker.llm_factory import LLMProviderFactory
from disinfo_relation_checker.model_registry import (
    ModelRegistry,
    create_model_metadata_from_config,
)
from disinfo_relation_checker.optimization_strategies import (
    GeneticOptimizationStrategy,
)
from disinfo_relation_checker.performance_monitoring import (
    MetricCollector,
    PerformanceMonitor,
)
from disinfo_relation_checker.prompt_optimizer import PromptOptimizer
from disinfo_relation_checker.settings import AppSettings, LLMConfig
from disinfo_relation_checker.training_data import TrainingDataManager


class VersionProvider(Protocol):
    """Protocol for version providers."""

    def get_version(self) -> str:
        """Get the version string."""


class VersionOutputter:
    """Handles version output formatting."""

    def format_version(self, version: str) -> str:
        """Format version string for output."""
        return version


class PackageVersionProvider:
    """Provides version from package."""

    def get_version(self) -> str:
        """Get version from package."""
        return __version__


class CliHandler:
    """Handles CLI commands with dependency injection."""

    def __init__(
        self,
        version_provider: VersionProvider,
        version_outputter: VersionOutputter,
    ) -> None:
        """Initialize with dependencies."""
        self._version_provider = version_provider
        self._version_outputter = version_outputter

    def handle_version_command(self) -> str:
        """Handle version command."""
        version = self._version_provider.get_version()
        return self._version_outputter.format_version(version)


def handle_classify_command(args: argparse.Namespace) -> None:
    """Handle classify command."""
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Load configuration
    settings = load_settings(getattr(args, "config", None))

    # Initialize components with configuration
    csv_processor = CsvProcessor()
    llm_provider = LLMProviderFactory().create_provider(settings.llm)
    classifier = TextClassifier(llm_provider=llm_provider)

    try:
        # Read input data
        data = csv_processor.read_input_data(input_path)

        # Extract texts for classification
        texts = [row["text"] for row in data]

        # Classify texts
        results = classifier.classify_batch(texts)

        # Save results
        csv_processor.save_classification_results(output_path, results)

        print(f"Classification completed. Results saved to {output_path}")

    except Exception as e:
        print(f"Error during classification: {e}", file=sys.stderr)
        sys.exit(1)


def handle_validate_command(args: argparse.Namespace) -> None:
    """Handle validate command."""
    labeled_path = Path(args.labeled_data)

    # Load configuration
    settings = load_settings(getattr(args, "config", None))

    # Initialize components with configuration
    csv_processor = CsvProcessor()
    llm_provider = LLMProviderFactory().create_provider(settings.llm)
    classifier = TextClassifier(llm_provider=llm_provider)

    try:
        # Read labeled data
        labeled_data = csv_processor.read_labeled_data(labeled_path)

        # Validate classifier performance
        metrics = classifier.validate(labeled_data)

        # Display results
        print("\nValidation Results:")
        print(f"Accuracy:  {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall:    {metrics['recall']:.3f}")
        print(f"F1 Score:  {metrics['f1']:.3f}")

    except Exception as e:
        print(f"Error during validation: {e}", file=sys.stderr)
        sys.exit(1)


def handle_optimize_command(args: argparse.Namespace) -> None:
    """Handle optimize command."""
    training_path = Path(args.training_data)
    target_accuracy = float(args.target_accuracy)
    max_iterations = int(args.max_iterations)

    # Load configuration
    settings = load_settings(getattr(args, "config", None))

    # Initialize components
    llm_provider = LLMProviderFactory().create_provider(settings.llm)
    classifier = TextClassifier(llm_provider=llm_provider)
    training_manager = TrainingDataManager()

    # Select optimization strategy
    strategy = GeneticOptimizationStrategy(population_size=5, mutation_rate=0.2)
    optimizer = PromptOptimizer(optimization_strategy=strategy, classifier=classifier)

    try:
        # Load and split training data
        dataset = training_manager.load_and_split_data(training_path)
        training_data = dataset.train_data

        print(f"Loaded {len(training_data)} training samples")
        print(f"Target accuracy: {target_accuracy}")
        print(f"Max iterations: {max_iterations}")
        print("Starting optimization...\n")

        # Run optimization
        best_candidate = optimizer.optimize_prompts(
            training_data=training_data,
            target_accuracy=target_accuracy,
            max_iterations=max_iterations,
        )

        # Display results
        print("Optimization completed!")
        print("\nBest prompt template:")
        print(f"{best_candidate.template}")
        print("\nPerformance metrics:")
        print(f"Accuracy:  {best_candidate.accuracy:.3f}")
        print(f"Precision: {best_candidate.precision:.3f}")
        print(f"Recall:    {best_candidate.recall:.3f}")
        print(f"F1 Score:  {best_candidate.f1:.3f}")

        # Save optimized prompt to file
        if hasattr(args, "output") and args.output:
            output_path = Path(args.output)
            prompt_config = {
                "prompt_template": best_candidate.template,
                "performance": {
                    "accuracy": best_candidate.accuracy,
                    "precision": best_candidate.precision,
                    "recall": best_candidate.recall,
                    "f1": best_candidate.f1,
                },
            }
            with output_path.open("w") as f:
                json.dump(prompt_config, f, indent=2)
            print(f"\nOptimized prompt saved to: {output_path}")

    except Exception as e:
        print(f"Error during optimization: {e}", file=sys.stderr)
        sys.exit(1)


def handle_evaluate_command(args: argparse.Namespace) -> None:
    """Handle evaluate command."""
    model_config_path = Path(args.model_config)
    test_data_path = Path(args.test_data)

    try:
        # Load model configuration
        with model_config_path.open() as f:
            model_config = json.load(f)

        prompt_template = model_config.get("prompt_template")
        if not prompt_template:
            msg = "Model config must contain 'prompt_template'"
            raise ValueError(msg)

        # Load LLM configuration from model config or use default
        llm_config = model_config.get("llm_config")
        if llm_config:
            # Convert dict to settings format
            adapter: TypeAdapter[LLMConfig] = TypeAdapter(LLMConfig)
            llm_settings = adapter.validate_python(llm_config)
        else:
            # Use default settings
            settings = load_settings(getattr(args, "config", None))
            llm_settings = settings.llm

        # Initialize components
        llm_provider = LLMProviderFactory().create_provider(llm_settings)
        classifier = TextClassifier(llm_provider=llm_provider)

        # Set custom prompt template
        classifier.set_prompt_template(prompt_template)

        # Load test data
        csv_processor = CsvProcessor()
        test_data = csv_processor.read_labeled_data(test_data_path)

        print(f"Loaded {len(test_data)} test samples")
        print(f"Using prompt template: {prompt_template[:100]}...")
        print("Running evaluation...\n")

        # Evaluate model
        metrics = classifier.validate(test_data)

        # Display results
        print("Evaluation results:")
        print(f"Accuracy:  {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall:    {metrics['recall']:.3f}")
        print(f"F1 Score:  {metrics['f1']:.3f}")

        # Save evaluation results
        if hasattr(args, "output") and args.output:
            output_path = Path(args.output)
            evaluation_results = {
                "model_config": model_config,
                "test_data_path": str(test_data_path),
                "evaluation_metrics": metrics,
                "test_samples": len(test_data),
            }
            with output_path.open("w") as f:
                json.dump(evaluation_results, f, indent=2)
            print(f"\nEvaluation results saved to: {output_path}")

    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        sys.exit(1)


def load_settings(config_path: str | None) -> AppSettings:
    """Load settings from config file or environment."""
    if config_path:
        return AppSettings.from_yaml(Path(config_path))
    return AppSettings()


def handle_register_model_command(args: argparse.Namespace) -> None:
    """Handle register-model command."""
    model_config_path = Path(args.model_config)
    model_name = args.model_name
    version = args.version
    description = args.description

    try:
        # Load model configuration
        with model_config_path.open() as f:
            config_data = json.load(f)

        # Create model metadata
        metadata = create_model_metadata_from_config(
            name=model_name,
            version=version,
            description=description,
            config_data=config_data,
            tags=getattr(args, "tags", []),
        )

        # Register model
        registry = ModelRegistry()
        registry.register_model(metadata)

        print(f"Model {model_name}:{version} registered successfully")

    except Exception as e:
        print(f"Error registering model: {e}", file=sys.stderr)
        sys.exit(1)


def handle_list_models_command(_args: argparse.Namespace) -> None:
    """Handle list-models command."""
    try:
        registry = ModelRegistry()
        models = registry.list_models()

        if not models:
            print("No registered models found")
            return

        print("Registered models:")
        print("-" * 80)

        for model in models:
            print(f"Name: {model.name}")
            print(f"Version: {model.version}")
            print(f"Description: {model.description}")
            print(f"F1 Score: {model.performance.f1:.3f}")
            print(f"Created: {model.created_at}")
            print(f"Tags: {', '.join(model.tags) if model.tags else 'None'}")
            print("-" * 80)

    except Exception as e:
        print(f"Error listing models: {e}", file=sys.stderr)
        sys.exit(1)


def handle_ab_test_setup_command(args: argparse.Namespace) -> None:
    """Handle ab-test-setup command."""
    model_a = args.model_a
    model_b = args.model_b
    test_data_path = args.test_data
    traffic_split = int(args.traffic_split)
    test_name = args.test_name

    try:
        # Create A/B test configuration

        config = ABTestConfig(
            test_name=test_name,
            model_a=model_a,
            model_b=model_b,
            traffic_split=traffic_split,
            test_data_path=test_data_path,
            created_at=datetime.now(UTC).isoformat(),
            status=ABTestStatus.ACTIVE,
        )

        # Setup A/B test
        registry = ModelRegistry()
        settings = load_settings(getattr(args, "config", None))
        llm_provider = LLMProviderFactory().create_provider(settings.llm)
        classifier = TextClassifier(llm_provider=llm_provider)

        runner = ABTestRunner(
            model_registry=registry,
            classifier=classifier,
        )

        runner.setup_ab_test(config)

        print(f"A/B test '{test_name}' configured successfully")
        print(f"Model A: {model_a} ({traffic_split}% traffic)")
        print(f"Model B: {model_b} ({100 - traffic_split}% traffic)")

    except Exception as e:
        print(f"Error setting up A/B test: {e}", file=sys.stderr)
        sys.exit(1)


def handle_ab_test_results_command(args: argparse.Namespace) -> None:
    """Handle ab-test-results command."""
    test_name = args.test_name

    try:
        # Get A/B test results
        registry = ModelRegistry()
        settings = load_settings(getattr(args, "config", None))
        llm_provider = LLMProviderFactory().create_provider(settings.llm)
        classifier = TextClassifier(llm_provider=llm_provider)

        runner = ABTestRunner(
            model_registry=registry,
            classifier=classifier,
        )

        result = runner.get_ab_test_result(test_name)

        if not result:
            print(f"No results found for A/B test: {test_name}")
            return

        print(f"A/B Test Results: {test_name}")
        print("-" * 50)
        print("Model A Performance:")
        print(f"  Accuracy:  {result.model_a_performance['accuracy']:.3f}")
        print(f"  Precision: {result.model_a_performance['precision']:.3f}")
        print(f"  Recall:    {result.model_a_performance['recall']:.3f}")
        print(f"  F1 Score:  {result.model_a_performance['f1']:.3f}")
        print(f"  Sample Size: {result.sample_size_a}")
        print()
        print("Model B Performance:")
        print(f"  Accuracy:  {result.model_b_performance['accuracy']:.3f}")
        print(f"  Precision: {result.model_b_performance['precision']:.3f}")
        print(f"  Recall:    {result.model_b_performance['recall']:.3f}")
        print(f"  F1 Score:  {result.model_b_performance['f1']:.3f}")
        print(f"  Sample Size: {result.sample_size_b}")
        print()
        print(f"Winner: {result.winner}")
        print(f"Statistical Significance: {result.statistical_significance:.3f}")
        print(f"Completed: {result.completed_at}")

    except Exception as e:
        print(f"Error getting A/B test results: {e}", file=sys.stderr)
        sys.exit(1)


def handle_monitor_performance_command(args: argparse.Namespace) -> None:
    """Handle monitor-performance command."""
    model_name = args.model_name
    time_range = args.time_range

    try:
        # Get performance monitoring data
        from disinfo_relation_checker.performance_monitoring import FilePerformanceStorage

        storage = FilePerformanceStorage(Path.home() / ".disinfo_relation_checker" / "performance")
        collector = MetricCollector(storage=storage)
        monitor = PerformanceMonitor(
            metric_collector=collector,
            storage=storage,
        )

        # Get performance summary
        summary = monitor.get_performance_summary(model_name, time_range)
        print(summary)

        # Get active alerts
        alerts = monitor.list_active_alerts(model_name)
        if alerts:
            print(f"\nActive Alerts ({len(alerts)}):")
            for alert in alerts:
                print(f"- {alert.severity.upper()}: {alert.message}")
        else:
            print("\nNo active alerts")

        # Get health status
        health = monitor.get_model_health_status(model_name)
        status = health["status"]
        if isinstance(status, str):
            print(f"\nOverall Health: {status.upper()}")
        else:
            print(f"\nOverall Health: {status}")

    except Exception as e:
        print(f"Error monitoring performance: {e}", file=sys.stderr)
        sys.exit(1)


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        prog="disinfo-relation-checker",
        description="Check if a given text is related to disinformation analysis",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Classify command
    classify_parser = subparsers.add_parser("classify", help="Classify texts in CSV file for disinformation relevance")
    classify_parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    classify_parser.add_argument("--output", type=str, required=True, help="Output CSV file path for results")
    classify_parser.add_argument("--config", type=str, help="Configuration file path (YAML)")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate classifier performance on labeled data")
    validate_parser.add_argument(
        "--labeled-data",
        type=str,
        required=True,
        help="Labeled CSV file path with text and label columns",
    )
    validate_parser.add_argument("--config", type=str, help="Configuration file path (YAML)")

    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize prompt templates for better performance")
    optimize_parser.add_argument(
        "--training-data",
        type=str,
        required=True,
        help="Training data CSV file path with text and label columns",
    )
    optimize_parser.add_argument(
        "--target-accuracy",
        type=str,
        required=True,
        help="Target accuracy threshold (e.g., 0.9)",
    )
    optimize_parser.add_argument(
        "--max-iterations",
        type=str,
        default="10",
        help="Maximum optimization iterations (default: 10)",
    )
    optimize_parser.add_argument("--output", type=str, help="Output file path for optimized prompt (JSON)")
    optimize_parser.add_argument("--config", type=str, help="Configuration file path (YAML)")

    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate model performance with specific configuration")
    evaluate_parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Model configuration file path (JSON) with prompt_template and optional llm_config",
    )
    evaluate_parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Test data CSV file path with text and label columns",
    )
    evaluate_parser.add_argument("--output", type=str, help="Output file path for evaluation results (JSON)")
    evaluate_parser.add_argument("--config", type=str, help="Configuration file path (YAML)")

    # Register model command
    register_parser = subparsers.add_parser("register-model", help="Register a model in the model registry")
    register_parser.add_argument("--model-config", type=str, required=True, help="Model configuration file path (JSON)")
    register_parser.add_argument("--model-name", type=str, required=True, help="Name of the model")
    register_parser.add_argument("--version", type=str, required=True, help="Version of the model (e.g., 1.0.0)")
    register_parser.add_argument("--description", type=str, required=True, help="Description of the model")
    register_parser.add_argument("--tags", type=str, nargs="*", help="Tags for the model")

    # List models command
    subparsers.add_parser("list-models", help="List all registered models")

    # A/B test setup command
    ab_setup_parser = subparsers.add_parser("ab-test-setup", help="Setup an A/B test for model comparison")
    ab_setup_parser.add_argument("--model-a", type=str, required=True, help="Model A (format: name:version)")
    ab_setup_parser.add_argument("--model-b", type=str, required=True, help="Model B (format: name:version)")
    ab_setup_parser.add_argument("--test-data", type=str, required=True, help="Test data CSV file path")
    ab_setup_parser.add_argument(
        "--traffic-split",
        type=str,
        required=True,
        help="Traffic split percentage for model A (0-100)",
    )
    ab_setup_parser.add_argument("--test-name", type=str, required=True, help="Name of the A/B test")
    ab_setup_parser.add_argument("--config", type=str, help="Configuration file path (YAML)")

    # A/B test results command
    ab_results_parser = subparsers.add_parser("ab-test-results", help="Get A/B test results")
    ab_results_parser.add_argument("--test-name", type=str, required=True, help="Name of the A/B test")
    ab_results_parser.add_argument("--config", type=str, help="Configuration file path (YAML)")

    # Monitor performance command
    monitor_parser = subparsers.add_parser("monitor-performance", help="Monitor model performance metrics")
    monitor_parser.add_argument("--model-name", type=str, required=True, help="Name of the model to monitor")
    monitor_parser.add_argument(
        "--time-range",
        type=str,
        default="24h",
        help="Time range for monitoring (e.g., 24h, 7d, 30d)",
    )

    return parser


def _handle_command(args: argparse.Namespace) -> None:
    """Handle command execution based on parsed arguments."""
    if args.command == "classify":
        handle_classify_command(args)
    elif args.command == "validate":
        handle_validate_command(args)
    elif args.command == "optimize":
        handle_optimize_command(args)
    elif args.command == "evaluate":
        handle_evaluate_command(args)
    elif args.command == "register-model":
        handle_register_model_command(args)
    elif args.command == "list-models":
        handle_list_models_command(args)
    elif args.command == "ab-test-setup":
        handle_ab_test_setup_command(args)
    elif args.command == "ab-test-results":
        handle_ab_test_results_command(args)
    elif args.command == "monitor-performance":
        handle_monitor_performance_command(args)


def main() -> None:
    """Entry point for the CLI."""
    parser = _create_argument_parser()
    args = parser.parse_args()

    if args.command is None:
        # Show help if no command is provided
        parser.print_help()
    else:
        _handle_command(args)


if __name__ == "__main__":
    main()
