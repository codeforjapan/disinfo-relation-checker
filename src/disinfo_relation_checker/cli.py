"""Command-line interface for disinfo-relation-checker."""

import argparse
from pathlib import Path
from typing import Any, Protocol

from disinfo_relation_checker.classifier import ClassifierImpl
from disinfo_relation_checker.csv_processor import CSVProcessorImpl


class Command(Protocol):
    """Protocol for command objects."""

    def execute(self, *, input_path: Path, output_path: Path) -> None:
        """Execute the command."""
        ...


class CSVProcessor(Protocol):
    """Protocol for CSV processing operations."""

    def read_csv(self, file_path: Path) -> list[dict[str, Any]]:
        """Read CSV file and return list of records."""
        ...

    def write_csv(self, file_path: Path, data: list[dict[str, Any]]) -> None:
        """Write data to CSV file."""
        ...


class Classifier(Protocol):
    """Protocol for text classification operations."""

    def classify_batch(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Classify a batch of texts and return results with classifications."""
        ...


class ClassifyCommand:
    """Command to classify texts in CSV files."""

    def __init__(self, csv_processor: CSVProcessor, classifier: Classifier) -> None:
        """Initialize with dependencies."""
        self.csv_processor = csv_processor
        self.classifier = classifier

    def execute(self, *, input_path: Path, output_path: Path) -> None:
        """Execute classification on input CSV and write results to output CSV."""
        # Read input data
        data = self.csv_processor.read_csv(input_path)

        # Classify the data
        results = self.classifier.classify_batch(data)

        # Write results
        self.csv_processor.write_csv(output_path, results)


class CommandFactory:
    """Factory for creating command objects."""

    def __init__(self, csv_processor: CSVProcessor, classifier: Classifier) -> None:
        """Initialize with dependencies."""
        self.csv_processor = csv_processor
        self.classifier = classifier

    def create_classify_command(self) -> ClassifyCommand:
        """Create a ClassifyCommand instance."""
        return ClassifyCommand(
            csv_processor=self.csv_processor,
            classifier=self.classifier,
        )


class CLIApplication:
    """Main CLI application."""

    def __init__(self, command_factory: CommandFactory) -> None:
        """Initialize with command factory."""
        self.command_factory = command_factory

    def run_classify(self, input_path: Path, output_path: Path) -> None:
        """Run classify command."""
        command = self.command_factory.create_classify_command()
        command.execute(input_path=input_path, output_path=output_path)


def main() -> None:
    """Initialize and run the CLI application."""
    parser = argparse.ArgumentParser(
        prog="disinfo-relation-checker",
        description="Check if a text is related to disinformation analysis topics",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="0.1.0",  # This should come from __version__
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Classify command
    classify_parser = subparsers.add_parser("classify", help="Classify texts in CSV file")
    classify_parser.add_argument("--input", type=Path, required=True, help="Input CSV file")
    classify_parser.add_argument("--output", type=Path, required=True, help="Output CSV file")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Wire up real dependencies

    csv_processor = CSVProcessorImpl()
    classifier = ClassifierImpl()
    command_factory = CommandFactory(csv_processor=csv_processor, classifier=classifier)
    app = CLIApplication(command_factory=command_factory)

    if args.command == "classify":
        app.run_classify(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
