"""Unit tests for the CLI module."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock

from disinfo_relation_checker.classifier import ClassifierImpl
from disinfo_relation_checker.cli import (
    ClassifyCommand,
    CLIApplication,
    Command,
    CommandFactory,
    main,
)
from disinfo_relation_checker.csv_processor import CSVProcessorImpl


def test_command_interface_defines_execute_method() -> None:
    """Test that Command interface defines execute method."""
    command = Mock(spec=Command)

    assert hasattr(command, "execute")


def test_classify_command_accepts_dependencies_via_constructor() -> None:
    """Test that ClassifyCommand follows Dependency Inversion Principle."""
    csv_processor = Mock(spec=CSVProcessorImpl)
    classifier = Mock(spec=ClassifierImpl)

    command = ClassifyCommand(
        csv_processor=csv_processor,
        classifier=classifier,
    )

    assert command.csv_processor is csv_processor
    assert command.classifier is classifier


def test_classify_command_execute_processes_input_file() -> None:
    """Test that ClassifyCommand.execute processes input file correctly."""
    csv_processor = Mock(spec=CSVProcessorImpl)
    classifier = Mock(spec=ClassifierImpl)
    input_path = Path("input.csv")
    output_path = Path("output.csv")

    csv_data = [
        {"text": "Sample text 1", "source": "news"},
        {"text": "Sample text 2", "source": "blog"},
    ]
    csv_processor.read_csv.return_value = csv_data

    classifications = [
        {"text": "Sample text 1", "source": "news", "classification": "relevant", "confidence": 0.8},
        {"text": "Sample text 2", "source": "blog", "classification": "not_relevant", "confidence": 0.6},
    ]
    classifier.classify_batch.return_value = classifications

    command = ClassifyCommand(
        csv_processor=csv_processor,
        classifier=classifier,
    )

    command.execute(input_path=input_path, output_path=output_path)

    csv_processor.read_csv.assert_called_once_with(input_path)
    classifier.classify_batch.assert_called_once_with(csv_data)
    csv_processor.write_csv.assert_called_once_with(output_path, classifications)


def test_command_factory_creates_classify_command() -> None:
    """Test that CommandFactory follows Factory Pattern."""
    csv_processor = Mock(spec=CSVProcessorImpl)
    classifier = Mock(spec=ClassifierImpl)

    factory = CommandFactory(
        csv_processor=csv_processor,
        classifier=classifier,
    )

    command = factory.create_classify_command()

    assert isinstance(command, ClassifyCommand)
    assert command.csv_processor is csv_processor
    assert command.classifier is classifier


def test_cli_application_accepts_command_factory() -> None:
    """Test that CLIApplication follows Dependency Inversion Principle."""
    command_factory = Mock(spec=CommandFactory)

    app = CLIApplication(command_factory=command_factory)

    assert app.command_factory is command_factory


def test_cli_application_run_classify_creates_and_executes_command() -> None:
    """Test that CLIApplication.run delegates to appropriate command."""
    command_factory = Mock(spec=CommandFactory)
    classify_command = Mock(spec=ClassifyCommand)
    command_factory.create_classify_command.return_value = classify_command

    app = CLIApplication(command_factory=command_factory)
    input_path = Path("input.csv")
    output_path = Path("output.csv")

    app.run_classify(input_path=input_path, output_path=output_path)

    command_factory.create_classify_command.assert_called_once()
    classify_command.execute.assert_called_once_with(
        input_path=input_path,
        output_path=output_path,
    )


def test_main_function_creates_cli_application_with_real_dependencies() -> None:
    """Test that main function wires up dependencies correctly."""
    # Test that main() properly creates the dependency graph without actual file processing

    # Mock sys.argv to test dependency creation without actual command execution
    original_argv = sys.argv

    # Create temporary files for the test
    with (
        tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as input_file,
        tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as output_file,
    ):
        input_file.write("text,source\ntest,news")
        input_file.flush()
        input_path = Path(input_file.name)

        output_path = Path(output_file.name)

        sys.argv = [
            "disinfo-relation-checker",
            "classify",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]

        try:
            # This should work now that we have implemented all dependencies
            main()  # Should not raise any exceptions

            # Verify output file was created
            assert output_path.exists()
        finally:
            sys.argv = original_argv
