"""End-to-end tests for CLI functionality."""

import csv
import json
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import pytest
import yaml

import disinfo_relation_checker as drc


@pytest.mark.e2e
def test_version_command() -> None:
    """Test that --version command returns the package version."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Run the CLI command
    uv_path = shutil.which("uv")
    assert uv_path is not None, "uv command not found"

    result = subprocess.run(
        [uv_path, "run", "disinfo-relation-checker", "--version"],
        check=False,
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    # Assert the command succeeded
    assert result.returncode == 0

    # Assert the output contains the actual package version
    assert drc.__version__ in result.stdout
    # Allow warning messages in stderr (uv environment warnings)
    assert "warning:" in result.stderr.lower() or result.stderr == ""


@pytest.mark.e2e
def test_classify_command() -> None:
    """Test that classify command processes CSV and generates predictions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test input CSV
        input_file = temp_path / "input.csv"
        with input_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text"])
            writer.writerow(["This politician is corrupt"])
            writer.writerow(["Today is a nice day"])

        output_file = temp_path / "output.csv"

        # Get the project root directory
        project_root = Path(__file__).parent.parent

        # Run the CLI command
        uv_path = shutil.which("uv")
        assert uv_path is not None, "uv command not found"

        result = subprocess.run(
            [
                uv_path,
                "run",
                "disinfo-relation-checker",
                "classify",
                "--input",
                str(input_file),
                "--output",
                str(output_file),
            ],
            check=False,
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        # Assert the command succeeded
        assert result.returncode == 0

        # Assert output file was created
        assert output_file.exists()

        # Assert output has predictions
        with output_file.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert "text" in rows[0]
            assert "prediction" in rows[0]
            assert "confidence" in rows[0]


@pytest.mark.e2e
def test_validate_command() -> None:
    """Test that validate command evaluates model performance."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test labeled data CSV
        labeled_file = temp_path / "labeled.csv"
        with labeled_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            writer.writerow(["This politician is corrupt", "1"])
            writer.writerow(["Today is a nice day", "0"])
            writer.writerow(["Vaccine conspiracy theory", "1"])

        # Get the project root directory
        project_root = Path(__file__).parent.parent

        # Run the CLI command
        uv_path = shutil.which("uv")
        assert uv_path is not None, "uv command not found"

        result = subprocess.run(
            [uv_path, "run", "disinfo-relation-checker", "validate", "--labeled-data", str(labeled_file)],
            check=False,
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        # Assert the command succeeded
        assert result.returncode == 0

        # Assert output contains performance metrics
        assert "accuracy" in result.stdout.lower()
        assert "precision" in result.stdout.lower()
        assert "recall" in result.stdout.lower()
        assert "f1" in result.stdout.lower()


@pytest.mark.e2e
def test_config_file_support() -> None:
    """Test that CLI supports configuration files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create config file
        config_file = temp_path / "config.yaml"
        config_data = {"llm": {"provider_type": "mock"}}
        with config_file.open("w") as f:
            yaml.dump(config_data, f)

        # Create test input CSV
        input_file = temp_path / "input.csv"
        with input_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text"])
            writer.writerow(["Test text"])

        output_file = temp_path / "output.csv"

        # Get the project root directory
        project_root = Path(__file__).parent.parent

        # Run CLI with config file
        uv_path = shutil.which("uv")
        assert uv_path is not None, "uv command not found"

        result = subprocess.run(
            [
                uv_path,
                "run",
                "disinfo-relation-checker",
                "classify",
                "--config",
                str(config_file),
                "--input",
                str(input_file),
                "--output",
                str(output_file),
            ],
            check=False,
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        # Assert the command succeeded
        assert result.returncode == 0
        assert output_file.exists()


@pytest.mark.e2e
@pytest.mark.skip(reason="Requires running Ollama server")
def test_ollama_provider() -> None:
    """Test Ollama provider integration (requires running Ollama server)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create Ollama config
        config_file = temp_path / "ollama_config.yaml"
        config_data = {
            "llm": {
                "provider_type": "ollama",
                "base_url": "http://localhost:11434",
                "model": "gemma3n:e4b",
                "timeout": 30,
            }
        }
        with config_file.open("w") as f:
            yaml.dump(config_data, f)

        # Create test labeled data
        labeled_file = temp_path / "labeled.csv"
        with labeled_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            writer.writerow(["This politician is corrupt", "1"])

        # Get the project root directory
        project_root = Path(__file__).parent.parent

        # Run validate with Ollama config
        uv_path = shutil.which("uv")
        assert uv_path is not None, "uv command not found"

        result = subprocess.run(
            [
                uv_path,
                "run",
                "disinfo-relation-checker",
                "validate",
                "--config",
                str(config_file),
                "--labeled-data",
                str(labeled_file),
            ],
            check=False,
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        # Assert the command succeeded (if Ollama is available)
        # This test will be skipped in CI but can be run manually
        assert result.returncode == 0
        assert "accuracy" in result.stdout.lower()


@pytest.mark.e2e
def test_optimize_command() -> None:
    """Test that optimize command performs prompt optimization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create training data
        training_file = temp_path / "training.csv"
        with training_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            writer.writerow(["This politician is corrupt", "1"])
            writer.writerow(["Today is a nice day", "0"])
            writer.writerow(["Vaccine conspiracy theory", "1"])
            writer.writerow(["Weather forecast is sunny", "0"])

        # Get the project root directory
        project_root = Path(__file__).parent.parent

        # Run the optimize command
        uv_path = shutil.which("uv")
        assert uv_path is not None, "uv command not found"

        result = subprocess.run(
            [
                uv_path,
                "run",
                "disinfo-relation-checker",
                "optimize",
                "--training-data",
                str(training_file),
                "--target-accuracy",
                "0.9",
                "--max-iterations",
                "3",
            ],
            check=False,
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        # Assert the command succeeded
        assert result.returncode == 0

        # Assert output contains optimization results
        assert "optimization completed" in result.stdout.lower()
        assert "best prompt" in result.stdout.lower()
        assert "accuracy" in result.stdout.lower()


@pytest.mark.e2e
def test_evaluate_command() -> None:
    """Test that evaluate command evaluates model configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create model config file
        config_file = temp_path / "model_config.json"
        config_data = {
            "prompt_template": "Classify this text for disinformation relevance: {text}\nAnswer with 0 or 1.",
            "llm_config": {"provider_type": "mock"},
        }
        with config_file.open("w") as f:
            json.dump(config_data, f)

        # Create test data
        test_file = temp_path / "test_data.csv"
        with test_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            writer.writerow(["Political corruption claims", "1"])
            writer.writerow(["Normal daily conversation", "0"])

        # Get the project root directory
        project_root = Path(__file__).parent.parent

        # Run the evaluate command
        uv_path = shutil.which("uv")
        assert uv_path is not None, "uv command not found"

        result = subprocess.run(
            [
                uv_path,
                "run",
                "disinfo-relation-checker",
                "evaluate",
                "--model-config",
                str(config_file),
                "--test-data",
                str(test_file),
            ],
            check=False,
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        # Assert the command succeeded
        assert result.returncode == 0

        # Assert output contains evaluation metrics
        assert "evaluation results" in result.stdout.lower()
        assert "accuracy" in result.stdout.lower()
        assert "precision" in result.stdout.lower()
        assert "recall" in result.stdout.lower()
        assert "f1" in result.stdout.lower()


@pytest.mark.e2e
def test_register_model_command() -> None:
    """Test that register-model command saves model to registry."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create model config file
        config_file = temp_path / "model_config.json"
        config_data = {
            "prompt_template": "Classify this text as related (1) or not related (0) to disinformation: {text}",
            "llm_config": {"provider_type": "mock"},
            "performance": {"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1": 0.85},
        }
        with config_file.open("w") as f:
            json.dump(config_data, f)

        # Get the project root directory
        project_root = Path(__file__).parent.parent

        # Run the register-model command
        uv_path = shutil.which("uv")
        assert uv_path is not None, "uv command not found"

        # Use a unique model name to avoid conflicts
        unique_model_name = f"test_model_{uuid.uuid4().hex[:8]}"

        result = subprocess.run(
            [
                uv_path,
                "run",
                "disinfo-relation-checker",
                "register-model",
                "--model-config",
                str(config_file),
                "--model-name",
                unique_model_name,
                "--version",
                "1.0.0",
                "--description",
                "Test model for unit testing",
            ],
            check=False,
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        # Assert the command succeeded
        assert result.returncode == 0

        # Assert output contains registration confirmation
        assert "registered successfully" in result.stdout.lower()


@pytest.mark.e2e
def test_list_models_command() -> None:
    """Test that list-models command shows registered models."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Run the list-models command
    uv_path = shutil.which("uv")
    assert uv_path is not None, "uv command not found"

    result = subprocess.run(
        [uv_path, "run", "disinfo-relation-checker", "list-models"],
        check=False,
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    # Assert the command succeeded
    assert result.returncode == 0

    # Assert output contains models listing
    assert "registered models" in result.stdout.lower()


@pytest.mark.e2e
def test_ab_test_setup_command() -> None:
    """Test that ab-test-setup command configures A/B testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test data
        test_file = temp_path / "test_data.csv"
        with test_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            writer.writerow(["Test political content", "1"])
            writer.writerow(["Weather is nice", "0"])

        # First register the required models
        project_root = Path(__file__).parent.parent
        uv_path = shutil.which("uv")
        assert uv_path is not None, "uv command not found"

        # Create model config files
        model_config_1 = temp_path / "model_1.json"
        with model_config_1.open("w") as f:
            json.dump(
                {
                    "prompt_template": "Classify this text: {text}",
                    "llm_config": {"provider_type": "mock"},
                    "performance": {"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1": 0.85},
                },
                f,
            )

        model_config_2 = temp_path / "model_2.json"
        with model_config_2.open("w") as f:
            json.dump(
                {
                    "prompt_template": "Determine relevance: {text}",
                    "llm_config": {"provider_type": "mock"},
                    "performance": {"accuracy": 0.87, "precision": 0.84, "recall": 0.90, "f1": 0.87},
                },
                f,
            )

        # Use unique model name to avoid conflicts
        unique_model_name = f"ab_test_model_{uuid.uuid4().hex[:8]}"

        # Register model A (version 1.0.0)
        subprocess.run(
            [
                uv_path,
                "run",
                "disinfo-relation-checker",
                "register-model",
                "--model-config",
                str(model_config_1),
                "--model-name",
                unique_model_name,
                "--version",
                "1.0.0",
                "--description",
                "Test model version 1.0.0",
            ],
            check=True,
            cwd=project_root,
            capture_output=True,
        )

        # Register model B (version 1.1.0)
        subprocess.run(
            [
                uv_path,
                "run",
                "disinfo-relation-checker",
                "register-model",
                "--model-config",
                str(model_config_2),
                "--model-name",
                unique_model_name,
                "--version",
                "1.1.0",
                "--description",
                "Test model version 1.1.0",
            ],
            check=True,
            cwd=project_root,
            capture_output=True,
        )

        # Run the ab-test-setup command
        unique_test_name = f"test_{uuid.uuid4().hex[:8]}"
        result = subprocess.run(
            [
                uv_path,
                "run",
                "disinfo-relation-checker",
                "ab-test-setup",
                "--model-a",
                f"{unique_model_name}:1.0.0",
                "--model-b",
                f"{unique_model_name}:1.1.0",
                "--test-data",
                str(test_file),
                "--traffic-split",
                "50",
                "--test-name",
                unique_test_name,
            ],
            check=False,
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        # Assert the command succeeded
        assert result.returncode == 0

        # Assert output contains A/B test configuration
        assert "a/b test" in result.stdout.lower()
        assert "configured" in result.stdout.lower()


@pytest.mark.e2e
def test_ab_test_results_command() -> None:
    """Test that ab-test-results command shows A/B test results."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Run the ab-test-results command
    uv_path = shutil.which("uv")
    assert uv_path is not None, "uv command not found"

    result = subprocess.run(
        [
            uv_path,
            "run",
            "disinfo-relation-checker",
            "ab-test-results",
            "--test-name",
            "nonexistent_test",
        ],
        check=False,
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    # Assert the command succeeded (even when no results are found)
    assert result.returncode == 0

    # Assert output indicates no results found (this is the expected behavior)
    assert "no results found" in result.stdout.lower()


@pytest.mark.e2e
def test_monitor_performance_command() -> None:
    """Test that monitor-performance command shows performance metrics."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Run the monitor-performance command
    uv_path = shutil.which("uv")
    assert uv_path is not None, "uv command not found"

    result = subprocess.run(
        [
            uv_path,
            "run",
            "disinfo-relation-checker",
            "monitor-performance",
            "--model-name",
            "test_model",
            "--time-range",
            "24h",
        ],
        check=False,
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    # Assert the command succeeded
    assert result.returncode == 0

    # Assert output contains performance summary (even with no data)
    assert "performance summary" in result.stdout.lower()
    assert "accuracy" in result.stdout.lower()
