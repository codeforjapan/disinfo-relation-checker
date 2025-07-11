"""End-to-end tests for the CLI interface."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


class ExecutableNotFoundError(Exception):
    """Exception raised when an executable is not found."""

    def __init__(self, executable_name: str, bin_dir: Path) -> None:
        """Initialize the exception."""
        self.executable_name = executable_name
        self.bin_dir = bin_dir

    def __str__(self) -> str:
        """Return the string representation of the exception."""
        return f"Executable '{self.executable_name}' not found in {self.bin_dir} or is not executable"


def find_venv_executable(executable_name: str) -> Path:
    """Find the executable in the virtual environment."""
    if not hasattr(sys, "real_prefix") and not (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix):
        pass

    python_executable = Path(sys.executable)

    if sys.platform == "win32":
        bin_dir = python_executable.parent
        executable_name_with_ext = f"{executable_name}.exe"
    else:
        bin_dir = python_executable.parent
        executable_name_with_ext = executable_name

    executable_path = bin_dir / executable_name_with_ext

    if executable_path.exists() and os.access(executable_path, os.X_OK):
        return executable_path

    raise ExecutableNotFoundError(executable_name, bin_dir)


@pytest.fixture
def disinfo_relation_checker_path() -> Path:
    """Get the path to the disinfo-relation-checker executable."""
    return find_venv_executable("disinfo-relation-checker")


# fixtures_dir and sample_input_csv are now defined in conftest.py


@pytest.mark.e2e
def test_cli_classify_command(disinfo_relation_checker_path: Path, sample_input_csv: Path) -> None:
    """Test that the classify command processes CSV input and produces output."""
    input_path = sample_input_csv

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "output.csv"

        result = subprocess.run(
            [
                disinfo_relation_checker_path,
                "classify",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

        assert output_path.exists(), "Output file was not created"

        output_content = output_path.read_text()
        assert "text" in output_content
        assert "classification" in output_content
        assert "confidence" in output_content


@pytest.mark.e2e
def test_cli_version_command(disinfo_relation_checker_path: Path) -> None:
    """Test that the version command works."""
    result = subprocess.run(
        [disinfo_relation_checker_path, "--version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() != ""
