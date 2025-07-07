"""End-to-end tests for CLI functionality."""

import shutil
import subprocess
from pathlib import Path

import pytest

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
