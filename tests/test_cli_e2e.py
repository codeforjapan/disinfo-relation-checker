"""End-to-end tests for CLI interface."""

import re
import subprocess

import pytest


@pytest.mark.e2e
def test_cli_version_command() -> None:
    """Test --version flag works from command line."""
    result = subprocess.run(
        ["disinfo-relation-checker", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    # Check version format: "disinfo-relation-checker X.Y.Z"
    version_pattern = r"disinfo-relation-checker \d+\.\d+\.\d+"
    assert re.match(version_pattern, result.stdout.strip())
    assert result.stderr == ""
