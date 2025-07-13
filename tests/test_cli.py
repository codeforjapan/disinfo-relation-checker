"""Unit tests for CLI interface."""

import pytest
import typer
from typer.testing import CliRunner

from disinfo_relation_checker import __version__
from disinfo_relation_checker.cli import create_app, version_callback


def test_create_app_returns_typer_app() -> None:
    """Test that create_app returns a Typer application."""
    app = create_app()
    assert isinstance(app, typer.Typer)
    assert app.info.name == "disinfo-relation-checker"
    help_text = app.info.help or ""
    assert "Check if a given text is related to disinformation analysis topic" in help_text


def test_version_callback_with_false_does_nothing() -> None:
    """Test version callback with False value does nothing."""
    # Should not raise any exception
    version_callback(value=False)


def test_version_callback_with_true_prints_version_and_exits() -> None:
    """Test version callback with True value prints version and exits."""
    with pytest.raises(typer.Exit):
        version_callback(value=True)


def test_cli_version_flag() -> None:
    """Test CLI --version flag using CliRunner."""
    app = create_app()
    runner = CliRunner()

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert f"disinfo-relation-checker {__version__}" in result.stdout


def test_cli_help() -> None:
    """Test CLI help message."""
    app = create_app()
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Check if a given text is related to disinformation analysis topic" in result.stdout
    assert "--version" in result.stdout


def test_cli_no_args() -> None:
    """Test CLI with no arguments shows error."""
    app = create_app()
    runner = CliRunner()

    result = runner.invoke(app, [])

    # Typer returns exit code 2 when no command is given
    expected_exit_code = 2  # Exit code for missing command
    assert result.exit_code == expected_exit_code
    assert "Missing command" in result.stderr
    assert "Usage: disinfo-relation-checker" in result.stderr
