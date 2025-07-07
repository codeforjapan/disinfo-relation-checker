"""Nox configuration for disinfo_relation_checker project."""

import nox

nox.options.default_venv_backend = "uv"


@nox.session(python=["3.11"])
def full_test(session: nox.Session) -> None:
    """Run tests with pytest."""
    session.install("-e", ".", "--group", "dev")
    session.run("pytest", "--cov=disinfo_relation_checker", "--cov-report=term-missing")


@nox.session(python="3.11")
def lint_format(session: nox.Session) -> None:
    """Run linting with ruff."""
    session.install("-e", ".", "--group", "dev")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session(python="3.11")
def type_check(session: nox.Session) -> None:
    """Run type checking with mypy."""
    session.install("-e", ".", "--group", "dev")
    session.run("mypy", "--strict", "src/disinfo_relation_checker", "tests")
