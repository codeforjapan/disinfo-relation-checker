# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python package called `disinfo-relation-checker` that checks if a given text is related to a topic. This is intended to be used to judge if we should include the text into our disinformation analysis.Uses modern Python tooling with uv for dependency management, nox for automation, ruff for linting/formatting, mypy for type checking, and pytest for testing with coverage.

## Development Commands

Primary development workflow uses nox:
- `uv run nox -s full_test` - Run tests with coverage reporting
- `uv run nox -s lint_format` - Run linting and format checking with ruff
- `uv run nox -s type_check` - Run strict type checking with mypy
- `uv run nox` - Run all sessions

Direct tool usage:
- `uv run pytest --cov=disinfo_relation_checker --cov-report=term-missing` - Run tests with coverage
- `uv run pytest tests/test_specific.py` - Run specific test file
- `uv run ruff check .` - Lint code
- `uv run ruff format .` - Format code
- `uv run mypy --strict src/disinfo_relation_checker tests` - Type check with strict settings

## Development Philosophy

Follow Test-Driven Development (TDD) practices with Outside-In approach:

### TDD Workflow:
1. **Start with E2E tests**: Write failing end-to-end tests first (marked with `@pytest.mark.e2e`)
2. **Run E2E tests**: `uv run pytest -m e2e` to see failure messages
3. **Identify next module**: Use failure messages to determine which module/component needs implementation
4. **Write unit tests with SOLID design**: Create failing unit tests that enforce SOLID principles from the start
   - Design interfaces that follow single responsibility
   - Use dependency injection in test setup
   - Mock unimplemented dependencies
5. **Run unit tests only**: `uv run pytest -m "not e2e"` to focus on unit test failures
6. **Implement module**: Write minimal code to make unit tests pass, following the SOLID design enforced by tests. After each implementation, run the nox command `uv run nox` to check if the tests, linting, formatting and type checking pass.
7. **Repeat**: Continue until E2E tests pass

### Design Principles:
- **SOLID principles**: Design tests to enforce Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion from the beginning
- **Test-driven design**: Use unit tests to drive interface design and enforce single responsibility
- **Dependency Injection**: Structure tests to require dependency injection for better testability
- **Mocking**: Mock external dependencies and unimplemented modules using pytest-mock during development
