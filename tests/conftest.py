"""Shared pytest fixtures for all test modules."""

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to fixtures directory.

    This fixture provides access to the test fixtures directory
    and can be used across all test modules.
    """
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def csv_processor_fixtures_dir(fixtures_dir: Path) -> Path:
    """Return path to CSV processor fixtures directory."""
    return fixtures_dir / "csv_processor"


@pytest.fixture
def basic_input_csv(csv_processor_fixtures_dir: Path) -> Path:
    """Return path to basic input CSV fixture."""
    return csv_processor_fixtures_dir / "input" / "basic_input.csv"


@pytest.fixture
def empty_csv(csv_processor_fixtures_dir: Path) -> Path:
    """Return path to empty CSV fixture."""
    return csv_processor_fixtures_dir / "input" / "empty_file.csv"


@pytest.fixture
def special_chars_csv(csv_processor_fixtures_dir: Path) -> Path:
    """Return path to CSV with special characters fixture."""
    return csv_processor_fixtures_dir / "special_cases" / "quotes_and_commas.csv"


@pytest.fixture
def sample_input_csv(fixtures_dir: Path) -> Path:
    """Return path to sample input CSV file for E2E tests."""
    return fixtures_dir / "sample_input.csv"
