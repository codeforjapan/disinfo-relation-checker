"""Unit tests for the CSV processor module."""

import tempfile
from pathlib import Path

import pytest

from disinfo_relation_checker.csv_processor import CSVProcessorImpl


def test_csv_processor_reads_csv_file(basic_input_csv: Path) -> None:
    """Test that CSVProcessor can read CSV files correctly."""
    processor = CSVProcessorImpl()
    data = processor.read_csv(basic_input_csv)

    expected_row_count = 2
    assert len(data) == expected_row_count
    assert data[0]["text"] == "This is test content"
    assert data[0]["source"] == "news_site"
    assert data[1]["text"] == "Another test text"
    assert data[1]["source"] == "blog"


def test_csv_processor_writes_csv_file() -> None:
    """Test that CSVProcessor can write CSV files correctly."""
    data = [
        {"text": "Sample text 1", "source": "news", "classification": "relevant", "confidence": 0.8},
        {"text": "Sample text 2", "source": "blog", "classification": "not_relevant", "confidence": 0.6},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as temp_file:
        temp_path = Path(temp_file.name)

        processor = CSVProcessorImpl()
        processor.write_csv(temp_path, data)

        assert temp_path.exists()
        content = temp_path.read_text()

        assert "text" in content
        assert "source" in content
        assert "classification" in content
        assert "confidence" in content

        assert "Sample text 1" in content
        assert "relevant" in content
        assert "0.8" in content


def test_csv_processor_handles_missing_file() -> None:
    """Test that CSVProcessor handles missing input files gracefully."""
    processor = CSVProcessorImpl()
    non_existent_path = Path("non_existent_file.csv")

    with pytest.raises(FileNotFoundError):
        processor.read_csv(non_existent_path)


def test_csv_processor_handles_empty_file(empty_csv: Path) -> None:
    """Test that CSVProcessor handles empty CSV files."""
    processor = CSVProcessorImpl()
    data = processor.read_csv(empty_csv)

    assert len(data) == 0


def test_csv_processor_preserves_column_order() -> None:
    """Test that CSVProcessor preserves column order when writing."""
    data = [
        {"text": "Text 1", "source": "news", "classification": "relevant", "confidence": 0.9, "metadata": "extra"},
        {"text": "Text 2", "source": "blog", "classification": "not_relevant", "confidence": 0.3, "metadata": "info"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as temp_file:
        temp_path = Path(temp_file.name)

        processor = CSVProcessorImpl()
        processor.write_csv(temp_path, data)

        content = temp_path.read_text()
        lines = content.strip().split("\n")
        header_line = lines[0]

        assert "text" in header_line
        assert "source" in header_line
        assert "classification" in header_line
        assert "confidence" in header_line
        assert "metadata" in header_line


def test_csv_processor_handles_special_characters(special_chars_csv: Path) -> None:
    """Test that CSVProcessor handles special characters and escaping correctly."""
    processor = CSVProcessorImpl()
    data = processor.read_csv(special_chars_csv)

    expected_row_count = 2
    assert len(data) == expected_row_count
    assert data[0]["text"] == 'Text with "quotes" and, commas'
    assert data[0]["source"] == "news_site"
    assert data[1]["text"] == "Text with\nnewlines"
    assert data[1]["source"] == "blog"
