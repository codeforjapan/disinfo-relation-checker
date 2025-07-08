"""Unit tests for CSV processing module with SOLID design."""

import csv
import tempfile
from pathlib import Path
from unittest.mock import Mock

from disinfo_relation_checker.csv_processor import (
    CsvProcessor,
    CsvReader,
    CsvWriter,
    DataValidator,
)


class TestCsvReader:
    """Test CsvReader following single responsibility principle."""

    def test_read_csv_file(self) -> None:
        """Test that CsvReader reads CSV files correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["text"])
            writer.writerow(["Sample text 1"])
            writer.writerow(["Sample text 2"])
            temp_path = Path(f.name)

        try:
            reader = CsvReader()
            data = reader.read(temp_path)

            assert len(data) == 2
            assert data[0]["text"] == "Sample text 1"
            assert data[1]["text"] == "Sample text 2"
        finally:
            temp_path.unlink()

    def test_read_labeled_csv_file(self) -> None:
        """Test that CsvReader reads labeled CSV files correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            writer.writerow(["Sample text 1", "1"])
            writer.writerow(["Sample text 2", "0"])
            temp_path = Path(f.name)

        try:
            reader = CsvReader()
            data = reader.read(temp_path)

            assert len(data) == 2
            assert data[0]["text"] == "Sample text 1"
            assert data[0]["label"] == "1"
            assert data[1]["text"] == "Sample text 2"
            assert data[1]["label"] == "0"
        finally:
            temp_path.unlink()


class TestCsvWriter:
    """Test CsvWriter following single responsibility principle."""

    def test_write_predictions_to_csv(self) -> None:
        """Test that CsvWriter writes prediction results correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = Path(f.name)

        try:
            data = [
                {"text": "Sample text 1", "prediction": "1", "confidence": "0.95"},
                {"text": "Sample text 2", "prediction": "0", "confidence": "0.85"},
            ]

            writer = CsvWriter()
            writer.write(temp_path, data)

            # Verify written data
            with temp_path.open() as file:
                reader = csv.DictReader(file)
                rows = list(reader)

            assert len(rows) == 2
            assert rows[0]["text"] == "Sample text 1"
            assert rows[0]["prediction"] == "1"
            assert rows[0]["confidence"] == "0.95"
        finally:
            temp_path.unlink()


class TestDataValidator:
    """Test DataValidator following single responsibility principle."""

    def test_validate_input_data_structure(self) -> None:
        """Test that DataValidator validates input data structure."""
        validator = DataValidator()

        # Valid data
        valid_data = [{"text": "Sample text"}]
        assert validator.validate_input_data(valid_data) is True

        # Invalid data - missing text column
        invalid_data = [{"content": "Sample text"}]
        assert validator.validate_input_data(invalid_data) is False

    def test_validate_labeled_data_structure(self) -> None:
        """Test that DataValidator validates labeled data structure."""
        validator = DataValidator()

        # Valid labeled data
        valid_data = [{"text": "Sample text", "label": "1"}]
        assert validator.validate_labeled_data(valid_data) is True

        # Invalid data - missing label column
        invalid_data = [{"text": "Sample text"}]
        assert validator.validate_labeled_data(invalid_data) is False


class TestCsvProcessor:
    """Test CsvProcessor with dependency injection."""

    def test_process_input_file_for_classification(self) -> None:
        """Test that CsvProcessor handles classification workflow correctly."""
        # Mock dependencies
        mock_reader = Mock()
        mock_reader.read.return_value = [{"text": "Sample text"}]

        mock_validator = Mock()
        mock_validator.validate_input_data.return_value = True

        # Create processor with injected dependencies
        processor = CsvProcessor(
            csv_reader=mock_reader,
            data_validator=mock_validator,
        )

        # Test processing
        input_path = Path("input.csv")
        result = processor.read_input_data(input_path)

        # Verify interactions
        mock_reader.read.assert_called_once_with(input_path)
        mock_validator.validate_input_data.assert_called_once_with([{"text": "Sample text"}])
        assert result == [{"text": "Sample text"}]

    def test_process_labeled_file_for_validation(self) -> None:
        """Test that CsvProcessor handles validation workflow correctly."""
        # Mock dependencies
        mock_reader = Mock()
        mock_reader.read.return_value = [{"text": "Sample text", "label": "1"}]

        mock_validator = Mock()
        mock_validator.validate_labeled_data.return_value = True

        # Create processor with injected dependencies
        processor = CsvProcessor(
            csv_reader=mock_reader,
            data_validator=mock_validator,
        )

        # Test processing
        labeled_path = Path("labeled.csv")
        result = processor.read_labeled_data(labeled_path)

        # Verify interactions
        mock_reader.read.assert_called_once_with(labeled_path)
        mock_validator.validate_labeled_data.assert_called_once_with([{"text": "Sample text", "label": "1"}])
        assert result == [{"text": "Sample text", "label": "1"}]

    def test_save_classification_results(self) -> None:
        """Test that CsvProcessor saves classification results correctly."""
        # Mock dependencies
        mock_writer = Mock()

        processor = CsvProcessor(csv_writer=mock_writer)

        # Test saving results
        output_path = Path("output.csv")
        results = [{"text": "Sample text", "prediction": "1", "confidence": "0.95"}]

        processor.save_classification_results(output_path, results)

        # Verify interactions
        mock_writer.write.assert_called_once_with(output_path, results)
