"""CSV processing module with SOLID design principles."""

import csv
from pathlib import Path


class CsvReader:
    """Handles CSV file reading operations."""

    def read(self, file_path: Path) -> list[dict[str, str]]:
        """Read CSV file and return list of dictionaries."""
        data: list[dict[str, str]] = []
        with file_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data.extend(dict(row) for row in reader)
        return data


class CsvWriter:
    """Handles CSV file writing operations."""

    def write(self, file_path: Path, data: list[dict[str, str]]) -> None:
        """Write list of dictionaries to CSV file."""
        if not data:
            return

        fieldnames = list(data[0].keys())
        with file_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


class DataValidator:
    """Validates data structure and content."""

    def validate_input_data(self, data: list[dict[str, str]]) -> bool:
        """Validate input data has required text column."""
        if not data:
            return False

        return all("text" in row for row in data)

    def validate_labeled_data(self, data: list[dict[str, str]]) -> bool:
        """Validate labeled data has required text and label columns."""
        if not data:
            return False

        return all(not ("text" not in row or "label" not in row) for row in data)


class CsvProcessor:
    """Main CSV processor with dependency injection."""

    def __init__(
        self,
        csv_reader: CsvReader | None = None,
        csv_writer: CsvWriter | None = None,
        data_validator: DataValidator | None = None,
    ) -> None:
        """Initialize with dependencies."""
        self._csv_reader = csv_reader or CsvReader()
        self._csv_writer = csv_writer or CsvWriter()
        self._data_validator = data_validator or DataValidator()

    def read_input_data(self, file_path: Path) -> list[dict[str, str]]:
        """Read and validate input data for classification."""
        data = self._csv_reader.read(file_path)

        if not self._data_validator.validate_input_data(data):
            msg = "Invalid input data: missing 'text' column"
            raise ValueError(msg)

        return data

    def read_labeled_data(self, file_path: Path) -> list[dict[str, str]]:
        """Read and validate labeled data for validation."""
        data = self._csv_reader.read(file_path)

        if not self._data_validator.validate_labeled_data(data):
            msg = "Invalid labeled data: missing 'text' or 'label' column"
            raise ValueError(msg)

        return data

    def save_classification_results(
        self,
        file_path: Path,
        results: list[dict[str, str]],
    ) -> None:
        """Save classification results to CSV file."""
        self._csv_writer.write(file_path, results)
