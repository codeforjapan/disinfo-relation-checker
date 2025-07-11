"""CSV processing module for reading and writing CSV files."""

import csv
from pathlib import Path
from typing import Any


class CSVProcessorImpl:
    """Implementation of CSV processing operations."""

    def read_csv(self, file_path: Path) -> list[dict[str, Any]]:
        """Read CSV file and return list of records."""
        if not file_path.exists():
            msg = f"CSV file not found: {file_path}"
            raise FileNotFoundError(msg)

        with file_path.open("r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            return [dict(row) for row in reader]

    def write_csv(self, file_path: Path, data: list[dict[str, Any]]) -> None:
        """Write data to CSV file."""
        if not data:
            # Write empty file with no headers if no data
            file_path.write_text("", encoding="utf-8")
            return

        # Get all unique column names from the data
        all_columns: set[str] = set()
        for row in data:
            all_columns.update(row.keys())

        # Sort columns for consistent output (text, source first if present)
        column_order = []
        priority_columns = ["text", "source", "classification", "confidence"]

        for col in priority_columns:
            if col in all_columns:
                column_order.append(col)
                all_columns.remove(col)

        # Add remaining columns alphabetically
        column_order.extend(sorted(all_columns))

        with file_path.open("w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=column_order)
            writer.writeheader()
            writer.writerows(data)
