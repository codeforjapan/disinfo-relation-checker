"""Training data management with SOLID design principles."""

import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .csv_processor import CsvProcessor


class ValidationDataset(BaseModel):
    """Represents a validation dataset split."""

    train_data: list[dict[str, Any]]
    validation_data: list[dict[str, Any]]
    test_data: list[dict[str, Any]]

    @property
    def train_size(self) -> int:
        """Size of training set."""
        return len(self.train_data)

    @property
    def validation_size(self) -> int:
        """Size of validation set."""
        return len(self.validation_data)

    @property
    def test_size(self) -> int:
        """Size of test set."""
        return len(self.test_data)

    @property
    def total_size(self) -> int:
        """Total size of all sets."""
        return self.train_size + self.validation_size + self.test_size


class DataSplitter:
    """Handles data splitting following single responsibility principle."""

    def random_split(
        self,
        data: list[dict[str, Any]],
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
        random_seed: int | None = None,
    ) -> ValidationDataset:
        """Split data randomly into train/validation/test sets."""
        if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
            msg = "Ratios must sum to 1.0"
            raise ValueError(msg)

        if random_seed is not None:
            random.seed(random_seed)

        # Shuffle data
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)

        # Calculate split indices
        total_size = len(data)
        train_size = int(total_size * train_ratio)
        validation_size = int(total_size * validation_ratio)

        # Split data
        train_data = shuffled_data[:train_size]
        validation_data = shuffled_data[train_size : train_size + validation_size]
        test_data = shuffled_data[train_size + validation_size :]

        return ValidationDataset(train_data=train_data, validation_data=validation_data, test_data=test_data)

    def stratified_split(
        self,
        data: list[dict[str, Any]],
        train_ratio: float,
        validation_ratio: float,
        test_ratio: float,
        label_key: str = "label",
        random_seed: int | None = None,
    ) -> ValidationDataset:
        """Split data maintaining label distribution."""
        if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
            msg = "Ratios must sum to 1.0"
            raise ValueError(msg)

        if random_seed is not None:
            random.seed(random_seed)

        # Group by labels
        label_groups: dict[str, list[dict[str, Any]]] = {}
        for item in data:
            label = item[label_key]
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(item)

        # Split each label group
        train_data: list[dict[str, Any]] = []
        validation_data: list[dict[str, Any]] = []
        test_data: list[dict[str, Any]] = []

        for label, group_data in label_groups.items():
            random.shuffle(group_data)

            group_size = len(group_data)
            train_size = int(group_size * train_ratio)
            validation_size = int(group_size * validation_ratio)

            train_data.extend(group_data[:train_size])
            validation_data.extend(group_data[train_size : train_size + validation_size])
            test_data.extend(group_data[train_size + validation_size :])

        # Shuffle final sets
        random.shuffle(train_data)
        random.shuffle(validation_data)
        random.shuffle(test_data)

        return ValidationDataset(train_data=train_data, validation_data=validation_data, test_data=test_data)

    def cross_validation_split(
        self, data: list[dict[str, Any]], k: int, random_seed: int | None = None
    ) -> list[list[dict[str, Any]]]:
        """Split data into k folds for cross-validation."""
        if k <= 1:
            msg = "k must be greater than 1"
            raise ValueError(msg)

        if random_seed is not None:
            random.seed(random_seed)

        # Shuffle data
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)

        # Calculate fold sizes
        fold_size = len(data) // k
        remainder = len(data) % k

        # Create folds
        folds = []
        start_idx = 0

        for i in range(k):
            # Add one extra item to first 'remainder' folds
            current_fold_size = fold_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_fold_size

            folds.append(shuffled_data[start_idx:end_idx])
            start_idx = end_idx

        return folds


class DataValidator:
    """Validates data quality following single responsibility principle."""

    def is_balanced(self, data: list[dict[str, Any]], threshold: float = 0.2, label_key: str = "label") -> bool:
        """Check if dataset is balanced within threshold."""
        if not data:
            return False

        # Count labels
        label_counts: dict[str, int] = {}
        for item in data:
            label = item[label_key]
            label_counts[label] = label_counts.get(label, 0) + 1

        if len(label_counts) < 2:
            return False

        # Calculate balance
        total_count = len(data)
        proportions = [count / total_count for count in label_counts.values()]

        # Check if all proportions are within threshold of 0.5
        target_proportion = 1.0 / len(label_counts)
        return all(abs(prop - target_proportion) <= threshold for prop in proportions)

    def has_minimum_samples_per_class(
        self, data: list[dict[str, Any]], min_samples: int, label_key: str = "label"
    ) -> bool:
        """Check if each class has minimum number of samples."""
        if not data:
            return False

        # Count labels
        label_counts: dict[str, int] = {}
        for item in data:
            label = item[label_key]
            label_counts[label] = label_counts.get(label, 0) + 1

        return all(count >= min_samples for count in label_counts.values())

    def validate_data_quality(
        self, data: list[dict[str, Any]], text_key: str = "text", label_key: str = "label"
    ) -> list[str]:
        """Validate comprehensive data quality and return issues."""
        issues = []

        if not data:
            issues.append("Dataset is empty")
            return issues

        # Check for empty texts
        empty_texts = sum(1 for item in data if not item.get(text_key, "").strip())
        if empty_texts > 0:
            issues.append(f"Found {empty_texts} empty text entries")

        # Check for very short texts
        short_texts = sum(1 for item in data if len(item.get(text_key, "").strip()) < 5)
        if short_texts > len(data) * 0.1:  # More than 10% are very short
            issues.append(f"Found {short_texts} very short text entries")

        # Check label distribution
        if not self.is_balanced(data, threshold=0.3, label_key=label_key):
            issues.append("Dataset is significantly imbalanced")

        # Check minimum samples per class
        if not self.has_minimum_samples_per_class(data, min_samples=2, label_key=label_key):
            issues.append("Some classes have insufficient samples")

        # Check for missing labels
        missing_labels = sum(1 for item in data if label_key not in item or item[label_key] is None)
        if missing_labels > 0:
            issues.append(f"Found {missing_labels} entries with missing labels")

        return issues


class TrainingDataManager:
    """Manages training data operations with dependency injection."""

    def __init__(
        self,
        csv_processor: CsvProcessor | None = None,
        data_splitter: DataSplitter | None = None,
        data_validator: DataValidator | None = None,
    ) -> None:
        """Initialize with injected dependencies."""
        self._csv_processor = csv_processor or CsvProcessor()
        self._data_splitter = data_splitter or DataSplitter()
        self._data_validator = data_validator or DataValidator()

    def load_and_split_data(
        self,
        file_path: Path,
        train_ratio: float = 0.7,
        validation_ratio: float = 0.2,
        test_ratio: float = 0.1,
        stratified: bool = True,
    ) -> ValidationDataset:
        """Load data from file and split into train/validation/test sets."""
        # Load data
        data = self._csv_processor.read_labeled_data(file_path)

        # Validate data quality
        issues = self._data_validator.validate_data_quality(data)
        if issues:
            msg = f"Data quality issues found: {', '.join(issues)}"
            raise ValueError(msg)

        # Split data
        if stratified:
            return self._data_splitter.stratified_split(data, train_ratio, validation_ratio, test_ratio)
        return self._data_splitter.random_split(data, train_ratio, validation_ratio, test_ratio)

    def prepare_cross_validation_data(self, file_path: Path, k: int = 5) -> list[list[dict[str, Any]]]:
        """Prepare data for k-fold cross-validation."""
        # Load data
        data = self._csv_processor.read_labeled_data(file_path)

        # Validate data quality
        issues = self._data_validator.validate_data_quality(data)
        if issues:
            msg = f"Data quality issues found: {', '.join(issues)}"
            raise ValueError(msg)

        # Create cross-validation folds
        return self._data_splitter.cross_validation_split(data, k)
