"""Unit tests for training data management with SOLID design."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from disinfo_relation_checker.training_data import (
    DataSplitter,
    DataValidator,
    TrainingDataManager,
    ValidationDataset,
)


class TestValidationDataset:
    """Test ValidationDataset following single responsibility principle."""

    def test_dataset_creation(self) -> None:
        """Test ValidationDataset creation and properties."""
        train_data = [{"text": "train1", "label": "1"}]
        val_data = [{"text": "val1", "label": "0"}]
        test_data = [{"text": "test1", "label": "1"}]

        dataset = ValidationDataset(train_data=train_data, validation_data=val_data, test_data=test_data)

        assert dataset.train_data == train_data
        assert dataset.validation_data == val_data
        assert dataset.test_data == test_data

    def test_dataset_sizes(self) -> None:
        """Test ValidationDataset size calculation."""
        dataset = ValidationDataset(
            train_data=[{"text": f"train{i}", "label": "1"} for i in range(10)],
            validation_data=[{"text": f"val{i}", "label": "0"} for i in range(3)],
            test_data=[{"text": f"test{i}", "label": "1"} for i in range(2)],
        )

        assert dataset.train_size == 10
        assert dataset.validation_size == 3
        assert dataset.test_size == 2
        assert dataset.total_size == 15


class TestDataSplitter:
    """Test DataSplitter following single responsibility principle."""

    def test_random_split_proportions(self) -> None:
        """Test random split maintains correct proportions."""
        splitter = DataSplitter()
        data = [{"text": f"text{i}", "label": str(i % 2)} for i in range(100)]

        dataset = splitter.random_split(data, train_ratio=0.7, validation_ratio=0.2, test_ratio=0.1)

        assert dataset.train_size == 70
        assert dataset.validation_size == 20
        assert dataset.test_size == 10

    def test_random_split_no_overlap(self) -> None:
        """Test random split creates non-overlapping sets."""
        splitter = DataSplitter()
        data = [{"text": f"unique_text_{i}", "label": "1"} for i in range(10)]

        dataset = splitter.random_split(data, 0.6, 0.2, 0.2)

        # Extract all texts from each split
        train_texts = {item["text"] for item in dataset.train_data}
        val_texts = {item["text"] for item in dataset.validation_data}
        test_texts = {item["text"] for item in dataset.test_data}

        # Verify no overlap
        assert len(train_texts & val_texts) == 0
        assert len(train_texts & test_texts) == 0
        assert len(val_texts & test_texts) == 0

        # Verify all data is included
        all_texts = train_texts | val_texts | test_texts
        original_texts = {item["text"] for item in data}
        assert all_texts == original_texts

    def test_stratified_split_maintains_distribution(self) -> None:
        """Test stratified split maintains label distribution."""
        splitter = DataSplitter()

        # Create data with 70% label "1" and 30% label "0"
        data = [{"text": f"pos{i}", "label": "1"} for i in range(70)] + [
            {"text": f"neg{i}", "label": "0"} for i in range(30)
        ]

        dataset = splitter.stratified_split(data, 0.8, 0.1, 0.1)

        # Check label distribution in train set
        train_labels = [item["label"] for item in dataset.train_data]
        train_pos_ratio = train_labels.count("1") / len(train_labels)

        # Should be approximately 70% (within 5% tolerance)
        assert abs(train_pos_ratio - 0.7) < 0.05

    def test_cross_validation_split(self) -> None:
        """Test cross-validation split creates correct folds."""
        splitter = DataSplitter()
        data = [{"text": f"text{i}", "label": str(i % 2)} for i in range(10)]

        folds = splitter.cross_validation_split(data, k=5)

        assert len(folds) == 5

        # Each fold should have 2 items (10 / 5)
        for fold in folds:
            assert len(fold) == 2

        # All folds together should contain all original data
        all_fold_data = []
        for fold in folds:
            all_fold_data.extend(fold)

        assert len(all_fold_data) == len(data)


class TestDataValidator:
    """Test DataValidator following single responsibility principle."""

    def test_validate_balanced_dataset(self) -> None:
        """Test validation of balanced datasets."""
        validator = DataValidator()

        # Balanced dataset
        balanced_data = [
            {"text": "text1", "label": "1"},
            {"text": "text2", "label": "0"},
            {"text": "text3", "label": "1"},
            {"text": "text4", "label": "0"},
        ]

        assert validator.is_balanced(balanced_data, threshold=0.1) is True

    def test_validate_imbalanced_dataset(self) -> None:
        """Test validation of imbalanced datasets."""
        validator = DataValidator()

        # Imbalanced dataset (80% vs 20%)
        imbalanced_data = [{"text": f"pos{i}", "label": "1"} for i in range(8)] + [
            {"text": f"neg{i}", "label": "0"} for i in range(2)
        ]

        assert validator.is_balanced(imbalanced_data, threshold=0.1) is False
        assert validator.is_balanced(imbalanced_data, threshold=0.35) is True

    def test_validate_minimum_samples_per_class(self) -> None:
        """Test validation of minimum samples per class."""
        validator = DataValidator()

        data = [{"text": "text1", "label": "1"}, {"text": "text2", "label": "0"}, {"text": "text3", "label": "1"}]

        assert validator.has_minimum_samples_per_class(data, min_samples=1) is True
        assert validator.has_minimum_samples_per_class(data, min_samples=2) is False

    def test_validate_data_quality(self) -> None:
        """Test comprehensive data quality validation."""
        validator = DataValidator()

        # High quality data
        good_data = [
            {"text": "Clear political statement", "label": "1"},
            {"text": "Normal weather discussion", "label": "0"},
            {"text": "Government corruption claim", "label": "1"},
            {"text": "Casual daily conversation", "label": "0"},
        ]

        issues = validator.validate_data_quality(good_data)
        assert len(issues) == 0

    def test_validate_data_with_issues(self) -> None:
        """Test validation of data with quality issues."""
        validator = DataValidator()

        # Data with issues
        problematic_data = [
            {"text": "", "label": "1"},  # Empty text
            {"text": "a", "label": "0"},  # Too short
            {"text": "Good text here", "label": "1"},
            {"text": "Another good text", "label": "1"},  # Imbalanced
            {"text": "More text", "label": "1"},  # More imbalanced
            {"text": "Even more text", "label": "1"},  # Even more imbalanced
        ]

        issues = validator.validate_data_quality(problematic_data)
        assert len(issues) > 0
        assert any("empty" in issue.lower() for issue in issues)
        assert any("imbalanced" in issue.lower() for issue in issues)


class TestTrainingDataManager:
    """Test TrainingDataManager with dependency injection."""

    def test_load_and_split_data(self) -> None:
        """Test loading and splitting training data."""
        # Mock dependencies
        mock_csv_processor = Mock()
        mock_csv_processor.read_labeled_data.return_value = [
            {"text": f"text{i}", "label": str(i % 2)} for i in range(10)
        ]

        mock_splitter = Mock()
        mock_splitter.stratified_split.return_value = ValidationDataset(
            train_data=[{"text": "train", "label": "1"}],
            validation_data=[{"text": "val", "label": "0"}],
            test_data=[{"text": "test", "label": "1"}],
        )

        mock_validator = Mock()
        mock_validator.validate_data_quality.return_value = []

        manager = TrainingDataManager(
            csv_processor=mock_csv_processor, data_splitter=mock_splitter, data_validator=mock_validator
        )

        # Test loading and splitting
        file_path = Path("test.csv")
        dataset = manager.load_and_split_data(file_path)

        # Verify interactions
        mock_csv_processor.read_labeled_data.assert_called_once_with(file_path)
        mock_validator.validate_data_quality.assert_called_once()
        mock_splitter.stratified_split.assert_called_once()

        assert dataset.train_data == [{"text": "train", "label": "1"}]

    def test_cross_validation_data_preparation(self) -> None:
        """Test cross-validation data preparation."""
        mock_csv_processor = Mock()
        mock_csv_processor.read_labeled_data.return_value = [{"text": f"text{i}", "label": "1"} for i in range(10)]

        mock_splitter = Mock()
        mock_splitter.cross_validation_split.return_value = [
            [{"text": "fold1_item1", "label": "1"}],
            [{"text": "fold2_item1", "label": "0"}],
        ]

        mock_validator = Mock()
        mock_validator.validate_data_quality.return_value = []

        manager = TrainingDataManager(
            csv_processor=mock_csv_processor, data_splitter=mock_splitter, data_validator=mock_validator
        )

        # Test cross-validation preparation
        file_path = Path("test.csv")
        folds = manager.prepare_cross_validation_data(file_path, k=2)

        # Verify interactions
        mock_csv_processor.read_labeled_data.assert_called_once_with(file_path)
        mock_splitter.cross_validation_split.assert_called_once_with(
            mock_csv_processor.read_labeled_data.return_value, 2
        )

        assert len(folds) == 2

    def test_data_quality_validation_failure(self) -> None:
        """Test handling of data quality validation failures."""
        mock_csv_processor = Mock()
        mock_csv_processor.read_labeled_data.return_value = []

        mock_validator = Mock()
        mock_validator.validate_data_quality.return_value = ["Data is empty", "Insufficient samples"]

        manager = TrainingDataManager(csv_processor=mock_csv_processor, data_validator=mock_validator)

        # Test validation failure
        file_path = Path("bad_data.csv")

        with pytest.raises(ValueError, match="Data quality issues found"):
            manager.load_and_split_data(file_path)
