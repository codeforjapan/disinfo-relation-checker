"""Unit tests for A/B testing framework with SOLID design."""

from unittest.mock import Mock

import pytest

from disinfo_relation_checker.ab_testing import (
    ABTest,
    ABTestConfig,
    ABTestResult,
    ABTestRunner,
    ABTestStatus,
    TrafficSplitter,
)


class TestABTestConfig:
    """Test ABTestConfig following single responsibility principle."""

    def test_ab_test_config_creation(self) -> None:
        """Test ABTestConfig creation and properties."""
        config = ABTestConfig(
            test_name="prompt_comparison",
            model_a="test_model:1.0.0",
            model_b="test_model:1.1.0",
            traffic_split=50,
            test_data_path="/tmp/test_data.csv",
            created_at="2025-01-01T00:00:00Z",
            status=ABTestStatus.ACTIVE,
        )

        assert config.test_name == "prompt_comparison"
        assert config.model_a == "test_model:1.0.0"
        assert config.model_b == "test_model:1.1.0"
        assert config.traffic_split == 50
        assert config.test_data_path == "/tmp/test_data.csv"
        assert config.created_at == "2025-01-01T00:00:00Z"
        assert config.status == "active"

    def test_ab_test_config_validation(self) -> None:
        """Test ABTestConfig validation."""
        with pytest.raises(ValueError):
            ABTestConfig(
                test_name="test",
                model_a="model_a:1.0.0",
                model_b="model_b:1.0.0",
                traffic_split=150,  # Invalid traffic split
                test_data_path="/tmp/test.csv",
                created_at="2025-01-01T00:00:00Z",
                status=ABTestStatus.ACTIVE,
            )

    def test_ab_test_config_serialization(self) -> None:
        """Test ABTestConfig serialization to dictionary."""
        config = ABTestConfig(
            test_name="test",
            model_a="model_a:1.0.0",
            model_b="model_b:1.0.0",
            traffic_split=50,
            test_data_path="/tmp/test.csv",
            created_at="2025-01-01T00:00:00Z",
            status=ABTestStatus.ACTIVE,
        )

        data = config.model_dump()

        assert data["test_name"] == "test"
        assert data["model_a"] == "model_a:1.0.0"
        assert data["traffic_split"] == 50

    def test_ab_test_config_from_dict(self) -> None:
        """Test ABTestConfig creation from dictionary."""
        data = {
            "test_name": "test",
            "model_a": "model_a:1.0.0",
            "model_b": "model_b:1.0.0",
            "traffic_split": 50,
            "test_data_path": "/tmp/test.csv",
            "created_at": "2025-01-01T00:00:00Z",
            "status": "active",
        }

        config = ABTestConfig.model_validate(data)

        assert config.test_name == "test"
        assert config.model_a == "model_a:1.0.0"
        assert config.traffic_split == 50


class TestABTestResult:
    """Test ABTestResult following single responsibility principle."""

    def test_ab_test_result_creation(self) -> None:
        """Test ABTestResult creation and properties."""
        result = ABTestResult(
            test_name="prompt_comparison",
            model_a_performance={"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1": 0.85},
            model_b_performance={"accuracy": 0.87, "precision": 0.84, "recall": 0.90, "f1": 0.87},
            sample_size_a=100,
            sample_size_b=100,
            statistical_significance=0.05,
            winner="model_b",
            completed_at="2025-01-01T00:00:00Z",
        )

        assert result.test_name == "prompt_comparison"
        assert result.model_a_performance["accuracy"] == 0.85
        assert result.model_b_performance["accuracy"] == 0.87
        assert result.sample_size_a == 100
        assert result.sample_size_b == 100
        assert result.statistical_significance == 0.05
        assert result.winner == "model_b"
        assert result.completed_at == "2025-01-01T00:00:00Z"

    def test_ab_test_result_determine_winner(self) -> None:
        """Test ABTestResult winner determination."""
        result = ABTestResult(
            test_name="test",
            model_a_performance={"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1": 0.85},
            model_b_performance={"accuracy": 0.87, "precision": 0.84, "recall": 0.90, "f1": 0.87},
            sample_size_a=100,
            sample_size_b=100,
            statistical_significance=0.05,
            winner="",
            completed_at="2025-01-01T00:00:00Z",
        )

        winner = result.determine_winner()
        assert winner == "model_b"

    def test_ab_test_result_serialization(self) -> None:
        """Test ABTestResult serialization to dictionary."""
        result = ABTestResult(
            test_name="test",
            model_a_performance={"accuracy": 0.85, "f1": 0.85},
            model_b_performance={"accuracy": 0.87, "f1": 0.87},
            sample_size_a=100,
            sample_size_b=100,
            statistical_significance=0.05,
            winner="model_b",
            completed_at="2025-01-01T00:00:00Z",
        )

        data = result.model_dump()

        assert data["test_name"] == "test"
        assert data["model_a_performance"]["accuracy"] == 0.85
        assert data["winner"] == "model_b"


class TestTrafficSplitter:
    """Test TrafficSplitter following single responsibility principle."""

    def test_traffic_splitter_initialization(self) -> None:
        """Test TrafficSplitter initialization."""
        splitter = TrafficSplitter(split_percentage=50, random_seed=42)

        assert splitter._split_percentage == 50
        assert splitter._random_seed == 42

    def test_traffic_splitter_assign_group(self) -> None:
        """Test TrafficSplitter group assignment."""
        splitter = TrafficSplitter(split_percentage=50, random_seed=42)

        # Test with consistent seed for predictable results
        group1 = splitter.assign_group("user_123")
        group2 = splitter.assign_group("user_123")  # Same user should get same group

        assert group1 == group2
        assert group1 in ["A", "B"]

    def test_traffic_splitter_distribution(self) -> None:
        """Test TrafficSplitter distribution across many users."""
        splitter = TrafficSplitter(split_percentage=50, random_seed=42)

        groups = []
        for i in range(1000):
            group = splitter.assign_group(f"user_{i}")
            groups.append(group)

        group_a_count = groups.count("A")
        group_b_count = groups.count("B")

        # Should be roughly 50-50 split (within reasonable tolerance)
        assert abs(group_a_count - 500) < 100
        assert abs(group_b_count - 500) < 100

    def test_traffic_splitter_different_splits(self) -> None:
        """Test TrafficSplitter with different split percentages."""
        splitter_70 = TrafficSplitter(split_percentage=70, random_seed=42)
        splitter_30 = TrafficSplitter(split_percentage=30, random_seed=42)

        groups_70 = []
        groups_30 = []

        for i in range(1000):
            group_70 = splitter_70.assign_group(f"user_{i}")
            group_30 = splitter_30.assign_group(f"user_{i}")
            groups_70.append(group_70)
            groups_30.append(group_30)

        group_a_count_70 = groups_70.count("A")
        group_a_count_30 = groups_30.count("A")

        # 70% splitter should have more A assignments
        assert group_a_count_70 > group_a_count_30


class TestABTestRunner:
    """Test ABTestRunner with dependency injection."""

    def test_ab_test_runner_initialization(self) -> None:
        """Test ABTestRunner initialization."""
        mock_model_registry = Mock()
        mock_classifier = Mock()
        mock_storage = Mock()

        runner = ABTestRunner(model_registry=mock_model_registry, classifier=mock_classifier, storage=mock_storage)

        assert runner._model_registry == mock_model_registry
        assert runner._classifier == mock_classifier
        assert runner._storage == mock_storage

    def test_setup_ab_test(self) -> None:
        """Test A/B test setup."""
        mock_model_registry = Mock()
        mock_classifier = Mock()
        mock_storage = Mock()

        runner = ABTestRunner(model_registry=mock_model_registry, classifier=mock_classifier, storage=mock_storage)

        config = ABTestConfig(
            test_name="test",
            model_a="model_a:1.0.0",
            model_b="model_b:1.0.0",
            traffic_split=50,
            test_data_path="/tmp/test.csv",
            created_at="2025-01-01T00:00:00Z",
            status=ABTestStatus.ACTIVE,
        )

        runner.setup_ab_test(config)

        # Verify storage was called
        mock_storage.save_ab_test_config.assert_called_once_with(config)

    def test_run_ab_test(self) -> None:
        """Test A/B test execution."""
        mock_model_registry = Mock()
        mock_classifier = Mock()
        mock_storage = Mock()

        runner = ABTestRunner(model_registry=mock_model_registry, classifier=mock_classifier, storage=mock_storage)

        config = ABTestConfig(
            test_name="test",
            model_a="model_a:1.0.0",
            model_b="model_b:1.0.0",
            traffic_split=50,
            test_data_path="/tmp/test.csv",
            created_at="2025-01-01T00:00:00Z",
            status=ABTestStatus.ACTIVE,
        )

        # Mock test data
        test_data = [{"text": "Test text 1", "label": "1"}, {"text": "Test text 2", "label": "0"}]

        mock_storage.get_ab_test_config.return_value = config
        mock_classifier.validate.return_value = {"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1": 0.85}

        result = runner.run_ab_test("test", test_data)

        assert result.test_name == "test"
        assert result.model_a_performance["accuracy"] == 0.85
        assert result.model_b_performance["accuracy"] == 0.85
        mock_storage.save_ab_test_result.assert_called_once()

    def test_get_ab_test_result(self) -> None:
        """Test A/B test result retrieval."""
        mock_model_registry = Mock()
        mock_classifier = Mock()
        mock_storage = Mock()

        runner = ABTestRunner(model_registry=mock_model_registry, classifier=mock_classifier, storage=mock_storage)

        expected_result = ABTestResult(
            test_name="test",
            model_a_performance={"accuracy": 0.85, "f1": 0.85},
            model_b_performance={"accuracy": 0.87, "f1": 0.87},
            sample_size_a=100,
            sample_size_b=100,
            statistical_significance=0.05,
            winner="model_b",
            completed_at="2025-01-01T00:00:00Z",
        )

        mock_storage.get_ab_test_result.return_value = expected_result

        result = runner.get_ab_test_result("test")

        assert result == expected_result
        mock_storage.get_ab_test_result.assert_called_once_with("test")

    def test_list_ab_tests(self) -> None:
        """Test listing all A/B tests."""
        mock_model_registry = Mock()
        mock_classifier = Mock()
        mock_storage = Mock()

        runner = ABTestRunner(model_registry=mock_model_registry, classifier=mock_classifier, storage=mock_storage)

        expected_configs = [
            ABTestConfig(
                test_name="test1",
                model_a="model_a:1.0.0",
                model_b="model_b:1.0.0",
                traffic_split=50,
                test_data_path="/tmp/test1.csv",
                created_at="2025-01-01T00:00:00Z",
                status=ABTestStatus.ACTIVE,
            ),
            ABTestConfig(
                test_name="test2",
                model_a="model_a:1.0.0",
                model_b="model_b:1.0.0",
                traffic_split=70,
                test_data_path="/tmp/test2.csv",
                created_at="2025-01-01T00:00:00Z",
                status=ABTestStatus.COMPLETED,
            ),
        ]

        mock_storage.list_ab_tests.return_value = expected_configs

        result = runner.list_ab_tests()

        assert result == expected_configs
        mock_storage.list_ab_tests.assert_called_once()


class TestABTest:
    """Test ABTest integration following single responsibility principle."""

    def test_ab_test_creation(self) -> None:
        """Test ABTest creation with config and result."""
        config = ABTestConfig(
            test_name="test",
            model_a="model_a:1.0.0",
            model_b="model_b:1.0.0",
            traffic_split=50,
            test_data_path="/tmp/test.csv",
            created_at="2025-01-01T00:00:00Z",
            status=ABTestStatus.ACTIVE,
        )

        result = ABTestResult(
            test_name="test",
            model_a_performance={"accuracy": 0.85, "f1": 0.85},
            model_b_performance={"accuracy": 0.87, "f1": 0.87},
            sample_size_a=100,
            sample_size_b=100,
            statistical_significance=0.05,
            winner="model_b",
            completed_at="2025-01-01T00:00:00Z",
        )

        ab_test = ABTest(config=config, result=result)

        assert ab_test.config == config
        assert ab_test.result == result

    def test_ab_test_is_significant(self) -> None:
        """Test ABTest statistical significance check."""
        config = ABTestConfig(
            test_name="test",
            model_a="model_a:1.0.0",
            model_b="model_b:1.0.0",
            traffic_split=50,
            test_data_path="/tmp/test.csv",
            created_at="2025-01-01T00:00:00Z",
            status=ABTestStatus.ACTIVE,
        )

        result_significant = ABTestResult(
            test_name="test",
            model_a_performance={"accuracy": 0.85, "f1": 0.85},
            model_b_performance={"accuracy": 0.87, "f1": 0.87},
            sample_size_a=100,
            sample_size_b=100,
            statistical_significance=0.03,  # < 0.05
            winner="model_b",
            completed_at="2025-01-01T00:00:00Z",
        )

        ab_test = ABTest(config=config, result=result_significant)
        assert ab_test.is_significant() is True

        result_not_significant = ABTestResult(
            test_name="test",
            model_a_performance={"accuracy": 0.85, "f1": 0.85},
            model_b_performance={"accuracy": 0.87, "f1": 0.87},
            sample_size_a=100,
            sample_size_b=100,
            statistical_significance=0.10,  # > 0.05
            winner="model_b",
            completed_at="2025-01-01T00:00:00Z",
        )

        ab_test = ABTest(config=config, result=result_not_significant)
        assert ab_test.is_significant() is False

    def test_ab_test_summary(self) -> None:
        """Test ABTest summary generation."""
        config = ABTestConfig(
            test_name="test",
            model_a="model_a:1.0.0",
            model_b="model_b:1.0.0",
            traffic_split=50,
            test_data_path="/tmp/test.csv",
            created_at="2025-01-01T00:00:00Z",
            status=ABTestStatus.ACTIVE,
        )

        result = ABTestResult(
            test_name="test",
            model_a_performance={"accuracy": 0.85, "f1": 0.85},
            model_b_performance={"accuracy": 0.87, "f1": 0.87},
            sample_size_a=100,
            sample_size_b=100,
            statistical_significance=0.03,
            winner="model_b",
            completed_at="2025-01-01T00:00:00Z",
        )

        ab_test = ABTest(config=config, result=result)
        summary = ab_test.summary()

        assert "test" in summary
        assert "model_b" in summary
        assert "0.87" in summary
