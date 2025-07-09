"""Unit tests for model registry with SOLID design."""

from unittest.mock import Mock

import pytest

from disinfo_relation_checker.model_registry import (
    ModelMetadata,
    ModelRegistry,
    ModelVersion,
    PerformanceMetrics,
)


class TestModelMetadata:
    """Test ModelMetadata following single responsibility principle."""

    def test_model_metadata_creation(self) -> None:
        """Test ModelMetadata creation and properties."""
        metrics = PerformanceMetrics(accuracy=0.85, precision=0.82, recall=0.88, f1=0.85)

        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            description="Test model for unit testing",
            prompt_template="Classify: {text}",
            llm_config={"provider_type": "mock"},
            performance=metrics,
            created_at="2025-01-01T00:00:00Z",
            tags=["classification", "disinformation"],
        )

        assert metadata.name == "test_model"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Test model for unit testing"
        assert metadata.prompt_template == "Classify: {text}"
        assert metadata.llm_config == {"provider_type": "mock"}
        assert metadata.performance.accuracy == 0.85
        assert metadata.created_at == "2025-01-01T00:00:00Z"
        assert metadata.tags == ["classification", "disinformation"]

    def test_model_metadata_serialization(self) -> None:
        """Test ModelMetadata serialization to dictionary."""
        metrics = PerformanceMetrics(accuracy=0.85, precision=0.82, recall=0.88, f1=0.85)

        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            description="Test model",
            prompt_template="Classify: {text}",
            llm_config={"provider_type": "mock"},
            performance=metrics,
            created_at="2025-01-01T00:00:00Z",
            tags=["test"],
        )

        data = metadata.model_dump()

        assert data["name"] == "test_model"
        assert data["version"] == "1.0.0"
        assert data["performance"]["accuracy"] == 0.85
        assert data["llm_config"]["provider_type"] == "mock"

    def test_model_metadata_from_dict(self) -> None:
        """Test ModelMetadata creation from dictionary."""
        data = {
            "name": "test_model",
            "version": "1.0.0",
            "description": "Test model",
            "prompt_template": "Classify: {text}",
            "llm_config": {"provider_type": "mock"},
            "performance": {"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1": 0.85},
            "created_at": "2025-01-01T00:00:00Z",
            "tags": ["test"],
        }

        metadata = ModelMetadata.model_validate(data)

        assert metadata.name == "test_model"
        assert metadata.version == "1.0.0"
        assert metadata.performance.accuracy == 0.85


class TestModelVersion:
    """Test ModelVersion following single responsibility principle."""

    def test_model_version_parsing(self) -> None:
        """Test ModelVersion parsing from string."""
        version = ModelVersion.from_string("1.2.3")

        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_model_version_comparison(self) -> None:
        """Test ModelVersion comparison operations."""
        v1 = ModelVersion(major=1, minor=0, patch=0)
        v2 = ModelVersion(major=1, minor=0, patch=1)
        v3 = ModelVersion(major=1, minor=1, patch=0)

        assert v1 < v2
        assert v2 < v3
        assert v1 < v3
        assert v2 > v1

    def test_model_version_string_representation(self) -> None:
        """Test ModelVersion string representation."""
        version = ModelVersion(major=1, minor=2, patch=3)
        assert str(version) == "1.2.3"

    def test_model_version_equality(self) -> None:
        """Test ModelVersion equality."""
        v1 = ModelVersion(major=1, minor=0, patch=0)
        v2 = ModelVersion(major=1, minor=0, patch=0)
        v3 = ModelVersion(major=1, minor=0, patch=1)

        assert v1 == v2
        assert v1 != v3


class TestModelRegistry:
    """Test ModelRegistry with dependency injection."""

    def test_registry_initialization(self) -> None:
        """Test ModelRegistry initialization."""
        mock_storage = Mock()
        registry = ModelRegistry(storage=mock_storage)

        assert registry._storage == mock_storage

    def test_register_model(self) -> None:
        """Test model registration."""
        mock_storage = Mock()
        # Mock get_model to return None (model doesn't exist)
        mock_storage.get_model.return_value = None

        registry = ModelRegistry(storage=mock_storage)

        metrics = PerformanceMetrics(accuracy=0.85, precision=0.82, recall=0.88, f1=0.85)
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            description="Test model",
            prompt_template="Classify: {text}",
            llm_config={"provider_type": "mock"},
            performance=metrics,
            created_at="2025-01-01T00:00:00Z",
            tags=["test"],
        )

        registry.register_model(metadata)

        # Verify storage was called
        mock_storage.get_model.assert_called_once_with("test_model", "1.0.0")
        mock_storage.save_model.assert_called_once_with(metadata)

    def test_get_model(self) -> None:
        """Test model retrieval."""
        mock_storage = Mock()
        registry = ModelRegistry(storage=mock_storage)

        metrics = PerformanceMetrics(accuracy=0.85, precision=0.82, recall=0.88, f1=0.85)
        expected_metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            description="Test model",
            prompt_template="Classify: {text}",
            llm_config={"provider_type": "mock"},
            performance=metrics,
            created_at="2025-01-01T00:00:00Z",
            tags=["test"],
        )

        mock_storage.get_model.return_value = expected_metadata

        result = registry.get_model("test_model", "1.0.0")

        assert result == expected_metadata
        mock_storage.get_model.assert_called_once_with("test_model", "1.0.0")

    def test_list_models(self) -> None:
        """Test listing all models."""
        mock_storage = Mock()
        registry = ModelRegistry(storage=mock_storage)

        metrics = PerformanceMetrics(accuracy=0.85, precision=0.82, recall=0.88, f1=0.85)
        expected_models = [
            ModelMetadata(
                name="model1",
                version="1.0.0",
                description="Model 1",
                prompt_template="Template 1",
                llm_config={"provider_type": "mock"},
                performance=metrics,
                created_at="2025-01-01T00:00:00Z",
                tags=["test"],
            ),
            ModelMetadata(
                name="model2",
                version="1.0.0",
                description="Model 2",
                prompt_template="Template 2",
                llm_config={"provider_type": "mock"},
                performance=metrics,
                created_at="2025-01-01T00:00:00Z",
                tags=["test"],
            ),
        ]

        mock_storage.list_models.return_value = expected_models

        result = registry.list_models()

        assert result == expected_models
        mock_storage.list_models.assert_called_once()

    def test_get_model_versions(self) -> None:
        """Test getting all versions of a model."""
        mock_storage = Mock()
        registry = ModelRegistry(storage=mock_storage)

        metrics = PerformanceMetrics(accuracy=0.85, precision=0.82, recall=0.88, f1=0.85)
        expected_versions = [
            ModelMetadata(
                name="test_model",
                version="1.0.0",
                description="Version 1.0.0",
                prompt_template="Template 1",
                llm_config={"provider_type": "mock"},
                performance=metrics,
                created_at="2025-01-01T00:00:00Z",
                tags=["test"],
            ),
            ModelMetadata(
                name="test_model",
                version="1.1.0",
                description="Version 1.1.0",
                prompt_template="Template 2",
                llm_config={"provider_type": "mock"},
                performance=metrics,
                created_at="2025-01-01T00:00:00Z",
                tags=["test"],
            ),
        ]

        mock_storage.get_model_versions.return_value = expected_versions

        result = registry.get_model_versions("test_model")

        assert result == expected_versions
        mock_storage.get_model_versions.assert_called_once_with("test_model")

    def test_get_latest_version(self) -> None:
        """Test getting the latest version of a model."""
        mock_storage = Mock()
        registry = ModelRegistry(storage=mock_storage)

        metrics = PerformanceMetrics(accuracy=0.85, precision=0.82, recall=0.88, f1=0.85)
        versions = [
            ModelMetadata(
                name="test_model",
                version="1.0.0",
                description="Version 1.0.0",
                prompt_template="Template 1",
                llm_config={"provider_type": "mock"},
                performance=metrics,
                created_at="2025-01-01T00:00:00Z",
                tags=["test"],
            ),
            ModelMetadata(
                name="test_model",
                version="1.1.0",
                description="Version 1.1.0",
                prompt_template="Template 2",
                llm_config={"provider_type": "mock"},
                performance=metrics,
                created_at="2025-01-01T00:00:00Z",
                tags=["test"],
            ),
        ]

        mock_storage.get_model_versions.return_value = versions

        result = registry.get_latest_version("test_model")

        assert result == versions[1]  # Latest version (1.1.0)
        mock_storage.get_model_versions.assert_called_once_with("test_model")


class TestPerformanceMetrics:
    """Test PerformanceMetrics following single responsibility principle."""

    def test_performance_metrics_creation(self) -> None:
        """Test PerformanceMetrics creation and properties."""
        metrics = PerformanceMetrics(accuracy=0.85, precision=0.82, recall=0.88, f1=0.85)

        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.82
        assert metrics.recall == 0.88
        assert metrics.f1 == 0.85

    def test_performance_metrics_validation(self) -> None:
        """Test PerformanceMetrics validation."""
        with pytest.raises(ValueError):
            PerformanceMetrics(accuracy=1.5, precision=0.82, recall=0.88, f1=0.85)

        with pytest.raises(ValueError):
            PerformanceMetrics(accuracy=0.85, precision=-0.1, recall=0.88, f1=0.85)

    def test_performance_metrics_comparison(self) -> None:
        """Test PerformanceMetrics comparison based on F1 score."""
        metrics1 = PerformanceMetrics(accuracy=0.85, precision=0.82, recall=0.88, f1=0.85)
        metrics2 = PerformanceMetrics(accuracy=0.90, precision=0.87, recall=0.92, f1=0.90)

        assert metrics1 < metrics2
        assert metrics2 > metrics1

    def test_performance_metrics_to_dict(self) -> None:
        """Test PerformanceMetrics serialization to dictionary."""
        metrics = PerformanceMetrics(accuracy=0.85, precision=0.82, recall=0.88, f1=0.85)

        data = metrics.model_dump()

        assert data["accuracy"] == 0.85
        assert data["precision"] == 0.82
        assert data["recall"] == 0.88
        assert data["f1"] == 0.85

    def test_performance_metrics_from_dict(self) -> None:
        """Test PerformanceMetrics creation from dictionary."""
        data = {"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1": 0.85}

        metrics = PerformanceMetrics.model_validate(data)

        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.82
        assert metrics.recall == 0.88
        assert metrics.f1 == 0.85
