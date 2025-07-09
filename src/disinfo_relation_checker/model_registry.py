"""Model registry and versioning with SOLID design principles."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field


class PerformanceMetrics(BaseModel):
    """Performance metrics for a model."""

    accuracy: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)

    def __lt__(self, other: "PerformanceMetrics") -> bool:
        """Compare metrics based on F1 score."""
        return self.f1 < other.f1

    def __gt__(self, other: "PerformanceMetrics") -> bool:
        """Compare metrics based on F1 score."""
        return self.f1 > other.f1


class ModelVersion(BaseModel):
    """Semantic version for models."""

    major: int = Field(ge=0)
    minor: int = Field(ge=0)
    patch: int = Field(ge=0)

    def __lt__(self, other: "ModelVersion") -> bool:
        """Compare versions."""
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __gt__(self, other: "ModelVersion") -> bool:
        """Compare versions."""
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, ModelVersion):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __str__(self) -> str:
        """Return string representation."""
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def from_string(cls, version_str: str) -> "ModelVersion":
        """Parse version from string."""
        parts = version_str.split(".")
        if len(parts) != 3:
            msg = f"Version must be in format 'major.minor.patch', got '{version_str}'"
            raise ValueError(msg)

        try:
            major, minor, patch = map(int, parts)
        except ValueError as e:
            msg = f"Version parts must be integers, got '{version_str}'"
            raise ValueError(msg) from e

        return cls(major=major, minor=minor, patch=patch)


class ModelMetadata(BaseModel):
    """Metadata for a registered model."""

    name: str
    version: str
    description: str
    prompt_template: str
    llm_config: dict[str, Any]
    performance: PerformanceMetrics
    created_at: str
    tags: list[str]


class ModelStorage(Protocol):
    """Protocol for model storage backends."""

    def save_model(self, metadata: ModelMetadata) -> None:
        """Save model metadata."""
        ...

    def get_model(self, name: str, version: str) -> ModelMetadata | None:
        """Get model metadata by name and version."""
        ...

    def list_models(self) -> list[ModelMetadata]:
        """List all registered models."""
        ...

    def get_model_versions(self, name: str) -> list[ModelMetadata]:
        """Get all versions of a model."""
        ...


class FileModelStorage:
    """File-based model storage implementation."""

    def __init__(self, storage_dir: Path) -> None:
        """Initialize with storage directory."""
        self._storage_dir = storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def get_storage_dir(self) -> Path:
        """Get the storage directory path."""
        return self._storage_dir

    def save_model(self, metadata: ModelMetadata) -> None:
        """Save model metadata to file."""
        model_dir = self._storage_dir / metadata.name
        model_dir.mkdir(exist_ok=True)

        model_file = model_dir / f"{metadata.version}.json"
        with model_file.open("w") as f:
            f.write(metadata.model_dump_json(indent=2))

    def get_model(self, name: str, version: str) -> ModelMetadata | None:
        """Get model metadata by name and version."""
        model_file = self._storage_dir / name / f"{version}.json"
        if not model_file.exists():
            return None

        with model_file.open() as f:
            return ModelMetadata.model_validate_json(f.read())

    def list_models(self) -> list[ModelMetadata]:
        """List all registered models."""
        models = []

        for model_dir in self._storage_dir.iterdir():
            if not model_dir.is_dir():
                continue

            for version_file in model_dir.glob("*.json"):
                with version_file.open() as f:
                    models.append(ModelMetadata.model_validate_json(f.read()))

        return models

    def get_model_versions(self, name: str) -> list[ModelMetadata]:
        """Get all versions of a model."""
        model_dir = self._storage_dir / name
        if not model_dir.exists():
            return []

        versions = []
        for version_file in model_dir.glob("*.json"):
            with version_file.open() as f:
                versions.append(ModelMetadata.model_validate_json(f.read()))

        # Sort by version
        versions.sort(key=lambda m: ModelVersion.from_string(m.version))
        return versions


class ModelRegistry:
    """Model registry with dependency injection."""

    def __init__(self, storage: ModelStorage | None = None) -> None:
        """Initialize with storage backend."""
        if storage is None:
            # Default to file storage in user's home directory
            default_dir = Path.home() / ".disinfo_relation_checker" / "models"
            storage = FileModelStorage(default_dir)
        self._storage = storage

    def register_model(self, metadata: ModelMetadata) -> None:
        """Register a model."""
        # Check if model version already exists
        existing = self._storage.get_model(metadata.name, metadata.version)
        if existing:
            msg = f"Model {metadata.name}:{metadata.version} already exists"
            raise ValueError(msg)

        # Save metadata
        self._storage.save_model(metadata)

    def get_model(self, name: str, version: str) -> ModelMetadata | None:
        """Get model metadata."""
        return self._storage.get_model(name, version)

    def list_models(self) -> list[ModelMetadata]:
        """List all registered models."""
        return self._storage.list_models()

    def get_model_versions(self, name: str) -> list[ModelMetadata]:
        """Get all versions of a model."""
        return self._storage.get_model_versions(name)

    def get_latest_version(self, name: str) -> ModelMetadata | None:
        """Get the latest version of a model."""
        versions = self.get_model_versions(name)
        if not versions:
            return None

        # Sort by version and return the latest
        versions.sort(key=lambda m: ModelVersion.from_string(m.version))
        return versions[-1]

    def delete_model(self, name: str, version: str) -> bool:
        """Delete a model version."""
        if isinstance(self._storage, FileModelStorage):
            model_dir = self._storage.get_storage_dir() / name
            model_file = model_dir / f"{version}.json"
            if model_file.exists():
                model_file.unlink()
                return True
        return False

    def update_model_tags(self, name: str, version: str, tags: list[str]) -> bool:
        """Update model tags."""
        metadata = self.get_model(name, version)
        if not metadata:
            return False

        # Create new metadata with updated tags
        updated_metadata = metadata.model_copy(update={"tags": tags})
        self._storage.save_model(updated_metadata)
        return True

    def search_models(self, query: str) -> list[ModelMetadata]:
        """Search models by name, description, or tags."""
        all_models = self.list_models()
        query_lower = query.lower()

        return [
            model
            for model in all_models
            if (
                query_lower in model.name.lower()
                or query_lower in model.description.lower()
                or any(query_lower in tag.lower() for tag in model.tags)
            )
        ]

    def get_best_performing_model(self, name: str) -> ModelMetadata | None:
        """Get the best performing version of a model based on F1 score."""
        versions = self.get_model_versions(name)
        if not versions:
            return None

        return max(versions, key=lambda m: m.performance.f1)


def create_model_metadata_from_config(
    name: str,
    version: str,
    description: str,
    config_data: dict[str, Any],
    tags: list[str] | None = None,
) -> ModelMetadata:
    """Create ModelMetadata from configuration data."""
    if "prompt_template" not in config_data:
        msg = "Configuration must contain 'prompt_template'"
        raise ValueError(msg)

    if "performance" not in config_data:
        msg = "Configuration must contain 'performance' metrics"
        raise ValueError(msg)

    llm_config = config_data.get("llm_config", {"provider_type": "mock"})
    performance = PerformanceMetrics.model_validate(config_data["performance"])
    created_at = datetime.now(UTC).isoformat()

    return ModelMetadata(
        name=name,
        version=version,
        description=description,
        prompt_template=config_data["prompt_template"],
        llm_config=llm_config,
        performance=performance,
        created_at=created_at,
        tags=tags or [],
    )
