"""A/B testing framework with SOLID design principles."""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .model_registry import ModelRegistry


@dataclass
class ABTestConfig:
    """Configuration for an A/B test."""

    test_name: str
    model_a: str  # Format: "model_name:version"
    model_b: str  # Format: "model_name:version"
    traffic_split: int  # Percentage of traffic for model A (0-100)
    test_data_path: str
    created_at: str
    status: str  # "active", "completed", "paused"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0 <= self.traffic_split <= 100:
            msg = f"Traffic split must be between 0 and 100, got {self.traffic_split}"
            raise ValueError(msg)

        valid_statuses = ["active", "completed", "paused"]
        if self.status not in valid_statuses:
            msg = f"Status must be one of {valid_statuses}, got '{self.status}'"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "traffic_split": self.traffic_split,
            "test_data_path": self.test_data_path,
            "created_at": self.created_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ABTestConfig":
        """Create from dictionary."""
        return cls(
            test_name=data["test_name"],
            model_a=data["model_a"],
            model_b=data["model_b"],
            traffic_split=data["traffic_split"],
            test_data_path=data["test_data_path"],
            created_at=data["created_at"],
            status=data["status"],
        )


@dataclass
class ABTestResult:
    """Results of an A/B test."""

    test_name: str
    model_a_performance: dict[str, float]
    model_b_performance: dict[str, float]
    sample_size_a: int
    sample_size_b: int
    statistical_significance: float
    winner: str  # "model_a", "model_b", or "no_significant_difference"
    completed_at: str

    def determine_winner(self) -> str:
        """Determine the winner based on F1 score."""
        f1_a = self.model_a_performance.get("f1", 0.0)
        f1_b = self.model_b_performance.get("f1", 0.0)

        if f1_a > f1_b:
            return "model_a"
        if f1_b > f1_a:
            return "model_b"
        return "no_significant_difference"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "model_a_performance": self.model_a_performance,
            "model_b_performance": self.model_b_performance,
            "sample_size_a": self.sample_size_a,
            "sample_size_b": self.sample_size_b,
            "statistical_significance": self.statistical_significance,
            "winner": self.winner,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ABTestResult":
        """Create from dictionary."""
        return cls(
            test_name=data["test_name"],
            model_a_performance=data["model_a_performance"],
            model_b_performance=data["model_b_performance"],
            sample_size_a=data["sample_size_a"],
            sample_size_b=data["sample_size_b"],
            statistical_significance=data["statistical_significance"],
            winner=data["winner"],
            completed_at=data["completed_at"],
        )


class TrafficSplitter:
    """Splits traffic between model variants."""

    def __init__(self, split_percentage: int, random_seed: int = 42) -> None:
        """Initialize traffic splitter."""
        self._split_percentage = split_percentage
        self._random_seed = random_seed

    def assign_group(self, identifier: str) -> str:
        """Assign identifier to group A or B consistently."""
        # Use hash of identifier for consistent assignment
        # Note: Using MD5 for non-cryptographic consistent hashing, not for security
        hash_value = hashlib.md5(identifier.encode()).hexdigest()  # noqa: S324
        # Convert first 8 characters to integer
        hash_int = int(hash_value[:8], 16)
        # Normalize to 0-100 range
        percentage = hash_int % 101

        return "A" if percentage <= self._split_percentage else "B"


class ABTestStorage(Protocol):
    """Protocol for A/B test storage backends."""

    def save_ab_test_config(self, config: ABTestConfig) -> None:
        """Save A/B test configuration."""
        ...

    def get_ab_test_config(self, test_name: str) -> ABTestConfig | None:
        """Get A/B test configuration."""
        ...

    def list_ab_tests(self) -> list[ABTestConfig]:
        """List all A/B tests."""
        ...

    def save_ab_test_result(self, result: ABTestResult) -> None:
        """Save A/B test result."""
        ...

    def get_ab_test_result(self, test_name: str) -> ABTestResult | None:
        """Get A/B test result."""
        ...


class FileABTestStorage:
    """File-based A/B test storage implementation."""

    def __init__(self, storage_dir: Path) -> None:
        """Initialize with storage directory."""
        self._storage_dir = storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self._configs_dir = self._storage_dir / "configs"
        self._results_dir = self._storage_dir / "results"
        self._configs_dir.mkdir(exist_ok=True)
        self._results_dir.mkdir(exist_ok=True)

    def save_ab_test_config(self, config: ABTestConfig) -> None:
        """Save A/B test configuration to file."""
        config_file = self._configs_dir / f"{config.test_name}.json"
        with config_file.open("w") as f:
            json.dump(config.to_dict(), f, indent=2)

    def get_ab_test_config(self, test_name: str) -> ABTestConfig | None:
        """Get A/B test configuration by name."""
        config_file = self._configs_dir / f"{test_name}.json"
        if not config_file.exists():
            return None

        with config_file.open() as f:
            data = json.load(f)

        return ABTestConfig.from_dict(data)

    def list_ab_tests(self) -> list[ABTestConfig]:
        """List all A/B test configurations."""
        configs = []

        for config_file in self._configs_dir.glob("*.json"):
            with config_file.open() as f:
                data = json.load(f)
            configs.append(ABTestConfig.from_dict(data))

        return configs

    def save_ab_test_result(self, result: ABTestResult) -> None:
        """Save A/B test result to file."""
        result_file = self._results_dir / f"{result.test_name}.json"
        with result_file.open("w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def get_ab_test_result(self, test_name: str) -> ABTestResult | None:
        """Get A/B test result by name."""
        result_file = self._results_dir / f"{test_name}.json"
        if not result_file.exists():
            return None

        with result_file.open() as f:
            data = json.load(f)

        return ABTestResult.from_dict(data)


class ABTestRunner:
    """Runs A/B tests with dependency injection."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        classifier: Any,
        storage: ABTestStorage | None = None,
    ) -> None:
        """Initialize with dependencies."""
        self._model_registry = model_registry
        self._classifier = classifier

        if storage is None:
            # Default to file storage
            default_dir = Path.home() / ".disinfo_relation_checker" / "ab_tests"
            storage = FileABTestStorage(default_dir)
        self._storage = storage

    def setup_ab_test(self, config: ABTestConfig) -> None:
        """Set up an A/B test."""
        # Validate that both models exist
        model_a_name, model_a_version = config.model_a.split(":")
        model_b_name, model_b_version = config.model_b.split(":")

        model_a = self._model_registry.get_model(model_a_name, model_a_version)
        model_b = self._model_registry.get_model(model_b_name, model_b_version)

        if not model_a:
            msg = f"Model A not found: {config.model_a}"
            raise ValueError(msg)

        if not model_b:
            msg = f"Model B not found: {config.model_b}"
            raise ValueError(msg)

        # Save configuration
        self._storage.save_ab_test_config(config)

    def run_ab_test(self, test_name: str, test_data: list[dict[str, Any]]) -> ABTestResult:
        """Run an A/B test on provided data."""
        # Get test configuration
        config = self._storage.get_ab_test_config(test_name)
        if not config:
            msg = f"A/B test not found: {test_name}"
            raise ValueError(msg)

        # Get models
        model_a_name, model_a_version = config.model_a.split(":")
        model_b_name, model_b_version = config.model_b.split(":")

        model_a = self._model_registry.get_model(model_a_name, model_a_version)
        model_b = self._model_registry.get_model(model_b_name, model_b_version)

        if not model_a or not model_b:
            msg = f"Models not found for test: {test_name}"
            raise ValueError(msg)

        # Split data based on traffic configuration
        splitter = TrafficSplitter(config.traffic_split)

        group_a_data = []
        group_b_data = []

        for i, item in enumerate(test_data):
            group = splitter.assign_group(f"{test_name}_{i}")
            if group == "A":
                group_a_data.append(item)
            else:
                group_b_data.append(item)

        # Evaluate both models
        self._classifier.set_prompt_template(model_a.prompt_template)
        model_a_performance = self._classifier.validate(group_a_data)

        self._classifier.set_prompt_template(model_b.prompt_template)
        model_b_performance = self._classifier.validate(group_b_data)

        # Calculate statistical significance (simplified)
        # In a real implementation, you'd use proper statistical tests
        f1_diff = abs(model_a_performance["f1"] - model_b_performance["f1"])
        significance = min(0.05, f1_diff)  # Simplified calculation

        # Create result
        result = ABTestResult(
            test_name=test_name,
            model_a_performance=model_a_performance,
            model_b_performance=model_b_performance,
            sample_size_a=len(group_a_data),
            sample_size_b=len(group_b_data),
            statistical_significance=significance,
            winner="",  # Will be determined
            completed_at=datetime.now(UTC).isoformat(),
        )

        # Determine winner
        result.winner = result.determine_winner()

        # Save result
        self._storage.save_ab_test_result(result)

        return result

    def get_ab_test_result(self, test_name: str) -> ABTestResult | None:
        """Get A/B test result."""
        return self._storage.get_ab_test_result(test_name)

    def list_ab_tests(self) -> list[ABTestConfig]:
        """List all A/B tests."""
        return self._storage.list_ab_tests()

    def stop_ab_test(self, test_name: str) -> bool:
        """Stop an active A/B test."""
        config = self._storage.get_ab_test_config(test_name)
        if not config:
            return False

        config.status = "paused"
        self._storage.save_ab_test_config(config)
        return True

    def get_active_tests(self) -> list[ABTestConfig]:
        """Get all active A/B tests."""
        all_tests = self.list_ab_tests()
        return [test for test in all_tests if test.status == "active"]


@dataclass
class ABTest:
    """Represents a complete A/B test with config and results."""

    config: ABTestConfig
    result: ABTestResult | None = None

    def is_significant(self, threshold: float = 0.05) -> bool:
        """Check if results are statistically significant."""
        if not self.result:
            return False
        return self.result.statistical_significance < threshold

    def summary(self) -> str:
        """Generate a summary of the A/B test."""
        if not self.result:
            return f"A/B Test '{self.config.test_name}': No results available"

        model_a_f1 = self.result.model_a_performance.get("f1", 0.0)
        model_b_f1 = self.result.model_b_performance.get("f1", 0.0)

        return f"""A/B Test Summary: {self.config.test_name}
Model A ({self.config.model_a}): F1 = {model_a_f1:.3f} (n={self.result.sample_size_a})
Model B ({self.config.model_b}): F1 = {model_b_f1:.3f} (n={self.result.sample_size_b})
Winner: {self.result.winner}
Statistical Significance: {self.result.statistical_significance:.3f}
Significant: {self.is_significant()}"""
