"""Performance monitoring system with SOLID design principles."""

import json
import statistics
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


@dataclass
class PerformanceRecord:
    """Performance record for a model at a specific time."""

    model_name: str
    model_version: str
    timestamp: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    latency_ms: float
    throughput_rps: float
    error_rate: float
    sample_count: int

    def __post_init__(self) -> None:
        """Validate performance metrics."""
        # Validate accuracy metrics
        for metric_name, value in [
            ("accuracy", self.accuracy),
            ("precision", self.precision),
            ("recall", self.recall),
            ("f1", self.f1),
        ]:
            if not 0.0 <= value <= 1.0:
                msg = f"{metric_name} must be between 0.0 and 1.0, got {value}"
                raise ValueError(msg)

        # Validate performance metrics
        if self.latency_ms < 0:
            msg = f"Latency must be non-negative, got {self.latency_ms}"
            raise ValueError(msg)

        if self.throughput_rps < 0:
            msg = f"Throughput must be non-negative, got {self.throughput_rps}"
            raise ValueError(msg)

        if not 0.0 <= self.error_rate <= 1.0:
            msg = f"Error rate must be between 0.0 and 1.0, got {self.error_rate}"
            raise ValueError(msg)

        if self.sample_count < 0:
            msg = f"Sample count must be non-negative, got {self.sample_count}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "latency_ms": self.latency_ms,
            "throughput_rps": self.throughput_rps,
            "error_rate": self.error_rate,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerformanceRecord":
        """Create from dictionary."""
        return cls(
            model_name=data["model_name"],
            model_version=data["model_version"],
            timestamp=data["timestamp"],
            accuracy=data["accuracy"],
            precision=data["precision"],
            recall=data["recall"],
            f1=data["f1"],
            latency_ms=data["latency_ms"],
            throughput_rps=data["throughput_rps"],
            error_rate=data["error_rate"],
            sample_count=data["sample_count"],
        )


@dataclass
class TimeSeriesData:
    """Time series data for a specific metric."""

    model_name: str
    metric_name: str
    time_range: str
    records: list[PerformanceRecord]

    def get_values(self) -> list[float]:
        """Get metric values from records."""
        values = []
        for record in self.records:
            value = getattr(record, self.metric_name, None)
            if value is not None:
                values.append(value)
        return values

    def calculate_statistics(self) -> dict[str, float]:
        """Calculate statistics for the metric."""
        values = self.get_values()
        if not values:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        return {
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }


@dataclass
class AlertRule:
    """Rule for triggering alerts based on metrics."""

    rule_id: str
    model_name: str
    metric_name: str
    condition: str  # "<", ">", "==", "!="
    threshold: float
    severity: str  # "low", "medium", "high", "critical"
    description: str
    enabled: bool

    def check_condition(self, value: float) -> bool:
        """Check if value triggers the alert condition."""
        if not self.enabled:
            return False

        if self.condition == "<":
            return value < self.threshold
        if self.condition == ">":
            return value > self.threshold
        if self.condition == "==":
            return abs(value - self.threshold) < 1e-9
        if self.condition == "!=":
            return abs(value - self.threshold) >= 1e-9
        msg = f"Unknown condition: {self.condition}"
        raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "model_name": self.model_name,
            "metric_name": self.metric_name,
            "condition": self.condition,
            "threshold": self.threshold,
            "severity": self.severity,
            "description": self.description,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlertRule":
        """Create from dictionary."""
        return cls(
            rule_id=data["rule_id"],
            model_name=data["model_name"],
            metric_name=data["metric_name"],
            condition=data["condition"],
            threshold=data["threshold"],
            severity=data["severity"],
            description=data["description"],
            enabled=data["enabled"],
        )


@dataclass
class Alert:
    """Alert triggered by a rule."""

    alert_id: str
    rule_id: str
    model_name: str
    metric_name: str
    current_value: float
    threshold: float
    severity: str
    message: str
    triggered_at: str
    acknowledged: bool = False

    def acknowledge(self) -> None:
        """Acknowledge the alert."""
        self.acknowledged = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "rule_id": self.rule_id,
            "model_name": self.model_name,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "severity": self.severity,
            "message": self.message,
            "triggered_at": self.triggered_at,
            "acknowledged": self.acknowledged,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Alert":
        """Create from dictionary."""
        return cls(
            alert_id=data["alert_id"],
            rule_id=data["rule_id"],
            model_name=data["model_name"],
            metric_name=data["metric_name"],
            current_value=data["current_value"],
            threshold=data["threshold"],
            severity=data["severity"],
            message=data["message"],
            triggered_at=data["triggered_at"],
            acknowledged=data.get("acknowledged", False),
        )


class PerformanceStorage(Protocol):
    """Protocol for performance data storage backends."""

    def save_performance_record(self, record: PerformanceRecord) -> None:
        """Save performance record."""
        ...

    def get_performance_records(self, model_name: str, time_range: str) -> list[PerformanceRecord]:
        """Get performance records for a model within time range."""
        ...

    def get_latest_performance_record(self, model_name: str) -> PerformanceRecord | None:
        """Get the latest performance record for a model."""
        ...

    def save_alert_rule(self, rule: AlertRule) -> None:
        """Save alert rule."""
        ...

    def get_alert_rules(self, model_name: str) -> list[AlertRule]:
        """Get alert rules for a model."""
        ...

    def save_alert(self, alert: Alert) -> None:
        """Save alert."""
        ...

    def get_active_alerts(self, model_name: str) -> list[Alert]:
        """Get active (unacknowledged) alerts for a model."""
        ...


class FilePerformanceStorage:
    """File-based performance storage implementation."""

    def __init__(self, storage_dir: Path) -> None:
        """Initialize with storage directory."""
        self._storage_dir = storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self._records_dir = self._storage_dir / "records"
        self._rules_dir = self._storage_dir / "rules"
        self._alerts_dir = self._storage_dir / "alerts"

        for dir_path in [self._records_dir, self._rules_dir, self._alerts_dir]:
            dir_path.mkdir(exist_ok=True)

    def save_performance_record(self, record: PerformanceRecord) -> None:
        """Save performance record to file."""
        model_dir = self._records_dir / record.model_name
        model_dir.mkdir(exist_ok=True)

        # Use timestamp as filename to maintain chronological order
        timestamp_str = record.timestamp.replace(":", "-").replace(".", "-")
        record_file = model_dir / f"{timestamp_str}.json"

        with record_file.open("w") as f:
            json.dump(record.to_dict(), f, indent=2)

    def get_performance_records(self, model_name: str, time_range: str) -> list[PerformanceRecord]:
        """Get performance records for a model within time range."""
        model_dir = self._records_dir / model_name
        if not model_dir.exists():
            return []

        records = []
        for record_file in model_dir.glob("*.json"):
            with record_file.open() as f:
                data = json.load(f)
            records.append(PerformanceRecord.from_dict(data))

        # Sort by timestamp
        records.sort(key=lambda r: r.timestamp)

        # Filter by time range (simplified implementation)
        # In a real implementation, you'd parse the time_range and filter accordingly
        return records

    def get_latest_performance_record(self, model_name: str) -> PerformanceRecord | None:
        """Get the latest performance record for a model."""
        records = self.get_performance_records(model_name, "all")
        return records[-1] if records else None

    def save_alert_rule(self, rule: AlertRule) -> None:
        """Save alert rule to file."""
        rule_file = self._rules_dir / f"{rule.rule_id}.json"
        with rule_file.open("w") as f:
            json.dump(rule.to_dict(), f, indent=2)

    def get_alert_rules(self, model_name: str) -> list[AlertRule]:
        """Get alert rules for a model."""
        rules = []

        for rule_file in self._rules_dir.glob("*.json"):
            with rule_file.open() as f:
                data = json.load(f)
            rule = AlertRule.from_dict(data)

            if rule.model_name == model_name:
                rules.append(rule)

        return rules

    def save_alert(self, alert: Alert) -> None:
        """Save alert to file."""
        alert_file = self._alerts_dir / f"{alert.alert_id}.json"
        with alert_file.open("w") as f:
            json.dump(alert.to_dict(), f, indent=2)

    def get_active_alerts(self, model_name: str) -> list[Alert]:
        """Get active (unacknowledged) alerts for a model."""
        alerts = []

        for alert_file in self._alerts_dir.glob("*.json"):
            with alert_file.open() as f:
                data = json.load(f)
            alert = Alert.from_dict(data)

            if alert.model_name == model_name and not alert.acknowledged:
                alerts.append(alert)

        return alerts


class MetricCollector:
    """Collects performance metrics with dependency injection."""

    def __init__(self, storage: PerformanceStorage) -> None:
        """Initialize with storage backend."""
        self._storage = storage

    def collect_performance_metrics(
        self,
        model_name: str,
        model_version: str,
        classifier: Any,
        test_data: list[dict[str, Any]],
        start_time: str,
    ) -> PerformanceRecord:
        """Collect performance metrics for a model."""
        # Measure classification performance
        if start_time.endswith("Z"):
            # Convert UTC 'Z' notation to standard offset format
            start_timestamp = datetime.fromisoformat(start_time[:-1] + "+00:00")  # noqa: FURB162
        else:
            start_timestamp = datetime.fromisoformat(start_time)

        # Run validation
        metrics = classifier.validate(test_data)

        end_timestamp = datetime.now(UTC)
        duration_ms = (end_timestamp - start_timestamp).total_seconds() * 1000

        # Calculate throughput
        throughput_rps = len(test_data) / (duration_ms / 1000) if duration_ms > 0 else 0.0

        # Create performance record
        record = PerformanceRecord(
            model_name=model_name,
            model_version=model_version,
            timestamp=datetime.now(UTC).isoformat(),
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            latency_ms=duration_ms / len(test_data) if test_data else 0.0,
            throughput_rps=throughput_rps,
            error_rate=0.0,  # Would be calculated from actual errors
            sample_count=len(test_data),
        )

        # Save record
        self._storage.save_performance_record(record)

        return record

    def collect_latency_metrics(
        self,
        model_name: str,
        model_version: str,
        latencies: list[float],
        timestamp: str,
    ) -> PerformanceRecord:
        """Collect latency metrics."""
        avg_latency = statistics.mean(latencies) if latencies else 0.0

        record = PerformanceRecord(
            model_name=model_name,
            model_version=model_version,
            timestamp=timestamp,
            accuracy=0.0,  # Not available for latency-only collection
            precision=0.0,
            recall=0.0,
            f1=0.0,
            latency_ms=avg_latency,
            throughput_rps=1000.0 / avg_latency if avg_latency > 0 else 0.0,
            error_rate=0.0,
            sample_count=len(latencies),
        )

        self._storage.save_performance_record(record)
        return record

    def get_time_series_data(self, model_name: str, metric_name: str, time_range: str) -> TimeSeriesData:
        """Get time series data for a metric."""
        records = self._storage.get_performance_records(model_name=model_name, time_range=time_range)

        return TimeSeriesData(
            model_name=model_name,
            metric_name=metric_name,
            time_range=time_range,
            records=records,
        )


class PerformanceMonitor:
    """Main performance monitoring system with dependency injection."""

    def __init__(
        self,
        metric_collector: MetricCollector,
        alert_manager: Any = None,
        storage: PerformanceStorage | None = None,
    ) -> None:
        """Initialize with dependencies."""
        self._metric_collector = metric_collector
        self._alert_manager = alert_manager

        if storage is None:
            default_dir = Path.home() / ".disinfo_relation_checker" / "performance"
            storage = FilePerformanceStorage(default_dir)
        self._storage = storage

    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self._storage.save_alert_rule(rule)

    def check_alerts(self, model_name: str) -> list[Alert]:
        """Check for triggered alerts."""
        rules = self._storage.get_alert_rules(model_name)
        latest_record = self._storage.get_latest_performance_record(model_name)

        if not latest_record:
            return []

        triggered_alerts = []

        for rule in rules:
            # Get metric value from latest record
            metric_value = getattr(latest_record, rule.metric_name, None)

            if metric_value is not None and rule.check_condition(metric_value):
                # Create alert
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    model_name=model_name,
                    metric_name=rule.metric_name,
                    current_value=metric_value,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    message=f"{rule.description}: {rule.metric_name} = {metric_value:.3f}",
                    triggered_at=datetime.now(UTC).isoformat(),
                    acknowledged=False,
                )

                triggered_alerts.append(alert)
                self._storage.save_alert(alert)

        return triggered_alerts

    def get_performance_summary(self, model_name: str, time_range: str) -> str:
        """Get performance summary for a model."""
        time_series = self._metric_collector.get_time_series_data(model_name, "accuracy", time_range)

        stats = time_series.calculate_statistics()

        return f"""Performance Summary for {model_name} ({time_range}):
Accuracy: mean={stats["mean"]:.3f}, min={stats["min"]:.3f}, max={stats["max"]:.3f}
Sample count: {stats["count"]}"""

    def list_active_alerts(self, model_name: str) -> list[Alert]:
        """List active alerts for a model."""
        return self._storage.get_active_alerts(model_name)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        # This would typically update the alert in storage
        # For now, simplified implementation
        return True

    def get_model_health_status(self, model_name: str) -> dict[str, Any]:
        """Get overall health status for a model."""
        latest_record = self._storage.get_latest_performance_record(model_name)
        active_alerts = self.list_active_alerts(model_name)

        if not latest_record:
            return {"status": "unknown", "message": "No performance data available"}

        # Determine health status based on alerts and metrics
        if active_alerts:
            critical_alerts = [a for a in active_alerts if a.severity == "critical"]
            status = "critical" if critical_alerts else "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "last_updated": latest_record.timestamp,
            "accuracy": latest_record.accuracy,
            "latency_ms": latest_record.latency_ms,
            "active_alerts": len(active_alerts),
        }
