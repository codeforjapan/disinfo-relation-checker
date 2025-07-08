"""Unit tests for performance monitoring with SOLID design."""

from unittest.mock import Mock

import pytest

from disinfo_relation_checker.performance_monitoring import (
    Alert,
    AlertRule,
    MetricCollector,
    PerformanceMonitor,
    PerformanceRecord,
    TimeSeriesData,
)


class TestPerformanceRecord:
    """Test PerformanceRecord following single responsibility principle."""

    def test_performance_record_creation(self) -> None:
        """Test PerformanceRecord creation and properties."""
        record = PerformanceRecord(
            model_name="test_model",
            model_version="1.0.0",
            timestamp="2025-01-01T00:00:00Z",
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1=0.85,
            latency_ms=150.5,
            throughput_rps=100.0,
            error_rate=0.01,
            sample_count=1000,
        )

        assert record.model_name == "test_model"
        assert record.model_version == "1.0.0"
        assert record.timestamp == "2025-01-01T00:00:00Z"
        assert record.accuracy == 0.85
        assert record.precision == 0.82
        assert record.recall == 0.88
        assert record.f1 == 0.85
        assert record.latency_ms == 150.5
        assert record.throughput_rps == 100.0
        assert record.error_rate == 0.01
        assert record.sample_count == 1000

    def test_performance_record_validation(self) -> None:
        """Test PerformanceRecord validation."""
        with pytest.raises(ValueError):
            PerformanceRecord(
                model_name="test_model",
                model_version="1.0.0",
                timestamp="2025-01-01T00:00:00Z",
                accuracy=1.5,  # Invalid accuracy
                precision=0.82,
                recall=0.88,
                f1=0.85,
                latency_ms=150.5,
                throughput_rps=100.0,
                error_rate=0.01,
                sample_count=1000,
            )

        with pytest.raises(ValueError):
            PerformanceRecord(
                model_name="test_model",
                model_version="1.0.0",
                timestamp="2025-01-01T00:00:00Z",
                accuracy=0.85,
                precision=0.82,
                recall=0.88,
                f1=0.85,
                latency_ms=-10.0,  # Invalid latency
                throughput_rps=100.0,
                error_rate=0.01,
                sample_count=1000,
            )

    def test_performance_record_serialization(self) -> None:
        """Test PerformanceRecord serialization to dictionary."""
        record = PerformanceRecord(
            model_name="test_model",
            model_version="1.0.0",
            timestamp="2025-01-01T00:00:00Z",
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1=0.85,
            latency_ms=150.5,
            throughput_rps=100.0,
            error_rate=0.01,
            sample_count=1000,
        )

        data = record.to_dict()

        assert data["model_name"] == "test_model"
        assert data["accuracy"] == 0.85
        assert data["latency_ms"] == 150.5

    def test_performance_record_from_dict(self) -> None:
        """Test PerformanceRecord creation from dictionary."""
        data = {
            "model_name": "test_model",
            "model_version": "1.0.0",
            "timestamp": "2025-01-01T00:00:00Z",
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1": 0.85,
            "latency_ms": 150.5,
            "throughput_rps": 100.0,
            "error_rate": 0.01,
            "sample_count": 1000,
        }

        record = PerformanceRecord.from_dict(data)

        assert record.model_name == "test_model"
        assert record.accuracy == 0.85
        assert record.latency_ms == 150.5


class TestTimeSeriesData:
    """Test TimeSeriesData following single responsibility principle."""

    def test_time_series_data_creation(self) -> None:
        """Test TimeSeriesData creation and properties."""
        records = [
            PerformanceRecord(
                model_name="test_model",
                model_version="1.0.0",
                timestamp="2025-01-01T00:00:00Z",
                accuracy=0.85,
                precision=0.82,
                recall=0.88,
                f1=0.85,
                latency_ms=150.5,
                throughput_rps=100.0,
                error_rate=0.01,
                sample_count=1000,
            ),
            PerformanceRecord(
                model_name="test_model",
                model_version="1.0.0",
                timestamp="2025-01-01T01:00:00Z",
                accuracy=0.87,
                precision=0.84,
                recall=0.90,
                f1=0.87,
                latency_ms=145.0,
                throughput_rps=105.0,
                error_rate=0.008,
                sample_count=1050,
            ),
        ]

        time_series = TimeSeriesData(model_name="test_model", metric_name="accuracy", time_range="24h", records=records)

        assert time_series.model_name == "test_model"
        assert time_series.metric_name == "accuracy"
        assert time_series.time_range == "24h"
        assert len(time_series.records) == 2

    def test_time_series_data_get_values(self) -> None:
        """Test TimeSeriesData metric value extraction."""
        records = [
            PerformanceRecord(
                model_name="test_model",
                model_version="1.0.0",
                timestamp="2025-01-01T00:00:00Z",
                accuracy=0.85,
                precision=0.82,
                recall=0.88,
                f1=0.85,
                latency_ms=150.5,
                throughput_rps=100.0,
                error_rate=0.01,
                sample_count=1000,
            ),
            PerformanceRecord(
                model_name="test_model",
                model_version="1.0.0",
                timestamp="2025-01-01T01:00:00Z",
                accuracy=0.87,
                precision=0.84,
                recall=0.90,
                f1=0.87,
                latency_ms=145.0,
                throughput_rps=105.0,
                error_rate=0.008,
                sample_count=1050,
            ),
        ]

        time_series = TimeSeriesData(model_name="test_model", metric_name="accuracy", time_range="24h", records=records)

        values = time_series.get_values()
        assert values == [0.85, 0.87]

        latency_series = TimeSeriesData(
            model_name="test_model", metric_name="latency_ms", time_range="24h", records=records
        )

        latency_values = latency_series.get_values()
        assert latency_values == [150.5, 145.0]

    def test_time_series_data_calculate_statistics(self) -> None:
        """Test TimeSeriesData statistics calculation."""
        records = [
            PerformanceRecord(
                model_name="test_model",
                model_version="1.0.0",
                timestamp="2025-01-01T00:00:00Z",
                accuracy=0.85,
                precision=0.82,
                recall=0.88,
                f1=0.85,
                latency_ms=150.5,
                throughput_rps=100.0,
                error_rate=0.01,
                sample_count=1000,
            ),
            PerformanceRecord(
                model_name="test_model",
                model_version="1.0.0",
                timestamp="2025-01-01T01:00:00Z",
                accuracy=0.87,
                precision=0.84,
                recall=0.90,
                f1=0.87,
                latency_ms=145.0,
                throughput_rps=105.0,
                error_rate=0.008,
                sample_count=1050,
            ),
            PerformanceRecord(
                model_name="test_model",
                model_version="1.0.0",
                timestamp="2025-01-01T02:00:00Z",
                accuracy=0.83,
                precision=0.80,
                recall=0.86,
                f1=0.83,
                latency_ms=160.0,
                throughput_rps=95.0,
                error_rate=0.012,
                sample_count=980,
            ),
        ]

        time_series = TimeSeriesData(model_name="test_model", metric_name="accuracy", time_range="24h", records=records)

        stats = time_series.calculate_statistics()

        assert abs(stats["mean"] - 0.85) < 0.01
        assert stats["min"] == 0.83
        assert stats["max"] == 0.87
        assert stats["count"] == 3


class TestAlertRule:
    """Test AlertRule following single responsibility principle."""

    def test_alert_rule_creation(self) -> None:
        """Test AlertRule creation and properties."""
        rule = AlertRule(
            rule_id="accuracy_low",
            model_name="test_model",
            metric_name="accuracy",
            condition="<",
            threshold=0.80,
            severity="high",
            description="Accuracy dropped below 80%",
            enabled=True,
        )

        assert rule.rule_id == "accuracy_low"
        assert rule.model_name == "test_model"
        assert rule.metric_name == "accuracy"
        assert rule.condition == "<"
        assert rule.threshold == 0.80
        assert rule.severity == "high"
        assert rule.description == "Accuracy dropped below 80%"
        assert rule.enabled is True

    def test_alert_rule_check_condition(self) -> None:
        """Test AlertRule condition checking."""
        rule_less_than = AlertRule(
            rule_id="accuracy_low",
            model_name="test_model",
            metric_name="accuracy",
            condition="<",
            threshold=0.80,
            severity="high",
            description="Accuracy dropped below 80%",
            enabled=True,
        )

        assert rule_less_than.check_condition(0.75) is True
        assert rule_less_than.check_condition(0.85) is False

        rule_greater_than = AlertRule(
            rule_id="latency_high",
            model_name="test_model",
            metric_name="latency_ms",
            condition=">",
            threshold=200.0,
            severity="medium",
            description="Latency exceeds 200ms",
            enabled=True,
        )

        assert rule_greater_than.check_condition(250.0) is True
        assert rule_greater_than.check_condition(150.0) is False

    def test_alert_rule_serialization(self) -> None:
        """Test AlertRule serialization to dictionary."""
        rule = AlertRule(
            rule_id="test_rule",
            model_name="test_model",
            metric_name="accuracy",
            condition="<",
            threshold=0.80,
            severity="high",
            description="Test rule",
            enabled=True,
        )

        data = rule.to_dict()

        assert data["rule_id"] == "test_rule"
        assert data["threshold"] == 0.80
        assert data["enabled"] is True


class TestAlert:
    """Test Alert following single responsibility principle."""

    def test_alert_creation(self) -> None:
        """Test Alert creation and properties."""
        alert = Alert(
            alert_id="alert_001",
            rule_id="accuracy_low",
            model_name="test_model",
            metric_name="accuracy",
            current_value=0.75,
            threshold=0.80,
            severity="high",
            message="Accuracy dropped to 75%",
            triggered_at="2025-01-01T00:00:00Z",
            acknowledged=False,
        )

        assert alert.alert_id == "alert_001"
        assert alert.rule_id == "accuracy_low"
        assert alert.model_name == "test_model"
        assert alert.metric_name == "accuracy"
        assert alert.current_value == 0.75
        assert alert.threshold == 0.80
        assert alert.severity == "high"
        assert alert.message == "Accuracy dropped to 75%"
        assert alert.triggered_at == "2025-01-01T00:00:00Z"
        assert alert.acknowledged is False

    def test_alert_acknowledge(self) -> None:
        """Test Alert acknowledgment."""
        alert = Alert(
            alert_id="alert_001",
            rule_id="accuracy_low",
            model_name="test_model",
            metric_name="accuracy",
            current_value=0.75,
            threshold=0.80,
            severity="high",
            message="Accuracy dropped to 75%",
            triggered_at="2025-01-01T00:00:00Z",
            acknowledged=False,
        )

        assert alert.acknowledged is False

        alert.acknowledge()

        assert alert.acknowledged is True

    def test_alert_serialization(self) -> None:
        """Test Alert serialization to dictionary."""
        alert = Alert(
            alert_id="alert_001",
            rule_id="accuracy_low",
            model_name="test_model",
            metric_name="accuracy",
            current_value=0.75,
            threshold=0.80,
            severity="high",
            message="Test alert",
            triggered_at="2025-01-01T00:00:00Z",
            acknowledged=False,
        )

        data = alert.to_dict()

        assert data["alert_id"] == "alert_001"
        assert data["current_value"] == 0.75
        assert data["acknowledged"] is False


class TestMetricCollector:
    """Test MetricCollector following single responsibility principle."""

    def test_metric_collector_initialization(self) -> None:
        """Test MetricCollector initialization."""
        mock_storage = Mock()
        collector = MetricCollector(storage=mock_storage)

        assert collector._storage == mock_storage

    def test_collect_performance_metrics(self) -> None:
        """Test performance metrics collection."""
        mock_storage = Mock()
        collector = MetricCollector(storage=mock_storage)

        mock_classifier = Mock()
        mock_classifier.validate.return_value = {"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1": 0.85}

        test_data = [{"text": "test", "label": "1"}]

        record = collector.collect_performance_metrics(
            model_name="test_model",
            model_version="1.0.0",
            classifier=mock_classifier,
            test_data=test_data,
            start_time="2025-01-01T00:00:00Z",
        )

        assert record.model_name == "test_model"
        assert record.model_version == "1.0.0"
        assert record.accuracy == 0.85
        assert record.sample_count == 1

        # Verify storage was called
        mock_storage.save_performance_record.assert_called_once_with(record)

    def test_collect_latency_metrics(self) -> None:
        """Test latency metrics collection."""
        mock_storage = Mock()
        collector = MetricCollector(storage=mock_storage)

        latencies = [150.0, 145.0, 160.0, 155.0, 140.0]

        record = collector.collect_latency_metrics(
            model_name="test_model", model_version="1.0.0", latencies=latencies, timestamp="2025-01-01T00:00:00Z"
        )

        assert record.model_name == "test_model"
        assert record.latency_ms == 150.0  # Average latency
        assert record.sample_count == 5

        mock_storage.save_performance_record.assert_called_once_with(record)

    def test_get_time_series_data(self) -> None:
        """Test time series data retrieval."""
        mock_storage = Mock()
        collector = MetricCollector(storage=mock_storage)

        mock_records = [
            PerformanceRecord(
                model_name="test_model",
                model_version="1.0.0",
                timestamp="2025-01-01T00:00:00Z",
                accuracy=0.85,
                precision=0.82,
                recall=0.88,
                f1=0.85,
                latency_ms=150.5,
                throughput_rps=100.0,
                error_rate=0.01,
                sample_count=1000,
            )
        ]

        mock_storage.get_performance_records.return_value = mock_records

        time_series = collector.get_time_series_data(model_name="test_model", metric_name="accuracy", time_range="24h")

        assert time_series.model_name == "test_model"
        assert time_series.metric_name == "accuracy"
        assert time_series.time_range == "24h"
        assert len(time_series.records) == 1

        mock_storage.get_performance_records.assert_called_once_with(model_name="test_model", time_range="24h")


class TestPerformanceMonitor:
    """Test PerformanceMonitor with dependency injection."""

    def test_performance_monitor_initialization(self) -> None:
        """Test PerformanceMonitor initialization."""
        mock_metric_collector = Mock()
        mock_alert_manager = Mock()
        mock_storage = Mock()

        monitor = PerformanceMonitor(
            metric_collector=mock_metric_collector, alert_manager=mock_alert_manager, storage=mock_storage
        )

        assert monitor._metric_collector == mock_metric_collector
        assert monitor._alert_manager == mock_alert_manager
        assert monitor._storage == mock_storage

    def test_add_alert_rule(self) -> None:
        """Test adding alert rule."""
        mock_metric_collector = Mock()
        mock_alert_manager = Mock()
        mock_storage = Mock()

        monitor = PerformanceMonitor(
            metric_collector=mock_metric_collector, alert_manager=mock_alert_manager, storage=mock_storage
        )

        rule = AlertRule(
            rule_id="test_rule",
            model_name="test_model",
            metric_name="accuracy",
            condition="<",
            threshold=0.80,
            severity="high",
            description="Test rule",
            enabled=True,
        )

        monitor.add_alert_rule(rule)

        mock_storage.save_alert_rule.assert_called_once_with(rule)

    def test_check_alerts(self) -> None:
        """Test alert checking."""
        mock_metric_collector = Mock()
        mock_alert_manager = Mock()
        mock_storage = Mock()

        monitor = PerformanceMonitor(
            metric_collector=mock_metric_collector, alert_manager=mock_alert_manager, storage=mock_storage
        )

        # Mock alert rules
        rules = [
            AlertRule(
                rule_id="accuracy_low",
                model_name="test_model",
                metric_name="accuracy",
                condition="<",
                threshold=0.80,
                severity="high",
                description="Accuracy too low",
                enabled=True,
            )
        ]

        # Mock performance record with low accuracy
        record = PerformanceRecord(
            model_name="test_model",
            model_version="1.0.0",
            timestamp="2025-01-01T00:00:00Z",
            accuracy=0.75,  # Below threshold
            precision=0.82,
            recall=0.88,
            f1=0.85,
            latency_ms=150.5,
            throughput_rps=100.0,
            error_rate=0.01,
            sample_count=1000,
        )

        mock_storage.get_alert_rules.return_value = rules
        mock_storage.get_latest_performance_record.return_value = record

        alerts = monitor.check_alerts("test_model")

        assert len(alerts) == 1
        assert alerts[0].rule_id == "accuracy_low"
        assert alerts[0].current_value == 0.75

        mock_storage.save_alert.assert_called_once()

    def test_get_performance_summary(self) -> None:
        """Test performance summary generation."""
        mock_metric_collector = Mock()
        mock_alert_manager = Mock()
        mock_storage = Mock()

        monitor = PerformanceMonitor(
            metric_collector=mock_metric_collector, alert_manager=mock_alert_manager, storage=mock_storage
        )

        # Mock time series data
        mock_time_series = TimeSeriesData(
            model_name="test_model",
            metric_name="accuracy",
            time_range="24h",
            records=[
                PerformanceRecord(
                    model_name="test_model",
                    model_version="1.0.0",
                    timestamp="2025-01-01T00:00:00Z",
                    accuracy=0.85,
                    precision=0.82,
                    recall=0.88,
                    f1=0.85,
                    latency_ms=150.5,
                    throughput_rps=100.0,
                    error_rate=0.01,
                    sample_count=1000,
                )
            ],
        )

        mock_metric_collector.get_time_series_data.return_value = mock_time_series

        summary = monitor.get_performance_summary("test_model", "24h")

        assert "test_model" in summary
        assert "Accuracy" in summary  # Capitalized in the summary
        assert "24h" in summary

        mock_metric_collector.get_time_series_data.assert_called_with("test_model", "accuracy", "24h")

    def test_list_active_alerts(self) -> None:
        """Test listing active alerts."""
        mock_metric_collector = Mock()
        mock_alert_manager = Mock()
        mock_storage = Mock()

        monitor = PerformanceMonitor(
            metric_collector=mock_metric_collector, alert_manager=mock_alert_manager, storage=mock_storage
        )

        expected_alerts = [
            Alert(
                alert_id="alert_001",
                rule_id="accuracy_low",
                model_name="test_model",
                metric_name="accuracy",
                current_value=0.75,
                threshold=0.80,
                severity="high",
                message="Accuracy dropped to 75%",
                triggered_at="2025-01-01T00:00:00Z",
                acknowledged=False,
            )
        ]

        mock_storage.get_active_alerts.return_value = expected_alerts

        alerts = monitor.list_active_alerts("test_model")

        assert alerts == expected_alerts
        mock_storage.get_active_alerts.assert_called_once_with("test_model")
