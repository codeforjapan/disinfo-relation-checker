"""Unit tests for CLI module with SOLID design."""

from unittest.mock import Mock

from disinfo_relation_checker.cli import CliHandler, VersionOutputter


class TestVersionOutputter:
    """Test VersionOutputter following single responsibility principle."""

    def test_format_version_output(self) -> None:
        """Test that VersionOutputter formats version correctly."""
        outputter = VersionOutputter()
        result = outputter.format_version("1.2.3")
        assert result == "1.2.3"


class TestCliHandler:
    """Test CliHandler with dependency injection."""

    def test_handle_version_command(self) -> None:
        """Test that CliHandler handles version command correctly."""
        # Mock dependencies
        mock_version_provider = Mock()
        mock_version_provider.get_version.return_value = "1.2.3"

        mock_outputter = Mock()
        mock_outputter.format_version.return_value = "1.2.3"

        # Create handler with injected dependencies
        handler = CliHandler(version_provider=mock_version_provider, version_outputter=mock_outputter)

        # Test version command handling
        result = handler.handle_version_command()

        # Verify interactions
        mock_version_provider.get_version.assert_called_once()
        mock_outputter.format_version.assert_called_once_with("1.2.3")
        assert result == "1.2.3"
