"""CLI module for disinfo-relation-checker."""

import argparse
from typing import Protocol

from disinfo_relation_checker import __version__


class VersionProvider(Protocol):
    """Protocol for version providers."""

    def get_version(self) -> str:
        """Get the version string."""


class VersionOutputter:
    """Handles version output formatting."""

    def format_version(self, version: str) -> str:
        """Format version string for output."""
        return version


class PackageVersionProvider:
    """Provides version from package."""

    def get_version(self) -> str:
        """Get version from package."""
        return __version__


class CliHandler:
    """Handles CLI commands with dependency injection."""

    def __init__(
        self,
        version_provider: VersionProvider,
        version_outputter: VersionOutputter,
    ) -> None:
        """Initialize with dependencies."""
        self._version_provider = version_provider
        self._version_outputter = version_outputter

    def handle_version_command(self) -> str:
        """Handle version command."""
        version = self._version_provider.get_version()
        return self._version_outputter.format_version(version)


def main() -> None:
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="disinfo-relation-checker",
        description="Check if a given text is related to a topic",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
    )

    # For now, just handle version
    parser.parse_args()


if __name__ == "__main__":
    main()
