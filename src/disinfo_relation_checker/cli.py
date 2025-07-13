"""CLI interface for disinfo-relation-checker."""

import typer

from disinfo_relation_checker import __version__


def version_callback(*, value: bool) -> None:
    """Print version information."""
    if value:
        typer.echo(f"disinfo-relation-checker {__version__}")
        raise typer.Exit


def create_app() -> typer.Typer:
    """Create and configure the Typer app."""
    app = typer.Typer(
        name="disinfo-relation-checker",
        help="Check if a given text is related to disinformation analysis topic",
    )

    @app.callback()
    def callback(
        *,
        version: bool = typer.Option(
            False,  # noqa: FBT003
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
    ) -> None:
        """Check if a given text is related to disinformation analysis topic."""

    return app


def main() -> None:
    """Start the main CLI application."""
    app = create_app()
    app()


if __name__ == "__main__":
    main()
