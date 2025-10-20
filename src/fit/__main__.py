"""Initialize and run `cli`."""

from fit.cli import cli
from fit.commands import train  # noqa: F401  # pyright: ignore[reportUnusedImport]


def main() -> None:
    """Entry point for the `fit` CLI command."""
    cli()


if __name__ == "__main__":
    main()
