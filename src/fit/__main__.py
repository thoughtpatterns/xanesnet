"""Initialize and run `cli`."""

from fit.cli import cli
from fit.commands import train  # noqa: F401  # pyright: ignore[reportUnusedImport]

if __name__ == "__main__":
    cli()
