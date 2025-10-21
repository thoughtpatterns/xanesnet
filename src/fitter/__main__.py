"""Initialize and run `cli`."""

# ruff: noqa: F401
# pyright: reportUnusedImport=false

from fitter.cli import cli
from fitter.commands import fit, train


def main() -> None:
    """Entry point for the `fitter` CLI command."""
    cli()


if __name__ == "__main__":
    main()
