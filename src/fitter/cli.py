"""Initialize `cli` and `console`."""

from rich.console import Console
from typer import Typer

cli = Typer(name="fit", add_completion=False)
console = Console()
