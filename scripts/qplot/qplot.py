"""Plot FDMNES output."""

from argparse import ArgumentParser
from pathlib import Path
from sys import stderr

import matplotlib.pyplot as plt
from pandas import read_csv


def qplot(path: Path, *, save: bool = False, show: bool = False) -> None:
    """Plot an FDMNES output file."""
    if not cli.save and not cli.show:
        print("must specify `--save` or `--show`", file=stderr)
        return

    fdmnes = read_csv(path, sep=r"\s+", skiprows=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fdmnes["energy"], fdmnes["<xanes>"], linestyle="-")
    ax.set_title(f"{path.with_suffix('').name}")

    if save:
        fig.savefig(f"{path.with_suffix('.svg').name}")

    if show:
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(prog="qplot")
    _ = parser.add_argument("path", type=Path)
    _ = parser.add_argument("--save", action="store_true")
    _ = parser.add_argument("--show", action="store_true")

    cli = parser.parse_args()
    qplot(cli.path, save=cli.save, show=cli.show)
