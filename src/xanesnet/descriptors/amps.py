"""Use a pre-computed `*.pk` file of normal mode amplitudes as our descriptor."""

# ruff: noqa: S301
# pyright: reportMissingTypeArgument=false

from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from pathlib import Path
from pickle import load
from typing import Final

import numpy as np
from ase import Atoms
from numpy.typing import NDArray
from typing_extensions import override

from xanesnet.descriptors.base_descriptor import BaseDescriptor
from xanesnet.registry import register_descriptor

# :::::::::::::::::::::::::::::: `_Pickle` implementation. :::::::::::::::::::::::::::::

# Each pickle file is a dictionary, with three keys: 0.35, 0.25, 0.2. Each key is
# a _threshold_, used to define uniqueness of LHS-generated structures. For 10,000
# structures, we selected them with a large threshold (0.35 for the 0.50 set, where
# 0.50 expresses the [-0.50, +0.50] range for which each of the 58 modes of CoCl2en2
# were permitted to vary), for which 89 distinct structures were found. Then, we
# used a threshold of 0.25 for another 10,000, to find 273 structures; and finally,
# a threshold of 0.20 on a third 10,000, to find 615 structures, which gave 977 total
# structures.

# For each key, there are three arrays, e.g., `raw[0.35] = (g2, amp, xyz)`, for
# which you can view the shape of each array via `print()`, as the `__str__` method
# below simply pretty-prints the shapes of a pickle's members, and the shape of their
# concatenation.  The concatenated matrices give a (977, 58) matrix, which can be used
# as a descriptor for the inputs in `/data/cocl2en2/lhs/trans/0.50`.

# For the 0.75 set, there are more rows in the matrix than there are XANES files, as
# some calculations failed. It _should_ be true that, for a file `trans_XXXXX.txt`, the
# `XXXXX` is its respective row number of the amplitude matrix.

# TODO: it would be best to find some intuitive standardization for pickle file inputs,
# or, better, to implement some way to re-calculate normal mode amplitudes from some
# "zero" molecule, if `*.xyz` files are given as input.


@dataclass(frozen=True)
class _Threshold:
    g2: NDArray[np.float64]
    amp: NDArray[np.float64]
    xyz: NDArray[np.float64]


class _Pickle:
    """A loaded `*.pk` file, which comprises an array of normal mode amplitudes."""

    def __init__(self, path: Path) -> None:
        """Read a `*.pk` file from the filesystem."""
        with path.open("rb") as f:
            raw = load(f)

        self._thresholds: dict[np.float64, _Threshold] = {
            threshold: _Threshold(g2=arrays[0], amp=arrays[1], xyz=arrays[2])
            for threshold, arrays in raw.items()
        }

    # We implement `__str__` for discovery's sake, as we can check the validity pickle
    # files via their shape.

    @override
    def __str__(self) -> str:
        result = [
            [
                f"{threshold}:",
                f"  g2.shape:  {array.g2.shape}",
                f"  amp.shape: {array.amp.shape}",
                f"  xyz.shape: {array.xyz.shape}",
            ]
            for threshold, array in self._thresholds.items()
        ] + [["", f"aggregate shape: {self.concat.shape}"]]

        return "\n".join(chain.from_iterable(result))

    @cached_property
    def concat(self) -> NDArray[np.float64]:
        """Concatenate amplitude matrices into an array of descriptors."""
        return np.concatenate([x.amp for x in self._thresholds.values()], axis=0)


# :::::::::::::::::::::::::::::::: `Amps` implementation. ::::::::::::::::::::::::::::::

_name: Final = "amps"


@register_descriptor(_name)
class Amps(BaseDescriptor):
    """Load a pre-computed `*.pk` file of normal mode amplitudes."""

    def __init__(self, pickle: Path | str, skips: list[int] | None = None) -> None:  # noqa: D107
        super().__init__()
        self.register_config(locals(), type=_name)

        self._amps: NDArray[np.float64] = _Pickle(Path(pickle)).concat
        self._index: int = 0
        self._skips: list[int] = skips or []
        self._test: list[int] = []

    @override
    def transform(self, system: Atoms) -> NDArray[np.float64]:
        while self._index in self._skips:
            self._index += 1

        aux = self._amps[self._index]
        self._test.append(self._index)
        self._index += 1

        return aux


    @override
    def get_nfeatures(self) -> int:
        return self._amps.shape[1]

    @override
    def get_type(self) -> str:
        return _name
