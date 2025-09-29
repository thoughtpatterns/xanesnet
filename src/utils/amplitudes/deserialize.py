# ruff: noqa: S301

"""Deserialize and read normal mode amplitude *.pk files."""

# Each pickle file is a dictionary, with three keys: 0.35, 0.25, 0.2. Each key is
# a _threshold_, used to define uniqueness of LHS-generated structures. For 10,000
# structures, we selected them with a large threshold (0.35 for the 0.50 set, where
# 0.50 expresses the [-0.50, +0.50] range for which each of the 58 modes of CoCl2en2
# were permitted to vary), for which 89 distinct structures were found. Then, we
# used a threshold of 0.25 for another 10,000, to find 273 structures; and finally,
# a threshold of 0.20 on a third 10,000, to find 615 structures, which gave 977 total
# structures.

# For each key, there are three arrays, e.g., raw[0.35] = [g2, amp, xyz], for which you
# can view the shape of each array via `print()`, as the `__str__` method below simply
# pretty-prints the shapes of a pickle's members, and the shape of their concatenation.
# The concatenated matrices give a (977, 58) matrix, which can be used as a descriptor
# for the inputs in `/data/cocl2en2/lhs/trans/0.50`.

# For the 0.75 set, there are more rows in the matrix than there are XANES files, as
# some calculations failed. It _should_ be true that, for a file `trans_XXXXX.txt`, the
# `XXXXX` is its respective row number of the amplitude matrix.

from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from pathlib import Path
from pickle import load

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

Array = NDArray[np.float64]


@dataclass(frozen=True)
class _Threshold:
    g2: Array
    amp: Array
    xyz: Array


class _Lhs:
    def __init__(self, path: Path) -> None:
        with path.open("rb") as f:
            raw = load(f)

        self.thresholds: dict[np.float64, _Threshold] = {
            threshold: _Threshold(g2=arrays[0], amp=arrays[1], xyz=arrays[2])
            for threshold, arrays in raw.items()
        }

    @override
    def __str__(self) -> str:
        result = [
            [
                f"{threshold}:",
                f"  g2.shape:  {array.g2.shape}",
                f"  amp.shape: {array.amp.shape}",
                f"  xyz.shape: {array.xyz.shape}",
            ]
            for threshold, array in self.thresholds.items()
        ] + [["", f"aggregate shape: {self.concat.shape}"]]

        return "\n".join(chain.from_iterable(result))

    @cached_property
    def concat(self) -> Array:
        return np.concatenate([x.amp for x in self.thresholds.values()], axis=0)


if __name__ == "__main__":
    lhs = _Lhs(Path("pickles/0.50.pk"))
    print(lhs)

    lhs = _Lhs(Path("pickles/0.75.pk"))
    print(lhs)
