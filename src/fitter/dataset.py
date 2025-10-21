"""Implement `Dataset`."""

from pathlib import Path

from numpy import load as load_npy
from torch import Tensor, float64, tensor
from torch.utils.data import Dataset as VirtualDataset
from typing_extensions import override

Return = tuple[Tensor, Tensor]


class Dataset(VirtualDataset[Return]):
    """Load normal mode amplitudes and spectra from `.npy` files."""

    def __init__(self, features: Path, targets: Path) -> None:  # noqa: D107
        self.features: Tensor = tensor(load_npy(features), dtype=float64)
        self.targets: Tensor = tensor(load_npy(targets), dtype=float64)

        if len(self.features) != len(self.targets):
            msg = f"`{len(self.features) = }` must equal `{len(self.targets) = }`"
            raise ValueError(msg)

    def __len__(self) -> int:  # noqa: D105
        return len(self.features)

    @override
    def __getitem__(self, index: int) -> Return:
        return self.features[index], self.targets[index]

    @property
    def dimensions(self) -> tuple[int, int]:
        """Return a tuple with `x = input_dim`, `y = output_dim`."""
        if self.features.ndim == 2 and self.targets.ndim == 2:
            return self.features.shape[1], self.targets.shape[1]

        msg = f"`{self.features.shape = }`, `{self.targets.shape = }` must each equal 2"
        raise ValueError(msg)
