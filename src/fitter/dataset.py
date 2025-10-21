"""Implement `Dataset`."""

from pathlib import Path

from numpy import load as load_npy
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from torch import Tensor, float32, tensor
from torch.utils.data import Dataset as VirtualDataset
from typing_extensions import Self, override

Return = tuple[Tensor, Tensor]


class Dataset(VirtualDataset[Return]):
    """Load normal mode amplitudes and spectra from two arrays."""

    def __init__(self, features: NDArray[float32], targets: NDArray[float32]) -> None:  # noqa: D107  # pyright: ignore[reportInvalidTypeForm]
        self.features: Tensor = tensor(features, dtype=float32)
        self.targets: Tensor = tensor(targets, dtype=float32)

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

    @classmethod
    def from_npy(cls, modes: Path, spectra: Path) -> tuple[Self, Self]:
        """Split two (X, Y) `.npy` files into train and validation datasets."""
        x, y = load_npy(modes), load_npy(spectra)
        tx, vx, ty, vy = train_test_split(x, y, random_state=42, shuffle=False)
        return cls(tx, ty), cls(vx, vy)
