"""
XANESNET

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either Version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import logging
import os
from dataclasses import dataclass

import torch

from pathlib import Path
from typing import List, Union
from tqdm import tqdm

from xanesnet.core_learn import Mode
from xanesnet.datasets.base_dataset import BaseDataset
from xanesnet.registry import register_dataset
from xanesnet.utils.fourier import fft
from xanesnet.utils.io import list_filestems, load_xanes, transform_xyz


@dataclass
class Data:
    x: torch.Tensor = None
    y: torch.Tensor = None
    e: torch.Tensor = None

    def to(self, device):
        # send batch do device
        self.x = self.x.to(device) if self.x is not None else None
        self.y = self.y.to(device) if self.y is not None else None

        return self


@register_dataset("xanesx")
class XanesXDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        xyz_path: List[str] | str = None,
        xanes_path: List[str] | str = None,
        mode: Mode = None,
        descriptors: list = None,
        **kwargs,
    ):
        # Unpack kwargs
        self.fft = kwargs.get("fourier", False)
        self.fft_concat = kwargs.get("fourier_concat", False)

        # dataset accepts only one path each for the XYZ and XANES datasets.
        xyz_path = self.unique_path(xyz_path)
        xanes_path = self.unique_path(xanes_path)

        BaseDataset.__init__(
            self, Path(root), xyz_path, xanes_path, mode, descriptors, **kwargs
        )

        # Save configuration
        params = {
            "fourier": self.fft,
            "fourier_concat": self.fft_concat,
        }
        self.register_config(locals(), type="xanesx")

    def set_file_names(self):
        """
        Get the list of valid file stems based on the
        xyz_path and/or xanes_path. If both are given, only common stems are kept.
        """
        xyz_path = self.xyz_path
        xanes_path = self.xanes_path

        if xyz_path and xanes_path:
            xyz_stems = set(list_filestems(xyz_path))
            xanes_stems = set(list_filestems(xanes_path))
            file_names = sorted(list(xyz_stems & xanes_stems))
        elif xyz_path:
            xyz_stems = set(list_filestems(xyz_path))
            file_names = sorted(list(xyz_stems))
        elif xanes_path:
            xanes_stems = set(list_filestems(xanes_path))
            file_names = sorted(list(xanes_stems))
        else:
            raise ValueError("At least one data dataset path must be provided.")

        if not file_names:
            raise ValueError("No matching files found in the provided paths.")

        self.file_names = file_names

    def process(self):
        """Processes raw XYZ and XANES file to convert them into data objects."""
        logging.info(f"Processing {len(self.file_names)} files to data objects...")
        for idx, stem in tqdm(enumerate(self.file_names), total=len(self.file_names)):
            xyz = xanes = e = None

            # transform xyz file into feature array
            if self.xyz_path:
                raw_path = os.path.join(self.xyz_path, f"{stem}.xyz")
                xyz = transform_xyz(raw_path, self.descriptors)

            # get xanes and energy arrays
            if self.xanes_path:
                raw_path = os.path.join(self.xanes_path, f"{stem}.txt")
                e, xanes = load_xanes(raw_path)
                if self.fft:
                    xanes = fft(xanes, self.fft_concat)

            if self.mode == Mode.XANES_TO_XYZ:
                x = xanes
                y = xyz
            else:
                x = xyz
                y = xanes

            data = Data(x=x, y=y, e=e)

            save_path = os.path.join(self.processed_dir, f"{stem}.pt")
            torch.save(data, save_path)

    def collate_fn(self, batch: list[Data]) -> Data:
        """
        Collates a list of Data objects into a single Data object  with batched tensors.
        """
        x_list = [sample.x for sample in batch]
        y_list = [sample.y for sample in batch]

        batched_x = torch.stack(x_list, dim=0).to(torch.float32)
        batched_y = torch.stack(y_list, dim=0).to(torch.float32)

        return Data(x=batched_x, y=batched_y)

    @property
    def x_size(self) -> Union[int, List[int]]:
        """Size of the feature array."""
        return len(self[0].x)

    @property
    def y_size(self) -> int:
        """Size of the label array."""
        return len(self[0].y)
