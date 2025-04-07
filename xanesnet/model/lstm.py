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

import numpy as np
import torch
from torch import nn

from xanesnet.model.base_model import Model
from xanesnet.utils_model import ActivationSwitch


class LSTM(Model):
    """
    A class for constructing a customisable LSTM (Long Short-Term Memory) model.
    The model consists of a bidirectional LSTM layer follow by a
    feedforward neural network with two dense layers.

    The defined LSTM is bidirectional, processing the input sequence in both
    forward and backward directions. The output size of the LSTM is
    2 x the number of features in the LSTM hidden state (hidden_size).
    The intermediate hidden size of the following FNN is user-specified
    (hidden_out_size)
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int,
        hidden_out_size: int,
        num_layers: int,
        dropout: float,
        activation: str,
    ):
        """
        Args:
            hidden_size (integer): Number of features in the hidden state of
                the LSTM layer.
            hidden_out_size (integer): Intermediate size of the two dense layers
                in the feedforward neural network.
            num_layers (integer): Number of recurrent layers (LSTM blocks)
                in the network.
            dropout (float): If none-zero, add a dropout layer on the outputs
                of the dense layer, with dropout probability equal to dropout.
            activation (string): Name of activation function for
                the dense layers.
            in_size (integer): Size of input data
            out_size (integer): Size of output data
        """
        super().__init__()

        self.nn_flag = 1

        # Instantiate ActivationSwitch for dynamic activation selection
        activation_switch = ActivationSwitch()
        act_fn = activation_switch.fn(activation)

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=in_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
        )

        # Define the dense layers
        self.dense_layers = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_out_size),
            act_fn(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_out_size, out_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass of the LSTM layer.
        x, _ = self.lstm(x)
        # Forward pass through dense layers
        out = self.dense_layers(x)

        return out
