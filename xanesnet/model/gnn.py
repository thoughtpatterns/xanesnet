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

from typing import Optional

import numpy as np
import torch
from torch import nn
import torch_geometric.nn as geom_nn

from xanesnet.model.base_model import Model
from xanesnet.utils_model import ActivationSwitch

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GATv2": geom_nn.GATv2Conv,
    "GraphConv": geom_nn.GraphConv,
}


class GNN(Model):
    """
    A class for constructing a customisable GNN (Graph Neural Network) model.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        mlp_feat_size: int,
        layer_name: str = "GATv2",
        layer_params: Optional[dict] = None,
        hidden_size: int = "512",
        dropout: float = 0.2,
        num_hidden_layers: int = 5,
        num_mlp_hidden_layers: int = 3,
        activation: str = "prelu",
    ):
        """
        Args:
            in_size (integer): Input size
            out_size (integer): Output size
            layer_name (string): Name of GNN layer (GAT, GATv2, GCN, GraphConv)
            layer_params (dict): parameters pass to GNN layers
            hidden_size (integer): Size of the hidden layer.
            mlp_feat_size (integer): Size of features added to the MLP layers.
                This should be the sum of feature sizes from all descriptors
            dropout (float): If none-zero, add dropout layer on the outputs
                of each hidden layer with dropout probability equal to dropout.
            num_hidden_layers (integer): Number of hidden layers
                in the network.
            activation (string): Name of activation function applied
                to the hidden layers.
        """
        super().__init__()

        if layer_params is None:
            layer_params = {
                "heads": 2,
                "concat": True,
                "edge_dim": 16,
            }
        heads = layer_params.get("heads")

        self.nn_flag = 1
        act_fn = ActivationSwitch().fn(activation)
        gnn_layer = gnn_layer_by_name[layer_name]

        # ---- GNN Layers ----
        layers = []
        for i in range(num_hidden_layers - 1):
            layers += [
                gnn_layer(
                    in_channels=in_size, out_channels=hidden_size, **layer_params
                ),
                nn.BatchNorm1d(hidden_size * heads),
                act_fn(),
                nn.Dropout(dropout),
            ]

            in_size = hidden_size * heads

        # GNN output layer
        layers += [
            gnn_layer(in_channels=in_size, out_channels=hidden_size, **layer_params)
        ]
        self.gnn_layers = nn.ModuleList(layers)

        # ---- MLP Layers ----
        mlp_in_size = in_size + mlp_feat_size
        mlp_hidden_size = in_size + mlp_feat_size * 2

        layers = []
        for i in range(num_mlp_hidden_layers):
            # First layer
            if i == 0:
                in_dim = mlp_in_size
                out_dim = mlp_hidden_size
            # Final layer
            elif i == num_mlp_hidden_layers - 1:
                in_dim = mlp_hidden_size
                out_dim = out_size
            # Hidden layers
            else:
                in_dim = mlp_hidden_size
                out_dim = mlp_hidden_size

            layer = [nn.Linear(in_dim, out_dim)]

            if i < num_mlp_hidden_layers - 1:
                layer += [nn.Dropout(dropout), act_fn()]

            layers.append(nn.Sequential(*layer))

        self.mlp_layers = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, edge_attr, graph_attr, edge_idx, batch_idx
    ) -> torch.Tensor:
        """
        Inputs:
            x - Input features per node
            edge_index - List of edge index pairs
            batch_idx - Index of batch element for each node
        """
        for layer in self.gnn_layers:
            if isinstance(layer, geom_nn.MessagePassing):
                if isinstance(layer, geom_nn.GATv2Conv) or (
                    isinstance(layer, geom_nn.GATConv) and edge_attr is not None
                ):
                    x = layer(x, edge_idx, edge_attr)
                else:
                    x = layer(x, edge_idx)
            else:
                x = layer(x)

        #       # Specific node
        #       node_idx = 0
        #       node_feature = x[node_idx]
        #       out = self.head(node_feature)
        # Average pooling
        x = geom_nn.global_mean_pool(x, batch_idx)
        # Append graph feature before sending to mlp layers
        x = torch.cat((x, graph_attr), dim=1)
        out = self.mlp_layers(x)

        return out
