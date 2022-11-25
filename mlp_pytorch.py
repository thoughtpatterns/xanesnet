import torch
from torch import nn, optim
import math

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import time

# setup tensorboard stuff
layout = {
    "Multi": {
        "loss": ["Multiline", ["loss/train", "loss/validation"]],
    },
}
writer = SummaryWriter(f"/tmp/tensorboard/{int(time.time())}")
writer.add_custom_scalars(layout)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, hl_size, out_dim, act_fn):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.hl_size = hl_size
        self.out_dim = out_dim
        self.act_fn = act_fn

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            self.act_fn,
            nn.Dropout(p=self.dropout),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hl_size),
            self.act_fn,
            nn.Dropout(p=self.dropout),
        )

        self.fc3 = nn.Sequential(nn.Linear(self.hl_size, self.out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)

        return out


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def earth_mover_distance(y_true, y_pred):
    return torch.mean(
        torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)),
        dim=-1,
    )


def activation_function(act_param):

    if act_param == "PReLU" or act_param == "prelu":
        act_fn = nn.PReLU()
    elif act_param == "ReLU" or act_param == "relu":
        act_fn = nn.ReLU(inplace=True)
    elif act_param == "LeakyReLU" or act_param == "leakyrelu":
        act_fn = nn.LeakyReLU(inplace=True)
    elif act_param == "Tanh" or act_param == "tanh":
        act_fn = nn.Tanh()

    return act_fn


def train_mlp(x, y, hyperparams, n_epoch):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_dim = y[0].size
    n_in = x.shape[1]

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    act_fn = activation_function(hyperparams["activation"])

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=hyperparams["batch_size"]
    )

    validset = torch.utils.data.TensorDataset(X_test, y_test)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=hyperparams["batch_size"]
    )

    mlp = MLP(
        n_in,
        hyperparams["hl_ini_dim"],
        hyperparams["dropout"],
        int(hyperparams["hl_ini_dim"] * hyperparams["hl_shrink"]),
        out_dim,
        act_fn,
    )
    mlp.to(device)
    mlp.apply(weight_init)
    mlp.train()
    optimizer = optim.Adam(mlp.parameters(), lr=hyperparams["lr"])

    criterion = nn.MSELoss()

    for epoch in range(n_epoch):
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.float()

            optimizer.zero_grad()
            logps = mlp(inputs)

            loss = criterion(logps, labels)
            loss.mean().backward()
            optimizer.step()
            running_loss += loss.item()

        valid_loss = 0
        mlp.eval()
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.float()

            target = mlp(inputs)

            loss = criterion(target, labels)
            valid_loss += loss.item()

        print("Training loss:", running_loss / len(trainloader))
        print("Validation loss:", valid_loss / len(validloader))

        writer.add_scalar("loss/train", (running_loss / len(trainloader)), epoch)
        writer.add_scalar("loss/validation", (valid_loss / len(validloader)), epoch)

    writer.close()
    return mlp, running_loss / len(trainloader)
