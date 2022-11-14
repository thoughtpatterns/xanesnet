import torch
from torch import nn, optim
import math
from pyemd import emd_samples
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, hl_size, out_dim):
        super().__init__()
    
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.hl_size = hl_size
        self.out_dim = out_dim

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.PReLU(),
            nn.Dropout(p=self.dropout)
        )
         
        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hl_size),
            nn.PReLU(),
            nn.Dropout(p=self.dropout)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(self.hl_size, self.out_dim)
        )

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
    return torch.mean(torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)), dim=-1)


def train_mlp (x, y, hyperparams, n_epoch):

    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    out_dim = y[0].size
    n_in = x.shape[1]
   
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    
    trainset = torch.utils.data.TensorDataset(x, y)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32)

    mlp = MLP(n_in, hyperparams['hl_ini_dim'], hyperparams['dropout'], int(hyperparams['hl_ini_dim'] * hyperparams['hl_shrink']), out_dim)
    mlp.to(device)
    mlp.apply(weight_init)
    mlp.train()
    optimizer = optim.Adam(mlp.parameters(), lr=hyperparams['lr'])
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
            
        # print(running_loss/len(trainloader))

    return mlp, running_loss/len(trainloader)
        