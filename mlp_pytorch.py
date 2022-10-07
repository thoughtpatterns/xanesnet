# transforming tensorflow mlp to pytorch
# the model summary from tensorflow
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 256)               12800

#  activation (Activation)     (None, 256)               0

#  dropout_ (Dropout_)         (None, 256)               0

#  dense_1 (Dense)             (None, 230)               59110

#  activation_1 (Activation)   (None, 230)               0

#  dropout__1 (Dropout_)       (None, 230)               0

#  dense_2 (Dense)             (None, 376)               86856

#  activation_2 (Activation)   (None, 376)               0

# =================================================================
# Total params: 158,766
# Trainable params: 158,766
# Non-trainable params: 0

import torch
from torch import nn, optim
import math

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, hl_shrink, out_dim):
        super().__init__()
    
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.hl_shrink = hl_shrink
        self.out_dim = out_dim

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(p=0.25)
        )
         
        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.Tanh(),
            nn.Dropout(p=0.25)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(512, self.out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)

        return out

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        # nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        nn.init.zeros_(m.bias)
        # nn.init.constant_(m.bias)


def train_mlp (x, y, hyperparams, n_epoch):

    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    out_dim = y[0].size
    n_in = x.shape[1]
   
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    
    trainset = torch.utils.data.TensorDataset(x, y)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32)
   
    mlp = MLP(n_in, 512, hyperparams['dropout'], hyperparams['hl_shrink'], out_dim)
    mlp.to(device)
    mlp.apply(weight_init)
    mlp.train()
    optimizer = optim.Adam(mlp.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    for epoch in range(n_epoch):
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.float()
            
            optimizer.zero_grad()
            logps = mlp(inputs)
            
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # print(loss.item())
            
        print(running_loss/len(trainloader))

    return epoch, mlp, optimizer

        