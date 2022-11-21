import torch
from torch import nn, optim

class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, hl_size, out_dim):
        super().__init__()
    

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.hl_size = hl_size
        self.out_dim = out_dim

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm1d(num_features=8),
            nn.PReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=self.dropout)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=1,
            ),
            nn.BatchNorm1d(num_features=16),
            nn.PReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=self.dropout)
        )

        self.dense_layer = nn.Sequential(
            nn.Linear(80, self.out_dim)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        out = self.dense_layer(x)

        return out


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def earth_mover_distance(y_true, y_pred):
    return torch.mean(torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)), dim=-1)


def train_cnn (x, y, hyperparams, n_epoch):

    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    out_dim = y[0].size
    n_in = x.shape[1]
   
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    
    trainset = torch.utils.data.TensorDataset(x, y)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32)

    mlp = CNN(n_in, hyperparams['hl_ini_dim'], hyperparams['dropout'], int(hyperparams['hl_ini_dim'] * hyperparams['hl_shrink']), out_dim)
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
       