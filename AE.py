import torch
from torch import nn, optim

class AE (nn.Module):
    def __init__(self, input_size, hidden_size, dropout, hl_shrink, out_dim):
        super().__init__()
    
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.hl_shrink = hl_shrink
        self.out_dim = out_dim

        self.encoder_hidden_1 = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.PReLU()
        )
         
        self.encoder_hidden_2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.PReLU()
        )

        self.encoder_output = nn.Sequential(
            nn.Linear(128, 128),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.Dropout(p=0.3),
            nn.PReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, self.out_dim),
            nn.PReLU(),
        )

        self.decoder_hidden_1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.PReLU()
        )

        self.decoder_hidden_2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.PReLU()
        )

        self.decoder_output = nn.Sequential(
            nn.Linear(256, self.input_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.encoder_hidden_1(x)
        x = self.encoder_hidden_2(x)
        
        latent_space = self.encoder_output(x)
        pred_y = self.fc1(latent_space)
        pred_y = self.fc2(pred_y)
        
        out = self.decoder_hidden_1(latent_space)
        out = self.decoder_hidden_2(out)
        recon = self.decoder_output(out)

        return recon, pred_y


def train_ae (x, y, hyperparams, n_epoch):

    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    out_dim = x[0].size
    n_in = y.shape[1]
   
    xyz = torch.from_numpy(x)
    xanes = torch.from_numpy(y)
    
    trainset = torch.utils.data.TensorDataset(xanes, xyz)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32)
   
    model = AE(n_in, 512, hyperparams['dropout'], hyperparams['hl_shrink'], out_dim)
    model.to(device)
    # mlp.apply(weight_init)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()

    for epoch in range(n_epoch):
        running_loss = 0
        running_recon = 0
        running_pred = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.float()

            optimizer.zero_grad()
            recon_input, outputs = model(inputs)
            

            loss_recon = criterion(recon_input, inputs) 
            loss_pred = criterion(outputs, labels)
            loss = loss_recon + (1 * loss_pred)
            # loss = custom_loss(logps, labels)
            # loss = earth_mover_distance(labels, logps)
            # print(loss.shape)
            # loss.backward()
            loss.mean().backward()

            optimizer.step()
            running_loss += loss.mean().item()
            running_recon += loss_recon.mean().item()
            running_pred += loss_pred.mean().item()
            # print(loss.item())
            
        print("total loss:", running_loss/len(trainloader))
        # print("recon loss:", running_recon/len(trainloader))
        # print("pred loss:", running_pred/len(trainloader))


    return epoch, model, optimizer