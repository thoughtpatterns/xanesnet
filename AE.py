import torch
from torch import nn, optim
import numpy as np
from sklearn import preprocessing


class AE_mlp(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, hl_shrink, out_dim):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.hl_shrink = hl_shrink
        self.out_dim = out_dim

        self.encoder_hidden_1 = nn.Sequential(
            nn.Linear(self.input_size, 256), nn.PReLU()
        )

        self.encoder_hidden_2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.PReLU(),
        )

        self.encoder_output = nn.Sequential(
            nn.Linear(128, 64),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.Dropout(p=0.2),
            nn.PReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(64, self.out_dim),
        )

        self.decoder_hidden_1 = nn.Sequential(
            nn.Linear(64, 128),
            nn.PReLU(),
        )

        self.decoder_hidden_2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.PReLU(),
        )

        self.decoder_output = nn.Sequential(
            nn.Linear(256, self.input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.encoder_hidden_1(x)
        x = self.encoder_hidden_2(x)

        latent_space = self.encoder_output(x)
        in_fc = self.fc1(latent_space)
        pred_y = self.fc2(in_fc)

        out = self.decoder_hidden_1(latent_space)
        out = self.decoder_hidden_2(out)
        recon = self.decoder_output(out)

        return recon, pred_y


class AE_cnn(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, hl_shrink, out_dim):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.hl_shrink = hl_shrink
        self.out_dim = out_dim

        self.encoder_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2),
            # nn.BatchNorm1d(num_features=16),
            nn.PReLU(),
            # nn.MaxPool1d(2),
        )

        self.encoder_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            # nn.BatchNorm1d(num_features=32),
            nn.PReLU(),
            # nn.MaxPool1d(2),
        )

        self.encoder_output = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            # nn.BatchNorm1d(num_features=64),
            nn.PReLU(),
            # nn.MaxPool1d(2),
        )

        self.dense_layers = nn.Sequential(
            nn.Linear(1600, 128),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.out_dim),
        )

        self.decoder_layer1 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=64,
                out_channels=32,
                kernel_size=5,
                stride=2,
                output_padding=1,
            ),
            # nn.BatchNorm1d(num_features=32),
            nn.PReLU(),
            # nn.MaxPool1d(2),
        )

        self.decoder_layer2 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=32,
                out_channels=16,
                kernel_size=5,
                stride=2,
                output_padding=1,
            ),
            # nn.BatchNorm1d(num_features=16),
            nn.PReLU(),
            # nn.MaxPool1d(2),
        )

        self.decoder_output = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=16,
                out_channels=1,
                kernel_size=5,
                stride=2,
                output_padding=1,
                padding=1,
            ),
            # nn.BatchNorm1d(num_features=1),
            nn.PReLU(),
            # nn.MaxPool1d(2),
        )

    def forward(self, x):

        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)

        out = self.encoder_layer1(x)
        out = self.encoder_layer2(out)
        out = self.encoder_output(out)

        pred = out.view(out.size(0), -1)
        pred = self.dense_layers(pred)

        recon = self.decoder_layer1(out)
        recon = self.decoder_layer2(recon)
        recon = self.decoder_output(recon)
        recon = recon.squeeze(dim=1)

        return recon, pred


def train_ae(x, y, hyperparams, n_epoch):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_dim = y[0].size
    n_in = x.shape[1]

    le = preprocessing.LabelEncoder()

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    trainset = torch.utils.data.TensorDataset(x, y)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32)

    model = AE_mlp(n_in, 512, hyperparams["dropout"], hyperparams["hl_shrink"], out_dim)
    model.to(device)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0000)
    criterion = nn.MSELoss()

    for epoch in range(n_epoch):
        running_loss = 0

        for inputs, labels in trainloader:
            inputs, labels = (
                inputs.to(device),
                labels.to(device),
                # element_label.to(device),
            )
            inputs, labels = (
                inputs.float(),
                labels.float(),
                # element_label.long(),
            )

            optimizer.zero_grad()
            recon_input, outputs = model(inputs)

            loss_recon = criterion(recon_input, inputs)
            loss_pred = criterion(outputs, labels)

            loss = loss_recon + loss_pred
            loss.backward()

            optimizer.step()
            running_loss += loss.mean().item()

        print("total loss:", running_loss / len(trainloader))

    return epoch, model, optimizer
