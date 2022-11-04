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
            nn.Linear(self.input_size, self.hidden_size), nn.PReLU()
        )

        self.encoder_hidden_2 = nn.Sequential(
            nn.Linear(self.hidden_size, int(self.hidden_size * self.hl_shrink)),
            nn.PReLU(),
        )

        # self.encoder_output = nn.Sequential(
        #     nn.Linear(128, 64),
        # )

        self.fc1 = nn.Sequential(
            nn.Linear(int(self.hidden_size * self.hl_shrink), 64),
            nn.Dropout(self.dropout),
            nn.PReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(64, self.out_dim),
        )

        self.decoder_hidden_1 = nn.Sequential(
            nn.Linear(int(self.hidden_size * self.hl_shrink), self.hidden_size),
            nn.PReLU(),
        )

        self.decoder_hidden_2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.input_size),
            nn.PReLU(),
        )

        # self.decoder_output = nn.Sequential(
        #     nn.Linear(256, self.input_size),
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.encoder_hidden_1(x)
        latent_space = self.encoder_hidden_2(x)

        # latent_space = self.encoder_output(x)
        in_fc = self.fc1(latent_space)
        pred_y = self.fc2(in_fc)

        out = self.decoder_hidden_1(latent_space)
        recon = self.decoder_hidden_2(out)
        # recon = self.decoder_output(out)

        return recon, pred_y


class AE_cnn(nn.Module):
    def __init__(
        self,
        input_size,
        out_channel,
        channel_mul,
        hidden_layer,
        out_dim,
        dropout,
        kernel_size,
        stride,
    ):
        super().__init__()

        self.input_size = input_size
        self.out_channel = out_channel
        self.channel_mul = channel_mul
        self.hidden_layer = hidden_layer
        self.out_dim = out_dim
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv1_shape = int(((self.input_size - self.kernel_size) / self.stride) + 1)
        self.conv2_shape = int(
            ((self.conv1_shape - self.kernel_size) / self.stride) + 1
        )
        self.conv3_shape = int(
            ((self.conv2_shape - self.kernel_size) / self.stride) + 1
        )

        self.convt1_shape = int(((self.conv3_shape - 1) * 2) + self.kernel_size)
        if self.convt1_shape != self.conv2_shape:
            if self.convt1_shape > self.conv2_shape:
                self.p1 = self.convt1_shape - self.conv2_shape
                self.op1 = 0
            elif self.convt1_shape < self.conv2_shape:
                self.op1 = self.conv2_shape - self.convt1_shape
                self.p1 = 0
        else:
            self.p1 = 0
            self.op1 = 0

        self.convt2_shape = int(((self.conv2_shape - 1) * 2) + self.kernel_size)
        # print(self.conv1_shape)
        if self.convt2_shape != self.conv1_shape:
            if self.convt2_shape > self.conv1_shape:
                self.p2 = self.convt2_shape - self.conv1_shape
                self.op2 = 0
                # print(self.op2)
            elif self.convt2_shape < self.conv1_shape:
                # print("here")
                self.op2 = self.conv1_shape - self.convt2_shape
                self.p2 = 0
        else:
            self.p2 = 0
            self.op2 = 0
            # print(self.op2)

        # print(self.convt2_shape)

        self.convt3_shape = int(((self.conv1_shape - 1) * 2) + self.kernel_size)
        if self.convt3_shape != self.input_size:
            if self.convt3_shape > self.input_size:
                self.p3 = self.convt3_shape - self.input_size
                self.op3 = 0
            elif self.convt3_shape < self.input_size:
                self.op3 = self.input_size - self.convt3_shape
                self.p3 = 0
        else:
            self.p3 = 0
            self.op3 = 0

        # print(self.convt_shape)
        # print(self.input_size)

        self.encoder_layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.out_channel,
                kernel_size=self.kernel_size,
                stride=self.stride,
            ),
            # nn.BatchNorm1d(num_features=16),
            nn.PReLU(),
            # nn.MaxPool1d(2),
        )

        self.encoder_layer2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.out_channel,
                out_channels=int(self.out_channel * self.channel_mul),
                kernel_size=self.kernel_size,
                stride=self.stride,
            ),
            # nn.BatchNorm1d(num_features=32),
            nn.PReLU(),
            # nn.MaxPool1d(2),
        )

        self.encoder_output = nn.Sequential(
            nn.Conv1d(
                in_channels=int(self.out_channel * self.channel_mul),
                out_channels=int(self.out_channel * self.channel_mul * 2),
                kernel_size=self.kernel_size,
                stride=self.stride,
            ),
            # nn.BatchNorm1d(num_features=64),
            nn.PReLU(),
            # nn.MaxPool1d(2),
        )
        # self.encoder_output = nn.Conv1d(
        #     in_channels=32, out_channels=64, kernel_size=5, stride=2
        # )

        self.dense_layers = nn.Sequential(
            nn.Linear(
                self.conv3_shape * int(self.out_channel * self.channel_mul * 2),
                self.hidden_layer,
            ),
            # nn.Lazylinear(128),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_layer, self.out_dim),
        )

        self.decoder_layer1 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=int(self.out_channel * self.channel_mul * 2),
                out_channels=int(self.out_channel * self.channel_mul),
                kernel_size=self.kernel_size,
                stride=self.stride,
                output_padding=self.op1,
                padding=self.p1,
            ),
            # nn.BatchNorm1d(num_features=32),
            nn.PReLU(),
            # nn.MaxPool1d(2),
        )

        self.decoder_layer2 = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=int(self.out_channel * self.channel_mul),
                out_channels=self.out_channel,
                kernel_size=self.kernel_size,
                stride=self.stride,
                output_padding=self.op2,
                padding=self.p2,
            ),
            # nn.BatchNorm1d(num_features=16),
            nn.PReLU(),
            # nn.MaxPool1d(2),
        )

        self.decoder_output = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=self.out_channel,
                out_channels=1,
                kernel_size=self.kernel_size,
                stride=self.stride,
                output_padding=self.op3,
                padding=self.p3,
            ),
            # nn.BatchNorm1d(num_features=1),
            nn.PReLU(),
            # nn.MaxPool1d(2),
        )

    # def encode(self, x):
    def forward(self, x):

        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)
        # x = nn.flatten(x)
        # print(x.shape)

        self.out = self.encoder_layer1(x)
        # print(self.out.shape)
        self.out = self.encoder_layer2(self.out)
        # print(self.out.shape)
        self.out = self.encoder_output(self.out)
        # print(self.out.shape)

        pred = self.out.view(self.out.size(0), -1)
        # print(pred.shape)
        self.hl = pred.size(-1)
        # print(self.hl)
        pred = self.dense_layers(pred)
        # print(pred.shape)
        # print(out.shape)

        recon = self.decoder_layer1(self.out)
        # print(recon.shape)
        recon = self.decoder_layer2(recon)
        # print(recon.shape)
        recon = self.decoder_output(recon)
        # print(recon.shape)
        recon = recon.squeeze(dim=1)
        # print(recon.shape)

        return recon, pred


def train_ae(x, y, hyperparams, n_epoch):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_dim = y[0].size
    n_in = x.shape[1]
    # print(x.shape)

    le = preprocessing.LabelEncoder()

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    trainset = torch.utils.data.TensorDataset(x, y)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32)

    # model = AE_mlp(
    #     n_in,
    #     hyperparams["hl_ini_dim"],
    #     hyperparams["dropout"],
    #     hyperparams["hl_shrink"],
    #     out_dim,
    # )

    model = AE_cnn(
        n_in,
        hyperparams["out_channel"],
        hyperparams["channel_mul"],
        hyperparams["hidden_layer"],
        out_dim,
        hyperparams["dropout"],
        hyperparams["kernel_size"],
        hyperparams["stride"],
    )

    model.to(device)

    model.train()
    optimizer = optim.Adam(
        model.parameters(), lr=hyperparams["lr"], weight_decay=0.0000
    )
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
            # print(inputs.shape)

            optimizer.zero_grad()
            recon_input, outputs = model(inputs)
            # outputs = model(inputs)
            # print(outputs.shape)

            loss_recon = criterion(recon_input, inputs)
            loss_pred = criterion(outputs, labels)

            loss = loss_recon + loss_pred
            loss.backward()

            optimizer.step()
            running_loss += loss.mean().item()

        print("total loss:", running_loss / len(trainloader))

    return epoch, model, optimizer
