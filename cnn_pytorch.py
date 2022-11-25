import torch
from torch import nn, optim
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


class CNN(nn.Module):
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
        print(self.conv1_shape)
        print(self.conv2_shape)

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.out_channel,
                kernel_size=self.kernel_size,
                stride=self.stride,
            ),
            nn.BatchNorm1d(num_features=self.out_channel),
            nn.PReLU(),
            # nn.MaxPool1d(2),
            # nn.Dropout(p=self.dropout),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.out_channel,
                out_channels=self.out_channel * self.channel_mul,
                kernel_size=self.kernel_size,
                stride=self.stride,
            ),
            nn.BatchNorm1d(num_features=self.out_channel * self.channel_mul),
            nn.PReLU(),
            # nn.MaxPool1d(2),
            nn.Dropout(p=self.dropout),
        )

        self.dense_layer = nn.Sequential(
            nn.Linear(
                self.conv2_shape * (self.out_channel * self.channel_mul), self.out_dim
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        out = self.dense_layer(x)

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
    
def train_cnn(x, y, hyperparams, n_epoch):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_dim = y[0].size
    n_in = x.shape[1]

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32)

    validset = torch.utils.data.TensorDataset(X_test, y_test)
    validloader = torch.utils.data.DataLoader(validset, batch_size=32)

    cnn = CNN(
        n_in,
        hyperparams["out_channel"],
        hyperparams["channel_mul"],
        hyperparams["hidden_layer"],
        out_dim,
        hyperparams["dropout"],
        hyperparams["kernel_size"],
        hyperparams["stride"],
    )

    cnn.to(device)
    cnn.apply(weight_init)
    cnn.train()
    optimizer = optim.Adam(cnn.parameters(), lr=hyperparams["lr"])

    criterion = nn.MSELoss()

    for epoch in range(n_epoch):
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.float()

            optimizer.zero_grad()
            logps = cnn(inputs)

            loss = criterion(logps, labels)
            loss.mean().backward()
            optimizer.step()
            running_loss += loss.item()

        valid_loss = 0
        cnn.eval()
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.float()

            target = cnn(inputs)

            loss = criterion(target, labels)
            valid_loss += loss.item()

        print("Training loss:", running_loss / len(trainloader))
        print("Validation loss:", valid_loss / len(validloader))

        writer.add_scalar("loss/train", (running_loss/len(trainloader)), epoch)
        writer.add_scalar("loss/validation", (valid_loss/len(validloader)), epoch)

    writer.close()

    return cnn, running_loss / len(trainloader)
