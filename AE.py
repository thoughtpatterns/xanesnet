import torch
from torch import nn, optim
import numpy as np
from sklearn import preprocessing

class AE (nn.Module):
    def __init__(self, input_size, hidden_size, dropout, hl_shrink, out_dim, n_class):
        super().__init__()
    
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.hl_shrink = hl_shrink
        self.out_dim = out_dim
        self.n_class = n_class

        self.encoder_hidden_1 = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.PReLU()
        )
         
        self.encoder_hidden_2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.PReLU()
        )

        self.encoder_output = nn.Sequential(
            nn.Linear(256, 128),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(p=0.2),
            nn.PReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(64, self.out_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, self.n_class),
        )

        self.decoder_hidden_1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.PReLU()
        )

        self.decoder_hidden_2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.PReLU()
        )

        self.decoder_output = nn.Sequential(
            nn.Linear(512, self.input_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.encoder_hidden_1(x)
        x = self.encoder_hidden_2(x)
        
        latent_space = self.encoder_output(x)
        in_fc = self.fc1(latent_space)
        pred_y = self.fc2(in_fc)

        e_class = self.classifier(in_fc)
        
        out = self.decoder_hidden_1(latent_space)
        out = self.decoder_hidden_2(out)
        recon = self.decoder_output(out)

        return recon, pred_y, e_class


def train_ae (x, y, element, hyperparams, n_epoch):

    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    out_dim = y[0].size
    n_in = x.shape[1]
    n_class = len(np.unique(element))

    le = preprocessing.LabelEncoder()
    element_label = le.fit_transform(element)
   
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    element_label = torch.as_tensor(element_label)
    
    trainset = torch.utils.data.TensorDataset(x, y, element_label)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32)
   
    model = AE(n_in, 512, hyperparams['dropout'], hyperparams['hl_shrink'], out_dim, n_class)
    model.to(device)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.004)
    criterion = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(n_epoch):
        running_loss = 0
        running_recon = 0
        running_pred = 0
        running_class = 0
        for inputs, labels, element_label in trainloader:
            inputs, labels, element_label = inputs.to(device), labels.to(device), element_label.to(device)
            inputs, labels, element_label = inputs.float(), labels.float(), element_label.long()

            optimizer.zero_grad()
            recon_input, outputs, class_pred = model(inputs)
            
            # print(class_pred)

            loss_recon = criterion(recon_input, inputs) 
            loss_pred = criterion(outputs, labels)
            loss_class = ce_loss(class_pred, element_label)
            loss = loss_recon + loss_pred + loss_class
            # loss = custom_loss(logps, labels)
            # loss = earth_mover_distance(labels, logps)
            # print(loss.shape)
            # loss.backward()
            loss.mean().backward()

            optimizer.step()
            running_loss += loss.mean().item()
            running_recon += loss_recon.mean().item()
            running_pred += loss_pred.mean().item()
            running_class += loss_class.mean().item()
            # print(loss.item())
            
        print("total loss:", running_loss/len(trainloader))
        print("recon loss:", running_recon/len(trainloader))
        print("pred loss:", running_pred/len(trainloader))
        print("class loss:", running_class/len(trainloader))


    return epoch, model, optimizer