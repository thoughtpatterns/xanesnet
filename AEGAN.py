import torch
from torch import nn, optim

from torch import nn
import torch
from torch.autograd import Variable
import numpy as np


class AEGANTrainer(nn.Module):
    def __init__(self, **kwargs):
        super(AEGANTrainer, self).__init__()
        # Initiate the networks
        self.gen_a = AEGen(n_input_features = kwargs['input_dim_a'], n_output_features = kwargs['input_dim_a'])  # auto-encoder for domain a
        self.gen_b = AEGen(n_input_features = kwargs['input_dim_b'], n_output_features = kwargs['input_dim_b'])  # auto-encoder for domain b

        self.enc_shared = SharedLayer()
        self.dec_shared = SharedLayer()

        self.dis_a = Dis(n_input_features = kwargs['input_dim_a'])  # discriminator for domain a
        self.dis_b = Dis(n_input_features = kwargs['input_dim_b'])  # discriminator for domain b
        
        self.dis_opt = torch.optim.Adam(self.parameters(),lr = 1e-3,weight_decay=1e-5)
        self.gen_opt = torch.optim.Adam(self.parameters(),lr = 1e-3,weight_decay=1e-5)

        # Network weight initialization
        self.apply(weight_init)

    def reconstruct_all_predict_all(self,x_a,x_b):

        enc_a = self.gen_a.encode(x_a)
        enc_b = self.gen_b.encode(x_b)

        shared_enc_a = self.enc_shared.forward(enc_a)
        shared_enc_b = self.enc_shared.forward(enc_b)

        shared_dec_a = self.dec_shared.forward(shared_enc_a)
        shared_dec_b = self.dec_shared.forward(shared_enc_b)

        x_ba = self.gen_a.decode(shared_dec_b)
        x_ab = self.gen_b.decode(shared_dec_a)

        x_a_recon = self.gen_a.decode(shared_dec_a)
        x_b_recon = self.gen_b.decode(shared_dec_b)

        return x_a_recon, x_b_recon, x_ba, x_ab


    def recon_criterion(self, pred, target):
        loss_fn = nn.MSELoss()
        loss = loss_fn(pred,target)
        return loss

    def forward(self, x_a, x_b):       
        enc_a = self.gen_a.encode(x_a)
        enc_b = self.gen_b.encode(x_b)

        shared_enc_a = self.enc_shared.forward(enc_a)
        shared_enc_b = self.enc_shared.forward(enc_b)

        shared_dec_a = self.dec_shared.forward(shared_enc_a)
        shared_dec_b = self.dec_shared.forward(shared_enc_b)

        x_ba = self.gen_a.decode(shared_dec_b)
        x_ab = self.gen_b.decode(shared_dec_a)
        return x_ab, x_ba

    def gen_update(self, x_a, x_b):
        self.gen_opt.zero_grad()

        # encode
        enc_a = self.gen_a.encode(x_a)
        enc_b = self.gen_b.encode(x_b)

        # encode shared layer
        shared_enc_a = self.enc_shared.forward(enc_a)
        shared_enc_b = self.enc_shared.forward(enc_b)
        # decode shared layer
        shared_dec_a = self.dec_shared.forward(shared_enc_a)
        shared_dec_b = self.dec_shared.forward(shared_enc_b)

        # decode (within domain)
        x_a_recon = self.gen_a.decode(shared_dec_a)
        x_b_recon = self.gen_b.decode(shared_dec_b)

        # decode (cross domain)
        x_ba = self.gen_a.decode(shared_dec_b)
        x_ab = self.gen_b.decode(shared_dec_a)

        # scale loss by mean-maximum value of input
        a_max = torch.max(torch.mean(x_a,dim = 1))
        b_max = torch.max(torch.mean(x_b,dim = 1))

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)/a_max
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)/b_max
        self.loss_gen_cyc_x_a = self.recon_criterion(x_ba, x_a)/a_max
        self.loss_gen_cyc_x_b = self.recon_criterion(x_ab, x_b)/b_max
        # GAN loss
        # self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        # self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # total loss
        self.loss_gen_total = self.loss_gen_cyc_x_a + self.loss_gen_cyc_x_b + self.loss_gen_recon_x_a + self.loss_gen_recon_x_b

        self.loss_gen_total.backward()
        self.gen_opt.step()

    def dis_update(self, x_a, x_b):
        self.dis_opt.zero_grad()
        # encode
        y_a = self.gen_a.encode(x_a)
        y_b = self.gen_b.encode(x_b)

        # encode shared layer
        shared_enc_a = self.enc_shared.forward(y_a)
        shared_enc_b = self.enc_shared.forward(y_b)
        # decode shared layer
        shared_dec_a = self.dec_shared.forward(shared_enc_a)
        shared_dec_b = self.dec_shared.forward(shared_enc_b)


        # decode (cross domain)
        x_ba = self.gen_a.decode(shared_dec_b)
        x_ab = self.gen_b.decode(shared_dec_a)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(shared_dec_a)
        x_b_recon = self.gen_b.decode(shared_dec_b)

        # Discriminator loss for real inputs
        self.loss_dis_adv_a = self.dis_a.calc_gen_loss(x_a)
        self.loss_dis_adv_b = self.dis_b.calc_gen_loss(x_b)

        self.loss_gen_adv_a = self.dis_a.calc_dis_loss(x_ba,x_a)
        self.loss_gen_adv_b = self.dis_b.calc_dis_loss(x_ab,x_b)

        self.loss_gen_recon_a = self.dis_a.calc_dis_loss(x_a_recon,x_a)
        self.loss_gen_recon_b = self.dis_b.calc_dis_loss(x_b_recon,x_b)

        self.loss_real = 0.5*(self.loss_dis_adv_a + self.loss_dis_adv_b)
        self.loss_fake = 0.25*(self.loss_gen_adv_a  + self.loss_gen_recon_a + self.loss_gen_adv_b + self.loss_gen_recon_b)

        self.loss_dis_total = (self.loss_real + self.loss_fake)

        self.loss_dis_total.backward()
        self.dis_opt.step()


class SharedLayer(nn.Module):
    # Autoencoder architecture
    def __init__(self, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128))

    def forward(self,features):
        z = self.layers(features)
        return z


class AEGen(nn.Module):
    # Autoencoder architecture
    def __init__(self, **kwargs):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(kwargs["n_input_features"], 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128))
        self.dec = nn.Sequential(
            nn.Linear(128,128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256,kwargs["n_output_features"]),
            nn.PReLU())

    def encode(self, features):
        hiddens = self.enc(features)
        return hiddens

    def decode(self, hiddens):
        hiddens = self.dec(hiddens)
        return hiddens

    def forward(self,features):
        hidden = self.encode(features)
        recon = self.decode(hidden)
        return recon,hidden


class Dis(nn.Module):
    # Discriminator architecture
    def __init__(self, **kwargs):
        super().__init__()
        self.discriminate = nn.Sequential(
            nn.Linear(kwargs["n_input_features"], 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid())
    
    def forward(self,features):
        discriminate = self.discriminate(features)
        return discriminate

    def calc_dis_loss(self,input_fake,input_real):
        # Calculate the loss to train D
        out0 = self.forward(input_fake)
        out1 = self.forward(input_real)
        loss_fn = nn.BCELoss()
        loss = loss_fn(out0,out1)
        return loss

    def calc_gen_loss(self,input_fake):
        # Calculate the loss to train G
        out0 = self.forward(input_fake)
        ones = torch.ones((input_fake.size(0),1))
        loss_fn = nn.BCELoss()
        loss = loss_fn(out0,ones)
        return loss


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.bias)
        # nn.init.xavier_uniform_(m.weight)



def train_aegan (x, y, hyperparams, n_epoch):

    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    n_x_features = x.shape[1]
    n_y_features = y.shape[1]
    
    trainset = torch.utils.data.TensorDataset(x, y)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32)


    model = AEGANTrainer(input_dim_a = n_x_features,input_dim_b = n_y_features)
    model.to(device)
    model.train()
    lossfn = nn.MSELoss()

    running_loss = [None]*n_epoch
    running_loss_recon_structure = [None]*n_epoch
    running_loss_recon_spectrum = [None]*n_epoch
    running_loss_pred_structure = [None]*n_epoch
    running_loss_pred_spectrum = [None]*n_epoch


    for epoch in range(n_epoch):
        loss_recon_structure = 0
        loss_recon_spectrum = 0
        loss_pred_structure = 0
        loss_pred_spectrum = 0
        for inputs_structure, inputs_spectrum in trainloader:
            inputs_structure, inputs_spectrum = inputs_structure.to(device), inputs_spectrum.to(device)
            inputs_structure, inputs_spectrum = inputs_structure.float(), inputs_spectrum.float()

            model.dis_update(inputs_structure,inputs_spectrum)
            model.gen_update(inputs_structure,inputs_spectrum)

            recon_structure, recon_spectrum, pred_structure, pred_spectrum  = model.reconstruct_all_predict_all(inputs_structure,inputs_spectrum)

            # xyz-xyz
            loss_recon_structure += lossfn(recon_structure,inputs_structure)
            # xanes-xanes
            loss_recon_spectrum += lossfn(recon_spectrum,inputs_spectrum)
            # xyz-xanes
            loss_pred_spectrum += lossfn(pred_spectrum,inputs_spectrum)
            # xanes-xyz
            loss_pred_structure += lossfn(pred_structure,inputs_structure)
            

        running_loss_recon_structure[epoch] = loss_recon_structure.detach().numpy()/len(trainloader)
        running_loss_recon_spectrum[epoch] = loss_recon_spectrum.detach().numpy()/len(trainloader)
        running_loss_pred_structure[epoch] = loss_pred_structure.detach().numpy()/len(trainloader)
        running_loss_pred_spectrum[epoch] = loss_pred_spectrum.detach().numpy()/len(trainloader)

        running_loss[epoch] = running_loss_recon_structure[epoch] + running_loss_recon_spectrum[epoch] + running_loss_pred_structure[epoch] + running_loss_pred_spectrum[epoch]
        print(f">>> Epoch {epoch}...")
        print(f"[+] Total loss:            {running_loss[epoch]}")
        print(f"[+] Reconstruction losses: {running_loss_recon_structure[epoch]},{running_loss_recon_spectrum[epoch]}")
        print(f"[+] Prediction losses:     {running_loss_pred_structure[epoch]},{running_loss_pred_spectrum[epoch]}")

        losses = {'total': running_loss, \
                 'loss_xyz_recon': running_loss_recon_structure, \
                 'loss_xanes_recon' : running_loss_recon_spectrum, \
                 'loss_xyz_pred' : running_loss_pred_structure, \
                 'loss_xanes_pred' : running_loss_pred_spectrum}


    return losses, model




