import numpy as np
import pickle as pickle
import tqdm as tqdm

from pathlib import Path
from inout import load_xyz
from inout import save_xyz
from inout import load_xanes
from inout import save_xanes

from utils import unique_path
from utils import linecount
from utils import list_filestems
from utils import print_cross_validation_scores
from structure.rdc import RDC
from structure.wacsf import WACSF
from spectrum.xanes import XANES

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from networks import AEGen, Dis, weight_init, GAN_AE_Trainer

from sklearn.preprocessing import minmax_scale

#-----------------------------------------------------------------------------

train_xyz = np.loadtxt('data-training-xyz.dat')
train_xanes = np.loadtxt('data-training-xanes.dat')

valid_xyz = np.loadtxt('data-valid-xyz.dat')
valid_xanes =np.loadtxt('data-valid-xanes.dat')

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


n_x_features = train_xyz.shape[1]
n_y_features = train_xanes.shape[1]


print('>>> Creating model...')
trainer = GAN_AE_Trainer(input_dim_a = n_x_features,input_dim_b = n_y_features).to(device)
print('>>> done!')


train_xyz = torch.tensor(train_xyz).float()
train_xanes = torch.tensor(train_xanes).float()
valid_xyz = torch.tensor(valid_xyz).float()
valid_xanes = torch.tensor(valid_xanes).float()


#-----------------------------------------------------------------------------
n_epoch = 10
bsz = 50
n_train_samples = train_xyz.shape[0]
seed = 2022
torch.manual_seed(seed)


print('>>> Training model....')
for i in range(n_epoch):
	print(f'>>> Epoch {i+1}')
	shuffle_idx=torch.randperm(n_train_samples)
	batches=torch.split(shuffle_idx,bsz)

	dis_loss = 0
	gen_loss = 0

	for idx in batches:
		x1 = train_xyz[idx,:]
		x2 = train_xanes[idx,:]
		trainer.gen_update(x1,x2)
		trainer.dis_update(x1,x2)
		dis_loss += trainer.loss_dis_total
		gen_loss += trainer.loss_gen_total
	print(f">>> Training dis loss = {dis_loss}")
	print(f">>> Training gen loss = {gen_loss}")


# Validation losses
z_a = valid_xyz
z_b = valid_xanes

# encode
enc_a = trainer.gen_a.encode(z_a)
enc_b = trainer.gen_b.encode(z_b)

# encode shared layer
shared_enc_a = trainer.enc_shared.model(enc_a)
shared_enc_b = trainer.enc_shared.model(enc_b)
# decode shared layer
shared_dec_a = trainer.dec_shared.model(shared_enc_a)
shared_dec_b = trainer.dec_shared.model(shared_enc_b)

# decode (within domain)
z_a_recon = trainer.gen_a.decode(shared_dec_a)
z_b_recon = trainer.gen_b.decode(shared_dec_b)
# decode (cross domain)
z_ba = trainer.gen_a.decode(shared_dec_b)
z_ab = trainer.gen_b.decode(shared_dec_a)

# reconstruction loss
valid_loss_recon_x_a = trainer.recon_criterion(z_a_recon, z_a)
valid_loss_recon_x_b = trainer.recon_criterion(z_b_recon, z_b)

# translation loss
valid_loss_translate_x_ab = trainer.recon_criterion(z_ab,z_b)
valid_loss_translate_x_ba = trainer.recon_criterion(z_ba,z_a)

total_loss = valid_loss_recon_x_a + valid_loss_recon_x_b + valid_loss_translate_x_ab + valid_loss_translate_x_ba

print(f">>>   Validation loss: {total_loss}")
print(f">>    Validation loss xyz-to-xyz     : {valid_loss_recon_x_a}")
print(f">>    Validation loss xanes-to-xanes : {valid_loss_recon_x_b}")
print(f">>    Validation loss xyz-to-xanes   : {valid_loss_translate_x_ab}")
print(f">>    Validation loss xanes-to-xyz   : {valid_loss_translate_x_ba}")
print(f">>    -----              ")

print('>>> done!')
#-----------------------------------------------------------------------------

# An example from validation data
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']



fig = plt.figure(figsize = (10,10))
plt.subplot(3,2,1)
plt.plot(z_a[0,:].detach().numpy(),color = cycle[0])
plt.title('Original XYZ')

plt.subplot(3,2,2)
plt.plot(z_b[0,:].detach().numpy(),color = cycle[0])
plt.title('Original XANES')

# plt.subplot(3,2,1)
# plt.plot(minmax_scale(z_a[0,:].detach().numpy()),color = cycle[0])
# plt.plot(minmax_scale(z_a_recon[0,:].detach().numpy()),color = cycle[1])
# plt.plot(minmax_scale(z_ba[0,:].detach().numpy()),color = cycle[2])
# plt.title('Original XYZ')

# plt.subplot(3,2,2)
# plt.plot(minmax_scale(z_b[0,:].detach().numpy()),color = cycle[0])
# plt.plot(minmax_scale(z_b_recon[0,:].detach().numpy()),color = cycle[1])
# plt.plot(minmax_scale(z_ab[0,:].detach().numpy()),color = cycle[2])
# plt.title('Original XANES')


plt.subplot(3,2,3)
plt.plot(z_a_recon[0,:].detach().numpy(),color = cycle[1])
plt.title('Autoencoder XYZ')

plt.subplot(3,2,4)
plt.plot(z_b_recon[0,:].detach().numpy(),color = cycle[1])
plt.title('Autoencoder XANES')


plt.subplot(3,2,5)
plt.plot(z_ba[0,:].detach().numpy(),color = cycle[2])
plt.title('XANES-to-XYZ')


plt.subplot(3,2,6)
plt.plot(z_ab[0,:].detach().numpy(),color = cycle[2])
plt.title('XYZ-to-XANES')

plt.show()


#-----------------------------------------------------------------------------

