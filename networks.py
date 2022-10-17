from torch import nn
import torch
from torch.autograd import Variable
import numpy as np

class GAN_AE_Trainer(nn.Module):
	def __init__(self, **kwargs):
		super(GAN_AE_Trainer, self).__init__()
		# Initiate the networks
		self.gen_a = AEGen(n_input_features = kwargs['input_dim_a'], n_output_features = kwargs['input_dim_a'])  # auto-encoder for domain a
		self.gen_b = AEGen(n_input_features = kwargs['input_dim_b'], n_output_features = kwargs['input_dim_b'])  # auto-encoder for domain b
		self.dis_a = Dis(n_input_features = kwargs['input_dim_a'])  # discriminator for domain a
		self.dis_b = Dis(n_input_features = kwargs['input_dim_b'])  # discriminator for domain b
		self.enc_shared = SharedLayer()
		self.dec_shared = SharedLayer()

		self.dis_opt = torch.optim.Adam(self.parameters(),lr = 1e-3,weight_decay=1e-5)
		self.gen_opt = torch.optim.Adam(self.parameters(),lr = 1e-3,weight_decay=1e-5)

		# Network weight initialization
		self.apply(weight_init)
		self.dis_a.apply(weight_init)
		self.dis_b.apply(weight_init)

		self.gen_a.apply(weight_init)
		self.gen_b.apply(weight_init)
		self.enc_shared.apply(weight_init)
		self.dec_shared.apply(weight_init)

	def recon_criterion(self, pred, target):
		loss_fn = nn.MSELoss()
		loss = loss_fn(pred,target)
		return loss

	def forward(self, x_a, x_b):
		self.eval()
		enc_a = self.gen_a.encode(x_a)
		enc_b = self.gen_b.encode(x_b)

		shared_enc_a = self.enc_shared.model(enc_a)
		shared_enc_b = self.enc_shared.model(enc_b)

		shared_dec_a = self.dec_shared.model(shared_enc_a)
		shared_dec_b = self.dec_shared.model(shared_enc_b)

		x_ba = self.gen_a.decode(shared_dec_b)
		x_ab = self.gen_b.decode(shared_dec_a)
		self.train()
		return x_ab, x_ba

	def gen_update(self, x_a, x_b):
		self.gen_opt.zero_grad()

		# encode
		enc_a = self.gen_a.encode(x_a)
		enc_b = self.gen_b.encode(x_b)

		# encode shared layer
		shared_enc_a = self.enc_shared.model(enc_a)
		shared_enc_b = self.enc_shared.model(enc_b)
		# decode shared layer
		shared_dec_a = self.dec_shared.model(shared_enc_a)
		shared_dec_b = self.dec_shared.model(shared_enc_b)


		# decode (within domain)
		x_a_recon = self.gen_a.decode(shared_dec_a)
		x_b_recon = self.gen_b.decode(shared_dec_b)

		# decode (cross domain)
		x_ba = self.gen_a.decode(shared_dec_b)
		x_ab = self.gen_b.decode(shared_dec_a)


		# encode again
		y_b_recon = self.gen_a.encode(x_ba)
		y_a_recon = self.gen_b.encode(x_ab)

		# encode shared layer
		shared_enc_b_recon = self.enc_shared.model(y_b_recon)
		shared_enc_a_recon = self.enc_shared.model(y_a_recon)
		# decode shared layer
		shared_dec_b_recon = self.dec_shared.model(shared_enc_b_recon)
		shared_dec_a_recon = self.dec_shared.model(shared_enc_a_recon)

		# decode again
		x_aba = self.gen_a.decode(shared_dec_a_recon)
		x_bab = self.gen_b.decode(shared_dec_b_recon)


		# reconstruction loss
		self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
		self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
		self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
		self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
		# GAN loss
		self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
		self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
		# total loss
		self.loss_gen_total = self.loss_gen_adv_a + \
						self.loss_gen_adv_b + \
						self.loss_gen_recon_x_a + \
						self.loss_gen_recon_x_b
		
		self.loss_gen_total.backward()
		self.gen_opt.step()

	def dis_update(self, x_a, x_b):
		self.dis_opt.zero_grad()
		# encode
		y_a = self.gen_a.encode(x_a)
		y_b = self.gen_b.encode(x_b)

		# encode shared layer
		shared_enc_a = self.enc_shared.model(y_a)
		shared_enc_b = self.enc_shared.model(y_b)
		# decode shared layer
		shared_dec_a = self.dec_shared.model(shared_enc_a)
		shared_dec_b = self.dec_shared.model(shared_enc_b)


		# decode (cross domain)
		x_ba = self.gen_a.decode(shared_dec_b)
		x_ab = self.gen_b.decode(shared_dec_a)
		# D loss
		self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
		self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)

		self.loss_dis_total = self.loss_dis_a + self.loss_dis_b

		self.loss_dis_total.backward()
		self.dis_opt.step()

	def update_learning_rate(self):
		if self.dis_scheduler is not None:
			self.dis_scheduler.step()
		if self.gen_scheduler is not None:
			self.gen_scheduler.step()

	def save(self, snapshot_dir, iterations):
		# Save generators, discriminators, and optimizers
		gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
		dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
		opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
		torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
		torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
		torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)



class SharedLayer(nn.Module):
	# Autoencoder architecture
	def __init__(self, **kwargs):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128,128))

	def model(self,features):
		z = self.mlp(features)
		return z


class AEGen(nn.Module):
	# Autoencoder architecture
	def __init__(self, **kwargs):
		super().__init__()
		self.enc = nn.Sequential(
			nn.Linear(kwargs["n_input_features"], 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128,128))
		self.dec = nn.Sequential(
			nn.Linear(128,128),
			nn.ReLU(),
			nn.Linear(128,256),
			nn.ReLU(),
			nn.Linear(256,kwargs["n_output_features"]),
			nn.ReLU())

	def encode(self, features):
		hiddens = self.enc(features)
		return hiddens

	def decode(self, hiddens):
		images = self.dec(hiddens)
		return images

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
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, 1),
			nn.Sigmoid())
	
	def forward(self,features):
		return self.discriminate(features)

	def calc_dis_loss(self,input_fake,input_real):
		# Calculate the loss to train D
		out0 = self.forward(input_fake)
		out1 = self.forward(input_real)
		# loss = nn.functional.cross_entropy(out0,out1)
		loss = 0
		for it, (out0, out1) in enumerate(zip(out0, out1)):
			loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
		return loss

	def calc_gen_loss(self,input_fake):
		# Calculate the loss to train G
		out0 = self.forward(input_fake)
		ones = torch.ones((input_fake.size(0),1))
		# loss = nn.functional.cross_entropy(out0,ones)
		# loss = nn.BCELoss()(out0,ones)
		loss = 0
		for it, (out0) in enumerate(out0):
			loss += torch.mean((out0 - 1)**2) # LSGAN
		return loss


def weight_init(m):
	if isinstance(m, nn.Linear):
		nn.init.zeros_(m.bias)