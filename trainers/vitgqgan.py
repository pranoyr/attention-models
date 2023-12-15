import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.autograd import Variable
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
from lpips import LPIPS
from einops import rearrange
from models.utils import NLayerDiscriminator
# import constant_learnign rate swith warm up
import logging
from transformers import get_constant_schedule_with_warmup
from PIL import Image
import cv2

def requires_grad(model, flag=True):
	for p in model.parameters():
		p.requires_grad = flag


def hinge_d_loss(fake, real):
	loss_fake = torch.mean(F.relu(1. + fake))
	loss_real = torch.mean(F.relu(1. - real))
	d_loss = 0.5 * (loss_real + loss_fake)
	return d_loss


def g_nonsaturating_loss(fake):
	loss = F.softplus(-fake).mean()

	return loss



class VQGANTrainer(nn.Module):
	def __init__(
		self, 
		cfg,
		model,
		dataloaders
		):
		super().__init__()
		
		# init accelerator
		self.accelerator = Accelerator(
			mixed_precision=cfg.training.mixed_precision,
			gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
			log_with="wandb"
		)
		self.accelerator.init_trackers(
				project_name=cfg.experiment.project_name,
				init_kwargs={"wandb": {
				"config" : cfg,
				"name" : cfg.experiment.name}
		})

		# models and dataloaders
		self.vqvae = model
		self.train_dl , self.val_dl = dataloaders
		self.global_step = 0
		
		# Training parameters
		lr = cfg.optimizer.params.learning_rate
		warmup_steps = cfg.lr_scheduler.params.warmup_steps
		beta1 = cfg.optimizer.params.beta1
		beta2 = cfg.optimizer.params.beta2
		
		# disciminator
		self.discr = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3)
		
		# Optimizer
		self.g_optim = Adam(self.vqvae.parameters(), lr=lr, betas=(beta1, beta2))
		self.d_optim = Adam(self.discr.parameters(), lr=lr, betas=(beta1, beta2))
		
		# Scheduler
		self.g_sched = get_constant_schedule_with_warmup(self.g_optim, warmup_steps)
		self.d_sched = get_constant_schedule_with_warmup(self.d_optim, warmup_steps)
  

		# define losses
		self.per_loss = LPIPS(net='vgg').to(self.device).eval()
		for param in self.per_loss.parameters():
			param.requires_grad = False
		self.d_loss = hinge_d_loss
		self.g_loss = g_nonsaturating_loss
		self.d_weight = 0.1
		
		(
			self.vqvae,
			self.discr,
			self.g_optim,
			self.d_optim,
	
		) = self.accelerator.prepare(
			self.vqvae,
			self.discr,
			self.g_optim,
			self.d_optim,
	 )
		
		# logging details
		self.num_epoch = cfg.training.num_epochs
		self.save_every = cfg.experiment.save_every
		self.sample_every = cfg.experiment.sample_every
		self.log_every = cfg.experiment.log_every
		self.max_grad_norm = 1
		
		# Checkpoint and generated images folder
		self.checkpoint_folder = os.path.join(cfg.experiment.output_folder, 'checkpoints')
		os.makedirs(self.checkpoint_folder, exist_ok=True)
		
		self.image_saved_dir = os.path.join(cfg.experiment.output_folder, 'images')
		os.makedirs(self.image_saved_dir, exist_ok=True)
		
		n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
		print(f'number of learnable parameters: {n_parameters//1e6}M')
	
	@property
	def device(self):
		return self.accelerator.device
	
	def calculate_gradient_penalty(self, real_images, fake_images, lambda_term=10):
		eta = torch.FloatTensor(real_images.shape[0],1,1,1).uniform_(0,1).to(self.device)
		eta = eta.expand(real_images.shape[0], real_images.size(1), real_images.size(2), real_images.size(3))
		
		interpolated = eta * real_images + ((1 - eta) * fake_images)
		interpolated = Variable(interpolated, requires_grad=True)
		prob_interpolated = self.discr(interpolated)
		
		gradients = torch.autograd.grad(
			outputs=prob_interpolated, 
			inputs=interpolated,
			grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
			create_graph=True, 
			retain_graph=True,)[0]
		
		grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
		return grad_penalty
	
	def train(self):
     
		start_epoch=self.global_step//len(self.train_dl)
	  
		for epoch in range(start_epoch, self.num_epoch):
			with tqdm(self.train_dl, dynamic_ncols=True, disable=not self.accelerator.is_main_process) as train_dl:
				for batch in train_dl:
					if isinstance(batch, tuple) or isinstance(batch, list):
						img = batch[0]
					else:
						img = batch
					img = img.to(self.device)
					# discriminator part
					requires_grad(self.vqvae, False)
					requires_grad(self.discr, True)
					with self.accelerator.accumulate(self.discr):
						with self.accelerator.autocast():
							rec, codebook_loss = self.vqvae(img)
		
							fake_pred = self.discr(rec)
							real_pred = self.discr(img)
							
							gp = self.calculate_gradient_penalty(img, rec)
							d_loss = self.d_loss(fake_pred, real_pred) + gp
							
						self.accelerator.backward(d_loss)
						if self.accelerator.sync_gradients:
							self.accelerator.clip_grad_norm_(self.discr.parameters(), self.max_grad_norm)
						self.d_optim.step()
						self.d_sched.step(self.global_step)
						self.d_optim.zero_grad()
						
					
					# generator part
					requires_grad(self.vqvae, True)
					requires_grad(self.discr, False)
					with self.accelerator.accumulate(self.vqvae):
						with self.accelerator.autocast():
							rec, codebook_loss = self.vqvae(img)
							# reconstruction loss
							rec_loss = F.l1_loss(rec, img) + F.mse_loss(rec, img)
							# perception loss
							per_loss = self.per_loss(rec, img).mean()
							# gan loss
							g_loss = self.g_loss(self.discr(rec))
							# combine
							loss = codebook_loss + rec_loss + per_loss + self.d_weight * g_loss
						
						self.accelerator.backward(loss)
						if self.accelerator.sync_gradients:
							self.accelerator.clip_grad_norm_(self.vqvae.parameters(), self.max_grad_norm)
						self.g_optim.step()
						self.g_sched.step(self.global_step)
						self.g_optim.zero_grad()   

				
					if not (self.global_step % self.save_every):
						self.save(rewrite=True)
					
					if not (self.global_step % self.sample_every):
						self.evaluate()
	  
					if not (self.global_step % self.log_every):
						lr = self.g_optim.param_groups[0]['lr']
						self.accelerator.log({"step": self.global_step, "lr": lr, 
											"d_loss": d_loss, "g_loss": g_loss, 
											"rec_loss": rec_loss, "per_loss": per_loss,
											"codebook_loss": codebook_loss})
			
					self.global_step += 1
	  
					
		self.accelerator.end_training()        
		print("Train finished!")
		
	def save(self, rewrite=False):
		"""Save checkpoint"""

		filename = os.path.join(self.checkpoint_folder, f'vit_vq_step_{self.global_step}.pt')
		if rewrite:
			filename = os.path.join(self.checkpoint_folder, f'vit_vq.pt')
		
		checkpoint={
				'step': self.global_step,
				'state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
				'optimizer': self.optimizer.state_dict(),
				'scheduler': self.scheduler.state_dict()
			}

		self.accelerator.save(checkpoint, filename)
		logging.info("Saving checkpoint: %s ...", filename)
   
   
	def resume_from_checkpoint(self, checkpoint_path):
		"""Resume from checkpoint"""
		checkpoint = self.accelerator.load(checkpoint_path)
		self.global_step = checkpoint['step']
		self.model.load_state_dict(checkpoint['state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer'])
		self.scheduler.load_state_dict(checkpoint['scheduler'])
		logging.info("Resume from checkpoint %s (global_step %d)", checkpoint_path, self.global_step)
					
	def save(self):
		self.accelerator.wait_for_everyone()
		state_dict = self.accelerator.unwrap_model(self.vqvae).state_dict()
		self.accelerator.save(state_dict, os.path.join(self.checkpoint_folder, f'vit_vq_step_{self.global_step}.pt'))
													   
	@torch.no_grad()
	def evaluate(self):
		self.vqvae.eval()
		with tqdm(self.val_dl, dynamic_ncols=True, disable=not self.accelerator.is_local_main_process) as valid_dl:
			for i, batch in enumerate(valid_dl):
				if isinstance(batch, tuple) or isinstance(batch, list):
					img = batch[0]
				else:
					img = batch
				if i == 10:
					break
				img = img.to(self.device)
				rec, _ = self.vqvae(img)
				imgs_and_recs = torch.stack((img, rec), dim=0)
				imgs_and_recs = rearrange(imgs_and_recs, 'r b ... -> (b r) ...')
				imgs_and_recs = imgs_and_recs.detach().cpu().float()

				grid = make_grid(imgs_and_recs, nrow=6, normalize=True, value_range=(-1, 1))
				save_image(grid, os.path.join(self.image_saved_dir, f'step_{i}.png'))
		self.vqvae.train()



	  


