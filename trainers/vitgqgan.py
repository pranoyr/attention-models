import os
import torch
import random
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.scheduler.cosine_lr import CosineLRScheduler
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
from transformers import get_cosine_schedule_with_warmup
# import constant_learnign rate swith warm up
import logging
from transformers import get_constant_schedule_with_warmup
from PIL import Image
import cv2
from .utils.base_trainer import BaseTrainer

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


class VQGANTrainer(BaseTrainer):
	def __init__(
		self, 
		cfg,
		model,
		dataloaders
		):
		super().__init__(cfg, model, dataloaders)
  
		# Training parameters
		lr = cfg.optimizer.params.learning_rate
		warmup_steps = cfg.lr_scheduler.params.warmup_steps
		beta1 = cfg.optimizer.params.beta1
		beta2 = cfg.optimizer.params.beta2
		decay_steps = cfg.lr_scheduler.params.decay_steps
		weight_decay = cfg.optimizer.params.weight_decay


		# disciminator
		self.discr = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3)
		
		# Optimizer
		self.g_optim = Adam(self.model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
		self.d_optim = Adam(self.discr.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
  
  
		num_iters_per_epoch = math.ceil(len(self.train_dl.dataset))
		total_iters = self.num_epoch * num_iters_per_epoch
		if decay_steps:
			total_iters = decay_steps
		self.g_sched = CosineLRScheduler(self.g_optim, t_initial=total_iters, warmup_t=warmup_steps, warmup_lr_init=1e-6, lr_min=5e-5)
		self.d_sched = CosineLRScheduler(self.d_optim, t_initial=total_iters, warmup_t=warmup_steps, warmup_lr_init=1e-6, lr_min=5e-5)


		# define losses
		self.per_loss = LPIPS(net='vgg').to(self.device).eval()
		for param in self.per_loss.parameters():
			param.requires_grad = False
		self.d_loss = hinge_d_loss
		self.g_loss = g_nonsaturating_loss

		self.per_loss_weight = cfg.losses.per_loss_weight
		self.adv_loss_weight = cfg.losses.adv_loss_weight
		self.logit_laplace_weight = cfg.losses.logit_laplace_weight

		
		
		(
			self.model,
			self.discr,
			self.g_optim,
			self.d_optim,
			self.g_sched,
			self.d_sched,
			self.train_dl
	
		) = self.accelerator.prepare(
			self.model,
			self.discr,
			self.g_optim,
			self.d_optim,
			self.g_sched,
			self.d_sched,
			self.train_dl
	 )
	
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
					requires_grad(self.model, False)
					requires_grad(self.discr, True)
					with self.accelerator.accumulate(self.discr):
						with self.accelerator.autocast():
							rec, codebook_loss = self.model(img)
		
							fake_pred = self.discr(rec)
							real_pred = self.discr(img)
							
							gp = self.calculate_gradient_penalty(img, rec)
							d_loss = self.d_loss(fake_pred, real_pred) + gp
							
						self.accelerator.backward(d_loss)
						if self.accelerator.sync_gradients and self.max_grad_norm:
							self.accelerator.clip_grad_norm_(self.discr.parameters(), self.max_grad_norm)
						self.d_optim.step()
						self.d_sched.step(self.global_step)
						self.d_optim.zero_grad()
						
					
					# generator part
					requires_grad(self.model, True)
					requires_grad(self.discr, False)
					with self.accelerator.accumulate(self.model):
						with self.accelerator.autocast():
							rec, codebook_loss = self.model(img)
						
							logit_laplacian = F.l1_loss(rec, img)
							l2_loss = F.mse_loss(rec, img)
							# perception loss
							per_loss = self.per_loss(rec, img).mean()
							# gan loss
							g_loss = self.g_loss(self.discr(rec))
							# combine
							loss = codebook_loss + self.adv_loss_weight * g_loss + self.per_loss_weight * per_loss  \
							+ self.logit_laplace_weight * logit_laplacian + l2_loss
							#  L = LVQ + 0.1 LAdv + 0.1 LPerceptual + 0.1 LLogit-laplace + 1.0L2.
						
						self.accelerator.backward(loss)
						if self.accelerator.sync_gradients and self.max_grad_norm:
							self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
						self.g_optim.step()
						self.g_sched.step(self.global_step)
						self.g_optim.zero_grad()   

				
					if not (self.global_step % self.save_every):
						self.save_ckpt(rewrite=True)
					
					if not (self.global_step % self.sample_every):
						self.evaluate()
	  
					if not (self.global_step % self.gradient_accumulation_steps):
						g_lr = self.g_optim.param_groups[0]['lr']
						d_lr = self.d_optim.param_groups[0]['lr']
						self.accelerator.log({"step": self.global_step, "d_lr": d_lr, "g_lr": g_lr,
											"d_loss": d_loss, "g_loss": g_loss, 
											"l2_loss": l2_loss, "per_loss": per_loss, "logit_laplace": logit_laplacian,
											"codebook_loss": codebook_loss}, step=self.global_step)
			
					self.global_step += 1
	
		self.accelerator.end_training()        
		print("Train finished!")

	@torch.no_grad()
	def evaluate(self):
		self.model.eval()
		with tqdm(self.val_dl, dynamic_ncols=True, disable=not self.accelerator.is_local_main_process) as valid_dl:
			for i, batch in enumerate(valid_dl):
				if isinstance(batch, tuple) or isinstance(batch, list):
					img = batch[0]
				else:
					img = batch
				if i == 10:
					break
				img = img.to(self.device)
				rec, _ = self.model(img)
				imgs_and_recs = torch.stack((img, rec), dim=0)
				imgs_and_recs = rearrange(imgs_and_recs, 'r b ... -> (b r) ...')
				imgs_and_recs = imgs_and_recs.detach().cpu().float()

				# grid = make_grid(imgs_and_recs, nrow=6, normalize=False, value_range=(-1, 1))
				grid = make_grid(imgs_and_recs, nrow=6, normalize=False)
				save_image(grid, os.path.join(self.image_saved_dir, f'step_{i}.png'))
		self.model.train()



	  


