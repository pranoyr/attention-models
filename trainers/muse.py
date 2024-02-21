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


class MuseTrainer(nn.Module):
	def __init__(
		self, 
		cfg,
		model,
		dataloaders
		):
		super().__init__()
  
		self.cfg = cfg
		
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
		self.model = model
		self.train_dl , self.val_dl = dataloaders
		self.global_step = 0
  
		# Resume from ckpt
		if cfg.experiment.resume_path_from_checkpoint:
			path = cfg.experiment.resume_path_from_checkpoint
			ckpt = torch.load(path, map_location=self.accelerator.device)
			self.model.load_state_dict(ckpt["state_dict"])
			self.global_step = ckpt["step"]
			logging.info(f"Resuming from checkpoint: {path} at step: {self.global_step}")
		
		# Training parameters
		lr = cfg.optimizer.params.learning_rate
		warmup_steps = cfg.lr_scheduler.params.warmup_steps
		beta1 = cfg.optimizer.params.beta1
		beta2 = cfg.optimizer.params.beta2
		self.gradient_accumulation_steps = cfg.training.gradient_accumulation_steps
		
		# Optimizer
		self.optim = Adam(self.model.parameters(), lr=lr, betas=(beta1, beta2))
	
		self.scheduler = get_constant_schedule_with_warmup(
			self.optim,
			num_warmup_steps=warmup_steps)

		(
			self.model,
			self.optim,
			self.scheduler,
			self.train_dl
	
		) = self.accelerator.prepare(
			self.model,
			self.optim,
			self.scheduler,
			self.train_dl
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


		logging.info(f"Train dataset size: {len(self.train_dl.dataset)}")
		logging.info(f"Val dataset size: {len(self.val_dl.dataset)}")

		num_iters_per_epoch = len(self.train_dl) 
		total_iters = self.num_epoch * num_iters_per_epoch
		logging.info(f"Number of iterations per epoch: {num_iters_per_epoch}")
		logging.info(f"Total training iterations: {total_iters}")
		
	
	@property
	def device(self):
		return self.accelerator.device
	
	
	def train(self):
	 
		start_epoch=self.global_step//len(self.train_dl)
	  
		for epoch in range(start_epoch, self.num_epoch):
			with tqdm(self.train_dl, dynamic_ncols=True, disable=not self.accelerator.is_main_process) as train_dl:
				for batch in train_dl:
					img, text = batch
					img = img.to(self.device)
				
					with self.accelerator.accumulate(self.model):
						with self.accelerator.autocast():
		  
							loss = self.model(text, img)
						
						self.accelerator.backward(loss)
						if self.accelerator.sync_gradients:
							self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
						self.optim.step()
						self.scheduler.step(self.global_step)
						self.optim.zero_grad()
						
						
					if not (self.global_step % self.save_every):
						self.save_ckpt(rewrite=True)
					
					if not (self.global_step % self.sample_every):
						self.evaluate()
	  
					if not (self.global_step % self.gradient_accumulation_steps):
						lr = self.optim.param_groups[0]['lr']
						self.accelerator.log({"loss": loss.item(), "lr": lr}, step=self.global_step)
			
					self.global_step += 1
	  
					
		self.accelerator.end_training()        
		print("Train finished!")
		
	def save_ckpt(self, rewrite=False):
		"""Save checkpoint"""

		filename = os.path.join(self.checkpoint_folder, f'{self.cfg.experiment.project_name}_step_{self.global_step}.pt')
		if rewrite:
			filename = os.path.join(self.checkpoint_folder, f'{self.cfg.experiment.project_name}.pt')
		
		checkpoint={
				'step': self.global_step,
				'state_dict': self.accelerator.unwrap_model(self.model).state_dict()
			}

		self.accelerator.save(checkpoint, filename)
		logging.info("Saving checkpoint: %s ...", filename)
   
   
	def resume_from_checkpoint(self, checkpoint_path):
		"""Resume from checkpoint"""
		checkpoint = self.accelerator.load(checkpoint_path)
		self.global_step = checkpoint['step']
		self.model.load_state_dict(checkpoint['state_dict'])
		logging.info("Resume from checkpoint %s (global_step %d)", checkpoint_path, self.global_step)


	@torch.no_grad()
	def evaluate(self):
		self.model.eval()
		with tqdm(self.val_dl, dynamic_ncols=True, disable=not self.accelerator.is_local_main_process) as valid_dl:
			for i, batch in enumerate(valid_dl):
				img, text = batch
	
				if i == 10:
					break
			
				img = self.model.generate(text)
				
				grid = make_grid(img, nrow=6, normalize=True, value_range=(-1, 1))
				save_image(grid, os.path.join(self.image_saved_dir, f'step_{i}.png'))
		self.model.train()



	  


