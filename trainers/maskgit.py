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
import wandb
from .utils.base_trainer import BaseTrainer
from .utils.scheduler import get_scheduler
from .utils.optimizer import get_optimizer


class MaskGitTrainer(BaseTrainer):
	def __init__(
		self, 
		cfg,
		model,
		dataloaders
		):
		super().__init__(cfg, model, dataloaders)
  
		decay_steps = cfg.lr_scheduler.params.decay_steps
  
		if not decay_steps:
			decay_steps = self.num_epoch * self.num_iters_per_epoch

		self.optim = get_optimizer(cfg, self.model.parameters())
		self.scheduler = get_scheduler(cfg, self.optim, decay_steps=decay_steps)

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
		
	def train(self):
	 
		start_epoch=self.global_step//len(self.train_dl)
	  
		for epoch in range(start_epoch, self.num_epoch):
			with tqdm(self.train_dl, dynamic_ncols=True, disable=not self.accelerator.is_main_process) as train_dl:
				for batch in train_dl:
					img, _ = batch
					img = img.to(self.device)
				
					with self.accelerator.accumulate(self.model):
						with self.accelerator.autocast():
							loss = self.model(img)
						
						self.accelerator.backward(loss)
						if self.accelerator.sync_gradients and self.max_grad_norm:
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
	
	@torch.no_grad()
	def evaluate(self):
		self.model.eval()
		
		with tqdm(self.val_dl, dynamic_ncols=True, disable=not self.accelerator.is_main_process) as val_dl:
			for i, batch in enumerate(val_dl):
				imgs, _ = batch

				if i > 10:
					break
 
				imgs = self.model(imgs)
				grid = make_grid(imgs, nrow=6, normalize=False, value_range=(-1, 1))
				# send this to wandb
				self.accelerator.log({"samples": [wandb.Image(grid, caption="Generated samples")]})
				save_image(grid, os.path.join(self.image_saved_dir, f'step.png'))
		self.model.train()



	  


