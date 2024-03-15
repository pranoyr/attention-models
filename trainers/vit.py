
from tqdm import tqdm
import os
import logging
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from torch.optim import AdamW
from .utils.base_trainer import BaseTrainer

class VitTrainer(BaseTrainer):
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
		
		# Optimizer
		self.optim = AdamW(self.model.parameters(), lr=lr, betas=(beta1, beta2))

		self.criterion = nn.CrossEntropyLoss()
	
		self.scheduler = get_cosine_schedule_with_warmup(optimizer=self.optim, num_warmup_steps=warmup_steps, num_training_steps=cfg.training.num_epochs*len(self.train_dl))

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
		self.eval_every = cfg.experiment.eval_every
		self.log_every = cfg.experiment.log_every
		self.max_grad_norm = cfg.training.max_grad_norm
		
  
	def train(self):
		start_epoch=self.global_step//len(self.train_dl)
		self.model.train()
		for epoch in range(start_epoch, self.num_epoch):
			with tqdm(self.train_dl, dynamic_ncols=True, disable=not self.accelerator.is_main_process) as train_dl:
				for batch in train_dl:
					img , target = batch
					img = img.to(self.device)

					with self.accelerator.accumulate(self.model):
						with self.accelerator.autocast():
							outputs = self.model(img)
						
						# cross entropy loss
						loss = self.criterion(outputs, target)
						self.accelerator.backward(loss)
						if self.accelerator.sync_gradients and self.max_grad_norm:
							self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
						self.optim.step()
						self.scheduler.step(self.global_step)
						self.optim.zero_grad()
						
						
					if not (self.global_step % self.save_every):
						self.save_ckpt(rewrite=True)
					
					if not (self.global_step % self.eval_every):
						self.model.eval()
						outputs = torch.softmax(outputs, dim=1)
						acc = (outputs.argmax(dim=1) == target).float().mean().item()
						self.accelerator.log({"acc": acc}, step=self.global_step)
						self.evaluate()
						self.model.train()
      
					if not (self.global_step % self.gradient_accumulation_steps):
						lr = self.optim.param_groups[0]['lr']
						self.accelerator.log({"loss": loss.item(), "lr": lr}, step=self.global_step)
			
					self.global_step += 1
	  
					
		self.accelerator.end_training()        
		print("Train finished!")
  
	def evaluate(self):
		with torch.no_grad():
			with tqdm(self.val_dl, dynamic_ncols=True, disable=not self.accelerator.is_main_process) as val_dl:
				for i, batch in enumerate(val_dl):
					img , target = batch
					img = img.to(self.device)
					target = target.to(self.device)
					outputs = self.model(img)
					outputs = torch.softmax(outputs, dim=1)
					acc = (outputs.argmax(dim=1) == target).float().mean().item()
					self.accelerator.log({"val_acc": acc}, step=self.global_step)
		print("Validation finished!")



