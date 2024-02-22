import os
import torch
import random
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from einops import rearrange
import logging



class BaseTrainer(object):
	def __init__(
		self, 
		cfg,
		model,
		dataloaders
		):

		self.project_name = cfg.experiment.project_name
		
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
		self.num_epoch = cfg.training.num_epochs
		self.gradient_accumulation_steps = cfg.training.gradient_accumulation_steps
		self.batch_size = cfg.dataset.params.batch_size
		
		# Resume from ckpt
		if cfg.experiment.resume_path_from_checkpoint:
			path = cfg.experiment.resume_path_from_checkpoint
			self.resume_from_checkpoint(path)
	
	
		# Checkpoint and generated images folder
		self.checkpoint_folder = os.path.join(cfg.experiment.output_folder, 'checkpoints')
		os.makedirs(self.checkpoint_folder, exist_ok=True)
		
		self.image_saved_dir = os.path.join(cfg.experiment.output_folder, 'images')
		os.makedirs(self.image_saved_dir, exist_ok=True)


		logging.info(f"Train dataset size: {len(self.train_dl.dataset)}")
		logging.info(f"Val dataset size: {len(self.val_dl.dataset)}")

		# effective iteration considering gradient accumulation
		effective_batch_size = self.batch_size * self.gradient_accumulation_steps
		num_iters_per_epoch = math.ceil(len(self.train_dl.dataset) / effective_batch_size)
		total_iters = self.num_epoch * num_iters_per_epoch
		logging.info(f"Number of iterations per epoch: {num_iters_per_epoch}")
		logging.info(f"Total training iterations: {total_iters}")
		
	
	@property
	def device(self):
		return self.accelerator.device
	
	
	def train(self):
		raise NotImplementedError("Train method not implemented")
		

	def save_ckpt(self, rewrite=False):
		"""Save checkpoint"""

		filename = os.path.join(self.checkpoint_folder, f'{self.project_name}_step_{self.global_step}.pt')
		if rewrite:
			filename = os.path.join(self.checkpoint_folder, f'{self.project_name}.pt')
		
		checkpoint={
				'step': self.global_step,
				'state_dict': self.accelerator.unwrap_model(self.model).state_dict()
			}

		self.accelerator.save(checkpoint, filename)
		logging.info("Saving checkpoint: %s ...", filename)
   
   
	def resume_from_checkpoint(self, checkpoint_path):
		"""Resume from checkpoint"""
		checkpoint = torch.load(checkpoint_path)
		self.global_step = checkpoint['step']
		self.model.load_state_dict(checkpoint['state_dict'])
		logging.info("Resume from checkpoint %s (global_step %d)", checkpoint_path, self.global_step)


	@torch.no_grad()
	def evaluate(self):
		raise NotImplementedError("Evaluate method not implemented")


	  


