import torch.nn as nn
import torch
import copy
import torch.nn.functional as F
from einops import rearrange, repeat, pack
import math
from models.transformer import Encoder as BidirectionalDecorder
from models.muse import filter_logits
from tqdm import tqdm
from typing import Optional
import numpy as np
import cv2

def cosine_schedule(t):
	return torch.cos(t * math.pi / 2)


def restore(x):
    x = (x + 1) * 0.5
    x = x.permute(1,2,0).detach().cpu().numpy()
    x = (255*x).astype(np.uint8)
    return x


def exists(val):
	return val is not None

class MaskGitTransformer(nn.Module):
	def __init__(
		self,
		dim,
		vq,
		vocab_size,
		n_heads=8,
		d_head=64,
		dec_depth=6,
		dropout=0.1
	):
		super().__init__()
  
		self.vq = vq

		self.input_proj = nn.Embedding(vocab_size+1, dim)
		num_patches = vq.num_patches
		self.pos_enc =  nn.Parameter(torch.randn(1, num_patches, dim))
		self.mask_token_id = vocab_size

		self.init_norm = nn.LayerNorm(dim)
		self.decoder = BidirectionalDecorder(
			dim=dim, n_heads=n_heads, d_head=d_head, depth=dec_depth, dropout=dropout
		)
		self.final_norm = nn.LayerNorm(dim)
		self.linear = nn.Linear(dim, vocab_size)

		# self.apply(self._init_weights)

		# freeze vq
		self.vq.eval()
		self.vq.requires_grad_(False)

	def _init_weights(self, module):
		"""
		Initialize the weights according to the original implementation.
		https://github.com/google-research/maskgit/blob/main/maskgit/nets/maskgit_transformer.py#L37
		"""
		# TODO: make this configurable
		if isinstance(module, nn.Linear):
			nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
			if module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, nn.Embedding):
			nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
		# elif isinstance(module, (nn.LayerNorm, RMSNorm)):
		# 	if hasattr(module, "weight") and module.weight is not None:
		# 		module.weight.data.fill_(1.0)
		# 	if hasattr(module, "bias") and module.bias is not None:
		# 		module.bias.data.zero_()
		
	def fill_mask(self, x):
		T = 8  # max number of timesteps during inference
		# sample the timestep from uniform distribution
		b , n = x.shape
		t = torch.randint(1, T, (1,))
		num_tokens_masked = cosine_schedule(t / T) * n
		num_tokens_masked = num_tokens_masked.clamp(min = 1.).int()
   
		# create mask 
		randm_perm = torch.rand(x.shape).argsort(dim = -1)
		mask = randm_perm < num_tokens_masked
		mask = mask.cuda()
		
		# fill x with mask_id, ignore the tokens that are not masked while computing loss
		tgt = x.masked_fill(~mask, -1)
		x = x.masked_fill(mask, self.mask_token_id)    
		return x, tgt, mask

	def fill_custom_mask(self, x):
		T = 8  # max number of timesteps during inference
		# sample the timestep from uniform distribution
		b , n = x.shape
		t = torch.tensor([7])
		num_tokens_masked = cosine_schedule(t / T) * n
		num_tokens_masked = num_tokens_masked.clamp(min = 1.).int()

		# create mask 
		randm_perm = torch.rand(x.shape).argsort(dim = -1)
		mask = randm_perm < num_tokens_masked
		mask = torch.zeros(x.shape).bool()
		# put first 200 as True
		mask[:, :100] = True
		# put last 200 as True
		mask[:, -200:] = True

		mask = mask.cuda()
  
		
		# fill x with mask_id, ignore the tokens that are not masked while computing loss
		tgt = x.masked_fill(~mask, -1)
		# x = x.masked_fill(mask, self.mask_token_id)    
		return x, tgt, mask

	def forward(self, imgs):
		# quantize images
		x = self.vq.encode_imgs(imgs)
		
		x, tgt, _ = self.fill_mask(x)
		x = self.input_proj(x)
		x += self.pos_enc

		# transformer decoder
		x = self.init_norm(x)
		dec_out = self.decoder(x)
		dec_out = self.final_norm(dec_out)
		output = self.linear(dec_out)

		if not self.training:
			pred_ids = torch.softmax(output, dim=-1).argmax(dim=-1)
			decoded_imgs = self.vq.decode_indices(pred_ids)
			return decoded_imgs
			
		# compute loss
		output = rearrange(output, 'b t c -> b c t')
		loss = torch.nn.functional.cross_entropy(output, tgt, ignore_index=-1)
		return loss    

	def generate(self, imgs : Optional[torch.Tensor] = None, timesteps = 18):
		     
		num_patches = self.vq.num_patches

		device = "cuda" if torch.cuda.is_available() else "cpu"
		b = 1

		# initialize decoder inputs
		ids = torch.ones(b, num_patches, dtype=torch.long, device=device) * self.mask_token_id
		scores = torch.zeros_like(ids).float().to(device)
		mask = torch.zeros_like(ids).bool().to(device)
  
  
		if exists(imgs):
			# quantize images
			ids = self.vq.encode_imgs(imgs)
			ids , _, mask = self.fill_custom_mask(ids)
			# print number of true
			print(f"Number of Tokens Masked: {mask.sum()}")
			scores = torch.zeros_like(ids).float().to(device)


			b , n = ids.shape
	

			ids2 = ids.masked_fill(mask, 100)
			decoded_imgs = self.vq.decode_indices(ids2)
			# display
			img = restore(decoded_imgs[0])
			img = img[:, :, ::-1]
			cv2.imshow('masked', img)
		
		outputs = []
		for timestep, steps_until_x0 in tqdm(zip(torch.linspace(0, 1, timesteps, device = device), reversed(range(timesteps))), total = timesteps):
			
			print(mask.sum())
			x = ids.masked_fill(mask, self.mask_token_id)    
			# decoder forward
			x = self.input_proj(x)
			x += self.pos_enc
			x = self.init_norm(x)
			logits = self.decoder(x)
			logits = self.final_norm(logits)
			logits = self.linear(logits)

			probs = F.softmax(logits, dim = -1)
   
			# decaying temperature
			temperature = 1 * (steps_until_x0 / timesteps) # temperature is annealed
			
			# sample with gumbel softmax
			logits = filter_logits(logits, p=0.9)

			pred_ids = F.gumbel_softmax(logits, tau = temperature, hard = False, dim = -1).argmax(dim = -1)

			# fill the masked tokens with predicted tokens
			ids[mask] = pred_ids[mask]
			
			# update scores
			scores = probs.gather(2, rearrange(pred_ids, 'b t -> b t 1'))
			scores = rearrange(scores, 'b t 1 -> b t')
   
			num_tokens_masked = mask.sum()// 2
			# rand_mask_prob = cosine_schedule(timestep)
			# num_tokens_masked = max(int((rand_mask_prob * 500).item()), 1)

   
			mask = torch.zeros_like(ids).bool()

			# rand_mask_prob = cosine_schedule(timestep)
	
			# find low probability tokens
			low_probs_indices = torch.argsort(scores, dim = -1)

			# indices of tokens to mask
			masked_indices = low_probs_indices[:, :num_tokens_masked]

			# True where the tokens are masked, False otherwise
			mask.scatter_(1, masked_indices, True)
   
			decoded_imgs = self.vq.decode_indices(ids)
			# display
			img = restore(decoded_imgs[0])
			img = img[:, :, ::-1]
			outputs.append(img)

		outputs  = np.hstack(outputs)
		cv2.imshow('iterations', outputs)
		imgs = self.vq.decode_indices(ids)
		return imgs