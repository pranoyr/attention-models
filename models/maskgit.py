import torch.nn as nn
import torch
import copy
import torch.nn.functional as F
from einops import rearrange, repeat, pack
import math
from models.transformer import Encoder 
from models.muse import filter_logits
from tqdm import tqdm
from typing import Optional
import numpy as np
import cv2

def cosine_schedule(t):
	return torch.cos(t * math.pi / 2)



class LayerNorm(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.gamma = nn.Parameter(torch.ones(dim))
		# we don't want to update this
		self.register_buffer("beta", torch.zeros(dim))

	def forward(self, x):
		return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


def weights_init(m):
	classname = m.__class__.__name__
	if "Linear" in classname or "Embedding" == classname:
		print(f"Initializing Module {classname}.")
		nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)
	# elif "Parameter" in classname:
	#     return nn.init.trunc_normal_(m, 0.0, 0.02)


def restore(x):
	# x = (x + 1) * 0.5
	x = x.clamp(0,1)
	x = x.permute(1,2,0).detach().cpu().numpy()
	x = (255*x).astype(np.uint8)
	return x


def exists(val):
	return val is not None


class BiDirectionalTransformer(nn.Module):
	def __init__(
		self,
		dim,
		vocab_size=8192,
		num_patches=256,
		n_heads=8,
		d_head=64,
		dec_depth=6,
		mult=4,
		dropout=0.1
	):
		super().__init__()


		self.input_proj = nn.Embedding(vocab_size+1, dim)
		# self.pos_enc =  nn.Parameter(torch.randn(1, num_patches, dim))
		self.pos_enc = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, num_patches, dim)), 0., 0.02)
		self.mask_token_id = vocab_size

		self.init_norm = LayerNorm(dim)
		self.decoder = Encoder(
			dim=dim, n_heads=n_heads, d_head=d_head, depth=dec_depth, mult=mult, dropout=dropout
		)
		self.final_norm = LayerNorm(dim)
		self.linear = nn.Linear(dim, vocab_size, bias=False)
		self.apply(weights_init)
	
	def forward(self, x):
		x = self.input_proj(x)
		x += self.pos_enc

		# transformer decoder
		x = self.init_norm(x)
		dec_out = self.decoder(x)
		dec_out = self.final_norm(dec_out)
		output = self.linear(dec_out)
		return output
		

class MaskGitTransformer(nn.Module):
	def __init__(
		self,
		dim,
		vq,
		vocab_size=8192,
		n_heads=8,
		d_head=64,
		dec_depth=6,
		mult=4,
		dropout=0.1
	):
		super().__init__()
  
		self.vq = vq


		self.bidirectional_transformer = BiDirectionalTransformer (
			dim=dim, vocab_size=vocab_size, num_patches=vq.num_patches,
			n_heads=n_heads, d_head=d_head, dec_depth=dec_depth, mult=mult, dropout=dropout
		)
		
		self.mask_token_id = vocab_size

		# freeze vq
		self.vq.eval()
		self.vq.requires_grad_(False)
		
	def fill_mask(self, x):

		b , n = x.shape
		timesteps = torch.random(b)
		num_tokens_masked = cosine_schedule(timesteps) * n
		num_tokens_masked = num_tokens_masked.clamp(min = 1.).int()
   
		# create mask 
		randm_perm = torch.rand(x.shape).argsort(dim = -1)
		mask = randm_perm < num_tokens_masked
		mask = mask.cuda()
		
		# fill x with mask_id, ignore the tokens that are not masked while computing loss
		tgt = x.masked_fill(~mask, -1)
		x = x.masked_fill(mask, self.mask_token_id)    
		return x, tgt, mask

	def fill_custom_mask(self, x, num_masked = 200):
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
		# mask[:, :num_masked] = True
		# put last 200 as True
		# mask[:, -num_masked:] = True
		mask[:, :num_masked] = True


		mask = mask.cuda()
  
		
		# fill x with mask_id, ignore the tokens that are not masked while computing loss
		tgt = x.masked_fill(~mask, -1)
		# x = x.masked_fill(mask, self.mask_token_id)    
		return x, tgt, mask

	def forward(self, imgs):
		# quantize images
		x = self.vq.encode_imgs(imgs)
		
		x, tgt, mask = self.fill_mask(x)

		# transformer decoder
		output = self.bidirectional_transformer(x)
		# x = self.init_norm(x)
		# dec_out = self.decoder(x, context=None)
		# dec_out = self.final_norm(dec_out)
		# output = self.linear(dec_out)

		if not self.training:
			pred_ids = torch.softmax(output, dim=-1).argmax(dim=-1)
			# replace the mask with pred_ids
			
			x[mask] = pred_ids[mask]
			
			decoded_imgs = self.vq.decode_indices(x)
			return decoded_imgs
			
		# compute loss
		output = rearrange(output, 'b t c -> b c t')
		loss = torch.nn.functional.cross_entropy(output, tgt, ignore_index=-1)
  
		return loss    

	def generate(self, imgs : Optional[torch.Tensor] = None, num_masked=200,  timesteps = 18):
			 
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
			ids , _, mask = self.fill_custom_mask(ids, num_masked)
			# print number of true
			print(f"Number of Tokens Masked: {mask.sum()}")
			scores = torch.zeros_like(ids).float().to(device)


			b , n = ids.shape
	

			ids2 = ids.masked_fill(mask, 100)
			decoded_imgs = self.vq.decode_indices(ids2)
			# display
			img = restore(decoded_imgs[0])
			img = img[:, :, ::-1]
			cv2.imwrite('outputs/maskgit/test_outputs/masked.jpg', img)
		
		outputs = []
		for timestep, steps_until_x0 in tqdm(zip(torch.linspace(0, 1, timesteps, device = device), reversed(range(timesteps))), total = timesteps):
			
			rand_mask_prob = cosine_schedule(timestep)
			num_tokens_masked = max(int((rand_mask_prob * num_masked).item()), 1)
			print(num_tokens_masked)
   
   
			low_probs_indices = torch.argsort(scores, dim = -1)	
			# indices of tokens to mask
			masked_indices = low_probs_indices[:, :num_tokens_masked]
   			# True where the tokens are masked, False otherwise
			mask.scatter_(1, masked_indices, True)
			
   
			ids2 = ids.masked_fill(mask, 100)
			decoded_imgs_1 = self.vq.decode_indices(ids2)
			# display
			img_1 = restore(decoded_imgs_1[0])
			img_1 = img_1[:, :, ::-1]
   
   
   
			x = ids.masked_fill(mask, self.mask_token_id)    
			# decoder forward
			logits = self.bidirectional_transformer(x)
		

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
			# scores = scores.masked_fill(~mask, 1.0)
			scores = scores.masked_fill(~mask, 1.0)
   

			mask = torch.zeros_like(ids).bool().to(device)
		
	

			decoded_imgs = self.vq.decode_indices(ids)
			# display
			img = restore(decoded_imgs[0])
			img = img[:, :, ::-1]
			img_combined = np.vstack([img, img_1])
			outputs.append(img_combined)
			
			

		outputs  = np.hstack(outputs)
		cv2.imwrite('outputs/maskgit/test_outputs/iterations.jpg', outputs)
		imgs = self.vq.decode_indices(ids)
		return imgs