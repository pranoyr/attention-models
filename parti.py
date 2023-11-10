import math
from random import random
from functools import partial
from torch import Tensor

import torch
import torch.nn.functional as F
from torch import nn, einsum
import pathlib
from pathlib import Path
import torchvision.transforms as T
from t5 import TextEncoder, get_encoded_dim
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from typing import Callable, Optional, List

from einops import rearrange, repeat
from vqgan import VQGAN
from transformer import Decoder

def exists(val):
	return val is not None


class Parti(nn.Module):
	def __init__(
		self,
		dim,
		t5_name,
		codebook_size,
		n_heads,
		d_head,
		depth,
		):
		super().__init__()
		self.dim = dim
		
		#### Text Encoder  ####
	
		text_encoder = TextEncoder(t5_name)
		text_embed_dim = get_encoded_dim(t5_name)
		text_embed_proj = nn.Linear(text_embed_dim, dim, bias = False) 
		
		self.text_encoder =  nn.Sequential(text_encoder, text_embed_proj)
		
  
		#### Image Decoder ####
	
		self.start_token = nn.Parameter(torch.randn(dim))
		self.token_emb = nn.Embedding(codebook_size, dim)
		self.pos_enc =  nn.Parameter(torch.randn(1, dim))
		self.vqgan = VQGAN(dim, codebook_size)
		self.transformer_decoder = Decoder(dim, n_heads, d_head, depth)
		self.to_logits = nn.Linear(dim, codebook_size)
		
	def forward(
		self,
		texts : List[str],
		imgs: Tensor = None
		):

		device = imgs.device
		b = imgs.shape[0]

		# text encoder
		text_embeds = self.text_encoder(texts) # (batch_size, seq_len, dim)
  
		if exists(imgs):
			# convert images to indices
			img_token_indices = self.vqgan.encode_imgs(imgs)
			labels = img_token_indices.clone()
			# remove the last token
			img_token_indices = img_token_indices[:, :-1]
			#  convert indices to embeddings
			img_token_embeds = self.token_emb(img_token_indices) # (batch_size, seq_len, dim)
			# add positional encoding
			img_token_embeds += self.pos_enc
			# add start token
			start_token = repeat(self.start_token, 'd -> b 1 d', b=b)
			img_token_embeds = torch.cat((start_token, img_token_embeds), dim=1)
			# decoder
			x = self.transformer_decoder(dec_in=img_token_embeds, context=text_embeds)
			# to logits
			logits = self.to_logits(x)

			# calculate loss
			loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels)
			return loss

		else:
				# encode text
				text_embeds = self.text_encoder(text_encoder)
			
				start = torch.empty((1,0), dtype=torch.long, device="cuda")

				for step in range(0, 1024):
					dec_output = self.forward_with_cond_scale(gen_seq, context=context, context_mask=context_mask)[:,-1]
					# dec_output = self.forward(gen_seq, context=context, context_mask=context_mask)[:,-1]

		
					# dec_output = F.softmax(dec_output, dim=1)
					# dec_output = torch.argmax(dec_output, dim=1) 
		

					filtered_logits = top_k(dec_output, thres = 0.9)
					dec_output = gumbel_sample(filtered_logits, temperature = 1, dim = -1)
					
					dec_output = rearrange(dec_output, 'b -> b 1')
					gen_seq = torch.cat([gen_seq, dec_output], dim=-1)  #  gen -> (1,1024)
					#break




if __name__=="__main__":
	imgs = torch.randn(2, 3, 256, 256).cuda()
	texts = ["this is a test", "this is another test"]
	

	dim = 512
	encoder_params = dict(
		t5_name = "google/t5-v1_1-base",
	)
 
	decoder_params = dict(
		codebook_size = 8192,
		n_heads = 8,
		d_head	= 64,
		depth= 6)
 
	# Training
	model = Parti(dim, **encoder_params, **decoder_params).cuda()
	loss = model(texts, imgs)
	loss.backward()
 
	# Inference
	model.eval()
	with torch.no_grad():
		imgs = model(texts)
	print(imgs.shape)
