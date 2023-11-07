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
from .t5 import TextEncoder, get_encoded_dim
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from typing import Callable, Optional, List

from einops import rearrange, repeat
from vqgan import VQGAN



class TransformerDecoder(nn.Module):
    def __init__(self, dim, n_heads, d_head, depth):
        super().__init__()

        encoder_layer = EncoderLayer(dim, n_heads, d_head)

        self.layers = _get_clones(encoder_layer, depth)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x



class Parti(nn.Module):
	def __init__(
		self,
		dim,
		**kwargs,
		decoder_kwargs
		):
		super().__init__()
		self.dim = dim
		
		#### Text Encoder  ####
	
		text_encoder = TextEncoder(encoder_kwargs)
		text_embed_dim = get_encoded_dim(t5_name)
		stext_embed_proj = nn.Linear(text_embed_dim, dim, bias = False) 
		
		self.text_encoder =  nn.Sequential(text_encoder, text_embed_proj)
		
  
		#### Image Decoder  ####
	
		# learnable start token for decoding
		self.start_token = nn.Parameter(torch.randn(dim))
		self.vqgan = VQGAN(dim, codebook_size)
		
  


	



	def forward(
		self,
		text : List[str],
		img: Tensor
	):
		device, b, n = x.device, *x.shape

		# encode text
		text_embeds, context_mask = self.text_encoder(texts, output_device=device)
		context = self.text_embed_proj(text_embeds)
		
		
		# encode image
		



		

		# image token embedding
		# if x.shape[1] > 0:
		x = self.token_emb(x)
		# add position embedding
		# pos_emb = self.pos_emb(x)
		# x = x + axial_pos_emb[:n]

		x = self.pos_emb(x) + x


		
		# add start token
		start_token = repeat(self.start_token, 'd -> b 1 d', b=b)
		x = torch.cat((start_token, x), dim=1)

		x = self.init_norm(x)
		context = self.enc_norm(context)


		# else:
		# 	x = repeat(self.start_token, 'd -> b 1 d', b=b)

		# context, context_mask = map(lambda t: t[:, :self.max_text_len], (context, context_mask))

		if cond_drop_prob > 0:
			keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device = device)
			context_mask = rearrange(keep_mask, 'b -> b 1') & context_mask

		# decoder
		x = self.transformer_blocks(x, context = context, context_mask = context_mask)

		# to logits
		logits = self.to_logits(x)

		if exists(labels):
			# calculate loss
			loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels)
			return loss, logits
		else:
			return logits


	def generate(self, src):
			with torch.no_grad():

				# encode text
				text_embeds, context_mask = self.text_encoder(src, "cuda")
				context = self.text_embed_proj(text_embeds)
				# context_mask = (text_embeds != 0).any(dim = -1)


				gen_seq = torch.empty((1,0), dtype=torch.long, device="cuda")

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
					
				return gen_seq




if __name__=="__main__":
	imgs = torch.randn(2, 3, 256, 256)
	texts = ["this is a test", "this is another test"]
	
	encoder_params = dict(
		t5_name = "t5-large",
	)
 
	decoder_params = dict(
		code_book_size = 8192,
		n_heads = 8,
		d_head	= 64,
		depth= 6)
 
 
	model = Parti(dim, **encoder_params, **decoder_params)
	out = model(imgs, texts)
	print(out.shape)
