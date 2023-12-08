from torch import Tensor

import torch
import torch.nn.functional as F
from torch import nn, einsum
import torchvision.transforms as T
from models.t5 import T5Encoder, get_encoded_dim
from einops import rearrange, repeat, pack
from typing import Callable, Optional, List
from models.positional_encoding import PositionalEncoding

from einops import rearrange, repeat
from models.transformer import Decoder
from models.vqgan import VQGAN

def exists(val):
	return val is not None


class TextEncoder(torch.nn.Module):
	def __init__(self, dim, t5_name, max_length):
		super().__init__()
	
		self.t5_encoder = T5Encoder(t5_name, max_length)
		text_embed_dim = get_encoded_dim(t5_name)
		self.text_embed_proj = nn.Linear(text_embed_dim, dim, bias = False)
		self.layer_norm = nn.LayerNorm(dim)
	
	def forward(self, texts: List[str]):
		context_mask, text_embeds = self.t5_encoder(texts)
		text_embeds = self.text_embed_proj(text_embeds)
		return context_mask, text_embeds
		

class Parti(nn.Module):
	def __init__(
		self,
		dim,
		vq,
		t5_name,
		max_length,
		n_heads,
		d_head,
		depth,
		):
		super().__init__()
		self.dim = dim
		self.vq = vq
		
		#### Text Encoder  ####
		self.text_encoder = TextEncoder(dim, t5_name, max_length)
		self.context_norm = nn.LayerNorm(dim)
		
		#### Transformer Decoder ####
		self.start_token = nn.Parameter(torch.randn(dim))
		codebook_size = vq.codebook.codebook_size
		self.token_emb = nn.Embedding(codebook_size, dim)
		num_patches = vq.num_patches - 1
		self.pos_enc =  PositionalEncoding(dim)
		
		self.transformer_decoder = Decoder(dim, n_heads, d_head, depth)
		self.init_norm = nn.LayerNorm(dim)
		self.final_norm = nn.LayerNorm(dim)
		self.to_logits = nn.Linear(dim, codebook_size)
		
	def forward(
		self,
		texts : List[str],
		imgs: Tensor = None
		):

		b = imgs.shape[0]
		device = imgs.device

		# text encoder
		context_mask , text_embeds = self.text_encoder(texts) # (batch_size, seq_len, dim)
		text_embeds = self.context_norm(text_embeds)
  
		# convert images to indices
		img_token_indices = self.vq.encode_imgs(imgs)
		# remove the last token for decoder input
		img_token_indices , labels = img_token_indices[:, :-1], img_token_indices
		#  convert indices to embeddings
		img_token_embeds = self.token_emb(img_token_indices) # (batch_size, seq_len, dim)
		# add positional encoding
		img_token_embeds = self.pos_enc(img_token_embeds)
		# add start token
		start_token = repeat(self.start_token, 'd -> b 1 d', b=b)
		img_token_embeds = torch.cat((start_token, img_token_embeds), dim=1)
   
		# decoder
		# causal mask for transformer decoder
		i = j = img_token_embeds.shape[1]
		causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
		# add init norm
		img_token_embeds = self.init_norm(img_token_embeds)
		# transformer decoder
		dec_out = self.transformer_decoder(dec_in=img_token_embeds, context=text_embeds, context_mask=context_mask, causal_mask=causal_mask)\
		# add final norm
		dec_out = self.final_norm(dec_out)
		# to logits
		logits = self.to_logits(dec_out)

		# calculate loss
		loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels)
		return loss

	def generate(self, texts : List[str]):
		b = len(texts)
		# text encoder
		context_mask , text_embeds = self.text_encoder(texts) # (batch_size, seq_len, dim)

		start_token = repeat(self.start_token, 'd -> b 1 d', b=b)
		
		indices = torch.zeros(b, 0, dtype=torch.long, device=text_embeds.device)	
		num_iters = self.vq.num_patches
		for i in range(num_iters):
			img_token_embeds = self.token_emb(indices) # (batch_size, seq_len, dim)
			# add positional encoding
			img_token_embeds = self.pos_enc(img_token_embeds)
			# add start token
			img_token_embeds = torch.cat((start_token, img_token_embeds), dim=1)
			# decoder
			self.init_norm(img_token_embeds)
			dec_out = self.transformer_decoder(dec_in=img_token_embeds, context=text_embeds, context_mask=context_mask)
			self.final_norm(dec_out)
			# to logits
			logits = self.to_logits(dec_out)
			# sample 
			last_token = F.gumbel_softmax(logits[:, -1, :], tau=1, hard=False)
			last_token = torch.argmax(last_token, dim=-1)
			last_token = rearrange(last_token, 'b -> b 1')
			indices = pack((indices, last_token), "b *")[0]
   
		imgs = self.vq.decode_indices(indices)
		return(imgs)