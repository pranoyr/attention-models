from torch import Tensor

import torch
import torch.nn.functional as F
from torch import nn, einsum
import torchvision.transforms as T
from einops import rearrange, repeat, pack
from typing import Callable, Optional, List
from models.positional_encoding import PositionalEncoding
import math

from einops import rearrange, repeat
from models.transformer import Decoder
from models.vqgan import VQGAN
from transformers import AutoTokenizer, CLIPTextModel

def exists(val):
	return val is not None

def filter_logits(logits, p=0.9):
	n_classes  = logits.shape[-1]
	k = math.ceil((1 - p) * n_classes)
	val, ind = logits.topk(k, dim = -1)
	filtered_logits = torch.full_like(logits, float('-inf'))
	filtered_logits.scatter_(2, ind, val)
	return filtered_logits


class TextEncoder(torch.nn.Module):
	def __init__(self, dim, enc_type, enc_name, max_length):
		super().__init__()
  
		self.max_length = max_length

		if enc_type == "clip":
			self.encoder = CLIPTextModel.from_pretrained(enc_name)
			self.tokenizer = AutoTokenizer.from_pretrained(enc_name)
		else:
			raise ValueError(f"Invalid encoder type {enc_type}")

	def forward(self, texts: List[str]):
		inputs = self.tokenizer(texts, return_tensors="pt", max_length=self.max_length, padding="max_length")["input_ids"]
		text_embeds = self.encoder(inputs.cuda()).last_hidden_state
		return text_embeds


# TODO: add classifier free guidance	

class Parti(nn.Module):
	def __init__(
		self,
		dim,
		vq,
		enc_type,
		enc_name,
		max_length,
		n_heads,
		d_head,
		depth,
		):
		super().__init__()
		self.dim = dim
		self.vq = vq
		
		#### Text Encoder  ####
		self.text_encoder = TextEncoder(dim, enc_type, enc_name, max_length)
		self.context_norm = nn.LayerNorm(dim)
		
		#### Transformer Decoder ####
		self.start_token = nn.Parameter(torch.randn(dim))
		codebook_size = vq.codebook.codebook_size
		self.token_emb = nn.Embedding(codebook_size, dim)
		self.pos_enc =  PositionalEncoding(dim)
		
		self.transformer_decoder = Decoder(dim, n_heads, d_head, depth)
		self.init_norm = nn.LayerNorm(dim)
		self.final_norm = nn.LayerNorm(dim)
		self.to_logits = nn.Linear(dim, codebook_size)

		# freeze the text encoder and vqgan
		self.text_encoder.requires_grad_(False)
		self.vq.requires_grad_(False)
		
	def forward(
		self,
		texts : List[str],
		imgs: Tensor = None
		):

		b = imgs.shape[0]
		device = imgs.device

		# text encoder
		text_embeds = self.text_encoder(texts) # (batch_size, seq_len, dim)
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
		dec_out = self.transformer_decoder(dec_in=img_token_embeds, context=text_embeds, causal_mask=causal_mask)
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
		text_embeds = self.text_encoder(texts) # (batch_size, seq_len, dim)

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
			dec_out = self.transformer_decoder(dec_in=img_token_embeds, context=text_embeds)
			self.final_norm(dec_out)
			# to logits
			logits = self.to_logits(dec_out)
			# sample 
			logits = filter_logits(logits, p=0.9)
			last_token = F.gumbel_softmax(logits[:, -1, :], tau=1, hard=False)
			last_token = torch.argmax(last_token, dim=-1)
			last_token = rearrange(last_token, 'b -> b 1')
			indices = pack((indices, last_token), "b *")[0]
   
		imgs = self.vq.decode_indices(indices)
		return(imgs)
