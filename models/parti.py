from torch import Tensor

import torch
import torch.nn.functional as F
from torch import nn, einsum
import torchvision.transforms as T
from models.t5 import T5Encoder, get_encoded_dim
from einops import rearrange, repeat, pack
from typing import Callable, Optional, List

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
		# add layer norm
		text_embeds = self.layer_norm(text_embeds)
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
		
		#### Text Encoder  ####
		self.text_encoder = TextEncoder(dim, t5_name, max_length)
		
		#### Transformer Decoder ####
		self.start_token = nn.Parameter(torch.randn(dim))
		codebook_size = vq.codebook.codebook_size
		self.token_emb = nn.Embedding(codebook_size, dim)
		self.pos_enc =  nn.Parameter(torch.randn(1, dim))
		self.vq = vq
		
		self.transformer_decoder = Decoder(dim, n_heads, d_head, depth)
		self.layer_norm = nn.LayerNorm(dim)
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
  
		# convert images to indices
		img_token_indices = self.vq.encode_imgs(imgs)
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
		# causal mask for transformer decoder
		i = j = img_token_embeds.shape[1]
		causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
		# add layer norm
		img_token_embeds = self.layer_norm(img_token_embeds)
		# transformer decoder
		dec_out = self.transformer_decoder(dec_in=img_token_embeds, context=text_embeds, context_mask=context_mask, causal_mask=causal_mask)
		# to logits
		logits = self.to_logits(dec_out)

		# calculate loss
		loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels)
		return loss

	def generate(self, texts : List[str]):

		b = len(texts)
		context_mask , text_embeds = self.text_encoder(texts) # (batch_size, seq_len, dim)

		start_token = repeat(self.start_token, 'd -> b 1 d', b=b)
  
		indices = torch.zeros(b, 0, dtype=torch.long, device=text_embeds.device)			
		for i in range(256):
		
			img_token_embeds = self.token_emb(indices) # (batch_size, seq_len, dim)
			# add positional encoding
			img_token_embeds += self.pos_enc
			# add start token
			img_token_embeds = torch.cat((start_token, img_token_embeds), dim=1)
			# decoder
			dec_out = self.transformer_decoder(dec_in=img_token_embeds, context=text_embeds, context_mask=context_mask)
			# to logits
			logits = self.to_logits(dec_out)
   
			# sample
			probs = F.softmax(logits, dim=-1)
			idx = torch.argmax(probs, dim=-1)[:,-1]
			idx = rearrange(idx, 'b -> b 1')
			indices = pack((indices, idx), "b *")[0]
		

		imgs = self.vq.decode_indices(indices)
		return(imgs)



if __name__=="__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	imgs = torch.randn(2, 3, 256, 256).to(device)
	texts = ["this is a test", "this is another test"]
	
	# Vector Quantizer 
	codebook_dim = 256
	codebook_size = 8192
	vq = VQGAN(codebook_dim, codebook_size)

	# Parti 
	dim = 512
	encoder_params = dict(
		t5_name = "google/t5-v1_1-base",
	)
 
	decoder_params = dict(
		n_heads = 8,
		d_head	= 64,
		depth= 6)
 
	model = Parti(dim, vq, **encoder_params, **decoder_params).to(device)
	loss = model(texts, imgs)
	loss.backward()
 
	# Inference
	model.eval()
	with torch.no_grad():
		imgs = model.generate(texts)
	print(imgs.shape)
