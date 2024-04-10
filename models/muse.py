import torch.nn as nn
import torch
import copy
import torch.nn.functional as F
from einops import rearrange, repeat, pack
from models.softmax_attention import SoftmaxAttention
import math
from typing import List
from models.transformer import Decoder
from .transformer import LayerNorm, FeedForward
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer
import random
from tqdm import tqdm


def log(t, eps = 1e-20):
	return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
	noise = torch.zeros_like(t).uniform_(0, 1)
	return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
	return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)


def cosine_schedule(t):
	return torch.cos(t * math.pi / 2)

def filter_logits(logits, p=0.9):
	n_classes  = logits.shape[-1]
	k = math.ceil((1 - p) * n_classes)
	val, ind = logits.topk(k, dim = -1)
	filtered_logits = torch.full_like(logits, float('-inf'))
	filtered_logits.scatter_(2, ind, val)
	return filtered_logits

def uniform(shape, min = 0, max = 1, device = None):
	return torch.zeros(shape, device = device).float().uniform_(0, 1)

def exists(val):
	return val is not None


class TextEncoder(torch.nn.Module):
	def __init__(self, dim, enc_type, enc_name, max_length):
		super().__init__()
  
		self.max_length = max_length
  
		# self.context_norm = LayerNorm(dim)

		if enc_type == "clip":
			self.encoder = CLIPTextModel.from_pretrained(enc_name)
			self.tokenizer = CLIPTokenizer.from_pretrained(enc_name)
		else:
			raise ValueError(f"Invalid encoder type {enc_type}")

		self.project_embeds = torch.nn.Linear(768, dim)


	# def forward(self, texts: List[str], device = None):
	# 	inputs = self.tokenizer(texts, return_tensors="pt", max_length=self.max_length, padding="max_length")
	# 	text_indices = inputs['input_ids'].to(device)
	# 	attention_mask = inputs['attention_mask'].bool().to(device)
	# 	text_embeds = self.encoder(text_indices).last_hidden_state
	# 	return text_embeds, attention_mask
 
	def forward(self, texts: List[str], device = None):
		inputs = self.tokenizer(texts, return_tensors="pt", max_length=self.max_length, padding="max_length")
		text_indices = inputs['input_ids'].to(device)
		text_embeds = self.encoder(text_indices,  return_dict=True, output_hidden_states=True)[0]
		text_embeds = self.project_embeds(text_embeds)
		return text_embeds, text_embeds



class BidirectionalDecoder(nn.Module):
	def __init__(self, dim, codebook_size, n_heads, d_head, depth, mult, dropout, num_patches):
		super().__init__()
  
		self.token_emb = nn.Embedding(codebook_size + 1, dim)
		self.pos_enc = nn.Parameter(torch.randn(1, num_patches, dim))
		# self.init_norm = LayerNorm(dim)
		self.decoder = Decoder(dim=dim, n_heads=n_heads, d_head=d_head, depth=depth, mult=mult, dropout=dropout)
		self.final_norm = LayerNorm(dim)
		self.linear = nn.Linear(dim, codebook_size, bias=False)
  
		self.apply(self._init_weights)
  

	def _init_weights(self, module):
		"""
		Initialize the weights according to the original implementation.
		https://github.com/google-research/maskgit/blob/main/maskgit/nets/maskgit_transformer.py#L37
		"""
		if isinstance(module, nn.Linear):
			nn.init.trunc_normal_(module.weight, std=0.02)
			if module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, nn.Embedding):
			nn.init.trunc_normal_(module.weight, std=0.02)
		elif isinstance(module, (nn.LayerNorm)):
			if hasattr(module, "weight") and module.weight is not None:
				module.weight.data.fill_(1.0)
			if hasattr(module, "bias") and module.bias is not None:
				module.bias.data.zero_()

	def forward(self, img_token_indices, context=None, context_mask=None):
		# add positional encoding
		img_token_embeds = self.token_emb(img_token_indices)
		img_token_embeds += self.pos_enc
  
		dec_out = self.decoder(dec_in=img_token_embeds, context=context, context_mask=context_mask)
		dec_out = self.final_norm(dec_out)
		logits = self.linear(dec_out)
		return logits


class MUSE(nn.Module):
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
		mult,
		dropout
	):
		super().__init__()

		#### Text Encoder  ####
		self.text_encoder= TextEncoder(dim, enc_type, enc_name, max_length)
		
	
		#### Vector Quantizer ####
		self.vq = vq
		codebook_size = vq.codebook.codebook_size
		# codebook_size = 16384
		self.mask_token_id = codebook_size
		num_patches = vq.num_patches
		# num_patches = 256

		#### Transformer Decoder ####
		self.decoder = BidirectionalDecoder(dim, codebook_size, n_heads, d_head, depth, mult, dropout, num_patches)
  
		self.ignore_index = -1

		# freeze the text encoder and vq
		self.text_encoder.requires_grad_(False)
  
		self.vq.requires_grad_(False)
 
	def fill_mask(self, image_tokens):
		batch_size, seq_len = image_tokens.shape

		# Sample a random timestep for each image
		timesteps = torch.rand(batch_size, device=image_tokens.device)
		mask_prob = cosine_schedule(timesteps)
		mask_prob = mask_prob.clip(0)
		# creat a random mask for each image
		num_token_masked = (seq_len * mask_prob).round().clamp(min=1)
		batch_randperm = torch.rand(batch_size, seq_len, device=image_tokens.device).argsort(dim=-1)
		mask = batch_randperm < num_token_masked.unsqueeze(-1)
		# mask images and create input and labels
		input_ids = image_tokens.masked_fill(mask, self.mask_token_id)
		labels = image_tokens.masked_fill(~mask, self.ignore_index)
	
		return input_ids, labels
  

	def forward(self, texts, imgs):
		# text encoder
		device = imgs.device
		b = len(texts)
  
		text_embeds, _ = self.text_encoder(texts, device)

		# quantize images
		with torch.no_grad():
			img_token_indices = self.vq.encode_imgs(imgs)
	
		# apply cosine schedule to img tokens
		img_token_indices, tgt = self.fill_mask(img_token_indices)
  
		# self conditioning (for classifier free guidance)
		context_mask = uniform((b,1,1), device=device) < 0.9
		text_embeds = text_embeds * context_mask
  


		logits = self.decoder(img_token_indices, context=text_embeds)
		
	 	
		logits = rearrange(logits, "b t c -> b c t")
		loss = torch.nn.functional.cross_entropy(logits, tgt, ignore_index=self.ignore_index)
		return loss

	def generate2(self, texts, timesteps = 18, device = None):
	 
		starting_temperature = 1
  
		b = len(texts)
		seq_len = self.vq.num_patches
		batch_size = b
  
		shape = (batch_size, seq_len)
  		# text encoder
		text_embeds, _ = self.text_encoder(texts, device=device)

		ids = torch.full(shape, self.mask_token_id, dtype = torch.long, device = device)
		scores = torch.zeros(shape, dtype = torch.float32, device = device) 
	 
		for timestep, steps_until_x0 in tqdm(zip(torch.linspace(0, 1, timesteps, device = device), reversed(range(timesteps))), total = timesteps):
	
			rand_mask_prob = cosine_schedule(timestep)
			num_token_masked = max(int((rand_mask_prob * seq_len).item()), 1)

			masked_indices = scores.topk(num_token_masked, dim = -1).indices

			ids = ids.scatter(1, masked_indices, self.mask_token_id)

			logits = self.decoder(ids, context=text_embeds)
   
   			# for classifier free guidance
			null_context = torch.zeros_like(text_embeds).bool().to(device)
			null_logits = self.decoder(ids, context=null_context)
			# scaled_logits = (1  + 3) * logits - 3 * null_logits
			scaled_logits = null_logits + 3 * (logits - null_logits)

			filtered_logits = filter_logits(scaled_logits, p=0.9)
   
		

			temperature = starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

			pred_ids = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

			is_mask = ids == self.mask_token_id

			ids = torch.where(
				is_mask,
				pred_ids,
				ids
			)

			
			probs_without_temperature = scaled_logits.softmax(dim = -1)

			scores = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
			scores = rearrange(scores, '... 1 -> ...')

	
		imgs = self.vq.decode_indices(ids)
		return imgs

	def generate(self, texts, timesteps = 18, device = None):
		b = len(texts)
		num_patches = self.vq.num_patches 
		# num_patches = 256

		if not device:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# text encoder
		text_embeds, _ = self.text_encoder(texts, device=device)
  
		# initialize decoder input
		ids = torch.ones(b, num_patches, dtype=torch.long, device=device) * self.mask_token_id
		scores = torch.zeros_like(ids).float().to(device)
		mask = torch.zeros_like(ids).bool().to(device)

		b , n = ids.shape

		for timestep, steps_until_x0 in tqdm(zip(torch.linspace(0, 1, timesteps, device = device), reversed(range(timesteps))), total = timesteps):
			# number of tokens to mask with cosine schedule
			rand_mask_prob = cosine_schedule(timestep)
			num_tokens_masked = max(int((rand_mask_prob * n).item()), 1)
			# find low probability tokens
			low_probs_indices = torch.argsort(scores, dim = -1)
			
			# indices of tokens to mask
			masked_indices = low_probs_indices[:, :num_tokens_masked]
			
			# True where the tokens are masked, False otherwise
			mask.scatter_(1, masked_indices, True)
			
			# mask the low probability tokens with mask_id
			ids = ids.masked_fill(mask, self.mask_token_id)

			# decoder forward
			logits = self.decoder(ids, context=text_embeds, context_mask=None)

			# for classifier free guidance
			zeros_mask = torch.zeros_like(text_embeds).to(device)
			null_logits = self.decoder(ids, context=zeros_mask)
			# scaled_logits = (1  + 3) * logits - 3 * null_logits
			scaled_logits = null_logits + 3 * (logits - null_logits)

			probs = F.softmax(scaled_logits, dim = -1)
   
			# decaying temperature
			temperature = 1 * (steps_until_x0 / timesteps) # temperature is annealed
			
			# sample with gumbel softmax
			scaled_logits = filter_logits(scaled_logits, p=0.9)

			pred_ids = F.gumbel_softmax(scaled_logits, tau = temperature, hard = False, dim = -1).argmax(dim = -1)

			# fill the masked tokens with predicted tokens
			ids[mask] = pred_ids[mask]
			
			# update scores
			scores = probs.gather(2, rearrange(pred_ids, 'b t -> b t 1'))
			scores = rearrange(scores, 'b t 1 -> b t')
			
			mask = torch.zeros_like(ids).bool()

		imgs = self.vq.decode_indices(ids)
		return imgs