import torch.nn as nn
import torch
import copy
import torch.nn.functional as F
from einops import rearrange, repeat, pack
from models.softmax_attention import SoftmaxAttention
import math
from typing import List
from models.transformer import Decoder
from transformers import AutoTokenizer, CLIPTextModel
import random
from tqdm import tqdm

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


class TextEncoder(torch.nn.Module):
	def __init__(self, dim, enc_type, enc_name, max_length):
		super().__init__()
  
		self.max_length = max_length

		if enc_type == "clip":
			self.encoder = CLIPTextModel.from_pretrained(enc_name)
			self.tokenizer = AutoTokenizer.from_pretrained(enc_name)
		else:
			raise ValueError(f"Invalid encoder type {enc_type}")


	def forward(self, texts: List[str], device = None):
		inputs = self.tokenizer(texts, return_tensors="pt", max_length=self.max_length, padding="max_length")
		text_indices = inputs['input_ids'].to(device)
		attention_mask = inputs['attention_mask'].bool().to(device)
		text_embeds = self.encoder(text_indices).last_hidden_state
		return text_embeds, attention_mask


class BidirectionalDecoder(nn.Module):
	def __init__(self, dim, codebook_size , n_heads, d_head, depth, num_patches):
		super().__init__()
  
		self.token_emb = nn.Embedding(codebook_size + 1, dim)
		self.pos_enc = nn.Parameter(torch.randn(1, num_patches, dim))
		self.init_norm = nn.LayerNorm(dim)
		self.decoder = Decoder(dim=dim, n_heads=n_heads, d_head=d_head, depth=depth)
		self.final_norm = nn.LayerNorm(dim)
		self.linear = nn.Linear(dim, codebook_size)

	def forward(self, img_token_indices, context, context_mask):
		# add positional encoding
		img_token_embeds = self.token_emb(img_token_indices)
		img_token_embeds += self.pos_enc

		# bidirectional decoder
		img_token_embeds = self.init_norm(img_token_embeds)
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
	):
		super().__init__()

		#### Text Encoder  ####
		self.text_encoder= TextEncoder(dim, enc_type, enc_name, max_length)
		self.context_norm = nn.LayerNorm(dim)
		
		#### Vector Quantizer ####
		self.vq = vq
		codebook_size = vq.codebook.codebook_size
		self.mask_token_id = codebook_size
		num_patches = vq.num_patches

		#### Transformer Decoder ####
		self.decoder = BidirectionalDecoder(dim, codebook_size, n_heads, d_head, depth, num_patches)
  
		self.ignore_index = -1

		# freeze the text encoder and vq
		self.text_encoder.requires_grad_(False)
		self.vq.requires_grad_(False)

	def fill_mask(self, x, T = 18):
		device = x.device

		# sample the timestep from uniform distribution
		b, n = x.shape
		# t = torch.randint(0, T, (1,))
		# num_tokens_masked = cosine_schedule(t / T) * n
		# num_tokens_masked = num_tokens_masked.clamp(min=1.0).int()
		# num_tokens_masked = num_tokens_masked.to(device)
		rand_time = uniform((b,), device = device)
		rand_mask_probs = cosine_schedule(rand_time)
		num_tokens_masked = (n * rand_mask_probs).round().clamp(min = 1)

		# create mask
		randm_perm = torch.rand((b, n), device = device).argsort(dim = -1)
		mask = randm_perm < rearrange(num_tokens_masked, 'b -> b 1')

		# ignore the tokens that are not masked while computing loss
		tgt = x.masked_fill(~mask, self.ignore_index)
		# fill x with mask_id where mask is True
		x = x.masked_fill(mask, self.mask_token_id)
		return x, tgt

	def forward(self, texts, imgs):
		# text encoder
		device = imgs.device
		text_embeds, context_mask = self.text_encoder(texts, device)
		text_embeds = self.context_norm(text_embeds)

		# quantize images
		img_token_indices = self.vq.encode_imgs(imgs)

		# apply cosine schedule to img tokens
		img_token_indices, tgt = self.fill_mask(img_token_indices)

		# decoder forward
		logits = self.decoder(img_token_indices, context=text_embeds, context_mask=context_mask)

	 	# self conditioning (for classifier free guidance)
		if random.random() < 0.9:
			context_mask = torch.zeros_like(context_mask).bool().to(device)
			logits = self.decoder(img_token_indices, context=text_embeds, context_mask=context_mask)

		# compute loss
		logits = rearrange(logits, "b t c -> b c t")
		loss = torch.nn.functional.cross_entropy(logits, tgt, ignore_index=self.ignore_index)
		return loss

	def generate(self, texts, timesteps = 18):
		b = len(texts)
		num_patches = self.vq.num_patches

		device = "cuda" if torch.cuda.is_available() else "cpu"

		# text encoder
		text_embeds, context_mask = self.text_encoder(texts, device=device)
		text_embeds = self.context_norm(text_embeds)
  
		# initialize decoder inputs
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
			logits = self.decoder(ids, context=text_embeds, context_mask=context_mask)

			# for classifier free guidance
			zeros_mask = torch.zeros_like(context_mask).bool().to(device)
			null_logits = self.decoder(ids, context=text_embeds, context_mask=zeros_mask)
			scaled_logits = null_logits + (logits - null_logits) * 3

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