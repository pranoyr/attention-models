import torch.nn as nn
import torch
import copy
import torch.nn.functional as F
from einops import rearrange, repeat, pack
import math
from models.transformer import Encoder as BidirectionalDecorder


def cosine_schedule(t):
	return torch.cos(t * math.pi / 2)


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
		return x, tgt

	def forward(self, imgs):
		# quantize images
		x = self.vq.encode_imgs(imgs)
		
		x, tgt = self.fill_mask(x)
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