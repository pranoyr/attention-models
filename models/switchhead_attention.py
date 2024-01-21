import torch
import torch.nn as nn
import math
from einops import rearrange, repeat, pack
from torch import einsum
from einops.layers.torch import Rearrange


# helper functions
def exists(val):
	return val is not None

def default(val, d):
	return val if exists(val) else d


class SwitchHeadAttention(nn.Module):
	def __init__(self, dim, num_heads=8, dim_head=64, num_experts=5, sel_experts=3,  act_fn="sigmoid", dropout=0.0):
		super(SwitchHeadAttention, self).__init__()

		self.dim = dim
		self.num_heads = num_heads
		self.dim_head = dim_head
		self.num_experts = num_experts
		self.sel_experts = sel_experts
		
		self.act_fn = nn.Sigmoid() if act_fn == "sigmoid" else nn.Softmax(dim=-1)

		self.q = nn.Sequential(
			nn.Linear(dim, num_heads * dim_head, bias=False),
			nn.Dropout(dropout),
			Rearrange('b t (h d) -> b h t d', h=self.num_heads)
		)

		self.k = nn.Sequential(
			nn.Linear(dim, num_heads * dim_head, bias=False),
			nn.Dropout(dropout),
			Rearrange('b t (h d) -> b h t d', h=self.num_heads)
		)

		self.v = nn.Sequential(
			nn.Linear(dim, num_heads * num_experts * dim_head, bias=False),
			nn.Dropout(dropout),
			Rearrange('b t (h e d) -> b t h e d', d = self.dim_head, h = self.num_heads, e=self.num_experts)
		)

		self.W_s = nn.Sequential(
			nn.Linear(dim, num_heads * num_experts, bias=False),
			Rearrange('b t (h e) -> b t h e', h=self.num_heads, e=self.num_experts)
		)

		self.W_d =  nn.Sequential(
			nn.Linear(dim, num_heads * num_experts, bias=False),
			Rearrange('b t (h e) -> b t h e', h=self.num_heads, e=self.num_experts)
		)
	
		self.W_o = nn.Sequential(nn.Conv2d(num_heads , num_heads * dim * num_experts , (1 , dim_head) , groups = num_heads, bias=False),
					Rearrange('b (h e d) t 1-> b t h e d' , h = self.num_heads , e = self.num_experts))

		self.scale = dim_head ** -0.5


	def topk_scores(self, scores, hard=False):
		eps = scores.topk(k=self.sel_experts, dim=-1).indices
	
		mask = torch.zeros_like(scores).scatter_(-1, eps, 1)

		if hard:
			return mask

		scores = scores * mask
		return scores
	
	def forward(self, x, context=None, causal_mask=None, context_mask=None):

		# prepare source-side
		ss = self.act_fn(self.W_s(x))
		ss = self.topk_scores(ss)

		# prepare destination-side
		sd = self.act_fn(self.W_d(x))
		sd = self.topk_scores(sd, True)
	
		# prepare query, key, value
		q = self.q(x) 
		x = default(context, x)
		k = self.k(x)
		
		v = torch.einsum('b t h e, b t h e d -> b h t d', ss, self.v(x))

		# compute attention scores
		attn_scores = einsum('b h i d, b h d j -> b h i j', q * self.scale, k.transpose(-1, -2))
		
		# context mask used in Cross-Attention (encoder-decoder) and Self-Attention (encoder)
		if exists(context_mask):
			context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
			attn_scores = attn_scores.masked_fill(~context_mask, -1e9)

		# causal mask used in Masked Multi-Head Attention (decoder)
		if exists(causal_mask):
			attn_scores = attn_scores.masked_fill(causal_mask, -1e9)
		attn_probs = torch.softmax(attn_scores, dim=-1)

		# Apply attention scores to V
		output = einsum('b h i j, b h j d -> b h i d', attn_probs, v)

		output = torch.einsum('b t h e , b t h e d -> b t h d', sd, self.W_o(output))

		# sum over heads
		output = output.sum(dim=-2)
		return output