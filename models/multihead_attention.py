import torch
import torch.nn as nn
import math
from einops import rearrange
from torch import einsum


# h - number of heads (num_heads)
# d - dimension of each head (dim_head)
# b - batch size

# t - number of timesteps of Q,K,V
# When Q,K have different lengths
# i - number of timesteps for Q
# j - number of timesteps for K

# helper function
def exists(val):
	return val is not None

class MultiHeadAttention(nn.Module):
	def __init__(self, dim, num_heads, dim_head):
		super(MultiHeadAttention, self).__init__()

		self.dim = dim
		self.num_heads = num_heads
		self.dim_head = dim_head

		self.W_q = nn.Linear(dim, num_heads * dim_head)
		self.W_k = nn.Linear(dim, num_heads * dim_head)
		self.W_v = nn.Linear(dim, num_heads * dim_head)
		self.W_o = nn.Linear(num_heads * dim_head, dim)

		self.scale = dim_head ** -0.5

	def forward(self, q, k, v, causal_mask=None, context_mask=None):

		# prepare Q, K, V for attention
		# Q,K,V - (b_size, n_timesteps, n_heads * dk)
		q = self.W_q(q)
		k = self.W_k(k)
		v = self.W_v(v)

		# split Q, K, V into multiple heads
		# (b, t, h * d) -> (b, h, t, d)
		q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
		k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
		v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)

		# compute attention scores
		# Attention Scores = Q * K^T / sqrt(d_k)
		#  Q(b, h, t, d) * K(b, h, d, t) ->  (b, h, t, t)
		k_transpose = rearrange(k, 'b h t d -> b h d t')
		attn_scores = einsum('b h i d, b h d j -> b h i j', q * self.scale, k_transpose)
		
		# context mask used in Cross-Attention (encoder-decoder) and Self-Attention (encoder)
		if exists(context_mask):
			context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
			attn_scores = attn_scores.masked_fill(~context_mask, -1e9)

		# causal mask used in Masked Multi-Head Attention (decoder)
		if exists(causal_mask):
			attn_scores = attn_scores.masked_fill(causal_mask, -1e9)
		attn_probs = torch.softmax(attn_scores, dim=-1)

		# Apply attention scores to V
		# (b, h, t, t) * V(b, h, t, d) -> (b, h, t, d)
		output = einsum('b h i j, b h j d -> b h i d', attn_probs, v)

		# combine heads
		output = rearrange(output, 'b h t d -> b t (h d)')
		output = self.W_o(output)
		return output