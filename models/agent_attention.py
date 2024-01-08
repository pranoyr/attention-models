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

class AtentAttention(nn.Module):
	def __init__(self, dim, num_heads=8, dim_head=64, dropout=0.0):
		super(AtentAttention, self).__init__()

		self.dim = dim
		self.num_heads = num_heads
		self.dim_head = dim_head

		self.qkv = nn.Linear(dim, dim * 3, bias=False)
		self.scale = dim_head ** -0.5

	def forward(self, x, causal_mask=None, context_mask=None):
		
		qkv = self.qkv(x)
		print(qkv.shape)

		qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

		# prepare Q, K, V for attention
		# Q,K,V - (b_size, n_timesteps, n_heads * dk)
		# q = self.W_q(q)
		# k = self.W_k(k)
		# v = self.W_v(v)

		# # split Q, K, V into multiple heads
		# # (b, t, h * d) -> (b, h, t, d)
		# q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
		# k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
		# v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)

		# # compute attention scores
		# # Attention Scores = Q * K^T / sqrt(d_k)
		# #  Q(b, h, t, d) * K(b, h, d, t) ->  (b, h, t, t)
		# k_transpose = rearrange(k, 'b h t d -> b h d t')
		# attn_scores = einsum('b h i d, b h d j -> b h i j', q * self.scale, k_transpose)
		
		# # context mask used in Cross-Attention (encoder-decoder) and Self-Attention (encoder)
		# if exists(context_mask):
		# 	context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
		# 	attn_scores = attn_scores.masked_fill(~context_mask, -1e9)

		# # causal mask used in Masked Multi-Head Attention (decoder)
		# if exists(causal_mask):
		# 	attn_scores = attn_scores.masked_fill(causal_mask, -1e9)
		# attn_probs = torch.softmax(attn_scores, dim=-1)

		# # Apply attention scores to V
		# # (b, h, t, t) * V(b, h, t, d) -> (b, h, t, d)
		# output = einsum('b h i j, b h j d -> b h i d', attn_probs, v)

		# # combine heads
		# output = rearrange(output, 'b h t d -> b t (h d)')
		# output = self.W_o(output)
		# output = self.dropout(output)
		# return output
	
# b , time, dim 
x = torch.randn(1, 16, 256)

attn = AtentAttention(256)

out = attn(x)