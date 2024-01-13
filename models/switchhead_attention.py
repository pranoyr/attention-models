import torch
import torch.nn as nn
import math
from einops import rearrange
from torch import einsum
from einops.layers.torch import Rearrange


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


class SwitchHeadAttention(nn.Module):
	def __init__(self, dim, num_heads=8, dim_head=64, dropout=0.0):
		super(SwitchHeadAttention, self).__init__()

		self.dim = dim
		self.num_heads = num_heads
		self.dim_head = dim_head


		self.q = nn.Linear(dim, num_heads * dim_head, bias=False)

		self.q = nn.Sequential(
			nn.Linear(dim, num_heads * dim_head, bias=False),
			nn.Dropout(dropout),
			Rearrange('b t (h d) -> b h t d', h=self.num_heads)
		)

		self.kv = nn.Sequential(
			nn.Linear(dim, 2 * num_heads * dim_head, bias=False),
			nn.Dropout(dropout),
			Rearrange('b t (kv h d) -> kv b h t d', d = self.dim_head, h = self.num_heads)
		)

	
		output_dim =  dim_head * (dim / (dim_head * num_heads))
		self.W_o = nn.ModuleList([nn.Linear(dim_head, int(output_dim)) for _ in range(num_heads)])

		self.dropout = nn.Dropout(dropout)

		self.scale = dim_head ** -0.5

	def forward(self, x, context=None, causal_mask=None, context_mask=None):

		# prepare Q, K, V for attention

		q = self.q(x)

		if exists(context):
			k, v = self.kv(context)
		else:
			k, v = self.kv(x)
		
		# compute attention scores
		# Attention Scores = Q * K^T / sqrt(d_k)
		#  Q(b, h, t, d) * K(b, h, d, t) ->  (b, h, t, t)
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
		# (b, h, t, t) * V(b, h, t, d) -> (b, h, t, d)
		output = einsum('b h i j, b h j d -> h b i d', attn_probs, v)

		output = [w(head) for w, head in zip(self.W_o, output)]

		# combine heads
		output = rearrange(output, 'h b t d -> b t (h d)')

		return output