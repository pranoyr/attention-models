import torch
import torch.nn as nn
import math
from einops import rearrange, reduce, repeat
from torch import einsum
from einops.layers.torch import Rearrange


# helper function
def exists(val):
	return val is not None




class SwitchHeadAttention(nn.Module):
	def __init__(self, dim, num_heads=8, dim_head=64, num_experts=5, dropout=0.0):
		super(SwitchHeadAttention, self).__init__()

		self.dim = dim
		self.num_heads = num_heads
		self.dim_head = dim_head
		self.num_experts = num_experts


		self.q = nn.Sequential(
			nn.Linear(dim, num_heads * num_experts * dim_head, bias=False),
			nn.Dropout(dropout),
			Rearrange('b t (h e d) -> b t h e d', h=self.num_heads, e=self.num_experts)
		)

		self.kv = nn.Sequential(
			nn.Linear(dim, 2 * num_heads * num_experts * dim_head, bias=False),
			nn.Dropout(dropout),
			Rearrange('b t (kv h e d) -> kv b t h e d', d = self.dim_head, h = self.num_heads, e=self.num_experts)
		)


		self.W_s = nn.Linear(dim, num_heads * num_experts, bias=False)
		self.W_d = nn.Linear(dim, num_heads * num_experts, bias=False)
	
		self.W_o = nn.ModuleList([nn.Linear(dim_head, dim) for _ in range(num_heads)])

		self.dropout = nn.Dropout(dropout)

		self.scale = dim_head ** -0.5

	def get_scores(self, eps, s):
		scores = torch.zeros_like(s)
		scores.scatter_(3, eps, s)
		scores = repeat(scores, 'b t h e -> b t h e d', d = self.dim_head)
		return scores


	def forward(self, x, context=None, causal_mask=None, context_mask=None):

		ss = torch.sigmoid(self.W_s(x))
		sd = torch.sigmoid(self.W_d(x))

		ss = rearrange(ss, 'b t (h e) -> b t h e', h=self.num_heads)
		sd = rearrange(sd, 'b t (h e) -> b t h e', h=self.num_heads)
		
		eps_s = ss.topk(k=3, dim=3).indices
		eps_d = sd.topk(k=3, dim=3).indices

		scores_q = self.get_scores(eps_d , sd)
		scores_k = self.get_scores(eps_s , ss)

	
		q = self.q(x) * scores_q  # b, h, t, 
		q = q.sum(dim=-2)
		q = rearrange(q, 'b t h d -> b h t d')


		if exists(context):
			k, v = self.kv(context)
		else:
			k, v = self.kv(x)

		k = k * scores_k
		k = k.sum(dim=-2)
		k = rearrange(k, 'b t h d -> b h t d')

		v = v * scores_k
		v = v.sum(dim=-2)
		v = rearrange(v, 'b t h d -> b h t d')

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
		output = rearrange(output, 'b h t d -> h b t d')


		output = [w(o_head)  for w, o_head   in zip(self.W_o, output)]

		output = torch.stack(output, dim=2).sum(dim=2)
	
		return output
	