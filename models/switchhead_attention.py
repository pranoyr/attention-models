import torch
import torch.nn as nn
import math
from einops import rearrange, repeat, pack
from torch import einsum
from einops.layers.torch import Rearrange
from torch.nn import functional as F


# helper functions
def exists(val):
	return val is not None

def default(val, d):
	return val if exists(val) else d


class SwitchHeadAttention(nn.Module):
	def __init__(self, dim, num_heads=8, dim_head=64, num_experts=5, sel_experts=2, dropout=0.0):
		super(SwitchHeadAttention, self).__init__()

		self.dim = dim
		self.num_heads = num_heads
		self.dim_head = dim_head
		self.num_experts = num_experts
		self.sel_experts = sel_experts
  

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

		self.W_s = nn.Sequential(
			nn.Linear(dim, num_heads * num_experts, bias=False),
			Rearrange('b t (h e) -> b t h e', h=self.num_heads, e=self.num_experts)
		)

		self.experts_v = nn.ModuleList([nn.Linear(dim, dim_head, bias=False) for _ in range(num_experts)])

		self.W_d =  nn.Sequential(
			nn.Linear(dim, num_heads * num_experts, bias=False),
			Rearrange('b t (h e) -> b t h e', h=self.num_heads, e=self.num_experts)
		)
	
		self.experts_out = nn.ModuleList([nn.Linear(dim_head, dim, bias=False) for _ in range(num_experts)])

		self.scale = dim_head ** -0.5
		self.inf = -1e9

	def moe_v(self, inputs):
		# inputs shape - (b, t, d)
		b , t = inputs.shape[:2]

		gate_logits = self.W_s(inputs) 
		weights, selected_experts = torch.topk(gate_logits, self.sel_experts)
		weights = torch.sigmoid(weights).to(inputs.dtype)

		results = torch.zeros(b, t, self.num_heads, self.dim_head, device=inputs.device)
		for i, expert in enumerate(self.experts_v):
			batch_idx, t, h, nth_expert = torch.where(selected_experts == i)
			results[batch_idx, t, h] += weights[batch_idx, t, h, nth_expert, None] * expert(
				inputs[batch_idx, t]
			)
		results = rearrange(results, 'b t h d -> b h t d')
		return results
	
	def moe_out(self, inputs, gate_inputs):
		# inputs shape - (b, t, h, d)
		# gate_inputs shape - (b, t, d)
		b , t = gate_inputs.shape[:2]

		gate_logits = self.W_d(gate_inputs) 
		weights, selected_experts = torch.topk(gate_logits, self.sel_experts)

		results = torch.zeros(b, t, self.num_heads, self.dim, device=inputs.device)
		for i, expert in enumerate(self.experts_out):
			batch_idx, t, h, nth_expert = torch.where(selected_experts == i)
			results[batch_idx, t, h] += expert(inputs[batch_idx, t, h]
			)
		return results
	
	def forward(self, x, context=None, causal_mask=None, context_mask=None):
		# prepare query, key, value
		q = self.q(x) 
		x = default(context, x)
		k = self.k(x)
		v = self.moe_v(x)
	
		# compute attention scores
		attn_scores = einsum('b h i d, b h d j -> b h i j', q * self.scale, k.transpose(-1, -2))
		
		# context mask used in Cross-Attention (encoder-decoder) and Self-Attention (encoder)
		if exists(context_mask):
			context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
			attn_scores = attn_scores.masked_fill(~context_mask, self.inf)

		# causal mask used in Masked Multi-Head Attention (decoder)
		if exists(causal_mask):
			attn_scores = attn_scores.masked_fill(causal_mask, self.inf)
		attn_probs = torch.softmax(attn_scores, dim=-1)

		# Apply attention scores to V
		output = einsum('b h i j, b h j d -> b i h d', attn_probs, v)
		output = self.moe_out(output, gate_inputs=x)

		# sum over heads
		output = output.sum(dim=-2)
		return output