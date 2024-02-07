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

		self.experts_v = nn.ModuleList([nn.Linear(dim, dim_head, bias=False) for _ in range(num_experts)])

		self.W_s = nn.Sequential(
			nn.Linear(dim, num_heads * num_experts, bias=False),
			Rearrange('b t (h e) -> b t h e', h=self.num_heads, e=self.num_experts)
		)

		self.W_d =  nn.Sequential(
			nn.Linear(dim, num_heads * num_experts, bias=False),
			Rearrange('b t (h e) -> b t h e', h=self.num_heads, e=self.num_experts)
		)
	
		self.experts_o = nn.ModuleList([nn.Linear(dim_head, dim, bias=False) for _ in range(num_experts)])

		self.scale = dim_head ** -0.5


	def compute_moe(self, inputs, gate, experts):
		# input shape - (b, t, d)
		gate_logits = gate(inputs) 
		weights, selected_experts = torch.topk(gate_logits, self.sel_experts)
		weights = torch.sigmoid(weights).to(inputs.dtype)

		# results should of shape - (b, t  h, d)
		results = torch.zeros(inputs.shape[0], inputs.shape[1], self.num_heads, self.dim_head)
		for i, expert in enumerate(experts):
			batch_idx, t, h, nth_expert = torch.where(selected_experts == i)
			results[batch_idx, t, h] += weights[batch_idx, t, h, nth_expert, None] * expert(
				inputs[batch_idx, t]
			)
		results = rearrange(results, 'b t h d -> b h t d')
		return results
	

	def compute_moe_o(self, inputs, gate_in, gate, experts):
		# gate int - (b, t, d)
		# input shape - (b, t, h, d)

		gate_logits = gate(gate_in) 
		weights, selected_experts = torch.topk(gate_logits, self.sel_experts)
		weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)

		# results should of shape - (b, t  h, d)
		results = torch.zeros(inputs.shape[0], inputs.shape[1], self.num_heads, self.dim)
		for i, expert in enumerate(experts):
			batch_idx, t, h, nth_expert = torch.where(selected_experts == i)
			results[batch_idx, t, h] += expert(inputs[batch_idx, t, h]
			)
		return results
	
	
	def forward(self, x, context=None, causal_mask=None, context_mask=None):

	
		# prepare query, key, value
		q = self.q(x) 
		x = default(context, x)
		k = self.k(x)
		
		
		v = self.compute_moe(x, gate=self.W_s, experts=self.experts_v)
	
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
		output = einsum('b h i j, b h j d -> b i h d', attn_probs, v)

		# output = torch.einsum('b t h e , b t h e d -> b t h d', sd, self.W_o(output))
		output = self.compute_moe_o(output, x, gate=self.W_d, experts=self.experts_o)

		# sum over heads
		output = output.sum(dim=-2)
		return output
	