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



# helper function
def exists(val):
	return val is not None


class AgentAttention(nn.Module):
	def __init__(self, dim, num_heads=8, dim_head=64, agent_num=47, dropout=0.0):
		super(AgentAttention, self).__init__()

		self.dim = dim
		self.num_heads = num_heads
		self.dim_head = dim_head
  
		inner_dimension = num_heads * dim_head

		self.qkv = nn.Linear(dim, 3 * inner_dimension , bias=False)
		self.scale = dim_head ** -0.5

		pool_size = int(agent_num ** 0.5)
		self.pool = nn.AdaptiveAvgPool2d((pool_size, dim_head))
  
		self.W_o = nn.Linear(num_heads * dim_head, dim)
		self.dropout = nn.Dropout(dropout)


		self.dwc = nn.Sequential(Rearrange('b h t d -> b d h t'), 
						   		nn.Conv2d(dim_head, dim_head, kernel_size=3, padding=1, groups=dim_head),
						  	 	Rearrange('b d h t -> b h t d'))

	def forward(self, x, context_mask=None):
		
		qkv = self.qkv(x)

		qkv = rearrange(qkv, 'b t (qkv h d) -> qkv b h t d', d = self.dim_head, h = self.num_heads)
		q , k , v = qkv[0], qkv[1], qkv[2]
		
		# pooling the queries to get agent tokens
		agent_tokens = self.pool(q)
  
		# Agent Aggregation

		attn_scores = einsum('b h i d, b h j d -> b h i j', agent_tokens * self.scale, k)
   
		attn_probs = torch.softmax(attn_scores, dim=-1)
   
		v_agent = einsum('b h i j, b h j d -> b h i d', attn_probs, v)
  

		# Agent Broadcast

		attn_scores = einsum('b h i d, b h j d -> b h i j', q * self.scale, agent_tokens)
  
		attn_probs = torch.softmax(attn_scores, dim=-1)

		output = einsum('b h i j, b h j d -> b h i d', attn_probs, v_agent) + self.dwc(v)
	
		# combine 
		output = rearrange(output, 'b h t d -> b t (h d)')
		output = self.W_o(output)
		output = self.dropout(output)
   
		return output
	