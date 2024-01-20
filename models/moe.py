import torch
import torch.nn as nn
import math
from einops import rearrange, repeat, pack
from torch import einsum
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F



class MoELayer(nn.Module):
	def __init__(self, input_dim, output_dim, num_experts, sel_experts, act_fn="sigmoid"):
		super().__init__()   

		self.sel_experts = sel_experts
		self.gate =  nn.Sequential(
					nn.Linear(input_dim, num_experts,  bias=False),
				)
		
		self.experts = nn.Sequential(
					nn.Linear(input_dim, output_dim * num_experts, bias=False),
					Rearrange('b t (e d) -> b t e d', d = input_dim, e=num_experts)
				)
		

	def topk_scores(self, scores, hard=False):
		eps = scores.topk(k=self.sel_experts, dim=-1).indices
		
		mask = torch.zeros_like(scores).scatter_(-1, eps, 1)
		scores = scores * mask

		return scores
		 
	def forward (self, x):
		gating_scores = torch.softmax(self.gate(x), dim=-1)
		gating_scores = self.topk_scores(gating_scores)

		output = torch.einsum('b t e, b t e d -> b t d', gating_scores, self.experts(x))
		return output


if __name__ == '__main__':
	moe = MoELayer(input_dim=512,  output_dim=512, num_experts=6, sel_experts=3)
	x = torch.randn(2, 10, 512)  # (b, timesteps_q, dim)
	output = moe(x)

	print(output.shape) # (b, timesteps, dim)	