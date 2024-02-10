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
	def __init__(self, input_dim, output_dim, num_experts, sel_experts):
		super().__init__()
  
		self.sel_experts = sel_experts
		self.gate = nn.Linear(input_dim, num_experts)
		
		self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
	
	def forward (self, inputs):
		b , t , d = inputs.shape

		gate_logits = self.gate(inputs) 
		weights, selected_experts = torch.topk(gate_logits, self.sel_experts)
		weights = torch.sigmoid(weights).to(inputs.dtype)

		# results should of shape - (b, t, d)
		results = torch.zeros(b, t, d, device=inputs.device)
		for i, expert in enumerate(self.experts):
			batch_idx, t, nth_expert = torch.where(selected_experts == i)
			results[batch_idx, t] += weights[batch_idx, t, nth_expert, None] * expert(
                inputs[batch_idx, t]
            )
   
		return results

if __name__ == '__main__':
	moe = MoELayer(input_dim=512,  output_dim=512, num_experts=6, sel_experts=2)
	x = torch.randn(2, 10, 512)  # (b, t, d)
	output = moe(x)

	print(output.shape) # (b, timesteps, dim)	