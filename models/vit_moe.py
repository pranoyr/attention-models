import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat, pack
from einops.layers.torch import Rearrange
from torch import einsum
from models import SwitchHeadAttention
from torch.nn import functional as F
from models import MoELayer

class Encoder(nn.Module):
	def __init__(self, dim, n_heads, d_head, depth, n_experts, sel_experts, dropout):
		super().__init__()
	
		self.layers = nn.ModuleList([EncoderLayer(dim, n_heads, d_head, n_experts, sel_experts, dropout) for _ in range(depth)])
 
	def forward(self, x, context_mask=None):
		for layer in self.layers:
			x = layer(x, context_mask=context_mask)
		return x


class EncoderLayer(nn.Module):
	def __init__(self, dim, n_heads, d_head, n_experts, sel_experts, dropout):
		super().__init__()

		self.self_attn = SwitchHeadAttention(dim, n_heads, d_head, num_experts=n_experts, sel_experts=sel_experts, dropout=dropout)
		self.moe = MoELayer(input_dim=dim,  output_dim=dim, num_experts=n_experts, sel_experts=sel_experts)
		self.norm1 = nn.LayerNorm(dim)
		self.norm2 = nn.LayerNorm(dim)
		
	def forward(self, x, context_mask=None):
		x_norm = self.norm1(x)
		# self attention
		attn_out = self.self_attn(x=x_norm, context_mask=context_mask)

		# ADD & NORM
		x = attn_out + x
		x_norm = self.norm2(x)

		# feed forward
		fc_out = self.moe(x_norm)

		# ADD
		x = fc_out + x
		return x



class ViTMoE(nn.Module):
	def __init__(self, 
				dim=1024,
				image_size=256, 
				patch_size = 32,
				n_heads = 16,
				d_head = 64,
				depth = 6,
				n_experts=32,
				sel_experts=2, 
				dropout=0.0, 
				num_classes = 1000):
		
		super(ViTMoE, self).__init__()
		
		self.dim = dim
		self.patch_size = patch_size
		
		# number of features inside a patch
		self.patch_dim = patch_size * patch_size * 3
		
		self.to_patch_embedding = nn.Sequential(
			Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
			nn.LayerNorm(self.patch_dim),
			nn.Linear(self.patch_dim, dim),
			nn.LayerNorm(dim))


		self.class_token = nn.Parameter(torch.randn(1, 1, dim))
			
		num_patches = (image_size // patch_size) ** 2  
		self.pos_enc =  nn.Parameter(torch.randn(1, num_patches + 1, dim)) # 1 extra for class token
		
		self.encoder = Encoder(dim, n_heads, d_head, depth, n_experts, sel_experts, dropout)
  
		self.norm = nn.LayerNorm(dim)
		
		self.class_embed = nn.Linear(dim, num_classes)

		
	def forward(self, x):
		x = self.to_patch_embedding(x)
		
		# add class token
		class_token = repeat(self.class_token, '1 1 d -> b 1 d', b=x.shape[0])
		x, _ = pack([class_token, x], "b * d")

		# add positional encoding
		x += self.pos_enc
		# transformer encoder
		x = self.encoder(x)
  
		x =  self.norm(x)

		x = x[:, 0, :]

		x = self.class_embed(x)
  
		return x