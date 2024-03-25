import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, pack
from einops.layers.torch import Rearrange
from models.softmax_attention import SoftmaxAttention
from typing import Optional
from xformers.ops import SwiGLU



def l2_norm(x):
	return F.normalize(x, p=2, dim=-1)


class FeedForward(SwiGLU):
	def __init__(
		self,
		in_features: int,
		hidden_features: Optional[int] = None,
		bias: bool = True,
	) -> None:
		out_features = in_features
		hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
		super().__init__(
			in_features=in_features,
			hidden_features=hidden_features,
			out_features=out_features,
			bias=bias,
		)



class EncoderLayer(nn.Module):
	def __init__(self, dim, n_heads, d_head, mlp_dim, dropout):
		super().__init__()

		self.self_attn = SoftmaxAttention(dim, n_heads, d_head, dropout)
		self.feed_forward = FeedForward(dim, mlp_dim)
		self.norm1 = nn.LayerNorm(dim)
		self.norm2 = nn.LayerNorm(dim)
		
	def forward(self, x):
		x_norm = self.norm1(x)
		# self attention
		attn_out = self.self_attn(x=x_norm)

		# ADD & NORM
		x = attn_out + x
		x_norm = self.norm2(x)

		# feed forward
		fc_out = self.feed_forward(x_norm)

		# ADD
		x = fc_out + x
		return x


class TransformerBlock(nn.Module):
	def __init__(self, dim, n_heads, d_head, depth, mlp_dim, dropout=0.):
		super().__init__()
  
		self.layers = nn.ModuleList([EncoderLayer(dim, n_heads, d_head, mlp_dim, dropout) for _ in range(depth)])
 
	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x



class ViTEncoder(nn.Module):
	def __init__(self, dim, img_size, patch_size, n_heads, d_head, depth, mlp_dim, dropout):
		super().__init__()

		self.dim = dim  # model dimension
		self.patch_size = patch_size
		self.img_size = img_size

		# number of features inside a patch
		patch_dim = patch_size * patch_size * 3
		num_patches = (img_size // patch_size) ** 2

		self.to_patch_embedding = nn.Sequential(
			Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_size, p2=self.patch_size),
			nn.LayerNorm(patch_dim),
			nn.Linear(patch_dim, dim),
			nn.LayerNorm(dim)
		)

		self.pos_enc = nn.Parameter(torch.randn(1, num_patches, dim))
		self.pre_norm = nn.LayerNorm(dim)
		self.encoder = TransformerBlock(dim, n_heads, d_head, depth, mlp_dim, dropout)

	def forward(self, x):
		# to patches
		x = self.to_patch_embedding(x)
		# add positional encoding
		x = self.pos_enc + x
		# encoder
		x = self.pre_norm(x)
		x = self.encoder(x)
		return x


class ViTDecoder(nn.Module):
	def __init__(self, dim, img_size, patch_size, n_heads, d_head, depth, mlp_dim, dropout):
		super().__init__()
		self.patch_size = patch_size
		self.img_size = img_size

		# number of features inside a patch
		patch_dim = patch_size * patch_size * 3
		num_patches = (img_size // patch_size) ** 2

		self.pos_enc = nn.Parameter(torch.randn(1, num_patches, dim))
		self.pre_norm = nn.LayerNorm(dim)
		self.decoder = TransformerBlock(dim, n_heads, d_head, depth, mlp_dim, dropout)
		
		self.fc = nn.Linear(dim, patch_dim)

	def forward(self, x):
		x = x + self.pos_enc

		x = self.pre_norm(x)
		x = self.decoder(x)
		x = self.fc(x)  # project to original patch dim

		# inverse patches to image
		x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", 
					  p1=self.patch_size, p2=self.patch_size, h=self.img_size // self.patch_size)
		return x


class Codebook(nn.Module):
	def __init__(self, codebook_size=8192, codebook_dim=32, beta=0.25):
		super(Codebook, self).__init__()
		self.codebook_size = codebook_size
		self.codebook_dim = codebook_dim
		self.beta = beta

		self.embedding = nn.Embedding(self.codebook_size, self.codebook_dim)
		# self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)
		self.embedding.weight.data.normal_()
	
	def forward(self, z):
		z = l2_norm(z)
		z_flattened = z.view(-1, self.codebook_dim)
		embedd_norm = l2_norm(self.embedding.weight)
		# distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

		d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
			torch.sum(embedd_norm**2, dim=1) - 2 * \
			torch.einsum('bd,nd->bn', z_flattened, embedd_norm)

		encoding_indices = torch.argmin(d, dim=1).view(*z.shape[:-1])
		z_q = self.embedding(encoding_indices)
		z_q = l2_norm(z_q)

		# compute loss for embedding
		loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q-z.detach())**2)

		# preserve gradients
		z_q = z + (z_q - z).detach()

		return z_q, encoding_indices, loss

	def indices_to_embeddings(self, indices):
		embeds = self.embedding(indices)
		embeds = l2_norm(embeds)
		return embeds



class ViTVQGAN(nn.Module):
	def __init__(self, vit_params, codebook_params):
		super(ViTVQGAN, self).__init__()

		self.encoder = ViTEncoder(**vit_params)
		self.pre_quant = nn.Linear(vit_params["dim"], codebook_params["codebook_dim"])
		self.codebook = Codebook(**codebook_params)
		self.post_quant = nn.Linear(codebook_params["codebook_dim"], vit_params["dim"])
		self.decoder = ViTDecoder(**vit_params)

	def forward(self, imgs):
		enc_imgs = self.encoder(imgs)
		enc_imgs = self.pre_quant(enc_imgs)
		embeds, indices, loss = self.codebook(enc_imgs)
		embeds = self.post_quant(embeds)
		out = self.decoder(embeds)
		return out, loss
	
	def decode_indices(self, indices):
		embeds = self.codebook.indices_to_embeddings(indices)
		embeds = self.post_quant(embeds)
		imgs = self.decoder(embeds)
		return  imgs
	
	def encode_imgs(self, imgs):
		b = imgs.shape[0]
		enc_imgs = self.encoder(imgs)
		enc_imgs = self.pre_quant(enc_imgs)
		_, indices, _ = self.codebook(enc_imgs)
		# indices = rearrange(indices, '(b i) -> b i', b=b)
		return indices

	@property
	def num_patches(self):
		num_patches = (self.encoder.img_size // self.encoder.patch_size)  ** 2
		return num_patches