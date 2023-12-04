import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, pack
from einops.layers.torch import Rearrange
from models.transformer import Encoder as TransformerBlock

class ViTEncoder(nn.Module):
	def __init__(self, dim, img_size, patch_size, n_heads, d_head, depth):
		super().__init__()

		self.dim = dim  # model dimension
		self.patch_size = patch_size
		self.img_size = img_size

		# number of features inside a patch
		patch_dim = patch_size * patch_size * 3
		num_patches = (img_size // patch_size) ** 2
		
		self.to_patch_embedding = nn.Sequential(
			Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
			nn.LayerNorm(patch_dim),
			nn.Linear(patch_dim, dim),
			nn.LayerNorm(dim))

		self.pos_enc =  nn.Parameter(torch.randn(1, num_patches, dim)) 
		self.encoder = TransformerBlock(dim, n_heads, d_head, depth)
  
		self.final_norm = nn.LayerNorm(dim)


	def forward(self, x):
		# to patches
		x = self.to_patch_embedding(x)
		# add positional encoding
		x = self.pos_enc + x
		# encoder 
		x = self.encoder(x)
		x = self.final_norm(x)
		return x
	
	
	
class ViTDecoder(nn.Module):
	def __init__(self, dim, img_size, patch_size, n_heads, d_head, depth):
		super().__init__()
		self.patch_size = patch_size
		self.img_size = img_size
  
  		# number of features inside a patch
		patch_dim = patch_size * patch_size * 3
		num_patches = (img_size // patch_size) ** 2

		self.pos_enc =  nn.Parameter(torch.randn(1, num_patches, dim)) 
		self.decoder = TransformerBlock(dim, n_heads, d_head, depth)
		self.norm = nn.LayerNorm(dim)
		self.fc = nn.Linear(dim, patch_dim)


	def forward(self, x):
		x = x + self.pos_enc
		x = self.decoder(x)
		x = self.norm(x)
		x = self.fc(x)   # project to original patch dim
	
		x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size,
					  h = self.img_size // self.patch_size)
		return x
	


class Codebook(nn.Module):
	def __init__(self, codebook_size=8192, codebook_dim=32, beta=0.25):
		super(Codebook, self).__init__()
		self.codebook_size = codebook_size
		self.codebook_dim = codebook_dim
		self.beta = beta

		self.embedding = nn.Embedding(self.codebook_size, self.codebook_dim)
		self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

	def forward(self, z):

		# for computing the difference between z and embeddings
		z_flattened = rearrange(z, 'b t d -> (b t) d')

		# D - distance between z and embeddings
		d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
			torch.sum(self.embedding.weight**2, dim=1) - \
			2*(torch.matmul(z_flattened, self.embedding.weight.t()))

		min_encoding_indices = torch.argmin(d, dim=1)
		
		z_q = self.embedding(min_encoding_indices) 

		b , t , d = z.shape
		z_q = rearrange(z_q, '(b t) d -> b t d', t=t, d=d)

		loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

		z_q = z + (z_q - z).detach()

		return z_q, min_encoding_indices, loss


	

class ViTVQGAN(nn.Module):
	def __init__(self, vit_params, codebook_params):
		super(ViTVQGAN, self).__init__()
  
		self.encoder = ViTEncoder(**vit_params)
		self.pre_quant = nn.Linear(vit_params['dim'], codebook_params['codebook_dim'])
		self.codebook = Codebook(**codebook_params)
		self.post_quant = nn.Linear(codebook_params['codebook_dim'], vit_params['dim'])
		self.decoder = ViTDecoder(**vit_params)

	def forward(self, imgs):
	    enc_imgs = self.encoder(imgs)
	    enc_imgs = self.pre_quant(enc_imgs)
	    embeds, indices, loss = self.codebook(enc_imgs)
	    embeds = self.post_quant(embeds)
	    out = self.decoder(embeds)
	    return out, loss
	

if __name__ == '__main__':
		
	ViT_params = dict(
	dim =  256,   	
	img_size= 256,	 
	patch_size=8,
	n_heads= 8,
	d_head= 64,
	depth= 6,
	)
 
	codebook_params = dict(
	codebook_size = 8192,
	codebook_dim = 32,
	beta = 0.25
	)

	imgs = torch.randn(2, 3, 256, 256)
	vitvqgan = ViTVQGAN(ViT_params, codebook_params)
	out, loss = vitvqgan(imgs)
	print(loss)