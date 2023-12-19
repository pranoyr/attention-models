import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat, pack
from einops.layers.torch import Rearrange
from torch import einsum
from models.transformer import Encoder



class ViT(nn.Module):
    def __init__(self, dim, image_size=256, patch_size = 64, n_heads = 8, d_head = 64, depth = 6, num_classes = None):
        super(ViT, self).__init__()
        
        self.dim = dim
        self.patch_size = patch_size
        
        # number of features inside a patch
        self.patch_dim = patch_size * patch_size * 3
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            nn.LayerNorm(dim))

        
        self.pre_norm = nn.LayerNorm(dim)
        self.final_fc = nn.Linear(dim, num_classes)

        self.class_token = nn.Parameter(torch.randn(dim))
        num_patches = (image_size // patch_size) ** 2
        self.pos_enc =  nn.Parameter(torch.randn(1, num_patches + 1, dim)) # 1 extra for class token

        self.encoder = Encoder(dim, n_heads, d_head, depth)
        


    def forward(self, x):
        # (batch_size, channels, height, width) --> (batch_size, timesteps, features)
        x = self.to_patch_embedding(x)

        # add class token
        class_token = repeat(self.class_token, 'd -> b 1 d', b=x.shape[0])
        x, _ = pack([class_token, x], "b * d")

        # add positional encoding
        x += self.pos_enc
        
        # pre norm
        x = self.pre_norm(x)

        # transformer encoder
        x = self.encoder(x)

        # get the class token output
        x = x[:, 0]
        x = self.final_fc(x)

        return x

    