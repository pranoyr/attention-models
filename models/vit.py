import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat, pack
from einops.layers.torch import Rearrange
from torch import einsum
from models.transformer import Encoder


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ViT(nn.Module):
    def __init__(self, dim, image_size=256, patch_size = 16, n_heads = 12, d_head = 64, depth = 12, mlp_dim=3072, dropout=0.0, num_classes = None):
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

        self.final_fc = nn.Linear(dim, num_classes)

        self.class_token = nn.Parameter(torch.randn(dim))
        num_patches = (image_size // patch_size) ** 2
        self.pos_enc =  nn.Parameter(torch.randn(1, num_patches + 1, dim)) # 1 extra for class token

        self.encoder = Encoder(dim, n_heads, d_head, depth, dropout)
        self.encoder.feed_forward = FeedForward(dim, mlp_dim)
        


    def forward(self, x):
        # (batch_size, channels, height, width) --> (batch_size, timesteps, features)
        x = self.to_patch_embedding(x)

        # add class token
        class_token = repeat(self.class_token, 'd -> b 1 d', b=x.shape[0])
        x, _ = pack([class_token, x], "b * d")

        # add positional encoding
        x += self.pos_enc

        # transformer encoder
        x = self.encoder(x)

        # get the class token output
        x = x[:, 0]
        x = self.final_fc(x)

        return x

    