import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat, pack
from torch import einsum
from models.transformer import Encoder



class ViT(nn.Module):
    def __init__(self, dim, patch_size = 64, n_heads = 8, d_head = 64, depth = 6, num_classes = None):
        super(ViT, self).__init__()
        
        self.dim = dim
        self.patch_size = patch_size

        # number of features inside a patch
        self.patch_dim = patch_size * patch_size * 3

        self.fc1 = nn.Linear(self.patch_dim, dim)
        self.final_fc = nn.Linear(dim, num_classes)

        self.class_token = nn.Parameter(torch.randn(dim))
        self.pos_enc =  nn.Parameter(torch.randn(1, 1, dim))

        self.encoder = Encoder(dim, n_heads, d_head, depth)
        


    def forward(self, x):
        # (batch_size, channels, height, width) --> (batch_size, timesteps, features)
        x = rearrange(x, 'b c (h p1) (w p2)  -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        x = nn.LayerNorm(self.patch_dim)(x)
        x = self.fc1(x) # to dim
        x = nn.LayerNorm(self.dim)(x)

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
    
if __name__ == '__main__':
    model = ViT(512, num_classes=10)

    img_batch = torch.randn(2, 3, 256, 256)
    out = model(img_batch)
    print(out.shape)


    