import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat, pack
from torch import einsum
import cv2
from positional_encoding import AbsolutePositionalEmbedding



class Vit(nn.Module):
    def __init__(self, patch_size = 64):
        super(Vit, self).__init__()
        
        self.patch_size = patch_size

        # number of features inside a patch
        self.features = patch_size * patch_size * 3

        self.class_token = nn.Parameter(torch.randn(1, self.features))
        self.pos_enc =  AbsolutePositionalEmbedding(self.features, max_len=1000)


    def forward(self, x):
        # (batch_size, channels, height, width) --> (batch_size, timesteps, features)
        x = rearrange(x, 'b c (h p1) (w p2)  -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)

        # add class token
        class_token = repeat(self.class_token, 't d -> b t d', b=x.shape[0])
        x = pack([class_token, x], "b * d")[0]
        
        # add absolute position embedding
        # x = self.pos_enc(x)
        # transformer
        return x
    
model = Vit()

img_batch = torch.randn(2, 3, 256, 256)
out = model(img_batch)
print(out.shape)

# patch = out[0][0]    
# print(torch.sum(patch))
# patch = patch.detach().numpy()
# cv2.imwrite('patch.jpg', patch)

    
    
    