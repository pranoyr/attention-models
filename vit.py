import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat, pack
from torch import einsum
import cv2
from positional_encoding import AbsolutePositionalEmbedding
from transformer import EncoderLayer, _get_clones


class Encoder(nn.Module):
    def __init__(self, dim, n_heads, d_head, depth):
        super().__init__()

        encoder_layer = EncoderLayer(dim, n_heads, d_head)

        self.layers = _get_clones(encoder_layer, depth)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return x



class Vit(nn.Module):
    def __init__(self, dim, patch_size = 64, n_heads = 8, d_head = 64, depth = 6, num_classes = None):
        super(Vit, self).__init__()
        
        self.dim = dim
        self.patch_size = patch_size

        # number of features inside a patch
        features = patch_size * patch_size * 3

        self.fc1 = nn.Linear(features, dim)
        self.final_fc = nn.Linear(dim, num_classes)

        self.class_token = nn.Parameter(torch.randn(1, dim))
        self.pos_enc =  nn.Parameter(torch.randn(1, dim))

        self.encoder = Encoder(dim, n_heads, d_head, depth)
        


    def forward(self, x):
        # (batch_size, channels, height, width) --> (batch_size, timesteps, features)
        x = rearrange(x, 'b c (h p1) (w p2)  -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        x = self.fc1(x)

        # add class token
        class_token = repeat(self.class_token, 't d -> b t d', b=x.shape[0])
        x, _ = pack([class_token, x], "b * d")

        # add positional encoding
        x = x + self.pos_enc

        # transformer encoder
        x = self.encoder(x)

        # get the class token output
        x = x[:, 0]
        x = self.final_fc(x)

        return x
    
model = Vit(512, num_classes=10)

img_batch = torch.randn(2, 3, 256, 256)
out = model(img_batch)
print(out.shape)

# patch = out[0][0]    
# print(torch.sum(patch))
# patch = patch.detach().numpy()
# cv2.imwrite('patch.jpg', patch)

    
    
    