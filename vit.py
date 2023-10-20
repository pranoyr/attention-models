import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from torch import einsum
import cv2


img_batch = torch.randn(2, 3, 256, 256)

img = cv2.imread('/home/pranoy/Downloads/foundation.jpg')
img = cv2.resize(img, (256, 256))
img = img.transpose(2, 0, 1)
img = torch.from_numpy(img).unsqueeze(0).float()
# (batch_size, channels, height, width) --> (batch_size, timesteps, features)


def pair (t):
    return t if isinstance(t, tuple) else (t, t)


class Vit(nn.Module):
    def __init__(self, patch_size = 64):
        super(Vit, self).__init__()
        self.patch_size = pair(patch_size)

    def forward(self, x):
        x = rearrange(x, 'b c (h p1) (w p2)  -> b (h w) p1 p2 c', h=64, w=64)
        return x
    
# model = Vit()
# out = model(img)
# print(out.shape)

# patch = out[0][0]    
# print(torch.sum(patch))
# patch = patch.detach().numpy()
# cv2.imwrite('patch.jpg', patch)

    
    
    