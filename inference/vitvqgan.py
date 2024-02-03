import torch
from models import ViTVQGAN
import cv2
import numpy as np

def restore(x):
    x = (x + 1) * 0.5
    x = x.permute(1,2,0).detach().cpu().numpy()
    x = (255*x).astype(np.uint8)
    return x


vit_params = dict(
        dim=256,
        img_size=256,
        patch_size=8,
        n_heads=8,
        d_head=64,
        depth=8,
        mlp_dim=2048,
        dropout=0.1)

codebook_params = dict(codebook_size=8192, codebook_dim=32, beta=0.25)

vitvqgan = ViTVQGAN(vit_params, codebook_params)


ckpt = torch.load('/Users/pranoy/Downloads/vit_vq_org.pt', map_location='cpu')

vitvqgan.load_state_dict(ckpt['state_dict'])
vitvqgan.eval()



imgs = cv2.imread('/Users/pranoy/Downloads/photo-1590992141027-1aaec7710e79.jpeg')

print(imgs.shape)
imgs = cv2.resize(imgs, (256, 256))
cv2.imwrite('test1.png', imgs)
imgs = torch.from_numpy(imgs).permute(2, 0, 1).unsqueeze(0).float()
indices = vitvqgan.encode_imgs(imgs)
imgs = vitvqgan.decode_indices(indices)

# imgs, loss = vitvqgan(imgs)
print(imgs.shape)

img = restore(imgs[0])
cv2.imshow('test', img)
cv2.waitKey(0)

# cv2.imwrite('test.png', img)

