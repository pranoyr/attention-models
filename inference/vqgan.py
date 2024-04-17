import torch
import sys
import cv2
import numpy as np
from torchvision import transforms as T
from PIL import Image
import argparse
sys.path.append('.')
from models import VQGAN



def restore(x):
    # x = (x + 1) * 0.5
    x = x.permute(1,2,0).detach().cpu().numpy()
    x = (255*x).astype(np.uint8)
    return x


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--image', type=str, default='data/images/scene.jpg', help='path to the image')
parser.add_argument('--ckpt', type=str, default='outputs/VQGAN/checkpoints/VQGAN.pt', help='path to the checkpoint')
args = parser.parse_args()


transforms = T.Compose([
    T.Resize((256, 256)),
	T.ToTensor(),
	# T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# params
vit_params = dict(
        dim=768,
        img_size=256,
        patch_size=8,
        n_heads=12,
        d_head=64,
        depth=12,
        mlp_dim=3072,
        dropout=0)
codebook_params = dict(codebook_size=8192, codebook_dim=32, beta=0.25)


# model

codebook_dim = 256
codebook_size = 8192

vq = VQGAN(codebook_dim, codebook_size)
# load checkpoint
ckpt = torch.load(args.ckpt, map_location='cpu')
vq.load_state_dict(ckpt['state_dict'])
vq.eval()

# load image
imgs = Image.open(args.image)
imgs = transforms(imgs)
imgs = imgs.unsqueeze(0)

# inference
with torch.no_grad():
	indices = vq.encode_imgs(imgs)
	imgs = vq.decode_indices(indices)

# display
img = restore(imgs[0])
img = img[:, :, ::-1]
cv2.imshow('test', img)
cv2.waitKey(0)