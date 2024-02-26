import torch
import sys
import cv2
import numpy as np
from torchvision import transforms as T
from PIL import Image
import argparse
sys.path.append('.')
from models import ViTVQGAN, MUSE



def restore(x):
    x = (x + 1) * 0.5
    x = x.permute(1,2,0).detach().cpu().numpy()
    x = (255*x).astype(np.uint8)
    return x


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--vae_ckpt', type=str, default='outputs/vitvqgan/checkpoints/VitVQGAN.pt', help='path to the checkpoint')
parser.add_argument('--ckpt', type=str, default='outputs/muse/checkpoints/MUSE.pt', help='path to the checkpoint')
parser.add_argument
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = T.Compose([
    T.Resize((256, 256)),
	T.ToTensor(),
	T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# define ViTVQGAN
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
vitvqgan = ViTVQGAN(vit_params, codebook_params)
ckpt = torch.load(args.vae_ckpt, map_location='cpu')
vitvqgan.load_state_dict(ckpt['state_dict'])
vitvqgan.eval()

# Muse  
dim = 512
encoder_params = dict(
	enc_type = 'clip',
	enc_name = 'openai/clip-vit-base-patch32',
	max_length = 77
)
 
decoder_params = dict(
	n_heads=8,
	d_head=64,
	depth=6)
 
muse = MUSE(dim, vitvqgan, **encoder_params, **decoder_params).to(device)
state_dict = torch.load(args.ckpt, map_location='cpu')['state_dict']
muse.load_state_dict(state_dict)
muse.eval()

# text input
texts = ["orange"]

# generate image
imgs = muse.generate(texts)

# display
img = restore(imgs[0])
img = img[:, :, ::-1]
cv2.imshow('test', img)
cv2.waitKey(0)