# ML From Scratch

Implementing some of the SOTA papers based on Transformers.

## Table of Contents
  * [Attention is All you Need](#attention-is-all-you-need) (Transformer)
  * [Multi Head Attention](#multi-head-attention) 
  * [Vision Transformer](#vision-transformer) (Image Classification Transformer)
  * [Vector Quantised GAN](#vqgan) (VQGAN)
  * [Parti](#parti) (Google's text to image)
  * [MaskGIT](#MaskGIT) (Masked Generative Image Transformer)
  * [Muse](#Muse) (Text-To-Image Generation via Masked Generative Transformers)
  * [ViTVQGAN](#vitvqgan) (Vector-Quantised Image Modeling with Improved VQGAN)
  


## Attention is All you Need

Implementation of <a href="https://arxiv.org/abs/1706.03762">Attention is all you need</a>, [[Code]](models/transformer.py)

```python
import torch
from models import Transformer
from einops import rearrange

transformer = Transformer(
        dim=512,
        vocab_size=1000,
        n_heads=16,
        d_head=64,
        enc_depth=6,
        dec_depth=6,
        n_classes=1000)
    
src_timesteps = 10
tgt_timesteps = 20
batch_size = 2
vocab_size = 1000

src_seq = torch.randint(1, vocab_size, (batch_size, src_timesteps)) 

# During Training -> end token should be last token in the sequence followed by padding
tgt_seq = torch.randint(1, vocab_size, (batch_size, tgt_timesteps))

# During Training -> start token should be the first token in the sequence
tgt_shifted = torch.randint(1, vocab_size, (batch_size, tgt_timesteps))

# forward pass
out = transformer(src_seq, tgt_shifted)

# compute loss
out = rearrange(out, 'b t c -> b c t')
loss = torch.nn.functional.cross_entropy(out, tgt_seq, ignore_index=0)
loss.backward()
```

## Multi Head Attention

Implementation of Multi-Head Attention, [[Code]](models/multihead_attention.py)

```python
import torch
from models import MultiHeadAttention

attention = MultiHeadAttention(dim=512, num_heads=16, dim_head=64)
	
q = torch.randn(2, 10, 512)  # (b, timesteps_q, dim)
k = torch.randn(2, 10, 512)  # (b, timesteps_k, dim)
v = torch.randn(2, 10, 512)  # (b, timesteps_v, dim)

# causal mask used in Masked Multi-Head Attention
i, j = q.shape[1], k.shape[1]
mask = torch.ones((i, j), dtype=torch.bool).triu(j - i + 1)

output = attention(q, k, v, causal_mask=mask)
print(output.shape) # (b, timesteps, dim

```




## Vision Transformer

Implementation of <a href="https://arxiv.org/abs/2010.11929">Vision Transformer</a>, [[Code]](models/vit.py)

```python
import torch
from models import ViT

model = ViT(512, num_classes=10)

img_batch = torch.randn(2, 3, 256, 256)
out = model(img_batch)
print(out.shape) # (b, num_classes)
```


## Parti

Implementation of <a href="https://sites.research.google/parti/">Parti</a>, [[Code]](models/parti.py)

```python
import torch
from models import VQGAN, Parti, ViTVQGAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
# Vector Quantizer 
vit_params = dict(
        dim=256,
        img_size=256,
        patch_size=8,
        n_heads=8,
        d_head=64,
        depth=6,
    )

codebook_params = dict(codebook_size=8192, codebook_dim=32)
vitvqgan = ViTVQGAN(vit_params, codebook_params)

# Parti 
dim = 512
encoder_params = dict(
	t5_name = "google/t5-v1_1-base",
	max_length = 77
)
 
decoder_params = dict(
	n_heads=8,
	d_head=64,
	depth=6)
 
model = Parti(dim, vq=vitvqgan, **encoder_params, **decoder_params).to(device)

imgs = torch.randn(2, 3, 256, 256).to(device)
texts = ["this is a test", "this is another test"]

loss = model(texts, imgs)
loss.backward()
 
# Inference
model.eval()
with torch.no_grad():
	imgs = model.generate(texts)
print(imgs.shape)
```


## VQGAN

Implementation of <a href="https://github.com/CompVis/taming-transformers">VQGAN</a>, [[Code]](models/vqgan.py)

```python
import torch
from models import VQGAN

codebook_dim = 256
codebook_size = 8192

vqgan = VQGAN(codebook_dim, codebook_size)

img = torch.randn(2, 3, 256, 256)
out, loss = vqgan(img)

imgs = torch.randn(2, 3, 256, 256)
indices = vqgan.encode_imgs(imgs)
imgs = vqgan.decode_indices(indices)
print(imgs.shape)
```


## MaskGIT

Implementation of <a href="https://arxiv.org/pdf/2202.04200.pdf">MaskGIT</a>, [[Code]](models/maskgit.py)

```python
from torch import nn
import torch
from models import MaskGitTransformer
from einops import rearrange
from models.vqgan import VQGAN

# VQGAN
codebook_dim = 256
codebook_size = 8192
vqgan = VQGAN(codebook_dim, codebook_size)

# MaskGitTransformer
transformer = MaskGitTransformer(
        dim=512,
        vq=vqgan,
        vocab_size=codebook_size,
        n_heads=16,
        d_head=64,
        dec_depth=6)
    
imgs = torch.randn(2, 3, 256, 256)

# forward pass
loss = transformer(imgs)
loss.backward()
```


## MUSE

Implementation of <a href="https://arxiv.org/pdf/2301.00704.pdf">MUSE</a>, [[Code]](models/muse.py)

```python
from torch import nn
import torch
from einops import rearrange
from models.vqgan import VQGAN
from models import MUSE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VQGAN
codebook_dim = 256
codebook_size = 8192
vq = VQGAN(codebook_dim, codebook_size)


# MUSE 
dim = 512
encoder_params = dict(
        t5_name = "google/t5-v1_1-base",
        max_length = 77
)
 
decoder_params = dict(
	n_heads=8,
	d_head=64,
	depth=6)
 
muse = MUSE(dim, vq, **encoder_params, **decoder_params).to(device)
    
imgs = torch.randn(2, 3, 256, 256).to(device)
texts = ["this is a test", "this is another test"]

# forward pass
loss = muse(texts, imgs)
loss.backward()

# generate images
imgs = muse.generate(texts)
```


## ViTVQGAN

Implementation of <a href="https://arxiv.org/pdf/2110.04627.pdf">ViTVQGAN</a>, [[Code]](models/vitvqgan.py)

```python
import torch
from models import ViTVQGAN


vit_params = dict(
        dim=256,
        img_size=256,
        patch_size=8,
        n_heads=8,
        d_head=64,
        depth=6,
        mlp_dim=2048,
        dropout=0.1)

codebook_params = dict(codebook_size=8192, codebook_dim=32, beta=0.25)

imgs = torch.randn(2, 3, 256, 256)
vitvqgan = ViTVQGAN(vit_params, codebook_params)
out, loss = vitvqgan(imgs)

imgs = torch.randn(2, 3, 256, 256)
indices = vitvqgan.encode_imgs(imgs)
imgs = vitvqgan.decode_indices(indices)
print(imgs.shape)
```
### Training
```bash
python main.py --config=/cfg/vitvqgan.yaml
```

## TODOs
Search for TODO Comments in the repo and contribute. 
Thank you for contributing to the repo in advance!



## Acknowledgement
- A Big Thanks to <a href="https://github.com/lucidrains">Lucidrians</a> for his open contributions. Your repos are always a reference book for me
- <a href="https://einops.rocks/">Einops</a> made my life easier
- Can't even think of Machine Learning without <a href="https://pytorch.org/"> Pytorch</a>



