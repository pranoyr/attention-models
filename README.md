# ML From Scratch

The purpose of this repository is not to give the best implementation , but to give the best explanation. :sunglasses:




## Table of Contents
  * [Attention is All you Need](#attention-is-all-you-need)
  * [Multi Head Attention](#multi-head-attention)
  * [Vision Transformer](#vision-transformer)
  * [VQ-GAN](#vqgan)
  * [Parti](#parti)


## Attention is All you Need

Implementation of <a href="https://arxiv.org/abs/1706.03762">Attention is all you need</a>,

```python
transformer = Transformer(
        d_model=512,
        n_heads=16,
        d_head=64,
        enc_depth=6,
        dec_depth=6,
        n_classes=10)

src_seq = torch.randn(2, 10, 512)  # (b, timesteps_src, d_model)
target_seq = torch.randn(2, 20, 512)  # (b, timesteps_tgt, d_model)

out = transformer(src_seq, target_seq)
print(out.shape)


```

## Multi Head Attention

Implementation of Multi-Head Attention

```python
attention = MultiHeadAttention(dim=512, num_heads=16, dim_head=64)
	
q = torch.randn(2, 10, 512)  # (b, timesteps_q, dim)
k = torch.randn(2, 10, 512)  # (b, timesteps_k, dim)
v = torch.randn(2, 10, 512)  # (b, timesteps_v, dim)

# causal mask used in Masked Multi-Head Attention
i, j = q.shape[1], k.shape[1]
mask = torch.ones((i, j), dtype=torch.bool).triu(j - i + 1)

output = attention(q, k, v, causal_mask=mask)
print(output.shape) # (b, timesteps, dim) 

```




## Vision Transformer

Implementation of <a href="https://arxiv.org/abs/2010.11929">Vision Transformer</a>,

```python
model = Vit(512, num_classes=10)

img_batch = torch.randn(2, 3, 256, 256)
out = model(img_batch)
print(out.shape) # (b, num_classes)


```


## Parti

Implementation of <a href="https://sites.research.google/parti/">Parti</a>,

```python
imgs = torch.randn(2, 3, 256, 256).to(device)
texts = ["this is a test", "this is another test"]
	
# Vector Quantizer 
codebook_dim = 256
codebook_size = 8192
vq = VQGAN(codebook_dim, codebook_size)

# Parti 
dim = 512
encoder_params = dict(
	t5_name = "google/t5-v1_1-base"
)
 
decoder_params = dict(
	n_heads = 8,
	d_head	= 64,
	depth= 6)
 
model = Parti(dim, vq, **encoder_params, **decoder_params).to(device)
loss = model(texts, imgs)
loss.backward()
 
# Inference
model.eval()
with torch.no_grad():
	imgs = model.generate(texts)
print(imgs.shape)
```
## VQGAN

Implementation of <a href="https://github.com/CompVis/taming-transformers">VQGAN</a>,

```python

codebook_dim = 256
codebook_size = 8192

vqgan = VQGAN(codebook_dim, codebook_size)

img = torch.randn(2, 3, 256, 256)
out, loss = vqgan(img)
print(loss)

img = torch.randn(2, 3, 256, 256)
indices = vqgan.encode_imgs(img)
imgs = vqgan.decode_indices(indices)
print(imgs.shape)
   

```


## Acknowledgement
- A Big Thanks to <a href="https://github.com/lucidrains">Lucidrians</a> for his contributions. Your repos are always a reference book for me
- <a href="https://einops.rocks/">Einops</a> made my life easier
- Can't even think of Machine Learning without <a href="https://pytorch.org/"> Pytorch</a>



