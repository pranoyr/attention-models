# ML From Scratch

The purpose of this repository is not to give the best implementation , but to give the best explanation. :sunglasses:




## Table of Contents
  * [Attention is All you Need](#attention-is-all-you-need)
  * [Vision Transformer](#vit)
  * [VQ-GAN](#vqgan)
  * [Parti](#parti)
  * [Multi Head Attention](#multi-head-attention)


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

src_seq = torch.randn(2, 10, 512)  # (b, timesteps_q, d_model)
target_seq = torch.randn(2, 20, 512)  # (b, timesteps_q, d_model)

out = transformer(src_seq, target_seq)
print(out.shape) # (b, timesteps, num_classes)


```

## ViT

Implementation of <a href="https://arxiv.org/abs/2010.11929">Vision Transformer</a>,

```python

model = Vit(512, num_classes=10)

img_batch = torch.randn(2, 3, 256, 256)
out = model(img_batch)
print(out.shape) # (b, num_classes)


```




## License
This project is licensed under the MIT License