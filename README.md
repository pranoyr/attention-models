# ML From Scratch

The purpose of this repository is not to give the best implementation , but to give the best explanation. :sunglasses:




## Table of Contents
  * [Attention is All you Need](#attention-is-all-you-need)
  * [Vision Transformer](#Download-dataset)
  * [VQ-GAN](#Train)
  * [Parti](#Inference)
  * [Multi Head Attention](#Inference)


## Attention is All you Need


[[Attention is all you need](https://arxiv.org/abs/1706.03762)] 

```

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
print(out.shape)



```



## License
This project is licensed under the MIT License