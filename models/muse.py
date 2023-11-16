import torch.nn as nn
import torch
import copy
import torch.nn.functional as F
from einops import rearrange, repeat, pack
from models.multihead_attention import MultiHeadAttention
from models.transformer import FeedForward, _get_clones
import math
from typing import List
from models.t5 import T5Encoder, get_encoded_dim
from models.transformer import Decoder


def cosine_schedule(t):
    return torch.cos(t * math.pi / 2)


class TextEncoder(torch.nn.Module):
    def __init__(self, dim, t5_name):
        super().__init__()

        self.t5_encoder = T5Encoder(t5_name)
        text_embed_dim = get_encoded_dim(t5_name)
        self.text_embed_proj = nn.Linear(text_embed_dim, dim, bias=False)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, texts: List[str]):
        context_mask, text_embeds = self.t5_encoder(texts)
        text_embeds = self.text_embed_proj(text_embeds)
        # add layer norm
        text_embeds = self.layer_norm(text_embeds)
        return context_mask, text_embeds


class MUSE(nn.Module):
    def __init__(
        self,
        dim,
        vq,
        t5_name,
        n_heads,
        d_head,
        depth,
    ):
        super().__init__()

        #### Text Encoder  ####
        self.text_encoder = TextEncoder(dim, t5_name)

        #### Transformer Decoder ####
        self.vq = vq
        codebook_size = vq.codebook.codebook_size
        self.mask_token_id = codebook_size
        self.token_emb = nn.Embedding(codebook_size + 1, dim)
        self.pos_enc =  nn.Parameter(torch.randn(1, dim))
        self.decoder = Decoder(dim=dim, n_heads=n_heads, d_head=d_head, depth=depth)
        self.linear = nn.Linear(dim, codebook_size)

    def fill_mask(self, x):
        device = x.device
        T = 8  # max number of timesteps during inference
        # sample the timestep from uniform distribution
        b, n = x.shape
        t = torch.randint(1, T, (1,))
        num_tokens_masked = cosine_schedule(t / T) * n
        num_tokens_masked = num_tokens_masked.clamp(min=1.0).int()
        num_tokens_masked = num_tokens_masked.to(device)

        # create mask
        randm_perm = torch.rand(x.shape).argsort(dim=-1).to(device)
        mask = randm_perm < num_tokens_masked

        # fill x with mask_id, ignore the tokens that are not masked while computing loss
        tgt = x.masked_fill(~mask, -1)
        x = x.masked_fill(mask, self.mask_token_id)
        return x, tgt

    def forward(self, texts, imgs):
        # text encoder
        context_mask, text_embeds = self.text_encoder(texts)  # (batch_size, seq_len, dim)

        # quantize images
        img_token_indices = self.vq.encode_imgs(imgs)

        # apply cosine schedule to img tokens
        img_token_indices, tgt = self.fill_mask(img_token_indices)
        
		# add positional encoding
        img_token_embeds = self.token_emb(img_token_indices)
        img_token_embeds += self.pos_enc

        # bidirectional decoder
        dec_out = self.decoder(
            dec_in=img_token_embeds, context=text_embeds, context_mask=context_mask
        )
        output = self.linear(dec_out)

        # compute loss
        output = rearrange(output, "b t c -> b c t")
        loss = torch.nn.functional.cross_entropy(output, tgt, ignore_index=-1)
        return loss