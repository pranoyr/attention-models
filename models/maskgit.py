import torch.nn as nn
import torch
import copy
import torch.nn.functional as F
from einops import rearrange, repeat, pack
import math
from models.transformer import Encoder as BidirectionalDecorder


def cosine_schedule(t):
    return torch.cos(t * math.pi / 2)


class MaskGitTransformer(nn.Module):
    def __init__(
        self,
        dim,
        vocab_size,
        n_heads=8,
        d_head=64,
        dec_depth=6,
    ):
        super().__init__()

        self.input_proj = nn.Embedding(vocab_size+1, dim)
        num_patches = vq.encoder.num_patches
        self.pos_enc =  nn.Parameter(torch.randn(1, num_patches, dim))
        self.mask_token_id = vocab_size

        self.init_norm = nn.LayerNorm(dim)
        self.decoder = BidirectionalDecorder(
            dim=dim, n_heads=n_heads, d_head=d_head, depth=dec_depth
        )
        self.final_norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, vocab_size)
        
    def fill_mask(self, x):
        T = 8  # max number of timesteps during inference
        # sample the timestep from uniform distribution
        b , n = x.shape
        t = torch.randint(1, T, (1,))
        num_tokens_masked = cosine_schedule(t / T) * n
        num_tokens_masked = num_tokens_masked.clamp(min = 1.).int()
   
        # create mask 
        randm_perm = torch.rand(x.shape).argsort(dim = -1)
        mask = randm_perm < num_tokens_masked
        
        # fill x with mask_id, ignore the tokens that are not masked while computing loss
        tgt = x.masked_fill(~mask, -1)
        x = x.masked_fill(mask, self.mask_token_id)    
        return x, tgt

    def forward(self, x):
        x, tgt = self.fill_mask(x)
        x = self.input_proj(x)
        x += self.pos_enc

        # transformer decoder
        x = self.init_norm(x)
        dec_out = self.decoder(x)
        dec_out = self.final_norm(dec_out)
        output = self.linear(dec_out)
        
        # compute loss
        output = rearrange(output, 'b t c -> b c t')
        loss = torch.nn.functional.cross_entropy(output, tgt, ignore_index=-1)
        return loss
