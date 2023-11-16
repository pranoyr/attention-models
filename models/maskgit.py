import torch.nn as nn
import torch
import copy
import torch.nn.functional as F
from einops import rearrange, repeat, pack
from models.multihead_attention import MultiHeadAttention
from models.transformer import FeedForward, _get_clones
import math


def cosine_schedule(t):
    return torch.cos(t * math.pi / 2)


class DecoderLayer(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.2):
        super().__init__()

        self.multihead_attention = MultiHeadAttention(dim, n_heads, d_head)
        self.feed_forward = FeedForward(dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_inp):
        # self attention
        attn_out = self.multihead_attention(q=dec_inp, k=dec_inp, v=dec_inp)

        # ADD & NORM
        dec_inp = attn_out + dec_inp
        dec_inp = self.dropout(self.norm(dec_inp))

        # feed forward
        fc_out = self.feed_forward(dec_inp)

        # ADD & NORM
        dec_inp = fc_out + dec_inp
        dec_out = self.dropout(self.norm(dec_inp))  # e.g.: 32x10x512

        return dec_out


class BidirectionalDecorder(nn.Module):
    def __init__(self, dim, n_heads, d_head, depth):
        super().__init__()

        decoder_layer = DecoderLayer(dim, n_heads, d_head)
        self.layers = _get_clones(decoder_layer, depth)

    def forward(self, dec_in):
        # input to the decoder is the previous dec output
        for layer in self.layers:
            dec_out = layer(dec_in)
            dec_in = dec_out

        return dec_out



class MaskGitTransformer(nn.Module):
    def __init__(
        self,
        dim,
        vocab_size,
        n_heads=8,
        d_head=64,
        enc_depth=6,
        dec_depth=6,
    ):
        super().__init__()

        self.input_proj = nn.Embedding(vocab_size+1, dim)
        self.pos_enc =  nn.Parameter(torch.randn(1, dim))
        self.mask_token_id = vocab_size
    
        self.decoder = BidirectionalDecorder(
            dim=dim, n_heads=n_heads, d_head=d_head, depth=dec_depth
        )
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
        
        # fill x with mask_id
        x = x.masked_fill(mask, self.mask_token_id)       
        return x 

    def forward(self, x):
        x = self.fill_mask(x)
        x = self.input_proj(x)
        x += self.pos_enc
        
        dec_out = self.decoder(x)

        output = self.linear(dec_out)
        return output
