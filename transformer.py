from multihead_attention import MultiHeadAttention
import torch.nn as nn
import torch
import copy
import torch.nn.functional as F
from positional_encoding import PositionalEncoding


# Attention is all you need

def _get_clones(block, N=6) -> nn.ModuleList:
    block_stack = nn.ModuleList([copy.deepcopy(block) for _ in range(N)])
    return block_stack


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        # we don't want to update this
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class GEGLU(nn.Module):
    """ https://arxiv.org/abs/2002.05202 """

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return gate * F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()

        inner_dim = int(dim * mult * 2 / 3)
        self.ff = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, inner_dim * 2, bias=False),
            GEGLU(),
            LayerNorm(inner_dim),
            nn.Linear(inner_dim, dim, bias=False)
        )

    def forward(self, x):
        return self.ff(x)


class EncoderLayer(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.2):
        super().__init__()

        self.multihead_attention = MultiHeadAttention(
            dim, n_heads, d_head)
        self.feed_forward = FeedForward(dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # self attention
        attn_out = self.multihead_attention(q=x, k=x, v=x)

       # ADD & NORM
        x = attn_out + x
        x = self.dropout(self.norm(x))

        # feed forward
        fc_out = self.feed_forward(x)

        # ADD & NORM
        x = fc_out + x
        x = self.dropout(self.norm(x))

        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.2):
        super().__init__()

        self.multihead_attention = MultiHeadAttention(
            dim, n_heads, d_head)
        self.feed_forward = FeedForward(dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_inp, enc_out, mask=None):

        # self attention
        attn_out = self.multihead_attention(q=dec_inp, k=dec_inp, v=dec_inp, mask=mask)

        # ADD & NORM
        dec_inp = attn_out + dec_inp
        dec_inp = self.dropout(self.norm(dec_inp))

        # cross attention
        attn_out = self.multihead_attention(q=dec_inp, k=enc_out, v=enc_out)

        # ADD & NORM
        dec_inp = attn_out + dec_inp
        dec_inp = self.dropout(self.norm(dec_inp))

        # feed forward
        fc_out = self.feed_forward(dec_inp)

        # ADD & NORM
        dec_inp = fc_out + dec_inp
        dec_out = self.dropout(self.norm(dec_inp))  # e.g.: 32x10x512

        return dec_out


class Decoder(nn.Module):
    def __init__(self, dim, n_heads, d_head, depth):
        super().__init__()

        decoder_layer = DecoderLayer(dim, n_heads, d_head)
        self.layers = _get_clones(decoder_layer, depth)
        self.pos = PositionalEncoding(dim)

    def forward(self, dec_in, enc_out, mask=None):

      #  tgt = tgt + pos
        dec_out = self.pos(dec_in)

        # input to the decoder is the previous dec output
        for layer in self.layers:
            dec_out = layer(dec_out, enc_out, mask=mask)

        return dec_out


class Encoder(nn.Module):
    def __init__(self, dim, n_heads, d_head, depth):
        super().__init__()

        encoder_layer = EncoderLayer(dim, n_heads, d_head)

        self.layers = _get_clones(encoder_layer, depth)
        self.pos_enc = PositionalEncoding(dim)

    def forward(self, x):

        # add positional encoding
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x)

        return x


class Transformer(nn.Module):
    def __init__(self,
                 d_model,
                 n_heads=8,
                 d_head=64,
                 enc_depth=6,
                 dec_depth=6,
                 n_classes=None):
        super().__init__()

        self.encoder = Encoder(dim=d_model, n_heads=n_heads,
                               d_head=d_head, depth=enc_depth)

        self.decoder = Decoder(dim=d_model, n_heads=n_heads,
                               d_head=d_head, depth=dec_depth)

        self.linear = nn.Linear(d_model, n_classes)
    
    def get_decoder_mask(self, tgt):
        # causal mask for decoder
        i, j = tgt.shape[1], tgt.shape[1]  # i, j - timestep of Q, V
        tgt_mask = torch.ones((i, j), dtype=torch.bool).triu(j - i + 1)
        return tgt_mask

    def forward(self, x, tgt):

        tgt_mask = self.get_decoder_mask(tgt)

        enc_out = self.encoder(x)
        x = self.decoder(tgt, enc_out, mask=tgt_mask)
        x = self.linear(x)
        return x


if __name__ == '__main__':
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
