from multihead_attention import MultiHeadAttention
import torch.nn as nn
import torch
import copy
import torch.nn.functional as F


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
    def __init__(self, d_model, num_heads, dim_head, dropout=0.2):
        super().__init__()

        self.multihead_attention = MultiHeadAttention(
            d_model, num_heads, dim_head)
        self.feed_forward = FeedForward(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        
        # self attention
        attn_out = self.multihead_attention(x, x, x, mask=mask)

       # ADD & NORM
        x = attn_out + x
        x = self.dropout(self.norm(x))

        # feed forward
        fc_out = self.feed_forward(x)

        # ADD & NORM
        x = fc_out + x
        x = self.dropout(self.norm(x))  

        return x


class Encoder(nn.Module):
    def __init__(self,
                 encoder_layer,
                 num_layers,
                 ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, x, mask=None, pos=None):

        #x = x + pos

        for layer in self.layers:
            x = layer(x, mask=mask)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_head, dropout=0.2):
        super().__init__()

        self.multihead_attention = MultiHeadAttention(
            d_model, num_heads, dim_head)
        self.feed_forward = FeedForward(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_out, mask=None):

        # self attention
        attn_out = self.multihead_attention(tgt, tgt, tgt, mask=mask)

        # ADD & NORM
        tgt = attn_out + tgt
        tgt = self.dropout(self.norm(tgt))

        # cross attention
        attn_out = self.multihead_attention(tgt, enc_out, enc_out)

        # ADD & NORM
        tgt = attn_out + tgt
        tgt = self.dropout(self.norm(tgt))

        # feed forward
        fc_out = self.feed_forward(tgt)

         # ADD & NORM
        tgt = fc_out + tgt
        tgt = self.dropout(self.norm(tgt))  # e.g.: 32x10x512

        return tgt


class Decoder(nn.Module):
    def __init__(
            self,
            decoder_layer,
            num_layers):
        super().__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, enc_out, mask=None):

      #  tgt = tgt + pos

        for layer in self.layers:
            tgt = layer(tgt, enc_out, mask=mask)

        return tgt


class Transformer(nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 dim_head,
                 enc_depth,
                 dec_depth,
                 num_classes):
        super().__init__()

        # a single encoder and decoder layer
        encoder = EncoderLayer(d_model, num_heads, dim_head)
        decoder = DecoderLayer(d_model, num_heads, dim_head)

        # stack of encoders
        self.encoder = Encoder(encoder, enc_depth)

        # stack of decoders
        self.decoder = Decoder(decoder, dec_depth)

        # final linear layer
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, x, tgt):

        # causal mask for decoder
        i, j = tgt.shape[1], tgt.shape[1]  # timestep of Q, V
        tgt_mask = torch.ones((i, j), dtype=torch.bool).triu(j - i + 1)

        # encoder 
        enc_out = self.encoder(x)
        # decoder
        x = self.decoder(tgt, enc_out, mask=tgt_mask)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    transformer = Transformer(
        d_model=512, num_heads=16, dim_head=64, enc_depth=6, dec_depth=6, num_classes=10)

    src_seq = torch.randn(2, 10, 512)  # (b, timesteps_q, d_model)
    target_seq = torch.randn(2, 20, 512)  # (b, timesteps_q, d_model)

    out = transformer(src_seq, target_seq)
    print(out.shape)
