from models.multihead_attention import MultiHeadAttention
import torch.nn as nn
import torch
import copy
import torch.nn.functional as F
from models.positional_encoding import PositionalEncoding
from einops import rearrange, repeat, pack


def _get_clones(block, N=6) -> nn.ModuleList:
    block_stack = nn.ModuleList([copy.deepcopy(block) for _ in range(N)])
    return block_stack


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        # we don't want to update this
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class GEGLU(nn.Module):
    """https://arxiv.org/abs/2002.05202"""

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return gate * F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()

        inner_dim = int(dim * mult * 2 / 3)
        self.ff = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias=False),
            GEGLU(),
            LayerNorm(inner_dim),
            nn.Linear(inner_dim, dim, bias=False),
        )

    def forward(self, x):
        return self.ff(x)


# Both Encoder and Decoder use Pre LayerNorm


class Encoder(nn.Module):
    def __init__(self, dim, n_heads, d_head, depth):
        super().__init__()

        encoder_layer = EncoderLayer(dim, n_heads, d_head)
        self.layers = _get_clones(encoder_layer, depth)

    def forward(self, x, context_mask=None):
        for layer in self.layers:
            x = layer(x, context_mask=context_mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.2):
        super().__init__()

        self.multihead_attention = MultiHeadAttention(dim, n_heads, d_head)
        self.feed_forward = FeedForward(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context_mask=None):
        x_norm = self.norm1(x)
        # self attention
        attn_out = self.multihead_attention(
            q=x_norm, k=x_norm, v=x_norm, context_mask=context_mask
        )

        # ADD & NORM
        x = attn_out + x
        x_norm = self.norm2(x)

        # feed forward
        fc_out = self.feed_forward(x_norm)

        # ADD
        x = fc_out + x
        return x


class Decoder(nn.Module):
    def __init__(self, dim, n_heads, d_head, depth):
        super().__init__()

        decoder_layer = DecoderLayer(dim, n_heads, d_head)
        self.layers = _get_clones(decoder_layer, depth)

    def forward(self, dec_in, context, context_mask=None, causal_mask=None):
        # input to the decoder is the previous dec output
        for layer in self.layers:
            dec_out = layer(
                dec_in, context, context_mask=context_mask, causal_mask=causal_mask
            )
            dec_in = dec_out

        return dec_out


class DecoderLayer(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.2):
        super().__init__()

        self.multihead_attention = MultiHeadAttention(dim, n_heads, d_head)
        self.feed_forward = FeedForward(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_inp, context, context_mask=None, causal_mask=None):
        dec_inp_norm = self.norm1(dec_inp)
        # self attention
        attn_out = self.multihead_attention(
            q=dec_inp_norm, k=dec_inp_norm, v=dec_inp_norm, causal_mask=causal_mask
        )

        # ADD & NORM
        dec_inp = attn_out + dec_inp
        dec_inp_norm = self.norm2(dec_inp)

        # cross attention
        attn_out = self.multihead_attention(
            q=dec_inp_norm, k=context, v=context, context_mask=context_mask
        )

        # ADD & NORM
        dec_inp = attn_out + dec_inp
        dec_inp_norm = self.norm3(dec_inp)

        # feed forward
        fc_out = self.feed_forward(dec_inp_norm)

        # ADD
        dec_out = fc_out + dec_inp
        return dec_out


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        vocab_size,
        n_heads=8,
        d_head=64,
        enc_depth=6,
        dec_depth=6,
        n_classes=None,
    ):
        super().__init__()

        # Encoder
        self.enc_input_proj = nn.Embedding(vocab_size, dim)
        self.dec_input_proj = nn.Embedding(vocab_size, dim)
        self.pos_enc = PositionalEncoding(dim)
        self.encoder_norm = LayerNorm(dim)
        self.encoder = Encoder(dim=dim, n_heads=n_heads, d_head=d_head, depth=enc_depth)
        
        # Decoder
        self.decoder = Decoder(dim=dim, n_heads=n_heads, d_head=d_head, depth=dec_depth)
        self.decoder_norm = LayerNorm(dim)
        self.linear = nn.Linear(dim, n_classes)

    def get_decoder_mask(self, src_seq, tgt_seq):
        # causal mask -> 2D triangular matrix with True values on the upper triangle.
        i = j = tgt_seq.shape[1]
        causal_mask = torch.ones((i, j), dtype=torch.bool).triu(j - i + 1)

        # context mask -> 2D mask with False values for all PAD tokens.
        b, t = src_seq.shape
        context_mask = torch.ones((b, t), dtype=torch.bool)

        return context_mask, causal_mask

    def generate(self, src_seq: torch.Tensor):
        src_seq = self.enc_input_proj(src_seq)
        src_seq = self.pos_enc(src_seq)

        # Encoder
        context = self.encoder(src_seq)

        end_token = 2
        b = src_seq.shape[0]

        # Auto-regressive decoding
        out_seq = torch.ones((b, 1), dtype=torch.long, device=src_seq.device)
        while True:
            dec_in = self.dec_input_proj(out_seq)
            dec_in = self.pos_enc(dec_in)
            dec_out = self.decoder(dec_in=dec_in, context=context)
            output = self.linear(dec_out)

            # sample the last token
            output = torch.argmax(output, dim=-1)[:, -1]
            if output[0] == end_token:
                break
            output = rearrange(output, "b -> b 1")

            out_seq = torch.cat((out_seq, output), dim=1)

        return out_seq

    def forward(self, src_seq, tgt_seq):
        # get masks
        context_mask, causal_mask = self.get_decoder_mask(src_seq, tgt_seq)

        # Encoder
        src_seq = self.enc_input_proj(src_seq)
        src_seq = self.pos_enc(src_seq)
        # added layer norm for better stability
        src_seq = self.encoder_norm(src_seq)
        context = self.encoder(src_seq, context_mask=context_mask)

        # Decoder
        dec_in = self.dec_input_proj(tgt_seq)
        dec_in = self.pos_enc(dec_in)
        # added layer norm for better stability
        dec_in = self.decoder_norm(dec_in)
        dec_out = self.decoder(
            dec_in=dec_in,
            context=context,
            context_mask=context_mask,
            causal_mask=causal_mask,
        )

        output = self.linear(dec_out)
        return output
