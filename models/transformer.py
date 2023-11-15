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

    def forward(self, x, context_mask=None):

        # self attention
        attn_out = self.multihead_attention(q=x, k=x, v=x, context_mask=context_mask)

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

    def forward(self, dec_inp, context, context_mask=None, causal_mask=None):

        # self attention
        attn_out = self.multihead_attention(q=dec_inp, k=dec_inp, v=dec_inp, causal_mask=causal_mask)

        # ADD & NORM
        dec_inp = attn_out + dec_inp
        dec_inp = self.dropout(self.norm(dec_inp))

        # cross attention
        attn_out = self.multihead_attention(q=dec_inp, k=context, v=context, context_mask=context_mask)

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
    
    def forward(self, dec_in, context, context_mask=None, causal_mask=None):

        # input to the decoder is the previous dec output
        for layer in self.layers:
            dec_out = layer(dec_in, context, context_mask=context_mask, causal_mask=causal_mask)
            dec_in = dec_out

        return dec_out


class Encoder(nn.Module):
    def __init__(self, dim, n_heads, d_head, depth):
        super().__init__()

        encoder_layer = EncoderLayer(dim, n_heads, d_head)
        self.layers = _get_clones(encoder_layer, depth)
    
    def forward(self, x, context_mask=None):

        for layer in self.layers:
            x = layer(x, context_mask=context_mask)

        return x


class Transformer(nn.Module):
    def __init__(self,
                 d_model,
                 vocab_size,
                 n_heads=8,
                 d_head=64,
                 enc_depth=6,
                 dec_depth=6,
                 n_classes=None):
        super().__init__()
           
        self.enc_input_proj = nn.Embedding(vocab_size, d_model)
        self.dec_input_proj = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        
        self.encoder = Encoder(dim=d_model, n_heads=n_heads,
                               d_head=d_head, depth=enc_depth)
        
        self.decoder = Decoder(dim=d_model, n_heads=n_heads,
                               d_head=d_head, depth=dec_depth)\
        
        self.linear = nn.Linear(d_model, n_classes)
    
    def get_decoder_mask(self, src_seq, tgt_seq):
        # causal mask | causal mask is a 2D triangular matrix with True values on the upper triangle.
        i = j = tgt_seq.shape[1]
        causal_mask = torch.ones((i, j), dtype=torch.bool).triu(j - i + 1)
        
        # context mask | context mask is 2D mask with True values on all elements. 
        b , t = src_seq.shape
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
            output = torch.argmax(output, dim=-1)[:,-1]
            if output[0] == end_token:
                break
            output = rearrange(output, 'b -> b 1')

            out_seq = torch.cat((out_seq, output), dim=1)
        
        return dec_in

   
    def forward(self, src_seq, tgt_seq):
                
        # get masks
        context_mask , causal_mask = self.get_decoder_mask(src_seq, tgt_seq)
        
        # Encoder
        src_seq = self.enc_input_proj(src_seq)
        src_seq = self.pos_enc(src_seq)    
        context = self.encoder(src_seq, context_mask=context_mask)
        
        # Decoder
        dec_in = self.dec_input_proj(tgt_seq)
        dec_in = self.pos_enc(dec_in)
        dec_out = self.decoder(dec_in=dec_in, context=context, context_mask=context_mask, causal_mask=causal_mask)
        
        output = self.linear(dec_out)
        return output