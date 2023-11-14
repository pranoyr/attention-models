from torch import nn
from models.multihead_attention import MultiHeadAttention

def _get_clones(block, N=6) -> nn.ModuleList:
    block_stack = nn.ModuleList([copy.deepcopy(block) for _ in range(N)])
    return block_stack


class MaskGitLayer(nn.Module):
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


class MaskGit(nn.Module):
    def __init__(self, dim, n_heads, d_head, depth):
        super().__init__()

        decoder_layer = MaskGitLayer(dim, n_heads, d_head)
        self.layers = _get_clones(decoder_layer, depth)
    
    def forward(self, dec_in, context, context_mask=None, causal_mask=None):

        # input to the decoder is the previous dec output
        for layer in self.layers:
            dec_out = layer(dec_in, context, context_mask=context_mask, causal_mask=causal_mask)
            dec_in = dec_out

        return dec_out