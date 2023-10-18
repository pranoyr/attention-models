import torch
import torch.nn as nn
import math
from einops import rearrange
from torch import einsum


# h - number of heads (num_heads)
# d - dimension of each head (dim_head)
# b - batch size

# t - number of timesteps of Q,K,V
# When Q,K have different lengths
# i - number of timesteps for Q
# j - number of timesteps for K

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dim_head):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_head = dim_head

        self.W_q = nn.Linear(d_model, num_heads * dim_head)
        self.W_k = nn.Linear(d_model, num_heads * dim_head)
        self.W_v = nn.Linear(d_model, num_heads * dim_head)
        self.W_o = nn.Linear(num_heads * dim_head, d_model)

        self.scale = dim_head ** -0.5

    def forward(self, q, k, v, mask=None):

        # prepare Q, K, V for attention
        # Q,K,V - (b_size, n_timesteps, n_heads * dk)
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        # split Q, K, V into multiple heads
        # (b, t, h * d) -> (b, h, t, d)
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)

        # compute attention scores
        # Attention Scores = Q * K^T / sqrt(d_k)
        #  Q(b, h, t, d) * K(b, h, d, t) ->  (b, h, t, t)
        k_transpose = rearrange(k, 'b h t d -> b h d t')
        attn_scores = einsum('b h i d, b h d j -> b h i j', q * self.scale, k_transpose)
        # apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Apply attention scores to V
        # (b, h, t, t) * V(b, h, t, d) -> (b, h, t, d)
        output = einsum('b h i j, b h j d -> b h i d', attn_probs, v)

        # combine heads
        output = rearrange(output, 'b h t d -> b t (h d)')
        output = self.W_o(output)
        return output


if __name__ == '__main__':

    attention = MultiHeadAttention(d_model=512,
                                   num_heads=16,
                                   dim_head=64)
    
    q = torch.randn(2, 10, 512)  # (b, timesteps_q, d_model)
    k = torch.randn(2, 10, 512)  # (b, timesteps_k, d_model)
    v = torch.randn(2, 10, 512)  # (b, timesteps_v, d_model)

    # causal mask used in Masked Multi-Head Attention
    i, j = q.shape[1], k.shape[1]
    mask = torch.ones((i, j), dtype=torch.bool).triu(j - i + 1)

    output = attention(q, k, v, mask)
    print(output.shape)
