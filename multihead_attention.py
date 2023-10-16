import torch
import torch.nn as nn
import math
from einops import rearrange
from torch import einsum



# h - number of heads (num_heads)
# d - dimension of each head (dim_head)
# b - batch size

# t - number of timesteps of Q,K if Q, K have the same length
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

    def forward(self, Q, K, V, mask=None):

        # prepare Q, K, V for attention
        # Q,K,V - (b_size, n_timesteps, n_heads * dk)
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # split Q, K, V into multiple heads
        # (b, t, h * d) -> (b, h, t, d)
        Q = rearrange(Q, 'b t (h d) -> b h t d', h=self.num_heads)
        K = rearrange(K, 'b t (h d) -> b h t d', h=self.num_heads)
        V = rearrange(V, 'b t (h d) -> b h t d', h=self.num_heads)

        # compute attention scores
        # Attention Scores = Q * K^T / sqrt(d_k)
        #  Q(b, h, t, d) * K(b, h, d, t) ->  (b, h, t, t)
        attn_scores = torch.matmul(Q, rearrange(K, 'b h t d -> b h d t')) / math.sqrt(self.dim_head)
        # apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Apply attention scores to V
        # (b, h, t, t) * V(b, h, t, d) -> (b, h, t, d)
        # output = torch.matmul(attn_probs, V)
        print(attn_probs.shape, V.shape)
        output = einsum('b h i j, b h j d -> b h i d', attn_probs, V)

        # combine heads
        output = rearrange(output, 'b h t d -> b t (h d)')
        output = self.W_o(output)
        return output


if __name__ == '__main__':

    attention = MultiHeadAttention(d_model=512, num_heads=16, dim_head=64)
    Q = torch.randn(2, 10, 512)  # (b, timesteps_q, d_model)
    K = torch.randn(2, 20, 512)  # (b, timesteps_k, d_model)
    V = torch.randn(2, 20, 512)  # (b, timesteps_v, d_model)

    mask = torch.zeros(2, 16, 10, 20)  # (b, h, timesteps_q, timesteps_k)

    output = attention(Q, K, V, mask)
    print(output.shape)
