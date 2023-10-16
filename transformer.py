import torch
import torch.nn as nn
import math
from einops import rearrange


# n - number of timesteps
# h - number of heads (num_heads)
# d - dimension of each head (dim_head)
# b - batch size

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
        # (b, n, h * d) -> (b, h, n, d)
        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)

        # compute attention scores
        #  (b, h, n, d) * (b, h, d, n) -> (b, h, n, n)
        attn_scores = torch.matmul(Q, rearrange(K, 'b h n d -> b h d n')) / math.sqrt(self.dim_head)
        
        # apply mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Apply attention scores to V
        # (b, h, n, n) * (b, h, n, d) -> (b, h, n, d)
        output = torch.matmul(attn_probs, V)

        # combine heads
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.W_o(output)
        return output
    


attention = MultiHeadAttention(d_model=512, num_heads=16, dim_head=64)
Q = torch.randn(2, 10, 512) # (b, n, d_model)
K = torch.randn(2, 20, 512)
V = torch.randn(2, 20, 512)

mask = torch.zeros(2, 16, 10, 20) # (b, h, timesteps_q, timesteps_k)

output = attention(Q, K, V, mask)
print(output.shape)

