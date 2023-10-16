import torch
import torch.nn as nn
import math
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        print(Q.shape)
        print(K.transpose(-2, -1).shape)
        print(attn_scores.shape)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        print(V.shape)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        # x - (batch_size, seq_length, d_model)
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        x = rearrange(x, 'b h n d -> b n (h d)')
        return x
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q)) # Q -(batch_size, num_heads, seq_length, d_k)
        K = self.split_heads(self.W_k(K)) # K -(batch_size, num_heads, seq_length, d_k)
        V = self.split_heads(self.W_v(V))   # V -(batch_size, num_heads, seq_length, d_k)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

attention = MultiHeadAttention(512, 8)
Q = torch.randn(2, 10, 512)
K = torch.randn(2, 10, 512)
V = torch.randn(2, 10, 512)
output = attention(Q, K, V)
# print(output.shape) 
