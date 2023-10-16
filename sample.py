from einops import rearrange, einsum
from torch import einsum
import torch


# 2d array
x = torch.tensor([[1,2,3,4],
                [5,6,7,8]])


y = torch.tensor([[1,2,3,4],
                [5,6,7,8]])

print("x: ", x.shape)
print("y: ", y.shape)


l = [x,x]

# cat
z = rearrange(l, 'b r c -> (b r) c')
print(f"cat: {z.shape}")

# stack
z = rearrange(l, 'b r c -> b r c')
print(f"stack: {z.shape}")


# matrix multiplication
y_t = rearrange(y, 'i j -> j i')
z = einsum('i j, j k -> i k', x, y_t)
print(f"matrix multiplication: {z.shape}")


# transpose
z = rearrange(x, 'i j -> j i')
print(f"transpose: {z.shape}")
z = einsum('i j -> j i', x)
print(f"transpose: {z.shape}")


