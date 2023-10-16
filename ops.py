from einops import rearrange, einsum, pack
from torch import einsum
import torch


# 2d array
x = torch.tensor([[1,2,3,4],
                [5,6,7,8]])


y = torch.tensor([[1,2,3,4],
                [5,6,7,8]])


m = torch.tensor([[1,2,3,4],
                [5,6,7,8],
                [9,10,11,12],
                [13,14,15,16]])

print("x: ", x.shape)
print("y: ", y.shape)
print("m: ", m.shape)
print()




# cat 
l = [x,x]
z = pack(l, '* i')[0]
print(f"cat: {z.shape}")
print()

# stack
z = pack(l, '* i j')[0]
print(f"stack: {z.shape}")
print()


# matrix multiplication
y_t = rearrange(y, 'i j -> j i')
z = einsum('i j, j k -> i k', x, y_t)
print(f"matrix multiplication: {z.shape}")
print()


# transpose
z = rearrange(x, 'i j -> j i')
print(f"transpose: {z.shape}")
z = einsum('i j -> j i', x)
print(f"transpose: {z.shape}")
print()


# chunk
z = x.chunk(2, dim=1)
print(f"chunk: {z}")
print()

