import torch
from torch import einsum
from einops import rearrange, repeat, reduce

##########################################################################
######################## Linear regression  ##############################
##########################################################################

num_samples = 10
num_features = 5

# Dataset
x , y = torch.randn(num_samples, num_features), torch.randn(num_samples, 1)


# make W and b trainable parameters
# W must contains as many paramters as there are features in X
W = torch.randn(num_features, requires_grad=True)
b = torch.randn(1, requires_grad=True)

W_t = rearrange(W, 'i -> i 1')

# Y  = XW^T + b   or  Y = x1w1 + x2w2 + x3w3 + x4w4 ... xnyn + b
# (b, n_feat) * (n_feat, 1) -> (b, 1)
y_pred = einsum('b i, i j -> b j', x, W_t) + b

# loss function
loss = ((y_pred - y)**2).mean()
print("loss: ", loss)


##########################################################################
######################## Logistic regression  ############################
##########################################################################


num_samples = 10
num_features = 5

# Dataset
x  = torch.randn(num_samples, num_features)
# generate random labels
y = torch.randint(0, 2, (num_samples, 1)).float()

# make W and b trainable parameters
# W must contains as many paramters as there are features in X
W = torch.randn(num_features, requires_grad=True)
b = torch.randn(1, requires_grad=True)

W_t = rearrange(W, 'i -> i 1')

# Y  = sigmoid(XW^T + b)  
# (b, n_feat) * (n_feat, 1) -> (b, 1)
z  = einsum('b i, i j -> b j', x, W_t) + b

# add sigmoid function
y_pred = torch.sigmoid(z)

# binary cross entropy loss
loss = -(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred)).mean()
print(loss)
