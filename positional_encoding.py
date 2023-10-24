
from torch.autograd import Variable
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# class PositionalEncoding(nn.Module):
# 	def __init__(self, d_model, max_len = 80):
# 		super().__init__()
# 		self.d_model = d_model
		
# 		pe = torch.zeros(max_len, d_model)
# 		for pos in range(max_len):
# 			for i in range(0, d_model, 2):
# 				# even indices
# 				pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
# 				# odd indices
# 				pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
	
# 		self.register_buffer('pe', pe)
 
	
# 	def forward(self, x):
# 		seq_len = x.size(1)
# 		x = x + self.pe[:seq_len]
# 		return x
class AbsolutePositionalEmbedding(nn.Module):
	def __init__(self, dim, max_len):
		super().__init__()
	  
		self.emb = nn.Embedding(max_len, dim)

	def forward(self, x):
		pos_emb = self.emb(x)
		pos_emb = F.normalize(pos_emb, p = 2, dim = -1)
		return pos_emb
	


class PositionalEncoding(nn.Module):
	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)

		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe = torch.zeros(max_len, d_model)
		pe[:, 0::2] = torch.sin(position * div_term) 	# even indices
		pe[:, 1::2] = torch.cos(position * div_term)		# odd indices
		self.register_buffer('pe', pe)
		
			
	def forward(self, x):
		"""
		Arguments:
			x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
		"""
		seq_len = x.size(1)
		x = x + self.pe[:seq_len]
		return self.dropout(x)
	  

if __name__ == '__main__':
	pe = PositionalEncoding(512, max_len=5000)
	x = torch.zeros(1, 100, 512) # (batch_size, seq_len, d_model)
	z = pe(x)
	print(z.shape)

	pe = AbsolutePositionalEmbedding(512, max_len=5000)
	x = torch.arange(100).unsqueeze(0)
	z = pe(x)
	print(z.shape)


