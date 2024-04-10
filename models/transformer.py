from models.softmax_attention import SoftmaxAttention
import torch.nn as nn
import torch
import copy
import torch.nn.functional as F
from models.positional_encoding import PositionalEncoding
from einops import rearrange, repeat, pack
import torch.nn as nn


class LayerNorm(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.gamma = nn.Parameter(torch.ones(dim))
		# we don't want to update this
		self.register_buffer("beta", torch.zeros(dim))

	def forward(self, x):
		return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class GEGLU(nn.Module):
	"""https://arxiv.org/abs/2002.05202"""

	def forward(self, x):
		x, gate = x.chunk(2, dim=-1)
		return gate * F.gelu(x)


class FeedForward(nn.Module):
	def __init__(self, dim, mult=4):
		super().__init__()

		inner_dim = int(dim * mult * 2 / 3)
		self.ff = nn.Sequential(
			nn.Linear(dim, inner_dim * 2, bias=False),
			GEGLU(),
			LayerNorm(inner_dim),
			nn.Linear(inner_dim, dim, bias=False),
		)

	def forward(self, x):
		return self.ff(x)


# Both Encoder and Decoder use Pre LayerNorm


class Encoder(nn.Module):
	def __init__(self, dim, n_heads=8, d_head=64, depth=6, mult=4, dropout=0.0):
		super().__init__()
	
		self.layers = nn.ModuleList([EncoderLayer(dim, n_heads, d_head, mult, dropout) for _ in range(depth)])
 
	def forward(self, x, context_mask=None):
		for layer in self.layers:
			x = layer(x, context_mask=context_mask)
		return x


class EncoderLayer(nn.Module):
	def __init__(self, dim, n_heads=8, d_head=64, mult=4, dropout=0.0):
		super().__init__()

		self.self_attn = SoftmaxAttention(dim, n_heads, d_head, dropout)
		self.feed_forward = FeedForward(dim, mult=mult)
		self.norm1 = LayerNorm(dim)
		self.norm2 = LayerNorm(dim)
		
	def forward(self, x, context_mask=None):
		x_norm = self.norm1(x)
		# self attention
		attn_out = self.self_attn(x=x_norm, context_mask=context_mask)

		# ADD & NORM
		x = attn_out + x
		x_norm = self.norm2(x)

		# feed forward
		fc_out = self.feed_forward(x_norm)

		# ADD
		x = fc_out + x
		return x


class Decoder(nn.Module):
	def __init__(self, dim, n_heads=8, d_head=64, depth=6, mult=4, dropout=0.):
		super().__init__()

		self.layers = nn.ModuleList([DecoderLayer(dim, n_heads, d_head, mult, dropout) for _ in range(depth)])

	def forward(self, dec_in, context, context_mask=None, causal_mask=None):
		# input to the decoder is the previous dec output
		for layer in self.layers:
			dec_out = layer(dec_in, context, context_mask=context_mask, causal_mask=causal_mask)
			dec_in = dec_out

		return dec_out


class DecoderLayer(nn.Module):
	def __init__(self, dim, n_heads=8, d_head=64, mult=4, dropout=0.0):
		super().__init__()

		self.self_attn = SoftmaxAttention(dim, n_heads, d_head, dropout)
		self.cross_attn = SoftmaxAttention(dim, n_heads, d_head, dropout)
		
		self.feed_forward = FeedForward(dim, mult)
		self.norm1 = LayerNorm(dim)
		self.norm2 = LayerNorm(dim)
		self.norm3 = LayerNorm(dim)

	def forward(self, dec_inp, context, context_mask=None, causal_mask=None):
		dec_inp_norm = self.norm1(dec_inp)
		# self attention
		attn_out = self.self_attn(x=dec_inp_norm, causal_mask=causal_mask)

		# ADD & NORM
		dec_inp = attn_out + dec_inp
		dec_inp_norm = self.norm2(dec_inp)

		# cross attention
		attn_out = self.cross_attn(x=dec_inp_norm, context=context, context_mask=context_mask)

		# ADD & NORM
		dec_inp = attn_out + dec_inp
		dec_inp_norm = self.norm3(dec_inp)

		# feed forward
		fc_out = self.feed_forward(dec_inp_norm)

		# ADD
		dec_out = fc_out + dec_inp
		return dec_out


class Transformer(nn.Module):
	def __init__(
		self,
		dim,
		vocab_size=1000,
		n_heads=8,
		d_head=64,
		enc_depth=6,
		dec_depth=6,
		n_classes=None,
	):
		super().__init__()

		# Encoder
		self.enc_input_proj = nn.Embedding(vocab_size, dim)
		self.dec_input_proj = nn.Embedding(vocab_size, dim)
		self.pos_enc = PositionalEncoding(dim)
		self.enc_init_norm = LayerNorm(dim)
		self.encoder = Encoder(dim=dim, n_heads=n_heads, d_head=d_head, depth=enc_depth)
		self.enc_final_norm = LayerNorm(dim)
		
		# Decoder
		self.dec_init_norm = LayerNorm(dim)
		self.decoder = Decoder(dim=dim, n_heads=n_heads, d_head=d_head, depth=dec_depth)
		self.dec_final_norm = LayerNorm(dim)
		self.linear = nn.Linear(dim, n_classes)

	def get_decoder_mask(self, src_seq, tgt_seq):
		# causal mask -> 2D triangular matrix with True values on the upper triangle.
		i = j = tgt_seq.shape[1]
		causal_mask = torch.ones((i, j), dtype=torch.bool).triu(j - i + 1)

		# context mask -> 2D mask with False values for all PAD tokens.
		b, t = src_seq.shape
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
			logits = self.linear(dec_out)
			# sample
			last_token = F.gumbel_softmax(logits[:, -1, :], tau=1, hard=False)
			last_token = torch.argmax(last_token, dim=-1)
			if last_token[0] == end_token:
				break
			last_token = rearrange(last_token, "b -> b 1")

			out_seq = torch.cat((out_seq, last_token), dim=1)

		return out_seq

	def forward(self, src_seq, tgt_seq):
		# get masks
		context_mask, causal_mask = self.get_decoder_mask(src_seq, tgt_seq)

		# Encoder
		src_seq = self.enc_input_proj(src_seq)
		src_seq = self.pos_enc(src_seq)
		src_seq = self.enc_init_norm(src_seq)
		context = self.encoder(src_seq, context_mask=context_mask)
		context = self.enc_final_norm(context)

		# Decoder
		dec_in = self.dec_input_proj(tgt_seq)
		dec_in = self.pos_enc(dec_in)
		dec_in = self.dec_init_norm(dec_in)
		dec_out = self.decoder(
			dec_in=dec_in,
			context=context,
			context_mask=context_mask,
			causal_mask=causal_mask,
		)
		dec_out = self.dec_final_norm(dec_out)

		output = self.linear(dec_out)
		return output
