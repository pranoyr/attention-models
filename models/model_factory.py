from .vitvqgan import ViTVQGAN
from .vqgan import VQGAN
from .muse import MUSE
from .vit import ViT
from .transformer import Transformer
from .parti import Parti
from .muse import MUSE
import torch


def load_model(model, checkpoint):
	ckpt = torch.load(checkpoint)
	model.load_state_dict(ckpt['state_dict'])
	print(f"Loaded checkpoint from {checkpoint}")


def build_model(cfg):
	if cfg.model.name == "vitvqgan":
		vit_params = dict(
			dim = cfg.model.transformer.dim,
			img_size = cfg.dataset.preprocessing.resolution,
			patch_size = cfg.model.transformer.patch_size,
			n_heads = cfg.model.transformer.n_heads,
			d_head = cfg.model.transformer.d_head,
			depth = cfg.model.transformer.depth,
			mlp_dim = cfg.model.transformer.mlp_dim,
			dropout = cfg.model.transformer.dropout
		)
		codebook_params = dict(
			codebook_dim = cfg.codebook.codebook_dim,
			codebook_size = cfg.codebook.codebook_size
		)
		model = ViTVQGAN(vit_params, codebook_params)
		return model

	if cfg.model.name == "muse":

		# ViTVQGAN
		vit_params = dict(
			dim = cfg.vitvqgan.transformer.dim,
			img_size = cfg.dataset.preprocessing.resolution,
			patch_size = cfg.vitvqgan.transformer.patch_size,
			n_heads = cfg.vitvqgan.transformer.n_heads,
			d_head = cfg.vitvqgan.transformer.d_head,
			depth = cfg.vitvqgan.transformer.depth,
			mlp_dim = cfg.vitvqgan.transformer.mlp_dim,
			dropout = cfg.vitvqgan.transformer.dropout
		)
		codebook_params = dict(
			codebook_dim = cfg.codebook.codebook_dim,
			codebook_size = cfg.codebook.codebook_size
		)
		vq = ViTVQGAN(vit_params, codebook_params)
		load_model(vq, cfg.vitvqgan.checkpoint)

		# MUSE 
		dim = cfg.muse.dim
		encoder_params = dict(
				t5_name = cfg.muse.encoder.t5_name,
				max_length = cfg.muse.encoder.max_length
		)
		
		decoder_params = dict(
			n_heads=cfg.muse.n_heads,
			d_head=cfg.muse.d_head,
			depth=cfg.muse.depth)


		model = MUSE(dim, vq, **encoder_params, **decoder_params)
		return model
			
			
 