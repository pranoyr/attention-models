from .vitvqgan import ViTVQGAN
from .vqgan import VQGAN
from .muse import MUSE
from .vit import ViT
from .transformer import Transformer
from .parti import Parti
from .muse import MUSE
from .maskgit import MaskGitTransformer
import torch
import logging
from models import ViT, ViTMoE


def load_model(model, checkpoint):
	ckpt = torch.load(checkpoint)
	model.load_state_dict(ckpt['state_dict'])
	logging.info(f"Loaded pretrained ViTVQGAN from {checkpoint}")

def freeze_model(model):
	for param in model.parameters():
		param.requires_grad = False


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
		
	if cfg.model.name == "vqgan":
		
		codebook_dim = cfg.codebook.codebook_dim
		codebook_size = cfg.codebook.codebook_size
  
		model = VQGAN(codebook_dim, codebook_size)
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
		dim = cfg.model.dim
		encoder_params = dict(
				enc_type = cfg.model.encoder.type,
				enc_name = cfg.model.encoder.name,
				max_length = cfg.model.encoder.max_length
		)
		
		decoder_params = dict(
			n_heads=cfg.model.decoder.n_heads,
			d_head=cfg.model.decoder.d_head,
			depth=cfg.model.decoder.depth)


		model = MUSE(dim, vq, **encoder_params, **decoder_params)
		return model
	
	if cfg.model.name == "maskgit":
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

		# MaskGit 
		dim = cfg.model.dim
		# MaskGitTransformer
		model = MaskGitTransformer(
			dim=dim,
			vq=vq,
			vocab_size=cfg.codebook.codebook_size,
			n_heads=cfg.model.n_heads,
			d_head=cfg.model.d_head,
			dec_depth=cfg.model.depth)
		return model
	
	if cfg.model.name == "vit":
		# Vit
		model = ViT(
			dim = cfg.model.transformer.dim,
			image_size = cfg.dataset.preprocessing.resolution,
			patch_size = cfg.model.transformer.patch_size,
			depth = cfg.model.transformer.depth,
			n_heads = cfg.model.transformer.n_heads,
			mlp_dim = cfg.model.transformer.mlp_dim,
			dropout = cfg.model.transformer.dropout,
			num_classes = cfg.model.transformer.num_classes
		)
		return model

	if cfg.model.name == "vit_moe":
    		# Vit
		model = ViTMoE(
			dim = cfg.model.transformer.dim,
			image_size = cfg.dataset.preprocessing.resolution,
			n_heads=cfg.model.transformer.n_heads,
			patch_size = cfg.model.transformer.patch_size,
			depth = cfg.model.transformer.depth,
			n_experts=cfg.model.transformer.n_experts,
			sel_experts=cfg.model.transformer.sel_experts,
			dropout=cfg.model.transformer.dropout,
			num_classes = cfg.model.transformer.num_classes
		)
		return model
			
			
			
 