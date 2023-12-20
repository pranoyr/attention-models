from .vitvqgan import ViTVQGAN
from .vqgan import VQGAN
from .muse import MUSE
from .vit import ViT
from .transformer import Transformer
from .parti import Parti


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
            
            
 