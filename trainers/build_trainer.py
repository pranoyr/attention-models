from .vitgqgan import VQGANTrainer
from .muse import MuseTrainer
from .vit import VitTrainer
from .maskgit import MaskGitTrainer

def build_trainer(cfg, model, data_loaders):
    if cfg.model.name in ["vqgan", "vitvqgan"]:
        return VQGANTrainer(cfg, model, data_loaders)
    if cfg.model.name == "muse":
        return MuseTrainer(cfg, model, data_loaders)
    if cfg.model.name in ["vit", "vit_moe"]:
        return VitTrainer(cfg, model, data_loaders)
    if cfg.model.name == "maskgit":
        return MaskGitTrainer(cfg, model, data_loaders)

            
            
 