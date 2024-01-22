from .vitgqgan import VQGANTrainer
from .muse import MuseTrainer

def build_trainer(cfg, model, data_loaders):
    if cfg.model.name == "vitvqgan":
        return VQGANTrainer(cfg, model, data_loaders)
    if cfg.model.name == "muse":
        return MuseTrainer(cfg, model, data_loaders)

            
            
 