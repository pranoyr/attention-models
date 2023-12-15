from .vitgqgan import VQGANTrainer

def build_trainer(cfg, model, data_loaders):
    if cfg.model.name == "vitvqgan":
        return VQGANTrainer(cfg, model, data_loaders)

            
            
 