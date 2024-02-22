import torch.nn as nn
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch

def get_optimizer(cfg, model):
    if cfg.optimizer.name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.params.learning_rate, betas=(cfg.optimizer.params.beta1, cfg.optimizer.params.beta2))
    elif cfg.optimizer.name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.optimizer.params.learning_rate, momentum=cfg.optimizer.params.momentum, weight_decay=cfg.optimizer.params.weight_decay)
    return optimizer