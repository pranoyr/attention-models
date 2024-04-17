import torch.nn as nn
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch

def get_optimizer(cfg, params):
    lr = cfg.optimizer.params.learning_rate
    warmup_steps = cfg.lr_scheduler.params.warmup_steps
    beta1 = cfg.optimizer.params.beta1
    beta2 = cfg.optimizer.params.beta2
    decay_steps = cfg.lr_scheduler.params.decay_steps
    weight_decay = cfg.optimizer.params.weight_decay
    
    if cfg.optimizer.name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    elif cfg.optimizer.name == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

    return optimizer