import torch
import math
import os
import torch.nn as nn

from copy import deepcopy
from pathlib import Path
from typing import Union




def de_parallel(model):
    """Return single-GPU model if model is wrapped (e.g. DDP); else return model."""
    return model.module if hasattr(model, "module") else model


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """Cosine ramp from y1 to y2 over steps (for LR scheduler)"""
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1


def copy_attr(a, b, include=(), exclude=()):
    """Copy attributes from b to a, with optional include/exclude."""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        setattr(a, k, v)


class ModelEMA:
    """
    Exponential Moving Average of model parameters.
    Keeps a moving average of the model state_dict; use for validation and saving best.pt.
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = deepcopy(de_parallel(model)).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        if not self.enabled:
            return
        self.updates += 1
        d = self.decay(self.updates)
        msd = de_parallel(model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_((1 - d) * msd[k].detach())

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)


def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def intersect_dicts(da, db, exclude=()):
    """Intersecting keys with matching shapes. Exclude keys by name if needed."""
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def strip_optimizer(f='best.pt', s=''):
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':  # keys
        x[k] = None
    x['epoch'] = -1
    # Convert model weights to FP16 if it's a state dict
    if isinstance(x['model'], dict):
        for k, v in x['model'].items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                x['model'][k] = v.half()
    else:
        # If it's a model object, use the original approach
        x['model'].half()  # to FP16
        for p in x['model'].parameters():
            p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")

