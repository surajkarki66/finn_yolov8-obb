import numpy as np

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

def clip_gradients(model, max_norm=10.0):
    parameters = model.parameters()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


def set_params(model, decay):
    p1 = []
    p2 = []
    norm = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
    for m in model.modules():
        for n, p in m.named_parameters(recurse=0):
            if not p.requires_grad:
                continue
            if n == "bias":  # bias (no decay)
                p1.append(p)
            elif n == "weight" and isinstance(m, norm):  # norm-weight (no decay)
                p1.append(p)
            else:
                p2.append(p)  # weight (with decay)
    return [{'params': p1, 'weight_decay': 0.00},
            {'params': p2, 'weight_decay': decay}]
    
class LinearLR:
    def __init__(self, args, params, num_steps):
        max_lr = params['max_lr']
        min_lr = params['min_lr']

        total_steps = args.epochs * num_steps

        warmup_steps = int(max(params['warmup_epochs'] * num_steps, 100))
        # Prevent Warmup from exceeding Total Steps ---
        if warmup_steps >= total_steps:
            # If total training is shorter than defined warmup, 
            # we limit warmup to a fraction (e.g., 10%) of the total run
            # or just total_steps - 1 to prevent negative decay.
            print(f"Warning: Warmup steps ({warmup_steps}) > Total steps ({total_steps}). Adjusting warmup.")
            warmup_steps = int(total_steps * 0.1)

        # Ensure warmup is at least 0
        warmup_steps = max(warmup_steps, 0)
        
        decay_steps = int(args.epochs * num_steps - warmup_steps)
        print("warmup_steps: ", warmup_steps)
        print("decay_steps: ", decay_steps)

        warmup_lr = np.linspace(min_lr, max_lr, int(warmup_steps), endpoint=False)
        decay_lr = np.linspace(max_lr, min_lr, decay_steps)

        self.total_lr = np.concatenate((warmup_lr, decay_lr))

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            # Safety check to prevent index out of bounds if training runs long
            if step < len(self.total_lr):
                param_group['lr'] = self.total_lr[step]
            else:
                param_group['lr'] = self.total_lr[-1]