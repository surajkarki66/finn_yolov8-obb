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


def strip_optimizer(
    f: Union[str, Path] = "best.pt",
    s: Union[str, Path] = "",
    half: bool = True,
) -> None:
    """
    Strip optimizer and extra state from a checkpoint to get actual model size.
    Keeps only model weights and nc; optionally saves in FP16.

    Args:
        f: Path to checkpoint (e.g. best.pt or last.pt).
        s: If set, save stripped checkpoint here; otherwise overwrite f.
        half: If True, save model weights in FP16 to reduce size.
    """
    f = Path(f)
    out = Path(s) if s else f
    try:
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(f, map_location="cpu")
    if "model" not in ckpt:
        print(f"Skipping {f}, no 'model' key.")
        return
    model_or_state = ckpt.get("ema") if ckpt.get("ema") is not None else ckpt["model"]
    if isinstance(model_or_state, nn.Module):
        model_or_state = model_or_state.module if hasattr(model_or_state, "module") else model_or_state
        state = model_or_state.state_dict()
        nc = getattr(model_or_state, "nc", ckpt.get("nc"))
    else:
        state = model_or_state
        nc = ckpt.get("nc")
    first = next(iter(state.values()), None)
    if half and first is not None and first.dtype != torch.float16:
        state = {k: v.half() for k, v in state.items()}
    stripped = {"model": state, "epoch": -1, "nc": nc}
    torch.save(stripped, out)
    mb = os.path.getsize(out) / 1e6
    print(f"Stripped optimizer from {f}{f' -> {out}' if s else ''}, {mb:.1f}MB")
