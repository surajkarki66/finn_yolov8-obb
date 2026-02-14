from typing import Any


import ast
import yaml
import re
import torch
import torch.nn as nn

from copy import deepcopy
from pathlib import Path
from brevitas.nn import QuantReLU
from src.models.quant_common import CommonUintActQuant

from src.nn.head import OBB
from src.utils.loss import v8OBBLoss
from src.nn.conv import Conv, Concat
from src.nn.block import C2f, SPPF
from src.models.quant import QuantConv, QuantC2f, QuantSPPF, QuantOBB
from src.utils.torch_utils import make_divisible, initialize_weights


def yaml_load(path):
    if isinstance(path, dict):
        return path
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def guess_model_scale(model_path):
    try:
        return re.search(r"yolov\d+([nslmx])", Path(model_path).stem).group(1)
    except Exception:
        return "n"


def parse_model(d, ch, verbose=True):
    max_channels = float("inf")
    nc = int(d.get("nc", 80))
    act = d.get("activation")
    scales = d.get("scales")
    depth, width = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple"))
    if scales:
        scale = d.get("scale") or tuple[Any, ...](scales.keys())[0]
        depth, width, max_channels = scales[scale]
    if act:
        Conv.default_act = eval(act)
    common_activations: dict[int, str] | None = d.get("common_activations")
    if "QuantC2f_common_act" in d:
        QuantC2f.use_common_act = d["QuantC2f_common_act"]
    activations_lookup: dict[str, QuantReLU] = {}
    if common_activations:
        bitwidth = common_activations.get("bitwidth", 8)
        for act_name in set[str](v for k, v in common_activations.items() if isinstance(k, int) and isinstance(v, str)):
            activations_lookup[act_name] = QuantReLU(
                act_quant=CommonUintActQuant,
                bit_width=bitwidth,
                scaling_per_channel=False,
                return_quant_tensor=False,
            )
    ch: list[int] = [ch]
    layers: list[nn.Module] = []
    save: list[int] = []
    c2 = ch[-1]
    for i, (f, n, m, args) in enumerate[Any](d["backbone"] + d["head"]):
        args = list[Any](args)
        for j, a in enumerate[Any](args):
            if isinstance(a, str) and a in ("nc", "ch", "depth", "width"):
                args[j] = locals().get(a, a)
            elif isinstance(a, str):
                try:
                    args[j] = ast.literal_eval(a)
                except (ValueError, SyntaxError):
                    pass
        if isinstance(m, str) and m.startswith("nn."):
            m = getattr(torch.nn, m[3:])
        else:
            orig_m_name = m
            m = globals().get[Any](m)
            if m is None and orig_m_name in ("QuantConv", "QuantC2f", "QuantSPPF", "QuantOBB"):
                raise RuntimeError(
                    "Quantized config is used but QAT modules failed to load. "
                    "Please check if the QAT modules are installed correctly."
                )
            elif m is None:
                raise RuntimeError(f"Unknown module name in config: {orig_m_name!r}")
        n = max(round(n * depth), 1) if n > 1 else n
        kwargs: dict[str, Any] = {}
        if common_activations and i in common_activations:
            kwargs["act"] = activations_lookup[common_activations[i]]
        if m in (Conv,):
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        elif m is QuantConv:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            wb = args[3] if len(args) > 4 else 8
            ab = args[4] if len(args) > 4 else 8
            args = [c1, c2, args[1], args[2], wb, ab]
        elif m is C2f:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            args.insert(2, n)
            n = 1
        elif m is QuantC2f:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            wb = args[2] if len(args) > 3 else 8
            ab = args[3] if len(args) > 3 else 8
            args = [c1, c2, n, args[1], wb, ab]
            n = 1
        elif m is SPPF:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        elif m is QuantSPPF:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(min(c2, max_channels) * width, 8)
            wb = args[2] if len(args) > 3 else 8
            ab = args[3] if len(args) > 3 else 8
            args = [c1, c2, args[1], wb, ab]
        elif m is nn.Upsample:
            c2 = ch[f]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is OBB:
            args.append([ch[x] for x in f])
        elif m is QuantOBB:
            wb = args[2] if len(args) > 3 else 8
            ab = args[3] if len(args) > 3 else 8
            args = [args[0], args[1], wb, ab]
            args.append([ch[x] for x in f])
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*(m(*args, **kwargs) for _ in range(n))) if n > 1 else m(*args, **kwargs)
        t = str(m)[8:-2].replace("__main__.", "")
        m.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class BaseModel(nn.Module):
    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False):
        return self._predict_once(x)

    def _predict_once(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def loss(self, batch, preds=None):
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        raise NotImplementedError


class DetectionModel(BaseModel):
    def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_load(cfg)
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        if nc is not None and nc != self.yaml.get("nc"):
            self.yaml["nc"] = int(nc)
        if "scale" not in self.yaml and "scales" in self.yaml:
            self.yaml["scale"] = guess_model_scale(cfg) if isinstance(cfg, (str, Path)) else "n"
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.names = {i: str(i) for i in range(self.yaml["nc"])}
        self.inplace = self.yaml.get("inplace", True)
        m = self.model[-1]
        if isinstance(m, OBB) or (isinstance(m, QuantOBB)):
            s = 256
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0]
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])
            self.stride = m.stride
            m.bias_init()
        else:
            self.stride = torch.tensor([32])
        initialize_weights(self)
        if verbose:
            n = sum(x.numel() for x in self.parameters())
            print(f"Model: {n:,} parameters")

    def init_criterion(self):
        raise NotImplementedError("Use OBBModel for OBB training.")


class OBBModel(DetectionModel):
    def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        return v8OBBLoss(self)


