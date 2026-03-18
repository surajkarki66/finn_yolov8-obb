import math
from typing import Any
import torch
import torch.nn as nn

from brevitas.nn import QuantConv2d, QuantReLU
 
from src.models.quant_common import CommonUintActQuant, CommonIntWeightPerChannelQuant

from src.nn.block import DFL
from src.nn.conv import autopad
from src.utils.tal import dist2rbox, make_anchors


def _act(act_bit_width, out_channels, act=True):
    if act is True or not isinstance(act, nn.Module):
        return QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=act_bit_width,
            per_channel_broadcastable_shape=(1, out_channels, 1, 1),
            scaling_per_channel=True,
            return_quant_tensor=False,
        )
    return act


class QuantConv(nn.Module):
    """Quantized Conv: QuantConv2d + BN + optional QuantReLU (or shared common_act)."""

    def __init__(
        self,
        c1,
        c2,
        k=1,
        s=1,
        weight_bit_width=8,
        act_bit_width=8,
        g=1,
        act=True,
    ):
        super().__init__()
        p = autopad(k)
        self.conv = QuantConv2d(
            c1,
            c2,
            k,
            stride=s,
            padding=p,
            groups=g,
            bias=False,
            weight_quant=CommonIntWeightPerChannelQuant,
            weight_bit_width=weight_bit_width,
        )
        self.bn = nn.BatchNorm2d(c2, eps=1e-5)
        self.act = act if isinstance(act, nn.Module) else _act(act_bit_width, c2, act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class QuantBottleneck(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        shortcut=True,
        weight_bit_width=8,
        act_bit_width=8,
        g=1,
        k=(3, 3),
        e=0.5,
        cv2_act=True,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = QuantConv(c1, c_, k[0], 1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)
        self.cv2 = QuantConv(
            c_,
            c2,
            k[1],
            1,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            act=cv2_act,
            g=g,
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class QuantC2f(nn.Module):
    """Quantized C2f with optional common activation (shared QuantReLU by name)."""

    use_common_act = False

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        weight_bit_width=8,
        act_bit_width=8,
        g=1,
        e=0.5,
        act=True,
    ):
        super().__init__()
        if self.use_common_act:
            self.common_act = QuantReLU(
                act_quant=CommonUintActQuant,
                bit_width=act_bit_width,
                scaling_per_channel=True,
                return_quant_tensor=False,
            )
        else:
            self.common_act = True
        self.c = int(c2 * e)
        self.cv1 = QuantConv(
            c1,
            2 * self.c,
            1,
            1,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            act=self.common_act,
        )
        self.cv2 = QuantConv(
            (2 + n) * self.c,
            c2,
            1,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            act=act,
        )
        self.m = nn.ModuleList(
            QuantBottleneck(
                self.c,
                self.c,
                shortcut=shortcut,
                weight_bit_width=weight_bit_width,
                act_bit_width=act_bit_width,
                cv2_act=self.common_act,
                g=g,
                k=(3, 3),
                e=1.0,
            )
            for _ in range(n)
        )

    def forward(self, x):
        y = list[Any](self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Use split instead of chunk for hardware compatibility."""
        y = list[Any](self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class QuantSPPF(nn.Module):
    def __init__(self, c1, c2, k=5, weight_bit_width=8, act_bit_width=8, act=True):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = QuantConv(c1, c_, 1, 1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)
        self.cv2 = QuantConv(
            c_ * 4,
            c2,
            1,
            1,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            act=act,
        )
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class QuantOBB(nn.Module):
    """Quantized OBB head: box (cv2), class (cv3), angle (cv4); same forward contract as OBB for v8OBBLoss."""

    finn_export = False
    finn_export_angle_raw = True
    dynamic = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ne=1, weight_bit_width=8, act_bit_width=8, ch=()):
        super().__init__()
        self.nc = nc
        self.ne = ne
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2 = max(16, ch[0] // 4, self.reg_max * 4)
        c3 = max(ch[0], min(self.nc, 100))
        c4 = max(ch[0] // 4, self.ne)

        self.cv2 = nn.ModuleList(
            nn.Sequential(
                QuantConv(x, c2, 3, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width),
                QuantConv(c2, c2, 3, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width),
                QuantConv2d(
                    c2,
                    4 * self.reg_max,
                    1,
                    weight_quant=CommonIntWeightPerChannelQuant,
                    weight_bit_width=weight_bit_width,
                ),
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                QuantConv(x, c3, 3, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width),
                QuantConv(c3, c3, 3, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width),
                QuantConv2d(
                    c3,
                    self.nc,
                    1,
                    weight_quant=CommonIntWeightPerChannelQuant,
                    weight_bit_width=weight_bit_width,
                ),
            )
            for x in ch
        )
        self.cv4 = nn.ModuleList(
            nn.Sequential(
                QuantConv(x, c4, 3, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width),
                QuantConv(c4, c4, 3, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width),
                QuantConv2d(
                    c4,
                    self.ne,
                    1,
                    weight_quant=CommonIntWeightPerChannelQuant,
                    weight_bit_width=weight_bit_width,
                ),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        bs = x[0].shape[0]
        # Raw angle logits per scale
        angle_raw = [self.cv4[i](x[i]) for i in range(self.nl)]
        angle = torch.cat([a.view(bs, self.ne, -1) for a in angle_raw], 2)
        angle = (angle.sigmoid() - 0.25) * math.pi
        if not self.training:
            self.angle = angle
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x, angle
        if self.finn_export:
            if self.finn_export_angle_raw:
                return (x[0], x[1], x[2], angle_raw[0], angle_raw[1], angle_raw[2])
            return (x[0], x[1], x[2], angle)
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return torch.cat([y, angle], 1), (x, angle)

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)

    def decode_bboxes(self, bboxes):
        return dist2rbox(self.dfl(bboxes), self.angle, self.anchors.unsqueeze(0), dim=1) * self.strides
