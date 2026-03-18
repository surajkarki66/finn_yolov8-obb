import math
import torch
import torch.nn as nn

from src.nn.block import DFL
from src.nn.conv import Conv
from src.utils.tal import dist2bbox, dist2rbox, make_anchors


class Detect(nn.Module):
    dynamic = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y, x

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)

    def decode_bboxes(self, bboxes):
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides


class OBB(Detect):
    def __init__(self, nc=80, ne=1, ch=()):
        super().__init__(nc, ch)
        self.ne = ne
        self.detect = Detect.forward
        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        bs = x[0].shape[0]
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)
        angle = (angle.sigmoid() - 0.25) * math.pi
        if not self.training:
            self.angle = angle
        x = self.detect(self, x)
        if self.training:
            return x, angle
        return torch.cat([x[0], angle], 1), (x[1], angle)

    def decode_bboxes(self, bboxes):
        return dist2rbox(self.dfl(bboxes), self.angle, self.anchors.unsqueeze(0), dim=1) * self.strides
