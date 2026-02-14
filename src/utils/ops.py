import numpy as np
import torch
import torchvision

from src.utils.metrics import batch_probiou


def xywh2xyxy(x):
    assert x.shape[-1] == 4
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
    dw = x[..., 2] / 2
    dh = x[..., 3] / 2
    y[..., 0] = x[..., 0] - dw
    y[..., 1] = x[..., 1] - dh
    y[..., 2] = x[..., 0] + dw
    y[..., 3] = x[..., 1] + dh
    return y


def clip_boxes(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0, shape[1])
        boxes[..., 1].clamp_(0, shape[0])
        boxes[..., 2].clamp_(0, shape[1])
        boxes[..., 3].clamp_(0, shape[0])
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])
    return boxes


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )
    else:
        gain = ratio_pad[0] if isinstance(ratio_pad[0], (int, float)) else ratio_pad[0][0]
        pad = ratio_pad[1]
    if padding:
        boxes[..., 0] -= pad[0]
        boxes[..., 1] -= pad[1]
        if not xywh:
            boxes[..., 2] -= pad[0]
            boxes[..., 3] -= pad[1]
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def nms_rotated(boxes, scores, threshold=0.45):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)
    sorted_idx = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes).triu_(diagonal=1)
    pick = torch.nonzero(ious.max(dim=0)[0] < threshold).squeeze_(-1)
    return sorted_idx[pick]


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    max_det=300,
    nc=0,
    rotated=True,
):
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 5)
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = prediction[:, 4:mi].amax(1) > conf_thres
    multi_label &= nc > 1
    prediction = prediction.transpose(-1, -2)
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        box, cls, mask = x.split((4, nc, nm), 1)
        conf, j = cls.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        n = x.shape[0]
        if not n:
            continue
        max_wh = 7680
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c
            i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        output[xi] = x[i]
    return output


def xywhr2xyxyxyxy(center):
    is_numpy = isinstance(center, np.ndarray)
    cos, sin = (np.cos, np.sin) if is_numpy else (torch.cos, torch.sin)
    ctr = center[..., :2]
    w, h, angle = (center[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = np.concatenate(vec1, axis=-1) if is_numpy else torch.cat(vec1, dim=-1)
    vec2 = np.concatenate(vec2, axis=-1) if is_numpy else torch.cat(vec2, dim=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return np.stack([pt1, pt2, pt3, pt4], axis=-2) if is_numpy else torch.stack([pt1, pt2, pt3, pt4], dim=-2)


def xyxyxyxy2xywhr(corners):
    import cv2
    is_torch = isinstance(corners, torch.Tensor)
    points = corners.cpu().numpy() if is_torch else corners
    points = points.reshape(len(corners), -1, 2)
    rboxes = []
    for pts in points:
        (x, y), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([x, y, w, h, angle / 180 * np.pi])
    rboxes = (
        torch.tensor(rboxes, device=corners.device, dtype=corners.dtype)
        if is_torch
        else np.asarray(rboxes, dtype=points.dtype)
    )
    return rboxes
