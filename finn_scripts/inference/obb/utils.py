import numpy as np


# -----------------------------------------------------------------
# Box scaling / clipping
# -----------------------------------------------------------------
def clip_boxes(boxes, shape):
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


# -----------------------------------------------------------------
# Rotated box corner conversion
# -----------------------------------------------------------------
def xywhr2xyxyxyxy(center):
    """center: (..., 5) array [cx, cy, w, h, angle(rad)] -> (..., 4, 2) corners."""
    center = np.asarray(center, dtype=np.float64)
    ctr = center[..., :2]
    w = center[..., 2:3]
    h = center[..., 3:4]
    angle = center[..., 4:5]
    cos_value, sin_value = np.cos(angle), np.sin(angle)
    vec1 = np.concatenate([w / 2 * cos_value, w / 2 * sin_value], axis=-1)
    vec2 = np.concatenate([-h / 2 * sin_value, h / 2 * cos_value], axis=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return np.stack([pt1, pt2, pt3, pt4], axis=-2)


# -----------------------------------------------------------------
# Rotated IoU (ProbIoU)
# -----------------------------------------------------------------
def _get_covariance_matrix(boxes):
    """boxes: (N, 5) xywhr -> (a, b, c) each shape (N,)."""
    boxes = np.asarray(boxes, dtype=np.float64)
    gbbs = np.concatenate([boxes[:, 2:4] ** 2 / 12.0, boxes[:, 4:5]], axis=-1)
    a, b, c = gbbs[:, 0], gbbs[:, 1], gbbs[:, 2]
    cos = np.cos(c)
    sin = np.sin(c)
    cos2 = cos ** 2
    sin2 = sin ** 2
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def batch_probiou(obb1, obb2, eps=1e-7):
    """obb1: (N,5), obb2: (M,5) xywhr -> IoU matrix (N,M)."""
    obb1 = np.asarray(obb1, dtype=np.float64)
    obb2 = np.asarray(obb2, dtype=np.float64)

    x1 = obb1[:, 0:1]
    y1 = obb1[:, 1:2]
    x2 = obb2[:, 0][None, :]
    y2 = obb2[:, 1][None, :]

    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)
    a1, b1, c1 = a1[:, None], b1[:, None], c1[:, None]
    a2, b2, c2 = a2[None, :], b2[None, :], c2[None, :]

    t1 = (
        ((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2)
        / ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)) * 0.5
    denom = 4 * (np.clip(a1 * b1 - c1 ** 2, 0, None) * np.clip(a2 * b2 - c2 ** 2, 0, None)) ** 0.5 + eps
    t3 = np.log(((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2) / denom + eps) * 0.5

    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(np.clip(1.0 - np.exp(-bd) + eps, 0, None))
    return 1 - hd


def nms_rotated(boxes, scores, threshold=0.45):
    """boxes: (N,5) xywhr, scores: (N,) -> kept indices (into the input order)."""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    sorted_idx = np.argsort(-scores)
    boxes_sorted = boxes[sorted_idx]
    ious = batch_probiou(boxes_sorted, boxes_sorted)
    ious = np.triu(ious, k=1)
    keep_mask = ious.max(axis=0) < threshold
    pick = np.nonzero(keep_mask)[0]
    return sorted_idx[pick]


# -----------------------------------------------------------------
# Rotated NMS post-processing
# -----------------------------------------------------------------
def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    max_det=300,
    nc=0,
):
    """
    prediction: (bs, 4+nc+nm, N) numpy array, nm=1 for OBB (angle channel).
    Returns a list (len bs) of (M, 6+nm) arrays: [x, y, w, h, conf, cls, mask...]
    """
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    prediction = np.asarray(prediction, dtype=np.float64)

    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 5)
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = prediction[:, 4:mi, :].max(axis=1) > conf_thres

    prediction = prediction.transpose(0, 2, 1)  # (bs, N, 4+nc+nm)
    output = [np.zeros((0, 6 + nm)) for _ in range(bs)]

    for xi in range(bs):
        x = prediction[xi][xc[xi]]
        if x.shape[0] == 0:
            continue

        box = x[:, :4]
        cls = x[:, 4:4 + nc]
        mask = x[:, 4 + nc:]

        j = cls.argmax(axis=1, keepdims=True)
        conf = np.take_along_axis(cls, j, axis=1)
        x = np.concatenate([box, conf, j.astype(np.float64), mask], axis=1)
        x = x[conf.reshape(-1) > conf_thres]

        if classes is not None:
            x = x[np.isin(x[:, 5], classes)]

        n = x.shape[0]
        if n == 0:
            continue

        max_wh = 7680
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]
        # rotated boxes: [x+c, y+c, w, h, angle]
        boxes = np.concatenate([x[:, :2] + c, x[:, 2:4], x[:, -1:]], axis=-1)
        i = nms_rotated(boxes, scores, iou_thres)
        i = i[:max_det]

        output[xi] = x[i]

    return output


# -----------------------------------------------------------------
# Anchor generation + rotated distance decode
# -----------------------------------------------------------------
def make_anchors(feats, strides, grid_cell_offset=0.5):
    """feats: list of (B, C, H, W) numpy arrays. Returns anchor_points (N,2), stride_tensor (N,1)."""
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = np.arange(w, dtype=np.float64) + grid_cell_offset
        sy = np.arange(h, dtype=np.float64) + grid_cell_offset
        sy, sx = np.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(np.stack((sx, sy), -1).reshape(-1, 2))
        stride_tensor.append(np.full((h * w, 1), stride, dtype=np.float64))
    return np.concatenate(anchor_points, axis=0), np.concatenate(stride_tensor, axis=0)


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=1):
    """
    pred_dist: (B, 4, N)   [l, t, r, b] distances
    pred_angle: (B, 1, N) or (B, N)
    anchor_points: (B, N, 2) or (1, N, 2)
    Returns (B, 4, N): [cx, cy, w, h]-style rotated box (dist2rbox convention).
    """
    lt, rb = np.split(pred_dist, 2, axis=dim)
    cos = np.cos(pred_angle)
    sin = np.sin(pred_angle)
    xf, yf = np.split((rb - lt) / 2.0, 2, axis=dim)
    x = xf * cos - yf * sin
    y = xf * sin + yf * cos
    xy = np.concatenate([x, y], axis=dim)

    if dim == 1 and anchor_points.ndim == 3 and anchor_points.shape[1] != xy.shape[1]:
        anchor_points = anchor_points.transpose(0, 2, 1)  # (1, N, 2) -> (1, 2, N)

    xy = xy + anchor_points
    return np.concatenate([xy, lt + rb], axis=dim)


# -----------------------------------------------------------------
# DFL decode
# -----------------------------------------------------------------
def dfl_decode(box_raw: np.ndarray, reg_max: int = 16) -> np.ndarray:
    """(B, reg_max*4, N) -> (B, 4, N) in distance space, via softmax-weighted sum."""
    b, c, n = box_raw.shape
    assert c == reg_max * 4
    x = box_raw.reshape(b, 4, reg_max, n).astype(np.float64)
    x = x - x.max(axis=2, keepdims=True)  # numerical stability
    e = np.exp(x)
    softmax = e / e.sum(axis=2, keepdims=True)
    proj = np.arange(reg_max, dtype=np.float64).reshape(1, 1, -1, 1)
    return (softmax * proj).sum(axis=2)