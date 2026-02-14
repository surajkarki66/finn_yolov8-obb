import math
import torch
import numpy as np

def _get_covariance_matrix(boxes):
    gbbs = torch.cat((torch.pow(boxes[:, 2:4], 2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    return (
        a * torch.cos(c) ** 2 + b * torch.sin(c) ** 2,
        a * torch.sin(c) ** 2 + b * torch.cos(c) ** 2,
        a * torch.cos(c) * torch.sin(c) - b * torch.sin(c) * torch.cos(c),
    )


def probiou(obb1, obb2, CIoU=False, eps=1e-7):
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)
    t1 = (
        ((a1 + a2) * (torch.pow(y1 - y2, 2)) + (b1 + b2) * (torch.pow(x1 - x2, 2)))
        / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)) * 0.5
    t3 = (
        torch.log(
            ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)))
            / (4 * torch.sqrt((a1 * b1 - torch.pow(c1, 2)).clamp_(0) * (a2 * b2 - torch.pow(c2, 2)).clamp_(0)) + eps)
            + eps
        )
        * 0.5
    )
    bd = t1 + t2 + t3
    bd = torch.clamp(bd, eps, 100.0)
    hd = torch.sqrt(1.0 - torch.exp(-bd) + eps)
    iou = 1 - hd
    if CIoU:
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha
    return iou


def batch_probiou(obb1, obb2, eps=1e-7):
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))
    t1 = (
        ((a1 + a2) * (torch.pow(y1 - y2, 2)) + (b1 + b2) * (torch.pow(x1 - x2, 2)))
        / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)) * 0.5
    t3 = (
        torch.log(
            ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)))
            / (4 * torch.sqrt((a1 * b1 - torch.pow(c1, 2)).clamp_(0) * (a2 * b2 - torch.pow(c2, 2)).clamp_(0)) + eps)
            + eps
        )
        * 0.5
    )
    bd = t1 + t2 + t3
    bd = torch.clamp(bd, eps, 100.0)
    hd = torch.sqrt(1.0 - torch.exp(-bd) + eps)
    return 1 - hd


def match_predictions_obb(pred_cls, true_classes, iou_matrix, iouv):
    """
    Match predictions to GT using OBB IoU matrix.
    pred_cls (N,), true_classes (M,), iou_matrix (N, M), iouv (10,) IoU thresholds.
    Returns correct (N, 10) bool.
    """
    pred_cls = np.asarray(pred_cls)
    true_classes = np.asarray(true_classes)
    correct = np.zeros((len(pred_cls), len(iouv)), dtype=bool)
    # correct_class (N, M): 1 if pred_cls[n] == true_classes[m]
    correct_class = (pred_cls[:, None] == true_classes[None, :])
    if hasattr(iou_matrix, "cpu"):
        iou = (iou_matrix * torch.tensor(correct_class, dtype=iou_matrix.dtype, device=iou_matrix.device)).cpu().numpy()
    else:
        iou = np.asarray(iou_matrix) * correct_class
    for i, threshold in enumerate(iouv):
        matches = np.nonzero(iou >= threshold)
        matches = np.array(matches).T
        if matches.shape[0] > 0:
            if matches.shape[0] > 1:
                iou_vals = iou[matches[:, 0], matches[:, 1]]
                matches = matches[iou_vals.argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 0].astype(int), i] = True
    return correct


def compute_ap(recall, precision):
    """AP from recall-precision curve (area under curve)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)
    return ap


def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = max(1, round(len(y) * f * 2) // 2 * 2 + 1)
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    tp (N, 10) bool, conf (N,), pred_cls (N,), target_cls (M,) - all numpy.
    Returns: p, r, ap50, ap50_95 (scalars), and per-class arrays.
    """
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = len(unique_classes)
    ap = np.zeros((nc, tp.shape[1]))
    p_curve = np.zeros((nc, 1000))
    r_curve = np.zeros((nc, 1000))
    x = np.linspace(0, 1, 1000)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]
        n_p = i.sum()
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        recall = tpc / (n_l + eps)
        precision = tpc / (tpc + fpc + eps)
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    idx = smooth(f1_curve.mean(0), 0.1).argmax()
    p = p_curve[:, idx].mean()
    r = r_curve[:, idx].mean()
    ap50 = ap[:, 0].mean() if ap.size else 0.0
    ap50_95 = ap.mean() if ap.size else 0.0
    return p, r, ap50, ap50_95, ap, unique_classes
