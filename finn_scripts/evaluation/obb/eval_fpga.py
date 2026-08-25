#!/usr/bin/env python3


import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from postprocess import REG_MAX, decode_npz, load_scales
from metrics import ap_per_class, batch_probiou, match_predictions_obb


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate FPGA raw outputs (OBB) against ground truth")
    p.add_argument("--raw-dir", default="raw_outputs", help="Dir of *_raw.npz from yolov8-obb_fpga.py")
    p.add_argument("--scale-dir", default=".", help="Dir with mul_0..5.npy / add_0..5.npy")
    p.add_argument("--data", type=str, required=True, help="data.yaml (for nc / names; labels found via image path)")
    p.add_argument("--labels-dir", type=str, default="", help="Override: look up all labels here as <stem>.txt "
                                                                 "instead of mirroring the image path")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--nc", type=int, default=None, help="Defaults to nc from data.yaml")
    p.add_argument("--reg-max", type=int, default=REG_MAX)
    p.add_argument("--conf", type=float, default=0.001, help="Low threshold recommended for mAP sweep")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--project", type=str, default="runs/yolov8-obb")
    p.add_argument("--name", type=str, default="eval_fpga")
    p.add_argument("--save-json", action="store_true")
    return p.parse_args()


def load_data_yaml(path):
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    names = d.get("names", {})
    if isinstance(names, list):
        names = dict(enumerate(names))
    nc = d.get("nc")
    if nc is None:
        nc = len(names) if names else 1
    return int(nc), names


def find_label_path(image_path, labels_dir_override=""):
    image_path = str(image_path)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    if labels_dir_override:
        return os.path.join(labels_dir_override, f"{stem}.txt")
    if f"{os.sep}images{os.sep}" in image_path:
        label_path = image_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
    elif "/images/" in image_path:
        label_path = image_path.replace("/images/", "/labels/")
    else:
        label_path = image_path
    return os.path.splitext(label_path)[0] + ".txt"


def load_gt_obb(label_path, orig_shape):
    """Read Ultralytics OBB txt (class + 4 normalized corner points) and
    convert to pixel-space xywhr (radians), matching ultralytics'
    ops.xyxyxyxy2xywhr (cv2.minAreaRect based) convention.

    Returns (gt_cls (M,) int, gt_xywhr (M,5) float) in ORIGINAL image pixels.
    """
    h0, w0 = orig_shape[:2]
    if not os.path.exists(label_path):
        return np.zeros((0,), dtype=int), np.zeros((0, 5), dtype=np.float64)

    classes, polys = [], []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 9:
                continue
            cls = int(float(parts[0]))
            coords = np.array(parts[1:9], dtype=np.float64).reshape(4, 2)
            coords[:, 0] *= w0
            coords[:, 1] *= h0
            classes.append(cls)
            polys.append(coords)

    if not classes:
        return np.zeros((0,), dtype=int), np.zeros((0, 5), dtype=np.float64)

    rboxes = []
    for pts in polys:
        (cx, cy), (w, h), angle_deg = cv2.minAreaRect(pts.astype(np.float32))
        rboxes.append([cx, cy, w, h, angle_deg / 180.0 * np.pi])

    return np.array(classes, dtype=int), np.array(rboxes, dtype=np.float64)


def evaluate_image(det, gt_cls, gt_xywhr, iouv, all_tp, all_conf, all_pred_cls, all_target_cls):
    """det: (N,7) [cx,cy,w,h,conf,cls,angle] in original-image pixels (numpy).
    gt_cls: (M,), gt_xywhr: (M,5). iouv: (10,) torch tensor of IoU thresholds."""
    all_target_cls.extend(gt_cls.tolist())

    if len(det) == 0:
        return

    pred_cls = det[:, 5].astype(int)
    conf = det[:, 4].astype(np.float64)
    pred_xywhr = torch.as_tensor(
        np.concatenate([det[:, :4], det[:, 6:7]], axis=1), dtype=torch.float64
    )

    if gt_xywhr.shape[0] == 0:
        tp = np.zeros((len(det), len(iouv)), dtype=bool)
    else:
        gt_t = torch.as_tensor(gt_xywhr, dtype=torch.float64)
        iou_m = batch_probiou(pred_xywhr, gt_t)
        tp = match_predictions_obb(pred_cls, gt_cls, iou_m, iouv.cpu().numpy())

    all_tp.append(tp)
    all_conf.append(conf)
    all_pred_cls.append(pred_cls)


def main():
    args = parse_args()
    nc_yaml, names = load_data_yaml(args.data)
    nc = args.nc if args.nc is not None else nc_yaml

    npz_paths = sorted(glob.glob(os.path.join(args.raw_dir, "*_raw.npz")))
    if not npz_paths:
        raise FileNotFoundError(f"No *_raw.npz files found in {args.raw_dir}")
    print(f"Found {len(npz_paths)} raw output file(s) in {args.raw_dir}")

    muls, adds = load_scales(args.scale_dir)
    iouv = torch.linspace(0.5, 0.95, 10, dtype=torch.float64)

    all_tp, all_conf, all_pred_cls, all_target_cls = [], [], [], []
    n_images, n_missing_labels = 0, 0

    for npz_path in npz_paths:
        det, meta = decode_npz(
            npz_path, muls, adds, imgsz=args.imgsz, nc=nc, reg_max=args.reg_max,
            conf_thres=args.conf, iou_thres=args.iou,
        )
        label_path = find_label_path(meta["image_path"], args.labels_dir)
        if not os.path.exists(label_path):
            n_missing_labels += 1
        gt_cls, gt_xywhr = load_gt_obb(label_path, meta["orig_shape"])

        evaluate_image(det, gt_cls, gt_xywhr, iouv, all_tp, all_conf, all_pred_cls, all_target_cls)
        n_images += 1

    if n_missing_labels:
        print(f"WARNING: {n_missing_labels}/{n_images} images had no label file found "
              f"(treated as zero ground-truth instances). Check --labels-dir if unexpected.")

    if not all_tp:
        print(f"No predictions passed the confidence threshold (conf={args.conf}). "
              f"Try a lower --conf, e.g. 0.01.")
        return
    if len(all_target_cls) == 0:
        print("No ground-truth labels found across the evaluated images.")
        return

    all_tp = np.concatenate(all_tp, axis=0)
    all_conf = np.concatenate(all_conf, axis=0)
    all_pred_cls = np.concatenate(all_pred_cls, axis=0)
    all_target_cls = np.array(all_target_cls, dtype=int)

    precision, recall, mAP50, mAP50_95, _, _ = ap_per_class(
        all_tp, all_conf, all_pred_cls, all_target_cls
    )

    print("\n" + "=" * 60)
    print("OBB FPGA metrics (raw accelerator outputs, probiou)")
    print("=" * 60)
    print(f"  Raw dir:      {args.raw_dir}")
    print(f"  Images:       {n_images}")
    print(f"  imgsz:        {args.imgsz}")
    print(f"  conf / iou:   {args.conf} / {args.iou}")
    print(f"  Precision:    {precision:.4f}")
    print(f"  Recall:       {recall:.4f}")
    print(f"  mAP50:        {mAP50:.4f}")
    print(f"  mAP50-95:     {mAP50_95:.4f}")
    print("=" * 60)

    if args.save_json:
        save_dir = Path(args.project) / args.name
        save_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = save_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "precision": float(precision),
                "recall": float(recall),
                "mAP50": float(mAP50),
                "mAP50-95": float(mAP50_95),
                "raw_dir": str(args.raw_dir),
                "data": str(args.data),
                "images": n_images,
                "missing_labels": n_missing_labels,
                "conf": args.conf,
                "iou": args.iou,
                "imgsz": args.imgsz,
                "reg_max": args.reg_max,
            }, f, indent=2)
        print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()