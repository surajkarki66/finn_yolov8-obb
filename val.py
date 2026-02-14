#!/usr/bin/env python3
"""
Validate YOLOv8-OBB
Computes: Precision, Recall, mAP50, mAP50-95 using probiou for OBB.

Usage:
  python val.py --weights runs/obb/train/last.pt --data path/to/data.yaml
"""

import argparse
import json
import torch
import numpy as np

from typing import Any
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader


from src.models.yolo import OBBModel
from src.utils.dataset import OBBDataset
from src.utils.metrics import batch_probiou
from src.utils.metrics import ap_per_class, match_predictions_obb
from src.utils.ops import non_max_suppression, scale_boxes

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--cfg", type=str, default="configs/models/yolov8-obb.yaml", help="Model config")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--project", type=str, default="runs/yolov8-obb")
    p.add_argument("--name", type=str, default="val")
    p.add_argument("--save-json", action="store_true", help="Save metrics to project/name/metrics.json")
    return p.parse_args()


def load_data_yaml(path):
    import yaml
    with open(path) as f:
        d = yaml.safe_load(f)
    root = Path(d.get("path", Path(path).parent))
    val = d.get("val")
    if val is None:
        raise KeyError("data.yaml must contain 'val' key (validation path)")
    if isinstance(val, str):
        val = [str(root / val)]
    elif isinstance(val, list):
        val = [str(root / x) for x in val]
    names = d.get("names", {})
    if isinstance(names, list):
        names = dict[int, Any](enumerate[Any](names))
    nc = d.get("nc")
    if nc is None:
        nc = len(names) if names else 80
    return {"val": val, "nc": int(nc), "names": names}


def compute_validation_metrics(model, data, device, imgsz=640, conf=0.25, iou=0.45, batch_size=8, use_tqdm=False):
    """
    Run OBB validation and return metrics dict (mAP50, mAP50_95, precision, recall).
    Returns None if no val data or no predictions/targets.
    """
    if not data.get("val"):
        return None
    model.eval()
    dataset = OBBDataset(img_path=data["val"], imgsz=imgsz, data=data, augment=False)
    loader = DataLoader[Any](
        dataset, batch_size=batch_size, shuffle=False, collate_fn=OBBDataset.collate_fn
    )
    iouv = torch.linspace(0.5, 0.95, 10)
    all_tp, all_conf, all_pred_cls = [], [], []
    all_target_cls = []

    batch_iter = tqdm[Any](loader, desc="Val", unit="batch") if use_tqdm else loader
    with torch.no_grad():
        for batch in batch_iter:
            im = batch["img"].to(device)
            pred = model(im)[0]
            pred = non_max_suppression(
                pred, conf_thres=conf, iou_thres=iou, rotated=True, nc=data["nc"]
            )
            imgsz_t = (imgsz, imgsz)
            bs = im.shape[0]
            for i in range(bs):
                ori_shape = batch["ori_shape"][i]
                ratio_pad = batch["ratio_pad"][i]
                idx = (batch["batch_idx"] == i).squeeze(-1)
                gt_cls = batch["cls"][idx].squeeze(-1)
                gt_bboxes = batch["bboxes"][idx].to(device)
                if gt_bboxes.numel():
                    gt_bboxes = gt_bboxes.clone()
                    gt_bboxes[:, :4] *= torch.tensor(
                        [imgsz, imgsz, imgsz, imgsz], device=gt_bboxes.device, dtype=gt_bboxes.dtype
                    )
                    scale_boxes(imgsz_t, gt_bboxes[:, :4], ori_shape, ratio_pad=ratio_pad, xywh=True)
                gt_xywhr = torch.cat([gt_bboxes[:, :4], gt_bboxes[:, 4:5]], dim=-1)
                gt_cls_np = gt_cls.cpu().numpy().astype(int)
                all_target_cls.extend(gt_cls_np.tolist())
                det = pred[i]
                if len(det) == 0:
                    continue
                det = det.to(device)
                det[:, :4] = scale_boxes(imgsz_t, det[:, :4], ori_shape, ratio_pad=ratio_pad, xywh=True)
                pred_xywhr = torch.cat([det[:, :4], det[:, -1:]], dim=-1)
                pred_cls = det[:, 5].detach().cpu().numpy().astype(int)
                conf_np = det[:, 4].detach().cpu().numpy()
                if gt_xywhr.shape[0] == 0:
                    tp = np.zeros((len(det), len(iouv)), dtype=bool)
                else:
                    iou_m = batch_probiou(pred_xywhr, gt_xywhr)
                    if iou_m.dim() == 1:
                        iou_m = iou_m.unsqueeze(1)
                    tp = match_predictions_obb(pred_cls, gt_cls_np, iou_m, iouv.cpu().numpy())
                all_tp.append(tp)
                all_conf.append(conf_np)
                all_pred_cls.append(pred_cls)

    if not all_tp or len(all_target_cls) == 0:
        return None
    all_tp = np.concatenate(all_tp, axis=0)
    all_conf = np.concatenate(all_conf, axis=0)
    all_pred_cls = np.concatenate(all_pred_cls, axis=0)
    all_target_cls = np.array(all_target_cls, dtype=int)
    precision, recall, mAP50, mAP50_95, _, _ = ap_per_class(
        all_tp, all_conf, all_pred_cls, all_target_cls
    )
    return {"precision": precision, "recall": recall, "mAP50": mAP50, "mAP50_95": mAP50_95}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data_yaml(args.data)
    model = OBBModel(args.cfg, ch=3, nc=data["nc"], verbose=False)
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model = model.to(device).eval()

    dataset = OBBDataset(img_path=data["val"], imgsz=args.imgsz, data=data, augment=False)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=False, collate_fn=OBBDataset.collate_fn
    )

    # IoU thresholds for mAP@0.5 and mAP@0.5:0.95
    iouv = torch.linspace(0.5, 0.95, 10)
    all_tp, all_conf, all_pred_cls = [], [], []
    all_target_cls = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", unit="batch"):
            im = batch["img"].to(device)
            pred = model(im)[0]
            pred = non_max_suppression(
                pred, conf_thres=args.conf, iou_thres=args.iou, rotated=True, nc=data["nc"]
            )
            imgsz = (args.imgsz, args.imgsz)
            bs = im.shape[0]

            for i in range(bs):
                ori_shape = batch["ori_shape"][i]
                ratio_pad = batch["ratio_pad"][i]
                idx = (batch["batch_idx"] == i).squeeze(-1)
                gt_cls = batch["cls"][idx].squeeze(-1)
                gt_bboxes = batch["bboxes"][idx].to(device)
                if gt_bboxes.numel():
                    gt_bboxes = gt_bboxes.clone()
                    gt_bboxes[:, :4] *= torch.tensor(
                        [args.imgsz, args.imgsz, args.imgsz, args.imgsz],
                        device=gt_bboxes.device, dtype=gt_bboxes.dtype
                    )
                    scale_boxes(imgsz, gt_bboxes[:, :4], ori_shape, ratio_pad=ratio_pad, xywh=True)
                gt_xywhr = torch.cat([gt_bboxes[:, :4], gt_bboxes[:, 4:5]], dim=-1)
                gt_cls_np = gt_cls.cpu().numpy().astype(int)
                all_target_cls.extend(gt_cls_np.tolist())

                det = pred[i]
                if len(det) == 0:
                    continue
                det = det.to(device)
                det[:, :4] = scale_boxes(
                    imgsz, det[:, :4], ori_shape, ratio_pad=ratio_pad, xywh=True
                )
                pred_xywhr = torch.cat([det[:, :4], det[:, -1:]], dim=-1)
                pred_cls = det[:, 5].detach().cpu().numpy().astype(int)
                conf = det[:, 4].detach().cpu().numpy()

                if gt_xywhr.shape[0] == 0:
                    tp = np.zeros((len(det), len(iouv)), dtype=bool)
                else:
                    iou = batch_probiou(pred_xywhr, gt_xywhr)
                    if iou.dim() == 1:
                        iou = iou.unsqueeze(1)
                    tp = match_predictions_obb(
                        pred_cls, gt_cls_np, iou, iouv.cpu().numpy()
                    )

                all_tp.append(tp)
                all_conf.append(conf)
                all_pred_cls.append(pred_cls)

        if not all_tp:
            print("No predictions to evaluate.")
            return

    all_tp = np.concatenate(all_tp, axis=0)
    all_conf = np.concatenate(all_conf, axis=0)
    all_pred_cls = np.concatenate(all_pred_cls, axis=0)
    all_target_cls = np.array(all_target_cls, dtype=int)

    if len(all_target_cls) == 0:
        print("No ground truth labels in validation set.")
        return

    precision, recall, mAP50, mAP50_95, _, _ = ap_per_class(
        all_tp, all_conf, all_pred_cls, all_target_cls
    )

    print("\n" + "=" * 60)
    print("OBB Validation metrics (probiou)")
    print("=" * 60)
    print(f"  Precision:    {precision:.4f}")
    print(f"  Recall:       {recall:.4f}")
    print(f"  mAP50:        {mAP50:.4f}")
    print(f"  mAP50-95:     {mAP50_95:.4f}")
    print("=" * 60)

    if args.save_json:
        save_dir = Path(args.project) / args.name
        save_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = save_dir / "metrics.json"
        metrics_dict = {
            "precision": float(precision),
            "recall": float(recall),
            "mAP50": float(mAP50),
            "mAP50-95": float(mAP50_95),
            "weights": str(args.weights),
            "data": str(args.data),
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
