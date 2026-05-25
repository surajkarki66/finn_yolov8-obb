#!/usr/bin/env python3
"""
Evaluate exported QONNX/ONNX model on val or test split.

Uses the same dataset, letterboxing, probiou matching, and mAP metrics as val.py
(training-time validation via compute_validation_metrics). Post-processing for the
6-output FINN export (P3, P4, P5 + angle per scale) matches predict_qonnx.py.

Usage:
  python val_qonnx.py --model exported.onnx --data data/data.yaml --imgsz 416
  python val_qonnx.py --model runs/.../exported.onnx --data data/data.yaml --split test --conf 0.01 --save-json
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from torch.utils.data import DataLoader
from tqdm import tqdm

from predict_qonnx import REG_MAX, qonnx_outputs_to_prediction, run_qonnx_inference
from src.utils.dataset import OBBDataset
from src.utils.metrics import ap_per_class, batch_probiou, match_predictions_obb
from src.utils.ops import non_max_suppression, scale_boxes


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate exported QONNX OBB model")
    p.add_argument("--model", type=str, required=True, help="Path to exported .onnx (QONNX)")
    p.add_argument("--data", type=str, required=True, help="Data YAML (paths, nc, names)")
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=("val", "test"),
        help="Split to evaluate (default: val, same as training validation)",
    )
    p.add_argument("--batch", type=int, default=8, help="DataLoader batch size (inference runs per image)")
    p.add_argument("--imgsz", type=int, default=416, help="Input size; must match export --input_shape")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (try 0.01 for quant ONNX)")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--reg-max", type=int, default=REG_MAX, help="DFL reg_max (default 16)")
    p.add_argument("--project", type=str, default="runs/yolov8-obb")
    p.add_argument("--name", type=str, default="val_qonnx")
    p.add_argument("--save-json", action="store_true", help="Save metrics to project/name/metrics.json")
    return p.parse_args()


def load_data_yaml(path: str, split: str) -> dict:
    """Load data.yaml and return dict for OBBDataset (val key = chosen split paths)."""
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    root = Path(d.get("path", Path(path).parent))
    split_paths = d.get(split)
    if split_paths is None:
        raise KeyError(f"data.yaml must contain '{split}' key for evaluation")
    if isinstance(split_paths, str):
        split_paths = [str(root / split_paths)]
    elif isinstance(split_paths, list):
        split_paths = [str(root / x) for x in split_paths]
    names = d.get("names", {})
    if isinstance(names, list):
        names = dict(enumerate(names))
    nc = d.get("nc")
    if nc is None:
        nc = len(names) if names else 80
    return {"val": split_paths, "nc": int(nc), "names": names}


def load_qonnx_model(model_path: str) -> tuple[ModelWrapper, str]:
    model = ModelWrapper(str(model_path))
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    input_name = model.graph.input[0].name
    return model, input_name


def evaluate_batch_predictions(
    pred_list,
    batch,
    imgsz: int,
    iouv: torch.Tensor,
    all_tp,
    all_conf,
    all_pred_cls,
    all_target_cls,
):
    """Accumulate TP/conf/cls for one batch (same logic as val.py)."""
    imgsz_t = (imgsz, imgsz)
    bs = len(pred_list)
    for i in range(bs):
        ori_shape = batch["ori_shape"][i]
        ratio_pad = batch["ratio_pad"][i]
        idx = (batch["batch_idx"] == i).squeeze(-1)
        gt_cls = batch["cls"][idx].squeeze(-1)
        gt_bboxes = batch["bboxes"][idx]
        if gt_bboxes.numel():
            gt_bboxes = gt_bboxes.clone()
            gt_bboxes[:, :4] *= torch.tensor(
                [imgsz, imgsz, imgsz, imgsz], dtype=gt_bboxes.dtype
            )
            scale_boxes(imgsz_t, gt_bboxes[:, :4], ori_shape, ratio_pad=ratio_pad, xywh=True)
        gt_xywhr = torch.cat([gt_bboxes[:, :4], gt_bboxes[:, 4:5]], dim=-1) if gt_bboxes.numel() else gt_bboxes
        gt_cls_np = gt_cls.cpu().numpy().astype(int)
        all_target_cls.extend(gt_cls_np.tolist())

        det = pred_list[i]
        if len(det) == 0:
            continue
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


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    data = load_data_yaml(args.data, args.split)
    nc = data["nc"]
    reg_max = args.reg_max
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, input_name = load_qonnx_model(str(model_path))
    dataset = OBBDataset(img_path=data["val"], imgsz=args.imgsz, data=data, augment=False)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=False, collate_fn=OBBDataset.collate_fn
    )

    iouv = torch.linspace(0.5, 0.95, 10)
    all_tp, all_conf, all_pred_cls = [], [], []
    all_target_cls = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval ({args.split})", unit="batch"):
            im = batch["img"]
            bs = im.shape[0]
            pred_list = []
            for i in range(bs):
                img_np = im[i : i + 1].numpy().astype(np.float32)
                box_outputs, angle_outputs = run_qonnx_inference(
                    model, input_name, img_np, nc=nc, reg_max=reg_max
                )
                pred = qonnx_outputs_to_prediction(
                    box_outputs,
                    angle_outputs,
                    imgsz=args.imgsz,
                    nc=nc,
                    reg_max=reg_max,
                    device=device,
                )
                pred = non_max_suppression(
                    pred,
                    conf_thres=args.conf,
                    iou_thres=args.iou,
                    rotated=True,
                    nc=nc,
                )[0]
                pred_list.append(pred.cpu())

            evaluate_batch_predictions(
                pred_list,
                batch,
                args.imgsz,
                iouv,
                all_tp,
                all_conf,
                all_pred_cls,
                all_target_cls,
            )

    if not all_tp:
        print(
            "No predictions to evaluate: no detections passed the confidence threshold "
            f"(conf={args.conf}). Try a lower threshold, e.g. --conf 0.01"
        )
        return

    if len(all_target_cls) == 0:
        print(f"No ground truth labels in {args.split} set.")
        return

    all_tp = np.concatenate(all_tp, axis=0)
    all_conf = np.concatenate(all_conf, axis=0)
    all_pred_cls = np.concatenate(all_pred_cls, axis=0)
    all_target_cls = np.array(all_target_cls, dtype=int)

    precision, recall, mAP50, mAP50_95, _, _ = ap_per_class(
        all_tp, all_conf, all_pred_cls, all_target_cls
    )

    print("\n" + "=" * 60)
    print(f"OBB QONNX metrics ({args.split}, probiou)")
    print("=" * 60)
    print(f"  Model:        {model_path}")
    print(f"  Split:        {args.split}")
    print(f"  Images:       {len(dataset)}")
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
        metrics_dict = {
            "precision": float(precision),
            "recall": float(recall),
            "mAP50": float(mAP50),
            "mAP50-95": float(mAP50_95),
            "model": str(model_path),
            "data": str(args.data),
            "split": args.split,
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "reg_max": reg_max,
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
