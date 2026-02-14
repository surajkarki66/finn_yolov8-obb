#!/usr/bin/env python3
"""
Run YOLOv8-OBB inference and annotate images.

Usage:
  python predict.py --weights runs/obb/train/best.pt --source image.jpg --save
  python predict.py --weights runs/obb/train/best.pt --source dir/of/images --save --data obb_data/data.yaml
"""

import argparse
import cv2
import numpy as np
import torch

from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.models.yolo import OBBModel
from src.utils.ops import non_max_suppression, scale_boxes, xywhr2xyxyxyxy
from src.utils.torch_utils import intersect_dicts

# Color palette (BGR for cv2)
HEX_PALETTE = (
    "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86",
    "1A9334", "00D4BB", "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF",
    "520085", "CB38FF", "FF95C8", "FF37C7",
)


def hex2bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


COLORS = [hex2bgr(f"#{c}") for c in HEX_PALETTE]


def get_names(data_path, nc):
    """Load class names from data.yaml if path given; else 0,1,...,nc-1."""
    if not data_path or not Path(data_path).exists():
        return {i: str(i) for i in range(nc)}
    import yaml
    with open(data_path) as f:
        d = yaml.safe_load(f) or {}
    names = d.get("names", {})
    if isinstance(names, list):
        return dict[int, Any](enumerate[Any](names))
    return names if names else {i: str(i) for i in range(nc)}


def draw_obb_label(img, box_xyxyxyxy, label, color, line_thickness=2, font_scale=0.5, font_thickness=1):
    """Draw rotated box (4 corners) and label with filled background."""
    box = np.asarray(box_xyxyxyxy, dtype=np.int32)
    cv2.polylines(img, [box], True, color, line_thickness, cv2.LINE_AA)
    if label:
        p1 = (int(box[0][0]), int(box[0][1]))
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        outside = p1[1] - h >= 3
        p2 = (p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3)
        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(
            img, label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True, help="Model weights (e.g. best.pt)")
    p.add_argument("--source", type=str, default=".", help="Image file or directory")
    p.add_argument("--data", type=str, default="", help="Data YAML for class names (optional)")
    p.add_argument("--cfg", type=str, default="configs/models/yolov8-obb.yaml")
    p.add_argument("--nc", type=int, default=80, help="Number of classes (if not in checkpoint)")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--save", action="store_true", help="Save annotated images")
    p.add_argument("--show", action="store_true", help="Show result in window")
    p.add_argument("--project", type=str, default="runs/yolov8-obb")
    p.add_argument("--name", type=str, default="predict")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.weights, map_location="cpu")
    nc = ckpt.get("nc", args.nc)
    names = get_names(args.data, nc)

    model = OBBModel(args.cfg, ch=3, nc=nc, verbose=False)
    raw = ckpt.get("model", ckpt)
    state = raw.float().state_dict() if hasattr(raw, "state_dict") else raw
    state = intersect_dicts(state, model.state_dict())
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    source = Path(args.source)
    if source.is_dir():
        files = list[Path](source.rglob("*.*"))
        files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
    else:
        files = [source] if source.exists() else []

    save_dir = Path(args.project) / args.name
    if args.save:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {save_dir}")

    for path in tqdm(files, desc="Predicting"):
        img = cv2.imread(str(path))
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        r = min(args.imgsz / h0, args.imgsz / w0)
        new_h, new_w = int(round(h0 * r)), int(round(w0 * r))
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        dh = (args.imgsz - new_h) / 2
        dw = (args.imgsz - new_w) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_in = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img_in = np.ascontiguousarray(img_in.transpose(2, 0, 1)[::-1])
        img_in = torch.from_numpy(img_in).float().unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            pred = model(img_in)[0]
        pred = non_max_suppression(pred, conf_thres=args.conf, iou_thres=args.iou, rotated=True, nc=nc)[0]

        if len(pred):
            pred[:, :4] = scale_boxes(
                (args.imgsz, args.imgsz), pred[:, :4], (h0, w0),
                ratio_pad=((r,), (dw, dh)), xywh=True,
            )
            obb = torch.cat([pred[:, :4], pred[:, -1:], pred[:, 4:6]], dim=-1)
            for *xywh, r_angle, conf, cls in obb.tolist():
                corners = xywhr2xyxyxyxy(torch.tensor([[xywh[0], xywh[1], xywh[2], xywh[3], r_angle]]))
                box_xyxyxyxy = corners[0].numpy()
                c = int(cls)
                name = names.get(c, str(c))
                label = f"{name} {conf:.2f}"
                color = COLORS[c % len(COLORS)]
                draw_obb_label(img, box_xyxyxyxy, label, color)

        if args.save:
            out_path = save_dir / path.name
            cv2.imwrite(str(out_path), img)
            print(f"  {path.name} -> {out_path}  ({len(pred)} detections)")
        elif not args.show:
            print(f"{path}: {len(pred)} detections")

        if args.show:
            cv2.imshow(str(path), img)
            cv2.waitKey(0 if len(files) == 1 else 1)

    if args.show:
        cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
