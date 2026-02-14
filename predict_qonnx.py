#!/usr/bin/env python3
import argparse
import math
import cv2
import numpy as np
import torch

from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes

from src.utils.ops import non_max_suppression, scale_boxes, xywhr2xyxyxyxy
from src.utils.tal import make_anchors, dist2rbox

REG_MAX = 16

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


def get_names(data_path: str, nc: int) -> dict:
    """Load class names from data.yaml if path given; else 0,1,...,nc-1."""
    if not data_path or not Path(data_path).exists():
        return {i: str(i) for i in range(nc)}
    import yaml
    with open(data_path) as f:
        d = yaml.safe_load(f) or {}
    names = d.get("names", {})
    if isinstance(names, list):
        return dict(enumerate(names))
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


def dfl_decode(box_raw: np.ndarray, reg_max: int = REG_MAX) -> np.ndarray:
    """Decode distribution focal layer: (B, reg_max*4, N) -> (B, 4, N) in distance space."""
    b, c, n = box_raw.shape
    assert c == reg_max * 4
    x = box_raw.reshape(b, 4, reg_max, n)
    x = torch.from_numpy(x).float()
    proj = torch.arange(reg_max, dtype=x.dtype, device=x.device)
    x = x.softmax(2)
    x = (x * proj.view(1, 1, -1, 1)).sum(2)
    return x.numpy()


def qonnx_outputs_to_prediction(
    box_outputs: List[np.ndarray],
    angle_outputs: List[np.ndarray],
    imgsz: int,
    nc: int,
    reg_max: int = REG_MAX,
    device: torch.device = None,
) -> torch.Tensor:
    """Convert 6 QONNX outputs to prediction tensor [1, 4+nc+1, total_anchors] for NMS.

    Mirrors QuantOBB post-processing that is omitted in FINN export (export returns
    raw P3,P4,P5 and angle_P3,angle_P4,angle_P5). Steps must match the model exactly:
    1) make_anchors from feats + strides  2) concat box outputs, split (box_raw, cls)
    3) DFL decode box_raw -> pred_dist   4) angle: concat raw per-scale, (sigmoid-0.25)*pi
    5) dist2rbox(pred_dist, angle, anchors, dim=1) * strides -> dbox
    6) cat(dbox, cls.sigmoid(), angle) -> [cx,cy,w,h, cls..., angle] for NMS.
    """
    device = device or torch.device("cpu")
    feats = [torch.from_numpy(x).float().to(device) for x in box_outputs]
    strides = torch.tensor(
        [imgsz / feats[i].shape[2] for i in range(len(feats))],
        dtype=feats[0].dtype,
        device=device,
    )
    anchor_points, stride_tensor = make_anchors(feats, strides)

    shape = feats[0].shape
    no = shape[1]
    assert no == nc + reg_max * 4
    x_cat = torch.cat([xi.view(shape[0], no, -1) for xi in feats], 2)

    box_raw, cls = x_cat.split((reg_max * 4, nc), 1)
    box_np = box_raw.cpu().numpy()
    dist = dfl_decode(box_np, reg_max)
    pred_dist = torch.from_numpy(dist).to(device)
    # Clip class logits for numerical stability
    cls = torch.clamp(cls, -20.0, 20.0)

    angle_list = []
    for a in angle_outputs:
        at = torch.from_numpy(a).float().to(device)
        angle_list.append(at.view(at.shape[0], -1))
    angle = torch.cat(angle_list, dim=1)
    # Raw angle logits: sigmoid then (x - 0.25)*pi. If already in [0,1] (e.g. after activation in graph), use directly.
    if angle.min() >= -0.01 and angle.max() <= 1.01:
        angle = (angle - 0.25) * math.pi
    else:
        angle = (torch.clamp(angle, -20.0, 20.0).sigmoid() - 0.25) * math.pi

    dbox = dist2rbox(pred_dist, angle, anchor_points.unsqueeze(0), dim=1) * stride_tensor.squeeze(-1)
    y = torch.cat((dbox, cls.sigmoid()), 1)
    pred = torch.cat([y, angle.unsqueeze(1)], 1)
    return pred


def run_qonnx_inference(
    model: ModelWrapper,
    input_name: str,
    img_tensor: np.ndarray,
    nc: int,
    reg_max: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Run QONNX model via execute_onnx. Use graph output order: first 3 = P3,P4,P5 (box+cls), next 3 = angle."""
    input_dict = {input_name: img_tensor}
    output_dict = execute_onnx(model, input_dict)

    c_box = nc + reg_max * 4
    # Preserve export order: P3, P4, P5, angle_P3, angle_P4, angle_P5 (model.graph.output order)
    ordered = []
    for out in model.graph.output:
        name = out.name
        if name not in output_dict:
            continue
        arr = output_dict[name]
        arr = np.asarray(arr, dtype=np.float32)
        if len(arr.shape) == 4:
            if arr.shape[-1] in (1, c_box) and arr.shape[1] not in (1, c_box):
                arr = arr.transpose(0, 3, 1, 2)
        ordered.append((name, arr))

    if len(ordered) != 6:
        raise ValueError(
            f"Expected 6 outputs (P3, P4, P5, angle_P3, angle_P4, angle_P5); got {len(ordered)}. "
            "Check export used default 6-output format (no --angle-legacy)."
        )

    box_outputs = [ordered[i][1] for i in range(3)]
    angle_outputs = [ordered[i][1] for i in range(3, 6)]
    return box_outputs, angle_outputs


def parse_args():
    p = argparse.ArgumentParser(description="QONNX YOLOv8-OBB inference and annotated image output")
    p.add_argument("--model", type=str, required=True, help="Path to exported QONNX .onnx model")
    p.add_argument("--source", type=str, default=".", help="Image file or directory")
    p.add_argument("--data", type=str, default="", help="Data YAML for class names (optional)")
    p.add_argument("--nc", type=int, default=1, help="Number of classes")
    p.add_argument("--imgsz", type=int, default=640, help="Input size (H and W); must match export")
    p.add_argument("--conf", type=float, default=0.20, help="Confidence threshold (QONNX often lower than .pt; try 0.15–0.25)")
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--save", action="store_true", help="Save annotated images")
    p.add_argument("--show", action="store_true", help="Show result in window")
    p.add_argument("--project", type=str, default="runs/yolov8-obb")
    p.add_argument("--name", type=str, default="predict_qonnx")
    p.add_argument("--reg-max", type=int, default=REG_MAX, help="DFL reg_max (default 16)")
    return p.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = ModelWrapper(str(model_path))
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    input_name = model.graph.input[0].name

    nc = args.nc
    reg_max = args.reg_max
    names = get_names(args.data, nc)
    imgsz = args.imgsz
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source = Path(args.source)
    if source.is_dir():
        files = list(source.rglob("*.*"))
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
        r = min(imgsz / h0, imgsz / w0)
        new_h, new_w = int(round(h0 * r)), int(round(w0 * r))
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        dh = (imgsz - new_h) / 2
        dw = (imgsz - new_w) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_in = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img_in = np.ascontiguousarray(img_in.transpose(2, 0, 1)[::-1])
        img_in = img_in.astype(np.float32) / 255.0
        img_in = np.expand_dims(img_in, axis=0)

        box_outputs, angle_outputs = run_qonnx_inference(
            model, input_name, img_in, nc=nc, reg_max=reg_max
        )
        if len(angle_outputs) != 3 or len(box_outputs) != 3:
            raise ValueError(
                f"Expected 3 box and 3 angle outputs; got {len(box_outputs)} and {len(angle_outputs)}. "
                "Use the default 6-output export (no --angle-legacy)."
            )

        pred = qonnx_outputs_to_prediction(
            box_outputs, angle_outputs, imgsz=imgsz, nc=nc, reg_max=reg_max, device=device
        )
        pred = non_max_suppression(
            pred, conf_thres=args.conf, iou_thres=args.iou, rotated=True, nc=nc
        )[0]

        if len(pred):
            pred[:, :4] = scale_boxes(
                (imgsz, imgsz), pred[:, :4], (h0, w0),
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

        if args.show:
            cv2.imshow(str(path), img)
            cv2.waitKey(0 if len(files) == 1 else 1)

    if args.show:
        cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
