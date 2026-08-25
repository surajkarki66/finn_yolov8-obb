#!/usr/bin/env python3

import argparse
import glob
import math
import os

import cv2
import numpy as np

from utils import (
    dfl_decode,
    dist2rbox,
    make_anchors,
    non_max_suppression,
    scale_boxes,
    xywhr2xyxyxyxy,
)

REG_MAX = 16

# Branch layout used by yolov8-obb_fpga.py / yolov8_obb_inference.py:
#   0: angle_P3   1: box_P3
#   2: angle_P4   3: box_P4
#   4: angle_P5   5: box_P5
ANGLE_BRANCH_IDX = [0, 2, 4]
BOX_BRANCH_IDX = [1, 3, 5]

HEX_PALETTE = (
    "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86",
    "1A9334", "00D4BB", "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF",
    "520085", "CB38FF", "FF95C8", "FF37C7",
)


def hex2bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


COLORS = [hex2bgr(f"#{c}") for c in HEX_PALETTE]


# -----------------------------------------------------------------
# Dequant
# -----------------------------------------------------------------
def load_scales(scale_dir):
    """mul_0..5.npy / add_0..5.npy, matching the branch order in the npz
    (output_0 .. output_5 == angle_P3, box_P3, angle_P4, box_P4, angle_P5, box_P5)."""
    muls = [np.load(os.path.join(scale_dir, f"mul_{i}.npy")) for i in range(6)]
    adds = [np.load(os.path.join(scale_dir, f"add_{i}.npy")) for i in range(6)]
    return muls, adds


def prepare_outputs_for_decode(outputs, muls, adds):
    """NHWC raw accelerator outputs -> NCHW float outputs, with affine dequant applied.

    outputs: list of 6 NHWC numpy arrays (output_0 .. output_5 from the npz,
             in ANGLE/BOX branch order defined above).
    Returns (box_outputs, angle_outputs), each a list of 3 NCHW arrays,
    ordered P3, P4, P5.
    """
    dequant = []
    for output, mul, add in zip(outputs, muls, adds):
        out = output.transpose(0, 3, 1, 2).astype(np.float64)  # NHWC -> NCHW
        out = out * mul + add
        dequant.append(out)

    angle_outputs = [dequant[i] for i in ANGLE_BRANCH_IDX]
    box_outputs = [dequant[i] for i in BOX_BRANCH_IDX]
    return box_outputs, angle_outputs


# -----------------------------------------------------------------
# YOLOv8-OBB postprocessing
# -----------------------------------------------------------------
def outputs_to_prediction(box_outputs, angle_outputs, imgsz, nc, reg_max=REG_MAX):
    """Convert dequantized accelerator outputs to prediction tensor
    [1, 4+nc+1, total_anchors] for non_max_suppression().
    """
    feats = box_outputs
    strides = np.array([imgsz / feats[i].shape[2] for i in range(len(feats))], dtype=np.float64)
    anchor_points, stride_tensor = make_anchors(feats, strides)
    anchor_points = anchor_points[None, :, :]      # (1, N, 2)
    stride_tensor = stride_tensor[None, :, 0]      # (1, N)

    shape = feats[0].shape
    no = shape[1]
    assert no == nc + reg_max * 4, f"box branch has {no} channels, expected {nc + reg_max * 4}"
    x_cat = np.concatenate([xi.reshape(shape[0], no, -1) for xi in feats], axis=2)

    box_raw = x_cat[:, : reg_max * 4, :]
    cls = x_cat[:, reg_max * 4:, :]

    pred_dist = dfl_decode(box_raw, reg_max)
    cls = np.clip(cls, -20.0, 20.0)

    angle_list = [a.reshape(a.shape[0], -1) for a in angle_outputs]
    angle = np.concatenate(angle_list, axis=1)[:, None, :]  # (1, 1, N)

    if angle.min() >= -0.01 and angle.max() <= 1.01:
        angle = (angle - 0.25) * math.pi
    else:
        angle_sig = 1.0 / (1.0 + np.exp(-np.clip(angle, -20.0, 20.0)))
        angle = (angle_sig - 0.25) * math.pi

    dbox = dist2rbox(pred_dist, angle, anchor_points, dim=1) * stride_tensor[:, None, :]
    cls_sig = 1.0 / (1.0 + np.exp(-cls))
    y = np.concatenate([dbox, cls_sig], axis=1)
    pred = np.concatenate([y, angle], axis=1)
    return pred


# -----------------------------------------------------------------
# End-to-end: raw npz -> final detections in original-image pixel space
# -----------------------------------------------------------------
def decode_npz(npz_path, muls, adds, imgsz=416, nc=1, reg_max=REG_MAX,
                conf_thres=0.20, iou_thres=0.45):
    """
    Returns:
      det: (M, 7) array [cx, cy, w, h, conf, cls, angle_rad] in ORIGINAL image
           pixel coordinates (already rescaled with scale_boxes).
      meta: dict with image_path, orig_shape, ratio, pad (as stored in the npz)
    """
    data = np.load(npz_path, allow_pickle=True)

    raw_outputs = [data[f"output_{i}"] for i in range(6)]
    box_outputs, angle_outputs = prepare_outputs_for_decode(raw_outputs, muls, adds)

    pred = outputs_to_prediction(box_outputs, angle_outputs, imgsz=imgsz, nc=nc, reg_max=reg_max)
    det = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, nc=nc)[0]

    orig_shape = tuple(int(v) for v in data["orig_shape"])  # (H, W, C)
    ratio = data["ratio"]
    pad = tuple(data["pad"].tolist())
    image_path = decode_image_path(data["image_path"])

    if len(det):
        r = float(ratio[0]) if np.ndim(ratio) else float(ratio)
        det = det.copy()
        det[:, :4] = scale_boxes(
            (imgsz, imgsz), det[:, :4], orig_shape[:2],
            ratio_pad=((r,), pad), xywh=True,
        )

    meta = {"image_path": image_path, "orig_shape": orig_shape, "ratio": ratio, "pad": pad}
    return det, meta


def decode_image_path(raw):
    """Robustly turn whatever was stored in the npz under 'image_path' into a
    clean string path. Handles plain strings, 0-d numpy arrays, numpy str_/bytes_
    scalars, and the common footgun where str(b'...') produces "b'...'" literally.
    """
    # Unwrap 0-d numpy arrays / numpy scalar types to a native python object.
    if isinstance(raw, np.ndarray):
        raw = raw.item() if raw.shape == () else raw.tolist()

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    else:
        raw = str(raw)

    # Fix the "b'/path/to/img.jpg'" case that comes from str(bytes_obj) being
    # applied twice (once by numpy, once by us / old code).
    if raw.startswith(("b'", 'b"')) and raw.endswith(("'", '"')):
        raw = raw[2:-1]

    return raw


def save_detections_txt(det, out_path):
    """One line per detection: cls conf cx cy w h angle_rad (pixel coords, original image)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for row in det:
            cx, cy, w, h, conf, cls, angle = row.tolist()
            f.write(f"{int(cls)} {conf:.6f} {cx:.3f} {cy:.3f} {w:.3f} {h:.3f} {angle:.6f}\n")


def draw_detections(img, det, names=None):
    for row in det:
        cx, cy, w, h, conf, cls, angle = row.tolist()
        corners = xywhr2xyxyxyxy(np.array([[cx, cy, w, h, angle]]))[0]
        box = corners.astype(np.int32)
        c = int(cls)
        color = COLORS[c % len(COLORS)]
        cv2.polylines(img, [box], True, color, 2, cv2.LINE_AA)
        name = names.get(c, str(c)) if names else str(c)
        label = f"{name} {conf:.2f}"
        p1 = (int(box[0][0]), int(box[0][1]))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        outside = p1[1] - th >= 3
        p2 = (p1[0] + tw, p1[1] - th - 3 if outside else p1[1] + th + 3)
        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (p1[0], p1[1] - 2 if outside else p1[1] + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def resolve_image_path(stored_path, img_dir=None):
    """Try to find a real, readable image file.

    1. If stored_path exists as-is, use it.
    2. Else, if --img-dir was given, try <img_dir>/<basename(stored_path)>.
    3. Else, give up and return the original (so the caller can report it).
    """
    if os.path.exists(stored_path):
        return stored_path

    if img_dir:
        candidate = os.path.join(img_dir, os.path.basename(stored_path))
        if os.path.exists(candidate):
            return candidate

    return stored_path


def parse_args():
    p = argparse.ArgumentParser(description="Decode raw FPGA .npz outputs into OBB detections")
    p.add_argument("--raw-dir", default="raw_outputs", help="Dir of *_raw.npz from yolov8-obb_fpga.py")
    p.add_argument("--scale-dir", default=".", help="Dir with mul_0..5.npy / add_0..5.npy")
    p.add_argument("--out-dir", default="predictions", help="Where to save per-image detection .txt")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--nc", type=int, default=1)
    p.add_argument("--reg-max", type=int, default=REG_MAX)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--save-vis", action="store_true", help="Also save annotated images")
    p.add_argument("--vis-dir", default="vis")
    p.add_argument("--img-dir", type=str, default="",
                    help="Fallback directory to look for source images if the path "
                         "stored inside the npz no longer exists (e.g. moved machines).")
    p.add_argument("--data", type=str, default="", help="data.yaml for class names (optional, for --save-vis)")
    return p.parse_args()


def get_names(data_path, nc):
    if not data_path or not os.path.exists(data_path):
        return {i: str(i) for i in range(nc)}
    import yaml
    with open(data_path) as f:
        d = yaml.safe_load(f) or {}
    names = d.get("names", {})
    if isinstance(names, list):
        return dict(enumerate(names))
    return names if names else {i: str(i) for i in range(nc)}


def main():
    args = parse_args()
    npz_paths = sorted(glob.glob(os.path.join(args.raw_dir, "*_raw.npz")))
    if not npz_paths:
        raise FileNotFoundError(f"No *_raw.npz files found in {args.raw_dir}")
    print(f"Found {len(npz_paths)} raw output file(s) in {args.raw_dir}")

    muls, adds = load_scales(args.scale_dir)
    names = get_names(args.data, args.nc)

    n_det_total = 0
    n_vis_saved = 0
    for npz_path in npz_paths:
        det, meta = decode_npz(
            npz_path, muls, adds, imgsz=args.imgsz, nc=args.nc, reg_max=args.reg_max,
            conf_thres=args.conf, iou_thres=args.iou,
        )
        stem = os.path.splitext(os.path.basename(npz_path))[0].removesuffix("_raw")
        out_path = os.path.join(args.out_dir, f"{stem}.txt")
        save_detections_txt(det, out_path)
        n_det_total += len(det)
        print(f"  {stem}: {len(det)} detection(s) -> {out_path}")

        if args.save_vis:
            stored_path = meta["image_path"]
            img_path = resolve_image_path(stored_path, args.img_dir)

            img = cv2.imread(img_path)
            if img is None:
                print(f"  [!] {stem}: could not load image for visualization.")
                print(f"      stored path in npz: {stored_path!r}")
                if img_path != stored_path:
                    print(f"      also tried:         {img_path!r}")
                print("      -> pass --img-dir <folder with source images> if the "
                      "path in the npz is stale (e.g. from a different machine/run).")
            else:
                img = draw_detections(img, det, names)
                os.makedirs(args.vis_dir, exist_ok=True)
                out_vis_path = os.path.join(args.vis_dir, os.path.basename(img_path))
                ok = cv2.imwrite(out_vis_path, img)
                if not ok:
                    print(f"  [!] {stem}: cv2.imwrite failed writing {out_vis_path!r} "
                          f"(check --vis-dir is writable and extension is valid)")
                else:
                    n_vis_saved += 1
                    print(f"      vis -> {out_vis_path}")

    print(f"\nDone. {n_det_total} total detection(s) across {len(npz_paths)} image(s).")
    print(f"Detections saved to {args.out_dir}")
    if args.save_vis:
        print(f"Annotated images saved: {n_vis_saved}/{len(npz_paths)} -> {args.vis_dir}")


if __name__ == "__main__":
    main()