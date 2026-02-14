#!/usr/bin/env python3
"""
Export quantized YOLOv8-OBB to QONNX for FINN compilation.
Uses forward_split for QuantC2f and finn_export for the head.
Default: 6 outputs (P3, P4, P5, angle_P3, angle_P4, angle_P5); angle = raw logits for FINN.
Use --angle-legacy for previous format: 4 outputs (P3, P4, P5, angle) with sigmoid+reshape in graph.

Usage:
  python export.py --weights runs/obb/quant8w8a/best.pt --cfg configs/quantyolov8_obb_8w8a_common_act.yaml --data obb_data/data.yaml --input_shape 640 640
  python export.py ... --angle-legacy   # 4 outputs (single angle)
  python export.py ... --load_ema --output exported_obb.onnx
"""

import torch
import argparse
import sys
import yaml

from pathlib import Path
from brevitas.export import export_qonnx
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames

from src.models.yolo import OBBModel
from src.models.quant import QuantC2f, QuantOBB


def load_data_yaml(path):
    with open(path) as f:
        d = yaml.safe_load(f)
    return int(d.get("nc", 1))


def main():
    p = argparse.ArgumentParser(description="Export quantized YOLOv8-OBB to QONNX for FINN")
    p.add_argument("--weights", type=str, required=True, help="Path to checkpoint (e.g. best.pt)")
    p.add_argument("--cfg", type=str, required=True, help="Model config YAML (quant config used for training)")
    p.add_argument("--data", type=str, required=True, help="Data YAML (for nc)")
    p.add_argument("--input_shape", nargs=2, type=int, required=True, help="Input shape H W (e.g. 640 640)")
    p.add_argument("--load_ema", action="store_true", help="Load EMA weights from checkpoint")
    p.add_argument("--angle-legacy", action="store_true", help="Use previous angle format: 4 outputs (P3, P4, P5, angle) with sigmoid+reshape in graph. Default: 6 outputs (raw angle per scale, FINN-friendly)")
    p.add_argument("--output", type=str, default="", help="Output ONNX path (default: same dir as weights file, exported.onnx)")
    args = p.parse_args()

    nc = load_data_yaml(args.data)
    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = Path.cwd() / cfg_path
    if not cfg_path.exists():
        # Try next to weights
        wdir = Path(args.weights).parent
        alt = wdir / "cfg.yaml"
        if alt.exists():
            cfg_path = alt

    model = OBBModel(str(cfg_path), ch=3, nc=nc, verbose=False)
    ckpt_path = Path(args.weights)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Weights not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # With --load_ema: use ckpt["ema"] if present. Else use "model".
    # Standalone train.py only saves "model" (already EMA when EMA was used), so --load_ema has no effect there.
    state = ckpt.get("ema" if args.load_ema else "model", ckpt.get("model", ckpt))
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    model.load_state_dict(state, strict=False)
    model = model.eval()

    # FINN export: use forward_split for C2f, finn_export for head
    for m in model.modules():
        if isinstance(m, QuantC2f):
            m.forward = m.forward_split
        if isinstance(m, QuantOBB):
            m.finn_export = True
            m.finn_export_angle_raw = not args.angle_legacy

    out_path = args.output
    if not out_path:
        out_path = str(ckpt_path.parent / "exported.onnx")
    elif not Path(out_path).is_absolute():
        out_path = str(Path.cwd() / out_path)

    h, w = args.input_shape[0], args.input_shape[1]
    dummy = torch.rand(1, 3, h, w)
    export_qonnx(model, dummy, out_path)

    # QONNX cleanup
    qonnx_model = ModelWrapper(out_path)
    qonnx_model = qonnx_model.transform(InferShapes())
    qonnx_model = qonnx_model.transform(GiveUniqueNodeNames())
    qonnx_model = qonnx_model.transform(GiveReadableTensorNames())
    qonnx_model = cleanup_model(qonnx_model)
    qonnx_model.save(out_path)

    out_desc = "4 outputs (P3, P4, P5, angle)" if args.angle_legacy else "6 outputs (P3, P4, P5, angle_P3, angle_P4, angle_P5; angle = raw logits)"
    print("Exported QONNX (OBB, " + out_desc + "):", out_path)


if __name__ == "__main__":
    main()
