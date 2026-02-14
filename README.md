# YOLOv8-OBB (standalone)

Standalone PyTorch implementation for **YOLOv8 Oriented Bounding Box (OBB)** training, validation, inference, and quantization. It follows the same architecture and loss as Ultralytics YOLOv8-OBB but **does not depend on the ultralytics package** — all code is self-contained for full control and customization.

---

## Features

- **Training & validation** — OBB detection with v8 loss, TAL assigner, probiou, optional EMA and hyperparameter YAML.
- **Model scales** — n / s / m / l / x via config (`scale` and `scales` in model YAML); compatible with Ultralytics pretrained OBB weights.
- **QAT** — Quantization-aware training (Brevitas) with common activation; 8w8a and 4w4a configs; export to QONNX for FINN.

---

## Setup

From the project root:

```bash
pip install -r requirements.txt
```

Requirements include: `torch`, `torchvision`, `numpy`, `opencv-python-headless`, `PyYAML`, `tqdm`, `Pillow`, `brevitas`, `qonnx`. No separate QAT requirements file; use the same `requirements.txt` for float and quant training.

---

## Data

- **Format**: OBB labels = one line per object: `class_id x1 y1 x2 y2 x3 y3 x4 y4` (normalized 0–1, four corners). Same as [Ultralytics OBB](https://docs.ultralytics.com/datasets/obb/).
- **Layout**: Use `images/` and `labels/` directories; paths are defined in a single **data YAML** (see below).
- **Details**: See [data/README.md](data/README.md) for directory layout, `data.yaml` keys, and label format.

Example `data.yaml`:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 15
names: {0: class0, 1: class1, ...}
```

---

## Quick start

### Train (float)

```bash
python train.py --data path/to/data.yaml --epochs 100 --batch 8
```

- **Config**: `--cfg configs/models/yolov8-obb.yaml` (default). Model scale is set by top-level `scale: n` in the YAML or inferred from the config filename (e.g. `yolov8n-obb.yaml` → n).
- **Options**: `--weights`, `--imgsz`, `--device`, `--workers`, `--project`, `--name`, `--hyp`, `--freeze`, `--no-ema`, `--cos-lr`, `--no-amp`.

### Finetune from Ultralytics pretrained OBB weights

State dicts are compatible. Download e.g. [yolov8n-obb.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-obb.pt) (see [pretrained/README.md](pretrained/README.md)) then:

```bash
python train.py --data your_data.yaml --weights pretrained/yolov8n-obb.pt --epochs 50 --batch 8
```

Supports both Ultralytics training checkpoint (`ckpt["model"]`) and raw state dict.

### Validate

```bash
python val.py --weights runs/yolov8-obb/train/last.pt --data path/to/data.yaml
```

Computes Precision, Recall, mAP50, mAP50-95 (probiou for OBB). Options: `--cfg`, `--batch`, `--imgsz`, `--conf`, `--iou`, `--project`, `--name`, `--save-json`.

### Predict

```bash
python predict.py --weights runs/yolov8-obb/train/best.pt --source path/to/images --save
```

Optional `--data path/to/data.yaml` for class names. Options: `--cfg`, `--imgsz`, `--conf`, `--iou`, `--device`.

---

## QAT (Quantization-Aware Training)

Quantized YOLOv8-OBB with **common activation** (shared activation quantizers per group). Quantizer code is in the repo (`src/models/quant_common.py`); no external brevitas examples needed.

### Train quantized (e.g. 8w8a)

```bash
python train.py --cfg configs/models/quant/quantyolov8_obb_8w8a_common_act.yaml --data path/to/data.yaml --weights runs/yolov8-obb/train/best.pt --epochs 100 --batch 8 --train-quant-scales
```

- **Configs**:  
  - `configs/models/quant/quantyolov8_obb_8w8a_common_act.yaml` — 8-bit weights and activations.  
  - `configs/models/quant/quantyolov8_obb_4w4a_common_act.yaml` — 4-bit weights and activations.

Same `val.py` and `predict.py` work with saved quant weights; use the same `--cfg` as for training.

### Export to QONNX (FINN)

Export produces QONNX with raw outputs for FINN. Default: 6 outputs (P3, P4, P5, angle_P3, angle_P4, angle_P5); use `--angle-legacy` for 4 outputs (P3, P4, P5, angle).

```bash
python export.py --weights runs/yolov8-obb/quant8w8a/best.pt --cfg configs/models/quant/quantyolov8_obb_8w8a_common_act.yaml --data path/to/data.yaml --input_shape 640 640
```

Options: `--load_ema`, `--output path/to/exported_obb.onnx`, `--angle-legacy`. Requires `brevitas` and `qonnx`.

---

## Config and project layout

| Path | Description |
|------|-------------|
| `configs/models/yolov8-obb.yaml` | Float model (backbone + OBB head, scale n/s/m/l/x). |
| `configs/models/quant/quantyolov8_obb_*_common_act.yaml` | QAT model configs (8w8a, 4w4a). |
| `configs/hyp/hyp.yaml` | Training hyperparameters (lr, loss gains, warmup, etc.). |
| `train.py`, `val.py`, `predict.py`, `export.py` | Entry scripts. |
| `src/` | Core package (no ultralytics). |
| `src/models/` | `yolo.py` (OBBModel, parse_model), `quant.py` (QuantConv, QuantC2f, QuantSPPF, QuantOBB), `quant_common.py` (common activation quantizers). |
| `src/nn/` | `conv.py`, `block.py` (Conv, C2f, SPPF), `head.py` (OBB). |
| `src/utils/` | `dataset.py` (OBBDataset), `loss.py` (v8OBBLoss, load_hyp), `tal.py`, `metrics.py`, `ops.py`, `torch_utils.py`. |
| `data/` | Place for datasets and `data.yaml`; see [data/README.md](data/README.md). |
| `pretrained/` | Optional pretrained weights; see [pretrained/README.md](pretrained/README.md). |
| `runs/` | Training and validation outputs (e.g. `runs/yolov8-obb/train/`, `runs/yolov8-obb/val/`). |

---

## Acknowledgements

This project builds on and acknowledges the following work:

- **[Ultralytics](https://github.com/ultralytics/ultralytics)** — YOLOv8 architecture, OBB format, and reference implementation.
- **[Brevitas](https://github.com/Xilinx/brevitas)** — Quantization-aware training and export for QAT and QONNX/FINN.
- **[yolo_finn](https://github.com/mdanilow/yolo_finn)** — Reference for FINN-oriented YOLO quantization and export (QAT, common activation, export flow).
