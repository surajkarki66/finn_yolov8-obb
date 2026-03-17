#!/usr/bin/env python3
from typing import Any


import torch
import argparse
import numpy as np

from pathlib import Path
from tqdm import tqdm
from torch.cuda import amp
from torch.utils.data import DataLoader

from src.models.yolo import OBBModel
from src.utils.dataset import OBBDataset
from src.utils.metrics import fitness
from src.utils.loss import load_hyp
from val import compute_validation_metrics
from src.utils.torch_utils import intersect_dicts, strip_optimizer, ModelEMA, one_cycle


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, default="configs/models/yolov8-obb.yaml", help="Model config YAML")
    p.add_argument("--data", type=str, required=True, help="Data YAML (train/val paths, nc, names)")
    p.add_argument("--weights", type=str, default="", help="Resume from weights")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", type=str, default="runs/yolov8-obb")
    p.add_argument("--name", type=str, default="train")
    p.add_argument("--no-ema", action="store_true", help="Disable EMA (exponential moving average) of model")
    p.add_argument(
        "--fresh-ema",
        action="store_true",
        help="Do not load EMA from checkpoint; start EMA from current model (default for QAT with pretrained).",
    )
    p.add_argument(
        "--freeze",
        type=str,
        default=None,
        help="Freeze first n layers (e.g. 10) or comma-separated indices (e.g. 0,1,2). DFL is always frozen.",
    )
    p.add_argument(
        "--hyp",
        type=str,
        default="configs/hyp/hyp.yaml",
        help="Path to hyperparameters YAML (box, cls, dfl, lr0, weight_decay, lrf, cos_lr). Overrides defaults.",
    )
    p.add_argument("--cos-lr", action="store_true", help="Use cosine LR schedule (overrides hyp cos_lr)")
    p.add_argument("--train-quant-scales", action="store_true", help="QAT: train activation quant scales (Brevitas scaling_impl)")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision (AMP)")
    return p.parse_args()


def load_data_yaml(path):
    import yaml
    with open(path) as f:
        d = yaml.safe_load(f)
    if "path" in d:
        root = Path(d["path"])
    else:
        root = Path(path).parent
    train = root / d["train"] if isinstance(d["train"], str) else [root / x for x in d["train"]]
    val = d.get("val")
    if val:
        val = root / val if isinstance(val, str) else [root / x for x in val]
    names = d.get("names", {})
    if isinstance(names, list):
        names = dict[int, Any](enumerate[Any](names))
    nc = int(d.get("nc", len(names) if names else 1))
    return {"train": train, "val": val, "nc": nc, "names": names}


def save_metrics(csv_path, epoch, metrics_dict):
    """Append one row to results.csv"""
    keys = list(metrics_dict.keys())
    vals = [metrics_dict[k] for k in keys]
    n = len(keys) + 1
    header = ("%23s," * n % tuple(["epoch"] + keys)).rstrip(",") + "\n"
    row = ("%23.5g," * n % tuple([epoch + 1] + vals)).rstrip(",") + "\n"
    if not csv_path.exists():
        with open(csv_path, "w") as f:
            f.write(header)
    with open(csv_path, "a") as f:
        f.write(row)


def move_brevitas_buffers_to_device(model, device):
    """
    Move any tensor attributes to device.
    """
    for m in model.modules():
        for name in dir(m):
            if name.startswith("_"):
                continue
            try:
                val = getattr(m, name)
                if isinstance(val, torch.Tensor) and val.device != device:
                    setattr(m, name, val.to(device, non_blocking=True))
            except (AttributeError, TypeError):
                pass


def freeze_layers(model, freeze_arg):
    """
    Freeze model parameters by name.
    - always_freeze: params whose name contains '.dfl'
    - freeze_arg: None, or int (freeze first n layers), or comma-separated indices (e.g. '0,1,2')
    """
    always_freeze_names = [".dfl"]
    if freeze_arg is None or (isinstance(freeze_arg, str) and freeze_arg.strip() == ""):
        freeze_list = []
    elif "," in str(freeze_arg):
        freeze_list = [int(x.strip()) for x in str(freeze_arg).split(",")]
    else:
        n = int(freeze_arg)
        # First n layers: model.0, model.1, ..., model.(n-1)
        freeze_list = list(range(n))
    freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
    n_frozen = 0
    for k, v in model.named_parameters():
        if any(x in k for x in freeze_layer_names):
            v.requires_grad = False
            n_frozen += 1
            print(f"  Freezing '{k}'")
    if n_frozen:
        print(f"Frozen {n_frozen} parameter groups (including .dfl)")


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data = load_data_yaml(args.data)
    nc = data["nc"]
    train_path = data["train"]
    if not isinstance(train_path, list):
        train_path = [str(train_path)]

    hyp = load_hyp(args.hyp)
    model = OBBModel(args.cfg, ch=3, nc=nc, verbose=True)
    model.args = hyp
    model = model.to(device)
    if args.hyp and Path(args.hyp).exists():
        print(
            f"Loaded hyperparameters from {args.hyp}  "
            f"(lr0={hyp.lr0}, lrf={hyp.lrf}, momentum={hyp.momentum}, warmup_epochs={hyp.warmup_epochs}, "
            f"box={hyp.box}, cls={hyp.cls}, dfl={hyp.dfl})"
        )

    ckpt_ema = None
    ckpt_updates = 0
    if args.weights and Path(args.weights).exists():
        try:
            ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(args.weights, map_location="cpu")
        if "model" in ckpt:
            m = ckpt["model"]
            csd = m.float().state_dict() if hasattr(m, "state_dict") else m
        else:
            csd = ckpt
        csd = intersect_dicts(csd, model.state_dict())
        model.load_state_dict(csd, strict=False)
        print(f"Transferred {len(csd)}/{len(model.state_dict())} items from {args.weights}")
        ckpt_ema = ckpt.get("ema")
        ckpt_updates = ckpt.get("updates", 0)
        del ckpt

    freeze_layers(model, args.freeze)
    move_brevitas_buffers_to_device(model, device)
    model = model.to(device)

    dataset = OBBDataset(
        img_path=train_path,
        imgsz=args.imgsz,
        data=data,
        augment=True,
    )
    loader = DataLoader[Any](
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=OBBDataset.collate_fn,
    )

    momentum = getattr(hyp, "momentum", 0.937)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=hyp.lr0,
        betas=(momentum, 0.999),
        weight_decay=hyp.weight_decay,
    )
    # LR scheduler: linear or cosine decay to lr0 * lrf
    cos_lr = getattr(hyp, "cos_lr", False) or args.cos_lr
    lrf = getattr(hyp, "lrf", 0.01)
    if cos_lr:
        lf = one_cycle(1, lrf, args.epochs)
    else:
        lf = lambda x: max(1 - x / args.epochs, 0) * (1.0 - lrf) + lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lf)
    nb = len(loader)
    warmup_epochs = getattr(hyp, "warmup_epochs", 3.0)
    nw = max(round(warmup_epochs * nb), 100) if warmup_epochs > 0 else -1
    warmup_momentum = getattr(hyp, "warmup_momentum", 0.8)
    warmup_bias_lr = getattr(hyp, "warmup_bias_lr", 0.1)
    print(
        f"LR scheduler: {'cosine' if cos_lr else 'linear'}  lr0={hyp.lr0}  lrf={lrf}  final_lr={hyp.lr0 * lrf:.2e}  "
        f"warmup={nw} iters (warmup_epochs={warmup_epochs})"
    )

    ema = ModelEMA(model) if not args.no_ema else None
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("Using mixed precision (AMP)")
    # QAT with pretrained: model layout differs from FP checkpoint, so use fresh EMA unless overridden
    use_fresh_ema = args.fresh_ema or (args.train_quant_scales and args.weights)
    if ema is not None and ckpt_ema is not None and not use_fresh_ema:
        ema.ema.load_state_dict(ckpt_ema, strict=False)
        ema.updates = ckpt_updates
        print(f"Loaded EMA from checkpoint (updates={ckpt_updates})")
    elif ema is not None and ckpt_ema is not None and use_fresh_ema:
        print("Starting with fresh EMA (QAT with pretrained or --fresh-ema)")
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    results_csv = save_dir / "results.csv"
    best_fitness = -float("inf")

    for epoch in range(args.epochs):
        # Step scheduler at start of epoch (after previous epoch's optimizer steps) to avoid PyTorch warning
        if epoch > 0:
            scheduler.step()
        model.train()
        total_loss = 0.0
        running = [0.0] * 3
        pbar = tqdm(
            enumerate(loader),
            total=int(nb),
            desc=f"Epoch {epoch+1}/{args.epochs}",
            unit="batch",
            leave=True,
        )
        for bi, batch in pbar:
            ni = bi + epoch * nb  # global iteration index
            # Warmup: interpolate lr and momentum over first nw iters
            if ni < nw:
                xi = [0, nw]
                target_lr = hyp.lr0 * lf(epoch)
                for g in opt.param_groups:
                    g["lr"] = np.interp(ni, xi, [warmup_bias_lr, target_lr])
                    g["betas"] = (np.interp(ni, xi, [warmup_momentum, momentum]), 0.999)
            for k in ("img", "cls", "bboxes", "batch_idx"):
                batch[k] = batch[k].to(device, non_blocking=True)
            opt.zero_grad()
            try:
                with amp.autocast(enabled=use_amp):
                    loss, loss_items = model.loss(batch)
            except RuntimeError as e:
                if "same device" in str(e) or "different devices" in str(e):
                    move_brevitas_buffers_to_device(model, device)
                    model = model.to(device)
                    with amp.autocast(enabled=use_amp):
                        loss, loss_items = model.loss(batch)
                else:
                    raise
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)
            total_loss += loss.item()
            for j in range(3):
                running[j] += loss_items[j].item()
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                box=f"{loss_items[0].item():.2f}",
                cls=f"{loss_items[1].item():.2f}",
                dfl=f"{loss_items[2].item():.2f}",
            )

        n_loader = len(loader)
        avg_loss = total_loss / n_loader
        avg_box = running[0] / n_loader
        avg_cls = running[1] / n_loader
        avg_dfl = running[2] / n_loader
        # Use EMA model for validation and saving
        eval_model = ema.ema.to(device) if ema is not None else model
        if ema is not None:
            ema.update_attr(model, include=("nc", "args"))

        # Fitness: use weighted [P, R, mAP50, mAP50_95] like reference, else use negative loss (higher = better)
        metrics = None
        if data.get("val"):
            val_paths = data["val"]
            if not isinstance(val_paths, list):
                val_paths = [val_paths]
            data_val = {**data, "val": [str(p) for p in val_paths]}
            metrics = compute_validation_metrics(
                eval_model, data_val, device, imgsz=args.imgsz, batch_size=args.batch, use_tqdm=True
            )
            if metrics:
                metric_vec = np.array(
                    [[
                        float(metrics.get("precision", 0.0)),
                        float(metrics.get("recall", 0.0)),
                        float(metrics.get("mAP50", 0.0)),
                        float(metrics.get("mAP50_95", 0.0)),
                    ]],
                    dtype=np.float32,
                )
                fi = float(fitness(metric_vec)[0])
            else:
                fi = -avg_loss
        else:
            fi = -avg_loss

        save_state = model.state_dict()
        save_ema = ema.ema.state_dict() if ema is not None else None
        save_updates = ema.updates if ema is not None else 0
        ckpt = {"model": save_state, "epoch": epoch, "nc": nc}
        if save_ema is not None:
            ckpt["ema"] = save_ema
            ckpt["updates"] = save_updates
        torch.save(ckpt, save_dir / "last.pt")
        if fi > best_fitness:
            best_fitness = fi
            torch.save(ckpt, save_dir / "best.pt")

        # End-of-epoch line: train loss + validation metrics
        train_part = (
            f"Epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}  "
            f"box={avg_box:.3f}  cls={avg_cls:.3f}  dfl={avg_dfl:.3f}"
        )
        if metrics is not None:
            val_part = (
                f"  val  mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50_95']:.4f}  "
                f"precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}"
            )
            best_mark = "  (best)" if fi >= best_fitness else ""
            print(train_part + val_part + best_mark)
        else:
            print(train_part)

        # Append to results.csv: fixed columns for consistent CSV
        current_lr = opt.param_groups[0]["lr"]
        row_metrics = {
            "train/box_loss": round(avg_box, 5),
            "train/cls_loss": round(avg_cls, 5),
            "train/dfl_loss": round(avg_dfl, 5),
            "train/loss": round(avg_loss, 5),
            "metrics/precision(B)": round(metrics["precision"], 5) if metrics else 0,
            "metrics/recall(B)": round(metrics["recall"], 5) if metrics else 0,
            "metrics/mAP50(B)": round(metrics["mAP50"], 5) if metrics else 0,
            "metrics/mAP50-95(B)": round(metrics["mAP50_95"], 5) if metrics else 0,
            "lr/pg0": round(current_lr, 6),
        }
        save_metrics(results_csv, epoch, row_metrics)

    # Strip optimizer so final checkpoints are actual model size
    strip_optimizer(save_dir / "best.pt", half=True)
    strip_optimizer(save_dir / "last.pt", half=True)
    print(f"Saved last.pt and best.pt (stripped) to {save_dir}")


if __name__ == "__main__":
    main()

