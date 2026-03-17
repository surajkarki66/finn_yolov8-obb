#!/usr/bin/env python3
from typing import Any


import torch
import argparse
import numpy as np
import torch.nn as nn

from pathlib import Path
from tqdm import tqdm
from torch.cuda import amp
from torch.utils.data import DataLoader

from src.models.yolo import OBBModel
from src.utils.dataset import OBBDataset
from src.utils.loss import load_hyp
from val import compute_validation_metrics
from src.utils.torch_utils import intersect_dicts, strip_optimizer, ModelEMA, one_cycle


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, default="configs/models/yolov8-obb.yaml", help="Model config YAML")
    p.add_argument("--data", type=str, default="data/data.yaml", help="Data YAML (train/val paths, nc, names)")
    p.add_argument("--weights", type=str, default="", help="Resume from weights")
    p.add_argument("--resume", action="store_true", help="Resume training state (epoch/optimizer/best_fitness) from --weights")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument(
        "--nominal-batch",
        type=int,
        default=64,
        help="Nominal batch size used to compute gradient accumulation steps.",
    )
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=("sgd", "adam", "adamw"),
        help="Optimizer type.",
    )
    p.add_argument("--project", type=str, default="runs/train")
    p.add_argument("--name", type=str, default="exp")
    p.add_argument("--no-ema", action="store_true", help="Disable EMA (exponential moving average) of model")
    p.add_argument(
        "--fresh-ema",
        action="store_true",
        help="Do not load EMA from checkpoint; start EMA from current model (default for QAT with pretrained).",
    )
    p.add_argument(
        "--fresh-optimizer",
        action="store_true",
        help="Do not load optimizer state from checkpoint when --resume is set.",
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
    p.add_argument(
        "--patience",
        type=int,
        default=80,
        help="Early stopping patience in epochs without fitness improvement (0 disables).",
    )
    p.add_argument(
        "--min-delta",
        type=float,
        default=0.001,
        help="Minimum fitness improvement to reset early stopping counter.",
    )
    p.add_argument(
        "--fitness-weights",
        nargs=4,
        type=float,
        default=[0.0, 0.0, 0.1, 0.9],
        metavar=("P", "R", "mAP50", "mAP50_95"),
        help="Weighted fitness for best model selection: P R mAP50 mAP50_95.",
    )
    p.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Save full epoch snapshots every N epochs as epoch_XXX.pt (-1 disables).",
    )
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


def compute_fitness(metrics, weights):
    """Weighted fitness over precision, recall, mAP50, mAP50_95."""
    if metrics is None:
        return None
    p = float(metrics.get("precision", 0.0))
    r = float(metrics.get("recall", 0.0))
    m50 = float(metrics.get("mAP50", 0.0))
    m5095 = float(metrics.get("mAP50_95", 0.0))
    return weights[0] * p + weights[1] * r + weights[2] * m50 + weights[3] * m5095


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


def build_optimizer(model, hyp, optimizer_name="adamw"):
    """Build optimizer with parameter groups: BN/no-decay, weights/decay, bias/no-decay."""
    pg_bn, pg_weight, pg_bias = [], [], []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if name == "bias":
                pg_bias.append(param)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                pg_bn.append(param)
            else:
                pg_weight.append(param)

    momentum = getattr(hyp, "momentum", 0.937)
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "sgd":
        opt = torch.optim.SGD(pg_bn, lr=hyp.lr0, momentum=momentum, nesterov=True, weight_decay=0.0)
    elif optimizer_name == "adam":
        opt = torch.optim.Adam(pg_bn, lr=hyp.lr0, betas=(momentum, 0.999), weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(pg_bn, lr=hyp.lr0, betas=(momentum, 0.999), weight_decay=0.0)

    if pg_weight:
        opt.add_param_group({"params": pg_weight, "weight_decay": hyp.weight_decay})
    if pg_bias:
        opt.add_param_group({"params": pg_bias, "weight_decay": 0.0})

    print(
        f"Optimizer: {optimizer_name.upper()}  groups -> "
        f"BN/no-decay={len(pg_bn)}, weights/decay={len(pg_weight)}, bias/no-decay={len(pg_bias)}"
    )
    return opt


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cuda = device.type != "cpu"

    data = load_data_yaml(args.data)
    fitness_weights = [float(x) for x in args.fitness_weights]
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
    ckpt_optimizer = None
    start_epoch = 0
    best_fitness = -float("inf")
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
        ckpt_optimizer = ckpt.get("optimizer")
        if args.resume:
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            best_fitness = float(ckpt.get("best_fitness", best_fitness))
            if start_epoch > 0:
                print(f"Resuming from epoch {start_epoch + 1} (best_fitness={best_fitness:.5f})")
        del ckpt
    elif args.resume:
        raise FileNotFoundError("--resume requires a valid --weights checkpoint path")

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
    nb = len(loader)
    base_accumulate = max(round(args.nominal_batch / args.batch), 1)  # gradient accumulation steps
    # Scale weight_decay by accumulation factor (reference behavior)
    hyp.weight_decay *= args.batch * base_accumulate / args.nominal_batch
    opt = build_optimizer(model, hyp, args.optimizer)
    # LR scheduler: linear or cosine decay to lr0 * lrf
    cos_lr = getattr(hyp, "cos_lr", False) or args.cos_lr
    lrf = getattr(hyp, "lrf", 0.01)
    if cos_lr:
        lf = one_cycle(1, lrf, args.epochs)
    else:
        lf = lambda x: max(1 - x / args.epochs, 0) * (1.0 - lrf) + lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lf)
    if start_epoch > 0:
        scheduler.last_epoch = start_epoch - 1
    warmup_epochs = getattr(hyp, "warmup_epochs", 3.0)
    nw = max(round(warmup_epochs * nb), 100) if warmup_epochs > 0 else -1
    warmup_momentum = getattr(hyp, "warmup_momentum", 0.8)
    warmup_bias_lr = getattr(hyp, "warmup_bias_lr", 0.1)
    print(
        f"LR scheduler: {'cosine' if cos_lr else 'linear'}  lr0={hyp.lr0}  lrf={lrf}  final_lr={hyp.lr0 * lrf:.2e}  "
        f"warmup={nw} iters (warmup_epochs={warmup_epochs})"
    )
    print(f"Gradient accumulation: nominal_batch={args.nominal_batch}, effective accumulate={base_accumulate}")

    ema = ModelEMA(model) if not args.no_ema else None
    scaler = amp.GradScaler(enabled=cuda)
    if cuda:
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
    results_txt = save_dir / "results.txt"
    results_tags = (
        "epoch",
        "train_loss",
        "box",
        "cls",
        "dfl",
        "precision",
        "recall",
        "mAP50",
        "mAP50_95",
        "fitness",
        "lr",
    )
    if not (args.resume and results_txt.exists()):
        results_txt.write_text(("%12s" * len(results_tags) % results_tags) + "\n")
    best_epoch = -1
    epochs_without_improve = 0
    if start_epoch > 0 and best_fitness > -float("inf"):
        best_epoch = start_epoch - 1

    if args.resume and ckpt_optimizer is not None and not args.fresh_optimizer:
        opt.load_state_dict(ckpt_optimizer)
        print("Loaded optimizer state from checkpoint")
    elif args.resume and ckpt_optimizer is not None and args.fresh_optimizer:
        print("Starting with fresh optimizer (--fresh-optimizer)")

    for epoch in range(start_epoch, args.epochs):
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
        opt.zero_grad(set_to_none=True)
        for bi, batch in pbar:
            ni = bi + epoch * nb  # global iteration index
            accumulate = base_accumulate
            # Warmup: interpolate lr and momentum over first nw iters
            if ni < nw:
                xi = [0, nw]
                target_lr = hyp.lr0 * lf(epoch)
                accumulate = max(1, int(round(float(np.interp(ni, xi, [1, base_accumulate])))))
                for g in opt.param_groups:
                    g["lr"] = np.interp(ni, xi, [warmup_bias_lr, target_lr])
                    if "betas" in g:
                        g["betas"] = (np.interp(ni, xi, [warmup_momentum, momentum]), g["betas"][1])
                    elif "momentum" in g:
                        g["momentum"] = np.interp(ni, xi, [warmup_momentum, momentum])
            for k in ("img", "cls", "bboxes", "batch_idx"):
                batch[k] = batch[k].to(device, non_blocking=True)
            try:
                with amp.autocast(enabled=cuda):
                    loss, loss_items = model.loss(batch)
            except RuntimeError as e:
                if "same device" in str(e) or "different devices" in str(e):
                    move_brevitas_buffers_to_device(model, device)
                    model = model.to(device)
                    with amp.autocast(enabled=cuda):
                        loss, loss_items = model.loss(batch)
                else:
                    raise
            scaler.scale(loss).backward()
            if ((ni + 1) % accumulate == 0) or (bi == nb - 1):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
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

        # Fitness: weighted val metrics when validation exists, else negative loss (higher = better)
        metrics = None
        if data.get("val"):
            val_paths = data["val"]
            if not isinstance(val_paths, list):
                val_paths = [val_paths]
            data_val = {**data, "val": [str(p) for p in val_paths]}
            metrics = compute_validation_metrics(
                eval_model, data_val, device, imgsz=args.imgsz, batch_size=args.batch, use_tqdm=True
            )
            fitness = compute_fitness(metrics, fitness_weights) if metrics else -avg_loss
        else:
            fitness = -avg_loss

        save_state = model.state_dict()
        save_ema = ema.ema.state_dict() if ema is not None else None
        save_updates = ema.updates if ema is not None else 0
        ckpt = {
            "model": save_state,
            "epoch": epoch,
            "nc": nc,
            "best_fitness": float(best_fitness if best_fitness is not None else -float("inf")),
            "fitness": float(fitness if fitness is not None else -avg_loss),
            "fitness_weights": fitness_weights,
            "optimizer": opt.state_dict(),
            "training_results": results_txt.read_text() if results_txt.exists() else "",
            "train_args": vars(args),
        }
        if save_ema is not None:
            ckpt["ema"] = save_ema
            ckpt["updates"] = save_updates
        torch.save(ckpt, save_dir / "last.pt")
        improved = fitness > (best_fitness + args.min_delta)
        if improved:
            best_fitness = fitness
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(ckpt, save_dir / "best.pt")
        else:
            epochs_without_improve += 1

        if args.save_period > 0 and (epoch + 1) % args.save_period == 0:
            torch.save(ckpt, save_dir / f"epoch_{epoch + 1:03d}.pt")

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
            best_mark = "  (best)" if improved else ""
            print(train_part + val_part + best_mark)
        else:
            print(train_part)

        p = float(metrics["precision"]) if metrics else 0.0
        r = float(metrics["recall"]) if metrics else 0.0
        m50 = float(metrics["mAP50"]) if metrics else 0.0
        m5095 = float(metrics["mAP50_95"]) if metrics else 0.0
        current_lr = float(opt.param_groups[0]["lr"])
        results_line = (
            "%12.5g" * len(results_tags)
            % (
                epoch + 1,
                avg_loss,
                avg_box,
                avg_cls,
                avg_dfl,
                p,
                r,
                m50,
                m5095,
                float(fitness if fitness is not None else -avg_loss),
                current_lr,
            )
        )
        with open(results_txt, "a") as f:
            f.write(results_line + "\n")

        if args.patience > 0 and epochs_without_improve >= args.patience:
            print(
                f"Early stopping at epoch {epoch + 1}: "
                f"no fitness improvement > {args.min_delta} for {args.patience} epochs "
                f"(best epoch: {best_epoch + 1}, best fitness: {best_fitness:.5f})"
            )
            break

        # Append to results.csv: fixed columns for consistent CSV
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

