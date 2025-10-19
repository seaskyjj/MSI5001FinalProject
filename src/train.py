"""Training script for the electrical component Faster R-CNN detector."""
from __future__ import annotations

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import DatasetConfig, TrainingConfig
from .dataset import ElectricalComponentsDataset, create_data_loaders
from .model import build_model
from .utils import MetricLogger, compute_detection_metrics, set_seed

LOGGER = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DatasetConfig().base_dir)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=TrainingConfig().num_workers)
    parser.add_argument("--no-augmentation", action="store_true", help="Disable data augmentation")
    parser.add_argument("--small-object", action="store_true", help="Use smaller RPN anchors")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--checkpoint", type=Path, default=TrainingConfig().checkpoint_path)
    parser.add_argument("--pretrained-path", type=Path, default=TrainingConfig().pretrained_weights_path)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--train-split", default=DatasetConfig().train_split)
    parser.add_argument("--valid-split", default=DatasetConfig().valid_split)
    parser.add_argument("--num-classes", type=int, default=DatasetConfig().num_classes)
    return parser.parse_args()


def prepare_configs(args: argparse.Namespace) -> tuple[DatasetConfig, TrainingConfig]:
    dataset_cfg = DatasetConfig(
        base_dir=args.data_dir,
        train_split=args.train_split,
        valid_split=args.valid_split,
        num_classes=args.num_classes,
    )

    train_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        augmentation=not args.no_augmentation,
        small_object=args.small_object,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        eval_interval=args.eval_interval,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        pretrained_weights_path=args.pretrained_path,
        log_every=args.log_every,
    )
    train_cfg.ensure_directories()
    return dataset_cfg, train_cfg


def move_to_device(targets: List[Dict[str, torch.Tensor]], device: torch.device) -> List[Dict[str, torch.Tensor]]:
    return [{k: v.to(device) for k, v in target.items()} for target in targets]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp: bool,
    log_every: int,
) -> float:
    model.train()
    metric_logger = MetricLogger()
    progress = tqdm(loader, desc="Train", leave=False)

    for step, (images, targets) in enumerate(progress, start=1):
        images = [img.to(device) for img in images]
        targets = move_to_device(targets, device)

        optimizer.zero_grad()
        with autocast(enabled=amp):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        if torch.isfinite(loss):
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:  # pragma: no cover - guard for invalid losses
            LOGGER.warning("Skipping step %s due to non-finite loss", step)
            scaler.update()
            continue

        metric_logger.update(loss=loss.item())
        if step % log_every == 0:
            progress.set_postfix_str(metric_logger.format())

    return metric_logger.meters.get("loss").avg if metric_logger.meters else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dataset_cfg: DatasetConfig,
    train_cfg: TrainingConfig,
) -> Dict[str, torch.Tensor | float | List[float]]:
    was_training = model.training
    model.eval()

    predictions = []
    targets_for_eval = []
    total_loss = 0.0
    num_batches = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets_device = move_to_device(targets, device)

        model.train()
        loss_dict = model(images, targets_device)
        total_loss += sum(loss_dict.values()).item()
        num_batches += 1
        model.eval()

        outputs = model(images)
        for output, target in zip(outputs, targets_device):
            scores = output["scores"].detach().cpu().numpy()
            keep = scores >= train_cfg.score_threshold
            predictions.append(
                {
                    "boxes": output["boxes"].detach().cpu().numpy()[keep],
                    "scores": scores[keep],
                    "labels": output["labels"].detach().cpu().numpy()[keep],
                }
            )
            targets_for_eval.append(
                {
                    "boxes": target["boxes"].detach().cpu().numpy(),
                    "labels": target["labels"].detach().cpu().numpy(),
                }
            )

    metrics = compute_detection_metrics(
        predictions, targets_for_eval, dataset_cfg.num_classes, train_cfg.iou_threshold
    )
    metrics["loss"] = total_loss / max(num_batches, 1)

    if was_training:
        model.train()
    return metrics


def _resolve_class_label(dataset_cfg: DatasetConfig, index: int) -> str:
    if index < len(dataset_cfg.class_names):
        label = dataset_cfg.class_names[index]
    else:
        label = f"class_{index:02d}"

    if label.startswith("class_") and label[6:].isdigit():
        return f"class {int(label[6:]):02d}"
    return label


def format_epoch_metrics(
    epoch: int,
    train_loss: float,
    metrics: Dict[str, torch.Tensor | float | List[float]],
    dataset_cfg: DatasetConfig,
) -> List[str]:
    lines: List[str] = []

    val_loss = float(metrics.get("loss", float("nan")))
    map_value = float(metrics.get("mAP", float("nan")))

    summary = f"Epoch {epoch:02d} | train loss {train_loss:.4f}"
    if np.isfinite(val_loss):
        summary += f" | val loss {val_loss:.4f}"
    if np.isfinite(map_value):
        summary += f" | mAP {map_value:.4f}"
    lines.append(summary)

    precision = np.asarray(metrics.get("precision", []), dtype=float)
    recall = np.asarray(metrics.get("recall", []), dtype=float)
    tp = np.asarray(metrics.get("TP", []), dtype=int)
    fp = np.asarray(metrics.get("FP", []), dtype=int)
    fn = np.asarray(metrics.get("FN", []), dtype=int)
    ap = np.asarray(metrics.get("AP", []), dtype=float)
    gt_counter = np.asarray(metrics.get("gt_counter", np.zeros_like(tp)), dtype=int)

    num_classes = min(len(tp), dataset_cfg.num_classes)
    for cls_idx in range(num_classes):
        gt_value = int(gt_counter[cls_idx]) if gt_counter.size > cls_idx else 0
        tp_value = int(tp[cls_idx])
        fp_value = int(fp[cls_idx])
        fn_value = int(fn[cls_idx])

        if gt_value == 0 and tp_value == 0 and fp_value == 0 and fn_value == 0:
            continue

        label = _resolve_class_label(dataset_cfg, cls_idx)
        p_val = float(np.nan_to_num(precision[cls_idx], nan=0.0)) if precision.size > cls_idx else 0.0
        r_val = float(np.nan_to_num(recall[cls_idx], nan=0.0)) if recall.size > cls_idx else 0.0
        line = f"{label} | P={p_val:.3f} R={r_val:.3f}  TP={tp_value} FP={fp_value} FN={fn_value}"
        if ap.size > cls_idx and np.isfinite(ap[cls_idx]):
            line += f" AP={ap[cls_idx]:.3f}"
        lines.append(line)

    return lines


def save_checkpoint(model: nn.Module, path: Path) -> None:
    torch.save(model.state_dict(), path)
    LOGGER.info("Saved checkpoint to %s", path)


def main() -> None:
    args = parse_args()
    dataset_cfg, train_cfg = prepare_configs(args)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(train_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(dataset_cfg, train_cfg, device=device)

    train_dataset = ElectricalComponentsDataset(
        root=dataset_cfg.base_dir,
        split=dataset_cfg.train_split,
        class_names=dataset_cfg.class_names,
        use_augmentation=train_cfg.augmentation,
    )
    valid_dataset = ElectricalComponentsDataset(
        root=dataset_cfg.base_dir,
        split=dataset_cfg.valid_split,
        class_names=dataset_cfg.class_names,
        use_augmentation=False,
    )

    train_loader = create_data_loaders(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
    )
    valid_loader = create_data_loaders(
        valid_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=max(1, train_cfg.num_workers // 2),
    )

    optimizer = AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    scaler = GradScaler(enabled=train_cfg.amp and device.type == "cuda")

    best_map = -float("inf")
    history: List[Dict[str, float]] = []

    for epoch in range(1, train_cfg.epochs + 1):
        LOGGER.info("Epoch %s/%s", epoch, train_cfg.epochs)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, train_cfg.amp, train_cfg.log_every
        )
        if epoch % train_cfg.eval_interval == 0:
            metrics = evaluate(model, valid_loader, device, dataset_cfg, train_cfg)
            for line in format_epoch_metrics(epoch, train_loss, metrics, dataset_cfg):
                LOGGER.info(line)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_loss": float(metrics["loss"]),
                    "mAP": float(metrics["mAP"]),
                }
            )

            if metrics["mAP"] > best_map:
                best_map = float(metrics["mAP"])
                save_checkpoint(model, train_cfg.checkpoint_path)
        else:
            LOGGER.info(
                "Epoch %02d | train loss %.4f | evaluation skipped (eval_interval=%d)",
                epoch,
                train_loss,
                train_cfg.eval_interval,
            )

    (train_cfg.output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    LOGGER.info("Training complete. Best mAP: %.4f", best_map)


if __name__ == "__main__":
    main()
