"""Training script for the electrical component Faster R-CNN detector."""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Set

import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import DEFAULT_CLASS_SCORE_THRESHOLDS, DatasetConfig, TrainingConfig
from .dataset import AugmentationParams, ElectricalComponentsDataset, create_data_loaders
from .model import build_model
from .utils import (
    MetricLogger,
    compute_detection_metrics,
    emit_metric_lines,
    format_epoch_metrics,
    identify_false_positive_predictions,
    parse_class_threshold_entries,
    save_detection_visual,
    score_threshold_mask,
    set_seed,
    write_false_positive_list,
    write_false_positive_report,
)

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
    parser.add_argument(
        "--mosaic-prob",
        type=float,
        default=TrainingConfig().mosaic_prob,
        help="Probability of applying mosaic augmentation (four-image collage)",
    )
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=TrainingConfig().mixup_prob,
        help="Probability of mixing an additional image into the current sample",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=TrainingConfig().mixup_alpha,
        help="Alpha parameter for the Beta distribution controlling MixUp strength",
    )
    parser.add_argument(
        "--scale-jitter", nargs=2, type=float, metavar=("MIN", "MAX"), default=None,
        help="Uniform scale jitter range applied to each image before flips/jitter",
    )
    parser.add_argument(
        "--rotation-prob",
        type=float,
        default=TrainingConfig().rotation_prob,
        help="Probability of applying a random in-place rotation.",
    )
    parser.add_argument(
        "--rotation-max-degrees",
        type=float,
        default=TrainingConfig().rotation_max_degrees,
        help="Maximum absolute angle in degrees for random rotations.",
    )
    parser.add_argument(
        "--affine-prob",
        type=float,
        default=TrainingConfig().affine_prob,
        help="Probability of applying a random affine transform (translate/scale/shear).",
    )
    parser.add_argument(
        "--affine-translate",
        nargs=2,
        type=float,
        metavar=("FRAC_X", "FRAC_Y"),
        default=None,
        help="Maximum absolute translation as a fraction of image width/height.",
    )
    parser.add_argument(
        "--affine-scale",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="Uniform scaling range applied during affine augmentation.",
    )
    parser.add_argument(
        "--affine-shear",
        nargs=2,
        type=float,
        metavar=("SHEAR_X", "SHEAR_Y"),
        default=None,
        help="Maximum absolute shear angles (degrees) for the affine augmentation.",
    )
    parser.add_argument("--small-object", action="store_true", help="Use smaller RPN anchors")
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument(
        "--class-threshold",
        action="append",
        default=[],
        metavar="CLS=THRESH",
        help="Override per-class score thresholds (e.g. --class-threshold 3=0.8)",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--checkpoint", type=Path, default=TrainingConfig().checkpoint_path)
    parser.add_argument("--pretrained-path", type=Path, default=TrainingConfig().pretrained_weights_path)
    parser.add_argument("--resume", action="store_true", help="Resume training from a saved checkpoint.")
    parser.add_argument(
        "--resume-path",
        type=Path,
        default=None,
        help="Optional path to the checkpoint used for resuming training. Defaults to the last-checkpoint file.",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--train-split", default=DatasetConfig().train_split)
    parser.add_argument("--valid-split", default=DatasetConfig().valid_split)
    parser.add_argument("--num-classes", type=int, default=DatasetConfig().num_classes)
    parser.add_argument(
        "--exclude-list",
        action="append",
        default=[],
        type=Path,
        help="Path(s) to files listing image stems to exclude from training (text or JSON).",
    )
    parser.add_argument(
        "--exclude-sample",
        action="append",
        default=[],
        help="Additional image stems to remove from the training split.",
    )
    parser.add_argument(
        "--fp-dir",
        type=Path,
        default=None,
        help="Directory where false-positive visualisations from the final epoch are written.",
    )
    parser.add_argument(
        "--fp-report",
        type=Path,
        help="Optional path to write a JSON report describing final-epoch false positives.",
    )
    parser.add_argument(
        "--fp-list",
        type=Path,
        help="Optional path to write newline separated image stems containing final-epoch false positives.",
    )
    parser.add_argument(
        "--fp-class",
        action="append",
        type=int,
        default=[],
        help="Restrict false-positive exports to specific classes (repeatable).",
    )
    return parser.parse_args()


def _normalise_stem(value: str) -> str:
    return Path(value).stem if value else value


def _load_exclusions_from_file(path: Path) -> List[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        LOGGER.warning("Unable to read exclude list %s: %s", path, exc)
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return [_normalise_stem(line) for line in lines]

    stems: List[str] = []
    if isinstance(payload, dict):
        for key in ("stems", "image_ids", "images"):
            if key in payload and isinstance(payload[key], list):
                stems.extend(_normalise_stem(str(item)) for item in payload[key])
        if "false_positive_images" in payload and isinstance(payload["false_positive_images"], list):
            for entry in payload["false_positive_images"]:
                if isinstance(entry, dict):
                    if "image_id" in entry:
                        stems.append(_normalise_stem(str(entry["image_id"])))
                    elif "image" in entry:
                        stems.append(_normalise_stem(str(entry["image"])))
    elif isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, dict):
                if "image_id" in entry:
                    stems.append(_normalise_stem(str(entry["image_id"])))
                elif "image" in entry:
                    stems.append(_normalise_stem(str(entry["image"])))
            else:
                stems.append(_normalise_stem(str(entry)))

    return [stem for stem in stems if stem]


def _resolve_exclusions(paths: List[Path], samples: List[str]) -> List[str]:
    stems: Set[str] = {_normalise_stem(sample) for sample in samples if sample}
    for path in paths:
        if path is None:
            continue
        if not path.exists():
            LOGGER.warning("Exclude list %s does not exist; skipping.", path)
            continue
        stems.update(_load_exclusions_from_file(path))
    return sorted(stem for stem in stems if stem)


def prepare_configs(args: argparse.Namespace) -> tuple[DatasetConfig, TrainingConfig]:
    default_train_cfg = TrainingConfig()
    dataset_cfg = DatasetConfig(
        base_dir=args.data_dir,
        train_split=args.train_split,
        valid_split=args.valid_split,
        num_classes=args.num_classes,
    )

    class_thresholds = DEFAULT_CLASS_SCORE_THRESHOLDS.copy()
    overrides = parse_class_threshold_entries(args.class_threshold)
    class_thresholds.update(overrides)

    exclude_samples = _resolve_exclusions(args.exclude_list, args.exclude_sample)

    if args.fp_class:
        fp_class_values = sorted({int(value) for value in args.fp_class})
        fp_classes = tuple(fp_class_values)
    else:
        fp_classes = default_train_cfg.fp_classes

    fp_dir = args.fp_dir if args.fp_dir is not None else default_train_cfg.fp_visual_dir

    if args.scale_jitter:
        scale_min, scale_max = args.scale_jitter
    else:
        scale_min = default_train_cfg.scale_jitter_min
        scale_max = default_train_cfg.scale_jitter_max

    rotation_prob = args.rotation_prob
    rotation_max = args.rotation_max_degrees

    affine_prob = args.affine_prob
    if args.affine_translate is not None:
        affine_translate = (float(args.affine_translate[0]), float(args.affine_translate[1]))
    else:
        affine_translate = default_train_cfg.affine_translate

    if args.affine_scale is not None:
        affine_scale_range = (float(args.affine_scale[0]), float(args.affine_scale[1]))
    else:
        affine_scale_range = default_train_cfg.affine_scale_range

    if args.affine_shear is not None:
        affine_shear = (float(args.affine_shear[0]), float(args.affine_shear[1]))
    else:
        affine_shear = default_train_cfg.affine_shear

    last_checkpoint_path = args.checkpoint.parent / "last_checkpoint.pth"
    if args.resume_path is not None:
        resume_path = args.resume_path
    elif args.resume:
        resume_path = last_checkpoint_path
    else:
        resume_path = None

    train_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        augmentation=not args.no_augmentation,
        mosaic_prob=args.mosaic_prob,
        mixup_prob=args.mixup_prob,
        mixup_alpha=args.mixup_alpha,
        scale_jitter_min=scale_min,
        scale_jitter_max=scale_max,
        rotation_prob=rotation_prob,
        rotation_max_degrees=rotation_max,
        affine_prob=affine_prob,
        affine_translate=affine_translate,
        affine_scale_range=affine_scale_range,
        affine_shear=affine_shear,
        small_object=args.small_object,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        eval_interval=args.eval_interval,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        pretrained_weights_path=args.pretrained_path,
        resume=args.resume,
        resume_path=resume_path,
        last_checkpoint_path=last_checkpoint_path,
        log_every=args.log_every,
        class_score_thresholds=class_thresholds,
        exclude_samples=tuple(exclude_samples),
        fp_visual_dir=fp_dir,
        fp_report_path=args.fp_report,
        fp_list_path=args.fp_list,
        fp_classes=fp_classes,
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
        autocast_enabled = amp and device.type == "cuda"
        autocast_context = (
            torch.amp.autocast(device_type="cuda") if autocast_enabled else contextlib.nullcontext()
        )
        with autocast_context:
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
    *,
    dataset: ElectricalComponentsDataset | None = None,
    collect_details: bool = False,
) -> tuple[Dict[str, torch.Tensor | float | List[float]], List[Dict[str, object]]]:
    was_training = model.training
    model.eval()

    predictions = []
    targets_for_eval = []
    total_loss = 0.0
    num_batches = 0

    sample_details: List[Dict[str, object]] = []
    dataset_ref = dataset if dataset is not None else getattr(loader, "dataset", None)

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets_device = move_to_device(targets, device)

        model.train()
        loss_dict = model(images, targets_device)
        total_loss += sum(loss_dict.values()).item()
        num_batches += 1
        model.eval()

        outputs = model(images)
        for output, target_device, target_cpu in zip(outputs, targets_device, targets):
            scores = output["scores"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()
            keep = score_threshold_mask(
                scores,
                labels,
                train_cfg.score_threshold,
                train_cfg.class_score_thresholds,
            )
            prediction_np = {
                "boxes": output["boxes"].detach().cpu().numpy()[keep],
                "scores": scores[keep],
                "labels": labels[keep],
            }
            target_np = {
                "boxes": target_device["boxes"].detach().cpu().numpy(),
                "labels": target_device["labels"].detach().cpu().numpy(),
            }

            predictions.append(prediction_np)
            targets_for_eval.append(target_np)

            if collect_details:
                image_identifier = target_cpu.get("image_id", -1)
                if isinstance(image_identifier, torch.Tensor):
                    image_index = int(image_identifier.item())
                else:
                    try:
                        image_index = int(image_identifier)
                    except Exception:
                        image_index = -1
                if dataset_ref is not None and 0 <= image_index < len(getattr(dataset_ref, "image_stems", [])):
                    image_id = dataset_ref.image_stems[image_index]
                else:
                    image_id = f"{dataset_cfg.valid_split}_{len(sample_details):04d}"

                fp_info = identify_false_positive_predictions(
                    prediction_np,
                    target_np,
                    dataset_cfg.num_classes,
                    train_cfg.iou_threshold,
                )

                sample_details.append(
                    {
                        "image_index": image_index,
                        "image_id": image_id,
                        "prediction": prediction_np,
                        "target": target_np,
                        "false_positives": fp_info,
                    }
                )

    metrics = compute_detection_metrics(
        predictions, targets_for_eval, dataset_cfg.num_classes, train_cfg.iou_threshold
    )
    metrics["loss"] = total_loss / max(num_batches, 1)

    if was_training:
        model.train()
    return metrics, sample_details


def save_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
    epoch: int,
    best_map: float,
    history: List[Dict[str, float]],
    train_cfg: TrainingConfig,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None and getattr(scaler, "is_enabled", lambda: True)() else None,
        "epoch": epoch,
        "best_map": float(best_map),
        "history": list(history),
        "config": train_cfg,
    }
    torch.save(checkpoint, path)
    LOGGER.info("Saved checkpoint to %s (epoch %d, best mAP %.4f)", path, epoch, best_map)


def load_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
    device: torch.device,
) -> tuple[int, float, List[Dict[str, float]]]:
    if not path.exists():
        LOGGER.warning("Checkpoint %s does not exist; starting fresh.", path)
        return 1, -float("inf"), []

    checkpoint_obj = torch.load(path, map_location=device)

    if isinstance(checkpoint_obj, dict) and "model" in checkpoint_obj:
        model.load_state_dict(checkpoint_obj["model"])
        if optimizer is not None and checkpoint_obj.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint_obj["optimizer"])
        if scaler is not None and checkpoint_obj.get("scaler") is not None:
            try:
                scaler.load_state_dict(checkpoint_obj["scaler"])
            except Exception as exc:  # pragma: no cover - fallback if scaler config changed
                LOGGER.warning("Unable to load scaler state from %s: %s", path, exc)
        epoch = int(checkpoint_obj.get("epoch", 0))
        best_map = float(checkpoint_obj.get("best_map", -float("inf")))
        history = checkpoint_obj.get("history", [])
        LOGGER.info(
            "Loaded checkpoint from %s (epoch %d, best mAP %.4f)",
            path,
            epoch,
            best_map,
        )
        return epoch + 1, best_map, list(history)

    # Legacy checkpoint: assume it's a plain state dict.
    model.load_state_dict(checkpoint_obj)
    LOGGER.info("Loaded model weights from legacy checkpoint %s", path)
    return 1, -float("inf"), []


def export_false_positive_visuals(
    *,
    dataset: ElectricalComponentsDataset,
    dataset_cfg: DatasetConfig,
    train_cfg: TrainingConfig,
    sample_details: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Export false-positive visuals for the configured classes and return records."""

    if not train_cfg.fp_visual_dir:
        return []

    relevant_classes = {int(cls) for cls in train_cfg.fp_classes}
    if not relevant_classes:
        return []

    output_dir = train_cfg.fp_visual_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep only the last-epoch visuals by clearing previous outputs.
    for existing in output_dir.glob("*.png"):
        try:
            existing.unlink()
        except OSError:
            LOGGER.debug("Unable to remove previous FP visual %s", existing)

    fp_records: List[Dict[str, object]] = []

    for detail in sample_details:
        raw_records = detail.get("false_positives", [])
        if not raw_records:
            continue

        filtered = [fp for fp in raw_records if int(fp.get("class", -1)) in relevant_classes]
        if not filtered:
            continue

        image_id = str(detail.get("image_id", detail.get("image_index", "unknown")))
        try:
            image_np, _, _ = dataset._load_raw_sample(image_id)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Skipping FP visual for %s due to load error: %s", image_id, exc)
            continue

        output_path = output_dir / f"{image_id}.png"
        save_detection_visual(
            image_np,
            detail.get("prediction", {}),
            detail.get("target", {}),
            dataset_cfg.class_names,
            train_cfg.score_threshold,
            train_cfg.class_score_thresholds,
            True,
            output_path,
        )
        fp_records.append({"image_id": image_id, "false_positives": filtered})

    return fp_records


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
        transform=AugmentationParams(
            mosaic_prob=train_cfg.mosaic_prob,
            mixup_prob=train_cfg.mixup_prob,
            mixup_alpha=train_cfg.mixup_alpha,
            scale_jitter_range=(train_cfg.scale_jitter_min, train_cfg.scale_jitter_max),
            rotation_prob=train_cfg.rotation_prob,
            rotation_max_degrees=train_cfg.rotation_max_degrees,
            affine_prob=train_cfg.affine_prob,
            affine_translate=train_cfg.affine_translate,
            affine_scale_range=train_cfg.affine_scale_range,
            affine_shear=train_cfg.affine_shear,
        ),
        use_augmentation=train_cfg.augmentation,
        exclude_stems=train_cfg.exclude_samples,
    )
    valid_dataset = ElectricalComponentsDataset(
        root=dataset_cfg.base_dir,
        split=dataset_cfg.valid_split,
        class_names=dataset_cfg.class_names,
        use_augmentation=False,
    )

    if train_cfg.exclude_samples:
        sample_preview = ", ".join(train_cfg.exclude_samples[:5])
        if len(train_cfg.exclude_samples) > 5:
            sample_preview += ", ..."
        LOGGER.info(
            "Excluding %d training samples: %s",
            len(train_cfg.exclude_samples),
            sample_preview,
        )

    train_loader = create_data_loaders(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
    )
    if train_cfg.num_workers > 1:
        valid_workers = max(1, train_cfg.num_workers // 2)
    else:
        valid_workers = train_cfg.num_workers

    valid_loader = create_data_loaders(
        valid_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=valid_workers,
    )

    optimizer = AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    scaler = GradScaler(enabled=train_cfg.amp and device.type == "cuda")

    start_epoch = 1
    best_map = -float("inf")
    history: List[Dict[str, float]] = []

    if train_cfg.resume or train_cfg.resume_path is not None:
        resume_source = train_cfg.resume_path or train_cfg.last_checkpoint_path
        start_epoch, best_map, history = load_checkpoint(
            path=resume_source,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

    for epoch in range(start_epoch, train_cfg.epochs + 1):
        LOGGER.info("Epoch %s/%s", epoch, train_cfg.epochs)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, train_cfg.amp, train_cfg.log_every
        )

        should_evaluate = (epoch % train_cfg.eval_interval == 0) or (epoch == train_cfg.epochs)
        collect_fp = should_evaluate and epoch == train_cfg.epochs

        if should_evaluate:
            metrics, sample_details = evaluate(
                model,
                valid_loader,
                device,
                dataset_cfg,
                train_cfg,
                dataset=valid_dataset,
                collect_details=collect_fp,
            )
            metric_lines = format_epoch_metrics(epoch, train_loss, metrics, dataset_cfg)
            emit_metric_lines(metric_lines, logger=LOGGER)

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
                save_checkpoint(
                    path=train_cfg.checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    best_map=best_map,
                    history=history,
                    train_cfg=train_cfg,
                )

            if collect_fp:
                fp_records = export_false_positive_visuals(
                    dataset=valid_dataset,
                    dataset_cfg=dataset_cfg,
                    train_cfg=train_cfg,
                    sample_details=sample_details,
                )
                if train_cfg.fp_report_path:
                    write_false_positive_report(
                        fp_records,
                        train_cfg.fp_report_path,
                        split=dataset_cfg.valid_split,
                        score_threshold=train_cfg.score_threshold,
                        class_score_thresholds=train_cfg.class_score_thresholds,
                        iou_threshold=train_cfg.iou_threshold,
                    )
                    LOGGER.info(
                        "Saved false-positive report for %d images to %s",
                        len(fp_records),
                        train_cfg.fp_report_path,
                    )
                if train_cfg.fp_list_path:
                    write_false_positive_list(fp_records, train_cfg.fp_list_path)
                    LOGGER.info(
                        "Saved list of %d false-positive image ids to %s",
                        len({str(record["image_id"]) for record in fp_records}),
                        train_cfg.fp_list_path,
                    )
        else:
            LOGGER.info(
                "Epoch %02d | train loss %.4f | evaluation skipped (eval_interval=%d)",
                epoch,
                train_loss,
                train_cfg.eval_interval,
            )

        save_checkpoint(
            path=train_cfg.last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_map=best_map,
            history=history,
            train_cfg=train_cfg,
        )

    (train_cfg.output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    LOGGER.info("Training complete. Best mAP: %.4f", best_map)


if __name__ == "__main__":
    main()
