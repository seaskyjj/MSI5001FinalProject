"""Inference utilities for evaluating trained models on the test split."""
from __future__ import annotations

import argparse
import logging
import pickle
import inspect
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.serialization import add_safe_globals
except ImportError:  # pragma: no cover - compatibility for older PyTorch
    add_safe_globals = None

from .config import (
    DEFAULT_CLASS_SCORE_THRESHOLDS,
    DatasetConfig,
    InferenceConfig,
    TrainingConfig,
)
from .dataset import ElectricalComponentsDataset, create_data_loaders
from .model import build_model
from .utils import (
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

LOGGER = logging.getLogger("inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DatasetConfig().base_dir)
    parser.add_argument("--split", default=DatasetConfig().test_split)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=InferenceConfig().output_dir)
    parser.add_argument("--score-threshold", type=float, default=InferenceConfig().score_threshold)
    parser.add_argument(
        "--class-threshold",
        action="append",
        default=[],
        metavar="CLS=THRESH",
        help="Override per-class score thresholds for inference",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=InferenceConfig().max_images)
    parser.add_argument("--num-classes", type=int, default=DatasetConfig().num_classes)
    parser.add_argument("--draw-ground-truth", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--pretrained-path", type=Path, default=TrainingConfig().pretrained_weights_path)
    parser.add_argument(
        "--fp-report",
        type=Path,
        help="Optional path to write a JSON report of images containing false positives.",
    )
    parser.add_argument(
        "--fp-list",
        type=Path,
        help="Optional path to write newline separated image stems that produced false positives.",
    )
    return parser.parse_args()


def _load_checkpoint_state(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    load_kwargs = {"map_location": device}
    try:
        checkpoint_obj = torch.load(path, **load_kwargs)
    except pickle.UnpicklingError:
        if add_safe_globals is not None:
            add_safe_globals([TrainingConfig])
        load_params = inspect.signature(torch.load).parameters
        if "weights_only" in load_params:
            load_kwargs["weights_only"] = False
        checkpoint_obj = torch.load(path, **load_kwargs)

    if isinstance(checkpoint_obj, dict) and "model" in checkpoint_obj:
        return checkpoint_obj["model"]

    return checkpoint_obj
@torch.no_grad()
def run_inference(args: argparse.Namespace) -> None:
    dataset_cfg = DatasetConfig(
        base_dir=args.data_dir,
        test_split=args.split,
        num_classes=args.num_classes,
    )
    class_thresholds = DEFAULT_CLASS_SCORE_THRESHOLDS.copy()
    overrides = parse_class_threshold_entries(args.class_threshold)
    class_thresholds.update(overrides)

    inference_cfg = InferenceConfig(
        score_threshold=args.score_threshold,
        max_images=args.max_images,
        output_dir=args.output_dir,
        draw_ground_truth=args.draw_ground_truth,
        class_score_thresholds=class_thresholds,
    )
    inference_cfg.ensure_directories()

    train_cfg = TrainingConfig(
        augmentation=False,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        pretrained_weights_path=args.pretrained_path,
        class_score_thresholds=class_thresholds,
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(dataset_cfg, train_cfg, device=device)
    state_dict = _load_checkpoint_state(args.checkpoint, device)
    model.load_state_dict(state_dict)
    model.eval()

    dataset = ElectricalComponentsDataset(
        root=dataset_cfg.base_dir,
        split=args.split,
        class_names=dataset_cfg.class_names,
        use_augmentation=False,
    )
    loader = create_data_loaders(dataset, batch_size=1, shuffle=False, num_workers=0)

    predictions: List[Dict[str, np.ndarray]] = []
    targets_for_eval: List[Dict[str, np.ndarray]] = []

    progress = tqdm(loader, desc="Infer")
    fp_records: List[Dict[str, object]] = []
    for idx, (images, targets) in enumerate(progress):
        image = images[0].to(device)
        output = model([image])[0]

        boxes_np = output["boxes"].detach().cpu().numpy()
        scores_np = output["scores"].detach().cpu().numpy()
        labels_np = output["labels"].detach().cpu().numpy().astype(np.int64, copy=False)
        keep = score_threshold_mask(
            scores_np,
            labels_np,
            inference_cfg.score_threshold,
            inference_cfg.class_score_thresholds,
        )
        boxes_np = boxes_np[keep]
        scores_np = scores_np[keep]
        labels_np = labels_np[keep].astype(np.int64, copy=True)
        if labels_np.size:
            labels_np -= 1

        target_boxes = targets[0]["boxes"].detach().cpu().numpy()
        target_labels = targets[0]["labels"].detach().cpu().numpy().astype(np.int64, copy=True)
        if target_labels.size:
            gt_keep = target_labels > 0
            target_boxes = target_boxes[gt_keep]
            target_labels = target_labels[gt_keep] - 1

        prediction_np = {
            "boxes": boxes_np,
            "scores": scores_np,
            "labels": labels_np,
        }
        target_np = {
            "boxes": target_boxes,
            "labels": target_labels,
        }

        predictions.append(prediction_np)
        targets_for_eval.append(target_np)

        fp_details = identify_false_positive_predictions(
            prediction_np,
            target_np,
            dataset_cfg.num_classes,
            args.iou_threshold,
        )
        if fp_details:
            image_id = dataset.image_stems[idx] if idx < len(dataset.image_stems) else f"{args.split}_{idx:04d}"
            fp_records.append(
                {
                    "image_id": image_id,
                    "false_positives": fp_details,
                }
            )

        if idx < inference_cfg.max_images:
            image_np = (images[0].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            output_path = inference_cfg.output_dir / f"{args.split}_{idx:04d}.png"
            save_detection_visual(
                image_np,
                prediction_np,
                target_np if args.draw_ground_truth else None,
                dataset_cfg.class_names,
                inference_cfg.score_threshold,
                inference_cfg.class_score_thresholds,
                args.draw_ground_truth,
                output_path,
            )

    metrics = compute_detection_metrics(
        predictions, targets_for_eval, dataset_cfg.num_classes, args.iou_threshold
    )
    metric_lines = format_epoch_metrics(
        epoch=None,
        train_loss=None,
        metrics=metrics,
        dataset_cfg=dataset_cfg,
        header=f"Inference @ IoU {args.iou_threshold:.2f}",
    )
    emit_metric_lines(metric_lines, logger=LOGGER)

    if args.fp_report:
        write_false_positive_report(
            fp_records,
            args.fp_report,
            split=args.split,
            score_threshold=inference_cfg.score_threshold,
            class_score_thresholds=inference_cfg.class_score_thresholds,
            iou_threshold=args.iou_threshold,
        )
        LOGGER.info(
            "Wrote false-positive report for %d images to %s",
            len(fp_records),
            args.fp_report,
        )

    if args.fp_list:
        write_false_positive_list(fp_records, args.fp_list)
        LOGGER.info(
            "Wrote %d image ids with false positives to %s",
            len({str(record["image_id"]) for record in fp_records}),
            args.fp_list,
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_inference(args)


if __name__ == "__main__":
    main()
