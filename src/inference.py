"""Inference utilities for evaluating trained models on the test split."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image as PILImage, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import DatasetConfig, InferenceConfig, TrainingConfig
from .dataset import ElectricalComponentsDataset, create_data_loaders
from .model import build_model
from .utils import compute_detection_metrics, set_seed

LOGGER = logging.getLogger("inference")
DEFAULT_COLORS = [
    "#FF6B6B",
    "#4ECDC4",
    "#556270",
    "#C44D58",
    "#FFB347",
    "#6B5B95",
    "#88B04B",
    "#92A8D1",
    "#955251",
    "#B565A7",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DatasetConfig().base_dir)
    parser.add_argument("--split", default=DatasetConfig().test_split)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=InferenceConfig().output_dir)
    parser.add_argument("--score-threshold", type=float, default=InferenceConfig().score_threshold)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=InferenceConfig().max_images)
    parser.add_argument("--num-classes", type=int, default=DatasetConfig().num_classes)
    parser.add_argument("--draw-ground-truth", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--pretrained-path", type=Path, default=TrainingConfig().pretrained_weights_path)
    return parser.parse_args()


def load_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=14)
    except Exception:  # pragma: no cover - fallback when font unavailable
        return ImageFont.load_default()


def draw_boxes(
    image: np.ndarray,
    prediction: Dict[str, np.ndarray],
    target: Dict[str, np.ndarray] | None,
    class_names: List[str],
    score_threshold: float,
    draw_ground_truth: bool,
    output_path: Path,
) -> None:
    pil = PILImage.fromarray(image)
    draw = ImageDraw.Draw(pil)
    font = load_font()

    colors = DEFAULT_COLORS
    boxes = prediction["boxes"]
    labels = prediction["labels"].astype(int)
    scores = prediction["scores"]

    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue
        color = colors[label % len(colors)]
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        caption = f"{class_names[label]} {score:.2f}"
        text_size = draw.textlength(caption, font=font)
        draw.rectangle([x1, y1 - 16, x1 + text_size + 8, y1], fill=color)
        draw.text((x1 + 4, y1 - 14), caption, fill="white", font=font)

    if draw_ground_truth and target is not None:
        gt_boxes = target["boxes"]
        gt_labels = target["labels"].astype(int)
        for box, label in zip(gt_boxes, gt_labels):
            color = "#FFFFFF"
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
            caption = f"GT {class_names[label]}"
            text_size = draw.textlength(caption, font=font)
            draw.rectangle([x1, y2, x1 + text_size + 6, y2 + 14], fill=color)
            draw.text((x1 + 3, y2), caption, fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(output_path)


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> None:
    dataset_cfg = DatasetConfig(
        base_dir=args.data_dir,
        test_split=args.split,
        num_classes=args.num_classes,
    )
    inference_cfg = InferenceConfig(
        score_threshold=args.score_threshold,
        max_images=args.max_images,
        output_dir=args.output_dir,
        draw_ground_truth=args.draw_ground_truth,
    )
    inference_cfg.ensure_directories()

    train_cfg = TrainingConfig(
        augmentation=False,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        pretrained_weights_path=args.pretrained_path,
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(dataset_cfg, train_cfg, device=device)
    state_dict = torch.load(args.checkpoint, map_location=device)
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
    for idx, (images, targets) in enumerate(progress):
        image = images[0].to(device)
        output = model([image])[0]

        prediction_np = {
            "boxes": output["boxes"].detach().cpu().numpy(),
            "scores": output["scores"].detach().cpu().numpy(),
            "labels": output["labels"].detach().cpu().numpy(),
        }
        target_np = {
            "boxes": targets[0]["boxes"].detach().cpu().numpy(),
            "labels": targets[0]["labels"].detach().cpu().numpy(),
        }

        predictions.append(prediction_np)
        targets_for_eval.append(target_np)

        if idx < inference_cfg.max_images:
            image_np = (images[0].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            output_path = inference_cfg.output_dir / f"{args.split}_{idx:04d}.png"
            draw_boxes(
                image_np,
                prediction_np,
                target_np if args.draw_ground_truth else None,
                dataset_cfg.class_names,
                args.score_threshold,
                args.draw_ground_truth,
                output_path,
            )

    metrics = compute_detection_metrics(
        predictions, targets_for_eval, dataset_cfg.num_classes, args.iou_threshold
    )
    LOGGER.info("mAP@%.2f: %.4f", args.iou_threshold, metrics["mAP"])
    for cls_idx, ap in enumerate(metrics["AP"]):
        if np.isnan(ap):
            LOGGER.info("Class %-15s: AP unavailable (no ground truth)", dataset_cfg.class_names[cls_idx])
        else:
            LOGGER.info(
                "Class %-15s | AP: %.3f | Precision: %.3f | Recall: %.3f | TP: %d FP: %d FN: %d",
                dataset_cfg.class_names[cls_idx],
                float(ap),
                float(metrics["precision"][cls_idx]),
                float(metrics["recall"][cls_idx]),
                int(metrics["TP"][cls_idx]),
                int(metrics["FP"][cls_idx]),
                int(metrics["FN"][cls_idx]),
            )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_inference(args)


if __name__ == "__main__":
    main()
