"""Utility helpers for training and evaluating the detection model."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image as PILImage, ImageDraw, ImageFont
from torch import Tensor

from .config import DatasetConfig


def set_seed(seed: int) -> None:
    """Set seeds for the Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_boxes_and_labels(
    boxes: Tensor, labels: Tensor, height: int, width: int, min_size: float = 1.0
) -> Tuple[Tensor, Tensor]:
    """Clamp bounding boxes to the image size and drop invalid boxes."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4).float(), labels.reshape(0).long()

    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, float(width))
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, float(height))

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep = (widths > min_size) & (heights > min_size)

    if keep.sum() == 0:
        return boxes.new_zeros((0, 4)), labels.new_zeros((0,), dtype=torch.long)
    return boxes[keep].float(), labels[keep].long()


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute the IoU matrix between two sets of boxes in ``xyxy`` format."""
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    inter_x1 = np.maximum(x11, x21.T)
    inter_y1 = np.maximum(y11, y21.T)
    inter_x2 = np.minimum(x12, x22.T)
    inter_y2 = np.minimum(y12, y22.T)

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0.0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0.0, a_max=None)
    inter_area = inter_w * inter_h

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)

    union = area1 + area2.T - inter_area
    return np.divide(inter_area, union, out=np.zeros_like(inter_area), where=union > 0)


def compute_average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute the interpolated Average Precision (AP) following COCO style."""
    if recalls.size == 0 or precisions.size == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    recall_points = np.linspace(0, 1, 101)
    precision_interp = np.interp(recall_points, mrec, mpre)
    return float(np.trapz(precision_interp, recall_points))


def accumulate_classification_stats(
    predictions: Sequence[Dict[str, np.ndarray]],
    targets: Sequence[Dict[str, np.ndarray]],
    num_classes: int,
    iou_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[float]], List[List[int]], np.ndarray]:
    """Accumulate TP/FP/FN statistics and matched predictions for AP computation."""
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)
    scores: List[List[float]] = [[] for _ in range(num_classes)]
    matches: List[List[int]] = [[] for _ in range(num_classes)]
    gt_counter = np.zeros(num_classes, dtype=np.int64)

    for pred, tgt in zip(predictions, targets):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_labels = pred["labels"].astype(np.int64)

        gt_boxes = tgt["boxes"]
        gt_labels = tgt["labels"].astype(np.int64)

        unique_classes = np.unique(np.concatenate((pred_labels, gt_labels)))
        for cls in unique_classes:
            pb = pred_boxes[pred_labels == cls]
            ps = pred_scores[pred_labels == cls]
            tb = gt_boxes[gt_labels == cls]
            gt_counter[cls] += len(tb)

            if len(tb) == 0:
                fp[cls] += len(pb)
                scores[cls].extend(ps.tolist())
                matches[cls].extend([0] * len(pb))
                continue

            order = np.argsort(-ps)
            pb = pb[order]
            ps = ps[order]
            iou_matrix = compute_iou_matrix(pb, tb)

            matched_gt: set[int] = set()
            for det_idx, score in enumerate(ps):
                if tb.size == 0:
                    fp[cls] += 1
                    scores[cls].append(float(score))
                    matches[cls].append(0)
                    continue

                best_gt = int(np.argmax(iou_matrix[det_idx]))
                best_iou = iou_matrix[det_idx, best_gt]

                if best_iou >= iou_threshold and best_gt not in matched_gt:
                    tp[cls] += 1
                    matched_gt.add(best_gt)
                    scores[cls].append(float(score))
                    matches[cls].append(1)
                else:
                    fp[cls] += 1
                    scores[cls].append(float(score))
                    matches[cls].append(0)

            fn[cls] += len(tb) - len(matched_gt)

    return tp, fp, fn, scores, matches, gt_counter


def identify_false_positive_predictions(
    prediction: Dict[str, np.ndarray],
    target: Dict[str, np.ndarray],
    num_classes: int,
    iou_threshold: float,
) -> List[Dict[str, Union[int, float, List[float]]]]:
    """Return detailed records for false-positive detections in a single sample."""

    boxes = np.asarray(prediction.get("boxes", np.empty((0, 4), dtype=np.float32)), dtype=np.float32)
    scores = np.asarray(prediction.get("scores", np.empty((0,), dtype=np.float32)), dtype=np.float32)
    labels = np.asarray(prediction.get("labels", np.empty((0,), dtype=np.int64)), dtype=np.int64)

    gt_boxes = np.asarray(target.get("boxes", np.empty((0, 4), dtype=np.float32)), dtype=np.float32)
    gt_labels = np.asarray(target.get("labels", np.empty((0,), dtype=np.int64)), dtype=np.int64)

    if boxes.size == 0:
        return []

    fp_records: List[Dict[str, Union[int, float, List[float]]]] = []
    unique_classes = np.unique(labels) if labels.size else np.asarray([], dtype=np.int64)

    for cls in unique_classes:
        cls = int(cls)
        cls_mask = labels == cls
        cls_indices = np.nonzero(cls_mask)[0]
        if cls_indices.size == 0:
            continue

        pb = boxes[cls_mask]
        ps = scores[cls_mask]
        order = np.argsort(-ps)
        pb_sorted = pb[order]
        ps_sorted = ps[order]
        original_indices = cls_indices[order]

        tb = gt_boxes[gt_labels == cls]
        iou_matrix = compute_iou_matrix(pb_sorted, tb) if tb.size else np.zeros((pb_sorted.shape[0], 0), dtype=np.float32)
        matched_gt: set[int] = set()

        for rank, (pred_idx, score_value) in enumerate(zip(original_indices, ps_sorted)):
            if iou_matrix.shape[1]:
                row = iou_matrix[rank]
                best_gt = int(row.argmax())
                best_iou = float(row[best_gt])
            else:
                best_gt = -1
                best_iou = 0.0

            is_true_positive = (
                iou_matrix.shape[1] > 0
                and best_iou >= iou_threshold
                and best_gt not in matched_gt
            )

            if is_true_positive:
                matched_gt.add(best_gt)
                continue

            fp_records.append(
                {
                    "index": int(pred_idx),
                    "class": cls,
                    "score": float(score_value),
                    "best_iou": best_iou,
                    "box": boxes[pred_idx].astype(float).tolist(),
                }
            )

    return fp_records


def compute_detection_metrics(
    predictions: Sequence[Dict[str, np.ndarray]],
    targets: Sequence[Dict[str, np.ndarray]],
    num_classes: int,
    iou_threshold: float,
) -> Dict[str, np.ndarray]:
    """Compute per-class metrics and mAP for detection results."""
    tp, fp, fn, scores, matches, gt_counter = accumulate_classification_stats(
        predictions, targets, num_classes, iou_threshold
    )

    precision = np.divide(tp, np.clip(tp + fp, a_min=1, a_max=None))
    recall = np.divide(tp, np.clip(tp + fn, a_min=1, a_max=None))

    ap = np.zeros(num_classes, dtype=np.float32)
    for cls in range(num_classes):
        if gt_counter[cls] == 0:
            ap[cls] = np.nan
            continue
        if not scores[cls]:
            ap[cls] = 0.0
            continue

        order = np.argsort(-np.asarray(scores[cls]))
        match_array = np.asarray(matches[cls], dtype=np.int32)[order]
        cumulative_tp = np.cumsum(match_array)
        cumulative_fp = np.cumsum(1 - match_array)

        recalls = cumulative_tp / gt_counter[cls]
        precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
        ap[cls] = compute_average_precision(recalls, precisions)

    valid_ap = ap[np.isfinite(ap)]
    map_value = float(valid_ap.mean()) if valid_ap.size else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "AP": ap,
        "mAP": map_value,
        "gt_counter": gt_counter,
    }


def score_threshold_mask(
    scores: np.ndarray,
    labels: np.ndarray,
    default_threshold: float,
    class_thresholds: Mapping[int, float],
) -> np.ndarray:
    """Return a boolean mask keeping predictions that pass per-class thresholds."""

    if scores.size == 0:
        return np.zeros_like(scores, dtype=bool)

    thresholds = np.full(scores.shape, default_threshold, dtype=scores.dtype)
    if class_thresholds:
        labels_int = labels.astype(np.int64, copy=False)
        for cls, value in class_thresholds.items():
            thresholds[labels_int == int(cls)] = float(value)

    return scores >= thresholds


def parse_class_threshold_entries(entries: Sequence[str]) -> Dict[int, float]:
    """Parse ``CLS=THRESH`` strings into a mapping of per-class thresholds."""

    thresholds: Dict[int, float] = {}
    for entry in entries:
        if not entry:
            continue

        if "=" in entry:
            key, value = entry.split("=", 1)
        elif ":" in entry:
            key, value = entry.split(":", 1)
        else:
            raise ValueError(f"Invalid class threshold format: {entry!r}")

        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid class threshold entry: {entry!r}")

        thresholds[int(key)] = float(value)

    return thresholds


def running_in_ipython_kernel() -> bool:
    """Return ``True`` when executing inside an IPython/Jupyter kernel."""

    try:  # ``IPython`` may be absent in some execution environments.
        from IPython import get_ipython  # type: ignore
    except Exception:  # pragma: no cover - depends on environment
        return False

    shell = get_ipython()
    return bool(shell and getattr(shell, "kernel", None))


def emit_metric_lines(
    lines: Sequence[str],
    *,
    logger: Optional[logging.Logger] = None,
    force_print: Optional[bool] = None,
) -> None:
    """Log metric lines and optionally echo them to ``stdout``.

    Kaggle notebooks buffer ``logging`` output differently from regular Python
    scripts, so we proactively mirror the messages with ``print`` when we detect
    an IPython kernel.  Callers may override this behaviour by passing
    ``force_print`` explicitly.
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    should_print = force_print if force_print is not None else running_in_ipython_kernel()

    for line in lines:
        if logger is not None:
            logger.info(line)
        if should_print:
            print(line)


def _resolve_class_label(dataset_cfg: DatasetConfig, index: int) -> str:
    if index < len(dataset_cfg.class_names):
        label = dataset_cfg.class_names[index]
    else:
        label = f"class_{index:02d}"

    if label.startswith("class_") and label[6:].isdigit():
        return f"class {int(label[6:]):02d}"
    return label


def format_epoch_metrics(
    epoch: Optional[int],
    train_loss: Optional[float],
    metrics: Dict[str, torch.Tensor | float | List[float]],
    dataset_cfg: DatasetConfig,
    *,
    header: Optional[str] = None,
) -> List[str]:
    lines: List[str] = []

    val_loss = float(metrics.get("loss", float("nan")))
    map_value = float(metrics.get("mAP", float("nan")))

    if header is not None:
        summary = header
    elif epoch is not None:
        summary = f"Epoch {epoch:02d}"
    else:
        summary = "Metrics"

    if train_loss is not None and np.isfinite(train_loss):
        summary += f" | train loss {train_loss:.4f}"
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
        tp_value = int(tp[cls_idx]) if tp.size > cls_idx else 0
        fp_value = int(fp[cls_idx]) if fp.size > cls_idx else 0
        fn_value = int(fn[cls_idx]) if fn.size > cls_idx else 0

        if gt_value == 0 and tp_value == 0 and fp_value == 0 and fn_value == 0:
            continue

        label = _resolve_class_label(dataset_cfg, cls_idx)
        p_val = (
            float(np.nan_to_num(precision[cls_idx], nan=0.0))
            if precision.size > cls_idx
            else 0.0
        )
        r_val = (
            float(np.nan_to_num(recall[cls_idx], nan=0.0))
            if recall.size > cls_idx
            else 0.0
        )
        line = f"{label} | P={p_val:.3f} R={r_val:.3f}  TP={tp_value} FP={fp_value} FN={fn_value}"
        if ap.size > cls_idx and np.isfinite(ap[cls_idx]):
            line += f" AP={ap[cls_idx]:.3f}"
        lines.append(line)

    return lines


def load_default_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a truetype font when available, otherwise fall back to default."""

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=14)
    except Exception:  # pragma: no cover - fallback when font unavailable
        return ImageFont.load_default()


def _resolve_class_name(class_names: Sequence[str], label: int) -> str:
    if 0 <= label < len(class_names):
        name = class_names[label]
    else:
        name = f"class_{label:02d}"

    if name.startswith("class_") and name[6:].isdigit():
        return f"class {int(name[6:]):02d}"
    return name


def render_detections(
    image: np.ndarray,
    prediction: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray] | None,
    class_names: Sequence[str],
    score_threshold: float,
    class_thresholds: Mapping[int, float],
    draw_ground_truth: bool,
) -> PILImage:
    """Render detection predictions (and optional ground truth) onto an image."""

    if image.dtype != np.uint8:
        image_array = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image_array = image

    pil = PILImage.fromarray(image_array)
    draw = ImageDraw.Draw(pil)
    font = load_default_font()

    boxes = np.asarray(prediction.get("boxes", np.empty((0, 4))), dtype=float)
    labels = np.asarray(prediction.get("labels", np.empty((0,), dtype=int)), dtype=int)
    scores = np.asarray(prediction.get("scores", np.empty((0,), dtype=float)), dtype=float)

    for box, label, score in zip(boxes, labels, scores):
        threshold = class_thresholds.get(int(label), score_threshold)
        if score < threshold:
            continue

        color = DEFAULT_COLORS[int(label) % len(DEFAULT_COLORS)] if len(DEFAULT_COLORS) else "#FF6B6B"
        x1, y1, x2, y2 = [float(coord) for coord in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        caption = f"{_resolve_class_name(class_names, int(label))} {score:.2f}"
        text_width = draw.textlength(caption, font=font)
        draw.rectangle([x1, y1 - 16, x1 + text_width + 8, y1], fill=color)
        draw.text((x1 + 4, y1 - 14), caption, fill="white", font=font)

    if draw_ground_truth and target is not None:
        gt_boxes = np.asarray(target.get("boxes", np.empty((0, 4))), dtype=float)
        gt_labels = np.asarray(target.get("labels", np.empty((0,), dtype=int)), dtype=int)
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = [float(coord) for coord in box]
            draw.rectangle([x1, y1, x2, y2], outline="#FFFFFF", width=1)
            caption = f"GT {_resolve_class_name(class_names, int(label))}"
            text_width = draw.textlength(caption, font=font)
            draw.rectangle([x1, y2, x1 + text_width + 6, y2 + 14], fill="#FFFFFF")
            draw.text((x1 + 3, y2), caption, fill="black", font=font)

    return pil


def save_detection_visual(
    image: np.ndarray,
    prediction: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray] | None,
    class_names: Sequence[str],
    score_threshold: float,
    class_thresholds: Mapping[int, float],
    draw_ground_truth: bool,
    output_path: Path,
) -> None:
    """Render and persist a detection visualisation to ``output_path``."""

    visual = render_detections(
        image,
        prediction,
        target,
        class_names,
        score_threshold,
        class_thresholds,
        draw_ground_truth,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visual.save(output_path)


def write_false_positive_report(
    fp_records: Sequence[Dict[str, object]],
    report_path: Path,
    *,
    split: str,
    score_threshold: float,
    class_score_thresholds: Mapping[int, float],
    iou_threshold: float,
) -> None:
    """Serialise detailed false-positive information to JSON."""

    report_payload = {
        "split": split,
        "score_threshold": score_threshold,
        "class_score_thresholds": dict(class_score_thresholds),
        "iou_threshold": iou_threshold,
        "false_positive_images": list(fp_records),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False))


def write_false_positive_list(fp_records: Sequence[Dict[str, object]], list_path: Path) -> None:
    """Write newline separated image identifiers that triggered false positives."""

    stems = sorted({str(record["image_id"]) for record in fp_records})
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(stems) + ("\n" if stems else ""))


class SmoothedValue:
    """Track a series of values and provide access to smoothed statistics."""

    def __init__(self, window_size: int = 20) -> None:
        self.window_size = window_size
        self.deque: List[float] = []
        self.total = 0.0
        self.count = 0

    def update(self, value: float) -> None:
        if len(self.deque) == self.window_size:
            self.total -= self.deque.pop(0)
        self.deque.append(value)
        self.total += value
        self.count += 1

    @property
    def avg(self) -> float:
        if not self.deque:
            return 0.0
        return self.total / len(self.deque)


class MetricLogger:
    """Helper class that logs running averages for multiple metrics."""

    def __init__(self) -> None:
        self.meters: Dict[str, SmoothedValue] = {}

    def update(self, **kwargs: float) -> None:
        for name, value in kwargs.items():
            if name not in self.meters:
                self.meters[name] = SmoothedValue()
            self.meters[name].update(float(value))

    def format(self) -> str:
        parts = [f"{name}: {meter.avg:.4f}" for name, meter in self.meters.items()]
        return " | ".join(parts)

