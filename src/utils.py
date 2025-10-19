"""Utility helpers for training and evaluating the detection model."""
from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


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
