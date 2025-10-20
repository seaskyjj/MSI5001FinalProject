"""Model building utilities for Faster R-CNN based detectors."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

from .config import DatasetConfig, TrainingConfig

LOGGER = logging.getLogger(__name__)


def _save_state_dict(model: nn.Module, path: Path) -> None:
    try:
        torch.save(model.state_dict(), path)
        LOGGER.info("Saved pretrained weights to %s", path)
    except Exception as exc:  # pragma: no cover - safety net
        LOGGER.warning("Unable to save pretrained weights: %s", exc)


def build_model(
    dataset_cfg: DatasetConfig,
    train_cfg: TrainingConfig,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Create a Faster R-CNN model adjusted for the project dataset."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _load_pretrained_model(train_cfg)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes_with_background = dataset_cfg.num_classes + 1
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes_with_background
    )

    if train_cfg.small_object:
        anchor_generator = AnchorGenerator(
            sizes=((16,), (32,), (64,), (128,), (256,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )
        model.rpn.anchor_generator = anchor_generator
        LOGGER.info("Using custom anchor sizes optimised for small objects")

    model.to(device)
    return model


def _load_pretrained_model(train_cfg: TrainingConfig) -> nn.Module:
    """Load a Faster R-CNN model with fallback to local weights when offline."""
    pretrained_path = train_cfg.pretrained_weights_path
    pretrained_path.parent.mkdir(parents=True, exist_ok=True)

    weights_enum = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    try:
        model = fasterrcnn_resnet50_fpn_v2(weights=weights_enum)
        LOGGER.info("Loaded torchvision Faster R-CNN weights")
        if not pretrained_path.exists():
            _save_state_dict(model, pretrained_path)
        return model
    except Exception:  # pragma: no cover - fallback when torchvision download fails
        LOGGER.warning("Falling back to locally saved pretrained detector weights")
        if not pretrained_path.exists():
            raise RuntimeError(
                "No pretrained weights available. Download the torchvision weights manually "
                "and place them at %s" % pretrained_path
            )
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model = fasterrcnn_resnet50_fpn_v2(weights=None)
        model.load_state_dict(state_dict)
        return model
