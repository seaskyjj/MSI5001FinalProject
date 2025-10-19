"""Configuration objects for the electrical component detection project."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


DEFAULT_PRETRAINED_URL = (
    "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
)


@dataclass
class DatasetConfig:
    """Configuration describing the dataset layout and metadata."""

    base_dir: Path = Path("data")
    train_split: str = "train"
    valid_split: str = "valid"
    test_split: str = "test"
    image_folder: str = "images"
    label_folder: str = "labels"
    num_classes: int = 32
    class_names: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.class_names:
            # Fallback names are useful for logging when a mapping file is not provided.
            self.class_names = tuple(f"class_{idx:02d}" for idx in range(self.num_classes))


@dataclass
class TrainingConfig:
    """Hyper-parameters and runtime settings for model training."""

    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    amp: bool = True
    augmentation: bool = True
    small_object: bool = False
    score_threshold: float = 0.05
    iou_threshold: float = 0.5
    eval_interval: int = 1
    seed: int = 2024
    output_dir: Path = Path("outputs")
    checkpoint_path: Path = Path("outputs/best_model.pth")
    pretrained_weights_path: Path = Path("weights/fasterrcnn_resnet50_fpn_v2_coco.pth")
    pretrained_weights_url: str = DEFAULT_PRETRAINED_URL
    log_every: int = 20

    def ensure_directories(self) -> None:
        """Create output directories if they do not exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pretrained_weights_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class InferenceConfig:
    """Options for running model inference and visualisation."""

    score_threshold: float = 0.3
    max_images: int = 50
    output_dir: Path = Path("outputs/inference")
    draw_ground_truth: bool = True
    class_colors: List[str] = field(default_factory=list)

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
