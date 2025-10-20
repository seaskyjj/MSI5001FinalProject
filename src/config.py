"""Configuration objects for the electrical component detection project."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_PRETRAINED_URL = (
    "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
)

#0.999 menas class does not exist
DEFAULT_CLASS_SCORE_THRESHOLDS = {
    3: 0.999,  
    6: 0.8,
    7: 0.9,
    8: 0.999,
    12: 0.999,
    16: 0.9,
    17: 0.999,
    20: 0.9,
    21: 0.8,
    24: 0.999,
    25: 0.9,
    26: 0.999,
    30: 0.9,
}


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
    mosaic_prob: float = 0.5
    mixup_prob: float = 0.5
    mixup_alpha: float = 0.4
    scale_jitter_min: float = 1.0
    scale_jitter_max: float = 1.0
    small_object: bool = True
    score_threshold: float = 0.6
    iou_threshold: float = 0.5
    eval_interval: int = 1
    seed: int = 2024
    output_dir: Path = Path("outputs")
    checkpoint_path: Path = Path("outputs/best_model.pth")
    pretrained_weights_path: Path = Path("weights/fasterrcnn_resnet50_fpn_v2_coco.pth")
    pretrained_weights_url: str = DEFAULT_PRETRAINED_URL
    log_every: int = 20
    class_score_thresholds: Dict[int, float] = field(
        default_factory=lambda: DEFAULT_CLASS_SCORE_THRESHOLDS.copy()
    )
    exclude_samples: Tuple[str, ...] = tuple()
    fp_visual_dir: Optional[Path] = Path("outputs/fp_images")
    fp_report_path: Optional[Path] = None
    fp_list_path: Optional[Path] = None
    fp_classes: Tuple[int, ...] = (16, 30)

    def ensure_directories(self) -> None:
        """Create output directories if they do not exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pretrained_weights_path.parent.mkdir(parents=True, exist_ok=True)
        if self.fp_visual_dir:
            self.fp_visual_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class InferenceConfig:
    """Options for running model inference and visualisation."""

    score_threshold: float = 0.6
    max_images: int = 200
    output_dir: Path = Path("outputs/inference")
    draw_ground_truth: bool = True
    class_colors: List[str] = field(default_factory=list)
    class_score_thresholds: Dict[int, float] = field(
        default_factory=lambda: DEFAULT_CLASS_SCORE_THRESHOLDS.copy()
    )

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
