"""Configuration objects for the electrical component detection project."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_PRETRAINED_URL = (
    "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
)

# A threshold of 0.999 effectively disables classes that are absent in the dataset.
DEFAULT_CLASS_SCORE_THRESHOLDS = {
    3: 0.999,  
    6: 0.8,
    7: 0.9,
    8: 0.999,
    12: 0.999,
    16: 0.97,
    17: 0.999,
    20: 0.9,
    21: 0.9,
    24: 0.999,
    25: 0.97,
    26: 0.999,
    30: 0.95,
}


@dataclass
class DatasetConfig:
    """Configuration describing the dataset layout and metadata."""

    base_dir: Path = Path("data")  # Root directory where all splits live.
    train_split: str = "train"  # Folder name for the training split.
    valid_split: str = "valid"  # Folder name for the validation split.
    test_split: str = "test"  # Folder name for the held-out/test split.
    image_folder: str = "images"  # Relative folder holding image tensors.
    label_folder: str = "labels"  # Relative folder holding CSV annotations.
    num_classes: int = 32  # Number of foreground classes (excluding background).
    class_names: Tuple[str, ...] = ()  # Optional mapping from class index to human name.

    def __post_init__(self) -> None:
        if not self.class_names:
            # Fallback names are useful for logging when a mapping file is not provided.
            self.class_names = tuple(f"class_{idx:02d}" for idx in range(self.num_classes))


@dataclass
class TrainingConfig:
    """Hyper-parameters and runtime settings for model training."""

    epochs: int = 20  # Total number of passes over the training split.
    batch_size: int = 4  # Images per iteration; constrained by GPU memory.
    learning_rate: float = 5e-5  # Base learning rate for the AdamW optimiser.
    weight_decay: float = 5e-5  # L2 regularisation strength to combat overfitting.
    num_workers: int = 0  # DataLoader worker processes (0 keeps everything in-process).
    amp: bool = True  # Enable PyTorch automatic mixed precision when available.
    augmentation: bool = True  # Master switch for all stochastic augmentations.
    mosaic_prob: float = 0.6  # Probability of four-image mosaic augmentation.
    mixup_prob: float = 0.6  # Probability of blending a second image into the batch.
    mixup_alpha: float = 0.4  # Beta distribution alpha controlling MixUp strength.
    scale_jitter_min: float = 0.8  # Lower bound for random resizing prior to affine ops.
    scale_jitter_max: float = 1.2  # Upper bound for random resizing prior to affine ops.
    rotation_prob: float = 0.5  # Chance of applying a random rotation augmentation.
    rotation_max_degrees: float = 30.0  # Maximum absolute rotation angle in degrees.
    affine_prob: float = 0.3  # Probability of running the generic affine augmentation.
    affine_translate: Tuple[float, float] = (0.1, 0.1)  # Max XY shift as a fraction of width/height.
    affine_scale_range: Tuple[float, float] = (0.9, 1.1)  # Multiplicative scaling range during affine.
    affine_shear: Tuple[float, float] = (5, 5)  # Max shear angles (deg) along X and Y axes.
    small_object: bool = True  # Use denser anchors tailored to small components.
    score_threshold: float = 0.6  # Default detection confidence threshold post-NMS.
    iou_threshold: float = 0.5  # IoU threshold for mAP computation and FP analysis.
    eval_interval: int = 1  # Run validation every N epochs.
    seed: int = 37  # Global RNG seed for reproducibility.
    output_dir: Path = Path("outputs")  # Root directory for training artefacts.
    checkpoint_path: Path = Path("outputs/best_model.pth")  # Best model snapshot location.
    pretrained_weights_path: Path = Path("weights/fasterrcnn_resnet50_fpn_v2_coco.pth")  # Local cache for torchvision weights.
    pretrained_weights_url: str = DEFAULT_PRETRAINED_URL  # Remote URL for the pretrained backbone.
    log_every: int = 20  # Frequency (steps) of progress bar/log updates.
    resume: bool = False  # Resume training from ``resume_path`` when True.
    resume_path: Optional[Path] = None  # Explicit checkpoint path for resuming.
    last_checkpoint_path: Path = Path("outputs/last_checkpoint.pth")  # Rolling checkpoint for fault tolerance.
    class_score_thresholds: Dict[int, float] = field(
        default_factory=lambda: DEFAULT_CLASS_SCORE_THRESHOLDS.copy()
    )
    exclude_samples: Tuple[str, ...] = tuple()  # Optional list of sample stems to skip during training.
    fp_visual_dir: Optional[Path] = Path("outputs/fp_images")  # Export directory for false-positive renders.
    fp_report_path: Optional[Path] = None  # JSON output summarising final-epoch false positives.
    fp_list_path: Optional[Path] = None  # Text output enumerating false-positive image stems.
    fp_classes: Tuple[int, ...] = (16, 30)  # Class indices prioritised for FP inspection.

    def ensure_directories(self) -> None:
        """Create output directories if they do not exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pretrained_weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if self.fp_visual_dir:
            self.fp_visual_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class InferenceConfig:
    """Options for running model inference and visualisation."""

    score_threshold: float = 0.7  # Default per-detection confidence cut-off during evaluation.
    max_images: int = 200  # Limit on rendered inference examples to avoid huge dumps.
    output_dir: Path = Path("outputs/inference")  # Where to store rendered predictions.
    draw_ground_truth: bool = True  # Overlay ground-truth boxes when available.
    class_colors: List[str] = field(default_factory=list)
    class_score_thresholds: Dict[int, float] = field(
        default_factory=lambda: DEFAULT_CLASS_SCORE_THRESHOLDS.copy()
    )

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
