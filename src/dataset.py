"""Dataset and data loading utilities for electrical component detection."""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage, ImageEnhance
from torch.utils.data import DataLoader, Dataset

from .utils import sanitize_boxes_and_labels


LOGGER = logging.getLogger(__name__)


@dataclass
class AugmentationParams:
    """Parameters controlling the dataset level image augmentations."""

    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.2
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.02


def load_image_hwc_uint8(path: Path) -> np.ndarray:
    """Load an ``.npy`` image stored as HWC and return an ``uint8`` array."""
    image = np.load(path, allow_pickle=False, mmap_mode="r")

    if image.dtype != np.uint8:
        image = image.astype(np.float32, copy=False)
        vmin, vmax = float(image.min()), float(image.max())
        if 0.0 <= vmin and vmax <= 1.0:
            image = (image * 255.0).round()
        elif -1.0 <= vmin and vmax <= 1.0:
            image = ((image + 1.0) * 0.5 * 255.0).round()
        image = np.clip(image, 0, 255).astype(np.uint8)

    channels = image.shape[2]
    if channels == 1:
        image = np.repeat(image, 3, axis=2)
    elif channels == 4:
        image = image[..., :3]
    return image


class ElectricalComponentsDataset(Dataset):
    """Dataset of electrical component detections stored as ``.npy`` images and CSV labels."""

    def __init__(
        self,
        root: Path,
        split: str,
        class_names: Iterable[str],
        transform: Optional[AugmentationParams] = None,
        use_augmentation: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.class_names = list(class_names)
        self.transform_params = transform or AugmentationParams()
        self.use_augmentation = use_augmentation

        self.image_dir = self.root / split / "images"
        self.label_dir = self.root / split / "labels"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Missing label directory: {self.label_dir}")

        self.image_stems = sorted(p.stem for p in self.label_dir.glob("*.csv"))
        if not self.image_stems:
            raise RuntimeError(f"No label files found in {self.label_dir}")

        # Pre-load all annotations to reduce I/O during training.
        self.annotations: Dict[str, pd.DataFrame] = {
            stem: pd.read_csv(self.label_dir / f"{stem}.csv") for stem in self.image_stems
        }

    def __len__(self) -> int:
        return len(self.image_stems)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        stem = self.image_stems[index]
        image_path = self.image_dir / f"{stem}.npy"
        image = load_image_hwc_uint8(image_path)
        height, width = image.shape[:2]

        ann = self.annotations[stem]
        x_center = ann["x_center"].to_numpy(dtype=np.float32)
        y_center = ann["y_center"].to_numpy(dtype=np.float32)
        box_width = ann["width"].to_numpy(dtype=np.float32)
        box_height = ann["height"].to_numpy(dtype=np.float32)

        # Auto-detect normalised coordinates and scale back to pixel space.
        if (
            (x_center.size == 0 or float(x_center.max()) <= 1.0)
            and (y_center.size == 0 or float(y_center.max()) <= 1.0)
            and (box_width.size == 0 or float(box_width.max()) <= 1.0)
            and (box_height.size == 0 or float(box_height.max()) <= 1.0)
        ):
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

        x1 = x_center - box_width / 2.0
        y1 = y_center - box_height / 2.0
        x2 = x_center + box_width / 2.0
        y2 = y_center + box_height / 2.0

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        labels = ann["class"].to_numpy(dtype=np.int64)

        if self.use_augmentation and len(boxes):
            image, boxes = self._apply_augmentations(image, boxes)
            height, width = image.shape[:2]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        boxes_tensor = torch.from_numpy(boxes).float()
        labels_tensor = torch.from_numpy(labels).long()

        boxes_tensor, labels_tensor = sanitize_boxes_and_labels(
            boxes_tensor, labels_tensor, height, width
        )

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            if boxes_tensor.numel()
            else torch.tensor([], dtype=torch.float32),
            "iscrowd": torch.zeros((boxes_tensor.shape[0],), dtype=torch.int64),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
        }

        return image_tensor, target

    def _apply_augmentations(
        self, image: np.ndarray, boxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        params = self.transform_params
        height, width = image.shape[:2]

        if random.random() < params.horizontal_flip_prob:
            image = np.ascontiguousarray(image[:, ::-1, :])
            x1 = width - boxes[:, 2]
            x2 = width - boxes[:, 0]
            boxes[:, 0], boxes[:, 2] = x1, x2

        if random.random() < params.vertical_flip_prob:
            image = np.ascontiguousarray(image[::-1, :, :])
            y1 = height - boxes[:, 3]
            y2 = height - boxes[:, 1]
            boxes[:, 1], boxes[:, 3] = y1, y2

        if params.brightness or params.contrast or params.saturation or params.hue:
            pil = PILImage.fromarray(image)
            if params.brightness:
                enhancer = ImageEnhance.Brightness(pil)
                factor = 1.0 + random.uniform(-params.brightness, params.brightness)
                pil = enhancer.enhance(max(0.1, factor))
            if params.contrast:
                enhancer = ImageEnhance.Contrast(pil)
                factor = 1.0 + random.uniform(-params.contrast, params.contrast)
                pil = enhancer.enhance(max(0.1, factor))
            if params.saturation:
                enhancer = ImageEnhance.Color(pil)
                factor = 1.0 + random.uniform(-params.saturation, params.saturation)
                pil = enhancer.enhance(max(0.1, factor))
            if params.hue:
                # Hue adjustment via simple conversion to HSV.
                hsv = np.array(pil.convert("HSV"), dtype=np.uint8)
                delta = int(params.hue * 255.0 * random.choice([-1, 1]))
                hsv[..., 0] = (hsv[..., 0].astype(int) + delta) % 255
                pil = PILImage.fromarray(hsv, mode="HSV").convert("RGB")
            image = np.array(pil)

        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        return image, boxes


def detection_collate(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    """Collate function for detection datasets returning lists of tensors."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def _safe_worker_count(requested: int) -> int:
    cpu_count = os.cpu_count() or 1
    if requested <= 0:
        return 0
    # Leave one core free so that the main process remains responsive on small machines.
    max_workers = max(1, cpu_count - 1)
    return min(requested, max_workers)


def _running_in_ipython_kernel() -> bool:
    """Return ``True`` when executing inside an IPython/Jupyter kernel."""

    try:  # ``IPython`` is an optional dependency in our runtime.
        from IPython import get_ipython  # type: ignore
    except Exception:  # pragma: no cover - depends on environment
        return False

    shell = get_ipython()
    return bool(shell and getattr(shell, "kernel", None))


def _should_force_single_worker(dataset: Dataset) -> bool:
    """Determine whether multiprocessing workers should be disabled."""

    module_name = getattr(dataset.__class__, "__module__", "")
    if module_name in {"__main__", "__mp_main__", "builtins"}:
        return True

    if module_name.startswith("ipykernel"):  # pragma: no cover - notebook specific
        return True

    return _running_in_ipython_kernel()


def create_data_loaders(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """Create a :class:`~torch.utils.data.DataLoader` for the detection dataset.

    Kaggle notebooks occasionally run in restricted multiprocessing environments where
    ``fork`` based workers cannot be reaped cleanly.  We therefore favour a conservative
    default (``num_workers=0``), detect in-notebook dataset definitions that cannot be
    spawned safely, and fall back to single-process loading automatically when worker
    start-up fails.
    """

    worker_count = _safe_worker_count(num_workers)
    if worker_count > 0 and _should_force_single_worker(dataset):
        LOGGER.info(
            "Detected interactive environment or in-notebook dataset definition; forcing num_workers=0."
        )
        worker_count = 0

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
        collate_fn=detection_collate,
    )

    if worker_count > 0:
        loader_kwargs["num_workers"] = worker_count
        loader_kwargs["persistent_workers"] = True
        # ``spawn`` avoids PID mismatches that surface as ``AssertionError: can only
        # test a child process`` when the notebook kernel re-uses processes.
        loader_kwargs["multiprocessing_context"] = mp.get_context("spawn")
    else:
        loader_kwargs["num_workers"] = 0

    try:
        return DataLoader(**loader_kwargs)
    except (RuntimeError, OSError, AssertionError) as exc:
        if worker_count == 0:
            raise
        warnings.warn(
            "Falling back to num_workers=0 because DataLoader worker initialisation "
            f"failed with: {exc}",
            RuntimeWarning,
        )
        LOGGER.warning("DataLoader workers failed to start (%s). Using num_workers=0 instead.", exc)
        loader_kwargs.pop("persistent_workers", None)
        loader_kwargs.pop("multiprocessing_context", None)
        loader_kwargs["num_workers"] = 0
        return DataLoader(**loader_kwargs)
