# Electrical Component Detection

This project trains a Faster R-CNN model to detect 32 classes of small electrical components.  
The repository contains a modular training and inference pipeline with configurable data
augmentation, detailed per-class metrics (TP/FP/FN, precision, recall, AP) and mAP reporting.

## Project structure

```
MSI5001FinalProject/
├── requirements.txt          # Python dependencies
├── src/
│   ├── config.py             # Configuration dataclasses
│   ├── dataset.py            # Dataset and augmentation utilities
│   ├── inference.py          # Script for model evaluation and visualisation
│   ├── model.py              # Faster R-CNN construction helpers
│   ├── train.py              # Main training script
│   └── utils.py              # Metrics, logging helpers and math utilities
└── data/
    ├── train/
    │   ├── images/*.npy
    │   └── labels/*.csv
    ├── valid/
    │   ├── images/*.npy
    │   └── labels/*.csv
    └── test/
        ├── images/*.npy
        └── labels/*.csv (optional)
```

Each label CSV is expected to contain the columns `class`, `x_center`, `y_center`, `width`
and `height`. Bounding boxes can be specified either in absolute pixel coordinates or in the
range `[0, 1]`; the loader automatically detects the correct interpretation.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python -m src.train \
  --data-dir /path/to/dataset \
  --epochs 25 \
  --batch-size 4 \
  --lr 1e-4 \
  --num-workers 0 \
  --small-object            # optional: use anchors better suited to tiny objects
```

Key flags:

* `--no-augmentation` disables geometric and photometric augmentations.
* `--small-object` enables smaller RPN anchors for tiny components.
* `--score-threshold` sets the global confidence cutoff (default `0.6`) while `--iou-threshold`
  controls matching for metric computation.
* `--class-threshold CLS=VALUE` overrides the confidence threshold for a specific class. Multiple
  overrides can be supplied (for example, `--class-threshold 3=0.7 --class-threshold 25=0.75`).
* `--checkpoint` defines where the best model (highest mAP) is saved.
* `--num-workers` defaults to `0` so the pipeline works in Kaggle notebooks without triggering
  multiprocessing shutdown assertions. Increase this gradually if you have spare CPU cores and a
  stable runtime environment.

Training produces `outputs/best_model.pth` and a `training_history.json` containing the
loss and mAP values for each evaluation epoch. Validation logs include per-class TP/FP/FN,
precision, recall and AP statistics alongside the global mAP figure.

## Working offline

When the environment has no internet access, download the official torchvision weights
(`fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`) and place them under
`weights/fasterrcnn_resnet50_fpn_v2_coco.pth`. The training and inference scripts load this
file automatically if online downloads are unavailable.

## Inference and visualisation

```bash
python -m src.inference \
  --data-dir /path/to/dataset \
  --split test \
  --checkpoint outputs/best_model.pth \
  --score-threshold 0.6 \
  --class-threshold 3=0.7 --class-threshold 6=0.7 --class-threshold 8=0.7 \
  --class-threshold 12=0.7 --class-threshold 17=0.7 --class-threshold 24=0.7 \
  --class-threshold 25=0.7 --class-threshold 26=0.7 \
  --draw-ground-truth
```

The script reports per-class metrics on the selected split (when labels are present), computes
the global mAP and saves annotated predictions to `outputs/inference/`. Ground-truth boxes are
drawn in white when `--draw-ground-truth` is set. By default the classes `03`, `06`, `08`, `12`,
`17`, `24`, `25` and `26` use a stricter `0.7` threshold while the remaining classes rely on the
global `0.6` cutoff; adjust these with `--class-threshold` as needed.

## Extending the project

* Adjust augmentation probabilities in `src/dataset.py::AugmentationParams` to experiment
  with different strategies.
* Update `DatasetConfig.class_names` if a mapping file containing human-readable names is
  available.
* Integrate additional metrics by extending `src/utils.py`.

## Reproducing the original notebooks

The training and inference logic previously implemented in the notebooks has been refactored
into Python modules. The notebooks are still available for reference but the scripts in `src/`
represent the canonical pipeline going forward.
