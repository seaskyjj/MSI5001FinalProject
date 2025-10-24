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

* `--no-augmentation` disables all geometric/photometric augmentations, including the composite
  options below.
* `--mosaic-prob` enables mosaic augmentation with the given probability (default `0.0`).
* `--mixup-prob` and `--mixup-alpha` control MixUp augmentation when `--mosaic-prob` is active; the
  alpha parameter shapes the Beta distribution used for image blending.
* `--scale-jitter MIN MAX` applies uniform scale jittering to each image before flips/jitter. Use
  values like `--scale-jitter 0.9 1.1` to randomly shrink/enlarge the image by ±10%.
* `--affine-prob` toggles random translations, scaling and shear. Combine it with
  `--affine-translate FRAC_X FRAC_Y`, `--affine-scale MIN MAX` and
  `--affine-shear SHEAR_X SHEAR_Y` to define the extent of each transform.
* `--small-object` enables smaller RPN anchors for tiny components.
* `--score-threshold` sets the global confidence cutoff (default `0.6`) while `--iou-threshold`
  controls matching for metric computation.
* `--class-threshold CLS=VALUE` overrides the confidence threshold for a specific class. Multiple
  overrides can be supplied (for example, `--class-threshold 3=0.7 --class-threshold 25=0.75`).
* `--checkpoint` defines where the best model (highest mAP) is saved.
* `--resume` restores the latest training state from `outputs/last_checkpoint.pth` (model, optimiser,
  GradScaler and metric history). Combine it with `--resume-path /path/to/checkpoint.pth` to resume
  from a different file (for example, the best-performing epoch).
* `--num-workers` defaults to `0` so the pipeline works in Kaggle notebooks without triggering
  multiprocessing shutdown assertions. Increase this gradually if you have spare CPU cores and a
  stable runtime environment.
* `--exclude-sample STEM` removes a specific training example (by stem) from the dataset. Repeat
  the flag to drop several items.
* `--exclude-list PATH` reads a newline-separated text file or JSON report (such as the inference
  report described below) containing stems to skip during training.

Training produces `outputs/best_model.pth` (best validation mAP), `outputs/last_checkpoint.pth`
which is updated after every epoch with the full training state, and a `training_history.json`
containing the loss and mAP values for each evaluation epoch. Validation logs include per-class
TP/FP/FN, precision, recall and AP statistics alongside the global mAP figure.

### Resuming training

Use the built-in resume support to pick up exactly where a previous run finished:

```bash
# Continue from outputs/last_checkpoint.pth
python -m src.train --data-dir /path/to/dataset --resume

# Resume from a specific checkpoint (e.g. the best model)
python -m src.train --data-dir /path/to/dataset --resume-path outputs/best_model.pth
```

The checkpoint stores the model parameters, optimiser state (including momentum and learning rate),
automatic mixed precision scaler and accumulated metric history. Continuing with `--resume` keeps
the learning schedule and optimiser warm-up intact, whereas launching without it only reloads the
pretrained weights and restarts optimisation from scratch.

### Choosing affine parameters

The affine augmentations are sampled only when `--affine-prob` (or `TrainingConfig.affine_prob`)
is greater than zero. Recommended ranges for the remaining parameters:

* `affine_translate`: fractions of image width/height. Values in the `0.05–0.15` range
  (`--affine-translate 0.1 0.1`) allow small shifts without dropping too many boxes.
* `affine_scale_range`: multiplicative zoom sampled uniformly between the bounds. Try
  `--affine-scale 0.9 1.1` to jitter size by ±10%; keep both bounds positive and swap order
  if needed.
* `affine_shear`: maximum absolute shear (in degrees) around the x/y axes. Mild shear such as
  `--affine-shear 5 5` keeps rectangular components recognisable. Larger angles increase
  distortion and should be paired with a higher `--affine-prob` only if the dataset already
  contains similar perspective effects.

### Skipping images that cause false positives

1. Run inference on a labelled split (`train` or `valid`) with `--draw-ground-truth` to export
   annotated predictions. Add `--fp-report outputs/fp_report.json` or
   `--fp-list outputs/fp_stems.txt` to save a structured summary of the samples that produced
   false positives.
2. Inspect the generated images to confirm the problematic detections. The JSON report records the
   per-image boxes, scores and IoU overlaps, while the plain-text list simply enumerates the image
   stems.
3. Launch a new training run with either `--exclude-list outputs/fp_stems.txt` or individual
   `--exclude-sample STEM` flags so those samples are removed from the training dataloader. The
   Kaggle notebook exposes the same functionality via `TrainingConfig.exclude_samples` and the
   `metrics["false_positive_stems"]` output returned by `run_inference`.

## Working offline

When the environment has no internet access, download the official torchvision weights
(`fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`) and place them under
`weights/fasterrcnn_resnet50_fpn_v2_coco.pth`. The training and inference scripts load this
file automatically if online downloads are unavailable.

## Running as a service (Ubuntu)

On Ubuntu servers you can keep a training job running in the background with `systemd`. The example
below assumes the project lives in `/opt/MSI5001FinalProject` and the Python virtual environment is
located at `/opt/MSI5001FinalProject/.venv`.

1. Create a unit file:

   ```bash
   sudo tee /etc/systemd/system/msi5001-train.service >/dev/null <<'EOF'
   [Unit]
   Description=Electrical Component Detector Training
   After=network.target

   [Service]
   Type=simple
   User=<your-username>
   WorkingDirectory=/opt/MSI5001FinalProject
   ExecStart=/opt/MSI5001FinalProject/.venv/bin/python -m src.train \
       --data-dir /opt/MSI5001FinalProject/data \
       --epochs 25 \
       --resume
   Restart=on-failure
   Environment=PYTHONUNBUFFERED=1

   [Install]
   WantedBy=multi-user.target
   EOF
   ```

   Replace `User`, `WorkingDirectory`, `ExecStart` arguments and data path with values that match your setup.

2. Reload systemd and start the service:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now msi5001-train.service
   ```

3. Monitor logs and manage the service:

   ```bash
   journalctl -u msi5001-train.service -f   # tail live logs
   sudo systemctl stop msi5001-train.service
   sudo systemctl restart msi5001-train.service
   ```

The service survives SSH disconnects and automatically restarts if the process exits with an error.
Update the unit file and rerun `daemon-reload` whenever you change the command line.

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
  --fp-report outputs/fp_report.json \
  --fp-list outputs/fp_stems.txt
```

The script reports per-class metrics on the selected split (when labels are present), computes
the global mAP and saves annotated predictions to `outputs/inference/`. Ground-truth boxes are
drawn in white when `--draw-ground-truth` is set. By default the classes `03`, `06`, `08`, `12`,
`17`, `24`, `25` and `26` use a stricter `0.7` threshold while the remaining classes rely on the
global `0.6` cutoff; adjust these with `--class-threshold` as needed. The optional `--fp-report`
and `--fp-list` flags mirror the notebook behaviour by writing a JSON report or a newline-separated
list of stems that produced false positives, allowing quick exclusion during subsequent training
runs.

## Extending the project

* Tune the augmentation settings either through the CLI flags above or by editing the
  `AugmentationParams` defaults in `src/dataset.py` (and the mirrored notebook cell) to evaluate
  different Mosaic, MixUp and scale jitter strengths.
* Update `DatasetConfig.class_names` if a mapping file containing human-readable names is
  available.
* Integrate additional metrics by extending `src/utils.py`.

## Reproducing the original notebooks

The training and inference logic previously implemented in the notebooks has been refactored
into Python modules. The notebooks are still available for reference but the scripts in `src/`
represent the canonical pipeline going forward.
