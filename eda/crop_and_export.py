#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量切分并导出子样本（.npy / .csv / .jpg），文件名为 <原文件名>_<class>.xxx。

两种目录组织均支持：
A) 同一目录树下成对：   --input /root
B) 图像与标注分目录：   --images /.../images  --labels /.../labels

用法示例：
    # 你当前的数据组织（分目录）
    python crop_and_export.py \
        --images /Project/dataset/train/images \
        --labels /Project/dataset/train/labels \
        --output /Project/dataset/train/outpus \
        --workers 4

    # 若图像和CSV在同一颗目录树下：
    python crop_and_export.py --input /path/to/dataset --output /path/to/output

CSV 要求列：class, x_center, y_center, width, height。
自动判断像素坐标或 YOLO 归一化坐标（0~1）。

依赖：numpy、pandas、Pillow、tqdm
    pip install numpy pandas pillow tqdm
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, List
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".npy"}
CSV_EXT = ".csv"

# ---------------------------- 文件配对 ---------------------------- #

def _scan_images(root: Path) -> Dict[str, Path]:
    """扫描图像目录，返回 {基名: 路径}。若同基名有多扩展，优先级 .npy>.jpg>.jpeg>.png。"""
    buckets: Dict[str, Dict[str, Path]] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in IMG_EXTS:
            continue
        base = p.with_suffix("").name
        b = buckets.setdefault(base, {})
        b[ext] = p
    priority = [".npy", ".jpg", ".jpeg", ".png"]
    picks: Dict[str, Path] = {}
    for base, mp in buckets.items():
        for ext in priority:
            if ext in mp:
                picks[base] = mp[ext]
                break
    return picks


def _scan_labels(root: Path) -> Dict[str, Path]:
    """扫描标注目录，返回 {基名: csv路径}。"""
    res: Dict[str, Path] = {}
    for p in root.rglob("*.csv"):
        if not p.is_file():
            continue
        base = p.with_suffix("").name
        res[base] = p
    return res


def find_pairs_mirrored(images_root: Path, labels_root: Path) -> List[Tuple[Path, Path]]:
    """在分离目录(images/labels)中，按同名基名配对。"""
    img_map = _scan_images(images_root)
    lbl_map = _scan_labels(labels_root)
    common = sorted(set(img_map.keys()) & set(lbl_map.keys()))
    pairs = [(img_map[k], lbl_map[k]) for k in common]
    missing_imgs = sorted(set(lbl_map.keys()) - set(img_map.keys()))
    missing_lbls = sorted(set(img_map.keys()) - set(lbl_map.keys()))
    if missing_imgs:
        warnings.warn(f"有 {len(missing_imgs)} 个标注找不到对应图像样本（举例）：{missing_imgs[:5]}")
    if missing_lbls:
        warnings.warn(f"有 {len(missing_lbls)} 个图像找不到对应标注（举例）：{missing_lbls[:5]}")
    return pairs


def find_pairs_flat(root: Path) -> Iterable[Tuple[Path, Path]]:
    """在单一目录树下按同名配对（向后兼容）。"""
    buckets: Dict[str, Dict[str, Path]] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in IMG_EXTS and ext != CSV_EXT:
            continue
        base = p.with_suffix("").name
        b = buckets.setdefault(base, {})
        b[ext] = p
    priority = [".npy", ".jpg", ".jpeg", ".png"]
    for base, mp in buckets.items():
        if CSV_EXT not in mp:
            continue
        img_path: Optional[Path] = None
        for ext in priority:
            if ext in mp:
                img_path = mp[ext]
                break
        if img_path is None:
            continue
        yield img_path, mp[CSV_EXT]

# ---------------------------- I/O 辅助 ---------------------------- #

def load_image(img_path: Path) -> np.ndarray:
    """加载图像，返回 HxWxC 的 uint8 RGB 数组。支持 .npy 或常见图片格式。"""
    ext = img_path.suffix.lower()
    if ext == ".npy":
        arr = np.load(img_path, allow_pickle=False)
        if arr.ndim == 2:  # 灰度 -> 伪 RGB
            arr = np.stack([arr] * 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    else:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            return np.array(im, dtype=np.uint8)


def save_jpg(arr: np.ndarray, path: Path) -> None:
    im = Image.fromarray(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=95)


def clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def ensure_unique(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    idx = 2
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1

# ---------------------------- 主处理 ---------------------------- #

def process_one(img_path: Path, csv_path: Path, out_dir: Path) -> None:
    arr = load_image(img_path)
    H, W = arr.shape[:2]

    df = pd.read_csv(csv_path, encoding="utf-8")
    required_cols = {"class", "x_center", "y_center", "width", "height"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少列: {missing} -> {csv_path}")

    def is_normalized(series: pd.Series) -> bool:
        return series.max() <= 1.0 and series.min() >= 0.0

    norm_x = is_normalized(df["x_center"]) and is_normalized(df["width"])
    norm_y = is_normalized(df["y_center"]) and is_normalized(df["height"])

    base_stem = img_path.name  # 使用带扩展的原文件名作为“原文件名”

    for _, row in df.iterrows():
        cls = row["class"]
        xc = float(row["x_center"]) * (W if norm_x else 1.0)
        yc = float(row["y_center"]) * (H if norm_y else 1.0)
        bw = float(row["width"])   * (W if norm_x else 1.0)
        bh = float(row["height"])  * (H if norm_y else 1.0)

        x1 = int(round(xc - bw / 2.0))
        y1 = int(round(yc - bh / 2.0))
        x2 = int(round(xc + bw / 2.0))
        y2 = int(round(yc + bh / 2.0))
        x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, W, H)

        crop = arr[y1:y2, x1:x2]

        out_stem = f"{cls}_{base_stem}"
        npy_out = ensure_unique(out_dir / f"{out_stem}.npy")
        jpg_out = ensure_unique(out_dir / f"{out_stem}.jpg")
        csv_out = ensure_unique(out_dir / f"{out_stem}.csv")

        npy_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_out, crop)
        save_jpg(crop, jpg_out)

        sub = {
            "class": cls,
            "x_center": row["x_center"],
            "y_center": row["y_center"],
            "width": row["width"],
            "height": row["height"],
            "coord_system": "normalized" if (norm_x and norm_y) else "pixel",
            "img_W": W,
            "img_H": H,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "source_image": img_path.name,
            "source_csv": csv_path.name,
        }
        pd.DataFrame([sub]).to_csv(csv_out, index=False, encoding="utf-8")

# ---------------------------- 入口 ---------------------------- #

def main():
    ap = argparse.ArgumentParser(description="裁剪并导出子样本为 .npy/.csv/.jpg")
    ap.add_argument("--input", type=Path, help="(可选) 单一目录树，自动成对")
    ap.add_argument("--images", type=Path, default="/Users/jiejia/AII/MSI5001 Introduction to AI/Project/dataset/train/images", help="(可选) 图像根目录")
    ap.add_argument("--labels", type=Path, default="/Users/jiejia/AII/MSI5001 Introduction to AI/Project/dataset/train/labels", help="(可选) 标注CSV根目录")
    ap.add_argument("--output", default="/Users/jiejia/AII/MSI5001 Introduction to AI/Project/dataset/train/output", type=Path, help="输出目录")
    ap.add_argument("--workers", type=int, default=2, help="并行线程数（0=单线程）")
    args = ap.parse_args()

    # 选择成对方式
    if args.images and args.labels:
        pairs = find_pairs_mirrored(args.images, args.labels)
    elif args.input:
        pairs = list(find_pairs_flat(args.input))
    else:
        print("[ERROR] 需要 (--images 与 --labels) 或者 --input 之一。", file=sys.stderr)
        return 2

    if not pairs:
        print("[WARN] 未找到任何 (图像,CSV) 成对文件。")
        return 0

    args.output.mkdir(parents=True, exist_ok=True)

    if args.workers and args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_one, ip, cp, args.output) for ip, cp in pairs]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                pass
    else:
        for img_path, csv_path in tqdm(pairs, desc="Processing"):
            process_one(img_path, csv_path, args.output)

    print("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())