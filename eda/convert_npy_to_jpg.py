#!/usr/bin/env python3
"""
Convert all .npy files under a given directory to .jpg images.

Usage:
    python convert_npy_to_jpg.py /path/to/root \
        [--outdir-name img] [--nonrecursive] [--overwrite] [--index 0]

Details:
- Scans for *.npy files. Recursive by default; use --nonrecursive to limit to the top level.
- Writes outputs to "<root>/<outdir-name>/" (default "img") preserving subfolder structure.
- Tries to interpret common image-like ndarray shapes:
    * (H, W)                -> grayscale
    * (H, W, 3/4)          -> RGB / RGBA (alpha dropped for JPEG)
    * (3/4, H, W)          -> CHW channel-first (converted)
    * (N, H, W[, C])       -> batched; takes slice N=--index (default 0)
- Dtype handling:
    * uint8 saved directly
    * float in [0,1] or [-1,1] -> scaled to [0,255]
    * other numeric -> min-max scaled per image
- Non-array containers:
    * dict / list / tuple: attempts to find a 2D/3D ndarray (prefers keys: image, img, array, data)
"""
import argparse
import sys
import json
from pathlib import Path
from typing import Any, Optional
import numpy as np
from PIL import Image

PREFERRED_KEYS = ["image", "img", "array", "arr", "data", "X", "mask"]

def find_image_array(x: Any, batch_index: int = 0) -> np.ndarray:
    """Extract a 2D/3D ndarray suitable for image saving from x. May use batch_index for 4D arrays."""
    if isinstance(x, np.ndarray):
        arr = x
    elif isinstance(x, dict):
        # Prefer typical keys
        for k in PREFERRED_KEYS:
            v = x.get(k, None)
            if isinstance(v, np.ndarray):
                arr = v
                break
        else:
            # fallback: largest ndarray value
            nds = [v for v in x.values() if isinstance(v, np.ndarray)]
            if not nds:
                raise ValueError("No ndarray found in dict.")
            nds.sort(key=lambda a: a.size, reverse=True)
            arr = nds[0]
    elif isinstance(x, (list, tuple)):
        nds = [v for v in x if isinstance(v, np.ndarray)]
        if not nds:
            raise ValueError("No ndarray found in list/tuple.")
        nds.sort(key=lambda a: a.size, reverse=True)
        arr = nds[0]
    else:
        raise ValueError(f"Unsupported container type: {type(x)}")

    # Now normalize shapes
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # (H,W,C) or (C,H,W) -> ensure HWC
        if arr.shape[-1] in (1, 3, 4):
            return arr
        if arr.shape[0] in (1, 3, 4):
            return np.moveaxis(arr, 0, -1)
        # Unknown channel position; best-effort: assume it's HWC
        return arr
    if arr.ndim == 4:
        # Try (N,H,W,C)
        if arr.shape[-1] in (1, 3, 4):
            n = arr.shape[0]
            idx = batch_index if 0 <= batch_index < n else 0
            return arr[idx]
        # Try (N,C,H,W)
        if arr.shape[1] in (1, 3, 4):
            n = arr.shape[0]
            idx = batch_index if 0 <= batch_index < n else 0
            chw = arr[idx]
            return np.moveaxis(chw, 0, -1)
        # Try (C,H,W,N) -> take first N
        if arr.shape[0] in (1, 3, 4):
            hwcN = np.moveaxis(arr, 0, -1)
            return hwcN[..., 0]
        raise ValueError(f"4D array with unrecognized layout: shape={arr.shape}")
    raise ValueError(f"Array has {arr.ndim} dimensions; expected 2D/3D/4D.")

def to_uint8_hwc(a: np.ndarray) -> np.ndarray:
    """Convert ndarray to uint8 image in HWC or 2D form."""
    arr = a
    # Channel-first to last
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)

    # Numeric conversion
    if arr.dtype == np.uint8:
        out = arr
    else:
        arr = arr.astype(np.float32, copy=False)
        vmin = float(np.nanmin(arr)) if arr.size else 0.0
        vmax = float(np.nanmax(arr)) if arr.size else 1.0
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            out = np.zeros_like(arr, dtype=np.uint8)
        elif vmin >= 0.0 and vmax <= 1.0:
            out = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
        elif vmin >= -1.0 and vmax <= 1.0:
            out = (((arr + 1.0) * 0.5) * 255.0).round().clip(0, 255).astype(np.uint8)
        else:
            out = ((arr - vmin) / (vmax - vmin) * 255.0).round().clip(0, 255).astype(np.uint8)

    # Squeeze single-channel HWC to 2D for grayscale
    if out.ndim == 3 and out.shape[-1] == 1:
        out = out[..., 0]
    return out

def save_image(img_u8: np.ndarray, out_path: Path) -> None:
    """Save uint8 ndarray as JPEG (RGB or L)."""
    if img_u8.ndim == 2:
        pil_img = Image.fromarray(img_u8, mode="L")
    elif img_u8.ndim == 3 and img_u8.shape[-1] == 3:
        pil_img = Image.fromarray(img_u8, mode="RGB")
    elif img_u8.ndim == 3 and img_u8.shape[-1] == 4:
        pil_img = Image.fromarray(img_u8[..., :3], mode="RGB")  # drop alpha for JPG
    else:
        raise ValueError(f"Unexpected image shape for saving: {img_u8.shape}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(str(out_path), format="JPEG", quality=95)

def convert_one(npy_path: Path, root: Path, out_root: Path, overwrite: bool, batch_index: int) -> str:
    rel = npy_path.relative_to(root)
    out_rel = rel.with_suffix(".jpg")
    out_path = out_root / out_rel
    if out_path.exists() and not overwrite:
        return f"[skip exists] {out_path}"
    try:
        data = np.load(str(npy_path), allow_pickle=True)
        # ``batch_index`` selects which slice to export when the .npy stores a stack of images.
        arr = find_image_array(data, batch_index=batch_index)
        img_u8 = to_uint8_hwc(arr)
        save_image(img_u8, out_path)
        return f"[ok] {npy_path} -> {out_path} (shape={getattr(arr, 'shape', None)}, dtype={getattr(arr, 'dtype', None)})"
    except Exception as e:
        return f"[error] {npy_path}: {e}"

def main():
    p = argparse.ArgumentParser(description="Convert .npy arrays to .jpg images.")
    p.add_argument("root", type=str, help="Root directory to scan for .npy files")
    p.add_argument("--outdir-name", type=str, default="img", help="Name of subfolder under root where JPGs are saved (default: img)")
    p.add_argument("--nonrecursive", action="store_true", help="Do not recurse into subdirectories (by default it recurses)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing .jpg files if present")
    p.add_argument("--index", type=int, default=0, help="Batch index to select when .npy contains a 4D batch (default: 0)")
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"Root path not found or not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    out_root = root / args.outdir-name if hasattr(args, 'outdir-name') else root / "img"  # safety fallback

    # Work around the hyphen in attribute name for older argparse behaviors
    outdir_name = getattr(args, "outdir_name", None) or "img"
    out_root = root / outdir_name

    # Toggle recursion based on ``--nonrecursive`` to support flat export directories.
    pattern = "**/*.npy" if not args.nonrecursive else "*.npy"
    files = sorted(root.glob(pattern))

    if not files:
        print("No .npy files found.")
        return

    print(f"Found {len(files)} .npy files under {root}")
    print(f"Saving JPGs under: {out_root}")
    ok, err, skip = 0, 0, 0
    for i, f in enumerate(files, 1):
        msg = convert_one(f, root=root, out_root=out_root, overwrite=args.overwrite, batch_index=args.index)
        if msg.startswith("[ok]"):
            ok += 1
        elif msg.startswith("[skip"):
            skip += 1
        else:
            err += 1
        print(f"{i:5d}/{len(files):5d} {msg}")
    print(f"Done. ok={ok}, skip={skip}, error={err}")

if __name__ == "__main__":
    main()
