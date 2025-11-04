#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import shutil
from pathlib import Path

# ===== 路径配置（按你的描述原样填写） =====
outputs_dir       = Path("/Users/jiejia/AII/MSI5001 Introduction to AI/dataset/valid/outputs")
valid_images_dir  = Path("/Users/jiejia/AII/MSI5001 Introduction to AI/Project/dataset/valid/images")
train_images_dir  = Path("/Users/jiejia/AII/MSI5001 Introduction to AI/Project/dataset/train/images")
valid_labels_dir  = Path("/Users/jiejia/AII/MSI5001 Introduction to AI/Project/dataset/valid/labels")
train_labels_dir  = Path("/Users/jiejia/AII/MSI5001 Introduction to AI/Project/dataset/train/labels")

SAMPLE_SIZE = 65
# 如需可复现抽样，取消下一行注释并设定种子
# random.seed(42)

def safe_move(src: Path, dst: Path) -> bool:
    """存在才移动；若目标已存在则跳过返回 False；成功移动返回 True。"""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[提示] 目标已存在，跳过覆盖：{dst}")
        return True  # 视为“处理完成”，不算失败
    shutil.move(str(src), str(dst))
    return True

def main():
    # 1) 列出所有 "0_*.csv"
    all_csvs = sorted(outputs_dir.glob("0_*.csv"))
    if len(all_csvs) == 0:
        raise FileNotFoundError(f"未在 {outputs_dir} 找到符合条件的 '0_*.csv' 文件。")
    if len(all_csvs) < SAMPLE_SIZE:
        raise ValueError(f"可用文件仅 {len(all_csvs)} 个，少于要求的 {SAMPLE_SIZE} 个。")

    # 2) 随机抽取 65 个
    selected_csvs = random.sample(all_csvs, SAMPLE_SIZE)

    moved_imgs_ok = 0
    moved_imgs_missing = 0
    moved_labels_ok = 0
    moved_labels_missing = 0

    for csv_path in selected_csvs:
        # 3) 去前缀 "0_" 与后缀 ".csv"，得到 img（可能是 xxx.npy）
        stem = csv_path.stem  # 去掉最后一个 .csv
        if not stem.startswith("0_"):
            # 理论上不会发生，因为用的 glob("0_*.csv")
            continue
        img = stem[2:]  # 去掉 "0_"

        # 4) 移动对应图片文件：valid/images/img -> train/images/img
        src_img = valid_images_dir / img
        dst_img = train_images_dir / img
        ok = safe_move(src_img, dst_img)
        if ok:
            moved_imgs_ok += 1
        else:
            moved_imgs_missing += 1
            print(f"[警告] 找不到对应图片，已跳过移动：{src_img}")

        # 5) 由 img 派生 label 名：去掉 ".npy" 换成 ".csv"
        label_name = (Path(img).with_suffix(".csv")).name

        # 6) 将 valid/labels 下的 label 文件移动到 train/labels
        src_label = valid_labels_dir / label_name
        dst_label = train_labels_dir / label_name
        ok_lbl = safe_move(src_label, dst_label)
        if ok_lbl:
            moved_labels_ok += 1
        else:
            moved_labels_missing += 1
            print(f"[警告] 找不到对应标签文件，已跳过移动：{src_label}")

    print("\n===== 完成 =====")
    print(f"匹配到的候选 CSV 文件总数：{len(all_csvs)}")
    print(f"本次随机处理：{len(selected_csvs)} 个")
    print(f"成功处理图片（含已存在跳过覆盖）：{moved_imgs_ok} 个")
    print(f"找不到对应图片：{moved_imgs_missing} 个")
    print(f"成功处理标签（含已存在跳过覆盖）：{moved_labels_ok} 个")
    print(f"找不到对应标签：{moved_labels_missing} 个")
    print(f"图片目标目录：{train_images_dir}")
    print(f"标签目标目录：{train_labels_dir}")

if __name__ == "__main__":
    main()
