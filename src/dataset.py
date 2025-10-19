# src/dataset.py

import torch
import cv2
import numpy as np
import os
import glob
import pandas as pd
from torch.utils.data import Dataset
from . import config

# 定义训练和验证时的数据变换
def get_transform(train):
    """
    获取数据增强/变换流程。
    使用 albumentations 库进行图像增强。
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    if train and config.APPLY_AUGMENTATION:
        # 训练时的数据增强策略
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ColorJitter(p=0.2),
            A.Rotate(limit=15, p=0.25),
            # 注意：ToTensorV2 会将图像数据从 HWC 转为 CHW，并从 uint8 转为 float32
            ToTensorV2(p=1.0),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        # 验证/测试时只进行 Tensor 转换
        return A.Compose([
            ToTensorV2(p=1.0),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

class SmallComponentsDataset(Dataset):
    def __init__(self, dir_path, classes, transform=None):
        self.dir_path = dir_path
        self.classes = classes
        self.transform = transform
        
        # 获取所有图片文件的路径
        self.image_paths = sorted(glob.glob(f"{self.dir_path}/*.npy"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # 读取图片
        # 从 .npy 文件加载 RGB 图像数组
        image = np.load(image_path)
        image_height, image_width, _ = image.shape
        
        # 解析对应的 CSV 标注文件
        annot_filename = os.path.splitext(os.path.basename(image_path))[0] + '.csv'
        annot_filepath = os.path.join(self.dir_path, annot_filename)
        
        boxes = []
        labels = []
        
        if os.path.exists(annot_filepath):
            annotations = pd.read_csv(annot_filepath)
            # 假设 CSV 列名为 'xmin', 'ymin', 'xmax', 'ymax', 'class'
            # 'class' 列包含的是类别名称
            for _, row in annotations.iterrows():
                label_name = row['class']
                if label_name not in self.classes:
                    continue

                # 获取类别索引 (config.CLASSES 中 0 是背景)
                labels.append(self.classes.index(label_name))

                # 获取并确保边界框坐标在图像范围内
                xmin = max(0, int(row['xmin']))
                ymin = max(0, int(row['ymin']))
                xmax = min(image_width, int(row['xmax']))
                ymax = min(image_height, int(row['ymax']))
                boxes.append([xmin, ymin, xmax, ymax])

        # 将数据转换为 torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        
        # 计算每个 box 的面积
        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            # 如果没有物体，创建一个空的 tensor
            area = torch.as_tensor([], dtype=torch.float32)

        # 假设不是拥挤的场景
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        
        # 准备 target 字典
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }
        
        # 应用数据增强
        if self.transform:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transform(**sample)
            image = sample['image']
            
            # 更新 target 中的 boxes
            if len(sample['bboxes']) > 0:
                target['boxes'] = torch.stack([torch.tensor(b) for b in sample['bboxes']])
            else:
                # 如果增强后没有 box，创建一个空的 tensor
                target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
        
        return image, target

def create_data_loader(dir_path, classes, batch_size, num_workers, is_train=True):
    """创建一个数据加载器"""
    transform = get_transform(train=is_train)
    dataset = SmallComponentsDataset(dir_path, classes, transform)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train, # 训练集需要打乱顺序
        num_workers=num_workers,
        collate_fn=collate_fn # 自定义批处理函数
    )
    return data_loader

def collate_fn(batch):
    """
    自定义的 collate_fn，因为每张图片的 object 数量不同，
    所以需要将 image 和 target 分别打包。
    """
    return tuple(zip(*batch))
