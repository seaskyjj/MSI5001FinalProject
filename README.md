# 小电器元器件识别项目 (MSI5001FinalProject)

本项目旨在使用 Faster R-CNN 模型对图像中的小型电子元器件（如电阻、电容等）进行目标检测。

## 项目特性

- **模型**: 基于 `torchvision` 的 Faster R-CNN (ResNet-50 FPN backbone)，在 COCO 数据集上进行了预训练。
- **数据增强**: 使用 `albumentations` 库实现，可在训练时对图像进行翻转、旋转、颜色抖动等操作，以提升模型泛化能力。
- **评估指标**: 采用标准的 **mAP (mean Average Precision)** 作为核心评估指标，并在验证过程中输出每个类别的 mAP。
- **模块化代码**: 代码结构清晰，分为配置、数据加载、模型、训练和工具函数等模块，易于维护和扩展。
- **可配置性**: 所有关键参数（如学习率、批大小、数据增强开关等）都集中在 `src/config.py` 文件中，方便调整。

## 环境设置

1.  **克隆项目**
    ```bash
    git clone <your-repo-url>
    cd MSI5001FinalProject
    ```

2.  **安装依赖**
    建议使用虚拟环境（如 venv 或 conda）。
    ```bash
    # 创建并激活虚拟环境 (可选)
    python -m venv venv
    source venv/bin/activate  # on Windows: venv\Scripts\activate

    # 安装所有必需的库
    pip install -r requirements.txt
    ```

## 数据准备

请按照以下结构组织您的数据集。标注文件应为 PASCAL VOC 格式的 `.xml` 文件，且与对应的图片文件名相同。

```
data/
├── train/
│   ├── image1.jpg
│   ├── image1.xml
│   ├── image2.jpg
│   ├── image2.xml
│   └── ...
├── valid/
│   ├── image3.jpg
│   ├── image3.xml
│   └── ...
└── test/
    ├── image4.jpg
    ├── image4.xml
    └── ...
```

**重要**: 请务必更新 `src/config.py` 文件中的 `CLASSES` 列表，使其与您数据集中的类别完全匹配。第一个类别必须是 `__background__`。

```python
# src/config.py
CLASSES = [
    '__background__', 
    'resistor', 
    'capacitor', 
    # ... 添加您所有的类别
]
```

## 如何使用

### 1. 调整配置

打开 `src/config.py` 文件，根据您的需求修改参数。特别是：
- `CLASSES`: 确保类别列表正确。
- `APPLY_AUGMENTATION`: 设置为 `True` 开启数据增强，`False` 关闭。
- `NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`: 调整训练超参数。

### 2. 开始训练

运行 `train.py` 脚本来启动训练和验证流程。

```bash
python -m src.train
```

训练过程中，脚本会：
- 在每个 epoch 结束后，在验证集上计算并打印 mAP 以及每个类别的 mAP。
- 将 mAP 最高的模型权重保存到 `outputs/models/` 目录下。

### 3. 进行推理 (可选)

您可以使用 `src/inference.py` (需要您根据需求自行编写，可参考训练脚本的评估部分) 对新图片进行预测，或在测试集上评估最终模型。

## 下一步计划

- 实现 `inference.py` 脚本，用于可视化单张图片的检测结果。
- 引入 TensorBoard 或 W&B 进行训练过程的可视化监控。
- 尝试其他模型架构或 backbone，如 EfficientDet。
