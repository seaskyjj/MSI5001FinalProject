# src/model.py

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from . import config

def create_model(num_classes):
    """
    创建 Faster R-CNN 模型。
    
    Args:
        num_classes (int): 类别数量 (包括背景)。
    
    Returns:
        model: A PyTorch Faster R-CNN model.
    """
    # 加载一个在 COCO 数据集上预训练好的 Faster R-CNN 模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    
    # 获取分类器的输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # 替换预训练的头部为一个新的头部
    # 新的头部的输出维度是 num_classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

if __name__ == '__main__':
    # 测试模型创建
    model = create_model(config.NUM_CLASSES)
    print(model)
    
    # 检查模型是否可以在指定设备上运行
    device = config.DEVICE
    model.to(device)
    print(f"Model moved to {device}")
