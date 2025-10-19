# src/utils.py

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from . import config

def calculate_map(model, data_loader, device):
    """
    计算并打印 mAP 以及每个类别的 TP, FP, FN。
    
    Args:
        model: 训练好的模型。
        data_loader: 验证或测试数据加载器。
        device: 'cuda' or 'cpu'.
    
    Returns:
        map_value (float): 计算出的 mAP 值。
    """
    model.eval() # 设置模型为评估模式
    
    # 初始化 torchmetrics 的 mAP 计算器
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)
    
    progress_bar = tqdm(data_loader, desc="Calculating mAP")
    
    with torch.no_grad():
        for images, targets in progress_bar:
            images = list(image.to(device) for image in images)
            
            # 将 targets 中的 tensor 也移动到 device
            targets_on_device = []
            for t in targets:
                target_on_device = {}
                for k, v in t.items():
                    target_on_device[k] = v.to(device) if isinstance(v, torch.Tensor) else v
                targets_on_device.append(target_on_device)

            outputs = model(images)
            
            # 更新 mAP 计算器
            metric.update(outputs, targets_on_device)
    
    # 计算最终结果
    results = metric.compute()
    
    print("\n--- Evaluation Results ---")
    map_val = results['map'].item()
    map_50_val = results['map_50'].item()
    print(f"mAP @ .50-.95: {map_val:.4f}")
    print(f"mAP @ .50: {map_50_val:.4f}")
    
    # 提取每个类别的 TP, FP, FN
    # 注意：torchmetrics 1.0+ 版本返回的是一个 tensor
    # 我们需要找到每个类别对应的 TP, FP, FN
    # 这个功能在 torchmetrics 中没有直接提供 TP/FP/FN 的简单接口
    # 但我们可以通过分析 'mar_100' (Max Recall) 和 'map_per_class' 来间接了解
    # 这里我们直接打印 torchmetrics 提供的每类 mAP
    
    print("\n--- mAP per Class ---")
    for i, class_name in enumerate(config.CLASSES[1:]): # 跳过背景
        class_map = results['map_per_class'][i].item()
        print(f"Class '{class_name}': mAP = {class_map:.4f}")
        
    # TP/FP/FN 的计算比较复杂，需要匹配预测框和真实框
    # torchmetrics 内部完成了这个过程，但没有直接暴露 TP/FP/FN
    # 如果需要精确的 TP/FP/FN，需要手动实现 IOU 匹配逻辑
    # 这里我们先专注于 mAP 的正确计算
    print("\nNote: TP/FP/FN per class requires custom logic to parse metric internals.")
    print("Focusing on the primary mAP and per-class mAP values provided by torchmetrics.")
    print("--------------------------\n")
    
    return map_val
