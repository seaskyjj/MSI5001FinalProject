# src/inference.py

import torch
import cv2
import numpy as np
import argparse
import os
import glob
import random

from . import config
from .model import create_model
from .utils import calculate_map
from .dataset import create_data_loader

# 为不同类别定义不同的颜色，方便可视化
# 使用 hsv 颜色空间生成区分度高的颜色
COLORS = np.random.uniform(0, 255, size=(len(config.CLASSES), 3))

def predict_and_visualize(model, image_path, device, confidence_threshold):
    """
    对单张图片进行预测并可视化结果。

    Args:
        model: 加载好的 PyTorch 模型。
        image_path (str): 待预测图片的路径。
        device (str): 'cuda' 或 'cpu'。
        confidence_threshold (float): 用于过滤低置信度预测的阈值。
    """
    # 1. 读取并预处理图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found at {image_path}")
        return
        
    original_image = image.copy() # 复制一份用于绘制
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 转换格式：HWC -> CHW, 并归一化
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device) # 增加 batch 维度

    # 2. 模型推理
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)

    # 3. 处理并可视化输出
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    
    if len(outputs[0]['boxes']) > 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        labels = outputs[0]['labels'].data.numpy()
        
        # 过滤掉低置信度的预测
        high_confidence_indices = scores >= confidence_threshold
        boxes = boxes[high_confidence_indices].astype(np.int32)
        scores = scores[high_confidence_indices]
        labels = labels[high_confidence_indices]
        
        print(f"Found {len(boxes)} objects with confidence > {confidence_threshold}")

        for i, box in enumerate(boxes):
            class_id = labels[i]
            class_name = config.CLASSES[class_id]
            score = scores[i]
            color = COLORS[class_id]

            # 绘制边界框
            cv2.rectangle(
                original_image,
                (box[0], box[1]),
                (box[2], box[3]),
                color, 2
            )
            # 准备标签文本
            label_text = f"{class_name}: {score:.2f}"
            
            # 计算文本框大小
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # 绘制文本背景
            cv2.rectangle(
                original_image,
                (box[0], box[1] - text_height - baseline),
                (box[0] + text_width, box[1]),
                color, -1 # -1 表示填充
            )
            
            # 绘制文本
            cv2.putText(
                original_image,
                label_text,
                (box[0], box[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1 # 黑色字体
            )
    else:
        print("No objects detected.")

    # 4. 保存结果
    output_dir = os.path.join(config.OUTPUT_DIR, 'inference_results')
    os.makedirs(output_dir, exist_ok=True)
    
    image_name = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f"result_{image_name}")
    cv2.imwrite(save_path, original_image)
    
    print(f"Visualization saved to: {save_path}")
    
    # 可选：直接显示图片
    # cv2.imshow('Prediction', original_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def main():
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="Inference script for object detection.")
    parser.add_argument(
        '-m', '--model-path', 
        default=None,
        help="Path to the trained model (.pth file). If not provided, finds the latest in outputs/models."
    )
    parser.add_argument(
        '-i', '--image-path', 
        default=None,
        help="Path to the image for visualization. If not provided, a random image from the test set is chosen."
    )
    parser.add_argument(
        '-t', '--threshold', 
        default=0.5, 
        type=float,
        help="Confidence threshold for predictions."
    )
    args = parser.parse_args()

    # --- 设备设置 ---
    device = config.DEVICE
    print(f"Using device: {device}")

    # --- 加载模型 ---
    model = create_model(num_classes=config.NUM_CLASSES)
    
    # 确定模型路径
    model_path = args.model_path
    if model_path is None:
        # 自动查找最新的模型文件
        model_files = glob.glob(os.path.join(config.MODEL_SAVE_PATH, '*.pth'))
        if not model_files:
            print("Error: No model file found in 'outputs/models/'. Please specify a path with --model-path.")
            return
        model_path = max(model_files, key=os.path.getctime)
        print(f"No model path specified. Using the latest model: {os.path.basename(model_path)}")
    
    # 加载模型权重
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    print("Model loaded successfully.")

    # --- 1. 在 Test 数据集上进行评估 ---
    print("\n--- Evaluating on Test Dataset ---")
    test_loader = create_data_loader(
        os.path.join(config.DATA_DIR, 'test'), 
        config.CLASSES, 
        config.BATCH_SIZE, 
        config.NUM_WORKERS, 
        is_train=False
    )
    if len(test_loader.dataset) > 0:
        calculate_map(model, test_loader, device)
    else:
        print("Test dataset is empty. Skipping evaluation.")

    # --- 2. 对单张图片进行可视化 ---
    print("\n--- Visualizing Single Image ---")
    image_path = args.image_path
    if image_path is None:
        # 自动从测试集随机选择一张图片
        test_image_paths = glob.glob(os.path.join(config.DATA_DIR, 'test', '*.jpg'))
        if not test_image_paths:
            print("No images found in 'data/test/' to visualize.")
            return
        image_path = random.choice(test_image_paths)
        print(f"No image path specified. Choosing a random image from test set: {os.path.basename(image_path)}")
    
    predict_and_visualize(model, image_path, device, args.threshold)


if __name__ == '__main__':
    main()
