# src/config.py

import torch

# --- 基础配置 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 4  # 数据加载器的工作进程数
RANDOM_SEED = 42 # 随机种子，确保结果可复现

# --- 路径配置 ---
ROOT_DIR = '..' # 项目根目录
DATA_DIR = f'{ROOT_DIR}/data'
OUTPUT_DIR = f'{ROOT_DIR}/outputs'
MODEL_SAVE_PATH = f'{OUTPUT_DIR}/models'
LOGS_PATH = f'{OUTPUT_DIR}/logs'

# --- 数据集配置 ---
# 请根据您的数据集修改类别名称，'__background__' 是必须的
CLASSES = [
    '__background__', 'resistor', 'capacitor', 'inductor', 'diode' 
    # 示例类别，请替换为您的真实类别
]
NUM_CLASSES = len(CLASSES)

# --- 训练超参数 ---
BATCH_SIZE = 4
LEARNING_RATE = 0.005
NUM_EPOCHS = 20
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# --- 数据增强开关 ---
# 设置为 True 来开启训练集的数据增强
APPLY_AUGMENTATION = True 

# --- 可视化配置 ---
VISUALIZE_TRANSFORMED_IMAGES = False # 是否可视化查看增强后的图片
