# src/config.py

import torch

# --- 基础配置 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 2  # 数据加载器的工作进程数
RANDOM_SEED = 37 # 随机种子，确保结果可复现

# --- 路径配置 ---
ROOT_DIR = '..' # 项目根目录
DATA_DIR = f'{ROOT_DIR}/data'
OUTPUT_DIR = f'{ROOT_DIR}/outputs'
MODEL_SAVE_PATH = f'{OUTPUT_DIR}/models'
LOGS_PATH = f'{OUTPUT_DIR}/logs'

# --- 数据集配置 ---
# 请根据您的数据集修改类别名称，'__background__' 是必须的
# 根据新的 README.md 更新类别
CLASSES = [
    '__background__',
    '1-5-Volt-Battery',
    '3-3-Volt-Battery',
    '7-Segment-Display',
    '9-Volt-Battery',
    'Arduino-Mega',
    'Arduino-Nano',
    'Arduino-Uno',
    'BJT-Transistor',
    'Bluetooth-Module',
    'Breadboard',
    'Bridge-Rectifier',
    'Buck-Converter',
    'Buzzer',
    'Capacitor-10mf',
    'Capacitor-470mf',
    'DC-Motor',
    'Diode',
    'ESP32',
    'ESP32-CAM',
    'FT-232-USB-Serial-Module',
    'Film-Capacitor',
    'Fuse',
    'Fuse-Base',
    'GSM-Module',
    'Gas-Sensor',
    'Heat-Sink',
    'High-Voltage-Ceramic-Capacitor',
    'Humidity-Sensor',
    'IC-Base-14-Pin',
    'IC-Base-28-Pin',
    'IC-Chip',
    'IGBT'
]
NUM_CLASSES = len(CLASSES)

# --- 训练超参数 ---
BATCH_SIZE = 4
LEARNING_RATE = 0.005
NUM_EPOCHS = 10
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# --- 数据增强开关 ---
# 设置为 True 来开启训练集的数据增强
APPLY_AUGMENTATION = False

# --- 可视化配置 ---
VISUALIZE_TRANSFORMED_IMAGES = False # 是否可视化查看增强后的图片
