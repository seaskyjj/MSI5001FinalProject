# src/train.py

import torch
import time
import os
from tqdm import tqdm

from . import config
from .dataset import create_data_loader
from .model import create_model
from .utils import calculate_map

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """训练一个 epoch"""
    model.train() # 设置模型为训练模式
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
    total_loss = 0.0
    
    for i, (images, targets) in enumerate(progress_bar):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 前向传播并计算损失
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        # 更新进度条显示
        progress_bar.set_postfix(loss=f"{losses.item():.4f}")
        
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1} Average Training Loss: {avg_loss:.4f}")
    return avg_loss

def main():
    # 设置随机种子
    torch.manual_seed(config.RANDOM_SEED)
    
    # 确保输出目录存在
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.LOGS_PATH, exist_ok=True)
    
    # 创建数据加载器
    train_loader = create_data_loader(
        os.path.join(config.DATA_DIR, 'train'), 
        config.CLASSES, 
        config.BATCH_SIZE, 
        config.NUM_WORKERS, 
        is_train=True
    )
    valid_loader = create_data_loader(
        os.path.join(config.DATA_DIR, 'valid'), 
        config.CLASSES, 
        config.BATCH_SIZE, 
        config.NUM_WORKERS, 
        is_train=False
    )
    
    # 创建模型
    model = create_model(num_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)
    
    # 创建优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr=config.LEARNING_RATE, 
        momentum=config.MOMENTUM, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 学习率调度器 (可选)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    print("--- Starting Training ---")
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Data Augmentation: {'Enabled' if config.APPLY_AUGMENTATION else 'Disabled'}")
    print("-------------------------")
    
    best_map = 0.0
    
    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()
        
        # 训练
        train_one_epoch(model, optimizer, train_loader, config.DEVICE, epoch)
        
        # 更新学习率
        if lr_scheduler:
            lr_scheduler.step()
        
        # 在验证集上评估
        print(f"\n--- Validating Epoch {epoch+1} ---")
        current_map = calculate_map(model, valid_loader, config.DEVICE)
        
        # 保存最佳模型
        if current_map > best_map:
            best_map = current_map
            model_save_name = f"best_model_map_{best_map:.4f}.pth"
            torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_PATH, model_save_name))
            print(f"🎉 New best model saved with mAP: {best_map:.4f} as {model_save_name}")
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch {epoch+1} duration: {epoch_duration:.2f} seconds\n")

    print("--- Training Finished ---")

if __name__ == '__main__':
    main()
