from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


def plot_training_curves(history: Dict[str, list], save_path: str):
    """绘制训练曲线
    
    Args:
        history: 训练历史记录，包含loss、val_loss等键
        save_path: 保存路径
    """
    epochs = len(history['loss'])
    
    plt.figure(figsize=(12, 6))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(range(1, epochs + 1), history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线（如果存在）
    if 'val_acc' in history or 'acc' in history:
        plt.subplot(1, 2, 2)
        if 'acc' in history:
            plt.plot(range(1, epochs + 1), history['acc'], label='Training Accuracy')
        if 'val_acc' in history:
            plt.plot(range(1, epochs + 1), history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
