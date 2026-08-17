from __future__ import annotations

import numpy as np
import joblib
from pathlib import Path
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class TrainingDataSampler:
    """从训练数据中采样，用于实时推理的mock数据"""
    
    def __init__(self, features_dir: str = "../features", hold_samples: int = 30):
        """
        Args:
            features_dir: 特征目录
            hold_samples: 每个情绪样本保持多少帧（让变化更慢）
        """
        self.features_dir = Path(features_dir)
        self.X_train_dict = {}
        self.y_train = None
        self.current_idx = 0
        self.hold_samples = hold_samples  # 每个样本保持的帧数
        self.hold_counter = 0  # 当前样本的保持计数器
        self._load_data()
    
    def _load_data(self):
        """加载训练数据"""
        logger.info("Loading training data for sampling...")
        
        modalities = ['filtered', 'powerspec', 'att', 'med']
        
        for mod in modalities:
            train_path = self.features_dir / f"X_train_{mod}.npy"
            if train_path.exists():
                self.X_train_dict[mod] = np.load(train_path)
                logger.info(f"Loaded {mod}: shape={self.X_train_dict[mod].shape}")
        
        # 加载标签
        y_train_path = self.features_dir / "y_train_filtered.npy"
        if y_train_path.exists():
            self.y_train = np.load(y_train_path)
            if self.y_train.ndim == 2:
                self.y_train = np.argmax(self.y_train, axis=1)
            logger.info(f"Loaded labels: shape={self.y_train.shape}")
        
        logger.info("Training data loaded successfully")
    
    def get_sample(self, idx: Optional[int] = None) -> tuple[Dict[str, np.ndarray], Optional[int]]:
        """获取一个样本
        
        Args:
            idx: 样本索引，如果为None则使用循环采样
            
        Returns:
            (多模态特征字典, 标签)
        """
        if idx is None:
            # 检查是否需要切换到下一个样本
            self.hold_counter += 1
            if self.hold_counter >= self.hold_samples:
                self.current_idx = (self.current_idx + 1) % len(self.y_train)
                self.hold_counter = 0
                logger.debug(f"[TRAIN_DATA] Switched to sample #{self.current_idx}")
            idx = self.current_idx
        
        result = {}
        for mod, arr in self.X_train_dict.items():
            result[mod] = arr[idx].copy()
        
        # 获取对应的标签
        label = None
        if self.y_train is not None:
            label = int(self.y_train[idx])
        
        return result, label
    
    def get_random_sample(self) -> tuple[Dict[str, np.ndarray], Optional[int]]:
        """随机获取一个样本"""
        idx = np.random.randint(0, len(self.y_train))
        return self.get_sample(idx)


# 全局采样器实例
_sampler = None


def get_sampler(features_dir: str = "../features", hold_samples: int = 30) -> TrainingDataSampler:
    """获取采样器单例"""
    global _sampler
    if _sampler is None:
        _sampler = TrainingDataSampler(features_dir, hold_samples=hold_samples)
    return _sampler
