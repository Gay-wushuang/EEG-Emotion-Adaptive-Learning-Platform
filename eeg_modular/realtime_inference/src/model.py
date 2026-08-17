from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import joblib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eeg_emotion.models.torch.multimodal_cvae_cnn import MultiModalCVAECNN, MultiModalCVAECNNConfig
from eeg_emotion.models.torch.cvae_model import CVAE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceConfig:
    model_path: Path
    model_type: str = "multimodal_cnn"
    device: str = "auto"
    num_classes: int = 3
    modalities: List[str] = ("filtered", "powerspec", "att", "med")
    time_steps: int = 10
    feat_dim: int = 4
    use_cvae: bool = True
    cvae_latent_dim: int = 64
    cvae_input_dim: int = 160
    cvae_checkpoint: Optional[str] = None
    dropout: float = 0.5
    scalers_dir: Optional[Path] = None
    skip_scaling: bool = False  # 跳过归一化（用于已经归一化的训练数据）


class EmotionInferenceModel:
    """情绪推理模型加载与推理模块"""

    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.device = self._init_device()
        self.scalers = self._load_scalers()
        self.class_names = self._load_class_names()
        self.model = self._load_model()

    def _init_device(self) -> torch.device:
        if self.cfg.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.cfg.device)

    def _load_scalers(self) -> Dict[str, any]:
        """Load and validate one training scaler for every configured modality."""
        scalers = {}
        
        if self.cfg.skip_scaling:
            logger.warning("Scaling explicitly disabled; inputs must already be standardized")
            return scalers
        if self.cfg.scalers_dir is None:
            raise ValueError("scalers_dir is required when skip_scaling=false")
        
        scalers_dir = Path(self.cfg.scalers_dir)
        if not scalers_dir.is_dir():
            raise FileNotFoundError(f"Scalers directory not found: {scalers_dir.resolve()}")
        
        for mod in self.cfg.modalities:
            scaler_path = scalers_dir / f"scaler_{mod}.joblib"
            if not scaler_path.is_file():
                raise FileNotFoundError(f"Missing scaler for modality '{mod}': {scaler_path.resolve()}")
            scaler = joblib.load(scaler_path)
            n_features = getattr(scaler, "n_features_in_", None)
            if n_features is None:
                raise TypeError(f"Scaler for '{mod}' has no n_features_in_: {scaler_path}")
            if int(n_features) != self.cfg.feat_dim:
                raise ValueError(
                    f"Scaler '{mod}' expects {n_features} features, model expects {self.cfg.feat_dim}"
                )
            scalers[mod] = scaler
            logger.info("Loaded scaler mapping %s -> %s", mod, scaler_path)
        if set(scalers) != set(self.cfg.modalities):
            raise RuntimeError("Scaler validation incomplete; refusing to start")
        
        return scalers

    def _load_class_names(self) -> List[str]:
        if self.cfg.scalers_dir is None:
            raise ValueError("scalers_dir is required to load label_encoder.joblib")
        encoder_path = Path(self.cfg.scalers_dir) / "label_encoder.joblib"
        if not encoder_path.is_file():
            raise FileNotFoundError(f"Missing label encoder: {encoder_path.resolve()}")
        encoder = joblib.load(encoder_path)
        class_names = [str(name) for name in getattr(encoder, "classes_", [])]
        if len(class_names) != self.cfg.num_classes:
            raise ValueError(
                f"Label encoder has {len(class_names)} classes, model expects {self.cfg.num_classes}"
            )
        logger.info("Loaded class order: %s", class_names)
        return class_names

    def _apply_scalers(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """应用归一化器 - 按照原始项目的方式"""
        result = {}
        
        # 如果配置了跳过归一化，直接返回原始数据
        if self.cfg.skip_scaling:
            logger.info("[SCALER] Skipping scaling (using pre-normalized training data)")
            for mod in self.cfg.modalities:
                result[mod] = data.get(mod, np.zeros((self.cfg.time_steps, self.cfg.feat_dim), dtype=np.float32))
            return result
        
        for mod in self.cfg.modalities:
            if mod not in data:
                raise KeyError(f"Missing required modality: {mod}")
            arr = np.asarray(data[mod], dtype=np.float32)
            if arr.ndim != 2 or arr.shape != (self.cfg.time_steps, self.cfg.feat_dim):
                raise ValueError(
                    f"Modality '{mod}' must have shape "
                    f"({self.cfg.time_steps}, {self.cfg.feat_dim}), got {arr.shape}"
                )
            if not np.isfinite(arr).all():
                raise ValueError(f"Modality '{mod}' contains NaN or Inf before scaling")
            
            logger.debug(f"[SCALER] {mod} - Before scaling: mean={arr.mean():.4f}, std={arr.std():.4f}")
            
            scaler = self.scalers[mod]
            result[mod] = scaler.transform(arr).astype(np.float32)
            if result[mod].shape != arr.shape or not np.isfinite(result[mod]).all():
                raise ValueError(f"Scaler output invalid for modality '{mod}': {result[mod].shape}")
            logger.debug(f"[SCALER] {mod} - After scaling: mean={result[mod].mean():.4f}, std={result[mod].std():.4f}")
        
        return result

    def _load_model(self) -> nn.Module:
        logger.info(f"Loading model from: {self.cfg.model_path}")
        
        try:
            cvae_model = None
            if self.cfg.use_cvae:
                cvae_model = CVAE(
                    input_dim=self.cfg.cvae_input_dim,
                    num_classes=self.cfg.num_classes,
                    latent_dim=self.cfg.cvae_latent_dim,
                    hidden_dim=256,
                )
                cvae_model.eval()
                for p in cvae_model.parameters():
                    p.requires_grad = False
            
            mcfg = MultiModalCVAECNNConfig(
                modalities=list(self.cfg.modalities),
                dropout=self.cfg.dropout,
                cvae_latent_dim=self.cfg.cvae_latent_dim,
                use_cvae=self.cfg.use_cvae,
            )
            
            model = MultiModalCVAECNN(
                num_classes=self.cfg.num_classes,
                cfg=mcfg,
                cvae_model=cvae_model,
            )
            
            checkpoint = torch.load(self.cfg.model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
            
            model.to(self.device)
            model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    @torch.no_grad()
    def predict(self, data: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray]:
        """
        执行情绪分类推理
        
        Args:
            data: 多模态数据字典，每个模态形状为 (time_steps, feat_dim)
            
        Returns:
            (emotion_label, probabilities)
        """
        logger.debug("[MODEL] Starting inference...")
        
        data = self._apply_scalers(data)
        
        x_dict = {}
        for mod in self.cfg.modalities:
            arr = data[mod]
            if arr.shape[0] < self.cfg.time_steps:
                pad = np.zeros((self.cfg.time_steps - arr.shape[0], arr.shape[1]), dtype=np.float32)
                arr = np.vstack([arr, pad])
                logger.debug(f"[MODEL] Padded modality {mod} from {arr.shape[0]-pad.shape[0]} to {self.cfg.time_steps} steps")
            elif arr.shape[0] > self.cfg.time_steps:
                arr = arr[:self.cfg.time_steps]
                logger.debug(f"[MODEL] Truncated modality {mod} from {arr.shape[0]+self.cfg.time_steps} to {self.cfg.time_steps} steps")
            
            x_dict[mod] = torch.from_numpy(arr).unsqueeze(0).to(self.device, dtype=torch.float32)
        
        if self.cfg.use_cvae:
            conditional_probs = []
            for label_idx in range(self.cfg.num_classes):
                labels = torch.full((1,), label_idx, dtype=torch.long, device=self.device)
                conditional_probs.append(torch.softmax(self.model(x_dict, labels=labels), dim=1))
            probs = torch.stack(conditional_probs, dim=0).mean(dim=0).cpu().numpy()[0]
        else:
            outputs = self.model(x_dict)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        emotion = self.class_names[pred_idx]
        
        logger.debug("[MODEL] Probabilities - %s", dict(zip(self.class_names, probs.tolist())))
        logger.info(f"[MODEL] Predicted emotion: {emotion} (confidence={probs[pred_idx]:.4f})")
        
        return emotion, probs

    @torch.no_grad()
    def predict_batch(self, batch_data: Dict[str, np.ndarray]) -> Tuple[List[str], np.ndarray]:
        """批量推理"""
        batch_data = self._apply_scalers(batch_data)
        
        batch_size = batch_data[self.cfg.modalities[0]].shape[0]
        
        x_dict = {}
        for mod in self.cfg.modalities:
            arr = batch_data[mod]
            x_dict[mod] = torch.from_numpy(arr).to(self.device, dtype=torch.float32)
        
        outputs = self.model(x_dict)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        pred_indices = np.argmax(probs, axis=1)
        emotions = [self.class_names[idx] for idx in pred_indices]
        
        return emotions, probs
