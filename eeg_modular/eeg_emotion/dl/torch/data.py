from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MultiModalNPYConfig:
    data_dir: str
    modalities: List[str]
    x_train_prefix: str = "X_train_"
    x_test_prefix: str = "X_test_"
    y_train_name: str = "y_train_filtered.npy"
    y_test_name: str = "y_test_filtered.npy"
    label_encoder_name: str = "label_encoder.joblib"
    time_steps: Optional[int] = 10
    feat_dim: Optional[int] = 4


def _normalize_seq(arr: np.ndarray, time_steps: Optional[int], feat_dim: Optional[int], name: str) -> np.ndarray:
    """Ensure arr is (N, T, F). Accept (N,F,T) or other reshapes if possible."""
    if arr.ndim != 3:
        raise ValueError(f"{name} expected 3D array, got shape={arr.shape}")
    n, a, b = arr.shape

    # If expected provided
    if time_steps is not None and feat_dim is not None:
        if (a, b) == (time_steps, feat_dim):
            return arr
        if (a, b) == (feat_dim, time_steps):
            return np.transpose(arr, (0, 2, 1))
        # attempt reshape if total matches
        if a * b == time_steps * feat_dim:
            return arr.reshape(n, time_steps, feat_dim)
        raise ValueError(f"{name} shape {arr.shape} cannot be normalized to (N,{time_steps},{feat_dim})")

    # Infer: prefer time_steps=10, feat_dim=4 pattern if present
    if (a, b) == (10, 4) or (a, b) == (128, 4):
        return arr
    if (a, b) == (4, 10) or (a, b) == (4, 128):
        return np.transpose(arr, (0, 2, 1))

    # fallback: keep as-is but ensure float32
    return arr


def load_multimodal_npy(cfg: MultiModalNPYConfig) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray, List[str]]:
    X_train_dict: Dict[str, np.ndarray] = {}
    X_test_dict: Dict[str, np.ndarray] = {}

    for m in cfg.modalities:
        tr_path = os.path.join(cfg.data_dir, f"{cfg.x_train_prefix}{m}.npy")
        te_path = os.path.join(cfg.data_dir, f"{cfg.x_test_prefix}{m}.npy")
        if not os.path.exists(tr_path):
            raise FileNotFoundError(f"Missing {tr_path}")
        if not os.path.exists(te_path):
            raise FileNotFoundError(f"Missing {te_path}")
        X_train_dict[m] = _normalize_seq(np.load(tr_path), cfg.time_steps, cfg.feat_dim, name=f"X_train_{m}").astype(np.float32)
        X_test_dict[m] = _normalize_seq(np.load(te_path), cfg.time_steps, cfg.feat_dim, name=f"X_test_{m}").astype(np.float32)

    y_train = np.load(os.path.join(cfg.data_dir, cfg.y_train_name))
    y_test = np.load(os.path.join(cfg.data_dir, cfg.y_test_name))

    # y_train might be one-hot
    if y_train.ndim == 2:
        y_train = np.argmax(y_train, axis=1)
    y_train = y_train.astype(np.int64).ravel()

    # y_test might be (N,1) or one-hot
    if y_test.ndim == 2 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)
    y_test = y_test.astype(np.int64).ravel()

    le_path = os.path.join(cfg.data_dir, cfg.label_encoder_name)
    if os.path.exists(le_path):
        le = joblib.load(le_path)
        class_names = list(getattr(le, "classes_", []))
    else:
        class_names = []
    if not class_names:
        # fallback by max label
        class_names = [str(i) for i in range(int(np.max(y_train)) + 1)]

    # sanity: all modalities aligned
    n = len(y_train)
    for k, v in X_train_dict.items():
        if len(v) != n:
            raise ValueError(f"Train modality {k} has {len(v)} samples but y_train has {n}")
    n2 = len(y_test)
    for k, v in X_test_dict.items():
        if len(v) != n2:
            raise ValueError(f"Test modality {k} has {len(v)} samples but y_test has {n2}")

    return X_train_dict, X_test_dict, y_train, y_test, class_names


class MultiModalTensorDataset(Dataset):
    def __init__(self, X_dict: Dict[str, np.ndarray], y: np.ndarray):
        self.X_dict = X_dict
        self.y = y.astype(np.int64)
        n = len(self.y)
        for k, v in self.X_dict.items():
            if len(v) != n:
                raise ValueError(f"Modality {k} has {len(v)} samples, but y has {n} samples")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        sample = {k: torch.tensor(v[idx], dtype=torch.float32) for k, v in self.X_dict.items()}
        return sample, torch.tensor(self.y[idx], dtype=torch.long)
