from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class ClassWeightConfig:
    sad_multiplier: float = 3.0
    normalize: bool = True


def compute_balanced_class_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(np.int64).ravel()
    counts = np.bincount(y)
    counts = np.where(counts == 0, 1, counts)
    total = len(y)
    w = total / (len(counts) * counts)
    return w.astype(np.float32)


def apply_manual_weights(class_names: Sequence[str], weights: np.ndarray, cfg: ClassWeightConfig) -> np.ndarray:
    """Mimic your original behavior: boost 'Sad' weight if present."""
    w = weights.astype(np.float32).copy()
    if not class_names:
        return w
    # find Sad ignoring case
    sad_idx = None
    for i, n in enumerate(class_names):
        if str(n).lower() == "sad":
            sad_idx = i
            break
    if sad_idx is not None and sad_idx < len(w):
        w[sad_idx] = w[sad_idx] * float(cfg.sad_multiplier)
    if cfg.normalize:
        w = w / w.sum() * len(w)
    return w


def to_tensor(weights: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(weights, dtype=torch.float32, device=device)
