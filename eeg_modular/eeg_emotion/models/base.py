from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class ModelAdapter(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "ModelAdapter":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        return None

    @abstractmethod
    def save(self, out_dir: str) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, in_dir: str) -> "ModelAdapter":
        ...
