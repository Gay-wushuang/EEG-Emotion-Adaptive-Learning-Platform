from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessConfig:
    impute_strategy: str = "mean"
    scale: bool = True
    pca_n_components: Optional[int] = None
    select_k_best: Optional[int] = None  # uses ANOVA F-test by default


class SklearnPreprocessPipeline:
    """A minimal, explicit preprocess pipeline with fit/transform + save/load.

    Why explicit instead of sklearn.Pipeline?
    - easier to serialize config + components together
    - you can later add modality-specific transforms without fighting Pipeline constraints
    """

    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self.imputer = SimpleImputer(strategy=cfg.impute_strategy)
        self.scaler = StandardScaler() if cfg.scale else None
        self.selector = SelectKBest(score_func=f_classif, k=cfg.select_k_best) if cfg.select_k_best else None
        self.pca = PCA(n_components=cfg.pca_n_components) if cfg.pca_n_components else None
        self._is_fit = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SklearnPreprocessPipeline":
        X1 = self.imputer.fit_transform(X)

        if self.scaler is not None:
            X1 = self.scaler.fit_transform(X1)

        if self.selector is not None:
            if y is None:
                raise ValueError("select_k_best requires y for fit()")
            X1 = self.selector.fit_transform(X1, y)

        if self.pca is not None:
            X1 = self.pca.fit_transform(X1)

        self._is_fit = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("Pipeline is not fit yet. Call fit() first.")

        X1 = self.imputer.transform(X)

        if self.scaler is not None:
            X1 = self.scaler.transform(X1)

        if self.selector is not None:
            X1 = self.selector.transform(X1)

        if self.pca is not None:
            X1 = self.pca.transform(X1)

        return X1

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y=y)
        return self.transform(X)

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        payload = {
            "cfg": self.cfg,
            "imputer": self.imputer,
            "scaler": self.scaler,
            "selector": self.selector,
            "pca": self.pca,
            "_is_fit": self._is_fit,
        }
        joblib.dump(payload, os.path.join(out_dir, "preprocess.joblib"))

    @classmethod
    def load(cls, in_dir: str) -> "SklearnPreprocessPipeline":
        payload = joblib.load(os.path.join(in_dir, "preprocess.joblib"))
        pipe = cls(payload["cfg"])
        pipe.imputer = payload["imputer"]
        pipe.scaler = payload["scaler"]
        pipe.selector = payload["selector"]
        pipe.pca = payload["pca"]
        pipe._is_fit = bool(payload["_is_fit"])
        return pipe


def augment_tabular(X: np.ndarray, y: np.ndarray, noise_std: float = 0.01, time_jitter: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """Simple augmentation: uniform jitter + gaussian noise (train-only)."""
    X_jitter = X + np.random.uniform(-time_jitter, time_jitter, size=X.shape)
    X_noise = X_jitter + np.random.normal(0.0, noise_std, size=X.shape)
    return X_noise, y.copy()
