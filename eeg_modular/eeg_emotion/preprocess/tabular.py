from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TabularPreprocessConfig:
    # core
    impute_strategy: str = "mean"
    scale: bool = True
    select_k_best: Optional[int] = None
    pca_n_components: Optional[int] = None

    # augmentation (train only)
    augment: bool = True
    noise_std: float = 0.01
    time_jitter: float = 0.02


class TabularPreprocessor:
    """Tabular preprocessing with strict no-leakage semantics.

    Rules:
    - fit() MUST be called only on train split (X_train, y_train)
    - transform() can be used on train/val/test after fit
    - augmentation (if enabled) is applied ONLY to training data and ONLY inside fit_transform_train()
    """

    def __init__(self, cfg: TabularPreprocessConfig):
        self.cfg = cfg
        self.imputer = SimpleImputer(strategy=cfg.impute_strategy)
        self.scaler = StandardScaler() if cfg.scale else None
        self.selector = SelectKBest(score_func=f_classif, k=cfg.select_k_best) if cfg.select_k_best else None
        self.pca = PCA(n_components=cfg.pca_n_components) if cfg.pca_n_components else None
        self._is_fit = False

    @staticmethod
    def _augment(X: np.ndarray, y: np.ndarray, noise_std: float, time_jitter: float) -> Tuple[np.ndarray, np.ndarray]:
        X_jitter = X + np.random.uniform(-time_jitter, time_jitter, size=X.shape)
        X_noise = X_jitter + np.random.normal(0.0, noise_std, size=X.shape)
        return X_noise, y.copy()

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "TabularPreprocessor":
        # imputer always fit on train
        X1 = self.imputer.fit_transform(X_train)

        # scaler fits after impute
        if self.scaler is not None:
            X1 = self.scaler.fit_transform(X1)

        # selector requires y
        if self.selector is not None:
            if y_train is None:
                raise ValueError("select_k_best requires y_train")
            X1 = self.selector.fit_transform(X1, y_train)

        if self.pca is not None:
            X1 = self.pca.fit_transform(X1)

        self._is_fit = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("TabularPreprocessor is not fit yet. Call fit() first with training data.")
        X1 = self.imputer.transform(X)
        if self.scaler is not None:
            X1 = self.scaler.transform(X1)
        if self.selector is not None:
            X1 = self.selector.transform(X1)
        if self.pca is not None:
            X1 = self.pca.transform(X1)
        return X1

    def fit_transform_train(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit on raw train data, then return transformed *and optionally augmented* train data."""
        # Fit/transform without leakage
        self.fit(X_train, y_train=y_train)
        X_t = self.transform(X_train)

        if self.cfg.augment:
            X_t, y_train = self._augment(X_t, y_train, noise_std=self.cfg.noise_std, time_jitter=self.cfg.time_jitter)

        return X_t, y_train

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
        joblib.dump(payload, os.path.join(out_dir, "tabular_preprocess.joblib"))

    @classmethod
    def load(cls, in_dir: str) -> "TabularPreprocessor":
        payload = joblib.load(os.path.join(in_dir, "tabular_preprocess.joblib"))
        obj = cls(payload["cfg"])
        obj.imputer = payload["imputer"]
        obj.scaler = payload["scaler"]
        obj.selector = payload["selector"]
        obj.pca = payload["pca"]
        obj._is_fit = bool(payload["_is_fit"])
        return obj
