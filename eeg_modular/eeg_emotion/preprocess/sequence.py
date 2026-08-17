from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from joblib import dump, load
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class SequencePreprocessConfig:
    impute_strategy: str = "mean"
    scale: bool = True
    pca_n_components: Optional[int] = 64


class SequencePreprocessor:
    def __init__(self, cfg: SequencePreprocessConfig):
        self.cfg = cfg
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        # X: (N, T, F)
        n, t, f = X.shape
        Xr = X.reshape(-1, f)

        self.imputer = SimpleImputer(strategy=self.cfg.impute_strategy)
        Xr = self.imputer.fit_transform(Xr)

        if self.cfg.scale:
            self.scaler = StandardScaler()
            Xr = self.scaler.fit_transform(Xr)

        pca_n = self.cfg.pca_n_components
        if pca_n is not None:
            pca_n = int(min(pca_n, f))
            self.pca = PCA(n_components=pca_n)
            Xr = self.pca.fit_transform(Xr)
            f2 = pca_n
        else:
            f2 = Xr.shape[1]

        return Xr.reshape(n, t, f2).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.imputer is None:
            raise RuntimeError("SequencePreprocessor is not fit.")
        n, t, f = X.shape
        Xr = X.reshape(-1, f)
        Xr = self.imputer.transform(Xr)

        if self.scaler is not None:
            Xr = self.scaler.transform(Xr)

        if self.pca is not None:
            Xr = self.pca.transform(Xr)

        return Xr.reshape(n, t, Xr.shape[1]).astype(np.float32)

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        if self.imputer is not None:
            dump(self.imputer, os.path.join(out_dir, "imputer.joblib"))
        if self.scaler is not None:
            dump(self.scaler, os.path.join(out_dir, "scaler.joblib"))
        if self.pca is not None:
            dump(self.pca, os.path.join(out_dir, "pca.joblib"))

    @classmethod
    def load(cls, in_dir: str, cfg: SequencePreprocessConfig) -> "SequencePreprocessor":
        obj = cls(cfg)
        imp = os.path.join(in_dir, "imputer.joblib")
        sca = os.path.join(in_dir, "scaler.joblib")
        pca = os.path.join(in_dir, "pca.joblib")
        if os.path.exists(imp):
            obj.imputer = load(imp)
        if os.path.exists(sca):
            obj.scaler = load(sca)
        if os.path.exists(pca):
            obj.pca = load(pca)
        return obj
