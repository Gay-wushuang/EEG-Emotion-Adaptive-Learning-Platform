from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.svm import SVC, LinearSVC


@dataclass
class SVMConfig:
    # Choose optimizer/solver backend
    # - "svc"      : kernel SVM (libsvm) – convex optimization, supports kernels
    # - "linearsvc": linear SVM (liblinear) – convex optimization, faster for large feature dims
    solver: str = "svc"

    # Common
    class_weight: Optional[str | Dict[int, float]] = "balanced"
    max_iter: int = -1
    tol: float = 1e-3

    # --- Kernel SVM (SVC) options ---
    kernel: str = "rbf"  # linear | rbf | poly | sigmoid
    C: float = 1.0
    gamma: str | float = "scale"
    degree: int = 3
    coef0: float = 0.0
    shrinking: bool = True
    probability: bool = False
    cache_size: float = 1024.0

    # --- LinearSVC options (convex) ---
    loss: str = "squared_hinge"  # hinge | squared_hinge
    dual: str | bool = "auto"    # "auto" or bool
    fit_intercept: bool = True
    intercept_scaling: float = 1.0


class SVMModel:
    """Sklearn SVM adapter with two backends:
    - SVC: supports kernels (convex QP solved by libsvm)
    - LinearSVC: linear SVM (convex) solved by liblinear (often faster)

    This module focuses on:
    1) Normalization handled upstream in preprocess (StandardScaler/MinMax).
    2) Kernel selection & hyperparams via config.
    3) Convex optimization controls via max_iter/tol and backend selection.
    """

    def __init__(self, cfg: SVMConfig):
        self.cfg = cfg
        self.model = self._build()

    def _build(self):
        if str(self.cfg.solver).lower() == "linearsvc":
            # Convex optimization (linear SVM). No kernel, but very stable.
            dual = self.cfg.dual
            if isinstance(dual, str) and dual.lower() == "auto":
                # sklearn will choose automatically if bool not provided (newer versions)
                dual = "auto"
            return LinearSVC(
                C=float(self.cfg.C),
                class_weight=self.cfg.class_weight,
                max_iter=int(self.cfg.max_iter if self.cfg.max_iter != -1 else 10000),
                tol=float(self.cfg.tol),
                loss=str(self.cfg.loss),
                dual=dual,
                fit_intercept=bool(self.cfg.fit_intercept),
                intercept_scaling=float(self.cfg.intercept_scaling),
            )

        # Default: SVC with kernel (convex QP)
        return SVC(
            C=float(self.cfg.C),
            kernel=str(self.cfg.kernel),
            gamma=self.cfg.gamma,
            degree=int(self.cfg.degree),
            coef0=float(self.cfg.coef0),
            class_weight=self.cfg.class_weight,
            shrinking=bool(self.cfg.shrinking),
            probability=bool(self.cfg.probability),
            cache_size=float(self.cfg.cache_size),
            max_iter=int(self.cfg.max_iter),
            tol=float(self.cfg.tol),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise RuntimeError("This SVM backend does not support predict_proba. Set probability=true for SVC.")

    def save(self, path: str) -> None:
        joblib.dump({"cfg": self.cfg, "model": self.model}, path)

    @staticmethod
    def load(path: str) -> "SVMModel":
        obj = joblib.load(path)
        cfg = obj["cfg"]
        m = SVMModel(cfg)
        m.model = obj["model"]
        return m


def from_dict(d: Dict[str, Any]) -> SVMConfig:
    """Convert config dict -> SVMConfig with defaults."""
    return SVMConfig(
        solver=str(d.get("solver", "svc")),
        class_weight=d.get("class_weight", "balanced"),
        max_iter=int(d.get("max_iter", -1)),
        tol=float(d.get("tol", 1e-3)),

        kernel=str(d.get("kernel", "rbf")),
        C=float(d.get("C", 1.0)),
        gamma=d.get("gamma", "scale"),
        degree=int(d.get("degree", 3)),
        coef0=float(d.get("coef0", 0.0)),
        shrinking=bool(d.get("shrinking", True)),
        probability=bool(d.get("probability", False)),
        cache_size=float(d.get("cache_size", 1024.0)),

        loss=str(d.get("loss", "squared_hinge")),
        dual=d.get("dual", "auto"),
        fit_intercept=bool(d.get("fit_intercept", True)),
        intercept_scaling=float(d.get("intercept_scaling", 1.0)),
    )
SVMClassifier = SVMModel