from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from eeg_emotion.models.base import ModelAdapter


@dataclass(frozen=True)
class RFConfig:
    param_grid: Optional[Dict[str, Any]] = None
    cv: int = 5
    n_jobs: int = -1
    random_state: int = 42


class RFAdapter(ModelAdapter):
    def __init__(self, cfg: RFConfig):
        self.cfg = cfg
        self.model: Optional[RandomForestClassifier] = None
        self.best_params_: Optional[Dict[str, Any]] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "RFAdapter":
        base = RandomForestClassifier(random_state=self.cfg.random_state, **kwargs)
        if self.cfg.param_grid:
            grid = GridSearchCV(
                base,
                self.cfg.param_grid,
                refit=True,
                cv=self.cfg.cv,
                n_jobs=self.cfg.n_jobs,
            )
            grid.fit(X, y)
            self.model = grid.best_estimator_
            self.best_params_ = dict(grid.best_params_)
        else:
            base.fit(X, y)
            self.model = base
            self.best_params_ = None
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("RF model is not fit.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray):
        if self.model is None:
            raise RuntimeError("RF model is not fit.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def save(self, out_dir: str) -> None:
        if self.model is None:
            raise RuntimeError("RF model is not fit.")
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(
            {"cfg": self.cfg, "model": self.model, "best_params": self.best_params_},
            os.path.join(out_dir, "rf.joblib"),
        )

    @classmethod
    def load(cls, in_dir: str) -> "RFAdapter":
        payload = joblib.load(os.path.join(in_dir, "rf.joblib"))
        obj = cls(payload["cfg"])
        obj.model = payload["model"]
        obj.best_params_ = payload.get("best_params")
        return obj
