from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from eeg_emotion.models.base import ModelAdapter
from eeg_emotion.models.sklearn.mlp import MLPAdapter, MLPConfig
from eeg_emotion.models.sklearn.rf import RFAdapter, RFConfig
from eeg_emotion.models.sklearn.svm import SVMModel, SVMConfig
from eeg_emotion.models.sklearn.xgb import XGBAdapter, XGBConfig


@dataclass(frozen=True)
class HybridConfig:
    mlp_config: MLPConfig
    rf_config: RFConfig
    svm_config: SVMConfig
    xgb_config: Optional[XGBConfig] = None  # XGBoost 配置
    voting_method: str = "soft"  # "soft" 或 "hard"
    use_stacking: bool = True
    stacking_meta_estimator: Optional[Any] = None
    random_state: int = 42


class HybridAdapter(ModelAdapter):
    def __init__(self, cfg: HybridConfig):
        self.cfg = cfg
        self.model: Optional[Any] = None
        self.base_models: List[Union[ModelAdapter, SVMModel]] = []
        self.best_params_: Optional[Dict[str, Any]] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> "HybridAdapter":
        # 初始化基础模型 - 4个机械模型
        mlp = MLPAdapter(self.cfg.mlp_config)
        rf = RFAdapter(self.cfg.rf_config)
        svm = SVMModel(self.cfg.svm_config)
        xgb = XGBAdapter(self.cfg.xgb_config if self.cfg.xgb_config else XGBConfig())
        
        # 训练每个基础模型
        mlp.fit(X, y, **kwargs)
        rf.fit(X, y, **kwargs)
        svm.fit(X, y)
        xgb.fit(X, y, **kwargs)
        
        self.base_models = [mlp, rf, svm, xgb]
        
        # 构建基础模型列表用于sklearn的集成方法
        estimators = [
            ("mlp", mlp.model),
            ("rf", rf.model),
            ("svm", svm.model),
            ("xgb", xgb.model),
        ]
        
        # 如果使用stacking
        if self.cfg.use_stacking:
            meta_estimator = self.cfg.stacking_meta_estimator or LogisticRegression(random_state=self.cfg.random_state)
            self.model = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_estimator,
                cv=5,
                stack_method="auto",
                n_jobs=-1,
            )
        else:
            # 使用voting
            self.model = VotingClassifier(
                estimators=estimators,
                voting=self.cfg.voting_method,
                n_jobs=-1,
            )
        
        # 训练集成模型
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Hybrid model is not fit.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            raise RuntimeError("Hybrid model is not fit.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def save(self, out_dir: str) -> None:
        if self.model is None:
            raise RuntimeError("Hybrid model is not fit.")
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(
            {
                "cfg": self.cfg,
                "model": self.model,
                "base_models": self.base_models,
                "best_params": self.best_params_
            },
            os.path.join(out_dir, "hybrid.joblib"),
        )

    @classmethod
    def load(cls, in_dir: str) -> "HybridAdapter":
        payload = joblib.load(os.path.join(in_dir, "hybrid.joblib"))
        obj = cls(payload["cfg"])
        obj.model = payload["model"]
        obj.base_models = payload.get("base_models", [])
        obj.best_params_ = payload.get("best_params")
        return obj