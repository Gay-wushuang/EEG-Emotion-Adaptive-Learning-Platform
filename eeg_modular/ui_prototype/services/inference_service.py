"""推理后台线程。

预留接口：mock 模式根据模拟情绪趋势生成概率；
真实模式预留对接后端 EmotionInferenceModel.predict()。

双层结果：
  1. 模型三分类概率（positive / neutral / negative）
  2. 独立信号可信度（quality_level: trusted / warning / rejected）
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from services.dashboard_state import (
    WARMUP_SECONDS, MAX_POOR_SIGNAL, CLASS_NAMES,
    INFERENCE_INTERVAL,
)


class InferenceWorker(QThread):
    """推理工作线程。

    Signals:
        result_ready(dict): probabilities, predicted_state, confidence
    """

    result_ready = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._emotion_trend = 1
        self._ewma: Optional[np.ndarray] = None
        self._alpha = 0.20

    def set_emotion_trend(self, trend: int):
        self._emotion_trend = trend

    def run(self):
        self._running = True
        while self._running:
            self._step()
            self.msleep(int(INFERENCE_INTERVAL * 1000))

    def stop(self):
        self._running = False
        self.wait(3000)

    def _step(self):
        trend = self._emotion_trend

        if trend == 0:      # positive
            base = np.array([0.55, 0.30, 0.15])
        elif trend == 1:    # neutral
            base = np.array([0.22, 0.55, 0.23])
        else:               # negative
            base = np.array([0.15, 0.28, 0.57])

        noisy = base + np.random.randn(3) * 0.06
        noisy = np.clip(noisy, 0.02, 0.98)
        probs = noisy / noisy.sum()

        if self._ewma is None:
            self._ewma = probs.copy()
        else:
            self._ewma = self._alpha * probs + (1 - self._alpha) * self._ewma

        smoothed = self._ewma / self._ewma.sum()
        pred_idx = int(np.argmax(smoothed))
        confidence = float(smoothed[pred_idx])

        self.result_ready.emit({
            "probabilities": smoothed.tolist(),
            "predicted_state": CLASS_NAMES[pred_idx],
            "confidence": confidence,
            "ewma_negative": float(smoothed[2]),
        })

    # ── 真实模式预留接口 ──

    def predict_real(self, multimodal_features: dict):
        """预留：对接后端 EmotionInferenceModel.predict()。尚未接入。"""
        raise NotImplementedError("真实推理模式尚未接入，请使用 mock 模式")


def compute_quality(
    poor_signal: Optional[int],
    warmup_progress: float,
) -> tuple:
    """计算质量等级和原因。

    返回 (quality_level, quality_reasons)。
    quality_level: trusted | warning | rejected
    """
    reasons = []

    if poor_signal is None:
        return "rejected", ["尚未接收到信号"]

    if poor_signal >= 200:
        reasons.append("无信号 (poor_signal=200)")
        return "rejected", reasons

    if poor_signal >= MAX_POOR_SIGNAL:
        reasons.append(f"信号质量差 (poor_signal={poor_signal} >= {MAX_POOR_SIGNAL})")
        # 信号差但不是完全无信号
        if warmup_progress < 1.0:
            reasons.append("预热未完成")
        return "rejected" if poor_signal >= 100 else "warning", reasons

    if warmup_progress < 1.0:
        reasons.append(f"预热中 ({warmup_progress*100:.0f}%)")
        return "warning", reasons

    # good signal and warmup done
    if poor_signal == 0:
        return "trusted", []
    else:
        reasons.append(f"信号轻微波动 (poor_signal={poor_signal})")
        return "trusted", reasons
