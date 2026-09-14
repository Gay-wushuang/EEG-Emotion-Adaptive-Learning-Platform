"""Mock数据编排服务。

协调教学演示采集轨迹，直接将演示指标写入 DashboardState。

所有写入操作在主线程执行（通过信号槽跨线程传递），
确保 DashboardState 的线程安全。

自适应决策委托给共享的 AdaptiveFeedbackEngine（与 Live 共用同一套逻辑）。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtCore import QObject, QTimer, Signal

from services.dashboard_state import (
    DashboardState, WARMUP_SECONDS, MAX_POOR_SIGNAL,
    CLASS_NAMES, INFERENCE_INTERVAL, MOCK_UI_REFRESH_HZ,
    DEVICE_TARGET_SAMPLE_HZ,
    DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD,
    DIFFICULTY_DISPLAY, AdaptiveAction,
)
from services.eeg_acquisition import EEGAcquisitionWorker, AcquisitionConfig
from services.adaptive_feedback_engine import (
    AdaptiveFeedbackEngine,
    AdaptiveDecision,
    apply_adaptive_decision,
)


# 阈值常量保留供外部引用（引擎内部也使用相同值）
NEGATIVE_THRESHOLD = 0.60
SUSTAIN_SECONDS = 20.0
COOLDOWN_SECONDS = 90.0


class MockDataService(QObject):
    """编排采集 + 推理，统一更新 DashboardState。"""

    def __init__(self, state: DashboardState, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.state = state
        self.sessions_dir = Path(__file__).resolve().parents[2] / "data" / "sessions"
        self.state.mode = "mock"
        # Persisted synthetic sessions are indexed only in Mock mode.  They
        # remain physically separated from formal Live history.
        self.state.configure_session_store(self.sessions_dir, include_demo=True)

        # 后台线程
        self.acq_config = AcquisitionConfig(mode="mock")
        self.acq_worker = EEGAcquisitionWorker(self.acq_config)

        # 共享自适应反馈引擎（与 Live 共用同一套决策逻辑）
        self.engine = AdaptiveFeedbackEngine(
            negative_threshold=NEGATIVE_THRESHOLD,
            sustain_seconds=SUSTAIN_SECONDS,
            cooldown_seconds=COOLDOWN_SECONDS,
        )

        # 连接信号
        self.acq_worker.data_ready.connect(self._on_acq_data)
        self.acq_worker.status_changed.connect(self._on_acq_status)

        # 定时器：更新会话时间和预热进度
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(200)  # 5 Hz
        self._tick_timer.timeout.connect(self._on_tick)

        self._session_running = False
        self._warmup_running = False
        self._last_tick_time = 0.0
        self._demo_started_at = time.monotonic()

    # ── 生命周期 ──

    def start_streaming(self):
        """启动独立教学演示；不加载或调用 Production 模型。"""
        self.state.mode = "mock"
        self.state.set_model_ready()
        self.state.connector_status = "online"
        self.state.device_status = "online"
        self.state.warmup_progress = 1.0
        self.state.quality_level = "trusted"
        self.state.quality_reasons = ["教学演示数据有效"]
        self.state.pipeline_state = "ready"
        if not self.acq_worker.isRunning():
            self.acq_worker.start()
        self._warmup_running = False
        self._demo_started_at = time.monotonic()
        self._tick_timer.start()

    def stop_streaming(self):
        """停止数据流。"""
        if self._session_running:
            self.end_session(status="interrupted")
        self._warmup_running = False
        self._tick_timer.stop()
        self.acq_worker.stop()

    def start_session(self):
        self.state.session_seconds = 0.0
        metadata_path = self.state.begin_session(source="mock", demo=True)
        self._session_running = True
        if not self._tick_timer.isActive():
            self._tick_timer.start()
        return metadata_path

    def pause_session(self):
        self.state.set_session_paused(True)
        self._session_running = False

    def resume_session(self):
        self.state.set_session_paused(False)
        self._session_running = True

    def end_session(self, status: str = "completed"):
        self._session_running = False
        self._warmup_running = False
        metadata_path = self.state.finalize_session(status=status)
        if metadata_path is None and self.state.last_session_save_error:
            self.state.feedback_text = (
                "会话记录保存失败，请检查 data/sessions 文件夹写入权限。"
            )
        return metadata_path

    # ── 信号回调（主线程执行）──

    def _on_acq_data(self, snap):
        """采集线程推送新数据。

        设备离线时不更新 poor_signal / attention / meditation，保持 None，
        避免出现"设备离线"与"Attention=72"并存的矛盾状态，
        也防止 _apply_adaptive_action 用残留的 attention 误判"Attention 偏低"。
        """
        s = self.state

        s.poor_signal = snap.poor_signal
        s.attention = float(snap.attention)
        s.meditation = float(snap.meditation)
        s._eeg_raw_buffer.append(snap.raw)
        s._attention_history.append(snap.attention)
        s._meditation_history.append(snap.meditation)
        s.quality_level = "trusted"
        s.quality_reasons = ["教学演示数据有效"]
        s.pipeline_state = "ready"

    def _on_acq_status(self, status: dict):
        s = self.state
        s.connector_status = "online"
        s.device_status = "online"
        s.mode = "mock"
        s.warmup_progress = 1.0
        s.quality_level = "trusted"
        s.quality_reasons = ["教学演示数据有效"]
        s.pipeline_state = "ready"

    def _on_inference(self, result: dict):
        """推理线程推送推理结果。"""
        s = self.state
        probs = result["probabilities"]

        # 只有质量合格时才写入概率
        if s.quality_level != "rejected":
            s.prob_positive = probs[0]
            s.prob_neutral = probs[1]
            s.prob_negative = probs[2]
            s.predicted_state = result["predicted_state"]
            s.confidence = result["confidence"]
        else:
            # 信号不合格时保留后台原始结果，但不得恢复旧的当前解释。
            self._invalidate_signal_analysis()
            return
        if s.pipeline_state != "error":
            s.pipeline_state = (
                "ready" if s.inference_eligible else "rejected"
            )

        # 概率历史（用于图表绘制）
        s._prob_history.append((time.time(), probs[0], probs[1], probs[2]))

        # 更新持续状态（传入原始概率供 Temporal Policy 使用）
        raw_probs = np.array(result.get("raw_probabilities", probs))
        self._update_stable_state(raw_probs)

        # 更新反馈文本
        s.feedback_text = self._generate_feedback()

    def _invalidate_signal_analysis(self):
        s = self.state
        s.clear_interpretation()
        s.feedback_text = "当前信号不可解释，请调整佩戴并等待信号恢复。"
        s.adaptive_feedback_text = ""
        s.adaptive_action = AdaptiveAction.NONE
        s.adaptive_action_reason = ""
        s.adaptive_action_time = None
        s._intervention_triggered = False
        s._intervention_cooldown = False
        s._negative_sustain_seconds = 0.0
        self.engine.reset()

    def _on_tick(self):
        """5Hz定时更新：预热进度、会话时间。"""
        now = time.time()
        if self._last_tick_time == 0.0:
            self._last_tick_time = now
        dt = now - self._last_tick_time
        self._last_tick_time = now

        s = self.state

        # Demo is immediately analysis-ready and independent of production warmup.
        s.warmup_progress = 1.0
        s.quality_level = "trusted"
        s.quality_reasons = ["教学演示数据有效"]
        s.pipeline_state = "ready"

        # 会话时间
        if self._session_running:
            s.session_seconds += dt

        demo_t = time.monotonic() - self._demo_started_at
        self._on_inference(self._demo_result(demo_t))

        if self._session_running:
            s.capture_session_snapshot()

        s.emit_update()

    @staticmethod
    def _demo_result(t: float) -> dict:
        """Deterministic probabilities matching the shortened teaching stages."""
        phase = t % 80.0
        if phase < 10.0:
            probs = np.array([0.28, 0.57, 0.15])
        elif phase < 25.0:
            progress = (phase - 10.0) / 15.0
            probs = np.array([0.48 + 0.24 * progress, 0.40 - 0.18 * progress, 0.12 - 0.06 * progress])
        elif phase < 35.0:
            progress = (phase - 25.0) / 10.0
            probs = np.array([0.60 - 0.40 * progress, 0.28 - 0.04 * progress, 0.12 + 0.44 * progress])
        elif phase < 60.0:
            probs = np.array([0.12, 0.18, 0.70])
        else:
            progress = min(1.0, (phase - 60.0) / 15.0)
            probs = np.array([0.18 + 0.20 * progress, 0.24 + 0.26 * progress, 0.58 - 0.46 * progress])
        probs = probs / probs.sum()
        predicted = CLASS_NAMES[int(np.argmax(probs))]
        return {"probabilities": probs.tolist(), "raw_probabilities": probs.tolist(),
                "predicted_state": predicted, "confidence": float(np.max(probs))}

    # ── 持续状态判定（委托给共享 AdaptiveFeedbackEngine）──

    def _update_stable_state(self, raw_probabilities: np.ndarray):
        """更新稳定状态和自适应决策。

        稳定状态（positive/neutral/negative）仍由 Mock 本地判定；
        自适应决策（EWMA + 持续负性 + 冷却 + action 选择）委托给
        共享引擎内部的 EWMASustainedNegativeDecision。

        Args:
            raw_probabilities: 原始三分类概率数组（未经预 EWMA）
        """
        s = self.state
        now = time.time()

        if not s.inference_eligible:
            s.stable_state = None
            s._intervention_triggered = False
            s._intervention_cooldown = False
            s._negative_sustain_seconds = 0.0
            self.engine.reset()
            return

        # 判定当前主导状态（仍由 Mock 本地做，与 UI 直接挂钩）
        if s.prob_positive is not None and s.prob_positive >= max(
            s.prob_neutral or 0, s.prob_negative or 0
        ):
            s.stable_state = "positive"
        elif s.prob_neutral is not None and s.prob_neutral >= (s.prob_negative or 0):
            s.stable_state = "neutral"
        else:
            s.stable_state = "negative"

        # ── 委托共享引擎做自适应决策（单一 EWMA 来源）──
        # Mock 始终 accepted=True（无 Production Baseline 拒识机制）
        decision = self.engine.decide(
            probabilities=raw_probabilities,
            accepted=True,
            quality_level=s.quality_level,
            attention=s.attention,
            task_difficulty=s.task_difficulty,
            timestamp=now,
            warmup_complete=s.warmup_complete,
        )

        # 同步引擎状态到 DashboardState（供 _generate_feedback 和 UI 使用）
        s._negative_sustain_seconds = decision.above_seconds
        s._intervention_cooldown = decision.in_cooldown

        if decision.should_emit_event:
            s._intervention_triggered = True
            self._apply_adaptive_action(decision)
        else:
            s._intervention_triggered = False

    # ── 自适应决策应用（决策层 → 学习场景正式接口）──

    def _apply_adaptive_action(self, decision: Optional[AdaptiveDecision] = None):
        """将 AdaptiveDecision 写入 DashboardState（委托给共享契约）。

        两种调用方式：
        1. 由 _update_stable_state 传入引擎决策（正式路径）
        2. 直接调用时从当前 state 构造决策（向后兼容测试）

        实际写入逻辑由 apply_adaptive_decision() 统一处理，
        确保 Mock 与 Live 的 DashboardState 写入契约完全一致。
        """
        s = self.state

        # 如果没有传入决策，从当前 state 构造（向后兼容测试）
        if decision is None:
            decision = self.engine.select_action(
                s.task_difficulty, s.attention
            )

        # 委托给共享写入契约
        apply_adaptive_decision(s, decision)

    # ── 反馈文本生成 ──

    def _generate_feedback(self) -> str:
        s = self.state
        if not s.inference_eligible:
            return "当前信号质量不足，暂不进行学习状态解释。"

        if s.stable_state == "positive":
            if s.attention is not None and s.attention > 70:
                return "学习状态良好，注意力集中，建议保持当前节奏。"
            return "情绪积极，可适当提升任务难度以保持投入。"
        elif s.stable_state == "neutral":
            if s.attention is not None and s.attention < 45:
                return "注意力偏低，建议切换任务或短暂休息后恢复。"
            return "状态平稳，建议维持当前学习计划。"
        else:  # negative
            if s._intervention_triggered:
                return "检测到持续消极状态，建议立即休息5分钟或切换至轻松任务。"
            if s._negative_sustain_seconds > 10:
                return "消极状态持续中，建议调整学习内容或进行放松练习。"
            return "情绪略有波动，建议关注任务难度是否偏高。"
