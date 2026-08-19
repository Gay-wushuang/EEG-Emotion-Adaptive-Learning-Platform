"""Mock数据编排服务。

协调 EEGAcquisitionWorker 和 InferenceWorker 两个后台线程，
将数据统一写入 DashboardState 并发射 ``state_updated`` 信号。

所有写入操作在主线程执行（通过信号槽跨线程传递），
确保 DashboardState 的线程安全。
"""

from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import QObject, QTimer, Signal

from services.dashboard_state import (
    DashboardState, WARMUP_SECONDS, MAX_POOR_SIGNAL,
    CLASS_NAMES, INFERENCE_INTERVAL, MOCK_UI_REFRESH_HZ,
    DEVICE_TARGET_SAMPLE_HZ,
    DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD,
    DIFFICULTY_DISPLAY, AdaptiveAction,
)
from services.eeg_acquisition import EEGAcquisitionWorker, AcquisitionConfig
from services.inference_service import InferenceWorker, compute_quality


NEGATIVE_THRESHOLD = 0.60
SUSTAIN_SECONDS = 20.0
COOLDOWN_SECONDS = 90.0


class MockDataService(QObject):
    """编排采集 + 推理，统一更新 DashboardState。"""

    def __init__(self, state: DashboardState, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.state = state

        # 后台线程
        self.acq_config = AcquisitionConfig(mode="mock")
        self.acq_worker = EEGAcquisitionWorker(self.acq_config)
        self.inf_worker = InferenceWorker()

        # EWMA 持续状态跟踪
        self._above_since: Optional[float] = None
        self._last_intervention: Optional[float] = None

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

    # ── 生命周期 ──

    def start_streaming(self):
        """启动数据流（采集+推理）。"""
        if not self.acq_worker.isRunning():
            self.acq_worker.start()
        if not self.inf_worker.isRunning():
            self.inf_worker.start()
        self._warmup_running = True
        self._tick_timer.start()

    def stop_streaming(self):
        """停止数据流。"""
        self._warmup_running = False
        self._tick_timer.stop()
        self.acq_worker.stop()
        self.inf_worker.stop()

    def start_session(self):
        self.state._session_active = True
        self.state.session_seconds = 0.0
        self._session_running = True
        if not self._tick_timer.isActive():
            self._tick_timer.start()

    def pause_session(self):
        self.state._session_active = False
        self._session_running = False

    def resume_session(self):
        self.state._session_active = True
        self._session_running = True

    def end_session(self):
        self.state._session_active = False
        self._session_running = False
        self._warmup_running = False

    # ── 信号回调（主线程执行）──

    def _on_acq_data(self, snap):
        """采集线程推送新数据。

        设备离线时不更新 poor_signal / attention / meditation，保持 None，
        避免出现"设备离线"与"Attention=72"并存的矛盾状态，
        也防止 _apply_adaptive_action 用残留的 attention 误判"Attention 偏低"。
        """
        s = self.state

        # 仅当设备真正在线时才接受 poor_signal / attention / meditation
        device_online = (
            s.device_status == "online" and s.connector_status == "online"
        )
        if device_online:
            s.poor_signal = snap.poor_signal
            s.attention = float(snap.attention)
            s.meditation = float(snap.meditation)
            s._eeg_raw_buffer.append(snap.raw)
            s._attention_history.append(snap.attention)
            s._meditation_history.append(snap.meditation)
        else:
            s.poor_signal = None
            s.attention = None
            s.meditation = None

        # 质量等级计算
        if device_online:
            quality_level, quality_reasons = compute_quality(
                s.poor_signal, s.warmup_progress
            )
        else:
            quality_level, quality_reasons = "rejected", ["设备未连接"]
        s.quality_level = quality_level
        s.quality_reasons = quality_reasons

        # 将模拟情绪趋势传给推理线程
        if hasattr(self.acq_worker, '_sim_state'):
            self.inf_worker.set_emotion_trend(self.acq_worker._sim_state.emotion_trend)

    def _on_acq_status(self, status: dict):
        s = self.state
        s.connector_status = status.get("connector_status", "offline")
        s.device_status = status.get("device_status", "offline")
        s.mode = status.get("mode", "live")

        # 设备离线时：清空 poor_signal / attention / meditation，
        # 强制标记质量为 rejected
        if s.device_status != "online" or s.connector_status != "online":
            s.poor_signal = None
            s.attention = None
            s.meditation = None
            s.quality_level = "rejected"
            s.quality_reasons = ["设备未连接"]

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
            # 信号不合格时保存概率日志但UI不展示
            s.prob_positive = None
            s.prob_neutral = None
            s.prob_negative = None
            s.predicted_state = None
            s.confidence = None

        # 概率历史（用于图表绘制）
        s._prob_history.append((time.time(), probs[0], probs[1], probs[2]))

        # 更新持续状态
        self._update_stable_state(result["ewma_negative"])

        # 更新反馈文本
        s.feedback_text = self._generate_feedback()

    def _on_tick(self):
        """5Hz定时更新：预热进度、会话时间。"""
        now = time.time()
        if self._last_tick_time == 0.0:
            self._last_tick_time = now
        dt = now - self._last_tick_time
        self._last_tick_time = now

        s = self.state

        # 预热进度
        if self._warmup_running and not s.warmup_complete:
            elapsed = s.warmup_progress * WARMUP_SECONDS + dt
            s.warmup_progress = min(1.0, elapsed / WARMUP_SECONDS)

        # 会话时间
        if self._session_running:
            s.session_seconds += dt

        # 质量等级重算（仅在设备在线时执行，离线时保持"设备未连接"原因）
        s = self.state
        device_online = (
            s.device_status == "online" and s.connector_status == "online"
        )
        if device_online:
            quality_level, quality_reasons = compute_quality(
                s.poor_signal, s.warmup_progress
            )
            s.quality_level = quality_level
            s.quality_reasons = quality_reasons

        s.emit_update()

    # ── 持续状态判定 ──

    def _update_stable_state(self, negative_ewma: float):
        s = self.state
        now = time.time()

        if not s.inference_eligible:
            self._above_since = None
            s._negative_sustain_seconds = 0.0
            s._intervention_triggered = False
            s.stable_state = None
            return

        # 判定当前主导状态
        if s.prob_positive is not None and s.prob_positive >= max(
            s.prob_neutral or 0, s.prob_negative or 0
        ):
            s.stable_state = "positive"
        elif s.prob_neutral is not None and s.prob_neutral >= (s.prob_negative or 0):
            s.stable_state = "neutral"
        else:
            s.stable_state = "negative"

        # 持续消极判定
        if negative_ewma >= NEGATIVE_THRESHOLD:
            if self._above_since is None:
                self._above_since = now
            s._negative_sustain_seconds = now - self._above_since

            cooled = (
                self._last_intervention is None
                or now - self._last_intervention >= COOLDOWN_SECONDS
            )
            if s._negative_sustain_seconds >= SUSTAIN_SECONDS and cooled:
                s._intervention_triggered = True
                s._intervention_cooldown = False
                self._last_intervention = now
                self._above_since = None
                # 触发正式自适应动作：降低难度 / 建议休息
                # （一次决策只产生一次 event，见 _apply_adaptive_action）
                self._apply_adaptive_action()
            elif not cooled:
                s._intervention_cooldown = True
        else:
            self._above_since = None
            s._negative_sustain_seconds = 0.0
            s._intervention_triggered = False

    # ── 自适应决策（决策层 → 学习场景正式接口）──

    def _apply_adaptive_action(self):
        """根据当前 task_difficulty 执行最小自适应动作。

        前置条件：调用方已确保 inference_eligible 且持续消极触发条件已满足。
        一次决策只产生一次 intervention event，避免 UI 刷新重复写入。

        - hard  → medium，reduce_difficulty
        - medium → easy，reduce_difficulty
        - easy  → suggest_break（不继续降低，不结束会话）
        """
        s = self.state
        now = time.time()

        # 安全护栏：信号 rejected 不得触发任何自适应动作
        if s.quality_level == "rejected":
            return

        att_low = s.attention is not None and s.attention < 50.0
        reason_neg = "持续负性状态趋势"
        reason_att = "且 Attention 偏低" if att_low else ""
        reason_full = f"{reason_neg}{reason_att}"

        prev_diff = s.task_difficulty
        if prev_diff == DIFFICULTY_HARD:
            s.task_difficulty = DIFFICULTY_MEDIUM
            s.adaptive_action = AdaptiveAction.REDUCE_DIFFICULTY
            s.adaptive_action_reason = reason_full
            prev_disp = DIFFICULTY_DISPLAY[DIFFICULTY_HARD]
            new_disp = DIFFICULTY_DISPLAY[DIFFICULTY_MEDIUM]
            s.adaptive_feedback_text = (
                "当前任务已从“" + prev_disp + "”调整为“" + new_disp + "”"
            )
            label = "自适应降低任务难度"
            note = (
                prev_disp + " -> " + new_disp
                + "；原因：" + reason_full
            )
        elif prev_diff == DIFFICULTY_MEDIUM:
            s.task_difficulty = DIFFICULTY_EASY
            s.adaptive_action = AdaptiveAction.REDUCE_DIFFICULTY
            s.adaptive_action_reason = reason_full
            prev_disp = DIFFICULTY_DISPLAY[DIFFICULTY_MEDIUM]
            new_disp = DIFFICULTY_DISPLAY[DIFFICULTY_EASY]
            s.adaptive_feedback_text = (
                "当前任务已从“" + prev_disp + "”调整为“" + new_disp + "”"
            )
            label = "自适应降低任务难度"
            note = (
                prev_disp + " -> " + new_disp
                + "；原因：" + reason_full
            )
        else:  # easy：不继续降低，转为建议休息
            s.adaptive_action = AdaptiveAction.SUGGEST_BREAK
            s.adaptive_action_reason = "学习状态持续不佳，已为最低难度"
            s.adaptive_feedback_text = "当前任务已为最低难度，建议短暂休息后继续"
            label = "建议短暂休息"
            note = "当前已为最低任务难度，进入休息建议"

        s.adaptive_action_time = now
        # 一次决策只产生一次 event
        s.add_event(label, "intervention", note)

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
