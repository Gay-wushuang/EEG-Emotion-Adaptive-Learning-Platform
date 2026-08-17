"""DashboardState - 严格遵循 eeg_modular/AGENTS.md 第6节定义的统一状态接口。

UI只能通过此对象接收业务数据，不直接读取socket、模型或CSV。
新增字段必须保持向后兼容或同步更新mock数据、UI和测试。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional
from collections import deque

from PySide6.QtCore import QObject, Signal


# ── 标签映射契约（AGENTS.md 第2节）──
# 0 happy  → positive（积极）
# 1 normal → neutral（中性）
# 2 sad    → negative（负性）
CLASS_NAMES = ["positive", "neutral", "negative"]
CLASS_DISPLAY = {"positive": "积极", "neutral": "中性", "negative": "消极"}

# ── 质量门控常量 ──
MAX_POOR_SIGNAL = 50          # poor_signal < 50 为合格
WARMUP_SECONDS = 30.0         # 30秒预热
INFERENCE_INTERVAL = 2.0      # 2秒推理间隔
MOCK_UI_REFRESH_HZ = 10       # Mock模式UI刷新率
DEVICE_TARGET_SAMPLE_HZ = 512  # 设备目标采样率（仅信息展示，Mock不等于真实采样）


@dataclass
class EventMarker:
    """事件标记。"""
    timestamp: float
    label: str
    category: str  # user / system / intervention
    note: str = ""


@dataclass
class SessionRecord:
    """历史会话摘要记录。"""
    session_id: str
    user_id: str
    start_time: str
    duration_seconds: float
    positive_ratio: float
    neutral_ratio: float
    negative_ratio: float
    avg_attention: float
    avg_meditation: float
    signal_quality: float
    event_count: int
    notes: str = ""
    demo: bool = False  # 演示数据标记


class DashboardState(QObject):
    """全局统一状态对象（AGENTS.md 第6节）。

    所有字段严格遵循接口定义，UI只消费这些字段。
    后台服务调用 set_* 方法更新状态，然后调用 emit_update() 发射信号。
    """

    state_updated = Signal(object)  # 发射 self
    event_added = Signal(object)    # 发射 EventMarker

    # ── AGENTS.md 第6节正式字段 ──
    run_id: str
    mode: str                        # live | replay
    connector_status: str            # offline | connecting | online
    device_status: str               # offline | waiting_raw | online
    sample_rate_hz: Optional[float]  # None in mock; 512.0 when real device reports
    poor_signal: Optional[int]
    quality_level: str               # trusted | warning | rejected
    quality_reasons: List[str]
    warmup_progress: float           # 0.0 ~ 1.0
    prob_positive: Optional[float]
    prob_neutral: Optional[float]
    prob_negative: Optional[float]
    predicted_state: Optional[str]   # positive | neutral | negative | None
    confidence: Optional[float]
    stable_state: Optional[str]      # positive | neutral | negative | None
    attention: Optional[float]
    meditation: Optional[float]
    feedback_text: str
    session_seconds: float

    # ── 内部簿记字段（不暴露给UI业务逻辑，仅用于图表缓冲）──
    # 这些字段不属于正式接口，UI图表组件可直接使用原始缓冲进行绘制，
    # 但不得从中推导业务状态。
    _eeg_raw_buffer: deque
    _attention_history: deque
    _meditation_history: deque
    _prob_history: deque

    # ── 会话簿记 ──
    _session_active: bool
    _events: list
    _history_sessions: list
    _user_id: str
    _user_name: str

    # ── 基线簿记 ──
    _baseline_phase: str   # idle / collecting / done
    _baseline_elapsed: float
    _baseline_target: float

    # ── 持续状态簿记 ──
    _negative_sustain_seconds: float
    _intervention_triggered: bool
    _intervention_cooldown: bool

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)

        # 正式字段初始化
        self.run_id = uuid.uuid4().hex[:12]
        self.mode = "live"
        self.connector_status = "offline"
        self.device_status = "offline"
        self.sample_rate_hz = None  # Mock模式下为None，不假装512Hz
        self.poor_signal = None
        self.quality_level = "rejected"
        self.quality_reasons = ["尚未接收到信号"]
        self.warmup_progress = 0.0
        self.prob_positive = None
        self.prob_neutral = None
        self.prob_negative = None
        self.predicted_state = None
        self.confidence = None
        self.stable_state = None
        self.attention = None
        self.meditation = None
        self.feedback_text = "等待信号稳定后将生成学习建议。"
        self.session_seconds = 0.0

        # 内部簿记
        self._eeg_raw_buffer = deque(maxlen=512 * 5)
        self._raw_sample_count = 0
        self._attention_history = deque(maxlen=900)
        self._meditation_history = deque(maxlen=900)
        self._prob_history = deque(maxlen=450)
        self._session_active = False
        self._events = []
        self._history_sessions = []
        self._user_id = "demo_user"
        self._user_name = "演示用户"
        self._baseline_phase = "idle"
        self._baseline_elapsed = 0.0
        self._baseline_target = 75.0
        self._negative_sustain_seconds = 0.0
        self._intervention_triggered = False
        self._intervention_cooldown = False

        self._seed_history()

    def _seed_history(self):
        """预填充模拟历史会话，全部标记 demo=True。"""
        from datetime import datetime, timedelta
        base = datetime.now()
        labels = ["数学练习", "英语阅读", "编程任务", "物理复习", "专注冥想", "语文写作"]
        for i in range(12):
            dt = base - timedelta(days=i * 2, hours=i % 3)
            dur = 600 + (i * 137) % 1200
            pr = 0.20 + (i * 0.07) % 0.35
            nr = 0.15 + (i * 0.05) % 0.30
            nu = 1.0 - pr - nr
            self._history_sessions.append(SessionRecord(
                session_id=f"S{dt.strftime('%Y%m%d%H%M')}",
                user_id=self._user_id,
                start_time=dt.strftime("%Y-%m-%d %H:%M"),
                duration_seconds=dur,
                positive_ratio=round(pr, 3),
                neutral_ratio=round(nu, 3),
                negative_ratio=round(nr, 3),
                avg_attention=round(55 + (i * 3) % 25, 1),
                avg_meditation=round(48 + (i * 5) % 20, 1),
                signal_quality=round(0.72 + (i * 0.02) % 0.25, 2),
                event_count=i * 2 + 1,
                notes=labels[i % len(labels)],
                demo=True,  # 全部标记为演示数据
            ))

    @property
    def warmup_complete(self) -> bool:
        """预热是否完成（从 warmup_progress 派生）。"""
        return self.warmup_progress >= 1.0

    @property
    def inference_eligible(self) -> bool:
        """推理是否可用：预热完成且质量不是rejected。"""
        return self.warmup_complete and self.quality_level != "rejected"

    def add_event(self, label: str, category: str = "user", note: str = ""):
        ev = EventMarker(
            timestamp=time.time(),
            label=label,
            category=category,
            note=note,
        )
        self._events.append(ev)
        self.event_added.emit(ev)

    def emit_update(self):
        self.state_updated.emit(self)

    def reset_session(self):
        self._eeg_raw_buffer.clear()
        self._attention_history.clear()
        self._meditation_history.clear()
        self._prob_history.clear()
        self._events.clear()
        self.warmup_progress = 0.0
        self.session_seconds = 0.0
        self._session_active = False
        self._negative_sustain_seconds = 0.0
        self._intervention_triggered = False
        self._intervention_cooldown = False
        self.stable_state = None
        self.prob_positive = None
        self.prob_neutral = None
        self.prob_negative = None
        self.predicted_state = None
        self.confidence = None
        self.run_id = uuid.uuid4().hex[:12]
