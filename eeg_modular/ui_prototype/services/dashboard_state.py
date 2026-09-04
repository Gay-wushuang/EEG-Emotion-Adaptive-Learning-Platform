"""DashboardState - 严格遵循 eeg_modular/AGENTS.md 第6节定义的统一状态接口。

UI只能通过此对象接收业务数据，不直接读取socket、模型或CSV。
新增字段必须保持向后兼容或同步更新mock数据、UI和测试。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional
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

# ── 学习任务难度枚举（业务层只允许这三个值）──
DIFFICULTY_EASY = "easy"
DIFFICULTY_MEDIUM = "medium"
DIFFICULTY_HARD = "hard"
DIFFICULTY_LEVELS = (DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD)
DIFFICULTY_DISPLAY = {
    DIFFICULTY_EASY: "简单",
    DIFFICULTY_MEDIUM: "中等",
    DIFFICULTY_HARD: "困难",
}

# ── 自适应动作枚举（决策层 → 学习场景正式契约）──
# maintain: 维持当前任务和难度，只更新反馈
# reduce_difficulty: hard→medium 或 medium→easy
# suggest_break: 已为 easy 时的替代动作，给出休息建议，不结束会话、不删除任务
class AdaptiveAction:
    NONE = "none"                       # 默认：无动作
    MAINTAIN = "maintain"
    REDUCE_DIFFICULTY = "reduce_difficulty"
    SUGGEST_BREAK = "suggest_break"
    ALL = (NONE, MAINTAIN, REDUCE_DIFFICULTY, SUGGEST_BREAK)

ADAPTIVE_ACTION_DISPLAY = {
    AdaptiveAction.NONE: "无",
    AdaptiveAction.MAINTAIN: "维持当前任务",
    AdaptiveAction.REDUCE_DIFFICULTY: "降低任务难度",
    AdaptiveAction.SUGGEST_BREAK: "建议休息",
}

SESSION_SOURCES = ("live", "mock", "replay")


def _normalise_source(value: Optional[str], *, demo: bool = False) -> str:
    source = str(value or "").strip().lower()
    if source in SESSION_SOURCES:
        return source
    return "mock" if demo else "live"


def _iso_time(timestamp: Optional[float] = None) -> str:
    value = time.time() if timestamp is None else float(timestamp)
    return datetime.fromtimestamp(value, timezone.utc).astimezone().isoformat(timespec="seconds")


def _timestamp(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return datetime.fromisoformat(str(value)).timestamp()
        except (TypeError, ValueError):
            return time.time()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class EventMarker:
    """事件标记，同时保留旧 UI 字段和正式数据契约字段。"""
    timestamp: float = field(default_factory=time.time)
    label: str = ""
    category: str = "user"  # legacy: user / system / intervention
    note: str = ""
    source: str = "live"
    type: str = "marker"
    session_id: str = ""
    task_id: str = ""
    time: str = ""
    content: str = ""
    event_id: str = field(default_factory=lambda: f"E{uuid.uuid4().hex[:12]}")

    def __post_init__(self) -> None:
        self.timestamp = _timestamp(self.timestamp or self.time)
        self.source = _normalise_source(self.source)
        self.type = str(self.type or self.category or "marker")
        self.time = str(self.time or _iso_time(self.timestamp))
        self.content = str(self.content or self.label)
        self.label = str(self.label or self.content)
        self.category = str(self.category or "user")
        self.note = str(self.note or "")

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "source": self.source,
            "type": self.type,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "time": self.time,
            "content": self.content,
            "note": self.note,
            # Legacy fields make old report/export code able to read v2 records.
            "timestamp": self.timestamp,
            "label": self.label,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "EventMarker":
        return cls(
            timestamp=_timestamp(payload.get("timestamp") or payload.get("time")),
            label=str(payload.get("label") or payload.get("content") or ""),
            category=str(payload.get("category") or "system"),
            note=str(payload.get("note") or ""),
            source=str(payload.get("source") or "live"),
            type=str(payload.get("type") or payload.get("category") or "marker"),
            session_id=str(payload.get("session_id") or ""),
            task_id=str(payload.get("task_id") or ""),
            time=str(payload.get("time") or ""),
            content=str(payload.get("content") or payload.get("label") or ""),
            event_id=str(payload.get("event_id") or f"E{uuid.uuid4().hex[:12]}"),
        )


@dataclass
class TaskRecord:
    """One task segment inside a session."""
    task_id: str = field(default_factory=lambda: f"T{uuid.uuid4().hex[:12]}")
    session_id: str = ""
    name: str = "自由学习"
    difficulty: str = DIFFICULTY_MEDIUM
    start_time: str = ""
    end_time: str = ""
    status: str = "running"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "name": self.name,
            "difficulty": self.difficulty,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TaskRecord":
        return cls(
            task_id=str(payload.get("task_id") or f"T{uuid.uuid4().hex[:12]}"),
            session_id=str(payload.get("session_id") or ""),
            name=str(payload.get("name") or payload.get("task_type") or "自由学习"),
            difficulty=str(payload.get("difficulty") or DIFFICULTY_MEDIUM),
            start_time=str(payload.get("start_time") or ""),
            end_time=str(payload.get("end_time") or ""),
            status=str(payload.get("status") or "completed"),
            notes=str(payload.get("notes") or ""),
        )


@dataclass
class SessionRecord:
    """历史会话摘要，兼容旧字段并携带 v2 审计信息。"""
    session_id: str = ""
    user_id: str = ""
    start_time: str = ""
    duration_seconds: float = 0.0
    positive_ratio: float = 0.0
    neutral_ratio: float = 0.0
    negative_ratio: float = 0.0
    avg_attention: float = 0.0
    avg_meditation: float = 0.0
    signal_quality: float = 0.0
    event_count: int = 0
    notes: str = ""
    demo: bool = False  # 演示数据标记
    source: str = "live"
    user_name: str = ""
    end_time: str = ""
    status: str = "completed"
    tasks: list[TaskRecord] = field(default_factory=list)
    events: list[EventMarker] = field(default_factory=list)
    quality_summary: dict = field(default_factory=dict)
    probability_summary: dict = field(default_factory=dict)
    data_files: dict = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        self.source = _normalise_source(self.source, demo=self.demo)
        self.tasks = [
            item if isinstance(item, TaskRecord) else TaskRecord.from_dict(item)
            for item in self.tasks if isinstance(item, (TaskRecord, dict))
        ]
        self.events = [
            item if isinstance(item, EventMarker) else EventMarker.from_dict(item)
            for item in self.events if isinstance(item, (EventMarker, dict))
        ]
        if not self.event_count:
            self.event_count = len(self.events)

    @property
    def primary_task(self) -> str:
        if self.tasks:
            return self.tasks[0].name
        return self.notes

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "source": self.source,
            "demo": self.demo,
            "status": self.status,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "tasks": [item.to_dict() for item in self.tasks],
            "events": [item.to_dict() for item in self.events],
            "quality_summary": dict(self.quality_summary),
            "probability_summary": dict(self.probability_summary),
            "data_files": dict(self.data_files),
            # Legacy summary fields remain first-class for old UI/report code.
            "positive_ratio": self.positive_ratio,
            "neutral_ratio": self.neutral_ratio,
            "negative_ratio": self.negative_ratio,
            "avg_attention": self.avg_attention,
            "avg_meditation": self.avg_meditation,
            "signal_quality": self.signal_quality,
            "event_count": len(self.events) if self.events else self.event_count,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SessionRecord":
        probabilities = payload.get("probability_summary") or {}
        quality = payload.get("quality_summary") or {}
        events = payload.get("events") or []
        return cls(
            session_id=str(payload.get("session_id") or ""),
            user_id=str(payload.get("user_id") or ""),
            start_time=str(payload.get("start_time") or ""),
            duration_seconds=_float(payload.get("duration_seconds")),
            positive_ratio=_float(
                payload.get("positive_ratio", probabilities.get("mean_positive", 0.0))
            ),
            neutral_ratio=_float(
                payload.get("neutral_ratio", probabilities.get("mean_neutral", 0.0))
            ),
            negative_ratio=_float(
                payload.get("negative_ratio", probabilities.get("mean_negative", 0.0))
            ),
            avg_attention=_float(payload.get("avg_attention")),
            avg_meditation=_float(payload.get("avg_meditation")),
            signal_quality=_float(
                payload.get("signal_quality", quality.get("usable_ratio", 0.0))
            ),
            event_count=int(payload.get("event_count") or len(events)),
            notes=str(payload.get("notes") or ""),
            demo=bool(payload.get("demo", False)),
            source=str(payload.get("source") or ""),
            user_name=str(payload.get("user_name") or ""),
            end_time=str(payload.get("end_time") or ""),
            status=str(payload.get("status") or "completed"),
            tasks=payload.get("tasks") or [],
            events=events,
            quality_summary=dict(quality),
            probability_summary=dict(probabilities),
            data_files=dict(payload.get("data_files") or {}),
            schema_version=int(payload.get("schema_version") or 1),
        )


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
    pipeline_state: str               # starting | waiting_data | warming_up | ready | rejected | error
    model_error_user: str             # 用户可见的安全提示
    model_error_detail: str           # 仅供设置/诊断页读取的技术详情

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

    # ── 自适应学习场景正式字段（AGENTS.md 第6节：新增字段须向后兼容）──
    # 这五个字段把"决策层 → 学习场景"的接口暴露给 UI，
    # 避免后台服务直接操作 QWidget。
    task_type: str                          # 当前任务类型（如"数学练习"）
    task_difficulty: str                   # easy | medium | hard
    task_running: bool                     # 当前任务是否进行中
    adaptive_action: str                   # AdaptiveAction.* 之一
    adaptive_action_reason: str            # 触发原因（中文展示用）
    adaptive_action_time: Optional[float]  # 最近一次动作时间戳
    adaptive_feedback_text: str            # AI 反馈文本（中文）

    # ── 持续状态簿记 ──
    _negative_sustain_seconds: float
    _intervention_triggered: bool
    _intervention_cooldown: bool

    def __init__(
        self,
        parent: Optional[QObject] = None,
        sessions_dir: Optional[Path | str] = None,
        seed_demo_history: bool = True,
    ):
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
        self.pipeline_state = "starting"
        self.model_error_user = ""
        self.model_error_detail = ""

        # 自适应学习场景字段（默认安全值）
        self.task_type = "自由学习"
        self.task_difficulty = DIFFICULTY_MEDIUM
        self.task_running = False
        self.adaptive_action = AdaptiveAction.NONE
        self.adaptive_action_reason = ""
        self.adaptive_action_time = None
        self.adaptive_feedback_text = ""

        # 内部簿记
        self._eeg_raw_buffer = deque(maxlen=512 * 5)
        self._raw_sample_count = 0
        self._attention_history = deque(maxlen=900)
        self._meditation_history = deque(maxlen=900)
        self._prob_history = deque(maxlen=450)
        self._quality_history = deque(maxlen=18000)
        self._session_active = False
        self._events = []
        self._history_sessions = []
        self._tasks: list[TaskRecord] = []
        self._current_task_id = ""
        self._session_started_at: Optional[float] = None
        self._session_source = "live"
        self._session_demo = False
        self._active_record: Optional[SessionRecord] = None
        self._sessions_dir = Path(sessions_dir) if sessions_dir else (
            Path(__file__).resolve().parents[2] / "data" / "sessions"
        )
        self._session_store = None
        self.last_session_save_error = ""
        self.history_load_errors: list[str] = []
        self._user_id = "demo_user"
        self._user_name = "演示用户"
        self._baseline_phase = "idle"
        self._baseline_elapsed = 0.0
        self._baseline_target = 75.0
        self._negative_sustain_seconds = 0.0
        self._intervention_triggered = False
        self._intervention_cooldown = False

        # Compatibility for callers that instantiate DashboardState in
        # isolation.  Both MockDataService and LiveDataService immediately
        # replace this in-memory sample list with their isolated persisted
        # history, so formal application history is never demo-polluted.
        if seed_demo_history:
            self._seed_history()

    def _seed_history(self):
        """Populate legacy in-memory samples; services replace them on init."""
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
                source="mock",
            ))

    @property
    def warmup_complete(self) -> bool:
        """预热是否完成（从 warmup_progress 派生）。"""
        return self.warmup_progress >= 1.0

    @property
    def inference_eligible(self) -> bool:
        """推理是否可用：预热完成且质量不是rejected。"""
        return self.warmup_complete and self.quality_level != "rejected"

    @property
    def history_sessions(self) -> tuple[SessionRecord, ...]:
        """Read-only public view used by pages and report builders."""
        return tuple(self._history_sessions)

    @property
    def current_task_id(self) -> str:
        return self._current_task_id

    def configure_session_store(
        self,
        sessions_dir: Optional[Path | str] = None,
        *,
        include_demo: bool = False,
    ):
        """Bind persistence and replace legacy in-memory demo history."""
        from services.session_store import SessionStore

        if sessions_dir is not None:
            self._sessions_dir = Path(sessions_dir)
        self._session_store = SessionStore(self._sessions_dir)
        self.reload_history(include_demo=include_demo)
        return self._session_store

    def reload_history(self, *, include_demo: bool = False) -> list[SessionRecord]:
        """Re-index saved sessions; formal history excludes demos by default."""
        if self._session_store is None:
            self.configure_session_store(self._sessions_dir, include_demo=include_demo)
            return list(self._history_sessions)
        records = []
        for payload in self._session_store.load(include_demo=include_demo):
            try:
                records.append(SessionRecord.from_dict(payload))
            except (TypeError, ValueError, AttributeError) as exc:
                self._session_store.last_errors.append(
                    f"{payload.get('session_id', '<unknown>')}: {exc}"
                )
        self._history_sessions = records
        self.history_load_errors = list(self._session_store.last_errors)
        return list(records)

    def begin_session(
        self,
        *,
        source: Optional[str] = None,
        demo: Optional[bool] = None,
        data_files: Optional[dict] = None,
    ) -> Optional[str]:
        """Start and immediately persist an auditable session envelope."""
        if self._session_active and self._active_record is not None:
            self.finalize_session(status="interrupted")
        if self._session_store is None:
            self.configure_session_store(self._sessions_dir, include_demo=False)

        self._session_source = _normalise_source(source or self.mode)
        self._session_demo = (
            self._session_source == "mock" if demo is None else bool(demo)
        )
        self._session_started_at = time.time()
        self._session_active = True
        self.task_running = False
        self._events.clear()
        self._tasks.clear()
        self._quality_history.clear()
        self._current_task_id = ""
        self.last_session_save_error = ""

        self._active_record = SessionRecord(
            session_id=self.run_id,
            user_id=self._user_id,
            user_name=self._user_name,
            start_time=_iso_time(self._session_started_at),
            source=self._session_source,
            demo=self._session_demo,
            status="running",
            notes=self.task_type,
            data_files=dict(data_files or {}),
        )
        path = self._persist_active_record()
        self.add_event(
            "会话开始", "system", event_type="session_start",
            source=self._session_source,
        )
        return path

    def begin_task(
        self,
        task_name: Optional[str] = None,
        difficulty: Optional[str] = None,
        notes: str = "",
    ) -> Optional[str]:
        """Begin a task segment and return its task_id."""
        if not self._session_active:
            return None
        if self._current_task_id:
            self.end_task(note="切换到新任务")
        task = self._begin_task_record(
            task_name or self.task_type,
            difficulty or self.task_difficulty,
            notes,
        )
        self.add_event(
            f"开始任务: {task.name}", "system", notes,
            event_type="task_start", task_id=task.task_id,
            _manage_task=False,
        )
        return task.task_id

    def end_task(self, note: str = "") -> Optional[TaskRecord]:
        """End the current task segment and retain it in the session record."""
        task = self._task_by_id(self._current_task_id)
        if task is None:
            self.task_running = False
            self._current_task_id = ""
            return None
        task_id = task.task_id
        task.end_time = _iso_time()
        task.status = "completed"
        if note:
            task.notes = "；".join(filter(None, (task.notes, note)))
        self.task_running = False
        self._current_task_id = ""
        self.add_event(
            "结束任务", "system", note,
            event_type="task_end", task_id=task_id,
            _manage_task=False,
        )
        return task

    def add_event(
        self,
        label: str = "",
        category: str = "user",
        note: str = "",
        *,
        source: Optional[str] = None,
        event_type: Optional[str] = None,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        event_time: Optional[float] = None,
        content: Optional[str] = None,
        _manage_task: bool = True,
    ) -> EventMarker:
        """Add one canonical event while retaining the old 3-argument API."""
        text = str(content if content is not None else label)
        kind = event_type or self._infer_event_type(text, category)
        linked_task = task_id or self._current_task_id

        # Older pages express task lifecycle as event labels.  Convert those
        # labels into task records without requiring a page rewrite.
        if _manage_task and self._session_active and kind == "task_start":
            task_name = text.split(":", 1)[1].strip() if ":" in text else self.task_type
            task = self._begin_task_record(task_name, self.task_difficulty, note)
            linked_task = task.task_id
        elif _manage_task and self._session_active and kind == "task_end":
            linked_task = self._current_task_id

        timestamp = time.time() if event_time is None else float(event_time)
        # DashboardPage emits the same start/end event immediately after the
        # service call; collapse that compatibility duplicate.
        if self._events:
            previous = self._events[-1]
            if (
                kind in {"session_start", "session_end"}
                and
                previous.content == text
                and previous.type == kind
                and abs(timestamp - previous.timestamp) < 1.0
            ):
                return previous

        ev = EventMarker(
            timestamp=timestamp,
            label=text,
            category=category,
            note=note,
            source=_normalise_source(source or self._session_source, demo=self._session_demo),
            type=kind,
            session_id=session_id or self.run_id,
            task_id=linked_task or "",
            content=text,
        )
        self._events.append(ev)

        if _manage_task and self._session_active and kind == "task_end":
            task = self._task_by_id(linked_task)
            if task is not None:
                task.end_time = ev.time
                task.status = "completed"
            self._current_task_id = ""
            self.task_running = False

        self._sync_active_record(persist=True)
        self.event_added.emit(ev)
        return ev

    def capture_session_snapshot(self) -> None:
        """Capture quality state for end-of-session summary generation."""
        if not self._session_active:
            return
        self._quality_history.append((
            time.time(), self.quality_level, self.poor_signal,
        ))

    def clear_interpretation(self) -> None:
        """Clear outputs that must not survive a rejected/error pipeline."""
        self.prob_positive = None
        self.prob_neutral = None
        self.prob_negative = None
        self.predicted_state = None
        self.confidence = None
        self.stable_state = None

    def set_pipeline_error(self, user_message: str, detail: str = "") -> None:
        """Expose a safe UI message while preserving diagnostic detail."""
        self.pipeline_state = "error"
        self.model_error_user = str(user_message)
        self.model_error_detail = str(detail)
        self.quality_level = "rejected"
        self.quality_reasons = [self.model_error_user]
        self.feedback_text = self.model_error_user
        self.clear_interpretation()

    def finalize_session(
        self,
        *,
        status: str = "completed",
        data_files: Optional[dict] = None,
    ) -> Optional[str]:
        """Finalize summaries, save metadata, then refresh isolated history."""
        if self._active_record is None:
            self._session_active = False
            return None
        if self._current_task_id:
            self.end_task(note="随会话结束")
        self.add_event(
            "会话结束", "system", event_type="session_end",
            source=self._session_source,
        )

        record = self._active_record
        now = time.time()
        record.end_time = _iso_time(now)
        record.status = status
        elapsed = now - self._session_started_at if self._session_started_at else 0.0
        record.duration_seconds = max(float(self.session_seconds), elapsed)
        record.user_id = self._user_id
        record.user_name = self._user_name
        if data_files:
            record.data_files.update(data_files)
        record.tasks = list(self._tasks)
        record.events = list(self._events)
        record.event_count = len(record.events)
        record.notes = self._task_summary()

        values = [float(value) for value in self._attention_history]
        record.avg_attention = sum(values) / len(values) if values else 0.0
        values = [float(value) for value in self._meditation_history]
        record.avg_meditation = sum(values) / len(values) if values else 0.0
        record.quality_summary = self._quality_summary()
        record.signal_quality = record.quality_summary["usable_ratio"]
        record.probability_summary = self._probability_summary()
        record.positive_ratio = record.probability_summary["mean_positive"]
        record.neutral_ratio = record.probability_summary["mean_neutral"]
        record.negative_ratio = record.probability_summary["mean_negative"]

        path = self._persist_active_record()
        self._session_active = False
        self.task_running = False
        self._active_record = None
        self.reload_history(include_demo=self._session_demo)
        return path

    def _begin_task_record(self, name: str, difficulty: str, notes: str) -> TaskRecord:
        if self._current_task_id:
            current = self._task_by_id(self._current_task_id)
            if current is not None and current.status == "running":
                current.end_time = _iso_time()
                current.status = "completed"
        task = TaskRecord(
            session_id=self.run_id,
            name=name or "自由学习",
            difficulty=(
                difficulty if difficulty in DIFFICULTY_LEVELS else DIFFICULTY_MEDIUM
            ),
            start_time=_iso_time(),
            notes=notes,
        )
        self._tasks.append(task)
        self._current_task_id = task.task_id
        self.task_type = task.name
        self.task_difficulty = task.difficulty
        self.task_running = True
        return task

    def _task_by_id(self, task_id: str) -> Optional[TaskRecord]:
        return next((item for item in reversed(self._tasks) if item.task_id == task_id), None)

    @staticmethod
    def _infer_event_type(content: str, category: str) -> str:
        if content.startswith("开始任务"):
            return "task_start"
        if content.startswith("结束任务"):
            return "task_end"
        if content.startswith("会话开始"):
            return "session_start"
        if content.startswith("会话结束"):
            return "session_end"
        if "暂停" in content:
            return "session_pause"
        if "恢复" in content:
            return "session_resume"
        if "难度" in content:
            return "difficulty_change"
        if category == "intervention":
            return "intervention"
        return "marker"

    def _task_summary(self) -> str:
        names = []
        for task in self._tasks:
            if task.name and task.name not in names:
                names.append(task.name)
        return "、".join(names) if names else self.task_type

    def _quality_summary(self) -> dict:
        levels = [item[1] for item in self._quality_history]
        counts = {name: levels.count(name) for name in ("trusted", "warning", "rejected")}
        total = len(levels)
        poor = [float(item[2]) for item in self._quality_history if item[2] is not None]
        usable = counts["trusted"] + counts["warning"]
        return {
            "sample_count": total,
            "trusted_count": counts["trusted"],
            "warning_count": counts["warning"],
            "rejected_count": counts["rejected"],
            "trusted_ratio": counts["trusted"] / total if total else 0.0,
            "usable_ratio": usable / total if total else 0.0,
            "average_poor_signal": sum(poor) / len(poor) if poor else None,
            "final_level": self.quality_level,
            "final_reasons": list(self.quality_reasons),
        }

    def _probability_summary(self) -> dict:
        samples = []
        for item in self._prob_history:
            try:
                samples.append((float(item[1]), float(item[2]), float(item[3])))
            except (IndexError, TypeError, ValueError):
                continue
        if not samples and all(value is not None for value in (
            self.prob_positive, self.prob_neutral, self.prob_negative,
        )):
            samples.append((
                float(self.prob_positive), float(self.prob_neutral),
                float(self.prob_negative),
            ))
        count = len(samples)
        means = [sum(row[index] for row in samples) / count for index in range(3)] if count else [0.0] * 3
        dominant = CLASS_NAMES[max(range(3), key=lambda index: means[index])] if count else None
        return {
            "sample_count": count,
            "mean_positive": means[0],
            "mean_neutral": means[1],
            "mean_negative": means[2],
            "dominant_state": dominant,
            "final_state": self.stable_state or self.predicted_state,
            "final_confidence": self.confidence,
        }

    def _sync_active_record(self, *, persist: bool) -> Optional[str]:
        if self._active_record is None:
            return None
        self._active_record.tasks = list(self._tasks)
        self._active_record.events = list(self._events)
        self._active_record.event_count = len(self._events)
        self._active_record.notes = self._task_summary()
        return self._persist_active_record() if persist else None

    def _persist_active_record(self) -> Optional[str]:
        if self._active_record is None or self._session_store is None:
            return None
        try:
            path = self._session_store.save(self._active_record)
            self.last_session_save_error = ""
            return str(path)
        except (OSError, ValueError, TypeError) as exc:
            self.last_session_save_error = str(exc)
            return None

    def emit_update(self):
        self.state_updated.emit(self)

    def reset_session(self):
        self._eeg_raw_buffer.clear()
        self._attention_history.clear()
        self._meditation_history.clear()
        self._prob_history.clear()
        self._quality_history.clear()
        self._events.clear()
        self._tasks.clear()
        self._current_task_id = ""
        self._active_record = None
        self._session_started_at = None
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

        # 复位自适应决策状态：task_type/task_difficulty 是用户选择，保留；
        # 但 task_running / adaptive_* 复位，避免新会话残留上次决策。
        self.task_running = False
        self.adaptive_action = AdaptiveAction.NONE
        self.adaptive_action_reason = ""
        self.adaptive_action_time = None
        self.adaptive_feedback_text = ""
