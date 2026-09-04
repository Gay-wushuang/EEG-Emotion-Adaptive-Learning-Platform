"""页面4：学习任务和事件标记页。

严格遵循 eeg_modular/ui_prototype/services/dashboard_state.py 中定义的
DashboardState 正式字段接口。UI 业务逻辑只消费正式字段，
内部簿记字段（_前缀）仅用于事件列表等簿记操作。

自适应学习场景：
- TaskPage 是 DashboardState 的状态消费者，不直接被后台服务调用；
- 任务类型/难度由 QComboBox 与 state.task_type / state.task_difficulty 双向同步；
- 自适应反馈区只展示 state.adaptive_* 字段，不参与决策逻辑；
- 信号质量 rejected 时，自适应反馈区不显示任何学习状态解释。
"""

from __future__ import annotations

import time
import inspect
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QLineEdit, QSplitter, QMessageBox,
)

from pages.base_page import BasePage
from widgets.card import Card
from services.dashboard_state import (
    CLASS_DISPLAY,
    DIFFICULTY_LEVELS, DIFFICULTY_DISPLAY,
    DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD,
    AdaptiveAction, ADAPTIVE_ACTION_DISPLAY,
)


TASK_TYPES = [
    "数学练习", "英语阅读", "编程任务", "物理复习",
    "语文写作", "专注冥想", "记忆训练", "自由学习",
]

# 业务层难度 <-> UI 索引映射（仅一处真相来源）
_DIFFICULTY_INDEX = {
    DIFFICULTY_EASY: 0,
    DIFFICULTY_MEDIUM: 1,
    DIFFICULTY_HARD: 2,
}
_INDEX_DIFFICULTY = {v: k for k, v in _DIFFICULTY_INDEX.items()}

QUICK_EVENTS = [
    ("开始专注", "self_report", "#4ADE80"),
    ("走神", "self_report", "#FBBF24"),
    ("感到困难", "self_report", "#F87171"),
    ("短暂休息", "manual", "#4FC3F7"),
    ("情绪波动", "self_report", "#FBBF24"),
    ("任务切换", "manual", "#94A3B8"),
]

EVENT_SOURCE_DISPLAY = {
    "user": "人工标记",
    "manual": "人工标记",
    "self_report": "主观反馈",
    "system": "系统记录",
    "inference": "模型检测",
    "intervention": "自适应动作",
    "adaptive": "自适应动作",
}


class TaskPage(BasePage):
    def __init__(self, state, service):
        self.state = state
        self.service = service
        self._task_start = 0.0
        self._task_active = False
        self._event_scopes = {}
        self._event_buttons = []
        # 防止程序同步 ComboBox 时回流触发用户回调
        self._syncing_combo = False
        super().__init__(
            "学习任务与事件标记",
            "管理学习任务并记录关键事件，用于后续会话分析与报告。"
        )
        self._build_ui()
        # 初始同步一次：state -> ComboBox
        self._sync_combos_from_state()
        self.set_role(self._role)

    def _build_ui(self):
        hierarchy_card = Card("记录层级")
        hierarchy_row = QHBoxLayout()
        hierarchy_row.setSpacing(12)
        relation = QLabel(
            "监测会话 = 一次连续学习记录；任务 = 会话中的学习活动；事件 = 会话或任务内的时间标记。"
        )
        relation.setWordWrap(True)
        relation.setStyleSheet("color: #AAB6C8; font-size: 12px;")
        hierarchy_row.addWidget(relation, 1)
        self._session_relation = QLabel("监测会话：未开始")
        self._session_relation.setObjectName("WarnLabel")
        hierarchy_row.addWidget(self._session_relation)
        self._btn_start_session = QPushButton("开始监测会话")
        self._btn_start_session.setObjectName("PrimaryButton")
        self._btn_start_session.clicked.connect(self._on_start_session)
        hierarchy_row.addWidget(self._btn_start_session)
        hierarchy_card.add_widget(self._wrap(hierarchy_row))
        hierarchy_card.setMaximumHeight(76)
        self.content_layout.addWidget(hierarchy_card)

        splitter = QSplitter(Qt.Horizontal)

        # ── 左侧：任务管理 ──
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # 任务设置卡片
        task_card = Card("当前任务")
        task_form = QGridLayout()
        task_form.setSpacing(8)

        task_form.addWidget(QLabel("任务类型:"), 0, 0)
        self._combo_task = QComboBox()
        self._combo_task.addItems(TASK_TYPES)
        # 从 state 初始化选中项
        if self.state.task_type in TASK_TYPES:
            self._combo_task.setCurrentIndex(TASK_TYPES.index(self.state.task_type))
        # 用户改动 -> 写回 state.task_type
        self._combo_task.currentIndexChanged.connect(self._on_user_changed_task)
        task_form.addWidget(self._combo_task, 0, 1)

        task_form.addWidget(QLabel("难度:"), 1, 0)
        self._combo_diff = QComboBox()
        self._combo_diff.addItems([
            DIFFICULTY_DISPLAY[DIFFICULTY_EASY],
            DIFFICULTY_DISPLAY[DIFFICULTY_MEDIUM],
            DIFFICULTY_DISPLAY[DIFFICULTY_HARD],
        ])
        # 从 state 初始化
        idx = _DIFFICULTY_INDEX.get(self.state.task_difficulty, 1)
        self._combo_diff.setCurrentIndex(idx)
        # 用户改动 -> 写回 state.task_difficulty
        self._combo_diff.currentIndexChanged.connect(self._on_user_changed_difficulty)
        task_form.addWidget(self._combo_diff, 1, 1)

        task_form.addWidget(QLabel("备注:"), 2, 0)
        self._input_note = QLineEdit()
        self._input_note.setPlaceholderText("可选备注信息")
        task_form.addWidget(self._input_note, 2, 1)

        task_card.add_widget(self._wrap(task_form))
        left_layout.addWidget(task_card)

        # AI 自适应反馈卡片（最小、不重做视觉）
        ai_card = Card("AI 自适应反馈")
        ai_form = QGridLayout()
        ai_form.setSpacing(6)

        ai_form.addWidget(QLabel("当前策略:"), 0, 0)
        self._ai_strategy = QLabel("无")
        self._ai_strategy.setStyleSheet("color: #4FC3F7; font-size: 13px;")
        ai_form.addWidget(self._ai_strategy, 0, 1)

        ai_form.addWidget(QLabel("AI 建议:"), 1, 0)
        self._ai_suggestion = QLabel("--")
        self._ai_suggestion.setWordWrap(True)
        self._ai_suggestion.setStyleSheet("color: #E8EDF3; font-size: 13px;")
        ai_form.addWidget(self._ai_suggestion, 1, 1)

        ai_form.addWidget(QLabel("触发原因:"), 2, 0)
        self._ai_reason = QLabel("--")
        self._ai_reason.setWordWrap(True)
        self._ai_reason.setStyleSheet("color: #94A3B8; font-size: 12px;")
        ai_form.addWidget(self._ai_reason, 2, 1)

        ai_card.add_widget(self._wrap(ai_form))
        left_layout.addWidget(ai_card)

        self._teacher_card = Card("教学端 · 班级过程")
        teacher_grid = QGridLayout()
        teacher_grid.setSpacing(6)
        self._teacher_trend = QLabel("班级状态趋势：暂无班级汇总")
        self._teacher_alert = QLabel("异常提醒：暂无")
        self._teacher_process = QLabel("过程记录：0 条")
        for row, label in enumerate(
            (self._teacher_trend, self._teacher_alert, self._teacher_process)
        ):
            label.setWordWrap(True)
            label.setStyleSheet("color: #C5CDD9; font-size: 12px;")
            teacher_grid.addWidget(label, row, 0)
        self._teacher_card.add_widget(self._wrap(teacher_grid))
        self._teacher_card.setVisible(False)
        left_layout.addWidget(self._teacher_card)

        # 任务计时
        timer_card = Card("任务计时")
        timer_layout = QVBoxLayout()

        self._task_time = QLabel("00:00")
        self._task_time.setObjectName("BigValue")
        self._task_time.setAlignment(Qt.AlignCenter)
        self._task_time.setStyleSheet("font-size: 48px; font-weight: bold; color: #4FC3F7;")
        timer_layout.addWidget(self._task_time)

        self._task_status = QLabel("未开始")
        self._task_status.setAlignment(Qt.AlignCenter)
        self._task_status.setStyleSheet("color: #6B7689; font-size: 14px;")
        timer_layout.addWidget(self._task_status)

        btn_row = QHBoxLayout()
        self._btn_task_start = QPushButton("开始任务")
        self._btn_task_start.setObjectName("PrimaryButton")
        self._btn_task_start.clicked.connect(self._on_task_start)
        btn_row.addWidget(self._btn_task_start)

        self._btn_task_stop = QPushButton("结束任务")
        self._btn_task_stop.setEnabled(False)
        self._btn_task_stop.clicked.connect(self._on_task_stop)
        btn_row.addWidget(self._btn_task_stop)
        timer_layout.addLayout(btn_row)

        task_card_inner = self._wrap(timer_layout)
        timer_card.add_widget(task_card_inner)
        left_layout.addWidget(timer_card)

        # 快速状态
        state_card = Card("当前学习状态")
        state_layout = QGridLayout()
        state_layout.setSpacing(6)

        self._state_label = QLabel("状态：--")
        self._state_label.setObjectName("CardValueSmall")
        state_layout.addWidget(self._state_label, 0, 0, 1, 2)

        self._state_att = QLabel("专注度：--")
        self._state_att.setStyleSheet("color: #4FC3F7; font-size: 13px;")
        state_layout.addWidget(self._state_att, 1, 0)

        self._state_med = QLabel("放松度：--")
        self._state_med.setStyleSheet("color: #4ADE80; font-size: 13px;")
        state_layout.addWidget(self._state_med, 1, 1)

        state_card.add_widget(self._wrap(state_layout))
        left_layout.addWidget(state_card)

        left_layout.addStretch()
        splitter.addWidget(left)

        # ── 右侧：事件标记 + 时间线 ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        # 快速事件标记
        event_card = Card("快速事件标记")
        event_help = QLabel(
            "事件用于在报告中定位关键时刻。主观感受与人工操作会标明来源，并关联当前会话 / 任务。"
        )
        event_help.setWordWrap(True)
        event_help.setStyleSheet("color: #8491A5; font-size: 12px;")
        event_card.add_widget(event_help)
        event_grid = QGridLayout()
        event_grid.setSpacing(8)
        for i, (label, cat, color) in enumerate(QUICK_EVENTS):
            btn = QPushButton(label)
            btn.setObjectName("EventButton")
            btn.clicked.connect(
                lambda checked, l=label, c=cat: self._add_quick_event(l, c)
            )
            btn.setEnabled(getattr(self.state, "_session_active", False))
            self._event_buttons.append(btn)
            event_grid.addWidget(btn, i // 3, i % 3)

        event_card.add_widget(self._wrap(event_grid))

        # 自定义事件
        custom_row = QHBoxLayout()
        self._input_custom = QLineEdit()
        self._input_custom.setPlaceholderText("输入自定义事件描述...")
        custom_row.addWidget(self._input_custom, 1)

        btn_custom = QPushButton("标记")
        btn_custom.clicked.connect(self._add_custom_event)
        btn_custom.setEnabled(getattr(self.state, "_session_active", False))
        self._event_buttons.append(btn_custom)
        custom_row.addWidget(btn_custom)
        event_card.add_widget(self._wrap(custom_row))

        right_layout.addWidget(event_card)

        # 事件时间线
        timeline_card = Card("事件时间线")
        timeline_layout = QVBoxLayout()

        self._event_table = QTableWidget()
        self._event_table.setColumnCount(5)
        self._event_table.setHorizontalHeaderLabels(["时间", "来源", "所属", "事件", "备注"])
        self._event_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._event_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._event_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._event_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._event_table.setAlternatingRowColors(True)
        self._event_table.verticalHeader().setVisible(False)
        self._event_table.setEditTriggers(QTableWidget.NoEditTriggers)
        timeline_layout.addWidget(self._event_table)

        # 清除按钮
        clear_row = QHBoxLayout()
        clear_hint = QLabel("仅删除人工标记和主观反馈；系统、模型及自适应审计记录保留。")
        clear_hint.setStyleSheet("color: #8491A5; font-size: 11px;")
        clear_hint.setWordWrap(True)
        clear_row.addWidget(clear_hint, 1)
        self._btn_clear = QPushButton("删除人工事件")
        self._btn_clear.setObjectName("DangerButton")
        self._btn_clear.clicked.connect(self._clear_events)
        clear_row.addWidget(self._btn_clear)
        timeline_layout.addLayout(clear_row)

        timeline_card.add_widget(self._wrap(timeline_layout))
        right_layout.addWidget(timeline_card, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 600])

        self.content_layout.addWidget(splitter)

        # 监听事件
        self.state.event_added.connect(self._on_event_added)

    def _wrap(self, layout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        return w

    def set_role(self, role: str):
        super().set_role(role)
        if hasattr(self, "_teacher_card"):
            self._teacher_card.setVisible(self._role == "teacher")

    def _on_start_session(self):
        """从任务页显式建立监测会话，解除隐含跨页前置条件。"""
        if getattr(self.state, "_session_active", False):
            return
        self.state.reset_session()
        event_count = len(getattr(self.state, "_events", []))
        self.service.start_session()
        if len(getattr(self.state, "_events", [])) == event_count:
            self.state.add_event("会话开始", "system")
        self._session_relation.setText("监测会话：进行中")
        self._session_relation.setObjectName("GoodLabel")
        self._session_relation.setStyleSheet("color: #4ADE80; font-size: 12px;")
        self._btn_start_session.setEnabled(False)

    def _on_task_start(self):
        if not getattr(self.state, "_session_active", False):
            answer = QMessageBox.question(
                self,
                "需要监测会话",
                "任务必须关联到一次监测会话。是否现在开始监测会话并继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer != QMessageBox.Yes:
                return
            self._on_start_session()
        self._task_start = time.time()
        self._task_active = True
        self.state.task_running = True
        self._btn_task_start.setEnabled(False)
        self._btn_task_stop.setEnabled(True)
        self._task_status.setText("进行中")
        self._task_status.setStyleSheet("color: #4ADE80; font-size: 14px;")
        task_name = self._combo_task.currentText()
        if hasattr(self.state, "begin_task"):
            self.state.begin_task(
                task_name, getattr(self.state, "task_difficulty", DIFFICULTY_MEDIUM),
                self._input_note.text(),
            )
        else:
            self._record_event(
                f"开始任务：{task_name}", "system", self._input_note.text(), scope="task"
            )

    def _on_task_stop(self):
        self._task_active = False
        self.state.task_running = False
        self._btn_task_start.setEnabled(True)
        self._btn_task_stop.setEnabled(False)
        self._task_status.setText("已结束")
        self._task_status.setStyleSheet("color: #6B7689; font-size: 14px;")
        if hasattr(self.state, "end_task"):
            self.state.end_task()
        else:
            self._record_event("结束任务", "system", scope="task")

    # ── 用户改动 ComboBox → 写回 DashboardState ──
    # 这是 UI 与 state 保持单一真相来源的关键。
    # 用 _syncing_combo 防止 _sync_combos_from_state 回流触发本回调。
    def _on_user_changed_task(self, idx: int):
        if self._syncing_combo:
            return
        text = self._combo_task.itemText(idx)
        if text and self.state.task_type != text:
            self.state.task_type = text
            if getattr(self.state, "_session_active", False):
                self._record_event(f"切换任务类型：{text}", "manual", scope="session")

    def _on_user_changed_difficulty(self, idx: int):
        if self._syncing_combo:
            return
        new_diff = _INDEX_DIFFICULTY.get(idx, DIFFICULTY_MEDIUM)
        if self.state.task_difficulty != new_diff:
            old_display = DIFFICULTY_DISPLAY.get(self.state.task_difficulty, "--")
            new_display = DIFFICULTY_DISPLAY.get(new_diff, "--")
            self.state.task_difficulty = new_diff
            if getattr(self.state, "_session_active", False):
                self._record_event(
                    f"手动调整难度：{old_display} → {new_display}",
                    "manual",
                    scope="task" if self._task_active else "session",
                )

    def _sync_combos_from_state(self):
        """state -> ComboBox 单向同步，用 blockSignals 防止回流。

        当 _apply_adaptive_action 修改 state.task_difficulty 后，
        update_state 调用本方法，让 QComboBox 显示新的正式状态。
        """
        self._syncing_combo = True
        try:
            # 任务类型
            if self.state.task_type in TASK_TYPES:
                target = TASK_TYPES.index(self.state.task_type)
                if self._combo_task.currentIndex() != target:
                    self._combo_task.setCurrentIndex(target)

            # 难度
            target_idx = _DIFFICULTY_INDEX.get(
                self.state.task_difficulty,
                _DIFFICULTY_INDEX[DIFFICULTY_MEDIUM],
            )
            if self._combo_diff.currentIndex() != target_idx:
                self._combo_diff.setCurrentIndex(target_idx)
        finally:
            self._syncing_combo = False

    def _event_scope_text(self, scope: str) -> str:
        if scope == "task":
            return f"任务：{getattr(self.state, 'task_type', '当前任务')}"
        return f"会话：{getattr(self.state, 'run_id', '--')}"

    def _record_event(
        self, label: str, category: str, note: str = "", scope: str | None = None
    ):
        """写入事件，并兼容后续扩展的 Session/Task/Event 字段。"""
        scope = scope or ("task" if self._task_active else "session")
        method = self.state.add_event
        supported = inspect.signature(method).parameters
        extra = {}
        if label.startswith("开始任务"):
            event_type = "task_start"
        elif label.startswith("结束任务"):
            event_type = "task_end"
        else:
            event_type = category
        candidates = {
            "source": getattr(self.state, "mode", "live"),
            "event_type": event_type,
            "scope": scope,
            "session_id": getattr(self.state, "run_id", ""),
            "task_id": (
                getattr(self.state, "_current_task_id", "")
                or getattr(self.state, "current_task_id", "")
            ) if scope == "task" else "",
        }
        for key, value in candidates.items():
            if key in supported:
                extra[key] = value
        method(label, category, note, **extra)
        events = getattr(self.state, "_events", [])
        if events:
            event = events[-1]
            self._event_scopes[getattr(event, "timestamp", id(event))] = self._event_scope_text(scope)
        self._refresh_table()

    def _ensure_event_session(self) -> bool:
        if getattr(self.state, "_session_active", False):
            return True
        QMessageBox.information(
            self,
            "尚未开始监测会话",
            "事件必须归属于监测会话。请先点击本页上方的“开始监测会话”。",
        )
        return False

    def _add_quick_event(self, label: str, category: str = "manual"):
        if self._ensure_event_session():
            self._record_event(label, category)

    def _add_custom_event(self):
        text = self._input_custom.text().strip()
        if text and self._ensure_event_session():
            self._record_event(text, "manual")
            self._input_custom.clear()

    def _clear_events(self):
        removable = {"user", "manual", "self_report"}
        events = getattr(self.state, "_events", [])
        count = sum(
            1 for event in events
            if getattr(event, "category", "") in removable
        )
        if not count:
            QMessageBox.information(self, "无需删除", "当前会话没有可删除的人工事件。")
            return
        answer = QMessageBox.question(
            self,
            "确认删除人工事件",
            f"将删除当前会话中的 {count} 条人工标记 / 主观反馈。\n"
            "系统、模型和自适应审计记录会保留。是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        events[:] = [
            event for event in events
            if getattr(event, "category", "") not in removable
        ]
        self._refresh_table()

    def _on_event_added(self, event):
        self._refresh_table()

    def _refresh_table(self):
        events = self.state._events
        self._event_table.setRowCount(len(events))
        for i, ev in enumerate(events):
            from datetime import datetime
            ts = datetime.fromtimestamp(ev.timestamp).strftime("%H:%M:%S")
            self._event_table.setItem(i, 0, QTableWidgetItem(ts))
            source = getattr(ev, "category", "")
            cat_item = QTableWidgetItem(EVENT_SOURCE_DISPLAY.get(source, source or "未知"))
            cat_item.setForeground(Qt.GlobalColor.white)
            self._event_table.setItem(i, 1, cat_item)
            task_id = getattr(ev, "task_id", "")
            session_id = getattr(ev, "session_id", "") or getattr(self.state, "run_id", "--")
            scope = f"任务：{task_id}" if task_id else f"会话：{session_id}"
            if not scope:
                scope = self._event_scopes.get(
                    getattr(ev, "timestamp", id(ev)),
                    self._event_scope_text("task" if self._task_active else "session"),
                )
            self._event_table.setItem(i, 2, QTableWidgetItem(str(scope)))
            self._event_table.setItem(i, 3, QTableWidgetItem(ev.label))
            self._event_table.setItem(i, 4, QTableWidgetItem(ev.note))
        self._event_table.scrollToBottom()

    def update_state(self, state):
        requested_role = self._normalize_role(getattr(state, "current_role", self._role))
        if requested_role != self._role:
            self.set_role(requested_role)

        session_active = bool(getattr(state, "_session_active", False))
        self._btn_start_session.setEnabled(not session_active)
        self._session_relation.setText(
            f"监测会话：{'进行中' if session_active else '未开始'}"
        )
        self._session_relation.setStyleSheet(
            f"color: {'#4ADE80' if session_active else '#FBBF24'}; font-size: 12px;"
        )
        for button in self._event_buttons:
            button.setEnabled(session_active)

        # 任务计时
        if self._task_active:
            elapsed = time.time() - self._task_start
            mins = int(elapsed) // 60
            secs = int(elapsed) % 60
            self._task_time.setText(f"{mins:02d}:{secs:02d}")

        # 任务运行状态：UI 的 _task_active 跟随 state.task_running
        # （但若用户未通过按钮开始，UI 不会自动启动）
        if state.task_running and not self._task_active:
            # 后台不应直接启动 UI 任务计时，这里只做显示同步
            self._task_status.setText("进行中")
            self._task_status.setStyleSheet("color: #4ADE80; font-size: 14px;")
        elif not state.task_running and self._task_active:
            # state 被外部复位（如 reset_session），UI 也复位
            self._task_active = False
            self._btn_task_start.setEnabled(True)
            self._btn_task_stop.setEnabled(False)
            self._task_status.setText("未开始")
            self._task_status.setStyleSheet("color: #6B7689; font-size: 14px;")

        # 当前状态：使用 stable_state 和 inference_eligible
        if state.inference_eligible:
            display = CLASS_DISPLAY.get(state.stable_state, "--")
            self._state_label.setText(f"状态：{display}")
            color_map = {
                "positive": "#4ADE80",
                "neutral": "#4FC3F7",
                "negative": "#F87171",
                "unknown": "#6B7689",
            }
            self._state_label.setStyleSheet(
                f"font-size: 22px; font-weight: bold; color: {color_map.get(state.stable_state, '#E8EDF3')};"
            )
        else:
            self._state_label.setText("状态：等待推理...")
            self._state_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #6B7689;")

        # Attention / Meditation：处理 None
        if state.attention is not None:
            self._state_att.setText(f"专注度：{state.attention:.0f}")
        else:
            self._state_att.setText("专注度：--")

        if state.meditation is not None:
            self._state_med.setText(f"放松度：{state.meditation:.0f}")
        else:
            self._state_med.setText("放松度：--")

        # ── 自适应反馈区（双层状态输出：rejected 时不解释）──
        self._refresh_adaptive_feedback(state)

        # ── state -> ComboBox 单向同步 ──
        # 当 _apply_adaptive_action 改了 task_difficulty 后，
        # ComboBox 必须显示新的正式难度。
        self._sync_combos_from_state()

        # 刷新事件表：使用 _events
        if not hasattr(self, "_last_event_count") or self._last_event_count != len(state._events):
            self._refresh_table()
            self._last_event_count = len(state._events)

        if self._role == "teacher":
            stable = CLASS_DISPLAY.get(getattr(state, "stable_state", None), "暂无有效趋势")
            trend = getattr(state, "class_trend_summary", "") or f"当前监测：{stable}"
            alerts = getattr(state, "class_alerts", None)
            if alerts:
                alert_text = f"{len(alerts)} 条待关注"
            elif getattr(state, "quality_level", "rejected") == "rejected":
                alert_text = "当前信号质量待处理"
            else:
                alert_text = "暂无异常提醒"
            self._teacher_trend.setText(f"班级状态趋势：{trend}")
            self._teacher_alert.setText(f"异常提醒：{alert_text}")
            self._teacher_process.setText(
                f"过程记录：{len(getattr(state, '_events', []))} 条"
            )

    def _refresh_adaptive_feedback(self, state):
        """刷新 AI 自适应反馈卡片。

        严格遵守 AGENTS.md：
        - quality_level == rejected 时不进行学习状态解释，只显示信号不足提示；
        - 不使用医疗化措辞；
        - negative 只表述为"负性状态/趋势"。
        """
        if state.quality_level == "rejected":
            self._ai_strategy.setText("暂不评估")
            self._ai_strategy.setStyleSheet("color: #6B7689; font-size: 13px;")
            self._ai_suggestion.setText("当前信号质量不足，暂不进行学习状态解释。")
            self._ai_suggestion.setStyleSheet("color: #6B7689; font-size: 13px;")
            self._ai_reason.setText("--")
            self._ai_reason.setStyleSheet("color: #6B7689; font-size: 12px;")
            return

        action = state.adaptive_action
        strategy_text = ADAPTIVE_ACTION_DISPLAY.get(action, "无")
        color = "#4FC3F7"
        if action == AdaptiveAction.REDUCE_DIFFICULTY:
            color = "#FBBF24"  # 琥珀：降低难度
        elif action == AdaptiveAction.SUGGEST_BREAK:
            color = "#F87171"  # 红：建议休息（非严重错误，但需注意）
        elif action == AdaptiveAction.MAINTAIN:
            color = "#4ADE80"  # 绿：维持

        self._ai_strategy.setText(strategy_text)
        self._ai_strategy.setStyleSheet(f"color: {color}; font-size: 13px;")

        # AI 建议文本：优先用 adaptive_feedback_text，否则用基础 feedback_text
        if state.adaptive_feedback_text:
            self._ai_suggestion.setText(state.adaptive_feedback_text)
        elif state.inference_eligible:
            self._ai_suggestion.setText(state.feedback_text or "维持当前学习计划。")
        else:
            self._ai_suggestion.setText("等待信号稳定后将生成学习建议。")
        self._ai_suggestion.setStyleSheet("color: #E8EDF3; font-size: 13px;")

        # 触发原因
        if state.adaptive_action_reason:
            self._ai_reason.setText(state.adaptive_action_reason)
        else:
            self._ai_reason.setText("--")
        self._ai_reason.setStyleSheet("color: #94A3B8; font-size: 12px;")

    def on_show(self):
        self._refresh_table()
