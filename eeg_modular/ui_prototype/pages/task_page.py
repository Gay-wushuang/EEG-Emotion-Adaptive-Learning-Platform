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
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QLineEdit, QSplitter, QMessageBox, QDialog,
    QDialogButtonBox, QFormLayout,
)

from pages.base_page import BasePage
from widgets.card import Card
from widgets.trend_plot import TrendPlotWidget
from services.dashboard_state import (
    CLASS_DISPLAY,
    DIFFICULTY_LEVELS, DIFFICULTY_DISPLAY,
    DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD,
    AdaptiveAction, ADAPTIVE_ACTION_DISPLAY,
)
from services.identity_store import IdentityStore
from services.teaching_store import (
    AssignmentStore, StudentRuntimeRegistry, TeacherSelectionContext,
    TeacherStudentStore,
)
from services.learning_readiness import learning_start_block_reason


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
    ("疑似走神", "teacher", "#FBBF24"),
    ("教师干预", "teacher", "#F87171"),
    ("环境干扰", "teacher", "#4FC3F7"),
    ("提问 / 课堂事件", "teacher", "#94A3B8"),
]

EVENT_SOURCE_DISPLAY = {
    "user": "人工标记",
    "manual": "人工标记",
    "self_report": "学生反馈",
    "system": "系统记录",
    "teacher": "教师观察",
    "inference": "模型检测",
    "intervention": "自适应动作",
    "adaptive": "自适应动作",
}


class TaskPage(BasePage):
    FEEDBACK_DIALOG_STYLE = (
        "QDialog { background: #161D2A; color: #E8EDF3; }"
        "QLabel { color: #E8EDF3; font-size: 13px; }"
        "QComboBox { background: #222B3A; color: #F3F6FA; "
        "border: 1px solid #3B475B; border-radius: 4px; padding: 6px; }"
        "QComboBox QAbstractItemView { background: #222B3A; color: #F3F6FA; "
        "selection-background-color: #315A7D; }"
        "QPushButton { color: #E8EDF3; background: #263246; "
        "border: 1px solid #46536A; border-radius: 4px; padding: 6px 12px; }"
        "QPushButton:hover { background: #30415A; }"
    )
    def __init__(self, state, service, *, binding_store=None, assignment_store=None,
                 runtime_registry=None, identity_store=None, selection_context=None):
        self.state = state
        self.service = service
        self._task_start = 0.0
        self._task_active = False
        self._task_owns_session = False
        self._event_scopes = {}
        self._event_buttons = []
        self._teacher_assignment = None
        self.identity_store = identity_store or IdentityStore()
        self.binding_store = binding_store or TeacherStudentStore(
            identity_store=self.identity_store
        )
        self.assignment_store = assignment_store or AssignmentStore()
        self.runtime_registry = runtime_registry or StudentRuntimeRegistry.shared()
        self.selection_context = selection_context or TeacherSelectionContext.shared()
        self._selected_assignment_id = ""
        self._teacher_form_dirty = False
        self._task_paused = False
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
        self._hierarchy_card = hierarchy_card
        hierarchy_row = QHBoxLayout()
        hierarchy_row.setSpacing(12)
        relation = QLabel(
            "监测会话 = 一次连续学习记录；任务 = 会话中的学习活动；事件 = 会话或任务内的时间标记。"
        )
        self._hierarchy_help = relation
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
        self._task_card = task_card
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
        self._input_note.textEdited.connect(self._mark_teacher_form_dirty)
        task_form.addWidget(self._input_note, 2, 1)

        self._label_student = QLabel("学生:")
        self._combo_student = QComboBox()
        self._combo_student.currentIndexChanged.connect(self._on_teacher_student_changed)
        task_form.addWidget(self._label_student, 3, 0)
        task_form.addWidget(self._combo_student, 3, 1)

        self._input_student = QLineEdit()
        self._input_student.setPlaceholderText("学生 ID，例如 st_001")
        self._btn_add_student = QPushButton("添加学生")
        self._btn_add_student.clicked.connect(self._add_student_binding)
        self._btn_remove_student = QPushButton("移除绑定")
        self._btn_remove_student.clicked.connect(self._remove_student_binding)
        bind_row = QHBoxLayout()
        bind_row.addWidget(self._input_student, 1)
        bind_row.addWidget(self._btn_add_student)
        bind_row.addWidget(self._btn_remove_student)
        self._binding_widget = self._wrap(bind_row)
        task_form.addWidget(self._binding_widget, 4, 0, 1, 2)

        self._combo_assignment = QComboBox()
        self._combo_assignment.setMinimumHeight(42)
        self._combo_assignment.setMinimumWidth(320)
        self._combo_assignment.setMaxVisibleItems(8)
        self._combo_assignment.setStyleSheet(
            "QComboBox { padding: 7px 10px; }"
            "QComboBox QAbstractItemView { min-width: 420px; }"
            "QComboBox QAbstractItemView::item { min-height: 34px; padding: 5px; }"
        )
        self._teacher_poll_timer = QTimer(self)
        self._teacher_poll_timer.setInterval(500)
        self._teacher_poll_timer.timeout.connect(self._poll_teacher_runtime)
        self._combo_assignment.currentIndexChanged.connect(self._on_student_assignment_changed)
        task_form.addWidget(self._combo_assignment, 5, 0, 1, 2)

        self._assignment_summary = QLabel("暂无教师布置任务")
        self._assignment_summary.setWordWrap(True)
        task_form.addWidget(self._assignment_summary, 6, 0, 1, 2)

        self._btn_publish_task = QPushButton("发布任务")
        self._btn_publish_task.clicked.connect(self._publish_teacher_task)
        self._btn_publish_task.setVisible(False)
        task_form.addWidget(self._btn_publish_task, 7, 1)

        task_card.add_widget(self._wrap(task_form))
        left_layout.addWidget(task_card)

        # AI 自适应反馈卡片（最小、不重做视觉）
        ai_card = Card("AI 自适应反馈")
        self._ai_card = ai_card
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

        self._teacher_card = Card("教师端 · 学生过程")
        teacher_grid = QGridLayout()
        teacher_grid.setSpacing(6)
        self._teacher_trend = QLabel("班级状态趋势：暂无班级汇总")
        self._teacher_alert = QLabel("异常提醒：暂无")
        self._teacher_process = QLabel("过程记录：0 条")
        self._teacher_advice = QLabel("AI 建议：暂无实时数据")
        for row, label in enumerate(
            (self._teacher_trend, self._teacher_alert, self._teacher_process,
             self._teacher_advice)
        ):
            label.setWordWrap(True)
            label.setStyleSheet("color: #C5CDD9; font-size: 12px;")
            teacher_grid.addWidget(label, row, 0)
        self._teacher_trend_plot = TrendPlotWidget()
        self._teacher_trend_plot.setMinimumHeight(120)
        teacher_grid.addWidget(self._teacher_trend_plot, 4, 0)
        self._teacher_card.add_widget(self._wrap(teacher_grid))
        self._teacher_card.setVisible(False)
        left_layout.addWidget(self._teacher_card)

        # 任务计时
        timer_card = Card("学习计时")
        self._timer_card = timer_card
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
        self._btn_task_start = QPushButton("开始学习")
        self._btn_task_start.setObjectName("PrimaryButton")
        self._btn_task_start.clicked.connect(self._on_task_start)
        btn_row.addWidget(self._btn_task_start)

        self._btn_task_stop = QPushButton("结束任务")
        self._btn_task_stop.setEnabled(False)
        self._btn_task_stop.clicked.connect(self._on_task_stop)
        btn_row.addWidget(self._btn_task_stop)
        self._btn_task_pause = QPushButton("暂停学习")
        self._btn_task_pause.setEnabled(False)
        self._btn_task_pause.clicked.connect(self._on_task_pause_resume)
        btn_row.addWidget(self._btn_task_pause)
        timer_layout.addLayout(btn_row)

        task_card_inner = self._wrap(timer_layout)
        timer_card.add_widget(task_card_inner)
        left_layout.addWidget(timer_card)

        # 快速状态
        state_card = Card("当前学习状态")
        self._state_card = state_card
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
        event_card = Card("教师观察事件")
        self._event_card = event_card
        event_help = QLabel(
            "用于记录课堂观察；每条记录都会关联教师、学生、会话与任务。"
        )
        self._event_help = event_help
        event_help.setWordWrap(True)
        event_help.setStyleSheet("color: #8491A5; font-size: 12px;")
        event_card.add_widget(event_help)
        event_grid = QGridLayout()
        event_grid.setSpacing(8)
        self._teacher_event_buttons = []
        for i, (label, cat, color) in enumerate(QUICK_EVENTS):
            btn = QPushButton(label)
            btn.setObjectName("EventButton")
            btn.clicked.connect(
                lambda checked, l=label, c=cat: self._add_quick_event(l, c)
            )
            btn.setEnabled(self._session_is_active())
            self._event_buttons.append(btn)
            self._teacher_event_buttons.append(btn)
            event_grid.addWidget(btn, i // 3, i % 3)

        event_card.add_widget(self._wrap(event_grid))

        # 自定义事件
        custom_row = QHBoxLayout()
        self._input_custom = QLineEdit()
        self._input_custom.setPlaceholderText("输入自定义事件描述...")
        custom_row.addWidget(self._input_custom, 1)

        btn_custom = QPushButton("标记")
        self._btn_custom_event = btn_custom
        btn_custom.clicked.connect(self._add_custom_event)
        btn_custom.setEnabled(self._session_is_active())
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
            teacher = self._role == "teacher"
            student = self._role == "student"
            if teacher:
                self._title_label.setText("任务发布与学生观察")
                self._desc_label.setText("发布学习任务、查看执行状态并记录教师观察事件。")
            elif student:
                self._title_label.setText("我的学习任务")
                self._desc_label.setText("开始教师发布的任务；系统将自动管理监测与学习记录。")
            else:
                self._title_label.setText("诊断记录")
                self._desc_label.setText("记录本机设备与模型诊断过程；数据仅归属于当前管理员。")
            self._teacher_card.setVisible(teacher)
            self._ai_card.setVisible(not teacher)
            self._state_card.setVisible(not teacher)
            self._btn_publish_task.setVisible(teacher)
            for widget in (
                self._label_student, self._combo_student, self._binding_widget,
            ):
                widget.setVisible(teacher)
            self._assignment_summary.setVisible(teacher or student)
            self._combo_assignment.setVisible(student)
            self._combo_task.setEnabled(not student)
            self._combo_diff.setEnabled(not student)
            self._input_note.setReadOnly(student)
            # Students operate the learning workflow, never the underlying Session.
            self._hierarchy_card.setVisible(not student and not teacher)
            self._timer_card.setVisible(not teacher)
            self._event_card.setVisible(teacher or self._role == "research")
            self._btn_start_session.setVisible(self._role == "research")
            self._btn_task_start.setVisible(not teacher)
            self._btn_task_stop.setVisible(not teacher)
            self._btn_task_pause.setVisible(not teacher)
            self._btn_clear.setVisible(self._role == "research")
            self._input_custom.setReadOnly(False)
            self._input_custom.setPlaceholderText(
                "输入自定义观察..." if teacher else "输入自定义事件描述..."
            )
            for button in self._event_buttons:
                button.setVisible(teacher or self._role == "research")
            if self._role == "research":
                self._hierarchy_card.set_title("本机监测")
                self._hierarchy_help.setText("本机监测用于设备、信号与模型诊断，不属于学生学习记录。")
                self._session_relation.setText("本机监测：未开始")
                self._btn_start_session.setText("开始本机监测")
                self._task_card.set_title("本机诊断项目")
                self._timer_card.set_title("监测时长")
                self._btn_task_start.setText("开始监测")
                self._btn_task_stop.setText("结束监测")
                self._btn_task_pause.setText("暂停监测")
                self._state_card.set_title("本机诊断状态")
                self._event_card.set_title("诊断事件")
                self._event_help.setText("记录设备、信号、模型测试、环境干扰或管理员备注。")
                self._input_custom.setPlaceholderText("输入管理员诊断备注...")
                for button in self._teacher_event_buttons:
                    button.setVisible(False)
            else:
                self._hierarchy_card.set_title("记录层级")
                self._btn_start_session.setText("开始监测会话")
                self._task_card.set_title("当前任务")
                self._timer_card.set_title("学习计时")
                self._btn_task_start.setText("开始学习")
                self._btn_task_stop.setText("结束任务")
                self._btn_task_pause.setText("暂停学习")
            if teacher:
                self.selection_context.set_teacher(getattr(self.state, "_user_id", ""))
                self._event_card.set_title("教师观察事件")
                self._event_help.setText("用于记录课堂观察；每条记录都会关联教师、学生、会话与任务。")
                self._teacher_poll_timer.start()
                self._refresh_teacher_students()
            elif student:
                self._teacher_poll_timer.stop()
                self._refresh_student_assignments()
            else:
                self._teacher_poll_timer.stop()

    def _poll_teacher_runtime(self):
        if self._role == "teacher":
            self._refresh_teacher_students()
            self._refresh_teacher_observation()

    def _mark_teacher_form_dirty(self, *args):
        if self._role == "teacher":
            self._teacher_form_dirty = True

    def _refresh_teacher_students(self):
        teacher_id = getattr(self.state, "_user_id", "")
        selected = self.selection_context.selected_student_id or self._combo_student.currentData()
        try:
            profiles = {
                item["user_id"]: item["name"]
                for item in self.identity_store.list_profiles()
            }
        except OSError:
            profiles = {}
        self._combo_student.blockSignals(True)
        self._combo_student.clear()
        self._combo_student.addItem("请选择学生", "")
        for student_id in self.binding_store.students_for(teacher_id):
            name = profiles.get(student_id, "")
            self._combo_student.addItem(f"{student_id} {name}".strip(), student_id)
        index = self._combo_student.findData(selected)
        self._combo_student.setCurrentIndex(max(0, index))
        self._combo_student.blockSignals(False)
        self._on_teacher_student_changed(self._combo_student.currentIndex())

    def _add_student_binding(self):
        if self._role != "teacher":
            return
        student_id = self._input_student.text().strip()
        try:
            self.binding_store.add(getattr(self.state, "_user_id", ""), student_id)
        except ValueError as exc:
            QMessageBox.warning(self, "无法添加学生", str(exc))
            return
        self._input_student.clear()
        self._refresh_teacher_students()
        index = self._combo_student.findData(student_id)
        if index >= 0:
            self._combo_student.setCurrentIndex(index)

    def _remove_student_binding(self):
        student_id = self._combo_student.currentData()
        if self._role == "teacher" and student_id:
            self.binding_store.remove(getattr(self.state, "_user_id", ""), student_id)
            self._refresh_teacher_students()

    def _on_teacher_student_changed(self, index):
        if self._role != "teacher":
            return
        self.selection_context.select(self._combo_student.currentData() or "")
        self._refresh_teacher_observation()

    def _refresh_teacher_observation(self):
        student_id = self.selection_context.selected_student_id
        snapshot = self.runtime_registry.get(student_id) if student_id else None
        assignments = self.assignment_store.for_teacher(
            getattr(self.state, "_user_id", ""), student_id
        ) if student_id else []
        if assignments:
            self._show_published_assignment(assignments[-1])
        elif student_id:
            self._assignment_summary.setText("该学生暂无已发布任务")
        active = bool(
            snapshot and not snapshot.get("stale")
            and snapshot.get("session_id") and snapshot.get("task_id")
        )
        for button in self._event_buttons:
            button.setEnabled(active)
        if not student_id:
            self._teacher_trend.setText("学生状态：请先选择学生")
            self._teacher_alert.setText("当前学生没有进行中的学习任务")
            self._teacher_trend_plot.reset()
            self._teacher_advice.setText("AI 建议：暂无实时数据")
            self._refresh_table([])
            return
        if not snapshot:
            self._teacher_trend.setText(f"{student_id}：暂无实时数据")
            self._teacher_alert.setText("当前学生没有进行中的学习任务")
            self._teacher_trend_plot.reset()
            self._teacher_advice.setText("AI 建议：暂无实时数据")
            self._refresh_table([])
            return
        elapsed = int(snapshot.get("elapsed_seconds") or 0)
        state_text = CLASS_DISPLAY.get(snapshot.get("stable_state"), "暂无有效分析")
        baseline_text = {"COMPLETED": "已完成", "COLLECTING": "采集中",
                         "FAILED": "采集失败", "EARLY_STOPPED": "未形成有效基线",
                         "IDLE": "待完成"}.get(snapshot.get("baseline_status"), "待完成")
        quality_text = {"trusted": "接触良好", "warning": "建议调整佩戴",
                        "rejected": "当前信号不可解释"}.get(
                            snapshot.get("quality_level"), "等待信号")
        demo_text = " · 教学演示 / 演示数据" if snapshot.get("data_mode") == "demo" else ""
        self._teacher_trend.setText(
            f"{snapshot.get('student_name') or student_id} {student_id}{demo_text}\n"
            f"基线：{baseline_text} · 学习进度：{'进行中' if active else '未进行'}\n"
            f"任务：{snapshot.get('task_name') or '无'} · "
            f"有效时长：{elapsed // 60:02d}:{elapsed % 60:02d}\n"
            f"接触质量：{quality_text} · 当前表现：{state_text} · "
            f"专注/放松：{snapshot.get('attention')}/{snapshot.get('meditation')}"
        )
        self._teacher_alert.setText(
            "异常提醒：暂无" if active else "当前学生没有进行中的学习任务"
        )
        self._teacher_advice.setText(
            f"AI 建议：{snapshot.get('advice') or '暂无有效建议'}"
        )
        self._teacher_trend_plot.reset()
        for attention, meditation in zip(
            snapshot.get("attention_history") or [],
            snapshot.get("meditation_history") or [],
        ):
            self._teacher_trend_plot.push_values(attention, meditation)
        self._refresh_table(snapshot.get("recent_events") or [])

    def _refresh_student_assignments(self):
        student_id = getattr(self.state, "_user_id", "")
        assignments = [item for item in self.assignment_store.for_student(student_id)
                       if item.status in {"pending", "running", "paused"}]
        current = self._combo_assignment.currentData() or self._selected_assignment_id
        ids = [item.assignment_id for item in assignments]
        existing = [self._combo_assignment.itemData(index)
                    for index in range(self._combo_assignment.count())]
        if ids != existing:
            self._combo_assignment.blockSignals(True)
            self._combo_assignment.clear()
            for item in assignments:
                status = {
                    "pending": "待开始", "running": "进行中", "paused": "已暂停"
                }[item.status]
                self._combo_assignment.addItem(f"{item.task_type} · {status}", item.assignment_id)
            index = self._combo_assignment.findData(current)
            self._combo_assignment.setCurrentIndex(index if index >= 0 else (0 if assignments else -1))
            self._combo_assignment.blockSignals(False)
        if not assignments:
            self._selected_assignment_id = ""
            self._assignment_summary.setText("暂无待执行的教师布置任务")
            return
        selected = self._combo_assignment.currentData()
        item = next((value for value in assignments if value.assignment_id == selected), assignments[0])
        self._selected_assignment_id = item.assignment_id
        self.state.task_type = item.task_type
        self.state.task_difficulty = item.difficulty
        self._input_note.setText(item.note)
        status_text = {
            "pending": "待开始", "running": "进行中", "paused": "已暂停"
        }.get(item.status, item.status)
        self._assignment_summary.setText(
            f"教师布置任务\n任务：{item.task_type} · 难度：{DIFFICULTY_DISPLAY.get(item.difficulty, item.difficulty)}\n"
            f"备注：{item.note or '无'} · 发布教师：{item.teacher_id} · 状态：{status_text}"
        )
        self._sync_combos_from_state()

    def _on_student_assignment_changed(self, index):
        if self._role == "student" and not self._task_active:
            self._refresh_student_assignments()

    def _publish_teacher_task(self):
        """Store a local teacher assignment draft without touching live Task state."""
        if self._role != "teacher":
            return
        student_id = self._combo_student.currentData()
        if not student_id:
            QMessageBox.warning(self, "无法发布", "发布任务前必须选择学生。")
            return
        item = self.assignment_store.publish(
            getattr(self.state, "_user_id", ""), student_id,
            self._combo_task.currentText(),
            _INDEX_DIFFICULTY.get(self._combo_diff.currentIndex(), DIFFICULTY_MEDIUM),
            self._input_note.text().strip(),
        )
        self._teacher_assignment = {
            "assignment_id": item.assignment_id, "student_id": item.student_id,
            "name": item.task_type, "difficulty": item.difficulty,
            "notes": item.note, "created_at": item.created_at,
            "status": "published",
        }
        self._teacher_form_dirty = False
        self._show_published_assignment(item)

    def _show_published_assignment(self, item):
        created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item.created_at))
        status = {"pending": "等待学生开始", "running": "进行中", "paused": "已暂停",
                  "completed": "已完成", "cancelled": "已取消"}.get(item.status, item.status)
        self._assignment_summary.setText(
            f"已发布\n学生：{item.student_id}\n任务类型：{item.task_type}\n"
            f"难度：{DIFFICULTY_DISPLAY.get(item.difficulty, item.difficulty)}\n"
            f"备注：{item.note or '无'}\n发布时间：{created}\n任务状态：{status}"
        )

    def _session_is_active(self) -> bool:
        return bool(getattr(
            self.state, "session_active",
            getattr(self.state, "_session_active", False),
        ))

    def _on_start_session(self):
        """从任务页显式建立监测会话，解除隐含跨页前置条件。"""
        if self._role == "teacher":
            return
        if self._session_is_active():
            return
        # An explicit session may contain multiple tasks and must be closed by
        # the user.  Only the task auto-start path sets ownership to True.
        self._task_owns_session = False
        self.state.reset_session()
        event_count = len(getattr(self.state, "_events", []))
        self.service.start_session()
        if not self._session_is_active():
            QMessageBox.warning(self, "会话启动失败", "未能创建监测会话，请查看系统诊断。")
            return
        if len(getattr(self.state, "_events", [])) == event_count:
            self.state.add_event("会话开始", "system")
        self._session_relation.setText("监测会话：进行中")
        self._session_relation.setObjectName("GoodLabel")
        self._session_relation.setStyleSheet("color: #4ADE80; font-size: 12px;")
        self._btn_start_session.setEnabled(False)
        self.state.emit_update()

    def _on_task_start(self):
        if self._role == "teacher":
            return
        reason = learning_start_block_reason(self.state) if self._role == "student" else ""
        if reason and self.isVisible():
            QMessageBox.warning(self, "暂不能开始学习", reason)
            return
        created_for_task = False
        if not self._session_is_active():
            # One student action owns the whole technical lifecycle.
            self._on_start_session()
            if not self._session_is_active():
                return
            created_for_task = True
        self._task_owns_session = created_for_task
        self._task_start = time.time()
        self._task_active = True
        self.state.task_running = True
        self._btn_task_start.setEnabled(False)
        self._btn_task_stop.setEnabled(True)
        self._btn_task_pause.setEnabled(True)
        self._btn_task_pause.setText("暂停监测" if self._role == "research" else "暂停学习")
        self._task_paused = False
        self._task_status.setText("进行中")
        self._task_status.setStyleSheet("color: #4ADE80; font-size: 14px;")
        task_name = self._combo_task.currentText()
        if hasattr(self.state, "begin_task"):
            self.state.begin_task(
                task_name, getattr(self.state, "task_difficulty", DIFFICULTY_MEDIUM),
                self._input_note.text(),
                assignment_id=self._selected_assignment_id,
            )
        else:
            self._record_event(
                f"开始任务：{task_name}", "system", self._input_note.text(), scope="task"
            )
        if self._selected_assignment_id:
            self.assignment_store.update_status(self._selected_assignment_id, "running")
        self.runtime_registry.publish(self.state, force=True)

    def _on_task_stop(self):
        if self._role == "teacher":
            return
        feedback = self._collect_post_task_feedback()
        if feedback:
            self.submit_self_report(feedback)
        close_owned_session = self._task_owns_session
        self._task_owns_session = False
        self._task_active = False
        self.state.task_running = False
        self._btn_task_start.setEnabled(True)
        self._btn_task_stop.setEnabled(False)
        self._btn_task_pause.setEnabled(False)
        self._btn_task_pause.setText("暂停监测" if self._role == "research" else "暂停学习")
        self._task_paused = False
        self._task_status.setText("已结束")
        self._task_status.setStyleSheet("color: #6B7689; font-size: 14px;")
        if hasattr(self.state, "end_task"):
            self.state.end_task()
        else:
            self._record_event("结束任务", "system", scope="task")
        if close_owned_session and self._session_is_active():
            self.service.end_session(status="completed")
        if self._selected_assignment_id:
            self.assignment_store.update_status(self._selected_assignment_id, "completed")
        self.runtime_registry.publish(self.state, force=True)
        self.state.emit_update()

    def _collect_post_task_feedback(self) -> dict:
        """Collect one short retrospective report, never during EEG learning."""
        if self._role != "student" or not self.isVisible():
            return {}
        dialog = QDialog(self)
        dialog.setWindowTitle("学习反馈（可选）")
        dialog.setStyleSheet(self.FEEDBACK_DIALOG_STYLE)
        form = QFormLayout(dialog)
        fields = {}
        choices = {
            "difficulty": ("主观难度", ["较低", "适中", "较高"]),
            "mind_wandering": ("主观走神", ["很少", "偶尔", "较多"]),
            "fatigue": ("疲劳程度", ["较低", "适中", "较高"]),
            "emotion_stress": ("情绪 / 压力", ["轻松", "一般", "较高"]),
        }
        for key, (label, options) in choices.items():
            combo = QComboBox()
            combo.addItems(options)
            combo.setCurrentIndex(1)
            fields[key] = combo
            form.addRow(label, combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("提交并结束")
        buttons.button(QDialogButtonBox.Cancel).setText("跳过")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.Accepted:
            return {}
        return {key: combo.currentText() for key, combo in fields.items()}

    def submit_self_report(self, feedback: dict) -> None:
        """Persist post-task subjective feedback as one typed event."""
        if not feedback or not self._session_is_active():
            return
        content = "；".join(f"{key}={value}" for key, value in feedback.items())
        self.state.add_event(
            "学习反馈", "self_report", content,
            source="self_report", event_type="self_report",
            task_id=(getattr(self.state, "_current_task_id", "") or ""),
            student_id=getattr(self.state, "_user_id", ""),
        )

    def _on_task_pause_resume(self):
        """Pause effective task time through the existing Session service."""
        if self._role == "teacher" or not self._task_active:
            return
        if self._task_paused:
            self.service.resume_session()
            self.state.add_event(
                "学习恢复", "system", source="system", event_type="resume"
            )
            self._task_paused = False
            self._btn_task_pause.setText("暂停监测" if self._role == "research" else "暂停学习")
            self._task_status.setText("进行中")
            if self._selected_assignment_id:
                self.assignment_store.update_status(self._selected_assignment_id, "running")
        else:
            self.service.pause_session()
            self.state.add_event(
                "学习暂停", "system", source="system", event_type="pause"
            )
            self._task_paused = True
            self._btn_task_pause.setText("恢复监测" if self._role == "research" else "恢复学习")
            self._task_status.setText("已暂停")
            if self._selected_assignment_id:
                self.assignment_store.update_status(self._selected_assignment_id, "paused")
        self.runtime_registry.publish(self.state, force=True)
        self.state.emit_update()

    # ── 用户改动 ComboBox → 写回 DashboardState ──
    # 这是 UI 与 state 保持单一真相来源的关键。
    # 用 _syncing_combo 防止 _sync_combos_from_state 回流触发本回调。
    def _on_user_changed_task(self, idx: int):
        if self._syncing_combo:
            return
        text = self._combo_task.itemText(idx)
        if self._role == "teacher":
            self._teacher_form_dirty = True
            return
        if text and self.state.task_type != text:
            self.state.task_type = text
            if self._session_is_active():
                self._record_event(
                    f"切换任务类型：{text}", "system", scope="session", source="system"
                )

    def _on_user_changed_difficulty(self, idx: int):
        if self._syncing_combo:
            return
        if self._role == "teacher":
            self._teacher_form_dirty = True
            return
        new_diff = _INDEX_DIFFICULTY.get(idx, DIFFICULTY_MEDIUM)
        if self.state.task_difficulty != new_diff:
            old_display = DIFFICULTY_DISPLAY.get(self.state.task_difficulty, "--")
            new_display = DIFFICULTY_DISPLAY.get(new_diff, "--")
            self.state.task_difficulty = new_diff
            if self._session_is_active():
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
        if self._role == "teacher":
            return
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
        self, label: str, category: str, note: str = "", scope: str | None = None,
        source: str | None = None,
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
        elif label.startswith("切换任务"):
            event_type = "task_switch"
        else:
            event_type = category
        candidates = {
            "source": source or getattr(self.state, "mode", "live"),
            "event_type": event_type,
            "scope": scope,
            "session_id": getattr(self.state, "run_id", ""),
            "task_id": (
                getattr(self.state, "_current_task_id", "")
                or getattr(self.state, "current_task_id", "")
            ) if scope == "task" else "",
            "observer_id": (
                getattr(self.state, "_user_id", "") if source == "teacher" else ""
            ),
            "student_id": (
                getattr(getattr(self.state, "_active_record", None), "user_id", "")
                if source == "teacher" else getattr(self.state, "_user_id", "")
            ),
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
        if self._role == "teacher":
            snapshot = self.runtime_registry.get(self.selection_context.selected_student_id)
            return bool(
                snapshot and not snapshot.get("stale")
                and snapshot.get("session_id") and snapshot.get("task_id")
            )
        if self._session_is_active():
            return True
        QMessageBox.information(
            self,
            "没有可观察的学习任务" if self._role == "teacher" else "尚未开始学习",
            "当前没有学生正在执行的任务，无法添加教师观察。"
            if self._role == "teacher" else "请先点击“开始学习”。",
        )
        return False

    def _add_quick_event(self, label: str, category: str = "manual"):
        if self._ensure_event_session():
            teacher = self._role == "teacher"
            if teacher:
                self.runtime_registry.add_teacher_event(
                    self.selection_context.selected_student_id,
                    getattr(self.state, "_user_id", ""), label
                )
                self._refresh_teacher_observation()
                return
            self._record_event(
                label, "teacher" if teacher else category,
                source="teacher" if teacher else None,
            )

    def _add_custom_event(self):
        text = self._input_custom.text().strip()
        if text and self._ensure_event_session():
            teacher = self._role == "teacher"
            if teacher:
                self.runtime_registry.add_teacher_event(
                    self.selection_context.selected_student_id,
                    getattr(self.state, "_user_id", ""),
                    text,
                )
                self._input_custom.clear()
                self._refresh_teacher_observation()
                return
            diagnostic = self._role == "research"
            self._record_event(
                text, "teacher" if teacher else ("system" if diagnostic else "manual"),
                source="teacher" if teacher else ("system" if diagnostic else None),
            )
            self._input_custom.clear()

    def _clear_events(self):
        if self._role == "teacher":
            return
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

    def _refresh_table(self, events=None):
        if events is None:
            events = self.state._events if self._role != "teacher" else []
        self._event_table.setRowCount(len(events))
        for i, ev in enumerate(events):
            from datetime import datetime
            value = ev.get("timestamp") if isinstance(ev, dict) else ev.timestamp
            ts = datetime.fromtimestamp(float(value)).strftime("%H:%M:%S")
            self._event_table.setItem(i, 0, QTableWidgetItem(ts))
            source = ev.get("source", "") if isinstance(ev, dict) else getattr(ev, "source", "")
            cat_item = QTableWidgetItem(EVENT_SOURCE_DISPLAY.get(source, source or "未知"))
            cat_item.setForeground(Qt.GlobalColor.white)
            self._event_table.setItem(i, 1, cat_item)
            task_id = ev.get("task_id", "") if isinstance(ev, dict) else getattr(ev, "task_id", "")
            session_id = (ev.get("session_id", "") if isinstance(ev, dict)
                          else getattr(ev, "session_id", "")) or getattr(self.state, "run_id", "--")
            scope = f"任务：{task_id}" if task_id else f"会话：{session_id}"
            if not scope:
                scope = self._event_scopes.get(
                    getattr(ev, "timestamp", id(ev)),
                    self._event_scope_text("task" if self._task_active else "session"),
                )
            self._event_table.setItem(i, 2, QTableWidgetItem(str(scope)))
            label = ev.get("label", ev.get("content", "")) if isinstance(ev, dict) else ev.label
            note = ev.get("note", "") if isinstance(ev, dict) else ev.note
            self._event_table.setItem(i, 3, QTableWidgetItem(label))
            self._event_table.setItem(i, 4, QTableWidgetItem(note))
        self._event_table.scrollToBottom()

    def update_state(self, state):
        requested_role = self._normalize_role(getattr(state, "current_role", self._role))
        if requested_role != self._role:
            self.set_role(requested_role)
        if self._role == "teacher":
            self._refresh_teacher_observation()
            return
        if self._role == "student":
            if not self._task_active:
                self._refresh_student_assignments()
            self.runtime_registry.publish(state)

        session_active = self._session_is_active()
        self._btn_start_session.setEnabled(self._role != "teacher" and not session_active)
        if self._role == "student" and not session_active and self.isVisible():
            reason = learning_start_block_reason(state, require_baseline=True)
            # Keep clickable so the guard can explain why start is unavailable.
            self._btn_task_start.setEnabled(True)
            self._btn_task_start.setToolTip(reason or "开始当前学习任务")
        relation_name = "本机监测" if self._role == "research" else "监测会话"
        self._session_relation.setText(
            f"{relation_name}：{'进行中' if session_active else '未开始'}"
        )
        self._session_relation.setStyleSheet(
            f"color: {'#4ADE80' if session_active else '#FBBF24'}; font-size: 12px;"
        )
        if self._role != "teacher":
            for button in self._event_buttons:
                button.setEnabled(session_active)

        # 任务计时
        if self._task_active:
            elapsed = float(getattr(
                state, "current_task_elapsed_seconds",
                time.time() - self._task_start,
            ))
            mins = int(elapsed) // 60
            secs = int(elapsed) % 60
            self._task_time.setText(f"{mins:02d}:{secs:02d}")
            if getattr(state, "session_paused", False):
                self._task_status.setText("已暂停")
                self._task_status.setStyleSheet("color: #FBBF24; font-size: 14px;")
            else:
                self._task_status.setText("进行中")
                self._task_status.setStyleSheet("color: #4ADE80; font-size: 14px;")

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
            self._refresh_teacher_observation()
            stable = CLASS_DISPLAY.get(getattr(state, "stable_state", None), "暂无有效趋势")
            trend = getattr(state, "class_trend_summary", "") or f"当前监测：{stable}"
            alerts = getattr(state, "class_alerts", None)
            if alerts:
                alert_text = f"{len(alerts)} 条待关注"
            elif getattr(state, "quality_level", "rejected") == "rejected":
                alert_text = "当前信号质量待处理"
            else:
                alert_text = "暂无异常提醒"
            if not self._combo_student.currentData():
                self._teacher_trend.setText(f"班级状态趋势：{trend}")
                self._teacher_alert.setText(f"异常提醒：{alert_text}")
            assignment = (
                self._teacher_assignment.get("name", "")
                if self._teacher_assignment else "暂无已发布任务"
            )
            execution = "学生执行中" if state.task_running else "学生未执行"
            self._teacher_process.setText(
                f"已发布：{assignment} · {execution} · "
                f"过程记录 {len(getattr(state, '_events', []))} 条"
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
