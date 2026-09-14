"""页面3：实时学习仪表盘（核心页面）。

严格遵循 eeg_modular/ui_prototype/services/dashboard_state.py 中定义的
DashboardState 正式字段接口。UI 业务逻辑只消费正式字段，
内部簿记字段（_前缀）仅用于图表缓冲绘制。
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QInputDialog,
    QMessageBox, QFileDialog, QComboBox, QGridLayout, QFrame,
    QDialog, QListWidget, QLineEdit,
)

from pages.base_page import BasePage
from widgets.card import Card
from widgets.status_indicator import StatusIndicator
from widgets.probability_bar import ProbabilityPanel
from widgets.eeg_plot import EEGPlotWidget
from widgets.trend_plot import TrendPlotWidget, ProbabilityTrendWidget
from widgets.gauge import ArcGauge
from services.dashboard_state import (
    WARMUP_SECONDS, MAX_POOR_SIGNAL, CLASS_DISPLAY,
    MOCK_UI_REFRESH_HZ, DEVICE_TARGET_SAMPLE_HZ,
    DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD,
    DIFFICULTY_DISPLAY,
)
from services.learning_readiness import learning_start_block_reason
from services.identity_store import IdentityStore
from services.teaching_store import (
    StudentRuntimeRegistry, TeacherSelectionContext, TeacherStudentStore,
)


LEARNER_TASK_TYPES = [
    "数学练习", "英语阅读", "编程任务", "物理复习", "语文写作", "自由学习",
]

_DIFFICULTY_VALUES = [DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD]


class DashboardPage(BasePage):
    MANAGE_STUDENTS_DIALOG_STYLE = (
        "QDialog { background: #161D2A; color: #E8EDF3; }"
        "QLabel { color: #E8EDF3; font-size: 13px; }"
        "QListWidget, QLineEdit { background: #222B3A; color: #F3F6FA; "
        "border: 1px solid #3B475B; border-radius: 4px; padding: 6px; }"
        "QLineEdit { selection-background-color: #2563EB; }"
        "QListWidget::item:selected { background: #2563EB; color: white; }"
        "QPushButton { background: #263449; color: #E8EDF3; border: 1px solid #41516A; "
        "border-radius: 4px; padding: 6px 12px; }"
        "QPushButton:hover { background: #31425C; }"
    )
    def __init__(self, state, service, *, binding_store=None, runtime_registry=None,
                 identity_store=None, selection_context=None):
        self.state = state
        self.service = service
        self.identity_store = identity_store or IdentityStore()
        self.binding_store = binding_store or TeacherStudentStore(
            identity_store=self.identity_store
        )
        self.runtime_registry = runtime_registry or StudentRuntimeRegistry.shared()
        self.selection_context = selection_context or TeacherSelectionContext.shared()
        self._session_started = False
        self._session_paused = False
        self._syncing_role_controls = False
        super().__init__(scrollable=True)
        self._teacher_poll_timer = QTimer(self)
        self._teacher_poll_timer.setInterval(500)
        self._teacher_poll_timer.timeout.connect(self._poll_teacher_runtime)
        self._build_ui()
        self.set_role(self._role)

    def _build_ui(self):
        layout = self.content_layout
        layout.setSpacing(8)

        # ── 标题行 ──
        header = QHBoxLayout()
        title = QLabel("实时学习仪表盘")
        self._page_title = title
        title.setObjectName("PageTitle")
        header.addWidget(title)

        self._teacher_student_label = QLabel("当前观察学生：")
        self._teacher_student_combo = QComboBox()
        self._teacher_student_combo.currentIndexChanged.connect(
            self._on_teacher_student_changed
        )
        self._btn_manage_students = QPushButton("管理学生")
        self._btn_manage_students.clicked.connect(self._manage_students)
        header.addWidget(self._teacher_student_label)
        header.addWidget(self._teacher_student_combo)
        header.addWidget(self._btn_manage_students)

        self._session_time = QLabel("会话时间 00:00")
        self._session_time.setObjectName("AccentLabel")
        self._session_time.setStyleSheet("font-size: 18px; font-weight: bold;")
        self._session_time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header.addStretch()
        header.addWidget(self._session_time)
        layout.addLayout(header)
        self._diagnostic_note = QLabel(
            "本机 EEG 数据仅用于设备与模型诊断，不计入任何学生学习记录。"
        )
        self._diagnostic_note.setStyleSheet("color: #FBBF24; font-size: 12px;")
        self._diagnostic_note.setVisible(False)
        layout.addWidget(self._diagnostic_note)

        # ── 第一行：5个状态卡片 ──
        status_row = QHBoxLayout()
        status_row.setSpacing(10)

        self._card_connector = self._make_status_card("ThinkGear Connector")
        status_row.addWidget(self._card_connector["frame"], 1)

        self._card_device = self._make_status_card("MindWave 设备")
        status_row.addWidget(self._card_device["frame"], 1)

        self._card_poor = self._make_status_card("Poor Signal")
        status_row.addWidget(self._card_poor["frame"], 1)

        self._card_conf = self._make_status_card("信号质量")
        status_row.addWidget(self._card_conf["frame"], 1)

        self._card_rate = self._make_status_card("采样率")
        status_row.addWidget(self._card_rate["frame"], 1)

        layout.addLayout(status_row)

        # ── 第二行：预热进度 ──
        warmup_card = Card(f"预热阶段（{WARMUP_SECONDS:.0f}秒）")
        warmup_h = QHBoxLayout()
        warmup_h.setSpacing(12)

        self._warmup_bar = QProgressBar()
        self._warmup_bar.setObjectName("WarmupBar")
        self._warmup_bar.setRange(0, 100)
        self._warmup_bar.setFixedHeight(20)
        warmup_h.addWidget(self._warmup_bar, 1)

        self._warmup_label = QLabel(f"0.0s / {WARMUP_SECONDS:.0f}s")
        self._warmup_label.setObjectName("DimLabel")
        self._warmup_label.setStyleSheet("font-size: 13px;")
        self._warmup_label.setFixedWidth(100)
        warmup_h.addWidget(self._warmup_label)

        warmup_card.add_widget(self._wrap(warmup_h))
        warmup_card.setMaximumHeight(64)
        layout.addWidget(warmup_card)

        # 分离“等待数据 / 预热 / 拒识 / 模型故障”，避免都表现为空概率。
        self._analysis_card = Card("分析状态")
        analysis_row = QHBoxLayout()
        analysis_row.setSpacing(10)
        self._analysis_label = QLabel("等待设备数据")
        self._analysis_label.setObjectName("DimLabel")
        self._analysis_label.setStyleSheet("font-size: 13px; font-weight: 600;")
        analysis_row.addWidget(self._analysis_label, 1)
        self._analysis_hint = QLabel("连接设备并收到原始脑电后开始预热。")
        self._analysis_hint.setWordWrap(True)
        self._analysis_hint.setStyleSheet("color: #8491A5; font-size: 12px;")
        analysis_row.addWidget(self._analysis_hint, 3)
        self._btn_diagnostics = QPushButton("查看系统诊断")
        self._btn_diagnostics.setVisible(False)
        self._btn_diagnostics.clicked.connect(self._open_diagnostics)
        analysis_row.addWidget(self._btn_diagnostics)
        self._analysis_card.add_widget(self._wrap(analysis_row))
        self._analysis_card.setMaximumHeight(72)
        layout.addWidget(self._analysis_card)

        # ── 第三行：左 EEG + 趋势 | 右 概率 + 持续状态 ──
        main_row = QHBoxLayout()
        main_row.setSpacing(10)

        # 左列
        left_col = QVBoxLayout()
        left_col.setSpacing(10)

        eeg_card = Card("EEG实时曲线")
        self._eeg_plot = EEGPlotWidget()
        self._eeg_plot.setMinimumHeight(100)
        eeg_card.add_widget(self._eeg_plot)
        left_col.addWidget(eeg_card, 1)

        trend_card = Card("专注度 / 放松度趋势（90秒）")
        self._trend_plot = TrendPlotWidget()
        self._trend_plot.setMinimumHeight(100)
        trend_card.add_widget(self._trend_plot)
        left_col.addWidget(trend_card, 1)

        main_row.addLayout(left_col, 3)

        # 右列
        right_col = QVBoxLayout()
        right_col.setSpacing(10)

        # 概率面板
        prob_card = Card("模型三分类概率")
        self._prob_panel = ProbabilityPanel()
        prob_card.add_widget(self._prob_panel)

        # 预测结果
        self._pred_label = QLabel("当前状态：--")
        self._pred_label.setObjectName("AccentLabel")
        self._pred_label.setStyleSheet("font-size: 16px; padding: 4px 0;")
        prob_card.add_widget(self._pred_label)

        right_col.addWidget(prob_card, 4)

        # Attention/Meditation 仪表
        gauge_row = QHBoxLayout()
        gauge_row.setSpacing(8)

        att_card = Card("专注度")
        self._att_gauge = ArcGauge("专注度", "#4FC3F7")
        self._att_gauge.setFixedSize(72, 72)
        att_card.add_widget(self._wrap_centered(self._att_gauge))
        gauge_row.addWidget(att_card, 1)

        med_card = Card("放松度")
        self._med_gauge = ArcGauge("放松度", "#4ADE80")
        self._med_gauge.setFixedSize(72, 72)
        med_card.add_widget(self._wrap_centered(self._med_gauge))
        gauge_row.addWidget(med_card, 1)

        right_col.addLayout(gauge_row, 3)

        # 持续状态
        sustain_card = Card("最近90秒稳定状态")
        sustain_layout = QVBoxLayout()
        sustain_layout.setSpacing(6)

        self._sustain_label = QLabel("主导状态：--")
        self._sustain_label.setObjectName("CardValueSmall")
        self._sustain_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        sustain_layout.addWidget(self._sustain_label)

        self._sustain_neg = QLabel("消极持续：0.0秒")
        self._sustain_neg.setStyleSheet("color: #F87171; font-size: 13px;")
        sustain_layout.addWidget(self._sustain_neg)

        self._intervention_label = QLabel()
        self._intervention_label.setStyleSheet(
            "color: #FBBF24; font-size: 12px; padding: 4px 8px; "
            "background-color: rgba(200,150,40,0.1); border-radius: 4px;"
        )
        self._intervention_label.setVisible(False)
        sustain_layout.addWidget(self._intervention_label)

        self._prob_trend = ProbabilityTrendWidget()
        self._prob_trend.setMinimumHeight(36)
        sustain_layout.addWidget(self._prob_trend)

        sustain_card.add_widget(self._wrap(sustain_layout))
        right_col.addWidget(sustain_card, 4)

        main_row.addLayout(right_col, 2)

        layout.addLayout(main_row, 1)

        # ── AI建议 ──
        ai_card = Card("AI学习建议")
        self._ai_label = QLabel("等待信号稳定后将生成学习建议。")
        self._ai_label.setWordWrap(True)
        self._ai_label.setStyleSheet("color: #C5CDD9; font-size: 14px;")
        ai_card.add_widget(self._ai_label)
        ai_card.setMaximumHeight(52)
        layout.addWidget(ai_card)

        # 角色业务重点：同一张轻量卡片按登录角色切换，不拆分页面架构。
        self._role_card = Card("角色工作区")
        role_grid = QGridLayout()
        role_grid.setHorizontalSpacing(14)
        role_grid.setVerticalSpacing(5)
        self._focus_keys = [QLabel() for _ in range(3)]
        self._focus_values = [QLabel("--") for _ in range(3)]
        for column, (key, value) in enumerate(zip(self._focus_keys, self._focus_values)):
            key.setStyleSheet("color: #8491A5; font-size: 12px;")
            value.setWordWrap(True)
            value.setStyleSheet("color: #E8EDF3; font-size: 13px; font-weight: 600;")
            role_grid.addWidget(key, 0, column)
            role_grid.addWidget(value, 1, column)

        self._learner_task = QComboBox()
        self._learner_task.addItems(LEARNER_TASK_TYPES)
        self._learner_task.currentTextChanged.connect(self._on_learner_task_changed)
        role_grid.addWidget(self._learner_task, 1, 0)

        self._learner_difficulty = QComboBox()
        self._learner_difficulty.addItems([
            DIFFICULTY_DISPLAY.get(level, level) for level in _DIFFICULTY_VALUES
        ])
        self._learner_difficulty.currentIndexChanged.connect(
            self._on_learner_difficulty_changed
        )
        role_grid.addWidget(self._learner_difficulty, 1, 2)

        self._role_card.add_widget(self._wrap(role_grid))
        self._role_card.setMaximumHeight(98)
        layout.addWidget(self._role_card)

        # ── 固定会话控制栏：放在滚动区域之外，始终可见 ──
        toolbar = QFrame()
        self._session_toolbar = toolbar
        toolbar.setObjectName("SessionToolbar")
        btn_row = QHBoxLayout(toolbar)
        btn_row.setContentsMargins(12, 8, 12, 8)
        btn_row.setSpacing(10)

        self._btn_start = QPushButton("开始学习记录")
        self._btn_start.setObjectName("PrimaryButton")
        self._btn_start.clicked.connect(self._on_start)
        btn_row.addWidget(self._btn_start)

        self._btn_pause = QPushButton("暂停")
        self._btn_pause.setEnabled(False)
        self._btn_pause.clicked.connect(self._on_pause)
        btn_row.addWidget(self._btn_pause)

        # Explicit parent is required because this compatibility button is not
        # inserted into the toolbar layout; otherwise it becomes a stray top-level window.
        self._btn_event = QPushButton("事件标记", toolbar)
        self._btn_event.setEnabled(False)
        self._btn_event.clicked.connect(self._on_event)
        # 详细事件统一在“任务与事件”页管理；保留属性兼容旧调用。
        self._btn_event.setVisible(False)

        self._btn_end = QPushButton("结束并保存")
        self._btn_end.setObjectName("DangerButton")
        self._btn_end.setEnabled(False)
        self._btn_end.clicked.connect(self._on_end)
        btn_row.addWidget(self._btn_end)

        self._btn_export = QPushButton("导出本次报告")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._on_export)
        btn_row.addWidget(self._btn_export)

        self._session_scope_hint = QLabel("学习记录包含任务、事件和分析结果")
        self._session_scope_hint.setStyleSheet("color: #8491A5; font-size: 12px;")
        btn_row.addStretch()
        btn_row.addWidget(self._session_scope_hint)
        self._main_layout.addWidget(toolbar, 0)

    # ── 辅助方法 ──

    def _make_status_card(self, title: str) -> dict:
        card = Card(title)
        # 标题、主状态和状态标签是三行内容。原先 72px 的上限小于
        # 三行文字与上下边距的实际高度，在 Windows 125%/150% 缩放下
        # 会把最后一行裁掉。给状态卡固定的完整高度，避免字体缩放时重叠。
        card.content_layout.setSpacing(4)
        val = QLabel("--")
        val.setObjectName("CardValueSmall")
        val.setMinimumHeight(24)
        ind = StatusIndicator()
        ind.setMinimumHeight(24)
        card.add_widget(val)
        card.add_widget(ind)
        card.setFixedHeight(96)
        return {"frame": card, "value": val, "indicator": ind}

    def _wrap(self, layout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        return w

    def _wrap_centered(self, widget) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        layout.addWidget(widget)
        layout.addStretch()
        return w

    def set_role(self, role: str):
        super().set_role(role)
        if not hasattr(self, "_role_card"):
            return
        is_student = self._role == "student"
        teacher = self._role == "teacher"
        self._teacher_student_label.setVisible(teacher)
        self._teacher_student_combo.setVisible(teacher)
        self._btn_manage_students.setVisible(teacher)
        self._learner_task.setVisible(is_student)
        self._learner_difficulty.setVisible(is_student)
        self._focus_values[0].setVisible(not is_student)
        self._focus_values[2].setVisible(not is_student)
        if is_student:
            self._page_title.setText("实时学习仪表盘")
            self._diagnostic_note.setVisible(False)
            self._card_poor["frame"].set_title("接触质量")
            self._btn_start.setText("开始学习记录")
            self._role_card.set_title("学生端 · 当前学习安排")
            titles = ["学习任务", "当前建议", "难度调整"]
        elif teacher:
            self._page_title.setText("学生实时观察")
            self._diagnostic_note.setVisible(False)
            self._card_poor["frame"].set_title("接触质量")
            self._role_card.set_title("教师端 · 学生过程概览")
            titles = ["班级状态趋势", "异常提醒", "过程记录"]
        else:
            self._page_title.setText("本机设备与模型诊断")
            self._diagnostic_note.setVisible(True)
            self._card_poor["frame"].set_title("Poor Signal 原始值")
            self._btn_start.setText("开始本机诊断记录")
            self._role_card.set_title("管理端 · 运行概览")
            titles = ["设备", "模型与数据质量", "实验记录"]
        can_control = self._role != "teacher"
        for button in (
            self._btn_start, self._btn_pause, self._btn_end,
        ):
            button.setVisible(can_control)
        self._btn_event.setVisible(False)
        for label, text in zip(self._focus_keys, titles):
            label.setText(text)
        self._refresh_role_focus(self.state)
        if teacher:
            self.selection_context.set_teacher(getattr(self.state, "_user_id", ""))
            self._teacher_poll_timer.start()
            self._refresh_teacher_students()
            self._update_teacher_snapshot()
        else:
            self._teacher_poll_timer.stop()

    def _poll_teacher_runtime(self):
        if self._role == "teacher":
            self._refresh_teacher_students()
            self._update_teacher_snapshot()

    def _refresh_teacher_students(self):
        teacher_id = getattr(self.state, "_user_id", "")
        current = (
            self.selection_context.selected_student_id
            or self._teacher_student_combo.currentData()
        )
        try:
            names = {item["user_id"]: item["name"] for item in self.identity_store.list_profiles()}
        except OSError:
            names = {}
        students = self.binding_store.students_for(teacher_id)
        existing = [self._teacher_student_combo.itemData(i)
                    for i in range(self._teacher_student_combo.count())]
        if [""] + students == existing:
            index = self._teacher_student_combo.findData(current)
            if index >= 0 and index != self._teacher_student_combo.currentIndex():
                self._teacher_student_combo.blockSignals(True)
                self._teacher_student_combo.setCurrentIndex(index)
                self._teacher_student_combo.blockSignals(False)
            return
        self._teacher_student_combo.blockSignals(True)
        self._teacher_student_combo.clear()
        self._teacher_student_combo.addItem("请选择学生", "")
        for student_id in students:
            self._teacher_student_combo.addItem(
                f"{names.get(student_id, '')} {student_id}".strip(), student_id
            )
        index = self._teacher_student_combo.findData(current)
        self._teacher_student_combo.setCurrentIndex(max(0, index))
        self._teacher_student_combo.blockSignals(False)

    def _on_teacher_student_changed(self, index):
        if self._role == "teacher":
            self.selection_context.select(self._teacher_student_combo.currentData() or "")
            self._update_teacher_snapshot()

    def _manage_students(self):
        if self._role != "teacher":
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("管理学生")
        dialog.setStyleSheet(self.MANAGE_STUDENTS_DIALOG_STYLE)
        layout = QVBoxLayout(dialog)
        students = QListWidget()
        layout.addWidget(QLabel("我的学生"))
        layout.addWidget(students)
        entry = QLineEdit()
        entry.setPlaceholderText("学生 ID，例如 st_001")
        layout.addWidget(entry)
        buttons = QHBoxLayout()
        add_button = QPushButton("添加学生")
        remove_button = QPushButton("移除选中绑定")
        close_button = QPushButton("关闭")
        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        buttons.addStretch()
        buttons.addWidget(close_button)
        layout.addLayout(buttons)
        teacher_id = getattr(self.state, "_user_id", "")

        def refresh():
            students.clear()
            try:
                names = {item["user_id"]: item["name"] for item in self.identity_store.list_profiles()}
            except OSError:
                names = {}
            for student_id in self.binding_store.students_for(teacher_id):
                students.addItem(f"{names.get(student_id, '')} {student_id}".strip())

        def add():
            student_id = entry.text().strip()
            try:
                added = self.binding_store.add(teacher_id, student_id)
            except ValueError as exc:
                QMessageBox.warning(dialog, "无法添加学生", str(exc))
                return
            if not added:
                QMessageBox.information(dialog, "提示", "该学生已经添加")
                return
            entry.clear()
            refresh()
            self._refresh_teacher_students()

        def remove():
            item = students.currentItem()
            if not item:
                return
            student_id = item.text().split()[-1]
            self.binding_store.remove(teacher_id, student_id)
            refresh()
            self._refresh_teacher_students()

        add_button.clicked.connect(add)
        remove_button.clicked.connect(remove)
        close_button.clicked.connect(dialog.accept)
        refresh()
        dialog.exec()

    def _update_teacher_snapshot(self):
        student_id = self.selection_context.selected_student_id
        snapshot = self.runtime_registry.get(student_id) if student_id else None
        self._teacher_snapshot = snapshot
        if not snapshot or snapshot.get("stale"):
            self._analysis_label.setText(
                "学生端未运行或实时数据已中断"
                if snapshot else "暂无实时数据"
            )
            self._analysis_hint.setText(
                "最后状态已超时，请确认学生端程序仍在运行。"
                if snapshot else "请选择已绑定且正在运行学生端的账号。"
            )
            self._session_time.setText("有效学习 00:00")
            for card in (self._card_connector, self._card_device, self._card_poor,
                         self._card_conf, self._card_rate):
                card["value"].setText("--")
                card["indicator"].set_state(StatusIndicator.LEVEL_NEUTRAL, "暂无数据")
            self._att_gauge.set_value(0)
            self._med_gauge.set_value(0)
            self._pred_label.setText("当前状态：--")
            self._ai_label.setText("暂无学生实时建议")
            self._eeg_plot.reset()
            return
        elapsed = int(snapshot.get("elapsed_seconds") or 0)
        demo = snapshot.get("data_mode") == "demo"
        self._session_time.setText(f"有效学习 {elapsed // 60:02d}:{elapsed % 60:02d}")
        online = bool(snapshot.get("online"))
        self._card_connector["value"].setText("学生在线" if online else "学生离线")
        self._card_device["value"].setText("在线" if online else "离线")
        self._card_poor["value"].setText("--")
        quality = {"trusted": "接触良好", "warning": "建议调整佩戴",
                   "rejected": "当前信号不可解释"}.get(snapshot.get("quality_level"), "等待信号")
        self._card_conf["value"].setText(quality)
        self._card_rate["value"].setText("教学演示 · 演示数据" if demo else "学生端实时采集")
        self._analysis_label.setText(
            f"{snapshot.get('student_name') or student_id} · {snapshot.get('task_name') or '当前无任务'}"
            + (" · 教学演示" if demo else "")
        )
        self._analysis_hint.setText(
            f"基线：{'已完成' if snapshot.get('baseline_status') == 'COMPLETED' else '待完成'} · "
            f"学习进度：{'进行中' if snapshot.get('task_id') else '当前无任务'}"
        )
        self._att_gauge.set_value(snapshot.get("attention") or 0)
        self._med_gauge.set_value(snapshot.get("meditation") or 0)
        stable = CLASS_DISPLAY.get(snapshot.get("stable_state"), "--")
        if snapshot.get("quality_level") == "rejected":
            self._pred_label.setText("当前状态：当前信号暂不可解释")
        else:
            self._pred_label.setText(f"当前状态：{stable}")
        self._ai_label.setText(snapshot.get("advice") or "暂无学生实时建议")
        probabilities = snapshot.get("probabilities") or {}
        for name, bar in self._prob_panel._bars.items():
            value = probabilities.get(name)
            bar.set_value(value or 0.0)
            bar.set_dimmed(
                value is None or snapshot.get("quality_level") == "rejected"
            )
        self._prob_panel._confidence_label.setText(
            f"接触质量：{quality}"
        )
        raw = snapshot.get("raw_eeg") or []
        self._eeg_plot.reset()
        if raw:
            self._eeg_plot.push_display_points(raw)
        att_history = snapshot.get("attention_history") or []
        med_history = snapshot.get("meditation_history") or []
        self._trend_plot.reset()
        for att, med in zip(att_history, med_history):
            self._trend_plot.push_values(att, med)
        updated = snapshot.get("updated_at") or 0
        self._focus_values[0].setText(f"学生：{student_id} · 状态：{stable}")
        self._focus_values[1].setText(
            "最后更新：" + time.strftime("%H:%M:%S", time.localtime(updated))
        )
        self._focus_values[2].setText(
            f"智能建议：{snapshot.get('advice') or '暂无建议'}"
        )

    def _on_learner_task_changed(self, text: str):
        if self._syncing_role_controls or self._role != "student" or not text:
            return
        old = getattr(self.state, "task_type", "")
        if old == text:
            return
        self.state.task_type = text
        self.state.add_event(f"选择学习任务：{text}", "manual")

    def _on_learner_difficulty_changed(self, index: int):
        if self._syncing_role_controls or self._role != "student":
            return
        if not 0 <= index < len(_DIFFICULTY_VALUES):
            return
        level = _DIFFICULTY_VALUES[index]
        if getattr(self.state, "task_difficulty", DIFFICULTY_MEDIUM) == level:
            return
        self.state.task_difficulty = level
        display = DIFFICULTY_DISPLAY.get(level, level)
        self.state.add_event(f"调整学习难度：{display}", "manual")

    def _sync_learner_controls(self, state):
        self._syncing_role_controls = True
        try:
            task = getattr(state, "task_type", "自由学习")
            if task in LEARNER_TASK_TYPES:
                self._learner_task.setCurrentIndex(LEARNER_TASK_TYPES.index(task))
            difficulty = getattr(state, "task_difficulty", DIFFICULTY_MEDIUM)
            if difficulty in _DIFFICULTY_VALUES:
                self._learner_difficulty.setCurrentIndex(
                    _DIFFICULTY_VALUES.index(difficulty)
                )
        finally:
            self._syncing_role_controls = False

    def _open_diagnostics(self):
        window = self.window()
        if hasattr(window, "_navigate_to"):
            window._navigate_to("settings")

    @staticmethod
    def _public_quality_reasons(reasons) -> list[str]:
        """过滤开发路径、哈希和内部文件名，避免普通界面泄露诊断细节。"""
        result = []
        blocked = (
            "checksum", "hash", "traceback", "exception", ".md", ".py",
            "package", "baseline_card", "\\", "/",
        )
        for reason in reasons or []:
            text = str(reason).strip()
            if text and not any(token in text.lower() for token in blocked):
                result.append(text)
        return result

    def _analysis_view(self, state) -> tuple[str, str, str]:
        """返回 kind/title/hint；优先消费统一 pipeline_state。"""
        pipeline = str(getattr(state, "pipeline_state", "") or "").lower()
        model_status = str(getattr(state, "model_status", "") or "").upper()
        if pipeline == "error" or model_status == "FAILED":
            friendly = str(
                getattr(state, "model_error_user", "")
                or "智能分析暂不可用"
            )
            return "error", friendly, "采集可继续；如问题持续，请联系管理员检查系统配置。"
        if pipeline == "waiting_data":
            return "waiting", "等待设备数据", "收到首个原始脑电数据后才会开始预热。"
        if pipeline == "warming_up":
            remaining = max(
                0, int(round((1.0 - float(getattr(state, "warmup_progress", 0.0))) * WARMUP_SECONDS))
            )
            return "warming", "正在积累分析数据", f"距离首个分析窗口约 {remaining} 秒。"
        if pipeline == "rejected":
            reasons = self._public_quality_reasons(getattr(state, "quality_reasons", []))
            low_confidence = any(
                token in " ".join(reasons).lower()
                for token in ("置信", "confidence", "ood", "不确定")
            )
            if low_confidence:
                return "rejected", "本窗口置信度不足", "结果已拒识，不计入状态趋势。"
            return "rejected", "当前信号暂不可解释", "请调整佩戴并保持静止；拒识窗口不计入状态。"

        # 兼容尚未提供 pipeline_state 的状态对象。
        if getattr(state, "device_status", "offline") != "online":
            return "waiting", "等待设备数据", "连接设备并收到原始脑电后开始预热。"
        if float(getattr(state, "warmup_progress", 0.0)) < 1.0:
            remaining = max(
                0, int(round((1.0 - float(state.warmup_progress)) * WARMUP_SECONDS))
            )
            return "warming", "正在积累分析数据", f"距离首个分析窗口约 {remaining} 秒。"
        if getattr(state, "quality_level", "rejected") == "rejected":
            return "rejected", "当前窗口已拒识", "信号或结果可信度不足，不进行学习状态解释。"
        if getattr(state, "predicted_state", None) is None:
            return "waiting_result", "等待首个分析结果", "分析窗口已形成，模型正在计算。"
        return "ready", "分析运行正常", "模型输出与信号质量已分层展示。"

    def _refresh_analysis_status(self, state) -> str:
        kind, title, hint = self._analysis_view(state)
        palette = {
            "ready": ("#4ADE80", "GoodLabel"),
            "warming": ("#4FC3F7", "AccentLabel"),
            "waiting_result": ("#4FC3F7", "AccentLabel"),
            "rejected": ("#FBBF24", "WarnLabel"),
            "error": ("#F87171", "DangerLabel"),
            "waiting": ("#94A3B8", "DimLabel"),
        }
        color, object_name = palette.get(kind, palette["waiting"])
        self._analysis_label.setObjectName(object_name)
        self._analysis_label.setText(title)
        self._analysis_label.setStyleSheet(
            f"font-size: 13px; font-weight: 600; color: {color};"
        )
        self._analysis_hint.setText(hint)
        self._btn_diagnostics.setVisible(kind == "error")
        return kind

    def _refresh_role_focus(self, state):
        if not hasattr(self, "_role_card"):
            return
        if self._role == "student":
            self._sync_learner_controls(state)
            suggestion = (
                getattr(state, "adaptive_feedback_text", "")
                or getattr(state, "feedback_text", "")
                or "等待信号稳定后生成建议。"
            )
            self._focus_values[1].setText(suggestion)
            return

        if self._role == "teacher":
            stable = CLASS_DISPLAY.get(getattr(state, "stable_state", None), "暂无有效趋势")
            class_trend = getattr(state, "class_trend_summary", "") or f"当前监测：{stable}"
            alerts = getattr(state, "class_alerts", None)
            if alerts:
                alert_text = f"{len(alerts)} 条待关注"
            elif getattr(state, "quality_level", "rejected") == "rejected":
                alert_text = "信号质量待处理"
            elif float(getattr(state, "_negative_sustain_seconds", 0.0)) > 0:
                alert_text = "存在持续负性趋势"
            else:
                alert_text = "暂无异常提醒"
            records = getattr(state, "process_record_count", None)
            if records is None:
                records = len(getattr(state, "_events", []))
            self._focus_values[0].setText(str(class_trend))
            self._focus_values[1].setText(alert_text)
            self._focus_values[2].setText(f"本次记录 {records} 条")
            return

        analysis_kind, _, _ = self._analysis_view(state)
        device = "在线" if getattr(state, "device_status", "offline") == "online" else "未就绪"
        model = "故障" if analysis_kind == "error" else "可用 / 等待数据"
        quality = {"trusted": "可信", "warning": "警告", "rejected": "拒识"}.get(
            getattr(state, "quality_level", "rejected"), "未知"
        )
        run_id = getattr(state, "run_id", "--")
        event_count = len(getattr(state, "_events", []))
        self._focus_values[0].setText(f"MindWave：{device}")
        self._focus_values[1].setText(f"模型：{model} · 数据：{quality}")
        self._focus_values[2].setText(f"运行ID {run_id} · {event_count} 条记录")

    # ── 按钮事件 ──

    def _on_start(self):
        if self._role == "teacher":
            return
        if getattr(self.state, "session_active", self.state._session_active):
            return
        reason = learning_start_block_reason(self.state) if self._role == "student" else ""
        if reason and self.isVisible():
            QMessageBox.warning(self, "暂不能开始学习", reason)
            return
        self.state.reset_session()
        self.service.start_session()
        self._session_started = True
        self._session_paused = False
        self._btn_start.setEnabled(False)
        self._btn_pause.setEnabled(True)
        self._btn_pause.setText("暂停")
        self._btn_event.setEnabled(False)
        self._btn_end.setEnabled(True)
        self._btn_export.setEnabled(False)
        self._trend_plot.reset()
        self._prob_trend.reset()
        self.state.add_event("会话开始", "system")

    def _on_pause(self):
        if self._role == "teacher":
            return
        if self._session_paused:
            self.service.resume_session()
            self._session_paused = False
            self._btn_pause.setText("暂停")
            self.state.add_event("会话恢复", "system")
        else:
            self.service.pause_session()
            self._session_paused = True
            self._btn_pause.setText("继续")
            self.state.add_event("会话暂停", "system")

    def _on_event(self):
        if self._role == "teacher":
            return
        text, ok = QInputDialog.getText(
            self, "事件标记", "输入事件描述："
        )
        if ok and text:
            self.state.add_event(text, "user")

    def _on_end(self):
        if self._role == "teacher":
            return
        self.service.end_session()
        self._session_started = False
        self._btn_start.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._btn_pause.setText("暂停")
        self._btn_event.setEnabled(False)
        self._btn_end.setEnabled(False)
        self._btn_export.setEnabled(True)
        self.state.add_event("会话结束", "system")
        QMessageBox.information(
            self, "会话结束",
            f"会话已结束，时长 {self.state.session_seconds:.0f} 秒。可导出报告。"
        )

    def _on_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "导出报告", "session_report.txt", "Text Files (*.txt)"
        )
        if path:
            self._export_report(path)
            QMessageBox.information(self, "导出成功", f"报告已导出至：\n{path}")

    def _export_report(self, path: str):
        s = self.state
        avg_att = sum(s._attention_history) / max(len(s._attention_history), 1)
        avg_med = sum(s._meditation_history) / max(len(s._meditation_history), 1)
        lines = [
            "智学脑机助手 - 会话报告",
            "=" * 40,
            f"用户：{s._user_name} ({s._user_id})",
            f"会话时长：{s.session_seconds:.0f} 秒",
            f"信号合格：{'是' if s.quality_level != 'rejected' else '否'}",
            f"平均Attention：{avg_att:.1f}",
            f"平均Meditation：{avg_med:.1f}",
            "",
            "事件记录：",
        ]
        for ev in s._events:
            lines.append(f"  [{ev.category}] {ev.label} - {ev.note}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ── 状态更新 ──

    def update_state(self, state):
        s = state
        requested_role = self._normalize_role(getattr(s, "current_role", self._role))
        if requested_role != self._role:
            self.set_role(requested_role)
        if self._role == "teacher":
            self._refresh_teacher_students()
            self._update_teacher_snapshot()
            return
        session_active = bool(getattr(s, "session_active", s._session_active))
        session_paused = bool(getattr(s, "session_paused", False))
        self._session_started = session_active
        self._session_paused = session_paused
        start_reason = learning_start_block_reason(
            s, require_baseline=self._role == "student"
        ) if self._role == "student" else ""
        # Keep clickable while idle so an attempted start receives a clear reason.
        self._btn_start.setEnabled(not session_active)
        self._btn_start.setToolTip(start_reason or "开始学习记录")
        self._btn_pause.setEnabled(session_active)
        self._btn_pause.setText("继续" if session_paused else "暂停")
        self._btn_event.setEnabled(session_active)
        self._btn_end.setEnabled(session_active)
        analysis_kind = self._refresh_analysis_status(s)

        # 会话时间
        mins = int(s.session_seconds) // 60
        secs = int(s.session_seconds) % 60
        self._session_time.setText(f"会话时间 {mins:02d}:{secs:02d}")

        # ── 状态卡片 ──

        # Connector 卡片：使用 connector_status
        if s.connector_status == "online":
            self._card_connector["value"].setText("已连接")
            self._card_connector["indicator"].set_state(StatusIndicator.LEVEL_GOOD, "在线")
        elif s.connector_status == "connecting":
            self._card_connector["value"].setText("连接中")
            self._card_connector["indicator"].set_state(StatusIndicator.LEVEL_WARN, "连接中")
        else:
            self._card_connector["value"].setText("未连接")
            self._card_connector["indicator"].set_state(StatusIndicator.LEVEL_ERROR, "离线")

        # Device 卡片：使用 device_status
        if s.device_status == "online":
            self._card_device["value"].setText("在线")
            self._card_device["indicator"].set_state(StatusIndicator.LEVEL_GOOD, "正常")
        elif s.device_status == "waiting_raw":
            self._card_device["value"].setText("等待数据")
            self._card_device["indicator"].set_state(StatusIndicator.LEVEL_WARN, "等待")
        else:
            self._card_device["value"].setText("离线")
            self._card_device["indicator"].set_state(StatusIndicator.LEVEL_ERROR, "离线")

        # Poor Signal 卡片：poor_signal 可能为 None
        poor = s.poor_signal
        if poor is None:
            self._card_poor["value"].setText("--")
            self._card_poor["indicator"].set_state(StatusIndicator.LEVEL_NEUTRAL, "无数据")
        elif poor < MAX_POOR_SIGNAL:
            self._card_poor["value"].setText(str(poor))
            self._card_poor["indicator"].set_state(StatusIndicator.LEVEL_GOOD, "合格")
        elif poor < 200:
            self._card_poor["value"].setText(str(poor))
            self._card_poor["indicator"].set_state(StatusIndicator.LEVEL_WARN, "警告")
        else:
            self._card_poor["value"].setText(str(poor))
            self._card_poor["indicator"].set_state(StatusIndicator.LEVEL_ERROR, "无信号")

        # 信号质量卡片：使用 quality_level（不使用数值置信度）
        ql = s.quality_level
        is_device_offline = (
            s.device_status != "online" or s.connector_status != "online"
        )
        if is_device_offline:
            self._card_conf["value"].setText("不可评估")
            self._card_conf["indicator"].set_state(
                StatusIndicator.LEVEL_ERROR, "设备未连接"
            )
        elif ql == "trusted":
            self._card_conf["value"].setText("可信")
            self._card_conf["indicator"].set_state(StatusIndicator.LEVEL_GOOD, "可信")
        elif ql == "warning":
            self._card_conf["value"].setText("警告")
            self._card_conf["indicator"].set_state(StatusIndicator.LEVEL_WARN, "警告")
        else:  # rejected
            self._card_conf["value"].setText("不合格")
            self._card_conf["indicator"].set_state(StatusIndicator.LEVEL_ERROR, "不合格")
        # quality_reasons 作为 tooltip 展示
        public_reasons = self._public_quality_reasons(
            getattr(s, "quality_reasons", [])
        )
        reasons_text = "、".join(public_reasons)
        if not reasons_text and getattr(s, "quality_reasons", []):
            reasons_text = "详细技术原因请查看系统诊断"
        self._card_conf["value"].setToolTip(reasons_text)
        self._card_conf["indicator"].setToolTip(reasons_text)

        # 采样率与来源分层展示，Live 页面不得出现 Mock 文案。
        measured_rate = getattr(s, "sample_rate_hz", None)
        self._card_rate["value"].setText(
            f"{measured_rate:.0f} Hz" if measured_rate is not None else "等待采样"
        )
        mode = str(getattr(s, "mode", "live") or "live").lower()
        mode_text = {"live": "实时采集", "mock": "教学演示数据", "replay": "离线回放数据"}.get(
            mode, "未知来源"
        )
        self._card_rate["indicator"].set_state(
            StatusIndicator.LEVEL_NEUTRAL, mode_text
        )

        # ── 预热进度：使用 warmup_progress（0.0~1.0）──
        self._warmup_bar.setValue(int(s.warmup_progress * 100))
        if getattr(s, "warmup_complete", s.warmup_progress >= 1.0):
            self._warmup_label.setText("已完成")
            self._warmup_label.setStyleSheet("font-size: 13px; color: #4ADE80;")
        else:
            self._warmup_label.setText(
                f"{s.warmup_progress * WARMUP_SECONDS:.1f}s / {WARMUP_SECONDS:.0f}s"
            )
            self._warmup_label.setStyleSheet("font-size: 13px;")

        # ── EEG曲线：使用内部缓冲 _eeg_raw_buffer ──
        if s._eeg_raw_buffer:
            self._eeg_plot.push_buffer(s._eeg_raw_buffer)
        self._eeg_plot.set_dimmed(s.quality_level == "rejected")

        # ── 趋势图：attention/meditation 可能为 None，以 0 填充 ──
        att = s.attention if s.attention is not None else 0
        med = s.meditation if s.meditation is not None else 0
        self._trend_plot.push_values(att, med)

        # ── 仪表：attention/meditation 可能为 None ──
        self._att_gauge.set_value(att)
        self._med_gauge.set_value(med)

        # ── 概率面板 ──
        self._prob_panel.update_state(s)

        quality_text = {
            "trusted": "信号质量：可信",
            "warning": "信号质量：警告",
            "rejected": "信号质量：暂不可用",
        }.get(getattr(s, "quality_level", "rejected"), "信号质量：未知")
        if public_reasons:
            quality_text += f"（{'、'.join(public_reasons)}）"
        self._prob_panel._confidence_label.setText(quality_text)
        if analysis_kind == "error":
            for bar in self._prob_panel._bars.values():
                bar.set_value(0.0)
                bar.set_dimmed(True)
            self._prob_panel._warning_label.setText(
                "智能分析暂不可用；如问题持续，请联系管理员"
            )
            self._prob_panel._warning_label.setVisible(True)
        elif analysis_kind == "rejected":
            self._prob_panel._warning_label.setText(
                "当前窗口已拒识，不进行学习状态解释"
            )
            self._prob_panel._warning_label.setVisible(True)

        # ── 预测结果：使用 predicted_state 和 confidence ──
        if analysis_kind == "error":
            self._pred_label.setText("当前状态：智能分析暂不可用")
            self._pred_label.setStyleSheet("font-size: 16px; padding: 4px 0; color: #F87171;")
        elif analysis_kind == "rejected":
            self._pred_label.setText("当前状态：已拒识（不计入趋势）")
            self._pred_label.setStyleSheet("font-size: 16px; padding: 4px 0; color: #FBBF24;")
        elif s.inference_eligible and s.predicted_state is not None:
            display = CLASS_DISPLAY.get(s.predicted_state, s.predicted_state)
            conf_text = f"{s.confidence * 100:.1f}%" if s.confidence is not None else "--"
            self._pred_label.setText(f"当前状态：{display}  (置信度 {conf_text})")
            color_map = {
                "positive": "#4ADE80",
                "neutral": "#4FC3F7",
                "negative": "#F87171",
            }
            self._pred_label.setStyleSheet(
                f"font-size: 16px; padding: 4px 0; color: {color_map.get(s.predicted_state, '#E8EDF3')};"
            )
        else:
            waiting_text = {
                "warming": "当前状态：正在积累分析数据",
                "waiting_result": "当前状态：等待首个分析结果",
                "waiting": "当前状态：等待设备数据",
            }.get(analysis_kind, "当前状态：等待分析")
            self._pred_label.setText(waiting_text)
            self._pred_label.setStyleSheet("font-size: 16px; padding: 4px 0; color: #6B7689;")

        # ── 概率趋势：使用内部缓冲 _prob_history ──
        if s._prob_history:
            latest = s._prob_history[-1]
            self._prob_trend.push_values(latest[1], latest[2], latest[3])

        # ── 持续状态：使用 stable_state ──
        if s.inference_eligible and analysis_kind not in {"error", "rejected"}:
            display = CLASS_DISPLAY.get(s.stable_state, "--")
            self._sustain_label.setText(f"主导状态：{display}")
            color_map = {
                "positive": "#4ADE80",
                "neutral": "#4FC3F7",
                "negative": "#F87171",
                "unknown": "#6B7689",
            }
            self._sustain_label.setStyleSheet(
                f"font-size: 16px; font-weight: bold; color: {color_map.get(s.stable_state, '#6B7689')};"
            )
        else:
            self._sustain_label.setText("主导状态：--")
            self._sustain_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #6B7689;")

        # 消极持续：使用内部簿记 _negative_sustain_seconds
        self._sustain_neg.setText(
            f"负性趋势持续：{float(getattr(s, '_negative_sustain_seconds', 0.0)):.1f}秒"
        )

        # 干预状态：使用内部簿记 _intervention_triggered / _intervention_cooldown
        if getattr(s, "_intervention_triggered", False):
            self._intervention_label.setText("已生成节奏调整建议（负性趋势持续超过20秒）")
            self._intervention_label.setVisible(True)
        elif getattr(s, "_intervention_cooldown", False):
            self._intervention_label.setText("建议冷却中（90秒内不重复触发）")
            self._intervention_label.setVisible(True)
        else:
            self._intervention_label.setVisible(False)

        # ── AI建议：使用 feedback_text ──
        if analysis_kind == "error":
            self._ai_label.setText("分析服务暂不可用，当前不生成学习状态建议。")
        else:
            self._ai_label.setText(s.feedback_text)
        if s.inference_eligible and analysis_kind != "error":
            self._ai_label.setStyleSheet("color: #C5CDD9; font-size: 14px;")
        else:
            self._ai_label.setStyleSheet("color: #FBBF24; font-size: 14px;")
        self._refresh_role_focus(s)
