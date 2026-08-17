"""页面4：学习任务和事件标记页。

严格遵循 eeg_modular/ui_prototype/services/dashboard_state.py 中定义的
DashboardState 正式字段接口。UI 业务逻辑只消费正式字段，
内部簿记字段（_前缀）仅用于事件列表等簿记操作。
"""

from __future__ import annotations

import time
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QLineEdit, QSplitter, QMessageBox,
)

from pages.base_page import BasePage
from widgets.card import Card
from services.dashboard_state import CLASS_DISPLAY


TASK_TYPES = [
    "数学练习", "英语阅读", "编程任务", "物理复习",
    "语文写作", "专注冥想", "记忆训练", "自由学习",
]

QUICK_EVENTS = [
    ("开始专注", "user", "#4ADE80"),
    ("走神", "user", "#FBBF24"),
    ("感到困难", "user", "#F87171"),
    ("短暂休息", "user", "#4FC3F7"),
    ("情绪波动", "user", "#FBBF24"),
    ("任务切换", "user", "#94A3B8"),
]


class TaskPage(BasePage):
    def __init__(self, state, service):
        self.state = state
        self.service = service
        self._task_start = 0.0
        self._task_active = False
        super().__init__(
            "学习任务与事件标记",
            "管理学习任务并记录关键事件，用于后续会话分析与报告。"
        )
        self._build_ui()

    def _build_ui(self):
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
        task_form.addWidget(self._combo_task, 0, 1)

        task_form.addWidget(QLabel("难度:"), 1, 0)
        self._combo_diff = QComboBox()
        self._combo_diff.addItems(["简单", "中等", "困难"])
        self._combo_diff.setCurrentIndex(1)
        task_form.addWidget(self._combo_diff, 1, 1)

        task_form.addWidget(QLabel("备注:"), 2, 0)
        self._input_note = QLineEdit()
        self._input_note.setPlaceholderText("可选备注信息")
        task_form.addWidget(self._input_note, 2, 1)

        task_card.add_widget(self._wrap(task_form))
        left_layout.addWidget(task_card)

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

        self._state_att = QLabel("Attention: --")
        self._state_att.setStyleSheet("color: #4FC3F7; font-size: 13px;")
        state_layout.addWidget(self._state_att, 1, 0)

        self._state_med = QLabel("Meditation: --")
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
        event_grid = QGridLayout()
        event_grid.setSpacing(8)
        for i, (label, cat, color) in enumerate(QUICK_EVENTS):
            btn = QPushButton(label)
            btn.setObjectName("EventButton")
            btn.clicked.connect(lambda checked, l=label: self._add_quick_event(l))
            event_grid.addWidget(btn, i // 3, i % 3)

        event_card.add_widget(self._wrap(event_grid))

        # 自定义事件
        custom_row = QHBoxLayout()
        self._input_custom = QLineEdit()
        self._input_custom.setPlaceholderText("输入自定义事件描述...")
        custom_row.addWidget(self._input_custom, 1)

        btn_custom = QPushButton("标记")
        btn_custom.clicked.connect(self._add_custom_event)
        custom_row.addWidget(btn_custom)
        event_card.add_widget(self._wrap(custom_row))

        right_layout.addWidget(event_card)

        # 事件时间线
        timeline_card = Card("事件时间线")
        timeline_layout = QVBoxLayout()

        self._event_table = QTableWidget()
        self._event_table.setColumnCount(4)
        self._event_table.setHorizontalHeaderLabels(["时间", "类别", "事件", "备注"])
        self._event_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._event_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._event_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._event_table.setAlternatingRowColors(True)
        self._event_table.verticalHeader().setVisible(False)
        self._event_table.setEditTriggers(QTableWidget.NoEditTriggers)
        timeline_layout.addWidget(self._event_table)

        # 清除按钮
        btn_clear = QPushButton("清空事件")
        btn_clear.clicked.connect(self._clear_events)
        timeline_layout.addWidget(btn_clear, 0, Qt.AlignRight)

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

    def _on_task_start(self):
        if not self.state._session_active:
            QMessageBox.information(
                self, "请先开始学习记录",
                "任务标记必须关联到一次学习记录。请先在“实时仪表盘”点击“开始学习记录”。"
            )
            return
        self._task_start = time.time()
        self._task_active = True
        self._btn_task_start.setEnabled(False)
        self._btn_task_stop.setEnabled(True)
        self._task_status.setText("进行中")
        self._task_status.setStyleSheet("color: #4ADE80; font-size: 14px;")
        task_name = self._combo_task.currentText()
        self.state.add_event(f"开始任务: {task_name}", "system", self._input_note.text())

    def _on_task_stop(self):
        self._task_active = False
        self._btn_task_start.setEnabled(True)
        self._btn_task_stop.setEnabled(False)
        self._task_status.setText("已结束")
        self._task_status.setStyleSheet("color: #6B7689; font-size: 14px;")
        self.state.add_event("结束任务", "system")

    def _add_quick_event(self, label: str):
        self.state.add_event(label, "user")

    def _add_custom_event(self):
        text = self._input_custom.text().strip()
        if text:
            self.state.add_event(text, "user")
            self._input_custom.clear()

    def _clear_events(self):
        self.state._events.clear()
        self._refresh_table()

    def _on_event_added(self, event):
        self._refresh_table()

    def _refresh_table(self):
        events = self.state._events
        self._event_table.setRowCount(len(events))
        cat_colors = {"user": "#4FC3F7", "system": "#94A3B8", "intervention": "#FBBF24"}
        for i, ev in enumerate(events):
            from datetime import datetime
            ts = datetime.fromtimestamp(ev.timestamp).strftime("%H:%M:%S")
            self._event_table.setItem(i, 0, QTableWidgetItem(ts))
            cat_item = QTableWidgetItem(ev.category)
            cat_item.setForeground(Qt.GlobalColor.white)
            self._event_table.setItem(i, 1, cat_item)
            self._event_table.setItem(i, 2, QTableWidgetItem(ev.label))
            self._event_table.setItem(i, 3, QTableWidgetItem(ev.note))
        self._event_table.scrollToBottom()

    def update_state(self, state):
        # 任务计时
        if self._task_active:
            elapsed = time.time() - self._task_start
            mins = int(elapsed) // 60
            secs = int(elapsed) % 60
            self._task_time.setText(f"{mins:02d}:{secs:02d}")

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
            self._state_att.setText(f"Attention: {state.attention:.0f}")
        else:
            self._state_att.setText("Attention: --")

        if state.meditation is not None:
            self._state_med.setText(f"Meditation: {state.meditation:.0f}")
        else:
            self._state_med.setText("Meditation: --")

        # 刷新事件表：使用 _events
        if not hasattr(self, "_last_event_count") or self._last_event_count != len(state._events):
            self._refresh_table()
            self._last_event_count = len(state._events)

    def on_show(self):
        self._refresh_table()
