"""页面5：历史会话与报告页。

UI 只消费 DashboardState 的正式字段与内部簿记字段 _history_sessions。
所有预填充的 SessionRecord 均标记 demo=True，属于演示数据。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame, QFileDialog, QMessageBox, QComboBox,
)

from pages.base_page import BasePage
from widgets.card import Card


class HistoryPage(BasePage):
    def __init__(self, state, service):
        self.state = state
        self.service = service
        super().__init__(
            "历史会话与报告",
            "查看历史学习记录，回顾状态倾向、注意力趋势与关键事件。"
        )
        self._build_ui()

    def _build_ui(self):
        # ── 演示数据警告横幅（始终可见）──
        demo_banner = QLabel("注意：以下为演示数据，非真实历史记录")
        demo_banner.setStyleSheet(
            "background-color: #1B2534; border: 1px solid #303C50; "
            "color: #AAB5C5; font-size: 13px; font-weight: 500; "
            "padding: 8px 12px; border-radius: 8px;"
        )
        demo_banner.setFixedHeight(38)
        demo_banner.setAlignment(Qt.AlignCenter)
        self.content_layout.addWidget(demo_banner)

        splitter = QSplitter(Qt.Horizontal)

        # ── 左侧：会话列表 ──
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # 筛选
        filter_card = Card("筛选")
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("排序:"))
        self._combo_sort = QComboBox()
        self._combo_sort.addItems(["按时间倒序", "按时长排序", "按信号质量排序"])
        self._combo_sort.currentIndexChanged.connect(self._refresh_table)
        filter_layout.addWidget(self._combo_sort, 1)
        filter_card.add_widget(self._wrap(filter_layout))
        left_layout.addWidget(filter_card)

        # 会话表（新增"标记"列用于显示演示数据）
        table_card = Card("历史会话")
        table_layout = QVBoxLayout()

        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels(
            ["会话ID", "开始时间", "时长", "任务", "信号质量", "事件数", "标记"]
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.itemSelectionChanged.connect(self._on_select)
        table_layout.addWidget(self._table)

        table_card.add_widget(self._wrap(table_layout))
        left_layout.addWidget(table_card, 1)

        # 数据操作
        actions = QHBoxLayout()
        btn_folder = QPushButton("打开真实会话文件夹")
        btn_folder.clicked.connect(self._open_sessions_folder)
        actions.addWidget(btn_folder)
        btn_export = QPushButton("导出演示记录 (CSV)")
        btn_export.clicked.connect(self._export_all)
        actions.addWidget(btn_export)
        left_layout.addLayout(actions)

        splitter.addWidget(left)

        # ── 右侧：会话详情 ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # 演示数据标记（详情区顶部，选中演示记录时显示）
        self._demo_badge = QLabel("演示数据")
        self._demo_badge.setStyleSheet(
            "background-color: rgba(200,150,40,0.15); "
            "color: #FBBF24; font-size: 12px; font-weight: 700; "
            "padding: 6px 10px; border-radius: 4px;"
        )
        self._demo_badge.setAlignment(Qt.AlignCenter)
        self._demo_badge.setVisible(False)
        right_layout.addWidget(self._demo_badge)

        # 详情卡片
        self._detail_card = Card("会话详情")
        self._detail_layout = QGridLayout()
        self._detail_layout.setSpacing(10)

        self._detail_labels = {}
        fields = [
            ("session_id", "会话ID"),
            ("user_id", "用户ID"),
            ("start_time", "开始时间"),
            ("duration", "时长"),
            ("task", "任务"),
            ("avg_attention", "平均Attention"),
            ("avg_meditation", "平均Meditation"),
            ("signal_quality", "信号质量"),
            ("positive_ratio", "Positive占比"),
            ("neutral_ratio", "Neutral占比"),
            ("negative_ratio", "Negative占比"),
            ("event_count", "事件数"),
        ]
        for i, (key, label) in enumerate(fields):
            row, col = i // 2, i % 2
            lbl = QLabel(label + ":")
            lbl.setStyleSheet("color: #6B7689; font-size: 13px;")
            self._detail_layout.addWidget(lbl, row, col * 2)
            val = QLabel("--")
            val.setStyleSheet("color: #E8EDF3; font-size: 14px; font-weight: 600;")
            self._detail_layout.addWidget(val, row, col * 2 + 1)
            self._detail_labels[key] = val

        self._detail_card.add_widget(self._wrap(self._detail_layout))
        right_layout.addWidget(self._detail_card)

        # 情绪分布卡片
        dist_card = Card("学习状态倾向分布")
        self._dist_layout = QVBoxLayout()
        self._dist_bars = {}
        for cls, color in [("positive", "#5B8DEF"), ("neutral", "#8EA3BF"), ("negative", "#6F82A0")]:
            row = QHBoxLayout()
            label = QLabel({"positive": "积极", "neutral": "中性", "negative": "消极"}[cls])
            label.setStyleSheet("color: #A5B0C0; font-size: 13px; font-weight: 600;")
            label.setFixedWidth(40)
            row.addWidget(label)

            bar_frame = QFrame()
            bar_frame.setFixedHeight(20)
            bar_frame.setStyleSheet("background-color: #1A1F2E; border-radius: 4px;")
            bar_layout = QHBoxLayout(bar_frame)
            bar_layout.setContentsMargins(0, 0, 0, 0)
            fill = QFrame()
            fill.setFixedWidth(0)
            fill.setStyleSheet(f"background-color: {color}; border-radius: 4px;")
            bar_layout.addWidget(fill)
            bar_layout.addStretch()
            row.addWidget(bar_frame, 1)

            val_label = QLabel("0%")
            val_label.setStyleSheet("color: #E7ECF3; font-size: 13px;")
            val_label.setFixedWidth(50)
            row.addWidget(val_label)

            self._dist_bars[cls] = {"fill": fill, "label": val_label, "frame": bar_frame}
            self._dist_layout.addLayout(row)

        dist_card.add_widget(self._wrap(self._dist_layout))
        right_layout.addWidget(dist_card)

        # 备注
        notes_card = Card("备注")
        self._notes_label = QLabel("--")
        self._notes_label.setWordWrap(True)
        self._notes_label.setStyleSheet("color: #C5CDD9; font-size: 13px;")
        notes_card.add_widget(self._notes_label)
        right_layout.addWidget(notes_card)

        right_layout.addStretch()
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([600, 500])

        self.content_layout.addWidget(splitter)

    def _wrap(self, layout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        return w

    def _get_sorted_sessions(self):
        """获取排序后的会话列表（从 _history_sessions 读取）。"""
        sessions = list(self.state._history_sessions)
        sort_idx = self._combo_sort.currentIndex()
        if sort_idx == 1:
            sessions.sort(key=lambda s: s.duration_seconds, reverse=True)
        elif sort_idx == 2:
            sessions.sort(key=lambda s: s.signal_quality, reverse=True)
        return sessions

    def _refresh_table(self):
        sessions = self._get_sorted_sessions()

        self._table.setRowCount(len(sessions))
        for i, s in enumerate(sessions):
            self._table.setItem(i, 0, QTableWidgetItem(s.session_id))
            self._table.setItem(i, 1, QTableWidgetItem(s.start_time))
            mins = int(s.duration_seconds) // 60
            secs = int(s.duration_seconds) % 60
            self._table.setItem(i, 2, QTableWidgetItem(f"{mins}分{secs}秒"))
            self._table.setItem(i, 3, QTableWidgetItem(s.notes))
            self._table.setItem(i, 4, QTableWidgetItem(f"{s.signal_quality*100:.0f}%"))
            self._table.setItem(i, 5, QTableWidgetItem(str(s.event_count)))

            # 标记列：演示数据显示"演示"
            demo_item = QTableWidgetItem("演示" if s.demo else "--")
            if s.demo:
                demo_item.setForeground(QColor("#FBBF24"))
            self._table.setItem(i, 6, demo_item)

    def _on_select(self):
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        sessions = self._get_sorted_sessions()

        if idx >= len(sessions):
            return
        s = sessions[idx]

        mins = int(s.duration_seconds) // 60
        secs = int(s.duration_seconds) % 60

        self._detail_labels["session_id"].setText(s.session_id)
        self._detail_labels["user_id"].setText(s.user_id)
        self._detail_labels["start_time"].setText(s.start_time)
        self._detail_labels["duration"].setText(f"{mins}分{secs}秒")
        self._detail_labels["task"].setText(s.notes)
        self._detail_labels["avg_attention"].setText(f"{s.avg_attention:.1f}")
        self._detail_labels["avg_meditation"].setText(f"{s.avg_meditation:.1f}")
        self._detail_labels["signal_quality"].setText(f"{s.signal_quality*100:.0f}%")
        self._detail_labels["positive_ratio"].setText(f"{s.positive_ratio*100:.1f}%")
        self._detail_labels["neutral_ratio"].setText(f"{s.neutral_ratio*100:.1f}%")
        self._detail_labels["negative_ratio"].setText(f"{s.negative_ratio*100:.1f}%")
        self._detail_labels["event_count"].setText(str(s.event_count))

        # 演示数据标记：选中演示记录时突出显示
        if getattr(s, "demo", False):
            self._demo_badge.setText("演示数据")
            self._demo_badge.setVisible(True)
        else:
            self._demo_badge.setVisible(False)

        # 情绪分布条
        for cls, key in [("positive", "positive_ratio"), ("neutral", "neutral_ratio"), ("negative", "negative_ratio")]:
            ratio = getattr(s, key)
            bar = self._dist_bars[cls]
            bar["label"].setText(f"{ratio*100:.1f}%")
            bar["fill"].setFixedWidth(int(ratio * 200))

        note_prefix = "【演示数据】" if getattr(s, "demo", False) else ""
        self._notes_label.setText(
            f"{note_prefix}"
            f"任务类型：{s.notes}\n"
            f"会话ID：{s.session_id}\n"
            f"该会话信号质量{'良好' if s.signal_quality > 0.7 else '一般' if s.signal_quality > 0.5 else '较差'}，"
            f"积极状态占比{s.positive_ratio*100:.1f}%。"
        )

    def _export_all(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "导出历史记录", "history_sessions.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        import csv
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "session_id", "user_id", "start_time", "duration_seconds",
                "positive_ratio", "neutral_ratio", "negative_ratio",
                "avg_attention", "avg_meditation", "signal_quality",
                "event_count", "notes", "demo",
            ])
            for s in self.state._history_sessions:
                writer.writerow([
                    s.session_id, s.user_id, s.start_time, s.duration_seconds,
                    s.positive_ratio, s.neutral_ratio, s.negative_ratio,
                    s.avg_attention, s.avg_meditation, s.signal_quality,
                    s.event_count, s.notes, getattr(s, "demo", False),
                ])
        QMessageBox.information(
            self, "导出成功",
            f"已导出 {len(self.state._history_sessions)} 条记录。"
        )

    def _open_sessions_folder(self):
        from pathlib import Path
        folder = Path(getattr(self.service, "sessions_dir", Path("data/sessions"))).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def on_show(self):
        self._refresh_table()
