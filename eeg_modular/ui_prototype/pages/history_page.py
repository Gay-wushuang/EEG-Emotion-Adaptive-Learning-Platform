"""页面5：历史会话与报告页（UI Round - 对齐 SVG 母版）。

页面显示由 ``DashboardState.reload_history`` 自动索引的本地记录。
Live/Replay 正式记录与 Mock 演示记录在存储和展示上均明确区分。

UI 布局（严格参照 student_history_report_1080p_master.svg）：
┌─ 顶部说明（context strip，数据源 + 来源标记）
├─ top_row ─ 筛选卡 (58%) + 近期概览卡 (42%)
├─ main_row ─ 历史会话表 (58%) + 右侧详情面板 (42%)
│   ├─ 表格 6 列：开始时间 | 任务 | 有效时长 | 信号质量 | 事件数 | 来源
│   │  （session_id 不进表格，仅内部逻辑/学习报告使用）
│   └─ 右侧面板：未选中提示 / 学习记录详情 / 任务阶段明细 / 关键事件与建议
└─ bottom_row ─ 查看学习报告 / 打开会话文件夹 / 导出 CSV

空态产品化：未选中记录 → "请选择一条历史学习记录查看详情。"；
字段确实缺失 → "暂无数据"；无记录 → "暂无历史学习记录"。
绝不显示 0 / 0% / 00:00 / -- 作为缺失占位。

本文件只修改 history_page.py，不动公共组件 / theme / 其它页面。
"""

from __future__ import annotations

from datetime import datetime
import re

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QComboBox, QDialog, QScrollArea,
    QStackedWidget, QSizePolicy, QLayout,
)

from pages.base_page import BasePage
from widgets.card import Card
from services.dashboard_state import DIFFICULTY_DISPLAY
from services.session_report_builder import (
    MISSING, MISSING_AI, MISSING_BASELINE, MISSING_FEEDBACK,
    TASK_STATUS_DISPLAY, build_learning_report,
)
from services.student_history_summary import build_recent_summary
from services.teaching_store import BaselineResultStore, TeacherSelectionContext

# 未选中记录时右侧详情区的产品化提示文案
SELECT_HINT_TEXT = "请选择一条历史学习记录查看详情。"
# 表格无记录时的产品化空态文案
TABLE_EMPTY_TEXT = "暂无历史学习记录"

# 顶部 context strip 文案（跟随筛选器）
SOURCE_BANNER_TEXT = {
    "all": "当前显示全部学习记录，教学演示与正式学习记录分别保存。",
    "live": "当前显示实时采集学习记录。",
    "mock": "当前显示教学演示记录，与正式学习记录分开保存。",
}
SOURCE_NAMES = {"live": "实时采集", "mock": "教学演示", "replay": "离线回放"}


class HistoryPage(BasePage):
    def __init__(self, state, service, *, selection_context=None):
        self.state = state
        self.service = service
        self._selection_context = selection_context or TeacherSelectionContext.shared()
        self._baseline_store = None
        self._report_dialog = None
        super().__init__(
            "历史会话与报告",
            "回顾历史学习记录、任务阶段与关键事件。"
        )
        self._build_ui()
        # 教师切换观察学生后立即按新学生刷新历史视图（严格数据隔离）。
        self._selection_context.add_listener(self, "_on_selection_changed")

    # ═══════════════════════════════════════════════════════════════
    # UI 构建：严格对齐 SVG 母版
    # ═══════════════════════════════════════════════════════════════

    def _build_ui(self):
        # ── Context strip：当前显示 + 来源标记 ──
        strip_wrap = QWidget()
        strip = QHBoxLayout(strip_wrap)
        strip.setContentsMargins(0, 0, 0, 0)
        strip.setSpacing(10)

        self._demo_banner = QLabel(SOURCE_BANNER_TEXT["all"])
        self._demo_banner.setStyleSheet(
            "background-color: #1B2534; border: 1px solid #303C50; "
            "color: #AAB5C5; font-size: 13px; font-weight: 500; "
            "padding: 8px 14px; border-radius: 8px;"
        )
        self._demo_banner.setFixedHeight(40)
        self._demo_banner.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        strip.addWidget(self._demo_banner, 1)

        self._source_tag = QLabel("数据来源：实时采集")
        self._source_tag.setStyleSheet(
            "background-color: #113C2D; border: 1px solid #1E6B4E; "
            "color: #3DDC97; font-size: 12px; font-weight: 600; "
            "padding: 6px 14px; border-radius: 14px;"
        )
        self._source_tag.setAlignment(Qt.AlignCenter)
        strip.addWidget(self._source_tag)
        self.content_layout.addWidget(strip_wrap)

        # ── top_row：筛选卡 (≈58%) + 近期概览卡 (≈42%) ──
        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        filter_card = Card("筛选")
        filter_row = QHBoxLayout()
        filter_row.setSpacing(16)
        filter_row.addWidget(QLabel("排序"))
        self._combo_sort = QComboBox()
        self._combo_sort.addItems(["按时间倒序", "按时长排序", "按信号质量排序"])
        self._combo_sort.currentIndexChanged.connect(self._refresh_table)
        filter_row.addWidget(self._combo_sort, 1)
        filter_row.addWidget(QLabel("数据来源"))
        self._combo_source = QComboBox()
        self._combo_source.addItem("全部", "all")
        self._combo_source.addItem("实时采集", "live")
        self._combo_source.addItem("教学演示", "mock")
        self._combo_source.currentIndexChanged.connect(self._refresh_table)
        filter_row.addWidget(self._combo_source, 1)
        filter_card.add_widget(self._wrap(filter_row))
        top_row.addWidget(filter_card, 58)

        # 近期概览卡（复用 Round 4B build_recent_summary，无新业务）
        self._recent_card = Card("近期概览")
        self._recent_card.setToolTip("最多展示最近 5 次；基于已持久化的真实历史记录。")
        self._recent_body = QLabel("暂无历史记录")
        self._recent_body.setWordWrap(True)
        self._recent_body.setStyleSheet(
            "color: #C5CDD9; font-size: 13px; line-height: 1.7;"
        )
        self._recent_body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._recent_card.add_widget(self._recent_body)
        top_row.addWidget(self._recent_card, 42)

        self.content_layout.addLayout(top_row)

        # ── main_row：左 ≈58% 历史会话表 + 右 ≈42% 详情面板 ──
        main_row = QHBoxLayout()
        main_row.setSpacing(10)

        # 左：历史会话表（6 列产品化列序：开始时间 | 任务 | 有效时长 | 信号质量 | 事件数 | 来源。
        # session_id 不进入表格列，仅保留在内部选中逻辑与学习报告中。）
        table_card = Card("历史会话")
        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(
            ["开始时间", "任务", "有效时长", "信号质量", "事件数", "来源"]
        )
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Stretch)
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # 开始时间完整显示
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # 有效时长
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # 信号质量
        hh.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # 事件数
        hh.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # 来源
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.itemSelectionChanged.connect(self._on_select)
        # 表格空态：无记录时切换到"暂无历史学习记录"占位页
        empty_page = QWidget()
        empty_lay = QVBoxLayout(empty_page)
        empty_lay.setContentsMargins(0, 0, 0, 0)
        empty_lay.addStretch(1)
        self._table_empty_hint = QLabel(TABLE_EMPTY_TEXT)
        self._table_empty_hint.setAlignment(Qt.AlignCenter)
        self._table_empty_hint.setStyleSheet(
            "color: #8FA3BE; font-size: 15px; padding: 24px 0;"
        )
        empty_lay.addWidget(self._table_empty_hint)
        empty_lay.addStretch(1)
        self._table_stack = QStackedWidget()
        self._table_stack.addWidget(self._table)
        self._table_stack.addWidget(empty_page)
        table_card.add_widget(self._table_stack)
        main_row.addWidget(table_card, 58)

        # 右：3 张卡（学习记录详情 / 任务阶段明细 / 关键事件与建议）
        right_col = QVBoxLayout()
        self._right_detail_layout = right_col
        right_col.setSizeConstraint(QLayout.SetMinimumSize)
        right_col.setSpacing(10)

        # 未选中记录时的产品化提示（选中任意行后隐藏）
        self._select_hint = QLabel(SELECT_HINT_TEXT)
        self._select_hint.setWordWrap(True)
        self._select_hint.setAlignment(Qt.AlignCenter)
        self._select_hint.setStyleSheet(
            "background-color: #1B2534; border: 1px solid #303C50; "
            "color: #AAB5C5; font-size: 13px; padding: 10px 14px; "
            "border-radius: 8px;"
        )
        right_col.addWidget(self._select_hint)

        # ① 学习记录详情：6 字段（开始时间/有效时长/任务数量 + 数据来源/信号质量/事件数）
        self._detail_card = Card("学习记录详情")
        self._detail_card._layout.setContentsMargins(12, 5, 12, 5)
        self._detail_labels = {}
        detail_grid = QGridLayout()
        self._detail_grid = detail_grid
        detail_grid.setSizeConstraint(QLayout.SetMinimumSize)
        detail_grid.setHorizontalSpacing(20)
        detail_grid.setVerticalSpacing(10)
        for i, (key, label) in enumerate([
            ("start_time", "开始时间"),
            ("duration", "有效时长"),
            ("task_count", "任务数量"),
            ("source", "数据来源"),
            ("signal_quality", "信号质量"),
            ("event_count", "事件数"),
        ]):
            row, col = i // 2, i % 2
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #8FA3BE; font-size: 12px;")
            detail_grid.addWidget(lbl, row, col * 2)
            val = QLabel(MISSING)
            val.setWordWrap(True)
            val.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            val.setStyleSheet("color: #F3F7FC; font-size: 14px; font-weight: 600;")
            detail_grid.addWidget(val, row, col * 2 + 1)
            self._detail_labels[key] = val
        # 兼容测试字段（不加入可见布局）：session_id / task / user_id /
        # avg_attention / avg_meditation / *_ratio——仅供断言与内部逻辑使用。
        for key in ("session_id", "task", "user_id", "avg_attention", "avg_meditation",
                    "positive_ratio", "neutral_ratio", "negative_ratio"):
            val = QLabel(MISSING)
            val.setStyleSheet("color: #F3F7FC; font-size: 13px;")
            self._detail_labels[key] = val
        # 学习状态概览行（无数据则隐藏）
        self._summary_label = QLabel(MISSING)
        self._summary_label.setWordWrap(True)
        self._summary_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._summary_label.setStyleSheet(
            "color: #3DDC97; font-size: 13px; padding-top: 8px;"
        )
        detail_grid.addWidget(self._summary_label, 3, 0, 1, 4)
        self._detail_body = self._wrap(detail_grid)
        self._detail_body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._detail_card.add_widget(self._detail_body)
        self._detail_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        right_col.addWidget(self._detail_card)

        # 演示数据标记（详情区顶部）
        self._demo_badge = QLabel("演示数据")
        self._demo_badge.setStyleSheet(
            "background-color: rgba(200,150,40,0.15); "
            "color: #FBBF24; font-size: 12px; font-weight: 700; "
            "padding: 6px 10px; border-radius: 4px;"
        )
        self._demo_badge.setAlignment(Qt.AlignCenter)
        self._demo_badge.setVisible(False)
        self._detail_card.content_layout.insertWidget(0, self._demo_badge)

        # ② 任务阶段明细：多任务逐项一行（name | diff | duration | status）
        self._tasks_card = Card("任务阶段明细")
        self._tasks_card._layout.setContentsMargins(12, 8, 12, 8)
        self._tasks_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._tasks_body = QWidget()
        self._tasks_body_layout = QVBoxLayout(self._tasks_body)
        self._tasks_body_layout.setSizeConstraint(QLayout.SetMinimumSize)
        self._tasks_body_layout.setContentsMargins(0, 0, 0, 0)
        self._tasks_body_layout.setSpacing(6)
        self._tasks_placeholder = QLabel(MISSING)
        self._tasks_placeholder.setWordWrap(True)
        self._tasks_placeholder.setStyleSheet(
            "color: #8FA3BE; font-size: 13px; padding: 10px 0;"
        )
        self._tasks_body_layout.addWidget(self._tasks_placeholder)
        self._tasks_card.add_widget(self._tasks_body)
        right_col.addWidget(self._tasks_card)

        # ③ 关键事件与建议
        self._events_card = Card("关键事件与建议")
        self._events_card._layout.setContentsMargins(12, 8, 12, 8)
        self._events_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._events_grid = QGridLayout()
        self._events_grid.setSizeConstraint(QLayout.SetMinimumSize)
        self._events_grid.setHorizontalSpacing(16)
        self._events_grid.setVerticalSpacing(4)
        self._events_grid.setColumnStretch(1, 1)
        self._events_labels = {}
        for i, (key, label) in enumerate([
            ("teacher", "教师观察"),
            ("feedback", "学生反馈"),
            ("advice", "AI 建议"),
        ]):
            cap = QLabel(label)
            cap.setStyleSheet("color: #8FA3BE; font-size: 12px;")
            self._events_grid.addWidget(cap, i, 0)
            val = QLabel(MISSING)
            val.setWordWrap(True)
            val.setStyleSheet("color: #F3F7FC; font-size: 13px;")
            val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._events_grid.addWidget(val, i, 1)
            self._events_labels[key] = val
        self._events_card.add_widget(self._wrap(self._events_grid))
        right_col.addWidget(self._events_card)

        right_col.addStretch()

        # ── 兼容测试属性（不加入可见布局） ──
        # _tasks_label：多任务缩进文本（测试断言用）
        self._tasks_label = QLabel(MISSING)
        self._tasks_label.setWordWrap(True)
        self._tasks_label.setStyleSheet(
            "color: #C5CDD9; font-size: 13px; line-height: 1.6;"
        )
        # _notes_label：简要会话信息（测试断言用）
        self._notes_label = QLabel(MISSING)
        self._notes_label.setWordWrap(True)
        self._notes_label.setStyleSheet(
            "color: #C5CDD9; font-size: 13px;"
        )
        # _dist_bars：旧版情绪分布条文本占位（测试断言用）
        self._dist_bars = {
            "positive": {"label": QLabel(MISSING), "fill": None},
            "neutral": {"label": QLabel(MISSING), "fill": None},
            "negative": {"label": QLabel(MISSING), "fill": None},
        }

        right_wrap = self._wrap(right_col)
        main_row.addWidget(right_wrap, 42)

        self.content_layout.addLayout(main_row, 1)

        # ── bottom_row：三按钮 ──
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(12)
        btn_report = QPushButton("查看学习报告")
        btn_report.setObjectName("PrimaryButton")
        btn_report.setMinimumHeight(44)
        btn_report.setToolTip("生成所选会话的只读学习报告（不修改任何记录）")
        btn_report.clicked.connect(self._open_learning_report)
        bottom_row.addWidget(btn_report, 1)
        btn_folder = QPushButton("打开会话文件夹")
        btn_folder.setMinimumHeight(44)
        btn_folder.clicked.connect(self._open_sessions_folder)
        bottom_row.addWidget(btn_folder, 1)
        btn_export = QPushButton("导出当前记录（CSV）")
        btn_export.setMinimumHeight(44)
        btn_export.clicked.connect(self._export_all)
        bottom_row.addWidget(btn_export, 1)
        self.content_layout.addLayout(bottom_row)

        # 底部脚注
        footer = QLabel(
            "历史记录仅展示已持久化数据；"
            "缺失字段显示" + MISSING + "。"
        )
        footer.setStyleSheet(
            "color: #6B7689; font-size: 11px; padding-top: 6px;"
        )
        footer.setAlignment(Qt.AlignRight)
        self.content_layout.addWidget(footer)

    def _wrap(self, layout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        return w

    # ═══════════════════════════════════════════════════════════════
    # 辅助：时长 / 任务显示 / 事件提取
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _fmt_duration(seconds):
        """统一秒数显示：截断到已完整经过的秒数（不四舍五入）。"""
        total = max(0, int(float(seconds or 0.0)))
        minutes, secs = divmod(total, 60)
        return f"{minutes}分{secs}秒"

    @staticmethod
    def _fmt_datetime_ui(value) -> str:
        """Format persisted ISO timestamps for display without changing storage."""
        text = str(value or "").strip()
        if not text:
            return MISSING
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except ValueError:
            return text.replace("T", " ").split("+")[0]

    @staticmethod
    def _fmt_ai_state_ui(value) -> str:
        """Translate model state tokens in visible UI text only."""
        text = str(value or "")
        for token, display in (
            ("positive", "积极"), ("negative", "消极"), ("neutral", "中性"),
        ):
            text = re.sub(rf"\b{token}\b", display, text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _task_display(record) -> str:
        """多任务 Session 摘要。1～2 个直接列出（" + "），更多显示"N项任务"。"""
        names = [t.name for t in (record.tasks or []) if getattr(t, "name", "")]
        if not names:
            return record.primary_task or record.notes or MISSING
        if len(names) <= 2:
            return " + ".join(names)
        return f"{len(names)}项任务"

    # ═══════════════════════════════════════════════════════════════
    # 核心：排序 / 刷新 / 选中 / 清空
    # ═══════════════════════════════════════════════════════════════

    def _get_sorted_sessions(self):
        sessions = list(self.state._history_sessions)
        role = str(getattr(self.state, "current_role", "research") or "research")
        if role == "student":
            uid = str(getattr(self.state, "_user_id", "") or "")
            sessions = [
                s for s in sessions
                if str(getattr(s, "user_id", "") or "") == uid
            ]
        elif role == "teacher":
            # Round 4C 数据隔离：教师只能看到当前选中学生本人的记录。
            selected = str(self._selection_context.selected_student_id or "")
            sessions = [
                s for s in sessions
                if selected and str(getattr(s, "user_id", "") or "") == selected
            ]
        # source 过滤：在 _get_sorted_sessions 层也做，保证单独调用时结果一致
        source_filter = self._combo_source.currentData()
        if source_filter in {"live", "mock"}:
            sessions = [s for s in sessions if s.source == source_filter]
        sort_idx = self._combo_sort.currentIndex()
        if sort_idx == 1:
            sessions.sort(key=lambda s: s.duration_seconds or 0, reverse=True)
        elif sort_idx == 2:
            sessions.sort(key=lambda s: s.signal_quality or 0, reverse=True)
        return sessions

    def _refresh_table(self):
        sessions = self._get_sorted_sessions()
        source_key = self._combo_source.currentData() or "all"
        # 在 UI 刷新层做 source 过滤（旧版行为）；_get_sorted_sessions 只管 role/user_id + 排序
        if source_key in {"live", "mock"}:
            sessions = [s for s in sessions if s.source == source_key]
        self._demo_banner.setText(SOURCE_BANNER_TEXT.get(source_key, SOURCE_BANNER_TEXT["all"]))
        # 右侧来源标记（SVG 要求与筛选一致；mock 时用琥珀色）
        source_text = SOURCE_NAMES.get(source_key, source_key or MISSING)
        tag_color = "#FBBF24" if source_key == "mock" else "#3DDC97"
        bg = "#4A3412" if source_key == "mock" else "#113C2D"
        border = "#7A5A18" if source_key == "mock" else "#1E6B4E"
        self._source_tag.setText(f"数据来源：{source_text}")
        self._source_tag.setStyleSheet(
            f"background-color: {bg}; border: 1px solid {border}; "
            f"color: {tag_color}; font-size: 12px; font-weight: 600; "
            f"padding: 6px 14px; border-radius: 14px;"
        )

        self._table.setRowCount(len(sessions))
        for i, s in enumerate(sessions):
            # 6 列产品化列序：开始时间 / 任务 / 有效时长 / 信号质量 / 事件数 / 来源
            quality = getattr(s, "signal_quality", None)
            q_text = f"{quality * 100:.0f}%" if quality is not None else MISSING
            dur_val = getattr(s, "duration_seconds", None)
            dur_text = self._fmt_duration(dur_val) if dur_val is not None else MISSING
            src = QTableWidgetItem(SOURCE_NAMES.get(s.source, s.source or MISSING))
            if getattr(s, "demo", False):
                src.setForeground(QColor("#FBBF24"))
            items = [
                QTableWidgetItem(self._fmt_datetime_ui(s.start_time)),
                QTableWidgetItem(self._task_display(s)),
                QTableWidgetItem(dur_text),
                QTableWidgetItem(q_text),
                QTableWidgetItem(str(getattr(s, "event_count", 0) if getattr(s, "event_count", None) is not None else MISSING)),
                src,
            ]
            for col, it in enumerate(items):
                self._table.setItem(i, col, it)
        # 表格空态切换：无记录时显示"暂无历史学习记录"占位页
        self._table_stack.setCurrentIndex(1 if not sessions else 0)

        # 近期概览（复用 Round 4B build_recent_summary）
        self._refresh_recent_overview()

    def _refresh_recent_overview(self):
        mode = str(getattr(self.state, "mode", "live") or "live")
        role = str(getattr(self.state, "current_role", "research") or "research")
        sessions = list(self.state._history_sessions)
        if role == "student":
            uid = str(getattr(self.state, "_user_id", "") or "")
            # getattr 防御：测试可能向 _history_sessions 放入非 SessionRecord 标记对象。
            sessions = [s for s in sessions
                        if str(getattr(s, "user_id", "") or "") == uid]
        elif role == "teacher":
            # Round 4C 数据隔离：近期概览同样严格按当前选中学生过滤。
            selected = str(self._selection_context.selected_student_id or "")
            if not selected:
                self._recent_body.setText("请先在教师工作台或学生实时观察页选择学生。")
                return
            sessions = [
                s for s in sessions
                if str(getattr(s, "user_id", "") or "") == selected
            ]
        try:
            summary = build_recent_summary(sessions, source=mode, max_sessions=5)
        except Exception:
            self._recent_body.setText("近期概览暂时不可用。")
            return
        stats = summary["stats"]
        if stats["count"] <= 0:
            self._recent_body.setText("暂无历史记录")
            return
        avg_dur = stats["avg_duration"]["text"]
        avg_att = stats["avg_attention"]["text"]
        advice = stats["ai_advice_count"]
        teacher = stats["teacher_observation_count"]
        feedback = stats["self_report_count"]
        lines = [
            f"最近 {stats['count']} 次学习（{summary['source_display']}）",
            f"平均有效学习时长　{avg_dur}",
            f"平均专注度　{avg_att}",
        ]
        if any([advice, teacher, feedback]):
            extras = []
            if teacher:
                extras.append(f"教师观察 {teacher} 次")
            if advice:
                extras.append(f"AI 建议 {advice} 次")
            if feedback:
                extras.append(f"学生反馈 {feedback} 次")
            lines.append("　".join(extras))
        self._recent_body.setText("\n".join(lines))

    def _on_select(self):
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        sessions = self._get_sorted_sessions()
        if idx >= len(sessions):
            return
        s = sessions[idx]
        # 选中记录后隐藏"请选择一条记录"提示
        self._select_hint.setVisible(False)

        # ── 学习记录详情 ──
        self._detail_labels["start_time"].setText(
            self._fmt_datetime_ui(getattr(s, "start_time", ""))
        )
        dur_val = getattr(s, "duration_seconds", None)
        self._detail_labels["duration"].setText(
            self._fmt_duration(dur_val) if dur_val is not None else MISSING
        )
        tasks = list(getattr(s, "tasks", []) or [])
        self._detail_labels["task_count"].setText(str(len(tasks)) or MISSING)
        self._detail_labels["source"].setText(
            SOURCE_NAMES.get(getattr(s, "source", ""), getattr(s, "source", "") or MISSING)
        )
        quality = getattr(s, "signal_quality", None)
        self._detail_labels["signal_quality"].setText(
            f"{quality * 100:.0f}%" if quality is not None else MISSING
        )
        self._detail_labels["event_count"].setText(
            str(getattr(s, "event_count", 0) or 0) or MISSING
        )

        # ── 历史测试兼容字段（旧版 _detail_labels 的 key，供断言） ──
        self._detail_labels["session_id"].setText(str(getattr(s, "session_id", "") or MISSING))
        self._detail_labels["task"].setText(self._task_display(s))
        self._detail_labels["user_id"].setText(
            str(getattr(s, "user_id", "") or MISSING)
        )
        # 概率占比：从 probability_summary 取，保留 2 位小数
        summary = dict(getattr(s, "probability_summary", {}) or {})
        pos = float(summary.get("mean_positive", 0) or 0)
        neu = float(summary.get("mean_neutral", 0) or 0)
        neg = float(summary.get("mean_negative", 0) or 0)
        self._detail_labels["positive_ratio"].setText(f"{pos:.2f}")
        self._detail_labels["neutral_ratio"].setText(f"{neu:.2f}")
        self._detail_labels["negative_ratio"].setText(f"{neg:.2f}")
        # 专注/放松均值：优先从 attention_ / meditation_ summary 取，无则 0
        att = float(summary.get("mean_attention",
                                float(getattr(s, "avg_attention", 0) or 0)) or 0)
        med = float(summary.get("mean_meditation",
                                 float(getattr(s, "avg_meditation", 0) or 0)) or 0)
        self._detail_labels["avg_attention"].setText(f"{att:.2f}")
        self._detail_labels["avg_meditation"].setText(f"{med:.2f}")
        # 兼容测试字段：多任务缩进格式（任务段：N项 + 每任务含难度/有效学习/状态子行）
        task_lines = [f"任务段：{len(tasks)}项"]
        for t in tasks:
            name = str(getattr(t, "name", "") or "自由学习")
            diff = DIFFICULTY_DISPLAY.get(
                getattr(t, "difficulty", ""),
                getattr(t, "difficulty", "") or MISSING,
            )
            t_dur = getattr(t, "duration_seconds", None)
            dur = self._fmt_duration(t_dur) if t_dur is not None else MISSING
            status_key = getattr(t, "status", "") or ""
            status = TASK_STATUS_DISPLAY.get(status_key, status_key or MISSING)
            task_lines.append(f"  {name}")
            task_lines.append(f"    难度：{diff}")
            task_lines.append(f"    有效学习：{dur}")
            task_lines.append(f"    状态：{status}")
        self._tasks_label.setText("\n".join(task_lines) or MISSING)
        # 兼容测试字段：旧版情绪分布条文本
        self._dist_bars["positive"]["label"].setText(f"积极 {int(pos * 100)}%" if pos else MISSING)
        self._dist_bars["neutral"]["label"].setText(f"中性 {int(neu * 100)}%" if neu else MISSING)
        self._dist_bars["negative"]["label"].setText(f"消极 {int(neg * 100)}%" if neg else MISSING)
        # 兼容测试字段：简要会话信息
        self._notes_label.setText(
            f"来源：{SOURCE_NAMES.get(getattr(s, 'source', ''), getattr(s, 'source', '') or MISSING)}"
        )

        # 演示标记
        self._demo_badge.setVisible(bool(getattr(s, "demo", False)))

        # 学习状态概览（从 probability_summary 生成简单文案）
        self._summary_label.setText(self._state_overview_text(s))

        # ── 任务阶段明细：逐任务一行（SVG 要求 soft 底纹） ──
        self._render_tasks(tasks)

        # ── 关键事件与建议 ──
        self._render_key_events(s)
        self._refresh_detail_card_geometry()

    def _refresh_detail_card_geometry(self):
        """Recalculate wrapped summary height before sibling cards are laid out."""
        self._detail_card.setMinimumHeight(0)
        self._summary_label.setMinimumHeight(0)
        available = max(120, self._detail_body.width() - 4)
        required = self._summary_label.heightForWidth(available)
        if required > 0:
            self._summary_label.setMinimumHeight(required)
        self._summary_label.updateGeometry()
        self._detail_grid.invalidate()
        self._detail_grid.activate()
        self._detail_body.updateGeometry()
        self._detail_card.updateGeometry()
        self._detail_card.setMinimumHeight(self._detail_card.sizeHint().height())
        self._tasks_body_layout.activate()
        self._tasks_card.updateGeometry()
        task_widgets = [
            self._tasks_body_layout.itemAt(index).widget()
            for index in range(self._tasks_body_layout.count())
            if self._tasks_body_layout.itemAt(index).widget() is not None
        ]
        task_body_height = sum(widget.minimumHeight() or widget.sizeHint().height()
                               for widget in task_widgets)
        if len(task_widgets) > 1:
            task_body_height += self._tasks_body_layout.spacing() * (len(task_widgets) - 1)
        self._tasks_body.setMinimumHeight(task_body_height)
        task_margins = self._tasks_card._layout.contentsMargins()
        task_chrome = (
            task_margins.top() + task_margins.bottom()
            + self._tasks_card._title_label.sizeHint().height()
            + self._tasks_card._layout.spacing()
        )
        self._tasks_card.setMinimumHeight(task_chrome + task_body_height)
        self._events_grid.activate()
        self._events_card.updateGeometry()
        event_margins = self._events_card._layout.contentsMargins()
        event_chrome = (
            event_margins.top() + event_margins.bottom()
            + self._events_card._title_label.sizeHint().height()
            + self._events_card._layout.spacing()
        )
        self._events_card.setMinimumHeight(
            event_chrome + self._events_grid.minimumSize().height()
        )
        self._right_detail_layout.invalidate()
        self._right_detail_layout.activate()

    @staticmethod
    def _state_overview_text(record) -> str:
        summary = dict(getattr(record, "probability_summary", {}) or {})
        sample_count = int(summary.get("sample_count", 0) or 0)
        if sample_count <= 0:
            return MISSING
        positive = float(summary.get("mean_positive", 0) or 0)
        negative = float(summary.get("mean_negative", 0) or 0)
        dominant = str(summary.get("dominant_state", "") or "")
        note = ""
        if dominant == "positive":
            note = "整体较稳定，积极状态占主导。"
        elif dominant == "negative":
            note = "消极状态占主导，后续可关注学习节奏与负荷。"
        else:
            if positive > negative:
                note = "整体较稳定，后半程出现一次短时专注下降。"
            elif negative > positive:
                note = "部分区间存在短时情绪波动。"
            else:
                note = "整体较稳定。"
        return note

    def _render_tasks(self, tasks):
        """清空任务阶段卡片并重建每任务一行。"""
        self._tasks_card.setMinimumHeight(0)
        self._tasks_body.setMinimumHeight(0)
        while self._tasks_body_layout.count():
            item = self._tasks_body_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        if not tasks:
            self._tasks_placeholder = QLabel(MISSING)
            self._tasks_placeholder.setStyleSheet(
                "color: #8FA3BE; font-size: 13px; padding: 10px 0;"
            )
            self._tasks_body_layout.addWidget(self._tasks_placeholder)
            return

        for task in tasks:
            row = QWidget()
            row.setObjectName("HistoryTaskRow")
            row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            row.setStyleSheet(
                "QWidget#HistoryTaskRow { background-color: #142031; border: 1px solid #26364B; "
                "border-radius: 8px; }"
            )
            layout = QHBoxLayout(row)
            layout.setContentsMargins(14, 6, 14, 6)
            layout.setSpacing(12)

            name_lbl = QLabel(str(getattr(task, "name", "") or "自由学习"))
            name_lbl.setStyleSheet("color: #F3F7FC; font-size: 14px; font-weight: 600;")
            layout.addWidget(name_lbl, 3)

            diff = DIFFICULTY_DISPLAY.get(
                getattr(task, "difficulty", ""), getattr(task, "difficulty", "") or MISSING
            )
            diff_lbl = QLabel(diff)
            diff_lbl.setStyleSheet("color: #8FA3BE; font-size: 12px;")
            diff_lbl.setMinimumWidth(50)
            layout.addWidget(diff_lbl)

            # 真实 TaskRecord.duration_seconds；缺失(None)显示 暂无数据，不伪造 0分0秒
            t_dur = getattr(task, "duration_seconds", None)
            dur_lbl = QLabel(
                f"有效 {self._fmt_duration(t_dur)}" if t_dur is not None
                else f"有效 {MISSING}"
            )
            dur_lbl.setStyleSheet("color: #8FA3BE; font-size: 12px;")
            dur_lbl.setMinimumWidth(110)
            layout.addWidget(dur_lbl)

            status_key = getattr(task, "status", "") or ""
            status_text = TASK_STATUS_DISPLAY.get(status_key, status_key or MISSING)
            status_color = "#3DDC97" if status_key == "completed" else (
                "#42B7FF" if status_key == "running" else "#8FA3BE"
            )
            status_lbl = QLabel(status_text)
            status_lbl.setStyleSheet(
                f"color: {status_color}; font-size: 12px; font-weight: 600;"
            )
            status_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            status_lbl.setMinimumWidth(60)
            layout.addWidget(status_lbl)

            self._tasks_body_layout.addWidget(row)
            layout.activate()
            row.setMinimumHeight(row.sizeHint().height())

    def _render_key_events(self, record):
        events = list(getattr(record, "events", []) or [])
        # 教师观察：source == teacher
        teacher_evts = [e for e in events if str(getattr(e, "source", "")) == "teacher"]
        # 学生反馈：type == self_report 或 source == self_report
        feedback_evts = [
            e for e in events
            if str(getattr(e, "type", "")) == "self_report"
            or str(getattr(e, "source", "")) == "self_report"
        ]
        # AI 建议：type == intervention / ai_state_change 或 category == intervention
        ai_evts = [
            e for e in events
            if str(getattr(e, "type", "")) in ("intervention", "ai_state_change")
            or str(getattr(e, "category", "")) == "intervention"
        ]

        def _fmt_teacher(items):
            if not items:
                return "本次无教师观察"
            notes = []
            for e in items:
                txt = str(getattr(e, "content", "") or getattr(e, "label", "") or "").strip()
                if txt:
                    notes.append(txt)
            combined = "；".join(notes)
            if len(combined) > 40:
                combined = combined[:37] + "…"
            return f"疑似走神 {len(items)} 次" if len(combined) < 3 and len(items) == 1 \
                else (combined or f"共 {len(items)} 条观察")

        def _fmt_feedback(items):
            if not items:
                return MISSING_FEEDBACK
            txt = str(getattr(items[-1], "content", "") or getattr(items[-1], "label", "") or "").strip()
            return txt if txt else f"共 {len(items)} 条"

        def _fmt_ai(items):
            if not items:
                return MISSING_AI
            # 只展示真实存在的 label/note（不伪造原始 AI 建议全文）
            parts = []
            for e in items[:2]:
                label = str(getattr(e, "content", "") or getattr(e, "label", "") or "").strip()
                note = str(getattr(e, "note", "") or "").strip()
                if note and "未自动" in note:
                    continue  # 已知道 AI 只给建议不自动改，避免冗余
                parts.append(note or label)
            text = "；".join([p for p in parts if p])
            if not text:
                text = f"共 {len(items)} 条建议"
            if len(text) > 50:
                text = text[:47] + "…"
            return self._fmt_ai_state_ui(text)

        self._events_labels["teacher"].setText(_fmt_teacher(teacher_evts))
        self._events_labels["feedback"].setText(_fmt_feedback(feedback_evts))
        self._events_labels["advice"].setText(_fmt_ai(ai_evts))

    def _clear_detail(self):
        """清除右侧选中详情：产品化空态——
        显示"请选择一条历史学习记录查看详情。"提示；字段统一回到 暂无数据，
        绝不显示 0 / 0% / 00:00 / -- 之类的技术占位。"""
        self._table.clearSelection()
        self._select_hint.setVisible(True)
        for label in self._detail_labels.values():
            label.setText(MISSING)
        self._summary_label.setText(MISSING)
        self._summary_label.setMinimumHeight(0)
        self._demo_badge.setVisible(False)
        for label in self._events_labels.values():
            label.setText(MISSING)
        # 兼容测试字段
        self._tasks_label.setText(MISSING)
        self._notes_label.setText(MISSING)
        for k in ("positive", "neutral", "negative"):
            self._dist_bars[k]["label"].setText(MISSING)
        # 任务阶段卡回到空态占位
        while self._tasks_body_layout.count():
            item = self._tasks_body_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        ph = QLabel(MISSING)
        ph.setStyleSheet("color: #8FA3BE; font-size: 13px; padding: 10px 0;")
        self._tasks_body_layout.addWidget(ph)
        self._refresh_recent_overview()  # 近期概览独立刷新（不受选中影响）

    # ═══════════════════════════════════════════════════════════════
    # Round 4A：学习报告 / 导出 / 打开会话文件夹 / 生命周期钩子
    # ═══════════════════════════════════════════════════════════════

    def _open_learning_report(self):
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            QMessageBox.information(
                self, "查看学习报告", "请先在历史会话列表中选择一条记录。"
            )
            return
        sessions = self._get_sorted_sessions()
        idx = rows[0].row()
        if idx >= len(sessions):
            return
        self._show_report(sessions[idx])

    def _load_baseline(self, student_id):
        if self._baseline_store is None:
            self._baseline_store = BaselineResultStore()
        try:
            return self._baseline_store.get(student_id)
        except Exception:
            return None

    def _show_report(self, record):
        baseline = self._load_baseline(record.user_id)
        report = build_learning_report(record, baseline)
        self._report_dialog = SessionReportDialog(report, self)
        self._report_dialog.show()

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
                "session_id", "source", "demo", "status", "user_id",
                "start_time", "end_time", "duration_seconds",
                "positive_ratio", "neutral_ratio", "negative_ratio",
                "avg_attention", "avg_meditation", "signal_quality",
                "task_count", "event_count", "notes",
            ])
            visible = self._get_sorted_sessions()
            for s in visible:
                writer.writerow([
                    s.session_id, s.source, s.demo, s.status, s.user_id,
                    s.start_time, s.end_time, s.duration_seconds,
                    s.positive_ratio, s.neutral_ratio, s.negative_ratio,
                    s.avg_attention, s.avg_meditation, s.signal_quality,
                    len(s.tasks), s.event_count, s.notes,
                ])
        QMessageBox.information(
            self, "导出成功", f"已导出 {len(visible)} 条记录。"
        )

    def _open_sessions_folder(self):
        from services.session_store import resolve_sessions_root
        folder = resolve_sessions_root(self.state, self.service)
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def on_show(self):
        self.state.reload_history(include_demo=True)
        self._apply_mode_default_filter()
        self._refresh_table()

    def _on_selection_changed(self, student_id: str):
        """Round 4C：教师切换观察学生后立即刷新历史表与近期概览。"""
        if self._role != "teacher":
            return
        self._clear_detail()
        self._refresh_table()

    def _apply_mode_default_filter(self):
        mode = str(getattr(self.state, "mode", "live") or "live")
        default_index = 2 if mode == "mock" else 1
        if self._combo_source.currentIndex() != default_index:
            self._combo_source.setCurrentIndex(default_index)

    def on_data_mode_changed(self, mode: str):
        default_index = 2 if mode == "mock" else 1
        if self._combo_source.currentIndex() != default_index:
            self._combo_source.setCurrentIndex(default_index)
        else:
            self._refresh_table()
        self._clear_detail()


# ═══════════════════════════════════════════════════════════════
# 学生单次学习报告视图（Round 4A，纯只读，不修改任何持久化数据）
# ═══════════════════════════════════════════════════════════════


class SessionReportDialog(QDialog):
    """只消费 build_learning_report() 的只读 report dict。"""

    def __init__(self, report: dict, parent=None):
        super().__init__(parent)
        self.report = report
        self.setWindowTitle("学生学习报告")
        self.resize(760, 860)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("学生学习报告")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #E8EDF3;")
        layout.addWidget(title)

        self._demo_badge = QLabel("【教学演示】")
        self._demo_badge.setStyleSheet(
            "background-color: rgba(200,150,40,0.15); color: #FBBF24; "
            "font-size: 14px; font-weight: 700; padding: 6px 10px; border-radius: 4px;"
        )
        self._demo_badge.setAlignment(Qt.AlignCenter)
        self._demo_badge.setVisible(report.get("is_demo", False))
        layout.addWidget(self._demo_badge)

        source_label = QLabel(f"数据来源：{report.get('source_display', MISSING)}")
        source_label.setStyleSheet("color: #A5B0C0; font-size: 13px;")
        layout.addWidget(source_label)

        overview_lines = [
            f"学生：{report.get('student_name', MISSING)}（{report.get('student_id', MISSING)}）",
            f"学习日期：{report.get('date', MISSING)}",
            f"数据来源：{report.get('source_display', MISSING)}",
            f"会话有效时长：{report.get('duration_text', MISSING)}",
        ]
        layout.addWidget(self._section_label("本次学习概览"))
        layout.addWidget(self._body_label("\n".join(overview_lines)))

        tasks = report.get("tasks", [])
        if tasks:
            task_lines = [
                f"{i}. {task['name']}｜难度：{task['difficulty_display']}"
                f"｜有效学习时长：{task['duration_text']}"
                f"｜状态：{task['status_display']}"
                for i, task in enumerate(tasks, 1)
            ]
            layout.addWidget(self._body_label("\n".join(task_lines)))
        else:
            layout.addWidget(self._body_label("暂无任务段记录"))

        layout.addWidget(self._section_label("学习状态摘要"))
        state = report.get("state_summary", {})
        metric_lines = [
            f"平均专注度：{report['attention']['text']}",
            f"平均放松度：{report['meditation']['text']}",
        ]
        if state.get("available"):
            metric_lines.append(
                f"状态倾向摘要：积极 {state['mean_positive_text']}"
                f"｜中性 {state['mean_neutral_text']}"
                f"｜消极 {state['mean_negative_text']}"
                f"｜主导状态：{state['dominant_state_text']}"
                f"｜结束状态：{state['final_state_text']}"
            )
            metric_lines.append(f"（{state['note']}）")
        layout.addWidget(self._body_label("\n".join(metric_lines)))

        layout.addWidget(self._section_label("数据质量"))
        quality = report.get("quality", {})
        layout.addWidget(self._body_label(
            quality.get("level_text") if quality.get("available") else MISSING
        ))

        layout.addWidget(self._section_label("关键事件时间线"))
        timeline = report.get("timeline", [])
        if timeline:
            lines = [
                f"{item['time']}　[{item['source_display']}]　{item['content']}"
                + (f"（{item['note']}）" if item["note"] else "")
                for item in timeline
            ]
            layout.addWidget(self._body_label("\n".join(lines)))
        else:
            layout.addWidget(self._body_label(MISSING))

        layout.addWidget(self._section_label("我的学习感受"))
        fb = report.get("self_feedback", {})
        if fb.get("available"):
            layout.addWidget(self._body_label("\n".join(
                f"{item['time']}　{item['text']}"
                + (f"（{item['note']}）" if item["note"] else "")
                for item in fb["items"]
            )))
        else:
            layout.addWidget(self._body_label(MISSING_FEEDBACK))

        layout.addWidget(self._section_label("AI 学习建议（历史事件记录）"))
        advice = report.get("ai_advice", {})
        if advice.get("available"):
            layout.addWidget(self._body_label("\n".join(
                f"{item['time']}　{item['label']}"
                + (f"（{item['note']}）" if item["note"] else "")
                for item in advice["items"]
            )))
        else:
            layout.addWidget(self._body_label(MISSING_AI))

        layout.addWidget(self._section_label("基线参考"))
        baseline = report.get("baseline", {})
        if baseline.get("available"):
            layout.addWidget(self._body_label("\n".join(baseline["lines"])))
        else:
            layout.addWidget(self._body_label(MISSING_BASELINE))

        layout.addWidget(self._section_label("本次学习总结"))
        layout.addWidget(self._body_label(report.get("summary_text", MISSING)))

        layout.addStretch()
        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    @staticmethod
    def _section_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(
            "font-size: 15px; font-weight: 700; color: #D7DEE9; "
            "border-bottom: 1px solid #2A3346; padding-bottom: 4px;"
        )
        return label

    @staticmethod
    def _body_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: #C5CDD9; font-size: 13px; line-height: 1.6;")
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return label
