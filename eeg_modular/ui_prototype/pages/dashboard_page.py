"""页面3：实时学习仪表盘（核心页面）。

严格遵循 eeg_modular/ui_prototype/services/dashboard_state.py 中定义的
DashboardState 正式字段接口。UI 业务逻辑只消费正式字段，
内部簿记字段（_前缀）仅用于图表缓冲绘制。
"""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QInputDialog,
    QMessageBox, QFileDialog, QComboBox, QGridLayout, QFrame,
    QDialog, QListWidget, QLineEdit, QSizePolicy, QStackedWidget,
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
    DIFFICULTY_DISPLAY, SessionRecord,
)
from services.session_store import SessionStore
from services.student_history_summary import build_recent_summary
from services.learning_readiness import (
    baseline_advisory_confirmed,
    learning_start_block_reason,
)
from services.identity_store import IdentityStore
from services.teaching_store import (
    StudentRuntimeRegistry, TeacherSelectionContext, TeacherStudentStore,
)
from services.notification_service import NotificationService, NotificationSound


_DIFFICULTY_VALUES = [DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD]
# 包根目录：相对 sessions_dir 的解析基准（与 ReplayPage 一致）。
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


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
        self.notification_service = NotificationService(self.runtime_registry.path)
        self._session_started = False
        self._session_paused = False
        super().__init__(scrollable=True)
        self._teacher_poll_timer = QTimer(self)
        self._teacher_poll_timer.setInterval(500)
        self._teacher_poll_timer.timeout.connect(self._poll_teacher_runtime)
        self._student_notification_timer = QTimer(self)
        self._student_notification_timer.setInterval(500)
        self._student_notification_timer.timeout.connect(self._poll_student_notifications)
        self._notification_sound = NotificationSound(self)
        self._build_ui()
        self._notification_toast = QLabel(self)
        self._notification_toast.setObjectName("TeacherNotificationToast")
        self._notification_toast.setWordWrap(True)
        self._notification_toast.setMinimumWidth(340)
        self._notification_toast.setMaximumWidth(440)
        self._notification_toast.setStyleSheet(
            "QLabel#TeacherNotificationToast{background:#22324A;color:#F8FAFC;"
            "border:1px solid #4F7DB8;border-radius:10px;padding:12px 16px;"
            "font-size:14px;font-weight:600;}"
        )
        self._notification_toast.hide()
        self.set_role(self._role)
        # Round 4C：任何入口切换观察学生后，本页立即同步刷新。
        self.selection_context.add_listener(self, "_on_selection_changed")

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
        self._student_subtitle = QLabel("查看当前学习状态、专注趋势与系统建议。")
        self._student_subtitle.setStyleSheet("color: #8491A5; font-size: 13px;")
        self._student_subtitle.setVisible(False)
        layout.addWidget(self._student_subtitle)
        self._teacher_subtitle = QLabel(
            "查看当前学生的学习状态、趋势与教学提示；教师端仅观察，不直接控制学生设备。"
        )
        self._teacher_subtitle.setStyleSheet("color: #8491A5; font-size: 13px;")
        self._teacher_subtitle.setVisible(False)
        layout.addWidget(self._teacher_subtitle)
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
        self._legacy_warmup_card = warmup_card
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
        self._legacy_eeg_card = eeg_card
        self._eeg_plot = EEGPlotWidget()
        self._eeg_plot.setMinimumHeight(100)
        eeg_card.add_widget(self._eeg_plot)
        left_col.addWidget(eeg_card, 1)

        trend_card = Card("专注度 / 放松度趋势（90秒）")
        self._legacy_trend_card = trend_card
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
        self._legacy_prob_card = prob_card
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
        self._legacy_att_card = att_card
        self._att_gauge = ArcGauge("专注度", "#4FC3F7")
        self._att_gauge.setFixedSize(72, 72)
        att_card.add_widget(self._wrap_centered(self._att_gauge))
        gauge_row.addWidget(att_card, 1)

        med_card = Card("放松度")
        self._legacy_med_card = med_card
        self._med_gauge = ArcGauge("放松度", "#4ADE80")
        self._med_gauge.setFixedSize(72, 72)
        med_card.add_widget(self._wrap_centered(self._med_gauge))
        gauge_row.addWidget(med_card, 1)

        right_col.addLayout(gauge_row, 3)

        # 持续状态
        sustain_card = Card("最近90秒稳定状态")
        self._legacy_sustain_card = sustain_card
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

        self._legacy_main_widget = self._wrap(main_row)
        layout.addWidget(self._legacy_main_widget, 1)

        # ── AI建议 ──
        ai_card = Card("AI学习建议")
        self._legacy_ai_card = ai_card
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

        # Round 4A-2：学生端不再提供任务/难度编辑入口（下拉框已删除），
        # 改为只读"当前学习任务"面板；任务选择只在"我的学习任务"页进行。
        self._learner_task_info = QLabel("当前暂无进行中的学习任务")
        self._learner_task_info.setWordWrap(True)
        self._learner_task_info.setVisible(False)
        self._learner_task_info.setStyleSheet(
            "color: #E8EDF3; font-size: 13px; font-weight: 600; line-height: 1.7;"
        )
        self._learner_task_info.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self._role_grid_widget = self._wrap(role_grid)
        self._role_card.add_widget(self._role_grid_widget)
        self._role_card.add_widget(self._learner_task_info)
        self._role_card.setMaximumHeight(98)
        layout.addWidget(self._role_card)

        self._student_home = self._build_student_home()
        self._student_home.setVisible(False)
        layout.addWidget(self._student_home, 1)

        self._teacher_home = self._build_teacher_home()
        self._teacher_home.setVisible(False)
        layout.addWidget(self._teacher_home, 1)

        self._admin_diagnostic_home = self._build_admin_diagnostic_home()
        self._admin_diagnostic_home.setVisible(False)
        layout.addWidget(self._admin_diagnostic_home, 1)

        # ── Round 4B：教师端"学生近期学习概览"（纯读取） ──
        self._recent_card = Card("学生近期学习概览")
        recent_header = QHBoxLayout()
        self._recent_source_combo = QComboBox()
        self._recent_source_combo.addItem("实时采集", "live")
        self._recent_source_combo.addItem("教学演示", "mock")
        self._recent_source_combo.currentIndexChanged.connect(
            self._on_recent_source_changed
        )
        recent_header.addWidget(QLabel("数据来源："))
        recent_header.addWidget(self._recent_source_combo)
        btn_recent_refresh = QPushButton("刷新")
        btn_recent_refresh.clicked.connect(self._load_recent_overview)
        recent_header.addStretch()
        recent_header.addWidget(btn_recent_refresh)
        self._recent_card.add_widget(self._wrap(recent_header))

        self._recent_overview_label = QLabel("请选择学生后查看近期学习概览")
        self._recent_overview_label.setWordWrap(True)
        self._recent_overview_label.setStyleSheet(
            "color: #C5CDD9; font-size: 13px; line-height: 1.7;"
        )
        self._recent_overview_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._recent_card.add_widget(self._recent_overview_label)
        self._recent_card.setVisible(False)
        layout.addWidget(self._recent_card)

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

    def _build_admin_diagnostic_home(self) -> QWidget:
        """管理员设备与模型诊断中心；只投影现有状态，不参与数据处理。"""
        home = QWidget()
        home.setObjectName("AdminModelDiagnosticHome")
        home.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root = QVBoxLayout(home)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        status_row = QHBoxLayout()
        status_row.setSpacing(10)
        self._admin_status_cards = {}
        for key, title in (
            ("connector", "ThinkGear 状态"),
            ("device", "MindWave 状态"),
            ("poor", "Poor Signal"),
            ("quality", "信号质量"),
            ("model", "模型状态"),
        ):
            card = QFrame()
            card.setObjectName("AdminModelDiagnosticCard")
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(14, 10, 14, 10)
            card_layout.setSpacing(3)
            caption = QLabel(title)
            caption.setObjectName("AdminModelDiagnosticCaption")
            value = QLabel("暂无数据")
            value.setObjectName("AdminModelDiagnosticStatus")
            detail = QLabel("不可评估")
            detail.setObjectName("AdminModelDiagnosticMuted")
            card_layout.addWidget(caption)
            card_layout.addWidget(value)
            card_layout.addWidget(detail)
            status_row.addWidget(card, 1)
            self._admin_status_cards[key] = {
                "frame": card, "value": value, "detail": detail,
            }
        root.addLayout(status_row)

        diagnostic_row = QHBoxLayout()
        diagnostic_row.setSpacing(10)

        eeg_card = QFrame()
        self._admin_eeg_card = eeg_card
        eeg_card.setObjectName("AdminModelDiagnosticCard")
        eeg_layout = QVBoxLayout(eeg_card)
        eeg_layout.setContentsMargins(16, 12, 16, 12)
        eeg_layout.setSpacing(6)
        eeg_header = QHBoxLayout()
        eeg_header.addWidget(self._admin_diagnostic_title("EEG 实时状态"))
        eeg_header.addStretch()
        self._admin_eeg_state = QLabel("等待设备数据")
        self._admin_eeg_state.setObjectName("AdminModelDiagnosticBadge")
        eeg_header.addWidget(self._admin_eeg_state)
        eeg_layout.addLayout(eeg_header)
        self._admin_eeg_plot = EEGPlotWidget()
        self._admin_eeg_plot.setMinimumHeight(150)
        self._admin_eeg_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        eeg_layout.addWidget(self._admin_eeg_plot, 1)
        self._admin_eeg_footer = QLabel("目标采样率：512 Hz · 等待设备数据")
        self._admin_eeg_footer.setObjectName("AdminModelDiagnosticMuted")
        eeg_layout.addWidget(self._admin_eeg_footer)
        diagnostic_row.addWidget(eeg_card, 56)

        model_card = QFrame()
        self._admin_model_card = model_card
        model_card.setObjectName("AdminModelDiagnosticCard")
        model_layout = QVBoxLayout(model_card)
        model_layout.setContentsMargins(16, 12, 16, 12)
        model_layout.setSpacing(7)
        model_layout.addWidget(self._admin_diagnostic_title("模型状态与数据质量"))
        metrics = QHBoxLayout()
        metrics.setSpacing(8)
        self._admin_metric_values = {}
        for key, title in (("attention", "Attention"), ("meditation", "Meditation")):
            box = QFrame()
            box.setObjectName("AdminModelDiagnosticInset")
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(12, 7, 12, 7)
            caption = QLabel(title)
            caption.setObjectName("AdminModelDiagnosticMuted")
            value = QLabel("暂无数据")
            value.setObjectName("AdminModelDiagnosticMetric")
            box_layout.addWidget(caption)
            box_layout.addWidget(value)
            metrics.addWidget(box, 1)
            self._admin_metric_values[key] = value
        model_layout.addLayout(metrics)

        probability_title = QLabel("三分类概率")
        probability_title.setObjectName("AdminModelDiagnosticCaption")
        model_layout.addWidget(probability_title)
        probability_grid = QGridLayout()
        probability_grid.setHorizontalSpacing(10)
        probability_grid.setVerticalSpacing(4)
        self._admin_probability_values = {}
        for row, (key, title) in enumerate((
            ("positive", "积极"), ("neutral", "中性"), ("negative", "消极"),
        )):
            caption = QLabel(title)
            caption.setObjectName("AdminModelDiagnosticMuted")
            value = QLabel("暂无数据")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value.setObjectName("AdminModelDiagnosticField")
            probability_grid.addWidget(caption, row, 0)
            probability_grid.addWidget(value, row, 1)
            self._admin_probability_values[key] = value
        probability_grid.setColumnStretch(1, 1)
        model_layout.addLayout(probability_grid)
        self._admin_model_quality = QLabel("当前模型与数据质量状态：不可评估")
        self._admin_model_quality.setWordWrap(True)
        self._admin_model_quality.setObjectName("AdminModelDiagnosticSummary")
        model_layout.addWidget(self._admin_model_quality)
        diagnostic_row.addWidget(model_card, 44)
        root.addLayout(diagnostic_row, 5)

        trend_card = QFrame()
        self._admin_trend_card = trend_card
        trend_card.setObjectName("AdminModelDiagnosticCard")
        trend_layout = QVBoxLayout(trend_card)
        trend_layout.setContentsMargins(16, 10, 16, 10)
        trend_layout.setSpacing(4)
        trend_layout.addWidget(self._admin_diagnostic_title("90 秒趋势分析"))
        self._admin_trend_stack = QStackedWidget()
        self._admin_trend_plot = TrendPlotWidget()
        self._admin_trend_plot.setMinimumHeight(80)
        self._admin_trend_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        self._admin_trend_empty = QLabel("等待设备数据")
        self._admin_trend_empty.setAlignment(Qt.AlignCenter)
        self._admin_trend_empty.setObjectName("AdminModelDiagnosticMuted")
        self._admin_trend_stack.addWidget(self._admin_trend_plot)
        self._admin_trend_stack.addWidget(self._admin_trend_empty)
        trend_layout.addWidget(self._admin_trend_stack, 1)
        root.addWidget(trend_card, 3)

        record_card = QFrame()
        self._admin_record_card = record_card
        record_card.setObjectName("AdminModelDiagnosticCard")
        record_layout = QVBoxLayout(record_card)
        record_layout.setContentsMargins(16, 9, 16, 9)
        record_layout.setSpacing(5)
        record_layout.addWidget(self._admin_diagnostic_title("诊断记录与运行信息"))
        record_grid = QGridLayout()
        record_grid.setHorizontalSpacing(18)
        record_grid.setVerticalSpacing(3)
        self._admin_runtime_values = {}
        for column, (key, title) in enumerate((
            ("record", "诊断记录"), ("source", "数据来源"),
            ("warmup", "模型预热"), ("session", "运行状态"),
        )):
            caption = QLabel(title)
            caption.setObjectName("AdminModelDiagnosticMuted")
            value = QLabel("暂无数据")
            value.setObjectName("AdminModelDiagnosticField")
            record_grid.addWidget(caption, 0, column)
            record_grid.addWidget(value, 1, column)
            record_grid.setColumnStretch(column, 1)
            self._admin_runtime_values[key] = value
        record_layout.addLayout(record_grid)
        root.addWidget(record_card, 0)

        home.setStyleSheet("""
            QWidget#AdminModelDiagnosticHome { background: transparent; }
            QFrame#AdminModelDiagnosticCard {
                background: #162235; border: 1px solid #2B3A50; border-radius: 12px;
            }
            QFrame#AdminModelDiagnosticInset {
                background: #1C293D; border: 1px solid #2B3A50; border-radius: 8px;
            }
            QLabel#AdminModelDiagnosticTitle { color: #E8EDF3; font-size: 15px; font-weight: 700; }
            QLabel#AdminModelDiagnosticCaption { color: #9FB0C6; font-size: 12px; }
            QLabel#AdminModelDiagnosticMuted { color: #8495AD; font-size: 11px; }
            QLabel#AdminModelDiagnosticStatus { color: #E8EDF3; font-size: 17px; font-weight: 700; }
            QLabel#AdminModelDiagnosticMetric { color: #E8EDF3; font-size: 19px; font-weight: 700; }
            QLabel#AdminModelDiagnosticField { color: #D8E1EC; font-size: 12px; font-weight: 600; }
            QLabel#AdminModelDiagnosticBadge {
                color: #FBBF24; background: #3A301D; border: 1px solid #80661F;
                border-radius: 6px; padding: 3px 8px; font-weight: 700;
            }
            QLabel#AdminModelDiagnosticSummary {
                color: #B8C7D9; background: #1C293D; border: 1px solid #2B3A50;
                border-radius: 7px; padding: 6px 8px; font-size: 11px;
            }
        """)
        return home

    @staticmethod
    def _admin_diagnostic_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("AdminModelDiagnosticTitle")
        return label

    def _build_student_home(self) -> QWidget:
        """Build the student-only presentation without changing dashboard services."""
        home = QWidget()
        home.setObjectName("StudentRealtimeHome")
        home.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root = QVBoxLayout(home)
        root.setContentsMargins(0, 2, 0, 0)
        root.setSpacing(10)

        home.setStyleSheet("""
            QFrame[studentCard="true"] {
                background: #1B2433;
                border: 1px solid #2D394C;
                border-radius: 12px;
            }
            QLabel[studentCaption="true"] { color: #8E9BAD; font-size: 12px; }
            QLabel[studentValue="true"] { color: #F1F5F9; font-size: 20px; font-weight: 700; }
            QLabel[studentHint="true"] { color: #9AA7B8; font-size: 12px; }
            QLabel[sectionTitle="true"] { color: #EAF0F7; font-size: 15px; font-weight: 700; }
        """)

        # Five compact state summaries. They deliberately use only labels so their
        # geometry is independent from the diagnostic StatusIndicator widget.
        self._student_status_cards = {}
        status_wrap = QWidget()
        self._student_status_row = status_wrap
        status_layout = QHBoxLayout(status_wrap)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(10)
        for index, (key, title) in enumerate((
            ("device", "设备连接"), ("contact", "接触质量"),
            ("analysis", "状态分析"), ("task", "当前任务"),
            ("state", "当前学习状态"),
        )):
            frame = QFrame()
            frame.setProperty("studentCard", True)
            frame.setMinimumHeight(88)
            card_layout = QVBoxLayout(frame)
            card_layout.setContentsMargins(14, 10, 14, 9)
            card_layout.setSpacing(2)
            caption = QLabel(title)
            caption.setProperty("studentCaption", True)
            value = QLabel("暂无数据")
            value.setProperty("studentValue", True)
            value.setMinimumHeight(28)
            hint = QLabel("等待实时状态")
            hint.setProperty("studentHint", True)
            hint.setWordWrap(True)
            hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            card_layout.addWidget(caption)
            card_layout.addWidget(value)
            card_layout.addWidget(hint)
            self._student_status_cards[key] = {
                "frame": frame, "value": value, "hint": hint,
            }
            status_layout.addWidget(frame, 7 if index == 4 else 6)
        root.addWidget(status_wrap, 0)

        charts_wrap = QWidget()
        self._student_charts_row = charts_wrap
        charts_wrap.setMinimumHeight(280)
        charts_wrap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        charts = QHBoxLayout(charts_wrap)
        charts.setContentsMargins(0, 0, 0, 0)
        charts.setSpacing(10)

        trend_card = QFrame()
        self._student_charts_left = trend_card
        trend_card.setProperty("studentCard", True)
        trend_layout = QVBoxLayout(trend_card)
        trend_layout.setContentsMargins(15, 11, 15, 10)
        trend_layout.setSpacing(3)
        trend_title = QLabel("专注度 / 放松度趋势")
        trend_title.setProperty("sectionTitle", True)
        trend_note = QLabel("最近 90 秒")
        trend_note.setProperty("studentHint", True)
        self._student_trend_empty = QLabel("等待设备数据")
        self._student_trend_empty.setAlignment(Qt.AlignCenter)
        self._student_trend_empty.setProperty("studentHint", True)
        self._student_trend_plot = TrendPlotWidget()
        self._student_trend_plot.setMinimumHeight(180)
        self._student_trend_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        trend_layout.addWidget(trend_title)
        trend_layout.addWidget(trend_note)
        trend_layout.addWidget(self._student_trend_empty)
        trend_layout.addWidget(self._student_trend_plot, 1)
        charts.addWidget(trend_card, 5)

        stable_card = QFrame()
        self._student_charts_right = stable_card
        stable_card.setProperty("studentCard", True)
        stable_layout = QVBoxLayout(stable_card)
        stable_layout.setContentsMargins(15, 11, 15, 10)
        stable_layout.setSpacing(4)
        stable_title = QLabel("最近 90 秒稳定状态")
        stable_title.setProperty("sectionTitle", True)
        self._student_stable_value = QLabel("暂无稳定状态")
        self._student_stable_value.setProperty("studentValue", True)
        self._student_stable_hint = QLabel("等待形成可解释的学习状态")
        self._student_stable_hint.setProperty("studentHint", True)
        self._student_negative_duration = QLabel("消极趋势持续：暂无数据")
        self._student_negative_duration.setStyleSheet("color: #D6A756; font-size: 12px;")
        self._student_prob_empty = QLabel("暂无可用状态概率")
        self._student_prob_empty.setAlignment(Qt.AlignCenter)
        self._student_prob_empty.setProperty("studentHint", True)
        self._student_prob_trend = ProbabilityTrendWidget()
        self._student_prob_trend.setMinimumHeight(125)
        self._student_prob_trend.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        stable_layout.addWidget(stable_title)
        stable_layout.addWidget(self._student_stable_value)
        stable_layout.addWidget(self._student_stable_hint)
        stable_layout.addWidget(self._student_negative_duration)
        stable_layout.addWidget(self._student_prob_empty)
        stable_layout.addWidget(self._student_prob_trend, 1)
        charts.addWidget(stable_card, 3)
        root.addWidget(charts_wrap, 1)

        advice_card = QFrame()
        self._student_advice_card = advice_card
        advice_card.setProperty("studentCard", True)
        advice_layout = QHBoxLayout(advice_card)
        advice_layout.setContentsMargins(15, 10, 15, 10)
        advice_layout.setSpacing(16)
        advice_title = QLabel("AI 学习建议")
        advice_title.setProperty("sectionTitle", True)
        advice_title.setMinimumWidth(105)
        self._student_advice_label = QLabel("等待信号稳定后将生成学习建议。")
        self._student_advice_label.setWordWrap(True)
        self._student_advice_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._student_advice_label.setStyleSheet("color: #D8E0EA; font-size: 13px;")
        advice_note = QLabel("建议仅供学习节奏参考，不会自动修改教师发布的任务。")
        advice_note.setWordWrap(True)
        advice_note.setProperty("studentHint", True)
        advice_note.setMaximumWidth(340)
        advice_layout.addWidget(advice_title)
        advice_layout.addWidget(self._student_advice_label, 1)
        advice_layout.addWidget(advice_note)
        root.addWidget(advice_card, 0)

        bottom_wrap = QWidget()
        self._student_bottom_row = bottom_wrap
        bottom = QHBoxLayout(bottom_wrap)
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(10)
        task_card = QFrame()
        self._student_task_card = task_card
        task_card.setProperty("studentCard", True)
        task_layout = QVBoxLayout(task_card)
        task_layout.setContentsMargins(15, 10, 15, 10)
        task_layout.setSpacing(3)
        task_title = QLabel("当前学习任务")
        task_title.setProperty("sectionTitle", True)
        task_head = QHBoxLayout()
        self._student_task_name = QLabel("当前暂无进行中的学习任务")
        self._student_task_name.setProperty("studentValue", True)
        self._student_task_name.setStyleSheet("font-size: 16px; font-weight: 700;")
        task_button = QPushButton("前往我的学习任务")
        task_button.clicked.connect(self._navigate_to_task_page)
        task_head.addWidget(self._student_task_name, 1)
        task_head.addWidget(task_button)
        self._student_task_detail = QLabel("开始学习后将在这里显示任务难度与有效学习时间。")
        self._student_task_detail.setProperty("studentHint", True)
        self._student_task_detail.setWordWrap(True)
        self._student_teacher_reminder = QLabel("最近教师提醒：暂无")
        self._student_teacher_reminder.setWordWrap(True)
        self._student_teacher_reminder.setStyleSheet(
            "color:#F6C85F;font-size:12px;background:#2B2A24;"
            "border:1px solid #554B2C;border-radius:6px;padding:5px 8px;"
        )
        task_layout.addWidget(task_title)
        task_layout.addLayout(task_head)
        task_layout.addWidget(self._student_task_detail)
        task_layout.addWidget(self._student_teacher_reminder)
        bottom.addWidget(task_card, 5)

        def gauge_card(title: str, color: str):
            frame = QFrame()
            frame.setProperty("studentCard", True)
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(12, 8, 12, 7)
            frame_layout.setSpacing(1)
            label = QLabel(title)
            label.setProperty("sectionTitle", True)
            gauge = ArcGauge(title, color)
            gauge.setFixedSize(90, 90)
            empty = QLabel("暂无数据")
            empty.setAlignment(Qt.AlignCenter)
            empty.setProperty("studentHint", True)
            frame_layout.addWidget(label)
            frame_layout.addWidget(gauge, 1, Qt.AlignCenter)
            frame_layout.addWidget(empty, 1, Qt.AlignCenter)
            return frame, gauge, empty

        self._student_att_card, self._student_att_gauge, self._student_att_empty = gauge_card(
            "专注度", "#4FC3F7"
        )
        self._student_med_card, self._student_med_gauge, self._student_med_empty = gauge_card(
            "放松度", "#4ADE80"
        )
        bottom.addWidget(self._student_att_card, 2)
        bottom.addWidget(self._student_med_card, 2)
        root.addWidget(bottom_wrap, 0)
        return home

    def _build_teacher_home(self) -> QWidget:
        """教师专属只读观察面板；仅消费现有学生 runtime snapshot。"""
        home = QWidget()
        root = QVBoxLayout(home)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        context_card = Card("")
        self._teacher_context_card = context_card
        context = QHBoxLayout()
        context.setSpacing(14)
        student_box = QVBoxLayout(); student_box.setSpacing(3)
        student_box.addWidget(QLabel("当前观察学生"))
        student_row = QHBoxLayout(); student_row.setSpacing(6)
        student_row.addWidget(self._teacher_student_combo, 1)
        student_row.addWidget(self._btn_manage_students)
        student_box.addLayout(student_row); context.addLayout(student_box, 3)
        self._teacher_context_values = {}
        for key, title in (("live", "实时状态"), ("elapsed", "有效学习"),
                           ("task", "当前任务"), ("updated", "最近更新")):
            box = QVBoxLayout(); box.setSpacing(3); caption = QLabel(title)
            caption.setStyleSheet("color:#8491A5;font-size:11px;")
            value = QLabel("--"); value.setWordWrap(True)
            value.setStyleSheet("color:#E8EDF3;font-size:14px;font-weight:600;")
            box.addWidget(caption); box.addWidget(value); context.addLayout(box, 2)
            self._teacher_context_values[key] = value
        context_card.add_widget(self._wrap(context)); root.addWidget(context_card)
        context_card.setMaximumHeight(96)

        self._teacher_status_row = QWidget()
        status = QHBoxLayout(self._teacher_status_row); status.setContentsMargins(0,0,0,0); status.setSpacing(8)
        self._teacher_status_cards = {}
        for key, title, stretch in (("connector","ThinkGear Connector",1),
                                    ("device","MindWave 设备",1),
                                    ("quality","接触 / 信号质量",1),
                                    ("analysis","当前分析状态",2)):
            card = Card(title); value = QLabel("--"); hint = QLabel("暂无实时数据")
            value.setStyleSheet("font-size:18px;font-weight:700;color:#E8EDF3;")
            hint.setStyleSheet("font-size:11px;color:#8491A5;"); hint.setWordWrap(True)
            card.add_widget(value); card.add_widget(hint); status.addWidget(card, stretch)
            self._teacher_status_cards[key] = {"frame":card,"value":value,"hint":hint}
        self._teacher_status_row.setMaximumHeight(92); root.addWidget(self._teacher_status_row)

        self._teacher_main_row = QWidget()
        main = QHBoxLayout(self._teacher_main_row); main.setContentsMargins(0,0,0,0); main.setSpacing(10)
        trend_card = Card("实时学习趋势")
        self._teacher_trend_card = trend_card
        trend_body = QVBoxLayout(); trend_body.setSpacing(6)
        trend_hint = QLabel("最近 90 秒 · Attention / Meditation / 学习状态")
        trend_hint.setStyleSheet("color:#8491A5;font-size:11px;"); trend_body.addWidget(trend_hint)
        self._teacher_eeg_plot = EEGPlotWidget(); self._teacher_eeg_plot.setMaximumHeight(72)
        # 教师页的 EEG 图高度较紧凑：缩短纵轴标题并固定轴宽，
        # 避免中文纵标题与刻度互相挤压，不改变绘图数据。
        eeg_axis = self._teacher_eeg_plot._plot.getAxis("left")
        eeg_axis.setLabel("幅度", color="#6B7689", **{"font-size": "9px"})
        eeg_axis.setWidth(42)
        self._teacher_eeg_empty = QLabel("等待学生实时数据")
        self._teacher_eeg_empty.setAlignment(Qt.AlignCenter); self._teacher_eeg_empty.setStyleSheet("color:#8491A5;")
        self._teacher_eeg_stack = QStackedWidget()
        self._teacher_eeg_stack.setMaximumHeight(72)
        self._teacher_eeg_stack.addWidget(self._teacher_eeg_plot)
        self._teacher_eeg_stack.addWidget(self._teacher_eeg_empty)
        eeg_wrap = QVBoxLayout(); eeg_wrap.setSpacing(2); eeg_wrap.addWidget(QLabel("EEG 最近 5 秒"))
        eeg_wrap.addWidget(self._teacher_eeg_stack)
        trend_body.addLayout(eeg_wrap)
        self._teacher_trend_plot = TrendPlotWidget(); self._teacher_trend_plot.setMinimumHeight(145)
        self._teacher_trend_empty = QLabel("等待学生实时数据")
        self._teacher_trend_empty.setAlignment(Qt.AlignCenter); self._teacher_trend_empty.setStyleSheet("color:#8491A5;")
        # 空状态与图表互斥显示：等待文字居中时不再与坐标轴同层。
        self._teacher_trend_stack = QStackedWidget()
        self._teacher_trend_stack.addWidget(self._teacher_trend_plot)
        self._teacher_trend_stack.addWidget(self._teacher_trend_empty)
        trend_body.addWidget(self._teacher_trend_stack, 1)
        trend_card.add_widget(self._wrap(trend_body)); main.addWidget(trend_card, 5)

        state_card = Card("当前学习状态")
        self._teacher_state_card = state_card
        state_body = QVBoxLayout(); state_body.setSpacing(6)
        self._teacher_dominant = QLabel("主导状态：--")
        self._teacher_dominant.setStyleSheet("font-size:22px;font-weight:700;color:#E8EDF3;")
        self._teacher_interpretable = QLabel("状态：等待有效分析窗口")
        self._teacher_interpretable.setStyleSheet("color:#FBBF24;font-size:12px;")
        state_body.addWidget(self._teacher_dominant); state_body.addWidget(self._teacher_interpretable)
        self._teacher_prob_panel = ProbabilityPanel(); state_body.addWidget(self._teacher_prob_panel)
        metrics = QGridLayout(); metrics.setSpacing(6)
        self._teacher_att = QLabel("--"); self._teacher_med = QLabel("--")
        self._teacher_negative = QLabel("负性趋势持续：--")
        self._teacher_state_quality = QLabel("当前信号质量：暂无数据")
        metrics.addWidget(QLabel("专注度"),0,0); metrics.addWidget(QLabel("放松度"),0,1)
        metrics.addWidget(self._teacher_att,1,0); metrics.addWidget(self._teacher_med,1,1)
        metrics.addWidget(self._teacher_negative,2,0,1,2); metrics.addWidget(self._teacher_state_quality,3,0,1,2)
        state_body.addLayout(metrics); state_card.add_widget(self._wrap(state_body)); main.addWidget(state_card, 3)
        self._teacher_main_row.setMaximumHeight(296); root.addWidget(self._teacher_main_row, 1)

        self._teacher_bottom_row = QWidget()
        bottom = QHBoxLayout(self._teacher_bottom_row); bottom.setContentsMargins(0,0,0,0); bottom.setSpacing(8)
        advice_card = Card("AI 学习建议"); self._teacher_advice_card = advice_card
        self._teacher_advice_text = QLabel("暂无有效建议"); self._teacher_advice_text.setWordWrap(True)
        self._teacher_advice_text.setStyleSheet("color:#C5CDD9;font-size:12px;"); advice_card.add_widget(self._teacher_advice_text)
        bottom.addWidget(advice_card, 3)
        process_card = Card("教师端 · 学生过程概览"); self._teacher_process_card = process_card
        self._teacher_process_text = QLabel("当前任务状态：暂无进行中任务\n最近事件：暂无\n最近更新时间：--\n智能建议：暂无有效建议")
        self._teacher_process_text.setWordWrap(True); self._teacher_process_text.setStyleSheet("color:#C5CDD9;font-size:11px;")
        process_card.add_widget(self._teacher_process_text); bottom.addWidget(process_card, 3)
        recent_card = Card("近期学习概览"); self._teacher_recent_summary_card = recent_card
        self._teacher_recent_summary = QLabel("暂无近期学习记录"); self._teacher_recent_summary.setWordWrap(True)
        self._teacher_recent_summary.setStyleSheet("color:#C5CDD9;font-size:11px;"); recent_card.add_widget(self._teacher_recent_summary)
        bottom.addWidget(recent_card, 3)
        self._teacher_bottom_row.setMaximumHeight(130); root.addWidget(self._teacher_bottom_row)
        return home

    def _navigate_to_task_page(self):
        window = self.window()
        if hasattr(window, "_navigate_to"):
            window._navigate_to("task")

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
        self._teacher_student_label.setVisible(False)
        self._teacher_student_combo.setVisible(teacher)
        self._btn_manage_students.setVisible(teacher)
        # Round 4B：近期学习概览仅教师可见（只读统计）。
        if hasattr(self, "_recent_card"):
            self._recent_card.setVisible(False)
        # 学生端：隐藏通用网格，显示只读"当前学习任务"面板；
        # 任务/难度编辑入口已按 Round 4A-2 移除。
        self._role_grid_widget.setVisible(not is_student)
        self._learner_task_info.setVisible(is_student)
        legacy_cards = (
            self._card_connector["frame"], self._card_device["frame"],
            self._card_poor["frame"], self._card_conf["frame"],
            self._card_rate["frame"], self._legacy_warmup_card,
            self._analysis_card, self._legacy_eeg_card,
            self._legacy_trend_card, self._legacy_prob_card,
            self._legacy_att_card, self._legacy_med_card,
            self._legacy_sustain_card, self._legacy_ai_card, self._role_card,
            self._legacy_main_widget,
        )
        for widget in legacy_cards:
            widget.setVisible(False)
        self._student_home.setVisible(is_student)
        self._teacher_home.setVisible(teacher)
        self._admin_diagnostic_home.setVisible(not is_student and not teacher)
        self._student_subtitle.setVisible(is_student)
        self._teacher_subtitle.setVisible(teacher)
        if is_student:
            self._page_title.setText("实时学习状态")
            self._diagnostic_note.setVisible(False)
            self._card_poor["frame"].set_title("接触质量")
            self._btn_start.setText("开始学习记录")
            self._role_card.set_title("当前学习任务")
            self._role_card.setMaximumHeight(150)
            titles = ["学习任务", "当前建议", "难度调整"]
        elif teacher:
            self._page_title.setText("学生实时观察")
            self._diagnostic_note.setVisible(False)
            self._card_poor["frame"].set_title("接触质量")
            self._role_card.set_title("教师端 · 学生过程概览")
            self._role_card.setMaximumHeight(98)
            titles = ["班级状态趋势", "异常提醒", "过程记录"]
        else:
            self._page_title.setText("本机设备与模型诊断")
            self._diagnostic_note.setVisible(True)
            self._card_poor["frame"].set_title("Poor Signal 原始值")
            self._btn_start.setText("开始本机诊断记录")
            self._role_card.set_title("管理端 · 运行概览")
            self._role_card.setMaximumHeight(98)
            titles = ["设备", "模型与数据质量", "实验记录"]
            self._sync_admin_diagnostic_home(self.state)
        can_control = self._role != "teacher"
        for button in (
            self._btn_start, self._btn_pause, self._btn_end,
        ):
            button.setVisible(can_control)
        self._session_time.setVisible(not teacher)
        self._session_toolbar.setVisible(not teacher)
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
        if is_student:
            self._student_notification_timer.start()
            self._poll_student_notifications()
        else:
            self._student_notification_timer.stop()
            self._notification_toast.hide()

    def _poll_student_notifications(self):
        if self._role != "student":
            return
        student_id = str(getattr(self.state, "_user_id", "") or "")
        notifications = self.notification_service.take_pending(student_id)
        if not notifications:
            return
        latest = notifications[-1]
        timestamp = time.strftime("%H:%M:%S", time.localtime(latest.created_at))
        self._student_teacher_reminder.setText(
            f"最近教师提醒：{latest.label}　{timestamp}"
        )
        suffix = f"\n另有 {len(notifications) - 1} 条新提醒" if len(notifications) > 1 else ""
        self._notification_toast.setText(f"教师提醒：{latest.label}{suffix}")
        self._notification_toast.adjustSize()
        self._position_notification_toast()
        self._notification_toast.show()
        self._notification_toast.raise_()
        self._notification_sound.play()
        QTimer.singleShot(5000, self._notification_toast.hide)

    def _position_notification_toast(self):
        if not hasattr(self, "_notification_toast"):
            return
        margin = 22
        self._notification_toast.move(
            max(margin, self.width() - self._notification_toast.width() - margin),
            margin,
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_notification_toast()

    def _poll_teacher_runtime(self):
        if self._role == "teacher":
            self._refresh_teacher_students()
            self._update_teacher_snapshot()

    def _refresh_teacher_students(self):
        teacher_id = getattr(self.state, "_user_id", "")
        # Round 4C：以 TeacherSelectionContext 为唯一数据源（不回退旧 combo 值）。
        current = self.selection_context.selected_student_id
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
            # Round 4B：切换观察学生后立即刷新近期学习概览（只读）。
            self._load_recent_overview()

    def _on_selection_changed(self, student_id: str):
        """Round 4C：其他页面切换观察学生后，立即同步本页全部学生视图。

        学生下拉框本身由 500ms 轮询 / update_state 负责回填；
        此处只刷新依赖当前学生的数据视图（快照 + 近期概览）。
        """
        if self._role != "teacher":
            return
        self._update_teacher_snapshot()
        self._load_recent_overview()

    # ── Round 4B：学生近期学习概览（纯读取） ──

    def _on_recent_source_changed(self):
        if self._role == "teacher":
            self._load_recent_overview()

    def _current_observed_student_id(self) -> str:
        # Round 4C：以 TeacherSelectionContext 为唯一数据源（严格数据隔离），
        # 下拉框仅作为选择入口，不作为学生身份的来源。
        return str(self.selection_context.selected_student_id or "")

    def _load_recent_overview(self):
        """加载当前观察学生的近期学习概览（只读，不写任何 History）。

        默认统计实时采集记录；教师主动选择"教学演示"时统计演示记录，
        并在概览中明确标注【教学演示数据】。Demo 与 Live 不混合计算。
        """
        if self._role != "teacher":
            return
        student_id = self._current_observed_student_id()
        if not student_id:
            self._recent_overview_label.setText("请选择学生后查看近期学习概览")
            self._teacher_recent_summary.setText("暂无近期学习记录")
            return
        source = str(self._recent_source_combo.currentData() or "live")
        try:
            # 与 ReplayPage/History 相同的目录解析：service.sessions_dir
            # 相对路径基于包根目录，直接可读已有 session.json 历史。
            folder = Path(getattr(self.service, "sessions_dir", Path("data/sessions")))
            if not folder.is_absolute():
                folder = _PACKAGE_ROOT / folder
            store = SessionStore(folder)
            payloads = store.load(include_demo=(source == "mock"))
            records = [
                SessionRecord.from_dict(p) for p in payloads
                if str(p.get("user_id") or "") == student_id
            ]
            summary = build_recent_summary(records, source=source)
        except Exception:
            self._recent_overview_label.setText("近期学习概览暂时不可用。")
            self._teacher_recent_summary.setText("近期学习概览暂时不可用")
            return
        self._recent_overview_label.setText(self._render_recent_summary(summary))
        stats = summary["stats"]
        if not stats["count"]:
            self._teacher_recent_summary.setText("暂无近期学习记录")
        else:
            self._teacher_recent_summary.setText(
                f"最近 {stats['count']} 次\n"
                f"平均专注度：{stats['avg_attention']['text']}　学习次数：{stats['count']}\n"
                f"教师观察：{stats['teacher_observation_count']} 次　"
                f"AI 建议：{stats['ai_advice_count']} 次\n"
                + (" ".join(summary["performance_summary"]) or "暂无趋势描述")
            )

    @staticmethod
    def _render_recent_summary(summary: dict) -> str:
        stats = summary["stats"]
        lines = []
        if summary.get("is_demo"):
            lines.append("【教学演示数据】")
        lines.append(
            f"统计范围：最近 {stats['count']} 次"
            f"{summary['source_display']}学习记录"
        )
        for i, row in enumerate(summary["sessions"], 1):
            attention = row["attention"]["text"]
            lines.append(
                f"{i}. {row['date']}｜{row['task_summary']}｜"
                f"{row['duration_text']}｜平均专注度 {attention}｜"
                f"AI建议 {row['ai_advice_count']} 次｜"
                f"教师观察 {row['teacher_observation_count']} 次"
            )
        if stats["count"]:
            lines.append(
                f"近期统计：学习次数 {stats['count']}｜"
                f"平均有效学习时长 {stats['avg_duration']['text']}｜"
                f"平均专注度 {stats['avg_attention']['text']}｜"
                f"AI建议 {stats['ai_advice_count']} 次｜"
                f"教师观察 {stats['teacher_observation_count']} 次｜"
                f"学生反馈 {stats['self_report_count']} 次"
            )
        lines.append(f"事件时间观察：{summary['event_timing']['text']}")
        lines.append("近期表现：" + " ".join(summary["performance_summary"]))
        if summary["teaching_suggestions"]:
            lines.append("教学参考：" + " ".join(summary["teaching_suggestions"]))
        return "\n".join(lines)

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
            self._teacher_context_values["live"].setText(
                "学生端未运行" if snapshot else "暂无实时数据"
            )
            self._teacher_context_values["elapsed"].setText("--")
            self._teacher_context_values["task"].setText("暂无进行中任务")
            self._teacher_context_values["updated"].setText("--")
            for item in self._teacher_status_cards.values():
                item["value"].setText("--"); item["hint"].setText("暂无实时数据")
            self._teacher_dominant.setText("主导状态：--")
            self._teacher_interpretable.setText("状态：等待有效分析窗口")
            for bar in self._teacher_prob_panel._bars.values():
                bar.set_value(0.0); bar.set_dimmed(True)
            self._teacher_att.setText("--"); self._teacher_med.setText("--")
            self._teacher_negative.setText("负性趋势持续：--")
            self._teacher_state_quality.setText("当前信号质量：暂无数据")
            self._teacher_advice_text.setText("暂无有效建议")
            self._teacher_process_text.setText(
                "当前任务状态：暂无进行中任务\n最近事件：暂无\n"
                "最近更新时间：--\n智能建议：暂无有效建议"
            )
            self._teacher_eeg_plot.reset(); self._teacher_trend_plot.reset()
            self._teacher_eeg_stack.setCurrentWidget(self._teacher_eeg_empty)
            self._teacher_trend_stack.setCurrentWidget(self._teacher_trend_empty)
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
            # Round 4C：无快照/超时也必须重置"学生过程概览"，
            # 否则切换学生后会残留上一个学生的 ID（数据隔离）。
            self._focus_values[0].setText(
                f"学生：{student_id or '未选择'} · 状态：暂无实时数据"
            )
            self._focus_values[1].setText("最后更新：--")
            self._focus_values[2].setText("智能建议：--")
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
        updated_text = time.strftime("%H:%M:%S", time.localtime(updated)) if updated else "--"
        task_name = snapshot.get("task_name") or "暂无进行中任务"
        self._teacher_context_values["live"].setText("在线观察中" if online else "学生端未运行")
        self._teacher_context_values["elapsed"].setText(
            f"{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"
        )
        self._teacher_context_values["task"].setText(task_name)
        self._teacher_context_values["updated"].setText(updated_text)
        status_values = {
            "connector": ("已连接" if online else "未连接", "在线" if online else "离线"),
            "device": ("在线" if online else "离线", "设备正常" if online else "暂无实时数据"),
            "quality": (quality, "可用于解释" if snapshot.get("quality_level") == "trusted" else "需要关注"),
            "analysis": (stable if stable != "--" else "等待分析", snapshot.get("advice") or "暂无有效建议"),
        }
        for key, (value, hint) in status_values.items():
            self._teacher_status_cards[key]["value"].setText(value)
            self._teacher_status_cards[key]["hint"].setText(hint)
        interpretable = snapshot.get("quality_level") != "rejected" and stable != "--"
        self._teacher_dominant.setText(f"主导状态：{stable if interpretable else '--'}")
        self._teacher_interpretable.setText("状态：可信" if interpretable else "状态：等待有效分析窗口")
        for name, bar in self._teacher_prob_panel._bars.items():
            value = probabilities.get(name)
            bar.set_value(value or 0.0); bar.set_dimmed(value is None or not interpretable)
        self._teacher_att.setText("--" if snapshot.get("attention") is None else str(round(snapshot["attention"])))
        self._teacher_med.setText("--" if snapshot.get("meditation") is None else str(round(snapshot["meditation"])))
        negative_seconds = snapshot.get("negative_sustain_seconds")
        self._teacher_negative.setText(
            "负性趋势持续：--" if negative_seconds is None else f"负性趋势持续：{float(negative_seconds):.1f} 秒"
        )
        self._teacher_state_quality.setText(f"当前信号质量：{quality}")
        advice = snapshot.get("advice") or "暂无有效建议"
        self._teacher_advice_text.setText(advice)
        recent_events = snapshot.get("recent_events") or []
        recent_label = "暂无" if not recent_events else str(
            (recent_events[-1].get("label") if isinstance(recent_events[-1], dict)
             else getattr(recent_events[-1], "label", "暂无")) or "暂无"
        )
        self._teacher_process_text.setText(
            f"当前任务状态：{task_name}\n最近事件：{recent_label}\n"
            f"最近更新时间：{updated_text}\n智能建议：{advice}"
        )
        self._teacher_eeg_stack.setCurrentWidget(
            self._teacher_eeg_plot if raw else self._teacher_eeg_empty
        )
        self._teacher_eeg_plot.reset()
        if raw: self._teacher_eeg_plot.push_display_points(raw)
        self._teacher_trend_stack.setCurrentWidget(
            self._teacher_trend_plot
            if att_history and med_history else self._teacher_trend_empty
        )
        self._teacher_trend_plot.reset()
        for att, med in zip(att_history, med_history):
            self._teacher_trend_plot.push_values(att, med)
        self._focus_values[0].setText(f"学生：{student_id} · 状态：{stable}")
        self._focus_values[1].setText(
            "最后更新：" + time.strftime("%H:%M:%S", time.localtime(updated))
        )
        self._focus_values[2].setText(
            f"智能建议：{snapshot.get('advice') or '暂无建议'}"
        )

    def _current_task_record(self):
        """只读获取当前进行中的 TaskRecord（不创建、不修改）。"""
        task_id = getattr(self.state, "current_task_id", "")
        if not task_id:
            return None
        return next(
            (t for t in reversed(getattr(self.state, "_tasks", []))
             if getattr(t, "task_id", "") == task_id),
            None,
        )

    def _refresh_learner_task_info(self, state):
        """Round 4A-2：学生端只读"当前学习任务"面板。

        任务来源语义：TaskRecord.assignment_id 非空 → 教师布置；否则自主学习。
        学生不能通过本页面选择/修改任务或难度（编辑入口已移除）。
        """
        task = self._current_task_record()
        if task is None or not getattr(state, "task_running", False):
            self._learner_task_info.setText(
                "当前暂无进行中的学习任务\n请前往“我的学习任务”开始学习"
            )
            return
        difficulty = DIFFICULTY_DISPLAY.get(task.difficulty, task.difficulty)
        source = "教师布置" if getattr(task, "assignment_id", "") else "自主学习"
        # 与任务页/History/报告一致的截断秒数规则
        elapsed = int(max(0.0, float(getattr(state, "current_task_elapsed_seconds", 0.0))))
        elapsed_text = f"{elapsed // 60}分{elapsed % 60}秒"
        advice = (
            getattr(state, "adaptive_feedback_text", "")
            or getattr(state, "feedback_text", "")
            or "暂无建议"
        )
        self._learner_task_info.setText(
            f"任务：{task.name}\n"
            f"来源：{source}\n"
            f"难度：{difficulty}\n"
            f"有效学习时间：{elapsed_text}\n"
            f"AI学习建议：{advice}"
        )

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
            # Round 4A-2：学生端只读任务面板（无任务/难度编辑控件）
            self._refresh_learner_task_info(state)
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
        # Round 4A-2：基线不再是绝对门槛（require_baseline=False），
        # 缺基线时改为推荐提示（去采集基线 / 暂时跳过）。
        reason = (
            learning_start_block_reason(self.state, require_baseline=False)
            if self._role == "student" else ""
        )
        if reason and self.isVisible():
            QMessageBox.warning(self, "暂不能开始学习", reason)
            return
        if self._role == "student" and self.isVisible() and not baseline_advisory_confirmed(self.state, self):
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

    def _set_student_status(self, key: str, value: str, hint: str, tone: str = "neutral"):
        item = self._student_status_cards[key]
        color = {
            "good": "#4ADE80", "warn": "#FBBF24", "error": "#F87171",
            "neutral": "#E8EDF3",
        }.get(tone, "#E8EDF3")
        item["value"].setText(value)
        item["value"].setStyleSheet(
            f"color: {color}; font-size: 20px; font-weight: 700;"
        )
        item["hint"].setText(hint)

    def _update_student_home(self, state, analysis_kind: str):
        """Map existing runtime state into the student-only, read-only view."""
        elapsed = int(max(0.0, float(
            getattr(state, "current_task_elapsed_seconds", 0.0)
        )))
        self._session_time.setText(f"有效学习 {elapsed // 60:02d}:{elapsed % 60:02d}")

        connector = getattr(state, "connector_status", "offline")
        device = getattr(state, "device_status", "offline")
        if connector == "online" and device == "online":
            self._set_student_status("device", "已连接", "设备数据持续接收中", "good")
        elif connector == "online" and device == "waiting_raw":
            self._set_student_status("device", "等待数据", "设备已连接，等待脑电数据", "warn")
        elif connector == "connecting":
            self._set_student_status("device", "连接中", "正在连接设备服务", "warn")
        else:
            self._set_student_status("device", "未连接", "请检查设备连接", "error")

        poor = getattr(state, "poor_signal", None)
        if poor is None:
            self._set_student_status("contact", "暂无数据", "等待接触质量数据")
        elif poor < MAX_POOR_SIGNAL:
            self._set_student_status("contact", "良好", "当前信号可用于分析", "good")
        elif poor < 200:
            self._set_student_status("contact", "需要调整", "请调整传感器位置", "warn")
        else:
            self._set_student_status("contact", "无信号", "请重新佩戴设备", "error")

        if analysis_kind == "error":
            self._set_student_status("analysis", "暂不可用", "分析服务暂不可用", "error")
        elif analysis_kind == "rejected":
            self._set_student_status("analysis", "暂不可解释", "当前信号不足以分析", "warn")
        elif analysis_kind == "warming":
            progress = int(max(0.0, min(1.0, getattr(state, "warmup_progress", 0.0))) * 100)
            self._set_student_status("analysis", "正在准备", f"分析预热 {progress}%", "warn")
        elif getattr(state, "inference_eligible", False):
            self._set_student_status("analysis", "已就绪", "学习状态持续分析中", "good")
        else:
            self._set_student_status("analysis", "等待分析", "正在积累有效数据", "warn")

        task = self._current_task_record()
        task_running = bool(getattr(state, "task_running", False)) and task is not None
        if task_running:
            difficulty = DIFFICULTY_DISPLAY.get(task.difficulty, task.difficulty)
            self._set_student_status("task", task.name, f"难度：{difficulty}", "good")
            self._student_task_name.setText(task.name)
            self._student_task_detail.setText(
                f"当前任务难度：{difficulty}　·　有效学习时间："
                f"{elapsed // 60:02d}:{elapsed % 60:02d}"
            )
        else:
            self._set_student_status("task", "暂无任务", "请前往任务页开始学习")
            self._student_task_name.setText("当前暂无进行中的学习任务")
            self._student_task_detail.setText("请前往“我的学习任务”选择并开始学习。")

        interpretable = (
            getattr(state, "inference_eligible", False)
            and analysis_kind not in {"error", "rejected"}
        )
        stable = getattr(state, "stable_state", "unknown")
        student_state_names = {
            "positive": "积极", "neutral": "中性", "negative": "消极",
        }
        if interpretable and stable in student_state_names:
            tone = {"positive": "good", "neutral": "neutral", "negative": "warn"}[stable]
            self._set_student_status(
                "state", student_state_names[stable], "基于最近有效信号", tone
            )
            self._student_stable_value.setText(student_state_names[stable])
            self._student_stable_value.setStyleSheet(
                "font-size: 20px; font-weight: 700; color: "
                + {"positive": "#4ADE80", "neutral": "#4FC3F7", "negative": "#FBBF24"}[stable]
                + ";"
            )
            self._student_stable_hint.setText("最近 90 秒有效分析的稳定状态")
            self._student_negative_duration.setText(
                f"消极趋势持续：{float(getattr(state, '_negative_sustain_seconds', 0.0)):.1f} 秒"
            )
        else:
            hint = "当前信号不可解释" if analysis_kind == "rejected" else "等待形成可解释的学习状态"
            self._set_student_status("state", "暂无状态", hint, "warn" if analysis_kind == "rejected" else "neutral")
            self._student_stable_value.setText("暂无稳定状态")
            self._student_stable_value.setStyleSheet(
                "font-size: 20px; font-weight: 700; color: #8E9BAD;"
            )
            self._student_stable_hint.setText(hint)
            self._student_negative_duration.setText("消极趋势持续：暂无数据")

        att = getattr(state, "attention", None)
        med = getattr(state, "meditation", None)
        has_metrics = att is not None and med is not None
        self._student_trend_empty.setVisible(not has_metrics)
        if has_metrics:
            self._student_trend_plot.push_values(att, med)
        for gauge, empty, value in (
            (self._student_att_gauge, self._student_att_empty, att),
            (self._student_med_gauge, self._student_med_empty, med),
        ):
            gauge.setVisible(value is not None)
            empty.setVisible(value is None)
            if value is not None:
                gauge.set_value(value)

        prob_history = getattr(state, "_prob_history", [])
        has_probability = interpretable and bool(prob_history)
        self._student_prob_empty.setVisible(not has_probability)
        if has_probability:
            latest = prob_history[-1]
            self._student_prob_trend.push_values(latest[1], latest[2], latest[3])

        if analysis_kind == "error":
            advice = "分析服务暂不可用，当前不生成学习状态建议。"
        else:
            advice = (
                getattr(state, "adaptive_feedback_text", "")
                or getattr(state, "feedback_text", "")
                or "等待信号稳定后将生成学习建议。"
            )
        self._student_advice_label.setText(advice)

    def _sync_admin_diagnostic_home(self, state):
        """把正式 DashboardState 投影到管理员诊断布局，不修改任何状态。"""
        if not hasattr(self, "_admin_diagnostic_home"):
            return

        def status_card(key, value, detail, tone="neutral"):
            colors = {
                "good": "#4ADE80", "warn": "#FBBF24",
                "error": "#F87171", "info": "#4FC3F7", "neutral": "#E8EDF3",
            }
            item = self._admin_status_cards[key]
            item["value"].setText(value)
            item["value"].setStyleSheet(
                f"color:{colors[tone]};font-size:17px;font-weight:700;"
            )
            item["detail"].setText(detail)

        connector = str(getattr(state, "connector_status", "offline") or "offline")
        if connector == "online":
            status_card("connector", "已连接", "ThinkGear Connector", "good")
        elif connector == "connecting":
            status_card("connector", "连接中", "等待本机服务响应", "warn")
        else:
            status_card("connector", "未连接", "本机服务不可用", "error")

        device = str(getattr(state, "device_status", "offline") or "offline")
        has_raw = bool(getattr(state, "_eeg_raw_buffer", []))
        if device == "online" and has_raw:
            status_card("device", "在线", "原始数据持续进入", "good")
        elif device in {"online", "waiting_raw"}:
            status_card("device", "等待数据", "MindWave 尚无原始数据", "warn")
        else:
            status_card("device", "离线", "未检测到 MindWave", "error")

        poor = getattr(state, "poor_signal", None)
        if poor is None:
            status_card("poor", "暂无数据", "不可评估", "neutral")
        elif poor < MAX_POOR_SIGNAL:
            status_card("poor", str(poor), "接触质量合格", "good")
        elif poor < 200:
            status_card("poor", str(poor), "需要调整佩戴", "warn")
        else:
            status_card("poor", str(poor), "当前无有效信号", "error")

        quality = str(getattr(state, "quality_level", "rejected") or "rejected")
        if not has_raw:
            status_card("quality", "不可评估", "等待设备数据", "neutral")
        elif quality == "trusted":
            status_card("quality", "可信", "可用于模型分析", "good")
        elif quality == "warning":
            status_card("quality", "警告", "数据质量需关注", "warn")
        else:
            status_card("quality", "不合格", "当前窗口已拒识", "error")

        model_status = str(getattr(state, "model_status", "") or "").upper()
        pipeline_state = str(getattr(state, "pipeline_state", "") or "").lower()
        if model_status == "READY":
            status_card("model", "正常", "模型已加载", "good")
        elif model_status == "FAILED" or pipeline_state == "error":
            status_card("model", "故障", "请查看系统诊断", "error")
        elif model_status == "LOADING":
            status_card("model", "正在准备", "等待模型预热", "warn")
        else:
            status_card("model", "暂无数据", "不可评估", "neutral")

        raw = getattr(state, "_eeg_raw_buffer", [])
        if raw:
            self._admin_eeg_plot.push_buffer(raw)
        else:
            self._admin_eeg_plot.reset()
        self._admin_eeg_plot.set_dimmed(quality == "rejected")
        measured_rate = getattr(state, "sample_rate_hz", None)
        self._admin_eeg_state.setText("实时监测" if raw else "等待设备数据")
        self._admin_eeg_footer.setText(
            f"目标采样率：{DEVICE_TARGET_SAMPLE_HZ} Hz · 实测："
            + (f"{float(measured_rate):.0f} Hz" if measured_rate is not None else "暂无数据")
        )

        attention = getattr(state, "attention", None)
        meditation = getattr(state, "meditation", None)
        self._admin_metric_values["attention"].setText(
            "暂无数据" if attention is None else f"{float(attention):.0f}"
        )
        self._admin_metric_values["meditation"].setText(
            "暂无数据" if meditation is None else f"{float(meditation):.0f}"
        )

        eligible = bool(getattr(state, "inference_eligible", False))
        probabilities = {
            "positive": getattr(state, "prob_positive", None),
            "neutral": getattr(state, "prob_neutral", None),
            "negative": getattr(state, "prob_negative", None),
        }
        for key, value in probabilities.items():
            self._admin_probability_values[key].setText(
                f"{float(value) * 100:.1f}%" if eligible and value is not None else "暂无数据"
            )
        if model_status == "FAILED" or pipeline_state == "error":
            summary = "当前模型与数据质量状态：模型运行故障"
        elif not raw:
            summary = "当前模型与数据质量状态：等待设备数据，暂不可评估"
        elif not eligible:
            summary = "当前模型与数据质量状态：信号不可解释，暂不输出概率"
        else:
            summary = "当前模型与数据质量状态：模型正常，当前数据可解释"
        self._admin_model_quality.setText(summary)

        if attention is not None and meditation is not None:
            self._admin_trend_plot.push_values(float(attention), float(meditation))
            self._admin_trend_stack.setCurrentWidget(self._admin_trend_plot)
        else:
            self._admin_trend_stack.setCurrentWidget(self._admin_trend_empty)

        mode = str(getattr(state, "mode", "live") or "live").lower()
        mode_text = {
            "live": "实时采集", "mock": "教学演示", "replay": "离线回放",
        }.get(mode, "暂无数据")
        active = bool(getattr(state, "session_active", getattr(state, "_session_active", False)))
        paused = bool(getattr(state, "session_paused", False))
        warmup = getattr(state, "warmup_progress", None)
        self._admin_runtime_values["record"].setText("记录中" if active else "未开始")
        self._admin_runtime_values["source"].setText(mode_text)
        self._admin_runtime_values["warmup"].setText(
            "等待设备数据" if not raw or warmup is None else f"{float(warmup) * 100:.0f}%"
        )
        self._admin_runtime_values["session"].setText(
            "已暂停" if active and paused else ("运行中" if active else "空闲")
        )

    def update_state(self, state):
        s = state
        requested_role = self._normalize_role(getattr(s, "current_role", self._role))
        if requested_role != self._role:
            self.set_role(requested_role)
        if self._role == "teacher":
            self._refresh_teacher_students()
            self._update_teacher_snapshot()
            return
        if self._role != "student":
            self._sync_admin_diagnostic_home(state)
        session_active = bool(getattr(s, "session_active", s._session_active))
        session_paused = bool(getattr(s, "session_paused", False))
        self._session_started = session_active
        self._session_paused = session_paused
        start_reason = (
            learning_start_block_reason(s, require_baseline=False)
            if self._role == "student" else ""
        )
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
        if self._role == "student":
            self._update_student_home(s, analysis_kind)
