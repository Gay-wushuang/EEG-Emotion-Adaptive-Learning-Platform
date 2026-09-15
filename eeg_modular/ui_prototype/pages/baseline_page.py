"""页面2：用户初始化与60～90秒基线采集页。"""

from __future__ import annotations

import time
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
    QPushButton, QLineEdit, QFrame, QProgressBar, QSpinBox, QSizePolicy, QComboBox,
)

from pages.base_page import BasePage
from widgets.card import Card
from widgets.status_indicator import StatusIndicator
from widgets.progress_ring import ProgressRing
from widgets.eeg_plot import EEGPlotWidget
from services.dashboard_state import (
    WARMUP_SECONDS, MAX_POOR_SIGNAL,
    MOCK_UI_REFRESH_HZ, DEVICE_TARGET_SAMPLE_HZ,
    BASELINE_COLLECTING, BASELINE_COMPLETED,
    BASELINE_EARLY_STOPPED, BASELINE_FAILED,
)
from services.identity_store import IdentityStore
from services.teaching_store import (
    BaselineResultStore, StudentRuntimeRegistry, TeacherSelectionContext,
    TeacherStudentStore,
)


class BaselinePage(BasePage):
    def __init__(self, state, service, *, identity_store=None, binding_store=None,
                 runtime_registry=None, selection_context=None, baseline_store=None):
        self.state = state
        self.service = service
        self.identity_store = identity_store or IdentityStore()
        self.binding_store = binding_store or TeacherStudentStore(identity_store=self.identity_store)
        self.runtime_registry = runtime_registry or StudentRuntimeRegistry.shared()
        self.selection_context = selection_context or TeacherSelectionContext.shared()
        self.baseline_store = baseline_store or BaselineResultStore()
        self._baseline_active = False
        self._baseline_done = False
        self._baseline_start = 0.0
        self._eeg_values = []
        self._att_values = []
        self._med_values = []
        self._poor_values = []
        self._baseline_start_raw_count = 0
        super().__init__(
            "用户初始化与基线采集",
            "采集60～90秒静息态基线数据，用于个人校准和后续状态对比。"
        )
        self._build_ui()
        self._teacher_timer = QTimer(self)
        self._teacher_timer.setInterval(500)
        self._teacher_timer.timeout.connect(self._refresh_teacher_baseline)
        self.set_role(self._role)
        # Round 4C：其他页面切换观察学生后，学生基线概览立即同步。
        self.selection_context.add_listener(self, "_on_selection_changed")

    def _on_selection_changed(self, student_id: str):
        if self._role == "teacher":
            self._refresh_teacher_baseline()

    def _build_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setSpacing(14)

        # ── 左侧：用户信息 + 采集控制 ──
        left = QVBoxLayout()
        left.setSpacing(14)

        # 教师端只读视图由独立 home 控件承载；旧卡片与详情标签保留仅用于兼容测试断言。
        self._teacher_baseline_card = Card("学生基线概览")
        self._teacher_baseline_card.setVisible(False)
        left.addWidget(self._teacher_baseline_card)
        self._teacher_baseline_detail = QLabel("请选择学生")
        self._teacher_baseline_detail.setWordWrap(True)
        self._teacher_baseline_detail.setVisible(False)

        # 用户信息卡片
        user_card = Card("用户信息")
        self._user_card = user_card
        form = QGridLayout()
        form.setSpacing(10)

        form.addWidget(QLabel("用户ID:"), 0, 0)
        user_id = getattr(self.state, "_user_id", "")
        if user_id == "demo_user":
            user_id = ""
        self._input_uid = QLineEdit(user_id)
        self._input_uid.setPlaceholderText("输入匿名用户ID，例如 S20260903001")
        self._input_uid.setReadOnly(True)
        form.addWidget(self._input_uid, 0, 1)

        form.addWidget(QLabel("显示名称:"), 0, 2)
        user_name = getattr(self.state, "_user_name", "")
        if user_name == "演示用户":
            user_name = ""
        self._input_name = QLineEdit(user_name)
        self._input_name.setPlaceholderText("可选；不建议填写真实姓名")
        form.addWidget(self._input_name, 0, 3)

        form.addWidget(QLabel("采集时长:"), 1, 0)
        self._combo_duration = QSpinBox()
        self._combo_duration.setRange(60, 90)
        self._combo_duration.setSingleStep(5)
        self._combo_duration.setSuffix(" 秒")
        self._combo_duration.setValue(int(getattr(self.state, "_baseline_target", 75)))
        self._combo_duration.valueChanged.connect(self._on_duration_change)
        form.addWidget(self._combo_duration, 1, 1)

        ownership = QLabel("学习任务与难度在监测页或“任务与事件”页设置；基线仅用于个人校准。")
        ownership.setWordWrap(True)
        ownership.setStyleSheet("color: #8491A5; font-size: 12px;")
        form.addWidget(ownership, 1, 2, 1, 2)

        user_card.add_widget(self._wrap_layout(form))
        left.addWidget(user_card)

        # 采集进度卡片
        collect_card = Card("基线采集")
        self._collect_card = collect_card

        progress_layout = QHBoxLayout()

        # 圆形进度环
        self._ring = ProgressRing()
        self._ring.setFixedSize(130, 130)
        progress_layout.addWidget(self._ring, 0, Qt.AlignCenter)

        # 进度信息
        info_layout = QVBoxLayout()
        info_layout.setSpacing(8)

        self._label_status = QLabel("就绪")
        self._label_status.setObjectName("AccentLabel")
        self._label_status.setStyleSheet("font-size: 16px;")
        info_layout.addWidget(self._label_status)

        self._label_time = QLabel(f"已用时间：0秒 / {int(self.state._baseline_target)}秒")
        self._label_time.setStyleSheet("color: #C5CDD9; font-size: 14px;")
        info_layout.addWidget(self._label_time)

        self._label_samples = QLabel("已采集样本：0")
        self._label_samples.setStyleSheet("color: #6B7689; font-size: 13px;")
        info_layout.addWidget(self._label_samples)

        self._label_quality = QLabel("信号合格率：--")
        self._label_quality.setStyleSheet("color: #6B7689; font-size: 13px;")
        info_layout.addWidget(self._label_quality)

        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("BaselineBar")
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        info_layout.addWidget(self._progress_bar)

        progress_layout.addLayout(info_layout, 1)
        collect_card.add_widget(self._wrap_layout(progress_layout))
        left.addWidget(collect_card)

        # 信号检查卡片
        signal_card = Card("实时信号检查")
        self._signal_card = signal_card
        sig_layout = QGridLayout()
        sig_layout.setSpacing(8)

        self._ind_poor = StatusIndicator("接触质量")
        sig_layout.addWidget(self._ind_poor, 0, 0)
        self._ind_att = StatusIndicator("专注度")
        sig_layout.addWidget(self._ind_att, 0, 1)
        self._ind_med = StatusIndicator("放松度")
        sig_layout.addWidget(self._ind_med, 1, 0)
        self._ind_conf = StatusIndicator("信号质量等级")
        sig_layout.addWidget(self._ind_conf, 1, 1)

        signal_card.add_widget(self._wrap_layout(sig_layout))
        left.addWidget(signal_card)

        # 操作按钮
        btn_layout = QHBoxLayout()
        self._btn_start = QPushButton("采集静息基线")
        self._btn_start.setObjectName("PrimaryButton")
        self._btn_start.clicked.connect(self._start_baseline)
        btn_layout.addWidget(self._btn_start)

        self._btn_stop = QPushButton("提前结束")
        self._btn_stop.setEnabled(False)
        self._btn_stop.setToolTip("提前结束不会生成有效基线，可重新采集")
        self._btn_stop.clicked.connect(self._stop_baseline)
        btn_layout.addWidget(self._btn_stop)

        self._btn_next = QPushButton("打开实时分析")
        self._btn_next.setObjectName("SuccessButton")
        self._btn_next.setEnabled(False)
        self._btn_next.setToolTip("完成一次基线采集后可打开实时分析")
        # 保留对象供旧版主窗口连接，但不再显示重复的页面跳转入口。
        self._btn_next.setVisible(False)

        self._baseline_actions = self._wrap_layout(btn_layout)
        left.addWidget(self._baseline_actions)
        main_layout.addLayout(left, 0)

        # ── 右侧：EEG实时曲线 + 采集统计 ──
        right = QVBoxLayout()
        right.setSpacing(14)

        eeg_card = Card("EEG实时信号")
        self._eeg_card = eeg_card
        self._eeg_plot = EEGPlotWidget()
        self._eeg_plot.setMinimumHeight(220)
        eeg_card.add_widget(self._eeg_plot)
        self._eeg_info = QLabel(f"目标采样率 {DEVICE_TARGET_SAMPLE_HZ} Hz · 等待设备数据")
        self._eeg_info.setStyleSheet("color: #6B7689; font-size: 12px;")
        eeg_card.add_widget(self._eeg_info)
        right.addWidget(eeg_card)

        # 基线统计预览
        stats_card = Card("基线统计预览")
        self._stats_card = stats_card
        stats_layout = QGridLayout()
        stats_layout.setSpacing(8)

        self._stat_att = self._make_stat("平均专注度", "--")
        stats_layout.addWidget(self._stat_att["card"], 0, 0)
        self._stat_med = self._make_stat("平均放松度", "--")
        stats_layout.addWidget(self._stat_med["card"], 0, 1)
        self._stat_qual = self._make_stat("信号合格率", "--")
        stats_layout.addWidget(self._stat_qual["card"], 1, 0)
        self._stat_samples = self._make_stat("总样本数", "--")
        stats_layout.addWidget(self._stat_samples["card"], 1, 1)

        stats_card.add_widget(self._wrap_layout(stats_layout))
        right.addWidget(stats_card)

        main_layout.addLayout(right, 1)
        self.content_layout.addLayout(main_layout)

        self._student_baseline_home = self._build_student_baseline_home()
        self._student_baseline_home.setVisible(False)
        self.content_layout.addWidget(self._student_baseline_home, 1)

        self._teacher_baseline_home = self._build_teacher_baseline_home()
        self._teacher_baseline_home.setVisible(False)
        self.content_layout.addWidget(self._teacher_baseline_home, 1)

        self._admin_diagnostic_home = self._build_admin_diagnostic_home()
        self._admin_diagnostic_home.setVisible(False)
        self.content_layout.addWidget(self._admin_diagnostic_home, 1)

    def _build_admin_diagnostic_home(self):
        """管理员专属的本机设备诊断台；所有控件只是现有采集状态的视图/代理。"""
        home = QWidget()
        home.setObjectName("AdminDiagnosticHome")
        root = QVBoxLayout(home)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)

        overview = QFrame()
        self._admin_status_strip = overview
        overview.setObjectName("AdminDiagnosticStatusStrip")
        overview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        overview_layout = QHBoxLayout(overview)
        overview_layout.setContentsMargins(20, 12, 20, 12)
        overview_layout.setSpacing(18)
        self._admin_overview = {}
        overview_specs = (
            ("device", "设备", "MindWave / ThinkGear"),
            ("connector", "连接状态", "未连接"),
            ("sampling", "采样率", "暂无数据"),
            ("analysis", "模型状态", "正在准备"),
        )
        for key, title, value_text in overview_specs:
            item = QWidget()
            layout = QHBoxLayout(item)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(9)
            title_label = QLabel(f"{title}：")
            title_label.setObjectName("AdminDiagnosticMuted")
            value = QLabel(value_text)
            value.setObjectName("AdminDiagnosticStripValue")
            hint = QLabel("")
            hint.hide()
            layout.addWidget(title_label)
            layout.addWidget(value, 1)
            overview_layout.addWidget(item, 1)
            self._admin_overview[key] = {"value": value, "hint": hint}
        root.addWidget(overview)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(14)

        left_widget = QWidget()
        self._admin_left_column = left_widget
        left = QVBoxLayout(left_widget)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(14)

        device_card = QFrame()
        self._admin_device_card = device_card
        device_card.setObjectName("AdminDiagnosticCard")
        device_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        device_layout = QVBoxLayout(device_card)
        device_layout.setContentsMargins(18, 14, 18, 14)
        device_layout.setSpacing(9)
        device_layout.addWidget(self._admin_title("设备信息"))
        device_grid = QGridLayout()
        device_grid.setHorizontalSpacing(22)
        device_grid.setVerticalSpacing(6)
        self._admin_device_values = {}
        for row, (key, caption, initial) in enumerate((
            ("model", "设备型号", "ThinkGear Connector"),
            ("protocol", "连接协议", "TCP"),
            ("port", "端口", "13854"),
            ("rate", "采样率", "暂无数据"),
            ("checked", "最后检测时间", "暂无数据"),
        )):
            name = QLabel(caption)
            name.setObjectName("AdminDiagnosticMuted")
            value = QLabel(initial)
            value.setObjectName("AdminDiagnosticFieldValue")
            device_grid.addWidget(name, row, 0)
            device_grid.addWidget(value, row, 1)
            self._admin_device_values[key] = value
        device_grid.setColumnStretch(1, 1)
        device_layout.addLayout(device_grid)
        left.addWidget(device_card)

        # 兼容既有同步接口；管理员诊断页不再展示学生式用户表单。
        self._admin_uid = QLineEdit(self._input_uid.text()); self._admin_uid.setReadOnly(True)
        self._admin_name = QLineEdit(self._input_name.text()); self._admin_name.setReadOnly(True)
        self._admin_duration = QSpinBox(); self._admin_duration.setRange(60, 90); self._admin_duration.setSingleStep(5)
        self._admin_duration.setSuffix(" 秒"); self._admin_duration.setValue(self._combo_duration.value())
        self._admin_duration.valueChanged.connect(self._combo_duration.setValue)

        sampling = QFrame()
        self._admin_baseline_card = sampling
        sampling.setObjectName("AdminDiagnosticCard")
        sampling.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sampling_layout = QVBoxLayout(sampling)
        sampling_layout.setContentsMargins(18, 14, 18, 14)
        sampling_layout.setSpacing(8)
        title_row = QHBoxLayout()
        title_row.addWidget(self._admin_title("基线采集"))
        title_row.addStretch()
        self._admin_sample_status = QLabel("未开始")
        self._admin_sample_status.setObjectName("AdminDiagnosticBadge")
        title_row.addWidget(self._admin_sample_status)
        sampling_layout.addLayout(title_row)
        sample_body = QHBoxLayout()
        sample_body.setSpacing(18)
        self._admin_ring = ProgressRing()
        self._admin_ring.setFixedSize(112, 112)
        sample_body.addWidget(self._admin_ring, 0, Qt.AlignCenter)
        sample_info = QVBoxLayout()
        sample_info.setSpacing(8)
        self._admin_sample_time = QLabel(f"采集时间：0 / {self._combo_duration.value()} 秒")
        self._admin_sample_count = QLabel("有效样本：暂无数据")
        self._admin_sample_quality = QLabel("信号质量：暂无数据")
        for label in (self._admin_sample_time, self._admin_sample_count,
                      self._admin_sample_quality):
            label.setObjectName("AdminDiagnosticMetric")
        sample_info.addWidget(self._admin_sample_time)
        sample_info.addWidget(self._admin_sample_count)
        sample_info.addWidget(self._admin_sample_quality)
        sample_info.addStretch()
        sample_body.addLayout(sample_info, 1)
        sampling_layout.addLayout(sample_body, 1)
        self._admin_progress = QProgressBar()
        self._admin_progress.setRange(0, 100)
        self._admin_progress.setValue(0)
        sampling_layout.addWidget(self._admin_progress)
        self._admin_hint = QLabel("设备就绪后可开始采集基线。")
        self._admin_hint.setObjectName("AdminDiagnosticMuted")
        self._admin_hint.setWordWrap(True)
        sampling_layout.addWidget(self._admin_hint)
        actions = QHBoxLayout()
        actions.setSpacing(10)
        self._admin_start = QPushButton("开始采集基线")
        self._admin_start.setObjectName("PrimaryButton")
        self._admin_start.setFixedHeight(40)
        self._admin_start.clicked.connect(self._btn_start.click)
        self._admin_stop = QPushButton("停止采集")
        self._admin_stop.setFixedHeight(40)
        self._admin_stop.clicked.connect(self._btn_stop.click)
        actions.addWidget(self._admin_start, 2)
        actions.addWidget(self._admin_stop, 1)
        sampling_layout.addLayout(actions)
        left.addWidget(sampling, 1)

        signal = QFrame()
        self._admin_signal_card = signal
        signal.setObjectName("AdminDiagnosticCard")
        signal.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        signal_layout = QVBoxLayout(signal)
        signal_layout.setContentsMargins(18, 14, 18, 14)
        signal_layout.setSpacing(8)
        signal_layout.addWidget(self._admin_title("实时信号质量"))
        signal_grid = QGridLayout()
        signal_grid.setSpacing(8)
        self._admin_signals = {}
        for index, (key, title) in enumerate((("poor", "接触质量"), ("attention", "专注度"),
                                              ("meditation", "放松度"), ("quality", "信号等级"))):
            item = StatusIndicator(title); item.set_state(StatusIndicator.LEVEL_NEUTRAL, "暂无数据")
            signal_grid.addWidget(item, index // 2, index % 2); self._admin_signals[key] = item
        signal_layout.addLayout(signal_grid)
        left.addWidget(signal)
        left.setStretch(0, 0)
        left.setStretch(1, 1)
        left.setStretch(2, 0)

        right_widget = QWidget()
        self._admin_right_column = right_widget
        right = QVBoxLayout(right_widget)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(14)
        eeg = QFrame()
        self._admin_eeg_card = eeg
        eeg.setObjectName("AdminDiagnosticCard")
        eeg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        eeg_layout = QVBoxLayout(eeg)
        eeg_layout.setContentsMargins(18, 14, 18, 14)
        eeg_layout.setSpacing(7)
        eeg_title_row = QHBoxLayout()
        eeg_title_row.addWidget(self._admin_title("EEG 实时波形"))
        eeg_title_row.addStretch()
        self._admin_eeg_status = QLabel("实时监测 · 等待数据")
        self._admin_eeg_status.setObjectName("AdminDiagnosticLiveBadge")
        eeg_title_row.addWidget(self._admin_eeg_status)
        eeg_layout.addLayout(eeg_title_row)
        self._admin_eeg_plot = EEGPlotWidget()
        self._admin_eeg_plot.setMinimumHeight(250)
        eeg_layout.addWidget(self._admin_eeg_plot, 1)
        self._admin_eeg_info = QLabel(f"目标采样率 {DEVICE_TARGET_SAMPLE_HZ} Hz")
        self._admin_eeg_info.setObjectName("AdminDiagnosticMuted")
        eeg_layout.addWidget(self._admin_eeg_info)
        right.addWidget(eeg, 5)

        stats = QFrame()
        self._admin_stats_card = stats
        stats.setObjectName("AdminDiagnosticCard")
        stats.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        stats_layout = QVBoxLayout(stats)
        stats_layout.setContentsMargins(18, 14, 18, 14)
        stats_layout.setSpacing(9)
        stats_layout.addWidget(self._admin_title("诊断结果"))
        stats_grid = QGridLayout()
        stats_grid.setSpacing(10)
        self._admin_stats = {}
        for index, (key, title) in enumerate((("attention", "平均专注度"), ("meditation", "平均放松度"),
                                              ("quality", "信号合格率"), ("samples", "总样本数"))):
            box = QFrame(); box.setObjectName("AdminDiagnosticStat")
            box_layout = QVBoxLayout(box); box_layout.setContentsMargins(14, 9, 14, 9); box_layout.setSpacing(3)
            caption = QLabel(title); caption.setObjectName("AdminDiagnosticMuted")
            value = QLabel("暂无数据"); value.setObjectName("AdminDiagnosticValue")
            box_layout.addWidget(caption); box_layout.addWidget(value); stats_grid.addWidget(box, index // 2, index % 2)
            self._admin_stats[key] = value
        stats_layout.addLayout(stats_grid)
        right.addWidget(stats, 0)

        body_layout.addWidget(left_widget, 40)
        body_layout.addWidget(right_widget, 60)
        root.addWidget(body, 1)

        home.setStyleSheet("""
            QWidget#AdminDiagnosticHome { background: transparent; }
            QFrame#AdminDiagnosticStatusStrip, QFrame#AdminDiagnosticCard {
                background: #162235; border: 1px solid #2B3A50; border-radius: 12px;
            }
            QFrame#AdminDiagnosticStat { background: #1C293D; border: 1px solid #2B3A50; border-radius: 9px; }
            QLabel#AdminDiagnosticTitle { color: #E8EDF3; font-size: 15px; font-weight: 700; }
            QLabel#AdminDiagnosticMuted { color: #8EA3BF; font-size: 12px; }
            QLabel#AdminDiagnosticStripValue, QLabel#AdminDiagnosticFieldValue,
            QLabel#AdminDiagnosticMetric { color: #E8EDF3; font-size: 13px; font-weight: 600; }
            QLabel#AdminDiagnosticBadge { color: #FBBF24; background: #3A301D; border: 1px solid #80661F; border-radius: 6px; padding: 3px 9px; font-weight: 700; }
            QLabel#AdminDiagnosticLiveBadge { color: #4ADE80; background: #173528; border: 1px solid #286246; border-radius: 6px; padding: 3px 9px; font-weight: 700; }
            QLabel#AdminDiagnosticValue { color: #E8EDF3; font-size: 18px; font-weight: 700; }
        """)
        return home

    @staticmethod
    def _admin_title(text):
        label = QLabel(text); label.setObjectName("AdminDiagnosticTitle"); return label

    def _build_student_baseline_home(self):
        """学生专用基线视图；控件只代理既有采集流程。"""
        home = QWidget()
        home.setObjectName("StudentBaselineHome")
        root = QHBoxLayout(home)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)

        left = QVBoxLayout()
        left.setSpacing(14)
        settings = QFrame()
        self._student_user_card = settings
        settings.setObjectName("StudentBaselineCard")
        settings.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(20, 13, 20, 13)
        settings_layout.setSpacing(7)
        settings_layout.addWidget(self._student_title("用户与采集设置"))
        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)
        self._student_uid = QLineEdit(self._input_uid.text())
        self._student_uid.setReadOnly(True)
        self._student_name = QLineEdit(self._input_name.text())
        self._student_name.textChanged.connect(self._input_name.setText)
        self._student_duration = QSpinBox()
        self._student_duration.setRange(60, 90)
        self._student_duration.setSingleStep(5)
        self._student_duration.setSuffix(" 秒")
        self._student_duration.setValue(self._combo_duration.value())
        self._student_duration.valueChanged.connect(self._combo_duration.setValue)
        form.addWidget(QLabel("用户 ID"), 0, 0)
        form.addWidget(self._student_uid, 0, 1)
        form.addWidget(QLabel("显示名称"), 0, 2)
        form.addWidget(self._student_name, 0, 3)
        form.addWidget(QLabel("采集时长"), 1, 0)
        form.addWidget(self._student_duration, 1, 1)
        note = QLabel("建议保持安静坐姿并减少动作；基线仅用于个人状态对比。")
        note.setObjectName("StudentBaselineMuted")
        note.setWordWrap(True)
        form.addWidget(note, 1, 2, 1, 2)
        settings_layout.addLayout(form)
        left.addWidget(settings)

        collect = QFrame()
        self._student_collect_card = collect
        collect.setObjectName("StudentBaselineCard")
        collect.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        collect_layout = QVBoxLayout(collect)
        collect_layout.setContentsMargins(20, 13, 20, 13)
        collect_layout.setSpacing(5)
        collect_layout.addWidget(self._student_title("基线采集"))
        state_row = QHBoxLayout()
        state_row.setContentsMargins(0, 0, 0, 0)
        state_column = QVBoxLayout()
        state_column.setSpacing(4)
        self._student_status_caption = QLabel("当前状态")
        self._student_status_caption.setObjectName("StudentBaselineMuted")
        state_column.addWidget(self._student_status_caption)
        self._student_status = QLabel("就绪")
        self._student_status.setObjectName("StudentBaselineState")
        state_column.addWidget(self._student_status)
        self._student_collect_badge = QLabel("等待信号")
        self._student_collect_badge.setObjectName("StudentBaselineBadge")
        self._student_collect_badge.setAlignment(Qt.AlignCenter)
        self._student_collect_badge.setMinimumWidth(86)
        self._student_collect_badge.setFixedHeight(30)
        state_column.addWidget(self._student_collect_badge, 0, Qt.AlignLeft)
        state_row.addLayout(state_column)
        state_row.addStretch()
        collect_layout.addLayout(state_row)
        body = QHBoxLayout()
        body.setSpacing(12)
        self._student_ring = ProgressRing()
        self._student_ring.setFixedSize(104, 104)
        body.addWidget(self._student_ring, 0, Qt.AlignCenter)
        metrics = QVBoxLayout()
        metrics.setSpacing(9)
        self._student_time = QLabel("已用时间：0秒")
        self._student_samples = QLabel("已采样本数：暂无数据")
        self._student_quality = QLabel("信号合格率：暂无数据")
        for label in (self._student_time, self._student_samples, self._student_quality):
            label.setObjectName("StudentBaselineMetric")
            metrics.addWidget(label)
        self._student_progress = QProgressBar()
        self._student_progress.setRange(0, 100)
        self._student_progress.setValue(0)
        body.addLayout(metrics, 1)
        collect_layout.addLayout(body)
        collect_layout.addWidget(self._student_progress)
        action_divider = QFrame()
        action_divider.setObjectName("StudentBaselineDivider")
        action_divider.setFixedHeight(1)
        collect_layout.addWidget(action_divider)
        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        self._student_start = QPushButton("开始采集")
        self._student_start.setObjectName("PrimaryButton")
        self._student_start.setMinimumHeight(42)
        self._student_start.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._student_start.clicked.connect(self._student_start_clicked)
        self._student_stop = QPushButton("提前结束")
        self._student_stop.setMinimumHeight(42)
        self._student_stop.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._student_stop.clicked.connect(self._student_stop_clicked)
        buttons.addWidget(self._student_start)
        buttons.addWidget(self._student_stop)
        self._student_button_row = buttons
        collect_layout.addLayout(buttons)
        collect_layout.setStretch(2, 1)
        left.addWidget(collect)

        signal = QFrame()
        self._student_signal_card = signal
        signal.setObjectName("StudentBaselineCard")
        signal.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        signal_layout = QVBoxLayout(signal)
        signal_layout.setContentsMargins(20, 13, 20, 13)
        signal_layout.setSpacing(7)
        signal_layout.addWidget(self._student_title("实时信号检查"))
        self._student_signal_indicators = {}
        for key, title in (("poor", "接触质量"), ("attention", "专注度"),
                           ("meditation", "放松度"), ("quality", "信号质量")):
            indicator = StatusIndicator(title)
            indicator.set_state(StatusIndicator.LEVEL_NEUTRAL, "暂无可用数据")
            signal_layout.addWidget(indicator)
            self._student_signal_indicators[key] = indicator
        signal_layout.addStretch()
        left.addWidget(signal)
        left.setStretch(0, 0)
        left.setStretch(1, 1)
        left.setStretch(2, 0)
        root.addLayout(left, 38)

        right = QVBoxLayout()
        right.setSpacing(14)
        eeg = QFrame()
        eeg.setObjectName("StudentBaselineCard")
        eeg.setMinimumHeight(520)
        eeg_layout = QVBoxLayout(eeg)
        eeg_layout.setContentsMargins(20, 16, 20, 16)
        eeg_layout.setSpacing(8)
        eeg_layout.addWidget(self._student_title("EEG 实时信号"))
        eeg_subtitle = QLabel("最近 5 秒")
        eeg_subtitle.setObjectName("StudentBaselineMuted")
        eeg_layout.addWidget(eeg_subtitle)
        self._student_eeg_plot = EEGPlotWidget()
        self._student_eeg_plot.setMinimumHeight(330)
        eeg_layout.addWidget(self._student_eeg_plot, 1)
        self._student_eeg_info = QLabel("等待设备数据")
        self._student_eeg_info.setObjectName("StudentBaselineMuted")
        self._student_eeg_info.setAlignment(Qt.AlignCenter)
        eeg_layout.addWidget(self._student_eeg_info)
        right.addWidget(eeg)

        stats = QFrame()
        stats.setObjectName("StudentBaselineCard")
        stats.setMinimumHeight(288)
        stats_layout = QVBoxLayout(stats)
        stats_layout.setContentsMargins(20, 16, 20, 16)
        stats_layout.setSpacing(9)
        stats_layout.addWidget(self._student_title("基线统计预览"))
        self._student_stats_hint = QLabel("暂无基线统计\n完成采集后将在这里显示个人基线摘要。")
        self._student_stats_hint.setObjectName("StudentBaselineMuted")
        self._student_stats_hint.setWordWrap(True)
        stats_layout.addWidget(self._student_stats_hint)
        stats_grid = QGridLayout()
        stats_grid.setSpacing(10)
        self._student_stats = {}
        for index, (key, title) in enumerate((
            ("attention", "平均专注度"), ("meditation", "平均放松度"),
            ("quality", "信号合格率"), ("samples", "总样本数"),
        )):
            card = QFrame()
            card.setObjectName("StudentBaselineStat")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(16, 10, 16, 10)
            caption = QLabel(title)
            caption.setObjectName("StudentBaselineMuted")
            value = QLabel("暂无数据")
            value.setObjectName("StudentBaselineStatValue")
            card_layout.addWidget(caption)
            card_layout.addWidget(value)
            stats_grid.addWidget(card, index // 2, index % 2)
            self._student_stats[key] = value
        stats_layout.addLayout(stats_grid)
        right.addWidget(stats)
        right.addStretch()
        root.addLayout(right, 62)

        home.setStyleSheet("""
            QWidget#StudentBaselineHome { background: transparent; }
            QFrame#StudentBaselineCard { background: #192130; border: 1px solid #293345; border-radius: 13px; }
            QFrame#StudentBaselineStat { background: #151E2C; border: 1px solid #293345; border-radius: 9px; }
            QFrame#StudentBaselineDivider { background: #26364B; border: none; }
            QLabel#StudentBaselineTitle { color: #E8EDF3; font-size: 17px; font-weight: 700; }
            QLabel#StudentBaselineMuted { color: #8491A5; font-size: 12px; }
            QLabel#StudentBaselineState { color: #E8EDF3; font-size: 26px; font-weight: 700; }
            QLabel#StudentBaselineBadge { color: #FBBF24; background: #3A301D; border: 1px solid #6B5526; border-radius: 5px; padding: 4px 8px; font-size: 12px; }
            QLabel#StudentBaselineBadgeGood { color: #4ADE80; background: #173528; border: 1px solid #286246; border-radius: 5px; padding: 4px 8px; font-size: 12px; }
            QLabel#StudentBaselineBadgeError { color: #F87171; background: #3B2027; border: 1px solid #7F3F47; border-radius: 5px; padding: 4px 8px; font-size: 12px; }
            QLabel#StudentBaselineMetric { color: #C5CDD9; font-size: 13px; }
            QLabel#StudentBaselineStatValue { color: #E8EDF3; font-size: 18px; font-weight: 700; }
        """)
        return home

    def _build_teacher_baseline_home(self):
        """教师端学生基线状态只读视图，严格参照 1920×1080 SVG 母版。"""
        home = QWidget()
        home.setObjectName("TeacherBaselineHome")
        root = QVBoxLayout(home)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)

        # ── 学生上下文卡（全宽）──
        ctx = QFrame()
        ctx.setObjectName("TeacherCard")
        ctx_layout = QHBoxLayout(ctx)
        ctx_layout.setContentsMargins(20, 14, 20, 14)
        ctx_layout.setSpacing(18)

        seg_student = QVBoxLayout()
        seg_student.setSpacing(6)
        seg_student.addWidget(self._teacher_caption("当前观察学生"))
        self._teacher_student_combo = QComboBox()
        self._teacher_student_combo.setObjectName("TeacherCombo")
        self._teacher_student_combo.setMinimumWidth(300)
        self._teacher_student_combo.currentIndexChanged.connect(
            self._teacher_student_changed
        )
        seg_student.addWidget(self._teacher_student_combo)
        ctx_layout.addLayout(seg_student, 0)

        seg_status = QVBoxLayout()
        seg_status.setSpacing(6)
        seg_status.addWidget(self._teacher_caption("基线状态"))
        status_row = QHBoxLayout()
        status_row.setSpacing(10)
        self._teacher_status_label = QLabel("未采集")
        self._teacher_status_label.setObjectName("TeacherStatusText")
        status_row.addWidget(self._teacher_status_label)
        self._teacher_status_badge = QLabel("不可用")
        self._teacher_status_badge.setObjectName("TeacherBadge")
        self._teacher_status_badge.setAlignment(Qt.AlignCenter)
        self._teacher_status_badge.setMinimumWidth(72)
        status_row.addWidget(self._teacher_status_badge)
        status_row.addStretch()
        seg_status.addLayout(status_row)
        ctx_layout.addLayout(seg_status, 0)

        seg_desc = QVBoxLayout()
        seg_desc.setSpacing(6)
        seg_desc.addWidget(self._teacher_caption("说明"))
        self._teacher_context_desc = QLabel("教师仅查看，不提供基线采集操作。")
        self._teacher_context_desc.setObjectName("TeacherText")
        self._teacher_context_desc.setWordWrap(True)
        seg_desc.addWidget(self._teacher_context_desc)
        seg_desc.addStretch()
        ctx_layout.addLayout(seg_desc, 1)

        root.addWidget(ctx)

        # ── 中间行：基线概览 + 教师查看说明 ──
        middle = QHBoxLayout()
        middle.setSpacing(14)

        # 左侧：基线概览
        overview = QFrame()
        overview.setObjectName("TeacherCard")
        ov_layout = QVBoxLayout(overview)
        ov_layout.setContentsMargins(20, 16, 20, 16)
        ov_layout.setSpacing(10)
        ov_layout.addWidget(self._teacher_card_title("基线概览"))
        ov_sub = QLabel("最近一次个人静息基线记录")
        ov_sub.setObjectName("TeacherMuted")
        ov_layout.addWidget(ov_sub)

        metrics = QFrame()
        metrics.setObjectName("TeacherMutedBg")
        m_layout = QGridLayout(metrics)
        m_layout.setContentsMargins(16, 12, 16, 12)
        m_layout.setHorizontalSpacing(12)
        m_layout.setVerticalSpacing(6)
        self._teacher_metrics = {}
        for col, key in enumerate(("completed_at", "duration", "samples", "quality")):
            cap = {"completed_at": "完成时间", "duration": "采集时长",
                   "samples": "总样本数", "quality": "信号合格率"}[key]
            caption = QLabel(cap)
            caption.setObjectName("TeacherMuted")
            value = QLabel("--")
            value.setObjectName("TeacherMetricValue")
            m_layout.addWidget(caption, 0, col)
            m_layout.addWidget(value, 1, col)
            self._teacher_metrics[key] = value
        ov_layout.addWidget(metrics)

        ov_layout.addWidget(self._teacher_card_title("核心统计"))
        stats_row = QHBoxLayout()
        stats_row.setSpacing(12)
        self._teacher_stats = {}
        for key, title in (("attention", "平均专注度"),
                           ("meditation", "平均放松度"),
                           ("usability", "基线可用性")):
            card = QFrame()
            card.setObjectName("TeacherMutedBg")
            c_layout = QVBoxLayout(card)
            c_layout.setContentsMargins(16, 12, 16, 12)
            c_layout.setSpacing(6)
            cap = QLabel(title)
            cap.setObjectName("TeacherMuted")
            val = QLabel("--")
            val.setObjectName("TeacherStatValue")
            c_layout.addWidget(cap)
            c_layout.addWidget(val)
            c_layout.addStretch()
            stats_row.addWidget(card, 1)
            self._teacher_stats[key] = val
        ov_layout.addLayout(stats_row, 1)

        middle.addWidget(overview, 60)

        # 右侧：教师查看说明
        explain = QFrame()
        explain.setObjectName("TeacherCard")
        ex_layout = QVBoxLayout(explain)
        ex_layout.setContentsMargins(20, 16, 20, 16)
        ex_layout.setSpacing(12)
        ex_layout.addWidget(self._teacher_card_title("教师查看说明"))
        line1 = QLabel("该基线由学生本人完成，用于后续学习状态对比。")
        line1.setObjectName("TeacherText")
        line1.setWordWrap(True)
        ex_layout.addWidget(line1)
        line2 = QLabel("教师端只读查看，不修改学生基线结果。")
        line2.setObjectName("TeacherText")
        line2.setWordWrap(True)
        ex_layout.addWidget(line2)

        div = QFrame()
        div.setObjectName("TeacherDivider")
        div.setFixedHeight(1)
        ex_layout.addWidget(div)

        ex_layout.addWidget(self._teacher_caption("基线曲线"))
        self._teacher_curve_text = QLabel("暂无可展示的历史采样曲线")
        self._teacher_curve_text.setObjectName("TeacherText")
        ex_layout.addWidget(self._teacher_curve_text)
        curve_note = QLabel("当前持久化结果仅保存统计摘要，不保存采样序列。")
        curve_note.setObjectName("TeacherMuted")
        curve_note.setWordWrap(True)
        ex_layout.addWidget(curve_note)

        note_box = QFrame()
        note_box.setObjectName("TeacherNoteBox")
        nb_layout = QHBoxLayout(note_box)
        nb_layout.setContentsMargins(14, 12, 14, 12)
        nb_layout.setSpacing(10)
        dot = QLabel("●")
        dot.setStyleSheet("color: #42b7ff; font-size: 14px;")
        nb_layout.addWidget(dot, 0, Qt.AlignTop)
        note_text = QLabel("后续学习报告可将本次表现与个人基线进行辅助对比。")
        note_text.setObjectName("TeacherText")
        note_text.setWordWrap(True)
        nb_layout.addWidget(note_text, 1)
        ex_layout.addWidget(note_box)
        ex_layout.addStretch()

        middle.addWidget(explain, 40)
        root.addLayout(middle, 1)

        # ── 底部：教学参考 ──
        guide = QFrame()
        guide.setObjectName("TeacherCard")
        g_layout = QVBoxLayout(guide)
        g_layout.setContentsMargins(20, 16, 20, 16)
        g_layout.setSpacing(10)
        g_layout.addWidget(self._teacher_card_title("教学参考"))
        g_sub = QLabel("基线状态主要用于判断学生是否已具备后续个体化分析条件。")
        g_sub.setObjectName("TeacherMuted")
        g_sub.setWordWrap(True)
        g_layout.addWidget(g_sub)

        guide_row = QHBoxLayout()
        guide_row.setSpacing(12)
        guide_items = [
            ("#3ddc97", "基线已完成",
             "后续状态可与该学生个人静息基线进行比较。",
             "不用于医学诊断，仅用于学习状态辅助参考。"),
            ("#42b7ff", "实时观察",
             "在“学生实时观察”页查看当前学习状态与趋势。",
             "基线仅作为对比背景，不直接替代实时状态判断。"),
            ("#f5b942", "异常情况",
             "若基线未完成或不可用，应提示学生重新完成个人基线。",
             "教师端仍保持只读，不出现“开始采集”等学生操作。"),
        ]
        for color, title, body, footnote in guide_items:
            card = QFrame()
            card.setObjectName("TeacherMutedBg")
            c_layout = QVBoxLayout(card)
            c_layout.setContentsMargins(16, 14, 16, 14)
            c_layout.setSpacing(8)
            title_row = QHBoxLayout()
            title_row.setSpacing(8)
            d = QLabel("●")
            d.setStyleSheet(f"color: {color}; font-size: 14px;")
            title_row.addWidget(d)
            t = QLabel(title)
            t.setObjectName("TeacherGuideTitle")
            title_row.addWidget(t)
            title_row.addStretch()
            c_layout.addLayout(title_row)
            b = QLabel(body)
            b.setObjectName("TeacherText")
            b.setWordWrap(True)
            c_layout.addWidget(b)
            f = QLabel(footnote)
            f.setObjectName("TeacherMuted")
            f.setWordWrap(True)
            c_layout.addWidget(f)
            guide_row.addWidget(card, 1)
        g_layout.addLayout(guide_row, 1)

        root.addWidget(guide)

        home.setStyleSheet("""
            QWidget#TeacherBaselineHome { background: transparent; }
            QFrame#TeacherCard { background: #182334; border: 1px solid #2b3950; border-radius: 12px; }
            QFrame#TeacherMutedBg { background: #131d2b; border-radius: 10px; }
            QFrame#TeacherDivider { background: #2b3950; border: none; }
            QFrame#TeacherNoteBox { background: #121b29; border: 1px solid #31425c; border-radius: 10px; }
            QLabel#TeacherCardTitle { color: #f4f7fb; font-size: 18px; font-weight: 700; }
            QLabel#TeacherCaption { color: #8da3bd; font-size: 13px; }
            QLabel#TeacherText { color: #dbe7f5; font-size: 14px; }
            QLabel#TeacherMuted { color: #8da3bd; font-size: 13px; }
            QLabel#TeacherStatusText { color: #3ddc97; font-size: 19px; font-weight: 700; }
            QLabel#TeacherBadge { color: #3ddc97; background: #123a2d; border: 1px solid #2a765d; border-radius: 6px; padding: 4px 10px; font-size: 13px; }
            QLabel#TeacherMetricValue { color: #f4f7fb; font-size: 18px; font-weight: 700; }
            QLabel#TeacherStatValue { color: #f4f7fb; font-size: 26px; font-weight: 700; }
            QLabel#TeacherGuideTitle { color: #f4f7fb; font-size: 15px; font-weight: 700; }
            QComboBox#TeacherCombo {
                background: #121b29; border: 1px solid #31425c; border-radius: 7px;
                color: #dbe7f5; font-size: 14px; padding: 6px 10px; min-height: 22px;
            }
            QComboBox#TeacherCombo:hover { border-color: #42b7ff; }
            QComboBox#TeacherCombo::drop-down { border: none; width: 20px; }
            QComboBox#TeacherCombo QAbstractItemView {
                background: #121b29; color: #dbe7f5; selection-background-color: #1e2d45;
                border: 1px solid #31425c;
            }
        """)
        return home

    @staticmethod
    def _teacher_card_title(text):
        label = QLabel(text)
        label.setObjectName("TeacherCardTitle")
        return label

    @staticmethod
    def _teacher_caption(text):
        label = QLabel(text)
        label.setObjectName("TeacherCaption")
        return label

    def _set_teacher_status(self, status_text: str, badge_text: str, *,
                            color: str = "#3ddc97", badge_color: str = "#3ddc97",
                            badge_bg: str = "#123a2d", badge_border: str = "#2a765d"):
        """统一刷新教师端状态文字与可用性徽章。"""
        self._teacher_status_label.setText(status_text)
        self._teacher_status_label.setStyleSheet(f"color: {color}; font-size: 19px; font-weight: 700;")
        self._teacher_status_badge.setText(badge_text)
        self._teacher_status_badge.setStyleSheet(
            f"color: {badge_color}; background: {badge_bg}; border: 1px solid {badge_border}; "
            f"border-radius: 6px; padding: 4px 10px; font-size: 13px;"
        )

    def _apply_teacher_baseline_result(self, result: dict):
        """将已完成基线结果填入教师端概览与核心统计。"""
        completed_at = result.get("completed_at") or result.get("baseline_completed_at") or "--"
        duration = float(result.get("duration_seconds", result.get("baseline_elapsed")) or 0)
        samples = result.get("sample_count", result.get("baseline_samples")) or 0
        quality_rate = result.get("quality_rate", result.get("baseline_quality_rate"))
        quality_text = "--" if quality_rate is None else f"{float(quality_rate) * 100:.0f}%"
        avg_att = result.get("avg_attention", result.get("baseline_avg_attention"))
        avg_med = result.get("avg_meditation", result.get("baseline_avg_meditation"))

        self._teacher_metrics["completed_at"].setText(str(completed_at))
        self._teacher_metrics["duration"].setText(f"{duration:.0f} 秒")
        self._teacher_metrics["samples"].setText(str(samples))
        qual = self._teacher_metrics["quality"]
        qual.setText(quality_text)
        if quality_rate is not None and float(quality_rate) >= 0.8:
            qual.setStyleSheet("color: #3ddc97; font-size: 18px; font-weight: 700;")
        else:
            qual.setStyleSheet("color: #f4f7fb; font-size: 18px; font-weight: 700;")

        att = self._teacher_stats["attention"]
        att.setText("--" if avg_att is None else f"{float(avg_att):.1f}")
        att.setStyleSheet("color: #42b7ff; font-size: 26px; font-weight: 700;")
        med = self._teacher_stats["meditation"]
        med.setText("--" if avg_med is None else f"{float(avg_med):.1f}")
        med.setStyleSheet("color: #3ddc97; font-size: 26px; font-weight: 700;")
        usable = self._teacher_stats["usability"]
        usable.setText("可用于后续对比")
        usable.setStyleSheet("color: #f4f7fb; font-size: 22px; font-weight: 700;")

    def _reset_teacher_baseline_view(self):
        """无基线时清空教师端统计为明确空状态。"""
        for key in self._teacher_metrics:
            self._teacher_metrics[key].setText("--")
            self._teacher_metrics[key].setStyleSheet("color: #f4f7fb; font-size: 18px; font-weight: 700;")
        for key in self._teacher_stats:
            self._teacher_stats[key].setText("--")
            self._teacher_stats[key].setStyleSheet("color: #f4f7fb; font-size: 26px; font-weight: 700;")
        self._teacher_stats["usability"].setText("暂不可用")
        self._teacher_curve_text.setText("暂无可展示的历史采样曲线")

    @staticmethod
    def _student_title(text):
        label = QLabel(text)
        label.setObjectName("StudentBaselineTitle")
        return label

    def _student_start_clicked(self):
        self._btn_start.click()
        self._sync_student_baseline_view()

    def _student_stop_clicked(self):
        self._btn_stop.click()
        self._sync_student_baseline_view()

    def _make_stat(self, title: str, value: str) -> dict:
        card = Card(title)
        val_label = QLabel(value)
        val_label.setObjectName("CardValueSmall")
        card.add_widget(val_label)
        return {"card": card, "label": val_label}

    def _wrap_layout(self, layout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        return w

    def _sync_student_baseline_view(self):
        """将既有基线控件状态镜像到学生视图，不参与状态变更。"""
        if not hasattr(self, "_student_baseline_home"):
            return
        status_text = self._label_status.text().replace("基线状态：", "")
        self._student_status.setText(status_text or "未采集")
        status = getattr(self.state, "baseline_status", "IDLE")
        color = "#F87171" if status == BASELINE_FAILED else (
            "#4ADE80" if status == BASELINE_COMPLETED else "#FBBF24"
        )
        self._student_status.setStyleSheet(f"color: {color};")
        poor = getattr(self.state, "poor_signal", None)
        device_ready = (
            getattr(self.state, "connector_status", "offline") == "online"
            and getattr(self.state, "device_status", "offline") == "online"
        )
        if status == BASELINE_FAILED:
            badge_text, badge_name = "采集失败", "StudentBaselineBadgeError"
        elif status == BASELINE_COMPLETED:
            badge_text, badge_name = "已完成", "StudentBaselineBadgeGood"
        elif poor is not None and poor < MAX_POOR_SIGNAL:
            badge_text, badge_name = "信号可用", "StudentBaselineBadgeGood"
        elif device_ready:
            badge_text, badge_name = "检查信号", "StudentBaselineBadge"
        else:
            badge_text, badge_name = "等待设备", "StudentBaselineBadge"
        self._student_collect_badge.setText(badge_text)
        self._student_collect_badge.setObjectName(badge_name)
        self._student_collect_badge.style().unpolish(self._student_collect_badge)
        self._student_collect_badge.style().polish(self._student_collect_badge)
        self._student_ring.set_progress(self._ring._progress)
        self._student_ring.set_text(self._ring._text)
        self._student_ring.set_subtext(self._ring._subtext)
        self._student_progress.setValue(self._progress_bar.value())
        self._student_time.setText(self._label_time.text())
        self._student_samples.setText(
            self._label_samples.text().replace(": 0", "：暂无数据").replace("：0", "：暂无数据")
        )
        self._student_quality.setText(
            self._label_quality.text().replace("--", "暂无数据")
        )
        self._student_start.setEnabled(self._btn_start.isEnabled())
        self._student_stop.setEnabled(self._btn_stop.isEnabled())
        self._student_start.setToolTip(self._btn_start.toolTip())
        if self._baseline_active:
            qualified = sum(1 for value in self._poor_values if value < MAX_POOR_SIGNAL)
            quality = qualified / len(self._poor_values) if self._poor_values else None
            raw_samples = max(
                0, int(getattr(self.state, "_raw_sample_count", 0))
                - self._baseline_start_raw_count,
            )
            values = (
                f"{sum(self._att_values) / len(self._att_values):.1f}" if self._att_values else "--",
                f"{sum(self._med_values) / len(self._med_values):.1f}" if self._med_values else "--",
                f"{quality * 100:.0f}%" if quality is not None else "--",
                str(raw_samples) if raw_samples else "--",
            )
        else:
            values = (
                self._stat_att["label"].text(), self._stat_med["label"].text(),
                self._stat_qual["label"].text(), self._stat_samples["label"].text(),
            )
        for key, value in zip(("attention", "meditation", "quality", "samples"), values):
            self._student_stats[key].setText("暂无数据" if value in {"", "--"} else value)
        has_stats = any(value not in {"", "--"} for value in values)
        if self._baseline_active:
            hint = "采集中，以下为当前临时统计。"
        elif has_stats:
            hint = "已显示本次真实基线统计。"
        else:
            hint = "暂无基线统计\n完成采集后将在这里显示个人基线摘要。"
        self._student_stats_hint.setText(hint)

    def _on_duration_change(self, seconds: int):
        if self._role == "teacher":
            return
        self.state._baseline_target = float(seconds)
        self._label_time.setText(f"已用时间：0秒 / {int(self.state._baseline_target)}秒")
        if hasattr(self, "_student_duration") and self._student_duration.value() != seconds:
            self._student_duration.blockSignals(True)
            self._student_duration.setValue(seconds)
            self._student_duration.blockSignals(False)
        self._sync_student_baseline_view()

    def set_identity(self, user_id: str, user_name: str = "") -> None:
        """Reset the reused page, then load only the selected account's baseline."""
        self._input_uid.setText(user_id)
        self._input_name.setText(user_name)
        self._student_uid.setText(user_id)
        self._student_name.blockSignals(True)
        self._student_name.setText(user_name)
        self._student_name.blockSignals(False)
        self._baseline_active = False
        self._eeg_values.clear()
        self._att_values.clear()
        self._med_values.clear()
        self._poor_values.clear()
        self._eeg_plot.reset()
        result = self.baseline_store.get(user_id)
        completed = bool(result and result.get("status") == BASELINE_COMPLETED)
        self._baseline_done = completed
        self.state.baseline_status = BASELINE_COMPLETED if completed else "IDLE"
        self.state._baseline_phase = "done" if completed else "idle"
        self.state._baseline_elapsed = float(result.get("duration_seconds") or 0) if result else 0.0
        self.state._baseline_samples = int(result.get("sample_count") or 0) if result else 0
        self.state._baseline_avg_attention = result.get("avg_attention") if result else None
        self.state._baseline_avg_meditation = result.get("avg_meditation") if result else None
        self.state._baseline_quality_rate = result.get("quality_rate") if result else None
        self.state._baseline_completed_at = result.get("completed_at", "") if result else ""
        progress = 1.0 if completed else 0.0
        self._ring.set_progress(progress)
        self._ring.set_text("100%" if completed else "0%")
        self._progress_bar.setValue(100 if completed else 0)
        self._label_status.setText("基线状态：已完成" if completed else "基线状态：未采集")
        self._label_time.setText(
            f"已用时间：{int(self.state._baseline_elapsed)}秒 / {int(self.state._baseline_target)}秒"
        )
        self._label_samples.setText(f"已采集样本：{self.state._baseline_samples}")
        quality = self.state._baseline_quality_rate
        self._label_quality.setText(
            f"信号合格率：{quality * 100:.0f}%" if quality is not None else "信号合格率：--"
        )
        values = (
            self.state._baseline_avg_attention,
            self.state._baseline_avg_meditation,
            f"{quality * 100:.0f}%" if quality is not None else None,
            self.state._baseline_samples if completed else None,
        )
        for card, value in zip(
            (self._stat_att, self._stat_med, self._stat_qual, self._stat_samples), values
        ):
            card["label"].setText("--" if value is None else str(value))
        self._btn_start.setEnabled(not completed)
        self._btn_stop.setEnabled(False)
        self._btn_next.setEnabled(completed)
        self._sync_student_baseline_view()

    def _start_baseline(self):
        if self._role == "teacher":
            return
        if self.state.device_status != "online" or self.state.connector_status != "online":
            self._label_status.setText("无法采集：设备未连接")
            self._label_status.setStyleSheet("font-size: 16px; color: #F87171;")
            return
        self.state._user_id = self._input_uid.text().strip() or "anonymous"
        self.state._user_name = self._input_name.text().strip() or "匿名用户"
        self.state._baseline_phase = "collecting"
        self.state.baseline_status = BASELINE_COLLECTING
        self._baseline_active = True
        self._baseline_done = False
        self._baseline_start = time.time()
        self._eeg_values.clear()
        self._att_values.clear()
        self._med_values.clear()
        self._poor_values.clear()
        self._baseline_start_raw_count = int(getattr(self.state, "_raw_sample_count", 0))
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._btn_next.setEnabled(False)
        self._label_status.setText("采集中...")
        self._label_status.setStyleSheet("font-size: 16px; color: #4FC3F7;")
        self._ring.set_progress(0.0)
        self._ring.set_text("0%")
        self._progress_bar.setValue(0)

    def _stop_baseline(self):
        """用户提前结束：保留统计预览，但不得宣称基线完成。"""
        if self._role == "teacher":
            return
        self._finish_baseline(completed=False)

    def set_role(self, role: str):
        super().set_role(role)
        if not hasattr(self, "_btn_start"):
            return
        read_only = self._role == "teacher"
        student = self._role == "student"
        admin = not read_only and not student
        legacy_visible = False
        self._user_card.setVisible(legacy_visible)
        self._baseline_actions.setVisible(legacy_visible)
        for card in (self._collect_card, self._signal_card, self._eeg_card, self._stats_card):
            card.setVisible(legacy_visible)
        self._student_baseline_home.setVisible(student)
        self._teacher_baseline_home.setVisible(read_only)
        self._admin_diagnostic_home.setVisible(admin)
        self._combo_duration.setEnabled(not read_only)
        self._btn_start.setVisible(not read_only)
        self._btn_stop.setVisible(not read_only)
        if read_only:
            self._title_label.setText("学生基线状态")
            self._desc_label.setText("查看学生个人基线完成情况与统计摘要；采集由学生本人执行。")
            self._label_status.setText("当前学生基线状态：等待数据")
            self._teacher_baseline_card.setVisible(False)
            self._teacher_timer.start()
            self._refresh_teacher_baseline()
        elif student:
            self._title_label.setText("用户初始化与基线采集")
            self._desc_label.setText("采集 60～90 秒静息态基线数据，用于个人校准和后续状态对比。")
            self._teacher_baseline_card.setVisible(False)
            self._teacher_timer.stop()
            self._sync_student_baseline_view()
        else:
            self._title_label.setText("本机设备诊断")
            self._desc_label.setText(
                "检测 EEG 设备连接状态、采样质量与模型运行环境。"
            )
            self._teacher_baseline_card.setVisible(False)
            self._teacher_timer.stop()
            self._sync_admin_diagnostic_view(self.state)

    def _teacher_student_changed(self, index):
        if self._role == "teacher":
            self.selection_context.select(self._teacher_student_combo.currentData() or "")
            self._refresh_teacher_baseline()

    def _refresh_teacher_baseline(self):
        if self._role != "teacher":
            return
        teacher_id = getattr(self.state, "_user_id", "")
        self.selection_context.set_teacher(teacher_id)
        students = self.binding_store.students_for(teacher_id)
        current = self.selection_context.selected_student_id
        existing = [self._teacher_student_combo.itemData(i)
                    for i in range(self._teacher_student_combo.count())]
        if students != existing:
            self._teacher_student_combo.blockSignals(True)
            self._teacher_student_combo.clear()
            for student_id in students:
                profile = self.identity_store.get_profile(student_id) or {}
                self._teacher_student_combo.addItem(
                    f"{profile.get('name', '')} {student_id}".strip(), student_id
                )
            index = self._teacher_student_combo.findData(current)
            self._teacher_student_combo.setCurrentIndex(index if index >= 0 else (0 if students else -1))
            self._teacher_student_combo.blockSignals(False)
            self.selection_context.select(self._teacher_student_combo.currentData() or "")
        else:
            index = self._teacher_student_combo.findData(current)
            if index >= 0 and index != self._teacher_student_combo.currentIndex():
                self._teacher_student_combo.blockSignals(True)
                self._teacher_student_combo.setCurrentIndex(index)
                self._teacher_student_combo.blockSignals(False)
        student_id = self.selection_context.selected_student_id
        snapshot = self.runtime_registry.get(student_id) if student_id else None
        persisted = self.baseline_store.get(student_id) if student_id else None
        if not student_id:
            self._teacher_baseline_detail.setText("请选择学生")
            self._set_teacher_status("未选择", "不可用", color="#8da3bd",
                                     badge_color="#8da3bd", badge_bg="#1c2636",
                                     badge_border="#31425c")
            self._reset_teacher_baseline_view()
            return
        live = snapshot if snapshot and not snapshot.get("stale") else None
        status_value = live.get("baseline_status") if live else None
        if status_value == "COLLECTING":
            progress = float(live.get("baseline_progress") or 0.0) * 100
            self._teacher_baseline_detail.setText(
                f"当前观察学生：{student_id}\n基线状态：采集中\n"
                f"当前进度：{progress:.0f}% · 接触质量：{live.get('quality_level') or '未知'}\n"
                f"采集时长：{float(live.get('baseline_elapsed') or 0):.0f} 秒 · "
                f"总样本数：{live.get('baseline_samples') or 0}"
            )
            self._set_teacher_status("采集中", "进行中", color="#42b7ff",
                                     badge_color="#42b7ff", badge_bg="#122a3a",
                                     badge_border="#2a5a76")
            self._reset_teacher_baseline_view()
            return
        if status_value in {"FAILED", "EARLY_STOPPED"}:
            message = "本次未形成有效基线" if status_value == "EARLY_STOPPED" else (
                live.get("baseline_failure_reason") or "基线采集失败"
            )
            self._teacher_baseline_detail.setText(
                f"当前观察学生：{student_id}\n基线状态："
                f"{'提前结束' if status_value == 'EARLY_STOPPED' else '失败'}\n{message}"
            )
            self._set_teacher_status(
                "提前结束" if status_value == "EARLY_STOPPED" else "失败",
                "不可用", color="#f5b942" if status_value == "EARLY_STOPPED" else "#ff6b6b",
                badge_color="#f5b942" if status_value == "EARLY_STOPPED" else "#ff6b6b",
                badge_bg="#3a2f1d" if status_value == "EARLY_STOPPED" else "#3a1d20",
                badge_border="#6b5526" if status_value == "EARLY_STOPPED" else "#7f3f47",
            )
            self._reset_teacher_baseline_view()
            return
        result = persisted or (live if status_value == "COMPLETED" else None)
        if not result:
            self._teacher_baseline_detail.setText(
                f"当前观察学生：{student_id}\n尚未完成基线采集"
            )
            self._set_teacher_status("未采集", "不可用", color="#8da3bd",
                                     badge_color="#8da3bd", badge_bg="#1c2636",
                                     badge_border="#31425c")
            self._reset_teacher_baseline_view()
            return
        status = {
            "IDLE": "未采集", "COLLECTING": "采集中", "COMPLETED": "已完成",
            "EARLY_STOPPED": "提前结束", "FAILED": "失败",
        }.get(result.get("status") or result.get("baseline_status"), "已完成")
        quality_rate = result.get("quality_rate", result.get("baseline_quality_rate"))
        quality_text = "--" if quality_rate is None else f"{float(quality_rate) * 100:.0f}%"
        self._teacher_baseline_detail.setText(
            f"当前观察学生：{student_id}\n基线状态：{status}\n"
            f"完成时间：{result.get('completed_at') or result.get('baseline_completed_at') or '--'}\n"
            f"采集时长：{float(result.get('duration_seconds', result.get('baseline_elapsed')) or 0):.0f} 秒 · "
            f"总样本数：{result.get('sample_count', result.get('baseline_samples')) or 0}\n"
            f"平均专注度：{result.get('avg_attention', result.get('baseline_avg_attention')) or '--'} · "
            f"平均放松度：{result.get('avg_meditation', result.get('baseline_avg_meditation')) or '--'}\n"
            f"信号合格率：{quality_text}\n"
            "基线曲线：当前持久化结果未保存采样序列"
        )
        self._set_teacher_status("已完成", "可用")
        self._apply_teacher_baseline_result(result)

    def _complete_baseline(self):
        """仅由达到目标时长的自动流程调用。"""
        self._finish_baseline(completed=True)

    def _finish_baseline(self, completed: bool):
        self._baseline_active = False
        self.state._baseline_phase = "done" if completed else "incomplete"
        self.state.baseline_status = (
            BASELINE_COMPLETED if completed else BASELINE_EARLY_STOPPED
        )
        self._baseline_done = completed
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_next.setEnabled(completed)
        if completed:
            self._ring.set_progress(1.0)
            self._ring.set_text("100%")
            self._progress_bar.setValue(100)
            self._label_status.setText("采集完成")
            self._label_status.setStyleSheet("font-size: 16px; color: #4ADE80;")
        else:
            self._label_status.setText("已提前结束，未形成有效基线")
            self._label_status.setStyleSheet("font-size: 16px; color: #FBBF24;")
        self._finalize_stats()

    def _finalize_stats(self):
        self.state._baseline_elapsed = max(0.0, time.time() - self._baseline_start)
        self.state._baseline_samples = len(self._att_values)
        if not self._att_values:
            return
        avg_att = sum(self._att_values) / len(self._att_values)
        avg_med = sum(self._med_values) / len(self._med_values)
        qualified = sum(1 for p in self._poor_values if p < MAX_POOR_SIGNAL)
        qual_rate = qualified / len(self._poor_values) if self._poor_values else 0.0
        self.state._baseline_avg_attention = avg_att
        self.state._baseline_avg_meditation = avg_med
        self.state._baseline_quality_rate = qual_rate
        if self.state.baseline_status == BASELINE_COMPLETED:
            self.state._baseline_completed_at = time.strftime("%Y-%m-%d %H:%M:%S")
            self.baseline_store.save(getattr(self.state, "_user_id", ""), {
                "status": BASELINE_COMPLETED,
                "completed_at": self.state._baseline_completed_at,
                "duration_seconds": self.state._baseline_elapsed,
                "avg_attention": avg_att,
                "avg_meditation": avg_med,
                "quality_rate": qual_rate,
                "sample_count": self.state._baseline_samples,
            })

        self._stat_att["label"].setText(f"{avg_att:.1f}")
        self._stat_med["label"].setText(f"{avg_med:.1f}")
        self._stat_qual["label"].setText(f"{qual_rate*100:.0f}%")
        self._stat_samples["label"].setText(f"{len(self._att_values)}")
        self._sync_student_baseline_view()

    def _sync_admin_diagnostic_view(self, state):
        """将既有设备/采样状态投影到管理员诊断台，不写入任何状态。"""
        if not hasattr(self, "_admin_diagnostic_home"):
            return
        connector_online = getattr(state, "connector_status", "offline") == "online"
        device_online = getattr(state, "device_status", "offline") == "online"
        has_raw = bool(getattr(state, "_eeg_raw_buffer", []))
        poor = getattr(state, "poor_signal", None)
        attention = getattr(state, "attention", None)
        meditation = getattr(state, "meditation", None)
        quality_level = getattr(state, "quality_level", "rejected")
        measured_rate = getattr(state, "sample_rate_hz", None)
        model_status = str(getattr(state, "model_status", "") or "").upper()

        self._admin_uid.setText(getattr(state, "_user_id", "") or "未设置")
        self._admin_name.setText(getattr(state, "_user_name", "") or "未设置")
        if self._admin_duration.value() != self._combo_duration.value():
            self._admin_duration.blockSignals(True)
            self._admin_duration.setValue(self._combo_duration.value())
            self._admin_duration.blockSignals(False)

        def set_overview_badge(key, text, tone):
            colors = {
                "good": ("#4ADE80", "#173528", "#286246"),
                "warn": ("#FBBF24", "#3A301D", "#80661F"),
                "error": ("#F87171", "#3B2027", "#7F3F47"),
                "info": ("#4FC3F7", "#173247", "#28607D"),
            }
            foreground, background, border = colors[tone]
            label = self._admin_overview[key]["value"]
            label.setText(text)
            label.setStyleSheet(
                f"color:{foreground};background:{background};border:1px solid {border};"
                "border-radius:6px;padding:3px 9px;font-weight:700;"
            )

        set_overview_badge("connector", "已连接" if connector_online else "未连接",
                           "good" if connector_online else "error")
        device_text = "已连接" if device_online and has_raw else ("等待原始数据" if device_online else "未连接")
        set_overview_badge("device", "MindWave / ThinkGear", "info")
        set_overview_badge(
            "sampling", "暂无数据" if measured_rate is None else f"{float(measured_rate):.0f} Hz",
            "good" if measured_rate is not None else "warn",
        )
        baseline_status = getattr(state, "baseline_status", "IDLE")
        warmup = float(getattr(state, "warmup_progress", 0.0) or 0.0)
        model_text = {"READY": "正常", "FAILED": "故障", "LOADING": "正在准备"}.get(
            model_status, "暂无数据"
        )
        set_overview_badge(
            "analysis", model_text,
            "good" if model_status == "READY" else ("error" if model_status == "FAILED" else "warn"),
        )
        self._admin_device_values["rate"].setText(
            "暂无数据" if measured_rate is None else f"{float(measured_rate):.0f} Hz"
        )
        self._admin_device_values["checked"].setText(
            time.strftime("%H:%M:%S") if connector_online or device_online or has_raw else "暂无数据"
        )

        if getattr(state, "_eeg_raw_buffer", None):
            self._admin_eeg_plot.push_buffer(state._eeg_raw_buffer)
        else:
            self._admin_eeg_plot.reset()
        self._admin_eeg_info.setText(
            f"目标采样率 {DEVICE_TARGET_SAMPLE_HZ} Hz"
        )
        self._admin_eeg_status.setText(
            "实时监测 · " + ("正在采样" if self._baseline_active else ("数据已就绪" if has_raw else "等待数据"))
        )

        if poor is None:
            self._admin_signals["poor"].set_state(StatusIndicator.LEVEL_ERROR, "无信号")
        elif poor < MAX_POOR_SIGNAL:
            self._admin_signals["poor"].set_state(StatusIndicator.LEVEL_GOOD, "良好")
        elif poor < 200:
            self._admin_signals["poor"].set_state(StatusIndicator.LEVEL_WARN, "需调整")
        else:
            self._admin_signals["poor"].set_state(StatusIndicator.LEVEL_ERROR, "无信号")
        self._admin_signals["attention"].set_state(
            StatusIndicator.LEVEL_NEUTRAL, "暂无数据" if attention is None else f"{attention:.0f}")
        self._admin_signals["meditation"].set_state(
            StatusIndicator.LEVEL_NEUTRAL, "暂无数据" if meditation is None else f"{meditation:.0f}")
        quality_text = {"trusted": "良好", "warning": "需关注", "rejected": "不可用"}.get(quality_level, "不可用")
        quality_state = (StatusIndicator.LEVEL_GOOD if quality_level == "trusted" else
                         StatusIndicator.LEVEL_WARN if quality_level == "warning" else StatusIndicator.LEVEL_ERROR)
        self._admin_signals["quality"].set_state(quality_state, quality_text)

        elapsed = int(max(0.0, time.time() - self._baseline_start)) if self._baseline_active else int(
            getattr(state, "_baseline_elapsed", 0.0) or 0.0)
        target = int(getattr(state, "_baseline_target", self._combo_duration.value()) or self._combo_duration.value())
        progress = min(100, round(elapsed / target * 100)) if target else 0
        sample_count = max(0, int(getattr(state, "_raw_sample_count", 0)) - self._baseline_start_raw_count) if self._baseline_active else int(
            getattr(state, "_baseline_samples", 0) or 0)
        status_text = {BASELINE_COLLECTING: "采样中", BASELINE_COMPLETED: "已完成",
                       BASELINE_EARLY_STOPPED: "提前结束", BASELINE_FAILED: "采样失败"}.get(
                           baseline_status, "等待设备数据" if not has_raw else "未采样")
        self._admin_sample_status.setText(status_text)
        self._admin_sample_time.setText(f"采集时间：{elapsed} / {target} 秒")
        self._admin_sample_count.setText(f"有效样本：{sample_count if sample_count else '暂无数据'}")
        quality_rate = getattr(state, "_baseline_quality_rate", None)
        if self._baseline_active and self._poor_values:
            quality_rate = sum(1 for value in self._poor_values if value < MAX_POOR_SIGNAL) / len(self._poor_values)
        quality_value = "暂无数据" if quality_rate is None else f"{float(quality_rate) * 100:.0f}%"
        self._admin_sample_quality.setText(f"信号质量：{quality_value}")
        self._admin_ring.set_progress(progress / 100); self._admin_ring.set_text(f"{progress}%")
        self._admin_progress.setValue(progress)

        values = {
            "attention": getattr(state, "_baseline_avg_attention", None),
            "meditation": getattr(state, "_baseline_avg_meditation", None),
            "quality": quality_rate,
            "samples": sample_count or None,
        }
        self._admin_stats["attention"].setText("暂无数据" if values["attention"] is None else f"{float(values['attention']):.1f}")
        self._admin_stats["meditation"].setText("暂无数据" if values["meditation"] is None else f"{float(values['meditation']):.1f}")
        self._admin_stats["quality"].setText("暂无数据" if values["quality"] is None else f"{float(values['quality']) * 100:.0f}%")
        self._admin_stats["samples"].setText("暂无数据" if values["samples"] is None else str(values["samples"]))
        self._admin_start.setEnabled(self._btn_start.isEnabled())
        self._admin_stop.setEnabled(self._btn_stop.isEnabled())
        self._admin_hint.setText(
            "设备就绪后可开始采集基线。" if connector_online and device_online and has_raw
            else "当前未检测到完整设备数据，可先完成设备连接再开始诊断。"
        )

    def update_state(self, state):
        if self._role == "teacher":
            self._refresh_teacher_baseline()
            return
        # EEG曲线 — 使用内部缓冲 _eeg_raw_buffer
        if state._eeg_raw_buffer:
            self._eeg_plot.push_buffer(state._eeg_raw_buffer)
            self._student_eeg_plot.push_buffer(state._eeg_raw_buffer)

        device_online = (
            state.device_status == "online" and state.connector_status == "online"
        )
        if not self._baseline_active and self._role != "teacher":
            self._btn_start.setEnabled(device_online)
            self._btn_start.setToolTip(
                "开始60～90秒静息基线采集" if device_online
                else "需先连接ThinkGear Connector并收到MindWave Raw数据"
            )
        self._eeg_info.setText(
            f"实时设备 · {DEVICE_TARGET_SAMPLE_HZ} Hz · 显示降采样（不影响模型）"
            if device_online else
            f"目标采样率 {DEVICE_TARGET_SAMPLE_HZ} Hz · 等待设备数据"
        )
        self._student_eeg_info.setText(
            "最近 5 秒 · 设备数据持续更新" if state._eeg_raw_buffer
            else "等待设备数据"
        )

        # 信号指示 — poor_signal 可能为 None
        poor = state.poor_signal
        if poor is None:
            self._ind_poor.set_state(StatusIndicator.LEVEL_NEUTRAL, "等待信号")
            self._student_signal_indicators["poor"].set_state(
                StatusIndicator.LEVEL_NEUTRAL, "暂无可用数据")
        elif poor < MAX_POOR_SIGNAL:
            self._ind_poor.set_state(StatusIndicator.LEVEL_GOOD, f"{poor}")
            self._student_signal_indicators["poor"].set_state(
                StatusIndicator.LEVEL_GOOD, "良好")
        elif poor < 200:
            self._ind_poor.set_state(StatusIndicator.LEVEL_WARN, f"{poor}")
            self._student_signal_indicators["poor"].set_state(
                StatusIndicator.LEVEL_WARN, "需要调整")
        else:
            self._ind_poor.set_state(StatusIndicator.LEVEL_ERROR, "无信号")
            self._student_signal_indicators["poor"].set_state(
                StatusIndicator.LEVEL_ERROR, "无信号")

        # Attention / Meditation — 现在为 float | None
        att = state.attention
        med = state.meditation
        self._ind_att.set_state(
            StatusIndicator.LEVEL_NEUTRAL,
            f"{att:.0f}" if att is not None else "--",
        )
        self._ind_med.set_state(
            StatusIndicator.LEVEL_NEUTRAL,
            f"{med:.0f}" if med is not None else "--",
        )
        self._student_signal_indicators["attention"].set_state(
            StatusIndicator.LEVEL_NEUTRAL,
            f"{att:.0f}" if att is not None else "暂无可用数据",
        )
        self._student_signal_indicators["meditation"].set_state(
            StatusIndicator.LEVEL_NEUTRAL,
            f"{med:.0f}" if med is not None else "暂无可用数据",
        )

        # 信号质量等级 — 使用 quality_level 替代 signal_confidence
        level = state.quality_level
        if level == "trusted":
            self._ind_conf.set_state(StatusIndicator.LEVEL_GOOD, "可信")
            self._student_signal_indicators["quality"].set_state(
                StatusIndicator.LEVEL_GOOD, "可用")
        elif level == "warning":
            self._ind_conf.set_state(StatusIndicator.LEVEL_WARN, "警告")
            self._student_signal_indicators["quality"].set_state(
                StatusIndicator.LEVEL_WARN, "需要关注")
        else:  # rejected
            self._ind_conf.set_state(StatusIndicator.LEVEL_NEUTRAL, "暂不可用")
            self._student_signal_indicators["quality"].set_state(
                StatusIndicator.LEVEL_NEUTRAL, "暂无可用数据")

        # 基线采集中
        if self._baseline_active:
            if not device_online:
                self.state._baseline_phase = "failed"
                self.state.baseline_status = BASELINE_FAILED
                self.state._baseline_failure_reason = "EEG 设备连接已断开，请重新连接后重新采集"
                self._baseline_active = False
                self._baseline_done = False
                self._btn_start.setEnabled(False)
                self._btn_stop.setEnabled(False)
                self._btn_next.setEnabled(False)
                self._label_status.setText("采集失败：设备连接已中断")
                self._label_status.setStyleSheet("font-size: 16px; color: #F87171;")
                self._sync_student_baseline_view()
                return
            elapsed = time.time() - self._baseline_start
            if att is not None:
                self._att_values.append(att)
            if med is not None:
                self._med_values.append(med)
            if poor is not None:
                self._poor_values.append(poor)

            target = state._baseline_target
            pct = min(1.0, elapsed / target)
            self._ring.set_progress(pct)
            self._ring.set_text(f"{pct*100:.0f}%")
            self._ring.set_subtext(f"{int(elapsed)}s / {int(target)}s")
            self._label_time.setText(
                f"已用时间：{int(elapsed)}秒 / {int(target)}秒"
            )
            raw_samples = max(
                0,
                int(getattr(state, "_raw_sample_count", 0))
                - self._baseline_start_raw_count,
            )
            self._label_samples.setText(f"已接收 Raw：{raw_samples}")

            qualified = sum(1 for p in self._poor_values if p < MAX_POOR_SIGNAL)
            qual_rate = qualified / len(self._poor_values) if self._poor_values else 0.0
            self._label_quality.setText(f"信号合格率：{qual_rate*100:.0f}%")

            self._student_stats["attention"].setText(
                f"{sum(self._att_values) / len(self._att_values):.1f}"
                if self._att_values else "暂无数据"
            )
            self._student_stats["meditation"].setText(
                f"{sum(self._med_values) / len(self._med_values):.1f}"
                if self._med_values else "暂无数据"
            )
            self._student_stats["quality"].setText(
                f"{qual_rate*100:.0f}%" if self._poor_values else "暂无数据"
            )
            self._student_stats["samples"].setText(str(raw_samples) if raw_samples else "暂无数据")

            self._progress_bar.setValue(int(pct * 100))

            # 自动结束
            if elapsed >= target:
                self._complete_baseline()
        self._sync_student_baseline_view()
        if self._role not in {"student", "teacher"}:
            self._sync_admin_diagnostic_view(state)

    def on_hide(self):
        pass
