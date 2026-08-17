"""页面2：用户初始化与60～90秒基线采集页。"""

from __future__ import annotations

import time
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
    QPushButton, QLineEdit, QFrame, QProgressBar, QComboBox, QSizePolicy,
)

from pages.base_page import BasePage
from widgets.card import Card
from widgets.status_indicator import StatusIndicator
from widgets.progress_ring import ProgressRing
from widgets.eeg_plot import EEGPlotWidget
from services.dashboard_state import (
    WARMUP_SECONDS, MAX_POOR_SIGNAL,
    MOCK_UI_REFRESH_HZ, DEVICE_TARGET_SAMPLE_HZ,
)


class BaselinePage(BasePage):
    def __init__(self, state, service):
        self.state = state
        self.service = service
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

    def _build_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setSpacing(14)

        # ── 左侧：用户信息 + 采集控制 ──
        left = QVBoxLayout()
        left.setSpacing(14)

        # 用户信息卡片
        user_card = Card("用户信息")
        form = QGridLayout()
        form.setSpacing(10)

        form.addWidget(QLabel("用户ID:"), 0, 0)
        self._input_uid = QLineEdit(self.state._user_id)
        self._input_uid.setPlaceholderText("输入用户ID")
        form.addWidget(self._input_uid, 0, 1)

        form.addWidget(QLabel("姓名:"), 0, 2)
        self._input_name = QLineEdit(self.state._user_name)
        self._input_name.setPlaceholderText("输入姓名")
        form.addWidget(self._input_name, 0, 3)

        form.addWidget(QLabel("任务类型:"), 1, 0)
        self._combo_task = QComboBox()
        self._combo_task.addItems(["数学练习", "英语阅读", "编程任务", "物理复习", "自由学习"])
        form.addWidget(self._combo_task, 1, 1)

        form.addWidget(QLabel("采集时长:"), 1, 2)
        self._combo_duration = QComboBox()
        self._combo_duration.addItems(["60秒", "75秒", "90秒"])
        self._combo_duration.setCurrentIndex(1)
        self._combo_duration.currentIndexChanged.connect(self._on_duration_change)
        form.addWidget(self._combo_duration, 1, 3)

        user_card.add_widget(self._wrap_layout(form))
        left.addWidget(user_card)

        # 采集进度卡片
        collect_card = Card("基线采集")

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
        sig_layout = QGridLayout()
        sig_layout.setSpacing(8)

        self._ind_poor = StatusIndicator("Poor Signal")
        sig_layout.addWidget(self._ind_poor, 0, 0)
        self._ind_att = StatusIndicator("Attention")
        sig_layout.addWidget(self._ind_att, 0, 1)
        self._ind_med = StatusIndicator("Meditation")
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

        self._btn_stop = QPushButton("结束采集")
        self._btn_stop.setEnabled(False)
        self._btn_stop.setToolTip("基线采集中可提前结束")
        self._btn_stop.clicked.connect(self._stop_baseline)
        btn_layout.addWidget(self._btn_stop)

        self._btn_next = QPushButton("打开实时分析")
        self._btn_next.setObjectName("SuccessButton")
        self._btn_next.setEnabled(False)
        self._btn_next.setToolTip("完成一次基线采集后可打开实时分析")
        btn_layout.addWidget(self._btn_next)

        left.addLayout(btn_layout)
        main_layout.addLayout(left, 0)

        # ── 右侧：EEG实时曲线 + 采集统计 ──
        right = QVBoxLayout()
        right.setSpacing(14)

        eeg_card = Card("EEG实时信号")
        self._eeg_plot = EEGPlotWidget()
        self._eeg_plot.setMinimumHeight(220)
        eeg_card.add_widget(self._eeg_plot)
        self._eeg_info = QLabel(f"目标采样率 {DEVICE_TARGET_SAMPLE_HZ} Hz · 等待设备数据")
        self._eeg_info.setStyleSheet("color: #6B7689; font-size: 12px;")
        eeg_card.add_widget(self._eeg_info)
        right.addWidget(eeg_card)

        # 基线统计预览
        stats_card = Card("基线统计预览")
        stats_layout = QGridLayout()
        stats_layout.setSpacing(8)

        self._stat_att = self._make_stat("平均Attention", "--")
        stats_layout.addWidget(self._stat_att["card"], 0, 0)
        self._stat_med = self._make_stat("平均Meditation", "--")
        stats_layout.addWidget(self._stat_med["card"], 0, 1)
        self._stat_qual = self._make_stat("信号合格率", "--")
        stats_layout.addWidget(self._stat_qual["card"], 1, 0)
        self._stat_samples = self._make_stat("总样本数", "--")
        stats_layout.addWidget(self._stat_samples["card"], 1, 1)

        stats_card.add_widget(self._wrap_layout(stats_layout))
        right.addWidget(stats_card)

        main_layout.addLayout(right, 1)
        self.content_layout.addLayout(main_layout)

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

    def _on_duration_change(self, idx: int):
        durations = [60.0, 75.0, 90.0]
        self.state._baseline_target = durations[idx]
        self._label_time.setText(f"已用时间：0秒 / {int(self.state._baseline_target)}秒")

    def _start_baseline(self):
        if self.state.device_status != "online" or self.state.connector_status != "online":
            self._label_status.setText("无法采集：设备未连接")
            self._label_status.setStyleSheet("font-size: 16px; color: #F87171;")
            return
        self.state._user_id = self._input_uid.text() or "demo_user"
        self.state._user_name = self._input_name.text() or "演示用户"
        self.state._baseline_phase = "collecting"
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

    def _stop_baseline(self):
        self._baseline_active = False
        self.state._baseline_phase = "done"
        self._baseline_done = True
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_next.setEnabled(True)
        self._label_status.setText("采集完成")
        self._label_status.setStyleSheet("font-size: 16px; color: #4ADE80;")
        self._finalize_stats()

    def _finalize_stats(self):
        if not self._att_values:
            return
        avg_att = sum(self._att_values) / len(self._att_values)
        avg_med = sum(self._med_values) / len(self._med_values)
        qualified = sum(1 for p in self._poor_values if p < MAX_POOR_SIGNAL)
        qual_rate = qualified / len(self._poor_values) if self._poor_values else 0.0

        self._stat_att["label"].setText(f"{avg_att:.1f}")
        self._stat_med["label"].setText(f"{avg_med:.1f}")
        self._stat_qual["label"].setText(f"{qual_rate*100:.0f}%")
        self._stat_samples["label"].setText(f"{len(self._att_values)}")

        self.state._baseline_elapsed = time.time() - self._baseline_start
        self.state._baseline_samples = len(self._att_values)

    def update_state(self, state):
        # EEG曲线 — 使用内部缓冲 _eeg_raw_buffer
        if state._eeg_raw_buffer:
            self._eeg_plot.push_buffer(state._eeg_raw_buffer)

        device_online = (
            state.device_status == "online" and state.connector_status == "online"
        )
        if not self._baseline_active:
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

        # 信号指示 — poor_signal 可能为 None
        poor = state.poor_signal
        if poor is None:
            self._ind_poor.set_state(StatusIndicator.LEVEL_NEUTRAL, "等待信号")
        elif poor < MAX_POOR_SIGNAL:
            self._ind_poor.set_state(StatusIndicator.LEVEL_GOOD, f"{poor}")
        elif poor < 200:
            self._ind_poor.set_state(StatusIndicator.LEVEL_WARN, f"{poor}")
        else:
            self._ind_poor.set_state(StatusIndicator.LEVEL_ERROR, "无信号")

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

        # 信号质量等级 — 使用 quality_level 替代 signal_confidence
        level = state.quality_level
        if level == "trusted":
            self._ind_conf.set_state(StatusIndicator.LEVEL_GOOD, "可信")
        elif level == "warning":
            self._ind_conf.set_state(StatusIndicator.LEVEL_WARN, "警告")
        else:  # rejected
            self._ind_conf.set_state(StatusIndicator.LEVEL_NEUTRAL, "暂不可用")

        # 基线采集中
        if self._baseline_active:
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

            self._progress_bar.setValue(int(pct * 100))

            # 自动结束
            if elapsed >= target:
                self._stop_baseline()

    def on_hide(self):
        pass
