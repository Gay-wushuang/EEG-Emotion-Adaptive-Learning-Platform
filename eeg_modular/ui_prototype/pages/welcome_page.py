"""页面1：欢迎与设备检查页。"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
    QPushButton, QFrame, QSizePolicy,
)

from pages.base_page import BasePage
from widgets.card import Card
from widgets.status_indicator import StatusIndicator
from services.dashboard_state import (
    MAX_POOR_SIGNAL, WARMUP_SECONDS,
    MOCK_UI_REFRESH_HZ, DEVICE_TARGET_SAMPLE_HZ,
)


class WelcomePage(BasePage):
    def __init__(self, state, service):
        self.state = state
        self.service = service
        super().__init__(
            "欢迎与设备检查",
            "在开始学习状态监测前，请确认设备连接与信号状态正常。"
        )
        self._build_ui()

    def _build_ui(self):
        # ── 顶部欢迎区 ──
        welcome_card = Card("系统简介")
        intro = QLabel(
            "智学脑机助手通过 MindWave 单通道脑电设备，持续观察学习过程中的状态变化。\n"
            "系统融合 EEG 时域与频域特征，输出积极、中性、负性三类状态概率，并独立评估信号可信度，\n"
            "为学习节奏调整、专注趋势观察和阶段复盘提供辅助信息。\n\n"
            "请先启动 ThinkGear Connector 并正确佩戴设备；只有信号质量合格后，系统才会提供状态解释。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #C5CDD9; font-size: 14px; line-height: 1.6;")
        welcome_card.add_widget(intro)
        self.content_layout.addWidget(welcome_card)

        # ── 设备检查卡片网格 ──
        check_layout = QGridLayout()
        check_layout.setSpacing(14)

        # 1. ThinkGear Connector
        self._card_connector = Card("ThinkGear Connector")
        self._ind_connector = StatusIndicator("连接状态")
        self._card_connector.add_widget(self._ind_connector)
        info_connector = QLabel("本地代理服务，负责与MindWave设备通信。\n地址：127.0.0.1:13854")
        info_connector.setStyleSheet("color: #6B7689; font-size: 12px;")
        info_connector.setWordWrap(True)
        self._card_connector.add_widget(info_connector)
        check_layout.addWidget(self._card_connector, 0, 0)

        # 2. MindWave设备
        self._card_device = Card("MindWave 设备")
        self._ind_device = StatusIndicator("设备状态")
        self._card_device.add_widget(self._ind_device)
        info_device = QLabel(
            f"NeuroSky MindWave Mobile2 单通道脑电头环。\n"
            f"设备目标采样率：{DEVICE_TARGET_SAMPLE_HZ}Hz | 连接方式：蓝牙/TCP"
        )
        info_device.setStyleSheet("color: #6B7689; font-size: 12px;")
        info_device.setWordWrap(True)
        self._card_device.add_widget(info_device)
        check_layout.addWidget(self._card_device, 0, 1)

        # 3. 信号质量
        self._card_signal = Card("信号质量")
        self._ind_signal = StatusIndicator("Poor Signal")
        self._card_signal.add_widget(self._ind_signal)
        info_signal = QLabel(f"阈值：< {MAX_POOR_SIGNAL} 为合格 | 200 = 无信号\n建议：调整电极接触，保持静止")
        info_signal.setStyleSheet("color: #6B7689; font-size: 12px;")
        info_signal.setWordWrap(True)
        self._card_signal.add_widget(info_signal)
        check_layout.addWidget(self._card_signal, 1, 0)

        # 4. 采样率
        self._card_sample = Card("采样率")
        self._ind_sample = StatusIndicator("数据流")
        self._card_sample.add_widget(self._ind_sample)
        info_sample = QLabel(
            f"设备目标采样率：{DEVICE_TARGET_SAMPLE_HZ}Hz\n"
            "实时数据：当前未接入\n"
            "支持数据：Raw EEG / Attention / Meditation"
        )
        info_sample.setStyleSheet("color: #6B7689; font-size: 12px;")
        info_sample.setWordWrap(True)
        self._card_sample.add_widget(info_sample)
        check_layout.addWidget(self._card_sample, 1, 1)

        self.content_layout.addLayout(check_layout)

        # ── 预热提示 ──
        self._card_warmup = Card("预热阶段")
        warmup_info = QLabel(
            f"系统启动后需要进行 {int(WARMUP_SECONDS)} 秒预热，期间采集数据填充分析窗口。\n"
            "预热完成后，模型推理结果方可用于状态解释。"
        )
        warmup_info.setWordWrap(True)
        warmup_info.setStyleSheet("color: #C5CDD9; font-size: 13px;")
        self._card_warmup.add_widget(warmup_info)

        self._warmup_progress_label = QLabel("预热进度：0%")
        self._warmup_progress_label.setObjectName("AccentLabel")
        self._warmup_progress_label.setStyleSheet("font-size: 14px;")
        self._card_warmup.add_widget(self._warmup_progress_label)

        self.content_layout.addWidget(self._card_warmup)

        # ── 操作按钮 ──
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self._btn_start = QPushButton("进入基线采集")
        self._btn_start.setObjectName("PrimaryButton")
        self._btn_start.setToolTip("进入用户信息与静息基线采集页面")
        btn_layout.addWidget(self._btn_start)

        self.content_layout.addLayout(btn_layout)
        self._next_hint = QLabel("设备在线后建议先完成60～90秒基线；也可先进入页面查看流程。")
        self._next_hint.setAlignment(Qt.AlignRight)
        self._next_hint.setStyleSheet("color: #8491A5; font-size: 12px;")
        self.content_layout.addWidget(self._next_hint)
        self.content_layout.addStretch()

    def update_state(self, state):
        # Connector
        if state.connector_status == "online":
            self._ind_connector.set_state(StatusIndicator.LEVEL_GOOD, "已连接")
        elif state.connector_status == "connecting":
            self._ind_connector.set_state(StatusIndicator.LEVEL_WARN, "连接中...")
        else:
            self._ind_connector.set_state(StatusIndicator.LEVEL_NEUTRAL, "等待启动")

        # Device — Mock模式不假装设备已连接
        if state.device_status == "online":
            self._ind_device.set_state(StatusIndicator.LEVEL_GOOD, "在线")
        elif state.device_status == "waiting_raw":
            self._ind_device.set_state(StatusIndicator.LEVEL_WARN, "等待原始数据")
        else:
            self._ind_device.set_state(StatusIndicator.LEVEL_NEUTRAL, "未检测到设备")

        # Signal — poor_signal 可能为 None
        poor = state.poor_signal
        if poor is None:
            self._ind_signal.set_state(StatusIndicator.LEVEL_NEUTRAL, "等待信号")
        elif poor < MAX_POOR_SIGNAL:
            self._ind_signal.set_state(StatusIndicator.LEVEL_GOOD, f"{poor}（合格）")
        elif poor < 200:
            self._ind_signal.set_state(StatusIndicator.LEVEL_WARN, f"{poor}（警告）")
        else:
            self._ind_signal.set_state(StatusIndicator.LEVEL_ERROR, "无信号")

        # Sample rate — 使用常量展示，不引用已移除的 raw_packet_count
        if state.mode != "replay" and state.device_status == "online":
            rate = f"{state.sample_rate_hz:.0f}Hz" if state.sample_rate_hz else "数据流活跃"
            self._ind_sample.set_state(StatusIndicator.LEVEL_GOOD, rate)
        elif state.mode != "replay":
            self._ind_sample.set_state(StatusIndicator.LEVEL_NEUTRAL, "无设备数据")
        else:
            self._ind_sample.set_state(StatusIndicator.LEVEL_NEUTRAL, "回放模式")

        # Warmup — 使用 warmup_progress (0.0~1.0)
        pct = state.warmup_progress * 100
        if state.warmup_complete:
            self._warmup_progress_label.setText("预热进度：100% ✓ 已完成")
            self._warmup_progress_label.setStyleSheet("font-size: 14px; color: #4ADE80;")
        else:
            self._warmup_progress_label.setText(f"预热进度：{pct:.0f}%")
        if state.connector_status == "online" and state.device_status == "online":
            self._next_hint.setText("设备与数据流已就绪，可以进入基线采集。")
            self._next_hint.setStyleSheet("color: #4ADE80; font-size: 12px;")
        else:
            self._next_hint.setText("尚未检测到完整设备数据；可进入基线页查看，但暂不能开始采集。")
            self._next_hint.setStyleSheet("color: #FBBF24; font-size: 12px;")
