"""主窗口：侧边导航 + 页面路由 + 底部状态栏。

状态栏严格使用 DashboardState 的正式字段（AGENTS.md 第6节）：
  - connector_status: offline | connecting | online
  - device_status: offline | waiting_raw | online
  - poor_signal: int | None
  - quality_level: trusted | warning | rejected
  - session_seconds: float
  - _session_active: bool（内部簿记）
  - mode: live | replay（mock 模式由采集线程报告为 "mock"）

Mock 模式下 connector_status / device_status 均为 offline，
状态栏不会将设备显示为已连接，以避免误导用户。
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QStackedWidget, QButtonGroup, QStatusBar,
    QFrame, QSizePolicy, QSpacerItem,
)

from services.dashboard_state import (
    DashboardState,
    WARMUP_SECONDS,
    MAX_POOR_SIGNAL,
    CLASS_DISPLAY,
    MOCK_UI_REFRESH_HZ,
    DEVICE_TARGET_SAMPLE_HZ,
)
from services.mock_data_service import MockDataService
from services.font_loader import ensure_chinese_font

from pages.welcome_page import WelcomePage
from pages.baseline_page import BaselinePage
from pages.dashboard_page import DashboardPage
from pages.task_page import TaskPage
from pages.history_page import HistoryPage
from pages.settings_page import SettingsPage
from pages.replay_page import ReplayPage


NAV_ITEMS = [
    ("welcome", "欢迎与设备检查", "1"),
    ("baseline", "基线采集", "2"),
    ("dashboard", "实时仪表盘", "3"),
    ("task", "任务与标记", "4"),
    ("history", "历史与报告", "5"),
    ("settings", "设置与诊断", "6"),
    ("replay", "CSV回放", "7"),
]


class MainWindow(QMainWindow):
    """主窗口。"""

    def __init__(self):
        super().__init__()
        ensure_chinese_font()
        self.setWindowTitle("智学脑机助手 - 单通道脑机接口学习状态辅助系统")
        self.resize(1920, 1080)
        self.setMinimumSize(1280, 700)

        # ── 核心状态与服务 ──
        self.state = DashboardState()
        self.service = MockDataService(self.state)
        self.service.start_streaming()

        # ── UI 构建 ──
        root = QWidget()
        root.setObjectName("RootWidget")
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 侧边栏
        sidebar = self._build_sidebar()
        root_layout.addWidget(sidebar)

        # 右侧主区域
        right_area = QWidget()
        right_layout = QVBoxLayout(right_area)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # 页面栈
        self.stack = QStackedWidget()
        self._pages = {}
        self._build_pages()
        right_layout.addWidget(self.stack, 1)

        # 状态栏
        self._status_bar = self._build_status_bar()
        right_layout.addWidget(self._status_bar)

        root_layout.addWidget(right_area, 1)

        # 连接状态更新
        self.state.state_updated.connect(self._on_state_updated)
        self.state.event_added.connect(self._on_event_added)

        # 默认显示欢迎页
        self._navigate_to("welcome")

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("SideBar")
        sidebar.setFixedWidth(220)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 20, 14, 20)
        layout.setSpacing(4)

        # Logo / 标题
        title = QLabel("智学脑机助手")
        title.setObjectName("AppTitle")
        layout.addWidget(title)

        subtitle = QLabel("Brain-Computer Learning Assistant")
        subtitle.setObjectName("AppSubtitle")
        layout.addWidget(subtitle)

        layout.addSpacing(16)

        # 导航按钮
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        self._nav_buttons = {}

        for key, label, num in NAV_ITEMS:
            btn = QPushButton(f"  {num}  {label}")
            btn.setObjectName("NavButton")
            btn.setCheckable(True)
            btn.setMinimumHeight(42)
            btn.clicked.connect(lambda checked, k=key: self._navigate_to(k))
            self._nav_group.addButton(btn)
            self._nav_buttons[key] = btn
            layout.addWidget(btn)

        layout.addStretch()

        # 底部版本信息（明确标注 Mock 演示数据）
        version = QLabel("v1.0.0  |  Mock模式 - 演示数据")
        version.setObjectName("AppSubtitle")
        version.setAlignment(Qt.AlignCenter)
        layout.addWidget(version)

        return sidebar

    def _build_pages(self):
        page_classes = [
            ("welcome", WelcomePage),
            ("baseline", BaselinePage),
            ("dashboard", DashboardPage),
            ("task", TaskPage),
            ("history", HistoryPage),
            ("settings", SettingsPage),
            ("replay", ReplayPage),
        ]
        for key, cls in page_classes:
            page = cls(self.state, self.service)
            self._pages[key] = page
            self.stack.addWidget(page)

    def _build_status_bar(self) -> QWidget:
        bar = QFrame()
        bar.setFixedHeight(32)
        bar.setStyleSheet("background-color: #141821; border-top: 1px solid #2A3142;")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(24)

        self._sb_connector = QLabel("ThinkGear Connector: --")
        self._sb_connector.setStyleSheet("color: #6B7689; font-size: 12px;")
        layout.addWidget(self._sb_connector)

        self._sb_device = QLabel("MindWave: --")
        self._sb_device.setStyleSheet("color: #6B7689; font-size: 12px;")
        layout.addWidget(self._sb_device)

        self._sb_signal = QLabel("Poor Signal: --")
        self._sb_signal.setStyleSheet("color: #6B7689; font-size: 12px;")
        layout.addWidget(self._sb_signal)

        layout.addStretch()

        self._sb_session = QLabel("会话: 未开始")
        self._sb_session.setStyleSheet("color: #6B7689; font-size: 12px;")
        layout.addWidget(self._sb_session)

        self._sb_mode = QLabel("模式: --")
        self._sb_mode.setStyleSheet("color: #6B7689; font-size: 12px;")
        layout.addWidget(self._sb_mode)

        return bar

    def _navigate_to(self, key: str):
        if key not in self._pages:
            return
        # 通知旧页面隐藏
        current = self.stack.currentWidget()
        if current and hasattr(current, "on_hide"):
            current.on_hide()

        self.stack.setCurrentWidget(self._pages[key])
        self._nav_buttons[key].setChecked(True)

        # 通知新页面显示
        new_page = self._pages[key]
        if hasattr(new_page, "on_show"):
            new_page.on_show()
        # 立即刷新一次
        new_page.update_state(self.state)

    def _on_state_updated(self, state):
        """根据 DashboardState 正式字段刷新状态栏。"""
        s = state

        # ── Connector 状态：offline | connecting | online ──
        cs = s.connector_status
        if cs == "online":
            self._sb_connector.setText("ThinkGear Connector: 已连接")
            self._sb_connector.setStyleSheet("color: #4ADE80; font-size: 12px;")
        elif cs == "connecting":
            self._sb_connector.setText("ThinkGear Connector: 连接中")
            self._sb_connector.setStyleSheet("color: #FBBF24; font-size: 12px;")
        else:  # offline（Mock 模式即在此分支）
            self._sb_connector.setText("ThinkGear Connector: 未连接")
            self._sb_connector.setStyleSheet("color: #F87171; font-size: 12px;")

        # ── Device 状态：offline | waiting_raw | online ──
        ds = s.device_status
        if ds == "online":
            self._sb_device.setText("MindWave: 在线")
            self._sb_device.setStyleSheet("color: #4ADE80; font-size: 12px;")
        elif ds == "waiting_raw":
            self._sb_device.setText("MindWave: 等待信号")
            self._sb_device.setStyleSheet("color: #FBBF24; font-size: 12px;")
        else:  # offline（Mock 模式即在此分支）
            self._sb_device.setText("MindWave: 离线")
            self._sb_device.setStyleSheet("color: #F87171; font-size: 12px;")

        # ── 信号质量：poor_signal (int | None) + quality_level ──
        poor = s.poor_signal
        ql = s.quality_level
        if poor is None:
            self._sb_signal.setText("Poor Signal: --")
            self._sb_signal.setStyleSheet("color: #6B7689; font-size: 12px;")
        else:
            if ql == "trusted":
                sig_color = "#4ADE80"
                sig_text = f"Poor Signal: {poor} (良好)"
            elif ql == "warning":
                sig_color = "#FBBF24"
                sig_text = f"Poor Signal: {poor} (警告)"
            else:  # rejected
                sig_color = "#F87171"
                sig_text = f"Poor Signal: {poor} (不合格)"
            self._sb_signal.setText(sig_text)
            self._sb_signal.setStyleSheet(f"color: {sig_color}; font-size: 12px;")

        # ── 会话时间：session_seconds + _session_active ──
        if s._session_active:
            secs_total = int(s.session_seconds)
            mins = secs_total // 60
            secs = secs_total % 60
            self._sb_session.setText(f"会话: {mins:02d}:{secs:02d}")
        else:
            self._sb_session.setText("会话: 未开始")

        # ── 模式：live | replay（mock 模式由采集线程报告为 "mock"）──
        mode = s.mode
        if mode == "live":
            self._sb_mode.setText(f"模式: 实时演示 · Mock {MOCK_UI_REFRESH_HZ}Hz刷新")
            self._sb_mode.setStyleSheet("color: #FBBF24; font-size: 12px;")
        elif mode == "replay":
            self._sb_mode.setText("模式: 回放")
            self._sb_mode.setStyleSheet("color: #60A5FA; font-size: 12px;")
        else:
            self._sb_mode.setText(f"模式: {mode}")
            self._sb_mode.setStyleSheet("color: #6B7689; font-size: 12px;")

        # 更新当前页面
        current = self.stack.currentWidget()
        if current and hasattr(current, "update_state"):
            current.update_state(state)

    def _on_event_added(self, event):
        pass  # 各页面自行监听 event_added

    def closeEvent(self, event):
        """关闭窗口时停止后台线程，确保采集/推理线程干净退出。"""
        self.service.stop_streaming()
        super().closeEvent(event)
