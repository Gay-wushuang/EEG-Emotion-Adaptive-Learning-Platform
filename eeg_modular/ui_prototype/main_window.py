"""主窗口：侧边导航 + 页面路由 + 底部状态栏。

状态栏严格使用 DashboardState 的正式字段（AGENTS.md 第6节）：
  - connector_status: offline | connecting | online
  - device_status: offline | waiting_raw | online
  - poor_signal: int | None
  - quality_level: trusted | warning | rejected
  - session_seconds: float
  - _session_active: bool（内部簿记）
  - mode: live | replay（mock 模式由采集线程报告为 "mock"）

支持两种模式：
  - mode="mock"  → MockDataService（模拟数据，无需设备）
  - mode="live"  → LiveDataService（真实 ThinkGear + Production Baseline v1）

Mock 模式下 connector_status / device_status 均为 offline，
状态栏不会将设备显示为已连接，以避免误导用户。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Literal

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QStackedWidget, QButtonGroup, QStatusBar,
    QFrame, QSizePolicy, QSpacerItem, QMessageBox, QComboBox,
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
from services.identity_store import (
    ROLE_LABELS,
    ROLE_RESEARCH,
    ROLE_STUDENT,
    ROLE_TEACHER,
    VALID_ROLES,
    role_for_user_id,
)
from services.teaching_store import StudentRuntimeRegistry, TeacherObserverService

from pages.welcome_page import WelcomePage
from pages.baseline_page import BaselinePage
from pages.dashboard_page import DashboardPage
from pages.task_page import TaskPage
from pages.history_page import HistoryPage
from pages.settings_page import SettingsPage
from pages.replay_page import ReplayPage

# Production Baseline v1 包路径（相对于项目根目录）
_PRODUCTION_PACKAGE_DIR = Path(__file__).resolve().parent.parent / "production_baseline_v1"


NAV_ITEMS = [
    ("welcome", "欢迎与设备检查", "1"),
    ("baseline", "基线采集", "2"),
    ("dashboard", "实时仪表盘", "3"),
    ("task", "任务与标记", "4"),
    ("history", "历史与报告", "5"),
    ("settings", "设置与诊断", "6"),
    ("replay", "离线回放", "7"),
]

ROLE_PAGE_KEYS = {
    ROLE_STUDENT: ("welcome", "baseline", "dashboard", "task", "history"),
    ROLE_TEACHER: ("welcome", "baseline", "dashboard", "task", "history"),
    ROLE_RESEARCH: tuple(item[0] for item in NAV_ITEMS),
}

ROLE_DEFAULT_PAGE = {
    ROLE_STUDENT: "welcome",
    ROLE_TEACHER: "welcome",
    ROLE_RESEARCH: "welcome",
}

ROLE_NAV_LABELS = {
    ROLE_STUDENT: {
        "welcome": "欢迎与设备检查", "baseline": "基线采集", "dashboard": "实时学习状态",
        "task": "我的学习任务", "history": "历史与报告",
    },
    ROLE_TEACHER: {
        "welcome": "教师工作台", "baseline": "学生基线", "dashboard": "学生实时观察",
        "task": "任务发布与观察", "history": "学生历史与报告",
    },
    ROLE_RESEARCH: {
        "welcome": "系统概览", "baseline": "本机设备诊断", "dashboard": "模型与信号诊断",
        "task": "诊断记录", "history": "数据与历史", "settings": "设置与诊断",
        "replay": "离线回放",
    },
}

ADMIN_NAV_ICONS = {
    "welcome": "◈", "baseline": "⌁", "dashboard": "◉",
    "task": "▣", "history": "▤", "settings": "⚙", "replay": "▶",
}


class MainWindow(QMainWindow):
    """主窗口。

    Args:
        mode: "mock" 使用 MockDataService（默认，无需设备）
              "live" 使用 LiveDataService（真实 ThinkGear + Production Baseline v1）
        package_dir: Production Baseline v1 包路径（仅 live 模式需要）
    """

    identity_switch_requested = Signal()

    def __init__(
        self,
        mode: Literal["mock", "live"] = "mock",
        package_dir: Optional[Path] = None,
        user_id: str = "admin_demo",
        user_name: str = "演示用户",
        role: str = ROLE_RESEARCH,
    ):
        super().__init__()
        ensure_chinese_font()
        self._mode = mode
        self._package_dir = package_dir
        self._user_id = user_id.strip() or "admin_demo"
        self._user_name = user_name.strip() or "演示用户"
        fixed_role = role_for_user_id(self._user_id)
        if role != fixed_role:
            raise ValueError("账号角色由 ID 前缀固定，不能切换使用端。")
        self._role = fixed_role
        self.setWindowTitle("智学脑机助手 - 单通道脑机接口学习状态辅助系统")
        self.resize(1920, 1080)
        self.setMinimumSize(1280, 700)

        # ── 核心状态与服务 ──
        self.state = DashboardState()
        self.runtime_registry = StudentRuntimeRegistry.shared()
        self._inject_identity_into_state()

        if self._role == ROLE_TEACHER:
            self.service = TeacherObserverService()
        elif mode == "live":
            # Live 模式：接入真实 ThinkGear + Production Baseline v1
            self._init_live_service(package_dir)
        else:
            # Mock 模式：默认，无需设备
            self.service = MockDataService(self.state)
            self._mock_service = self.service
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
        right_area.setObjectName("RightArea")
        right_layout = QVBoxLayout(right_area)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # 页面栈
        self.stack = QStackedWidget()
        self._pages = {}
        self._build_pages()
        self._connect_page_flows()
        right_layout.addWidget(self.stack, 1)

        # 状态栏
        self._status_bar = self._build_status_bar()
        right_layout.addWidget(self._status_bar)

        root_layout.addWidget(right_area, 1)

        # 连接状态更新
        self.state.state_updated.connect(self._on_state_updated)
        self.state.event_added.connect(self._on_event_added)

        # 直接实例化 MainWindow 时默认是 research，兼容原有全页面测试。
        self._apply_role_navigation()
        self._navigate_to(ROLE_DEFAULT_PAGE[self._role])

    def _init_live_service(self, package_dir: Optional[Path]) -> None:
        """初始化 Live 服务：加载 Production Baseline v1 包并启动。

        若 package_dir 为 None，使用默认路径 eeg_modular/production_baseline_v1。
        若包不存在，明确回退到 Mock 模式并将 _mode 改为 "mock"，
        绝不允许在 Mock service 上显示 Live 标签。
        """
        if package_dir is None:
            package_dir = _PRODUCTION_PACKAGE_DIR

        package_dir = Path(package_dir).resolve()

        # 检查生产包是否存在
        if not package_dir.exists():
            # 明确切换到 Mock 模式，防止侧边栏/状态栏误标 Live
            self._mode = "mock"
            self.state.mode = "mock"  # 立即同步状态，避免短暂的错误状态
            self.state._live_fallback_reason = str(package_dir)
            self.state.model_status = "FAILED"
            self.state.model_error_user = "智能分析暂不可用，请联系管理员检查系统配置。"
            self.state.model_error_detail = f"Production package not found: {package_dir}"
            self.state.quality_level = "rejected"
            self.state.quality_reasons = [
                "实时分析运行资产不完整，已进入教学演示模式。"
            ]
            self.state.emit_update()
            self.service = MockDataService(self.state)
            self._mock_service = self.service
            self.service.start_streaming()
            return

        # 延迟导入 LiveDataService（避免 Mock 模式不必要的依赖）
        from smart_learning_app.live_service import LiveDataService

        self.service = LiveDataService(self.state, package_dir)
        self._live_service = self.service
        self.state._production_package_dir = str(package_dir)
        self.service.start_streaming()

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

        self._admin_console_title = QLabel("管理员控制台")
        self._admin_console_title.setStyleSheet(
            "color: #8EC5FF; font-size: 12px; font-weight: 700; "
            "padding: 7px 9px; margin-top: 6px; background: #152338; "
            "border: 1px solid #294263; border-radius: 6px;"
        )
        self._admin_console_title.setVisible(self._role == ROLE_RESEARCH)
        layout.addWidget(self._admin_console_title)

        layout.addSpacing(12)

        self._identity_label = QLabel()
        self._identity_label.setWordWrap(True)
        self._identity_label.setStyleSheet(
            "color: #D8DFE9; font-size: 13px; font-weight: 600; padding-top: 4px;"
        )
        layout.addWidget(self._identity_label)

        self._role_label = QLabel()
        self._role_label.setObjectName("AppSubtitle")
        layout.addWidget(self._role_label)
        self._demo_badge = QLabel("教学演示 · 演示数据")
        self._demo_badge.setStyleSheet(
            "color: #FBBF24; background: rgba(251,191,36,0.10); "
            "padding: 5px 8px; border-radius: 4px; font-size: 12px;"
        )
        self._demo_badge.setVisible(self._mode == "mock")
        layout.addWidget(self._demo_badge)

        mode_row = QHBoxLayout()
        self._mode_caption = QLabel("数据模式：")
        mode_row.addWidget(self._mode_caption)
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("实时采集", "live")
        self._mode_combo.addItem("教学演示", "mock")
        self._mode_combo.setCurrentIndex(1 if self._mode == "mock" else 0)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_selected)
        mode_row.addWidget(self._mode_combo, 1)
        self._teacher_mode_source = QLabel("数据来源：跟随当前学生")
        self._teacher_mode_source.setWordWrap(True)
        self._teacher_mode_source.setVisible(False)
        mode_row.addWidget(self._teacher_mode_source, 1)
        layout.addLayout(mode_row)
        self._refresh_identity_labels()

        layout.addSpacing(10)

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

        switch_identity = QPushButton("切换账号")
        switch_identity.setMinimumHeight(36)
        switch_identity.clicked.connect(self._request_identity_switch)
        layout.addWidget(switch_identity)

        # 底部版本信息（标注当前模式）
        if self._mode == "live":
            mode_label = "v1.0.0  |  实时采集"
        else:
            mode_label = "v1.0.0  |  教学演示"
        self._version_label = QLabel(mode_label)
        self._version_label.setObjectName("AppSubtitle")
        self._version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._version_label)

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

    def _connect_page_flows(self):
        """Connect the two visible primary step buttons to page navigation."""
        self._pages["welcome"]._btn_start.clicked.connect(
            self._navigate_from_welcome
        )
        self._pages["baseline"]._btn_next.clicked.connect(
            lambda: self._navigate_to("dashboard")
        )

    def _navigate_from_welcome(self):
        self._navigate_to("dashboard" if self._role == ROLE_TEACHER else "baseline")

    def _build_status_bar(self) -> QWidget:
        bar = QFrame()
        bar.setFixedHeight(32)
        bar.setStyleSheet("background-color: #141821; border-top: 1px solid #2A3142;")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(24)

        self._sb_connector = QLabel("数据连接：--")
        self._sb_connector.setStyleSheet("color: #6B7689; font-size: 12px;")
        layout.addWidget(self._sb_connector)

        self._sb_device = QLabel("EEG 设备：--")
        self._sb_device.setStyleSheet("color: #6B7689; font-size: 12px;")
        layout.addWidget(self._sb_device)

        self._sb_signal = QLabel("接触质量：--")
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
        if key not in self._pages or key not in ROLE_PAGE_KEYS[self._role]:
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
        if self._role == ROLE_STUDENT:
            self.runtime_registry.publish(s)

        # ── Connector 状态：offline | connecting | online ──
        cs = s.connector_status
        if s.mode == "mock":
            self._sb_connector.setText("演示数据：已就绪")
            self._sb_connector.setStyleSheet("color: #FBBF24; font-size: 12px;")
        elif cs == "online":
            self._sb_connector.setText("数据连接：已连接")
            self._sb_connector.setStyleSheet("color: #4ADE80; font-size: 12px;")
        elif cs == "connecting":
            self._sb_connector.setText("数据连接：连接中")
            self._sb_connector.setStyleSheet("color: #FBBF24; font-size: 12px;")
        else:  # offline（Mock 模式即在此分支）
            self._sb_connector.setText("数据连接：未连接")
            self._sb_connector.setStyleSheet("color: #F87171; font-size: 12px;")

        # ── Device 状态：offline | waiting_raw | online ──
        ds = s.device_status
        if ds == "online":
            self._sb_device.setText("EEG 设备：在线" if s.mode != "mock" else "数据来源：教学演示")
            self._sb_device.setStyleSheet("color: #4ADE80; font-size: 12px;")
        elif ds == "waiting_raw":
            self._sb_device.setText("EEG 设备：等待信号")
            self._sb_device.setStyleSheet("color: #FBBF24; font-size: 12px;")
        else:  # offline（Mock 模式即在此分支）
            self._sb_device.setText("EEG 设备：离线")
            self._sb_device.setStyleSheet("color: #F87171; font-size: 12px;")

        # ── 信号质量：poor_signal (int | None) + quality_level ──
        poor = s.poor_signal
        ql = s.quality_level
        if poor is None:
            self._sb_signal.setText("接触质量：--")
            self._sb_signal.setStyleSheet("color: #6B7689; font-size: 12px;")
        else:
            if ql == "trusted":
                sig_color = "#4ADE80"
                sig_text = "接触质量：良好"
            elif ql == "warning":
                sig_color = "#FBBF24"
                sig_text = "接触质量：建议调整"
            else:  # rejected
                sig_color = "#F87171"
                sig_text = "接触质量：暂不可用"
            self._sb_signal.setText(sig_text)
            self._sb_signal.setStyleSheet(f"color: {sig_color}; font-size: 12px;")

        # ── 会话时间：session_seconds + _session_active ──
        if getattr(s, "session_active", s._session_active):
            secs_total = int(s.session_seconds)
            mins = secs_total // 60
            secs = secs_total % 60
            self._sb_session.setText(f"会话: {mins:02d}:{secs:02d}")
        else:
            self._sb_session.setText("会话: 未开始")

        # ── 模式：live | replay | mock ──
        mode = s.mode
        if mode == "live":
            self._sb_mode.setText("数据模式：实时采集")
            self._sb_mode.setStyleSheet("color: #4ADE80; font-size: 12px;")
        elif mode == "replay":
            self._sb_mode.setText("模式: 回放")
            self._sb_mode.setStyleSheet("color: #60A5FA; font-size: 12px;")
        elif mode == "mock":
            self._sb_mode.setText("数据模式：教学演示")
            self._sb_mode.setStyleSheet("color: #FBBF24; font-size: 12px;")
        else:
            self._sb_mode.setText(f"模式: {mode}")
            self._sb_mode.setStyleSheet("color: #6B7689; font-size: 12px;")

        # 更新当前页面
        current = self.stack.currentWidget()
        if current and hasattr(current, "update_state"):
            current.update_state(state)

    def _on_event_added(self, event):
        pass  # 各页面自行监听 event_added

    def _inject_identity_into_state(self) -> None:
        """Keep identity data on the existing state without changing its schema."""
        self.state._user_id = self._user_id
        self.state._user_name = self._user_name
        self.state._user_role = self._role
        self.state.current_role = self._role

    def _refresh_identity_labels(self) -> None:
        if hasattr(self, "_identity_label"):
            self._identity_label.setText(
                f"{self._user_name}\nID: {self._user_id}"
            )
        if hasattr(self, "_role_label"):
            self._role_label.setText(ROLE_LABELS[self._role])
        if hasattr(self, "_admin_console_title"):
            admin = self._role == ROLE_RESEARCH
            self._admin_console_title.setVisible(admin)
            self._identity_label.setStyleSheet(
                ("color: #F3F7FC; font-size: 13px; font-weight: 700; "
                 "padding: 9px 10px 4px 10px; background: #151F2E; "
                 "border-left: 2px solid #4A8DFF;")
                if admin else
                "color: #D8DFE9; font-size: 13px; font-weight: 600; padding-top: 4px;"
            )
        if hasattr(self, "_mode_combo"):
            teacher = self._role == ROLE_TEACHER
            self._mode_combo.setVisible(not teacher)
            self._mode_caption.setVisible(not teacher)
            self._teacher_mode_source.setVisible(teacher)
            self._demo_badge.setVisible(not teacher and self._mode == "mock")
            self._mode_combo.setToolTip(
                "教师端跟随当前学生的数据来源" if self._role == ROLE_TEACHER
                else "仅可在没有进行中的学习任务时切换"
            )

    def _apply_role_navigation(self) -> None:
        allowed = set(ROLE_PAGE_KEYS[self._role])
        for key, button in self._nav_buttons.items():
            button.setVisible(key in allowed)
            label = ROLE_NAV_LABELS[self._role].get(key, button.text())
            if self._role == ROLE_RESEARCH:
                button.setText(f"  {ADMIN_NAV_ICONS.get(key, '•')}   {label}")
                button.setStyleSheet(
                    "QPushButton { text-align: left; padding-left: 14px; "
                    "border-radius: 7px; color: #AEBBD0; }"
                    "QPushButton:hover { background: #192A42; color: #EAF2FF; }"
                    "QPushButton:checked { background: #203653; color: #70B7FF; "
                    "font-weight: 700; border-left: 3px solid #4A8DFF; }"
                )
            else:
                button.setText(label)
                button.setStyleSheet("")

        if hasattr(self, "stack"):
            current = self.stack.currentWidget()
            current_key = next(
                (key for key, page in self._pages.items() if page is current),
                None,
            )
            if current_key not in allowed:
                self._navigate_to(ROLE_DEFAULT_PAGE[self._role])

    def set_identity(self, user_id: str, user_name: str, role: str) -> None:
        """Apply a login result without rebuilding services or page widgets."""
        user_id = user_id.strip()
        user_name = user_name.strip()
        if not user_id or not user_name:
            raise ValueError("用户 ID 和姓名不能为空。")
        if role not in VALID_ROLES:
            raise ValueError("未知的系统角色。")
        fixed_role = role_for_user_id(user_id)
        if role != fixed_role:
            raise ValueError("账号角色由 ID 前缀固定，不能切换使用端。")

        previous_role = self._role
        self._user_id = user_id
        self._user_name = user_name
        self._role = role
        self._inject_identity_into_state()
        if previous_role != role:
            self._replace_service_for_role()
        self._refresh_identity_labels()
        self._apply_role_navigation()
        for page in self._pages.values():
            if hasattr(page, "set_role"):
                page.set_role(self._role)

        # 基线页已有兼容字段；同步输入框，防止切换身份后又写回旧用户。
        baseline = self._pages.get("baseline")
        if baseline is not None:
            if hasattr(baseline, "set_identity"):
                baseline.set_identity(user_id, user_name)
            else:
                if hasattr(baseline, "_input_uid"):
                    baseline._input_uid.setText(user_id)
                if hasattr(baseline, "_input_name"):
                    baseline._input_name.setText(user_name)

        self._navigate_to(ROLE_DEFAULT_PAGE[role])
        self.state.emit_update()

    def _replace_service_for_role(self) -> None:
        """Switch acquisition ownership when an account role changes."""
        old_service = getattr(self, "service", None)
        history_snapshot = list(getattr(self.state, "_history_sessions", []))
        if old_service is not None:
            old_service.stop_streaming()
        if self._role == ROLE_TEACHER:
            self.service = TeacherObserverService()
        elif self._mode == "live":
            self._init_live_service(self._package_dir)
        else:
            self.service = MockDataService(self.state)
            self.service.start_streaming()
        self.state._history_sessions = history_snapshot
        for page in getattr(self, "_pages", {}).values():
            if hasattr(page, "service"):
                page.service = self.service

    def _request_identity_switch(self) -> None:
        if getattr(self.state, "session_active", self.state._session_active):
            QMessageBox.information(
                self,
                "会话正在进行",
                "请先结束并保存当前会话，再切换身份或使用端。",
            )
            return
        self.identity_switch_requested.emit()

    def _on_mode_selected(self, index: int) -> None:
        requested = self._mode_combo.itemData(index)
        if requested and requested != self._mode and not self.switch_data_mode(requested):
            self._mode_combo.blockSignals(True)
            self._mode_combo.setCurrentIndex(1 if self._mode == "mock" else 0)
            self._mode_combo.blockSignals(False)

    def switch_data_mode(self, mode: str) -> bool:
        """Switch acquisition source only while no Session/Task is active."""
        if mode not in {"live", "mock"} or mode == self._mode:
            return mode == self._mode
        if bool(getattr(self.state, "task_running", False)) or bool(
            getattr(self.state, "session_active", self.state._session_active)
        ):
            QMessageBox.information(
                self, "暂不能切换数据模式",
                "当前学习任务正在进行，请先结束任务后再切换数据模式。",
            )
            return False
        old_service = self.service
        if self._mode == "live" and hasattr(old_service, "suspend_streaming"):
            old_service.suspend_streaming()
            self._live_service = old_service
        else:
            old_service.stop_streaming()
            if self._mode == "mock":
                self._mock_service = old_service
        self._mode = mode
        self.state._eeg_raw_buffer.clear()
        self.state._attention_history.clear()
        self.state._meditation_history.clear()
        self.state.clear_interpretation()
        self.state.attention = None
        self.state.meditation = None
        self.state.poor_signal = None
        self.state.warmup_progress = 0.0
        if self._role != ROLE_TEACHER and mode == "mock":
            self.service = getattr(self, "_mock_service", None) or MockDataService(self.state)
            self._mock_service = self.service
            self.service.start_streaming()
        elif self._role != ROLE_TEACHER:
            cached_live = getattr(self, "_live_service", None)
            if cached_live is not None:
                self.service = cached_live
                self.service.resume_streaming()
            else:
                self._init_live_service(self._package_dir)
        for page in self._pages.values():
            if hasattr(page, "service"):
                page.service = self.service
        self._demo_badge.setVisible(self._mode == "mock")
        self._version_label.setText(
            "v1.0.0  |  教学演示" if self._mode == "mock" else "v1.0.0  |  实时采集"
        )
        self._mode_combo.blockSignals(True)
        self._mode_combo.setCurrentIndex(1 if self._mode == "mock" else 0)
        self._mode_combo.blockSignals(False)
        # Round 4A-2：历史页可见时，模式切换后立即同步筛选/列表/说明文字。
        history_page = self._pages.get("history")
        if history_page is not None and hasattr(history_page, "on_data_mode_changed"):
            history_page.on_data_mode_changed(mode)
        self.state.emit_update()
        return True

    def closeEvent(self, event):
        """关闭窗口时停止后台线程，确保采集/推理线程干净退出。"""
        if self._role == ROLE_STUDENT:
            self.runtime_registry.mark_offline(self._user_id)
        self.service.stop_streaming()
        cached_live = getattr(self, "_live_service", None)
        if cached_live is not None and cached_live is not self.service:
            cached_live.stop_streaming()
        super().closeEvent(event)
