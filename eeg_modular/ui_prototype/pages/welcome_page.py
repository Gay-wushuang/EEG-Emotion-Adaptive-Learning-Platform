"""页面1：欢迎与设备检查页。"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
    QPushButton, QFrame, QSizePolicy, QComboBox,
)

from pages.base_page import BasePage
from widgets.card import Card
from widgets.status_indicator import StatusIndicator
from services.dashboard_state import (
    MAX_POOR_SIGNAL, WARMUP_SECONDS,
    MOCK_UI_REFRESH_HZ, DEVICE_TARGET_SAMPLE_HZ,
)
from services.identity_store import IdentityStore
from services.teaching_store import (
    StudentRuntimeRegistry, TeacherSelectionContext, TeacherStudentStore,
)


class WelcomePage(BasePage):
    def __init__(self, state, service, *, identity_store=None, binding_store=None,
                 runtime_registry=None, selection_context=None):
        self.state = state
        self.service = service
        self.identity_store = identity_store or IdentityStore()
        self.binding_store = binding_store or TeacherStudentStore(identity_store=self.identity_store)
        self.runtime_registry = runtime_registry or StudentRuntimeRegistry.shared()
        self.selection_context = selection_context or TeacherSelectionContext.shared()
        super().__init__(
            "欢迎与设备检查",
            "在开始学习状态监测前，请确认设备连接与信号状态正常。"
        )
        self._build_ui()
        self._teacher_timer = QTimer(self)
        self._teacher_timer.setInterval(500)
        self._teacher_timer.timeout.connect(self._refresh_teacher_workspace)
        self.set_role(getattr(state, "current_role", "research"))

    def _build_ui(self):
        # ── 顶部欢迎区 ──
        welcome_card = Card("系统简介")
        self._welcome_card = welcome_card
        intro = QLabel(
            "智学脑机助手通过 MindWave 单通道脑电设备，持续观察学习过程中的状态变化。\n"
            "系统融合脑电时域与频域特征，输出积极、中性、负性三类状态概率，并独立评估信号可信度，\n"
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
        self._ind_signal = StatusIndicator("接触质量")
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
            "支持数据：原始脑电 / 专注度 / 放松度"
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

        self._teacher_card = Card("教师工作台")
        teacher_layout = QVBoxLayout()
        self._teacher_identity = QLabel()
        self._teacher_students = QLabel()
        self._teacher_selected = QComboBox()
        self._teacher_selected.currentIndexChanged.connect(self._teacher_selection_changed)
        self._teacher_detail = QLabel("请选择学生")
        self._teacher_detail.setWordWrap(True)
        teacher_layout.addWidget(self._teacher_identity)
        teacher_layout.addWidget(self._teacher_students)
        teacher_layout.addWidget(QLabel("当前观察学生："))
        teacher_layout.addWidget(self._teacher_selected)
        teacher_layout.addWidget(self._teacher_detail)
        shortcuts = QHBoxLayout()
        for text, key in (("查看实时状态", "dashboard"), ("发布任务", "task"),
                          ("查看历史", "history")):
            button = QPushButton(text)
            button.clicked.connect(lambda checked=False, target=key: self._navigate(target))
            shortcuts.addWidget(button)
        teacher_layout.addLayout(shortcuts)
        self._teacher_card.add_widget(self._wrap_layout(teacher_layout))
        self._teacher_card.setVisible(False)
        self.content_layout.addWidget(self._teacher_card)

        self._flow_card = Card("我的学习流程")
        self._flow_steps = QLabel()
        self._flow_steps.setWordWrap(True)
        self._flow_steps.setStyleSheet("color: #C5CDD9; font-size: 14px; line-height: 1.7;")
        self._flow_action = QPushButton("查看下一步")
        self._flow_action.setObjectName("PrimaryButton")
        self._flow_target = "baseline"
        self._flow_action.clicked.connect(lambda: self._navigate(self._flow_target))
        self._flow_card.add_widget(self._flow_steps)
        self._flow_card.add_widget(self._flow_action)
        self.content_layout.addWidget(self._flow_card)

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

    @staticmethod
    def _wrap_layout(layout):
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _navigate(self, key):
        window = self.window()
        if hasattr(window, "_navigate_to"):
            window._navigate_to(key)

    def set_role(self, role: str):
        super().set_role(role)
        if not hasattr(self, "_teacher_card"):
            return
        teacher = self._role == "teacher"
        student = self._role == "student"
        for widget in (self._welcome_card, self._card_connector, self._card_device,
                       self._card_signal, self._card_sample, self._card_warmup,
                       self._btn_start, self._next_hint):
            widget.setVisible(not teacher)
        self._teacher_card.setVisible(teacher)
        self._flow_card.setVisible(student)
        if teacher:
            self._title_label.setText("教师工作台")
            self._desc_label.setText("查看已绑定学生的实时学习状态并进入教师工作流程。")
            self._teacher_timer.start()
            self._refresh_teacher_workspace()
        else:
            self._title_label.setText("欢迎与设备检查")
            self._desc_label.setText("在开始学习状态监测前，请确认设备连接与信号状态正常。")
            self._teacher_timer.stop()

    def _teacher_selection_changed(self, index):
        if self._role == "teacher":
            self.selection_context.select(self._teacher_selected.currentData() or "")
            self._refresh_teacher_workspace()

    def _refresh_teacher_workspace(self):
        if self._role != "teacher":
            return
        teacher_id = getattr(self.state, "_user_id", "")
        self.selection_context.set_teacher(teacher_id)
        students = self.binding_store.students_for(teacher_id)
        current = self.selection_context.selected_student_id
        existing = [self._teacher_selected.itemData(i)
                    for i in range(self._teacher_selected.count())]
        if students != existing:
            self._teacher_selected.blockSignals(True)
            self._teacher_selected.clear()
            for student_id in students:
                profile = self.identity_store.get_profile(student_id) or {}
                self._teacher_selected.addItem(
                    f"{profile.get('name', '')} {student_id}".strip(), student_id
                )
            index = self._teacher_selected.findData(current)
            self._teacher_selected.setCurrentIndex(index if index >= 0 else (0 if students else -1))
            self._teacher_selected.blockSignals(False)
            self.selection_context.select(self._teacher_selected.currentData() or "")
        else:
            index = self._teacher_selected.findData(current)
            if index >= 0 and index != self._teacher_selected.currentIndex():
                self._teacher_selected.blockSignals(True)
                self._teacher_selected.setCurrentIndex(index)
                self._teacher_selected.blockSignals(False)
        snapshots = [self.runtime_registry.get(student_id) for student_id in students]
        online = sum(bool(item and not item.get("stale")) for item in snapshots)
        learning = sum(bool(item and item.get("task_id") and not item.get("stale")) for item in snapshots)
        attention = sum(bool(item and not item.get("stale") and (
            item.get("quality_level") == "rejected"
            or item.get("stable_state") == "negative"
            or (item.get("attention") is not None and item.get("attention") < 45)
        )) for item in snapshots)
        self._teacher_identity.setText(f"当前教师：{teacher_id}")
        self._teacher_students.setText(
            f"我的学生：{len(students)} 人 · 当前在线：{online} 人 · "
            f"正在学习：{learning} 人 · 建议关注：{attention} 人"
        )
        snapshot = self.runtime_registry.get(self.selection_context.selected_student_id)
        if not snapshot or snapshot.get("stale"):
            self._teacher_detail.setText("当前观察学生：暂无实时数据")
            return
        elapsed = int(snapshot.get("elapsed_seconds") or 0)
        quality = {"trusted": "接触良好", "warning": "建议调整佩戴",
                   "rejected": "当前信号不可解释"}.get(snapshot.get("quality_level"), "等待信号")
        learning_state = {"positive": "积极", "neutral": "稳定",
                          "negative": "学习负荷偏高"}.get(snapshot.get("stable_state"), "暂无")
        mode_text = " · 教学演示 / 演示数据" if snapshot.get("data_mode") == "demo" else ""
        self._teacher_detail.setText(
            f"{snapshot.get('student_name')} {snapshot.get('student_id')} · "
            f"{'在线' if snapshot.get('online') else '离线'}{mode_text}\n"
            f"任务：{snapshot.get('task_name') or '无'} · 有效学习 {elapsed // 60:02d}:{elapsed % 60:02d}\n"
            f"接触质量：{quality} · 学习状态：{learning_state}\n"
            f"智能建议：{snapshot.get('advice') or '暂无建议'}"
        )

    def _refresh_student_flow(self, state):
        if self._role != "student":
            return
        device_ready = state.mode == "mock" or state.device_status == "online"
        baseline_done = getattr(state, "baseline_status", "IDLE") == "COMPLETED"
        task_active = bool(getattr(state, "task_running", False))
        has_history = bool(getattr(state, "_history_sessions", []))
        marks = [device_ready, baseline_done, task_active, task_active, has_history]
        labels = ["设备与数据准备", "完成个人基线", "开始学习任务", "查看实时状态", "查看学习报告"]
        self._flow_steps.setText("\n".join(
            f"{'✓' if done else '○'}  {index}. {label}"
            for index, (label, done) in enumerate(zip(labels, marks), 1)
        ))
        if not device_ready:
            text, target = "请先连接 EEG 设备", "welcome"
        elif not baseline_done:
            text, target = "下一步：完成基线采集", "baseline"
        elif task_active:
            text, target = "继续：查看实时学习状态", "dashboard"
        elif has_history:
            text, target = "查看最近学习报告", "history"
        else:
            text, target = "下一步：查看我的学习任务", "task"
        self._flow_action.setText(text)
        self._flow_action.setEnabled(target != "welcome")
        self._flow_target = target

    def update_state(self, state):
        if self._role == "teacher":
            self._refresh_teacher_workspace()
            return
        self._refresh_student_flow(state)
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
            self._ind_signal.set_state(StatusIndicator.LEVEL_GOOD, "接触良好")
        elif poor < 200:
            self._ind_signal.set_state(StatusIndicator.LEVEL_WARN, "请调整电极接触")
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
        if state.mode == "mock":
            self._card_warmup.set_title("教学演示状态")
            self._warmup_progress_label.setText("演示分析：已就绪（不使用生产模型预热）")
            self._warmup_progress_label.setStyleSheet("font-size: 14px; color: #FBBF24;")
        else:
            self._card_warmup.set_title("智能分析准备")
        if state.mode != "mock" and state.warmup_complete:
            self._warmup_progress_label.setText("预热进度：100% ✓ 已完成")
            self._warmup_progress_label.setStyleSheet("font-size: 14px; color: #4ADE80;")
        elif state.mode != "mock":
            self._warmup_progress_label.setText(f"预热进度：{pct:.0f}%")
        if state.connector_status == "online" and state.device_status == "online":
            self._next_hint.setText("设备与数据流已就绪，可以进入基线采集。")
            self._next_hint.setStyleSheet("color: #4ADE80; font-size: 12px;")
        else:
            self._next_hint.setText("尚未检测到完整设备数据；可进入基线页查看，但暂不能开始采集。")
            self._next_hint.setStyleSheet("color: #FBBF24; font-size: 12px;")
