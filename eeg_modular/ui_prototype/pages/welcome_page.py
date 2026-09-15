"""页面1：欢迎与设备检查页。"""

from __future__ import annotations

from datetime import datetime

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
        # Round 4C：其他页面切换观察学生后，教师工作台立即同步。
        self.selection_context.add_listener(self, "_on_selection_changed")

    def _on_selection_changed(self, student_id: str):
        if self._role == "teacher":
            self._refresh_teacher_workspace()

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

        self._teacher_home = self._build_teacher_home()
        # 保留旧属性名，兼容现有角色可见性检查；实际教师首页由专属容器承载。
        self._teacher_card = self._teacher_home
        self._teacher_home.setVisible(False)
        self.content_layout.addWidget(self._teacher_home, 1)

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

        self._student_home = self._build_student_home()
        self._student_home.setVisible(False)
        self.content_layout.addWidget(self._student_home, 1)
        self._admin_home = self._build_admin_home()
        self._admin_home.setVisible(False)
        self.content_layout.addWidget(self._admin_home, 1)
        self.content_layout.addStretch()

    def _build_admin_home(self) -> QWidget:
        """Administrator system overview using existing read-only state sources."""
        home = QWidget()
        home.setObjectName("AdminSystemOverview")
        root = QVBoxLayout(home)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        section = QLabel("平台运行概览")
        section.setStyleSheet("color: #E8EDF3; font-size: 17px; font-weight: 700;")
        root.addWidget(section)
        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        self._admin_metrics = {}
        for key, title, hint, color in (
            ("devices", "在线设备", "当前检测到的 EEG 设备", "#4ADE80"),
            ("sessions", "今日 Session", "今日已保存学习会话", "#4FC3F7"),
            ("students", "学生账号", "当前本地学生账号", "#A78BFA"),
            ("model", "AI 模型状态", "生产分析服务状态", "#FBBF24"),
        ):
            card = Card(title)
            value = QLabel("暂无数据")
            value.setStyleSheet(f"color: {color}; font-size: 28px; font-weight: 700;")
            helper = QLabel(hint)
            helper.setWordWrap(True)
            helper.setStyleSheet("color: #8491A5; font-size: 11px;")
            card.add_widget(value)
            card.add_widget(helper)
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            metrics.addWidget(card, 1)
            self._admin_metrics[key] = value
        root.addLayout(metrics)

        main = QHBoxLayout()
        main.setSpacing(12)
        health = Card("系统运行状态")
        health_body = QVBoxLayout()
        health_body.setContentsMargins(0, 0, 0, 0)
        health_body.setSpacing(9)
        health_note = QLabel("核心服务状态概览")
        health_note.setStyleSheet("color: #8491A5; font-size: 11px;")
        health_body.addWidget(health_note)
        self._admin_health = {}
        for key, title in (
            ("connector", "数据连接服务"), ("device", "MindWave 设备"),
            ("signal", "当前信号质量"), ("analysis", "AI 分析服务"),
        ):
            row = QHBoxLayout()
            label = QLabel(title)
            label.setStyleSheet("color: #AAB6C8; font-size: 13px;")
            value = QLabel("暂无数据")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value.setStyleSheet("color: #94A3B8; font-size: 13px; font-weight: 600;")
            row.addWidget(label)
            row.addStretch()
            row.addWidget(value)
            health_body.addLayout(row)
            self._admin_health[key] = value
        health_body.addStretch()
        health.add_widget(self._wrap_layout(health_body))
        main.addWidget(health, 3)

        access = Card("管理入口")
        access_body = QVBoxLayout()
        access_body.setContentsMargins(0, 0, 0, 0)
        access_body.setSpacing(8)
        access_note = QLabel("快速进入设备、模型、数据与离线诊断工具。")
        access_note.setWordWrap(True)
        access_note.setStyleSheet("color: #8491A5; font-size: 11px;")
        access_body.addWidget(access_note)
        for text, target in (
            ("本机设备诊断", "baseline"), ("模型与信号诊断", "dashboard"),
            ("数据与历史", "history"), ("离线数据回放", "replay"),
        ):
            button = QPushButton(text)
            button.setMinimumHeight(36)
            button.clicked.connect(lambda checked=False, key=target: self._navigate(key))
            access_body.addWidget(button)
        access_body.addStretch()
        access.add_widget(self._wrap_layout(access_body))
        main.addWidget(access, 2)
        root.addLayout(main, 1)

        device_section = QLabel("设备与数据状态")
        device_section.setStyleSheet("color: #E8EDF3; font-size: 16px; font-weight: 700;")
        root.addWidget(device_section)
        status_card = Card("")
        status_row = QHBoxLayout()
        status_row.setContentsMargins(2, 2, 2, 2)
        status_row.setSpacing(18)
        self._admin_status_labels = {}
        for key, title in (
            ("source", "数据来源"), ("sample", "采样状态"),
            ("contact", "接触质量"), ("session", "当前 Session"),
        ):
            column = QVBoxLayout()
            caption = QLabel(title)
            caption.setStyleSheet("color: #8491A5; font-size: 11px;")
            value = QLabel("暂无数据")
            value.setStyleSheet("color: #DDE6F2; font-size: 13px; font-weight: 600;")
            column.addWidget(caption)
            column.addWidget(value)
            status_row.addLayout(column, 1)
            self._admin_status_labels[key] = value
        status_card.add_widget(self._wrap_layout(status_row))
        root.addWidget(status_card)
        return home

    def _build_teacher_home(self) -> QWidget:
        """按教师工作台母版构建纯展示布局，数据由现有 store/registry 刷新。"""
        home = QWidget()
        page = QVBoxLayout(home)
        page.setContentsMargins(0, 0, 0, 0)
        page.setSpacing(10)

        overview_title = QLabel("教学概览")
        overview_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #E8EDF3;")
        page.addWidget(overview_title)

        self._teacher_overview_row = QWidget()
        overview = QHBoxLayout(self._teacher_overview_row)
        overview.setContentsMargins(0, 0, 0, 0)
        overview.setSpacing(12)
        self._teacher_metrics = {}
        self._teacher_metric_cards = {}
        metric_specs = (
            ("students", "我的学生", "已绑定学生总数", "#E8EDF3"),
            ("online", "当前在线", "可进行实时观察", "#4ADE80"),
            ("learning", "正在学习", "当前进行中的学习任务", "#4FC3F7"),
            ("attention", "建议关注", "存在需要跟进的学习提示", "#FBBF24"),
        )
        for key, title, helper, color in metric_specs:
            card = Card(title)
            metric = QLabel("暂无数据")
            metric.setStyleSheet(f"font-size: 28px; font-weight: 700; color: {color};")
            hint = QLabel(helper)
            hint.setStyleSheet("color: #8491A5; font-size: 11px;")
            hint.setWordWrap(True)
            card.add_widget(metric)
            card.add_widget(hint)
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            overview.addWidget(card, 1)
            self._teacher_metrics[key] = metric
            self._teacher_metric_cards[key] = card
        page.addWidget(self._teacher_overview_row)

        self._teacher_middle = QWidget()
        main_row = QHBoxLayout(self._teacher_middle)
        main_row.setContentsMargins(0, 0, 0, 0)
        main_row.setSpacing(12)

        observation = Card("当前观察学生")
        self._teacher_observation_card = observation
        # 兼容既有只读测试接口；不加入布局，页面使用下方语义化字段展示。
        self._teacher_detail = QLabel()
        observation_body = QVBoxLayout()
        observation_body.setContentsMargins(0, 0, 0, 0)
        observation_body.setSpacing(7)
        observation_help = QLabel(
            "聚合教师需要的学习信息，不展示模型内部概率与原始技术指标。"
        )
        observation_help.setStyleSheet("color: #8491A5; font-size: 11px;")
        observation_body.addWidget(observation_help)

        selector_row = QHBoxLayout()
        selector_row.setSpacing(10)
        selector_row.addWidget(QLabel("当前学生"))
        self._teacher_selected = QComboBox()
        self._teacher_selected.setMinimumHeight(38)
        self._teacher_selected.currentIndexChanged.connect(self._teacher_selection_changed)
        selector_row.addWidget(self._teacher_selected, 1)
        self._teacher_online_badge = QLabel("未选择")
        self._teacher_online_badge.setAlignment(Qt.AlignCenter)
        self._teacher_online_badge.setMinimumWidth(72)
        self._teacher_online_badge.setStyleSheet(
            "color: #94A3B8; background: #222B3A; border-radius: 6px; padding: 7px 10px;"
        )
        selector_row.addWidget(self._teacher_online_badge)
        observation_body.addLayout(selector_row)

        detail_grid = QGridLayout()
        detail_grid.setHorizontalSpacing(24)
        detail_grid.setVerticalSpacing(4)
        self._teacher_task = QLabel("暂无进行中的任务")
        self._teacher_elapsed = QLabel("有效学习时长：暂无数据")
        self._teacher_learning_state = QLabel("暂无稳定状态")
        self._teacher_quality = QLabel("接触质量：暂无数据")
        for label in (self._teacher_task, self._teacher_learning_state):
            label.setStyleSheet("font-size: 17px; font-weight: 600; color: #E8EDF3;")
        for label in (self._teacher_elapsed, self._teacher_quality):
            label.setStyleSheet("color: #94A3B8; font-size: 12px;")
        detail_grid.addWidget(QLabel("当前任务"), 0, 0)
        detail_grid.addWidget(QLabel("学习状态"), 0, 1)
        detail_grid.addWidget(self._teacher_task, 1, 0)
        detail_grid.addWidget(self._teacher_learning_state, 1, 1)
        detail_grid.addWidget(self._teacher_elapsed, 2, 0)
        detail_grid.addWidget(self._teacher_quality, 2, 1)
        detail_grid.setColumnStretch(0, 1)
        detail_grid.setColumnStretch(1, 1)
        observation_body.addLayout(detail_grid)

        observation_body.addWidget(QLabel("智能建议"))
        self._teacher_advice = QLabel("暂无建议")
        self._teacher_advice.setWordWrap(True)
        self._teacher_advice.setStyleSheet("color: #C5CDD9; font-size: 12px;")
        observation_body.addWidget(self._teacher_advice)
        shortcuts = QHBoxLayout()
        shortcuts.setSpacing(10)
        for index, (text, key) in enumerate((
            ("查看实时状态", "dashboard"),
            ("发布任务", "task"),
            ("查看学生历史与报告", "history"),
        )):
            button = QPushButton(text)
            if index == 0:
                button.setObjectName("PrimaryButton")
            button.setMinimumHeight(36)
            button.clicked.connect(lambda checked=False, target=key: self._navigate(target))
            shortcuts.addWidget(button, 1)
        observation_body.addLayout(shortcuts)
        observation.add_widget(self._wrap_layout(observation_body))
        main_row.addWidget(observation, 3)

        right_column = QVBoxLayout()
        right_column.setSpacing(10)
        attention_card = Card("需要关注")
        self._teacher_attention_card = attention_card
        attention_hint = QLabel("仅在持续趋势或教师事件需要跟进时出现。")
        attention_hint.setStyleSheet("color: #8491A5; font-size: 11px;")
        self._teacher_attention_text = QLabel("当前暂无需要特别关注的学生")
        self._teacher_attention_text.setWordWrap(True)
        self._teacher_attention_text.setStyleSheet("color: #4ADE80; font-size: 13px; font-weight: 600;")
        self._teacher_attention_helper = QLabel("继续观察学习过程，必要时记录教师观察事件。")
        self._teacher_attention_helper.setWordWrap(True)
        self._teacher_attention_helper.setStyleSheet("color: #94A3B8; font-size: 11px;")
        attention_card.add_widget(attention_hint)
        attention_card.add_widget(self._teacher_attention_text)
        attention_card.add_widget(self._teacher_attention_helper)
        right_column.addWidget(attention_card, 1)

        activity_card = Card("最近教学动态")
        self._teacher_activity_card = activity_card
        self._teacher_activity = QLabel("暂无最近教学动态")
        self._teacher_activity.setWordWrap(True)
        self._teacher_activity.setStyleSheet("color: #94A3B8; font-size: 12px;")
        activity_card.add_widget(self._teacher_activity)
        history_button = QPushButton("查看全部历史")
        history_button.clicked.connect(lambda: self._navigate("history"))
        activity_card.add_widget(history_button)
        right_column.addWidget(activity_card, 1)
        main_row.addLayout(right_column, 2)
        self._teacher_middle.setMaximumHeight(365)
        page.addWidget(self._teacher_middle, 1)

        workflow_title = QLabel("教师工作流程")
        workflow_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #E8EDF3;")
        page.addWidget(workflow_title)
        workflow_card = Card("")
        self._teacher_workflow_card = workflow_card
        workflow = QHBoxLayout()
        workflow.setContentsMargins(4, 0, 4, 0)
        workflow.setSpacing(8)
        for number, title, helper in (
            ("1", "选择学生", "确定当前观察对象"),
            ("2", "查看实时", "观察学习状态与趋势"),
            ("3", "发布任务 / 标记", "布置任务并记录观察"),
            ("4", "复盘报告", "查看历史与近期趋势"),
        ):
            step = QVBoxLayout()
            step.setSpacing(3)
            number_label = QLabel(number)
            number_label.setAlignment(Qt.AlignCenter)
            number_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #4FC3F7;")
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("font-size: 13px; font-weight: 600; color: #E8EDF3;")
            helper_label = QLabel(helper)
            helper_label.setAlignment(Qt.AlignCenter)
            helper_label.setStyleSheet("color: #8491A5; font-size: 10px;")
            step.addWidget(number_label)
            step.addWidget(title_label)
            step.addWidget(helper_label)
            workflow.addLayout(step, 1)
        workflow_card.add_widget(self._wrap_layout(workflow))
        page.addWidget(workflow_card)
        return home

    def _build_student_home(self):
        """学生专用首页；仅重排现有状态与导航，不改变业务判断。"""
        home = QWidget()
        home.setObjectName("StudentHome")
        layout = QVBoxLayout(home)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        section = QLabel("状态总览")
        section.setObjectName("StudentSectionTitle")
        layout.addWidget(section)

        cards = QHBoxLayout()
        cards.setSpacing(14)
        self._student_status = {}
        status_specs = (
            ("connector", "THINKGEAR CONNECTOR", "本地通信服务状态"),
            ("device", "MINDWAVE 设备", "设备连接与采样状态"),
            ("signal", "接触质量", "当前电极接触状态"),
            ("analysis", "分析准备", "学习状态分析准备情况"),
        )
        for key, title, hint in status_specs:
            card = QFrame()
            card.setObjectName("StudentStatusCard")
            card.setFixedHeight(142)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(20, 15, 20, 14)
            card_layout.setSpacing(6)
            caption = QLabel(title)
            caption.setObjectName("StudentCaption")
            value = QLabel("暂无数据")
            value.setObjectName("StudentStatusValue")
            indicator = StatusIndicator("")
            indicator.set_state(StatusIndicator.LEVEL_NEUTRAL, "暂无数据")
            helper = QLabel(hint)
            helper.setObjectName("StudentHelper")
            helper.setWordWrap(True)
            card_layout.addWidget(caption)
            card_layout.addWidget(value)
            card_layout.addWidget(indicator)
            card_layout.addWidget(helper)
            self._student_status[key] = {
                "card": card, "value": value, "indicator": indicator, "helper": helper,
            }
            cards.addWidget(card, 1)
        layout.addLayout(cards)

        middle_title = QLabel("学习准备")
        middle_title.setObjectName("StudentSectionTitle")
        layout.addWidget(middle_title)
        middle = QHBoxLayout()
        middle.setSpacing(14)

        readiness = QFrame()
        readiness.setObjectName("StudentPanel")
        readiness.setFixedHeight(300)
        ready_layout = QVBoxLayout(readiness)
        ready_layout.setContentsMargins(22, 18, 22, 18)
        ready_layout.setSpacing(10)
        ready_layout.addWidget(self._student_panel_title("当前准备状态"))
        self._student_ready_value = QLabel("暂不可开始学习")
        self._student_ready_value.setObjectName("StudentReadyValue")
        ready_layout.addWidget(self._student_ready_value)
        self._student_ready_badge = QLabel("等待设备")
        self._student_ready_badge.setObjectName("StudentBadgeNeutral")
        self._student_ready_badge.setAlignment(Qt.AlignCenter)
        self._student_ready_badge.setFixedWidth(112)
        ready_layout.addWidget(self._student_ready_badge)
        divider = QFrame()
        divider.setObjectName("StudentDivider")
        divider.setFixedHeight(1)
        ready_layout.addWidget(divider)
        self._student_check_values = {}
        for key, label in (("device", "设备检查"), ("baseline", "个人静息基线"),
                           ("analysis", "状态分析")):
            row = QHBoxLayout()
            name = QLabel(label)
            name.setObjectName("StudentRowName")
            value = QLabel("暂无数据")
            value.setObjectName("StudentRowValue")
            dot = QLabel("●")
            dot.setObjectName("StudentDotNeutral")
            dot.setFixedWidth(18)
            row.addWidget(name)
            row.addWidget(value, 1)
            row.addWidget(dot)
            ready_layout.addLayout(row)
            self._student_check_values[key] = (value, dot)
        ready_layout.addStretch()
        middle.addWidget(readiness, 5)

        actions = QFrame()
        actions.setObjectName("StudentPanel")
        actions.setFixedHeight(300)
        action_layout = QVBoxLayout(actions)
        action_layout.setContentsMargins(22, 18, 22, 18)
        action_layout.setSpacing(12)
        action_layout.addWidget(self._student_panel_title("下一步"))
        explanation = QLabel("你可以先建立个人基线，或按当前学习准备状态进入下一步。")
        explanation.setObjectName("StudentBody")
        explanation.setWordWrap(True)
        action_layout.addWidget(explanation)
        action_layout.addStretch()
        baseline_button = QPushButton("进入基线采集")
        baseline_button.setObjectName("StudentSecondaryButton")
        baseline_button.setFixedHeight(48)
        baseline_button.clicked.connect(self._btn_start.click)
        action_layout.addWidget(baseline_button)
        self._student_flow_action = QPushButton("查看下一步")
        self._student_flow_action.setObjectName("PrimaryButton")
        self._student_flow_action.setFixedHeight(48)
        self._student_flow_action.clicked.connect(lambda: self._navigate(self._flow_target))
        action_layout.addWidget(self._student_flow_action)
        self._student_next_hint = QLabel("设备状态更新后，将在这里给出可用入口。")
        self._student_next_hint.setObjectName("StudentHelper")
        self._student_next_hint.setWordWrap(True)
        action_layout.addWidget(self._student_next_hint)
        middle.addWidget(actions, 3)
        layout.addLayout(middle)

        flow_title = QLabel("我的学习流程")
        flow_title.setObjectName("StudentSectionTitle")
        layout.addWidget(flow_title)
        flow = QFrame()
        flow.setObjectName("StudentFlow")
        flow.setFixedHeight(156)
        flow_layout = QHBoxLayout(flow)
        flow_layout.setContentsMargins(22, 18, 22, 18)
        flow_layout.setSpacing(10)
        flow_specs = (
            ("1", "设备检查", "确认连接与信号"),
            ("2", "个人基线", "按当前规则完成"),
            ("3", "开始任务", "进入本人学习任务"),
            ("4", "实时观察", "查看状态与建议"),
            ("5", "查看报告", "回顾本次学习"),
        )
        self._student_flow_markers = []
        for index, (number, title, detail) in enumerate(flow_specs):
            step = QVBoxLayout()
            step.setSpacing(5)
            marker = QLabel(number)
            marker.setObjectName("StudentStepPending")
            marker.setAlignment(Qt.AlignCenter)
            marker.setFixedSize(38, 38)
            title_label = QLabel(title)
            title_label.setObjectName("StudentStepTitle")
            detail_label = QLabel(detail)
            detail_label.setObjectName("StudentHelper")
            detail_label.setWordWrap(True)
            step.addWidget(marker, 0, Qt.AlignHCenter)
            step.addWidget(title_label, 0, Qt.AlignHCenter)
            step.addWidget(detail_label, 0, Qt.AlignHCenter)
            self._student_flow_markers.append(marker)
            flow_layout.addLayout(step, 1)
            if index < len(flow_specs) - 1:
                line = QFrame()
                line.setObjectName("StudentFlowLine")
                line.setFixedHeight(2)
                flow_layout.addWidget(line, 0, Qt.AlignVCenter)
        layout.addWidget(flow)

        home.setStyleSheet("""
            QWidget#StudentHome { background: transparent; }
            QLabel#StudentSectionTitle, QLabel#StudentPanelTitle {
                color: #E8EDF3; font-size: 17px; font-weight: 700;
            }
            QFrame#StudentStatusCard, QFrame#StudentPanel, QFrame#StudentFlow {
                background-color: #192130; border: 1px solid #293345; border-radius: 13px;
            }
            QLabel#StudentCaption { color: #8491A5; font-size: 12px; font-weight: 600; }
            QLabel#StudentStatusValue { color: #E8EDF3; font-size: 20px; font-weight: 700; }
            QLabel#StudentHelper { color: #8491A5; font-size: 12px; }
            QLabel#StudentBody { color: #C5CDD9; font-size: 13px; }
            QLabel#StudentReadyValue { color: #E8EDF3; font-size: 26px; font-weight: 700; }
            QLabel#StudentBadgeGood { color: #4ADE80; background: #173528; padding: 5px 9px; border-radius: 5px; }
            QLabel#StudentBadgeNeutral { color: #FBBF24; background: #3A301D; padding: 5px 9px; border-radius: 5px; }
            QFrame#StudentDivider, QFrame#StudentFlowLine { background-color: #2A3142; border: none; }
            QLabel#StudentRowName { color: #8491A5; font-size: 13px; min-width: 150px; }
            QLabel#StudentRowValue { color: #E8EDF3; font-size: 13px; }
            QLabel#StudentDotGood { color: #4ADE80; }
            QLabel#StudentDotWarn { color: #FBBF24; }
            QLabel#StudentDotNeutral { color: #6B7689; }
            QPushButton#StudentSecondaryButton { background: #222C3D; border: 1px solid #344056; }
            QPushButton#StudentSecondaryButton:hover { background: #2A364A; border-color: #52627B; }
            QLabel#StudentStepPending, QLabel#StudentStepDone {
                border-radius: 19px; color: #E8EDF3; font-size: 15px; font-weight: 700;
            }
            QLabel#StudentStepPending { background: #222C3D; border: 1px solid #3A4458; }
            QLabel#StudentStepDone { background: #2563EB; border: 1px solid #3975F0; }
            QLabel#StudentStepTitle { color: #E8EDF3; font-size: 14px; font-weight: 600; }
        """)
        return home

    @staticmethod
    def _student_panel_title(text):
        label = QLabel(text)
        label.setObjectName("StudentPanelTitle")
        return label

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
        admin = not teacher and not student
        for widget in (self._welcome_card, self._card_connector, self._card_device,
                       self._card_signal, self._card_sample, self._card_warmup,
                       self._btn_start, self._next_hint):
            widget.setVisible(False)
        self._teacher_card.setVisible(teacher)
        self._flow_card.setVisible(False)
        self._student_home.setVisible(student)
        self._admin_home.setVisible(admin)
        if teacher:
            self._title_label.setText("教师工作台")
            self._desc_label.setText("查看学生学习概况，快速进入观察、任务与复盘流程。")
            self._teacher_timer.start()
            self._refresh_teacher_workspace()
        elif student:
            self._title_label.setText("欢迎与设备检查")
            self._desc_label.setText(
                "确认设备、信号与分析状态，准备开始一次学习。"
            )
            self._teacher_timer.stop()
        else:
            self._title_label.setText("系统总览")
            self._desc_label.setText(
                "实时查看脑机设备、AI 模型以及学习系统运行状态。"
            )
            self._teacher_timer.stop()
            self._refresh_admin_home(self.state)

    def _refresh_admin_home(self, state):
        if self._role != "research":
            return
        mode = str(getattr(state, "mode", "live") or "live").lower()
        device_online = (
            mode == "live" and getattr(state, "device_status", "offline") == "online"
        )
        self._admin_metrics["devices"].setText("1" if device_online else "0")

        today = datetime.now().astimezone().date()
        today_sessions = 0
        for session in getattr(state, "_history_sessions", []) or []:
            raw = str(getattr(session, "start_time", "") or "")
            try:
                if datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone().date() == today:
                    today_sessions += 1
            except ValueError:
                continue
        self._admin_metrics["sessions"].setText(str(today_sessions))
        try:
            profiles = self.identity_store.list_profiles()
            student_count = sum(
                str(item.get("last_role", "")) == "student" for item in profiles
            )
            self._admin_metrics["students"].setText(str(student_count))
        except OSError:
            self._admin_metrics["students"].setText("暂无数据")

        model_status = str(getattr(state, "model_status", "") or "").upper()
        if mode == "mock":
            model_text, model_color = "教学演示", "#FBBF24"
        elif model_status == "READY":
            model_text, model_color = "运行正常", "#4ADE80"
        elif model_status == "FAILED":
            model_text, model_color = "运行异常", "#F87171"
        else:
            model_text, model_color = "正在准备", "#FBBF24"
        self._admin_metrics["model"].setText(model_text)
        self._admin_metrics["model"].setStyleSheet(
            f"color: {model_color}; font-size: 24px; font-weight: 700;"
        )

        connector = getattr(state, "connector_status", "offline")
        device = getattr(state, "device_status", "offline")
        poor = getattr(state, "poor_signal", None)
        self._admin_health["connector"].setText({
            "online": "运行正常", "connecting": "正在连接", "offline": "未连接",
        }.get(connector, "暂无数据"))
        self._admin_health["device"].setText({
            "online": "在线", "waiting_raw": "等待数据", "offline": "离线",
        }.get(device, "暂无数据"))
        self._admin_health["signal"].setText(
            "暂无数据" if poor is None else
            "良好" if poor < MAX_POOR_SIGNAL else
            "需要关注" if poor < 200 else "无信号"
        )
        if mode == "mock":
            self._admin_health["connector"].setText("教学演示")
            self._admin_health["device"].setText("未连接真实设备")
            self._admin_health["signal"].setText("演示信号")
        self._admin_health["analysis"].setText(model_text)
        for label in self._admin_health.values():
            good = label.text() in {"运行正常", "在线", "良好"}
            warning = label.text() in {
                "正在连接", "等待数据", "需要关注", "正在准备", "教学演示", "演示信号"
            }
            color = "#4ADE80" if good else "#FBBF24" if warning else "#94A3B8"
            label.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: 600;")

        source_text = {"live": "实时采集", "mock": "教学演示", "replay": "离线回放"}.get(
            mode, "暂无数据"
        )
        sample_rate = getattr(state, "sample_rate_hz", None)
        self._admin_status_labels["source"].setText(source_text)
        self._admin_status_labels["sample"].setText(
            f"{sample_rate:.0f} Hz" if sample_rate is not None else "暂无设备数据"
        )
        self._admin_status_labels["contact"].setText(self._admin_health["signal"].text())
        self._admin_status_labels["session"].setText(
            "运行中" if getattr(state, "session_active", False) else "未开始"
        )

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
        for key, value in (
            ("students", len(students)), ("online", online),
            ("learning", learning), ("attention", attention),
        ):
            self._teacher_metrics[key].setText(str(value))

        attention_names = []
        for student_id, item in zip(students, snapshots):
            if not item or item.get("stale"):
                continue
            needs_attention = (
                item.get("quality_level") == "rejected"
                or item.get("stable_state") == "negative"
                or (item.get("attention") is not None and item.get("attention") < 45)
            )
            if needs_attention:
                profile = self.identity_store.get_profile(student_id) or {}
                attention_names.append(profile.get("name") or student_id)
        if attention_names:
            self._teacher_attention_text.setText("、".join(attention_names))
            self._teacher_attention_text.setStyleSheet(
                "color: #FBBF24; font-size: 13px; font-weight: 600;"
            )
            self._teacher_attention_helper.setText("建议进入实时状态页查看当前学习过程。")
        else:
            self._teacher_attention_text.setText("当前暂无需要特别关注的学生")
            self._teacher_attention_text.setStyleSheet(
                "color: #4ADE80; font-size: 13px; font-weight: 600;"
            )
            self._teacher_attention_helper.setText(
                "继续观察学习过程，必要时记录教师观察事件。"
            )

        snapshot = self.runtime_registry.get(self.selection_context.selected_student_id)
        if not snapshot or snapshot.get("stale"):
            self._teacher_online_badge.setText("未在线")
            self._teacher_online_badge.setStyleSheet(
                "color: #94A3B8; background: #222B3A; border-radius: 6px; padding: 7px 10px;"
            )
            self._teacher_task.setText("暂无进行中的任务")
            self._teacher_elapsed.setText("有效学习时长：暂无数据")
            self._teacher_learning_state.setText("暂无稳定状态")
            self._teacher_quality.setText("接触质量：暂无数据")
            self._teacher_advice.setText("暂无建议")
            self._teacher_activity.setText("暂无最近教学动态")
            self._teacher_detail.setText("当前观察学生：暂无实时数据")
            return
        elapsed = int(snapshot.get("elapsed_seconds") or 0)
        quality = {"trusted": "接触良好", "warning": "建议调整佩戴",
                   "rejected": "当前信号不可解释"}.get(snapshot.get("quality_level"), "等待信号")
        learning_state = {"positive": "积极", "neutral": "中性",
                          "negative": "消极"}.get(snapshot.get("stable_state"), "暂无稳定状态")
        is_online = bool(not snapshot.get("stale"))
        self._teacher_online_badge.setText("在线" if is_online else "离线")
        self._teacher_online_badge.setStyleSheet(
            ("color: #4ADE80; background: #15392D; border-radius: 6px; padding: 7px 10px;"
             if is_online else
             "color: #94A3B8; background: #222B3A; border-radius: 6px; padding: 7px 10px;")
        )
        self._teacher_task.setText(snapshot.get("task_name") or "暂无进行中的任务")
        self._teacher_elapsed.setText(
            f"有效学习时长：{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"
        )
        self._teacher_learning_state.setText(learning_state)
        self._teacher_quality.setText(f"接触质量：{quality}")
        self._teacher_advice.setText(snapshot.get("advice") or "暂无建议")
        legacy_state = {"positive": "积极", "neutral": "稳定",
                        "negative": "学习负荷偏高"}.get(
                            snapshot.get("stable_state"), "暂无"
                        )
        self._teacher_detail.setText(
            f"{snapshot.get('student_name')} {snapshot.get('student_id')} · "
            f"{'在线' if is_online else '离线'}\n"
            f"任务：{snapshot.get('task_name') or '无'} · "
            f"有效学习 {elapsed // 60:02d}:{elapsed % 60:02d}\n"
            f"接触质量：{quality} · 学习状态：{legacy_state}\n"
            f"智能建议：{snapshot.get('advice') or '暂无建议'}"
        )

        recent = snapshot.get("recent_events") or []
        lines = []
        for event in recent[-3:]:
            if isinstance(event, dict):
                label = event.get("label") or event.get("content") or ""
            else:
                label = getattr(event, "label", "") or getattr(event, "content", "")
            if label:
                lines.append(f"• {label}")
        self._teacher_activity.setText("\n".join(lines) if lines else "暂无最近教学动态")

    def _refresh_student_flow(self, state):
        if self._role != "student":
            return
        device_ready = state.mode == "mock" or state.device_status == "online"
        connector_ready = state.connector_status == "online"
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
        self._student_flow_action.setText(text)
        self._student_flow_action.setEnabled(target != "welcome")
        self._flow_target = target

        self._student_ready_value.setText(
            "可以开始学习" if device_ready else "暂不可开始学习"
        )
        self._student_ready_badge.setText("设备状态正常" if device_ready else "等待设备")
        self._student_ready_badge.setObjectName(
            "StudentBadgeGood" if device_ready else "StudentBadgeNeutral"
        )
        self._repolish(self._student_ready_badge)

        device_check_text = (
            "已完成" if device_ready
            else "等待设备数据" if connector_ready
            else "等待设备连接"
        )
        checks = {
            "device": (device_check_text, device_ready, False),
            "baseline": (
                "已有记录，可直接学习" if baseline_done else "尚未完成，请按当前规则操作",
                baseline_done, not baseline_done,
            ),
            "analysis": (
                "已准备，可解释学习状态" if state.warmup_complete else "准备中，按现有规则进入学习",
                bool(state.warmup_complete), not state.warmup_complete,
            ),
        }
        for key, (value_text, good, warn) in checks.items():
            value, dot = self._student_check_values[key]
            value.setText(value_text)
            dot.setObjectName("StudentDotGood" if good else "StudentDotWarn" if warn else "StudentDotNeutral")
            self._repolish(dot)
        current_step = 0 if not device_ready else 1 if not baseline_done else 2
        for index, marker in enumerate(self._student_flow_markers):
            marker.setObjectName("StudentStepDone" if index == current_step else "StudentStepPending")
            self._repolish(marker)

    @staticmethod
    def _repolish(widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _set_student_status(self, key, value, level, badge, helper):
        status = self._student_status[key]
        status["value"].setText(value)
        status["indicator"].set_state(level, badge)
        status["helper"].setText(helper)

    def update_state(self, state):
        if self._role == "teacher":
            self._refresh_teacher_workspace()
            return
        if self._role == "research":
            self._refresh_admin_home(state)
        self._refresh_student_flow(state)
        # Connector
        if state.connector_status == "online":
            self._ind_connector.set_state(StatusIndicator.LEVEL_GOOD, "已连接")
            self._set_student_status("connector", "已连接", StatusIndicator.LEVEL_GOOD,
                                     "在线", "本地通信服务正常")
        elif state.connector_status == "connecting":
            self._ind_connector.set_state(StatusIndicator.LEVEL_WARN, "连接中...")
            self._set_student_status("connector", "正在连接", StatusIndicator.LEVEL_WARN,
                                     "连接中", "正在连接本地通信服务")
        else:
            self._ind_connector.set_state(StatusIndicator.LEVEL_NEUTRAL, "等待启动")
            self._set_student_status("connector", "未连接", StatusIndicator.LEVEL_NEUTRAL,
                                     "等待", "请先启动 ThinkGear Connector")

        # Device — Mock模式不假装设备已连接
        if state.device_status == "online":
            self._ind_device.set_state(StatusIndicator.LEVEL_GOOD, "在线")
            self._set_student_status("device", "在线", StatusIndicator.LEVEL_GOOD,
                                     "正常", "设备已连接，等待稳定采样")
        elif state.device_status == "waiting_raw":
            self._ind_device.set_state(StatusIndicator.LEVEL_WARN, "等待原始数据")
            self._set_student_status("device", "等待数据", StatusIndicator.LEVEL_WARN,
                                     "等待", "设备已连接，正在等待脑电数据")
        else:
            self._ind_device.set_state(StatusIndicator.LEVEL_NEUTRAL, "未检测到设备")
            self._set_student_status("device", "未检测到", StatusIndicator.LEVEL_NEUTRAL,
                                     "离线", "请检查设备连接与佩戴状态")

        # Signal — poor_signal 可能为 None
        poor = state.poor_signal
        if poor is None:
            self._ind_signal.set_state(StatusIndicator.LEVEL_NEUTRAL, "等待信号")
            self._set_student_status("signal", "暂无数据", StatusIndicator.LEVEL_NEUTRAL,
                                     "等待", "收到信号后显示接触质量")
        elif poor < MAX_POOR_SIGNAL:
            self._ind_signal.set_state(StatusIndicator.LEVEL_GOOD, "接触良好")
            self._set_student_status("signal", "良好", StatusIndicator.LEVEL_GOOD,
                                     "合格", "当前信号可用于分析")
        elif poor < 200:
            self._ind_signal.set_state(StatusIndicator.LEVEL_WARN, "请调整电极接触")
            self._set_student_status("signal", "需要调整", StatusIndicator.LEVEL_WARN,
                                     "提醒", "请调整电极接触并保持静止")
        else:
            self._ind_signal.set_state(StatusIndicator.LEVEL_ERROR, "无信号")
            self._set_student_status("signal", "无信号", StatusIndicator.LEVEL_ERROR,
                                     "异常", "未检测到可用脑电信号")

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
            self._set_student_status("analysis", "已就绪", StatusIndicator.LEVEL_WARN,
                                     "教学演示", "演示分析不使用生产模型预热")
        else:
            self._card_warmup.set_title("智能分析准备")
        if state.mode != "mock" and state.warmup_complete:
            self._warmup_progress_label.setText("预热进度：100% ✓ 已完成")
            self._warmup_progress_label.setStyleSheet("font-size: 14px; color: #4ADE80;")
            self._set_student_status("analysis", "已准备", StatusIndicator.LEVEL_GOOD,
                                     "已就绪", "当前可以解释学习状态")
        elif state.mode != "mock":
            self._warmup_progress_label.setText(f"预热进度：{pct:.0f}%")
            remaining = max(0, int(WARMUP_SECONDS * (1 - state.warmup_progress)))
            self._set_student_status("analysis", "正在准备", StatusIndicator.LEVEL_WARN,
                                     f"预热 {pct:.0f}%", f"预计约 {remaining} 秒后完成分析准备")
        if state.connector_status == "online" and state.device_status == "online":
            self._next_hint.setText("设备与数据流已就绪，可以进入基线采集。")
            self._next_hint.setStyleSheet("color: #4ADE80; font-size: 12px;")
        else:
            self._next_hint.setText("尚未检测到完整设备数据；可进入基线页查看，但暂不能开始采集。")
            self._next_hint.setStyleSheet("color: #FBBF24; font-size: 12px;")
        self._student_next_hint.setText(self._next_hint.text())
