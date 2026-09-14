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

    def _build_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setSpacing(14)

        # ── 左侧：用户信息 + 采集控制 ──
        left = QVBoxLayout()
        left.setSpacing(14)

        self._teacher_baseline_card = Card("学生基线概览")
        teacher_layout = QVBoxLayout()
        self._teacher_student_combo = QComboBox()
        self._teacher_student_combo.currentIndexChanged.connect(
            self._teacher_student_changed
        )
        self._teacher_baseline_detail = QLabel("请选择学生")
        self._teacher_baseline_detail.setWordWrap(True)
        teacher_layout.addWidget(QLabel("当前观察学生："))
        teacher_layout.addWidget(self._teacher_student_combo)
        teacher_layout.addWidget(self._teacher_baseline_detail)
        self._teacher_baseline_card.add_widget(self._wrap_layout(teacher_layout))
        self._teacher_baseline_card.setVisible(False)
        left.addWidget(self._teacher_baseline_card)

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

    def _on_duration_change(self, seconds: int):
        if self._role == "teacher":
            return
        self.state._baseline_target = float(seconds)
        self._label_time.setText(f"已用时间：0秒 / {int(self.state._baseline_target)}秒")

    def set_identity(self, user_id: str, user_name: str = "") -> None:
        """Reset the reused page, then load only the selected account's baseline."""
        self._input_uid.setText(user_id)
        self._input_name.setText(user_name)
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
        self._user_card.setVisible(not read_only)
        self._baseline_actions.setVisible(not read_only)
        for card in (self._collect_card, self._signal_card, self._eeg_card, self._stats_card):
            card.setVisible(not read_only)
        self._combo_duration.setEnabled(not read_only)
        self._btn_start.setVisible(not read_only)
        self._btn_stop.setVisible(not read_only)
        if read_only:
            self._title_label.setText("学生基线状态")
            self._desc_label.setText("查看学生基线完成情况、信号质量与统计；采集由学生本人执行。")
            self._label_status.setText("当前学生基线状态：等待数据")
            self._teacher_baseline_card.setVisible(True)
            self._teacher_timer.start()
            self._refresh_teacher_baseline()
        else:
            self._title_label.setText("用户初始化与基线采集")
            self._desc_label.setText("采集60～90秒静息态基线数据，用于个人校准和后续状态对比。")
            self._teacher_baseline_card.setVisible(False)
            self._teacher_timer.stop()

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
            return
        if status_value in {"FAILED", "EARLY_STOPPED"}:
            message = "本次未形成有效基线" if status_value == "EARLY_STOPPED" else (
                live.get("baseline_failure_reason") or "基线采集失败"
            )
            self._teacher_baseline_detail.setText(
                f"当前观察学生：{student_id}\n基线状态："
                f"{'提前结束' if status_value == 'EARLY_STOPPED' else '失败'}\n{message}"
            )
            return
        result = persisted or (live if status_value == "COMPLETED" else None)
        if not result:
            self._teacher_baseline_detail.setText(
                f"当前观察学生：{student_id}\n尚未完成基线采集"
            )
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

    def update_state(self, state):
        if self._role == "teacher":
            self._refresh_teacher_baseline()
            return
        # EEG曲线 — 使用内部缓冲 _eeg_raw_buffer
        if state._eeg_raw_buffer:
            self._eeg_plot.push_buffer(state._eeg_raw_buffer)

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

            self._progress_bar.setValue(int(pct * 100))

            # 自动结束
            if elapsed >= target:
                self._complete_baseline()

    def on_hide(self):
        pass
