"""页面3：实时学习仪表盘（核心页面）。

严格遵循 eeg_modular/ui_prototype/services/dashboard_state.py 中定义的
DashboardState 正式字段接口。UI 业务逻辑只消费正式字段，
内部簿记字段（_前缀）仅用于图表缓冲绘制。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QInputDialog,
    QMessageBox, QFileDialog,
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
)


class DashboardPage(BasePage):
    def __init__(self, state, service):
        self.state = state
        self.service = service
        self._session_started = False
        self._session_paused = False
        super().__init__(scrollable=True)
        self._build_ui()

    def _build_ui(self):
        layout = self.content_layout
        layout.setSpacing(8)

        # ── 标题行 ──
        header = QHBoxLayout()
        title = QLabel("实时学习仪表盘")
        title.setObjectName("PageTitle")
        header.addWidget(title)

        self._session_time = QLabel("会话时间 00:00")
        self._session_time.setObjectName("AccentLabel")
        self._session_time.setStyleSheet("font-size: 18px; font-weight: bold;")
        self._session_time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header.addStretch()
        header.addWidget(self._session_time)
        layout.addLayout(header)

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

        # ── 第三行：左 EEG + 趋势 | 右 概率 + 持续状态 ──
        main_row = QHBoxLayout()
        main_row.setSpacing(10)

        # 左列
        left_col = QVBoxLayout()
        left_col.setSpacing(10)

        eeg_card = Card("EEG实时曲线")
        self._eeg_plot = EEGPlotWidget()
        self._eeg_plot.setMinimumHeight(100)
        eeg_card.add_widget(self._eeg_plot)
        left_col.addWidget(eeg_card, 1)

        trend_card = Card("Attention / Meditation 趋势（90秒）")
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

        att_card = Card("Attention")
        self._att_gauge = ArcGauge("Attention", "#4FC3F7")
        self._att_gauge.setFixedSize(72, 72)
        att_card.add_widget(self._wrap_centered(self._att_gauge))
        gauge_row.addWidget(att_card, 1)

        med_card = Card("Meditation")
        self._med_gauge = ArcGauge("Meditation", "#4ADE80")
        self._med_gauge.setFixedSize(72, 72)
        med_card.add_widget(self._wrap_centered(self._med_gauge))
        gauge_row.addWidget(med_card, 1)

        right_col.addLayout(gauge_row, 3)

        # 持续状态
        sustain_card = Card("最近90秒稳定状态")
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

        layout.addLayout(main_row, 1)

        # ── AI建议 ──
        ai_card = Card("AI学习建议")
        self._ai_label = QLabel("等待信号稳定后将生成学习建议。")
        self._ai_label.setWordWrap(True)
        self._ai_label.setStyleSheet("color: #C5CDD9; font-size: 14px;")
        ai_card.add_widget(self._ai_label)
        ai_card.setMaximumHeight(52)
        layout.addWidget(ai_card)

        # ── 控制按钮 ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self._btn_start = QPushButton("开始会话")
        self._btn_start.setObjectName("PrimaryButton")
        self._btn_start.clicked.connect(self._on_start)
        btn_row.addWidget(self._btn_start)

        self._btn_pause = QPushButton("暂停")
        self._btn_pause.setEnabled(False)
        self._btn_pause.clicked.connect(self._on_pause)
        btn_row.addWidget(self._btn_pause)

        self._btn_event = QPushButton("事件标记")
        self._btn_event.setEnabled(False)
        self._btn_event.clicked.connect(self._on_event)
        btn_row.addWidget(self._btn_event)

        self._btn_end = QPushButton("结束会话")
        self._btn_end.setObjectName("DangerButton")
        self._btn_end.setEnabled(False)
        self._btn_end.clicked.connect(self._on_end)
        btn_row.addWidget(self._btn_end)

        self._btn_export = QPushButton("导出报告")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._on_export)
        btn_row.addWidget(self._btn_export)

        btn_row.addStretch()
        layout.addLayout(btn_row)

    # ── 辅助方法 ──

    def _make_status_card(self, title: str) -> dict:
        card = Card(title)
        val = QLabel("--")
        val.setObjectName("CardValueSmall")
        ind = StatusIndicator()
        card.add_widget(val)
        card.add_widget(ind)
        card.setMaximumHeight(72)
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

    # ── 按钮事件 ──

    def _on_start(self):
        self.state.reset_session()
        self.service.start_session()
        self._session_started = True
        self._session_paused = False
        self._btn_start.setEnabled(False)
        self._btn_pause.setEnabled(True)
        self._btn_pause.setText("暂停")
        self._btn_event.setEnabled(True)
        self._btn_end.setEnabled(True)
        self._btn_export.setEnabled(False)
        self._trend_plot.reset()
        self._prob_trend.reset()
        self.state.add_event("会话开始", "system")

    def _on_pause(self):
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
        text, ok = QInputDialog.getText(
            self, "事件标记", "输入事件描述："
        )
        if ok and text:
            self.state.add_event(text, "user")

    def _on_end(self):
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

    def update_state(self, state):
        s = state

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
        reasons_text = "、".join(s.quality_reasons) if s.quality_reasons else ""
        self._card_conf["value"].setToolTip(reasons_text)
        self._card_conf["indicator"].setToolTip(reasons_text)

        # 采样率卡片：使用常量展示，不使用 sample_rate_hz（Mock 模式下为 None）
        self._card_rate["value"].setText(f"设备目标: {DEVICE_TARGET_SAMPLE_HZ}Hz")
        self._card_rate["indicator"].set_state(
            StatusIndicator.LEVEL_NEUTRAL, f"Mock刷新: {MOCK_UI_REFRESH_HZ}Hz"
        )

        # ── 预热进度：使用 warmup_progress（0.0~1.0）──
        self._warmup_bar.setValue(int(s.warmup_progress * 100))
        if s.warmup_complete:
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

        # ── 预测结果：使用 predicted_state 和 confidence ──
        if s.inference_eligible and s.predicted_state is not None:
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
            self._pred_label.setText("当前状态：等待推理...")
            self._pred_label.setStyleSheet("font-size: 16px; padding: 4px 0; color: #6B7689;")

        # ── 概率趋势：使用内部缓冲 _prob_history ──
        if s._prob_history:
            latest = s._prob_history[-1]
            self._prob_trend.push_values(latest[1], latest[2], latest[3])

        # ── 持续状态：使用 stable_state ──
        if s.inference_eligible:
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
        self._sustain_neg.setText(f"消极持续：{s._negative_sustain_seconds:.1f}秒")

        # 干预状态：使用内部簿记 _intervention_triggered / _intervention_cooldown
        if s._intervention_triggered:
            self._intervention_label.setText("⚠ 已触发干预建议（消极状态持续超过20秒）")
            self._intervention_label.setVisible(True)
        elif s._intervention_cooldown:
            self._intervention_label.setText("干预冷却中（90秒内不重复触发）")
            self._intervention_label.setVisible(True)
        else:
            self._intervention_label.setVisible(False)

        # ── AI建议：使用 feedback_text ──
        self._ai_label.setText(s.feedback_text)
        if s.inference_eligible:
            self._ai_label.setStyleSheet("color: #C5CDD9; font-size: 14px;")
        else:
            self._ai_label.setStyleSheet("color: #FBBF24; font-size: 14px;")
