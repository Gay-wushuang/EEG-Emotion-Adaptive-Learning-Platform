"""页面7：历史CSV回放模式。

回放页面使用自身内部状态（self._data）驱动播放，不依赖 DashboardState 的
实时字段。DashboardState 的 prob_* 等字段在 Mock 模式下可能为 None，
因此概率面板通过直接 set_value 方式更新，绕过 update_state(state)。
"""

from __future__ import annotations

import csv
import io
import math
import random
from collections import Counter
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
    QPushButton, QFrame, QSlider, QFileDialog, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QMessageBox,
)

from pages.base_page import BasePage
from widgets.card import Card
from widgets.eeg_plot import EEGPlotWidget
from widgets.trend_plot import TrendPlotWidget
from widgets.probability_bar import ProbabilityPanel
from widgets.gauge import ArcGauge
from services.dashboard_state import CLASS_DISPLAY


def _generate_sample_csv() -> str:
    """生成模拟CSV回放数据（演示数据）。"""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "timestamp", "raw", "attention", "meditation", "poor_signal",
        "delta", "theta", "alpha1", "alpha2", "beta1", "beta2", "gamma1", "gamma2",
        "prob_positive", "prob_neutral", "prob_negative", "predicted_class",
    ])
    t = 0.0
    for i in range(600):  # 60秒 @ 10Hz
        t += 0.1
        cycle = (t % 120.0) / 120.0
        if cycle < 0.35:
            trend, probs = 0, [0.55, 0.30, 0.15]
        elif cycle < 0.70:
            trend, probs = 1, [0.22, 0.55, 0.23]
        else:
            trend, probs = 2, [0.15, 0.28, 0.57]

        att = int(max(0, min(100, [70, 60, 38][trend] + 10 * math.sin(t / 12) + random.gauss(0, 3))))
        med = int(max(0, min(100, [65, 52, 35][trend] + 12 * math.sin(t / 15 + 1.5) + random.gauss(0, 3))))
        raw = int(400 * math.sin(t * 8) + 150 * math.sin(t * 23) + random.gauss(0, 80))
        poor = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 25])

        writer.writerow([
            f"{t:.1f}", raw, att, med, poor,
            *[int(50000 + 20000 * math.sin(t / (14 + j))) for j in range(8)],
            f"{probs[0]:.4f}", f"{probs[1]:.4f}", f"{probs[2]:.4f}",
            ["positive", "neutral", "negative"][probs.index(max(probs))],
        ])
    return output.getvalue()


class ReplayPage(BasePage):
    def __init__(self, state, service):
        self.state = state
        self.service = service
        self._data = []
        self._index = 0
        self._playing = False
        self._speed = 1.0
        self._is_sample = False  # 标记当前是否为示例（演示）数据
        self._timer = QTimer()
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)
        super().__init__(
            "历史CSV回放",
            "加载历史会话CSV数据进行回放分析，支持播放控制与变速回放。"
        )
        self._build_ui()

    def _build_ui(self):
        splitter = QSplitter(Qt.Vertical)

        # ── 上部：控制 + 图表 ──
        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)

        # 回放模式 - 演示数据 标记（加载示例数据后显示）
        self._demo_label = QLabel("回放模式 - 演示数据")
        self._demo_label.setStyleSheet(
            "background-color: rgba(200,150,40,0.15); "
            "color: #FBBF24; font-size: 12px; font-weight: 600; "
            "padding: 6px 10px; border-radius: 4px;"
        )
        self._demo_label.setAlignment(Qt.AlignCenter)
        self._demo_label.setVisible(False)
        top_layout.addWidget(self._demo_label)

        # 文件加载与控制
        control_card = Card("回放控制")
        control_layout = QHBoxLayout()
        control_layout.setSpacing(10)

        self._btn_load = QPushButton("加载一个或多个CSV")
        self._btn_load.clicked.connect(self._load_file)
        control_layout.addWidget(self._btn_load)

        self._btn_folder = QPushButton("打开会话文件夹")
        self._btn_folder.clicked.connect(self._open_sessions_folder)
        control_layout.addWidget(self._btn_folder)

        self._btn_sample = QPushButton("加载示例数据")
        self._btn_sample.clicked.connect(self._load_sample)
        control_layout.addWidget(self._btn_sample)

        control_layout.addSpacing(20)

        self._btn_play = QPushButton("播放")
        self._btn_play.setObjectName("PrimaryButton")
        self._btn_play.setEnabled(False)
        self._btn_play.setToolTip("请先加载包含Raw EEG的会话CSV")
        self._btn_play.clicked.connect(self._play)
        control_layout.addWidget(self._btn_play)

        self._btn_pause = QPushButton("暂停")
        self._btn_pause.setEnabled(False)
        self._btn_pause.clicked.connect(self._pause)
        control_layout.addWidget(self._btn_pause)

        self._btn_stop = QPushButton("停止")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop)
        control_layout.addWidget(self._btn_stop)

        control_layout.addSpacing(20)

        control_layout.addWidget(QLabel("速度:"))
        self._combo_speed = QComboBox()
        self._combo_speed.addItems(["0.5x", "1x", "2x", "4x"])
        self._combo_speed.setCurrentIndex(1)
        self._combo_speed.currentIndexChanged.connect(self._change_speed)
        control_layout.addWidget(self._combo_speed)

        control_layout.addStretch()

        self._label_file = QLabel("未加载文件")
        self._label_file.setStyleSheet("color: #6B7689; font-size: 13px;")
        control_layout.addWidget(self._label_file)

        control_card.add_widget(self._wrap(control_layout))
        top_layout.addWidget(control_card)

        # 进度条
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_seek)
        top_layout.addWidget(self._slider)

        self._label_progress = QLabel("0 / 0")
        self._label_progress.setStyleSheet("color: #6B7689; font-size: 12px;")
        self._label_progress.setAlignment(Qt.AlignCenter)
        top_layout.addWidget(self._label_progress)

        # 图表区
        charts_layout = QHBoxLayout()
        charts_layout.setSpacing(10)

        eeg_card = Card("EEG回放")
        self._eeg_plot = EEGPlotWidget()
        self._eeg_plot.setMinimumHeight(180)
        eeg_card.add_widget(self._eeg_plot)
        charts_layout.addWidget(eeg_card, 1)

        trend_card = Card("Attention / Meditation 回放")
        self._trend_plot = TrendPlotWidget()
        self._trend_plot.setMinimumHeight(180)
        trend_card.add_widget(self._trend_plot)
        charts_layout.addWidget(trend_card, 1)

        top_layout.addLayout(charts_layout)
        splitter.addWidget(top)

        # ── 下部：回放数据详情 ──
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(10)

        # 概率面板
        prob_card = Card("回放 - 模型概率")
        self._prob_panel = ProbabilityPanel()
        prob_card.add_widget(self._prob_panel)

        self._replay_pred = QLabel("预测状态：--")
        self._replay_pred.setObjectName("AccentLabel")
        self._replay_pred.setStyleSheet("font-size: 14px;")
        prob_card.add_widget(self._replay_pred)

        self._replay_dominant = QLabel("整场主导状态（有效预测众数）：--")
        self._replay_dominant.setStyleSheet("font-size: 13px; color: #AAB6C8;")
        prob_card.add_widget(self._replay_dominant)

        bottom_layout.addWidget(prob_card, 1)

        # 仪表
        gauge_row = QHBoxLayout()
        att_card = Card("Attention")
        self._att_gauge = ArcGauge("Attention", "#4FC3F7")
        self._att_gauge.setFixedSize(120, 120)
        att_card.add_widget(self._wrap_centered(self._att_gauge))
        gauge_row.addWidget(att_card)

        med_card = Card("Meditation")
        self._med_gauge = ArcGauge("Meditation", "#4ADE80")
        self._med_gauge.setFixedSize(120, 120)
        med_card.add_widget(self._wrap_centered(self._med_gauge))
        gauge_row.addWidget(med_card)

        bottom_layout.addLayout(gauge_row, 1)

        # 数据表
        table_card = Card("回放数据预览")
        table_layout = QVBoxLayout()
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["时间", "Raw", "Att", "Med", "预测"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        table_layout.addWidget(self._table)
        table_card.add_widget(self._wrap(table_layout))
        bottom_layout.addWidget(table_card, 2)

        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        self.content_layout.addWidget(splitter)

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

    def _load_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择一个或多个会话CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not paths:
            return
        self.load_paths(paths)

    def load_paths(self, paths):
        """Load one session or a sequence of compatible session CSV files."""
        try:
            combined = []
            missing_predictions = False
            for path in paths:
                with open(path, "r", encoding="utf-8-sig") as f:
                    rows = list(csv.DictReader(f))
                if not rows or not any("raw" in row or "raw_eeg" in row for row in rows):
                    raise ValueError(f"{Path(path).name} 不包含 Raw EEG 列")
                normalized = [self._normalize_row(row, Path(path).name) for row in rows]
                missing_predictions |= not any(row["predicted_class"] for row in normalized)
                combined.extend(normalized)
            self._data = combined
            self._is_sample = False
            self._demo_label.setVisible(False)
            self._label_file.setText(f"已加载 {len(paths)} 个文件，共 {len(self._data)} 行")
            self._init_playback()
            if missing_predictions:
                QMessageBox.information(
                    self, "原始数据回放",
                    "部分文件没有模型概率，因此只能回放 Raw/ATT/MED，不能补造状态结果。\n"
                    "应用新保存的 session.csv 是包含采集、质量与推理结果的综合CSV。"
                )
        except Exception as e:
            QMessageBox.warning(self, "加载失败", f"无法加载文件：{e}")

    def _open_sessions_folder(self):
        folder = Path(getattr(self.service, "sessions_dir", Path("data/sessions"))).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    @staticmethod
    def _normalize_row(row: dict, source: str = "") -> dict:
        pred = (row.get("predicted_class") or row.get("prediction") or "").strip()
        pred = {"happy": "positive", "normal": "neutral", "sad": "negative"}.get(pred, pred)
        return {
            **row,
            "source_file": source,
            "timestamp": row.get("signal_time_seconds") or row.get("timestamp") or row.get("timestamp_unix") or "",
            "raw": row.get("raw") if row.get("raw") not in (None, "") else row.get("raw_eeg", ""),
            "attention": row.get("attention") if row.get("attention") not in (None, "") else row.get("att", ""),
            "meditation": row.get("meditation") if row.get("meditation") not in (None, "") else row.get("med", ""),
            "predicted_class": pred,
        }

    def _load_sample(self):
        """加载内置示例数据（演示数据）。"""
        csv_text = _generate_sample_csv()
        reader = csv.DictReader(io.StringIO(csv_text))
        self._data = list(reader)
        self._is_sample = True
        self._demo_label.setVisible(True)
        self._label_file.setText(f"示例数据(演示) ({len(self._data)}行)")
        self._init_playback()

    def _init_playback(self):
        self._index = 0
        self._slider.setEnabled(True)
        self._slider.setRange(0, len(self._data) - 1)
        self._slider.setValue(0)
        self._btn_play.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._trend_plot.reset()
        self._update_frame(0)
        self._populate_table()
        self._update_dominant_state()

    def _update_dominant_state(self):
        votes = []
        seen_inferences = set()
        for row in self._data:
            pred = row.get("predicted_class", "")
            if pred not in ("positive", "neutral", "negative"):
                continue
            inference_id = row.get("inference_index", "")
            if inference_id:
                key = (row.get("source_file", ""), inference_id)
                if key in seen_inferences:
                    continue
                seen_inferences.add(key)
            votes.append(pred)
        if not votes:
            self._replay_dominant.setText("整场主导状态（有效预测众数）：--（无模型预测）")
            return
        counts = Counter(votes)
        highest = max(counts.values())
        tied = {name for name, count in counts.items() if count == highest}
        dominant = next(name for name in reversed(votes) if name in tied)
        self._replay_dominant.setText(
            f"整场主导状态（有效预测众数）：{CLASS_DISPLAY.get(dominant, dominant)} · {highest}/{len(votes)}"
        )

    def _populate_table(self):
        self._table.setRowCount(min(50, len(self._data)))
        for i in range(min(50, len(self._data))):
            row = self._data[i]
            self._table.setItem(i, 0, QTableWidgetItem(row.get("timestamp", "")))
            self._table.setItem(i, 1, QTableWidgetItem(row.get("raw", "")))
            self._table.setItem(i, 2, QTableWidgetItem(row.get("attention", "")))
            self._table.setItem(i, 3, QTableWidgetItem(row.get("meditation", "")))
            self._table.setItem(i, 4, QTableWidgetItem(row.get("predicted_class", "")))

    def _play(self):
        if not self._data:
            return
        self._playing = True
        self._btn_play.setEnabled(False)
        self._btn_pause.setEnabled(True)
        self._btn_stop.setEnabled(True)
        self._timer.start()

    def _pause(self):
        self._playing = False
        self._timer.stop()
        self._btn_play.setEnabled(True)
        self._btn_pause.setEnabled(False)

    def _stop(self):
        self._playing = False
        self._timer.stop()
        self._index = 0
        self._slider.setValue(0)
        self._btn_play.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)

    def _change_speed(self, idx: int):
        speeds = [0.5, 1.0, 2.0, 4.0]
        self._speed = speeds[idx]
        self._timer.setInterval(int(100 / self._speed))

    def _on_seek(self, value: int):
        self._index = value
        self._update_frame(value)

    def _tick(self):
        if self._index >= len(self._data) - 1:
            self._pause()
            return
        self._index += 1
        self._slider.setValue(self._index)
        self._update_frame(self._index)

    def _safe_float(self, row: dict, key: str, default: float = 0.0) -> float:
        """安全解析 CSV 行中的浮点数，避免 None 或非法值导致崩溃。"""
        val = row.get(key)
        if val is None or val == "":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _safe_int(self, row: dict, key: str, default: int = 0) -> int:
        """安全解析 CSV 行中的整数。"""
        val = row.get(key)
        if val is None or val == "":
            return default
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return default

    def _update_frame(self, idx: int):
        if not self._data or idx >= len(self._data):
            return
        row = self._data[idx]

        raw = self._safe_int(row, "raw")
        att = self._safe_int(row, "attention")
        med = self._safe_int(row, "meditation")
        pp = self._safe_float(row, "prob_positive")
        pn = self._safe_float(row, "prob_neutral")
        ng = self._safe_float(row, "prob_negative")
        pred = row.get("predicted_class", "") or ""

        self._eeg_plot.push_value(raw)
        self._trend_plot.push_values(att, med)
        self._att_gauge.set_value(att)
        self._med_gauge.set_value(med)

        # 直接操作概率面板的条形组件，绕过 DashboardState
        # （回放模式下 DashboardState 的 prob_* 字段为 None，不能通过
        #   _prob_panel.update_state(state) 更新）
        self._prob_panel._bars["positive"].set_value(pp)
        self._prob_panel._bars["neutral"].set_value(pn)
        self._prob_panel._bars["negative"].set_value(ng)
        for bar in self._prob_panel._bars.values():
            bar.set_dimmed(False)
        self._prob_panel._confidence_label.setText("回放模式 - 信号可信度: N/A")
        self._prob_panel._confidence_label.setStyleSheet(
            "font-size: 12px; color: #6B7689; padding-top: 4px;"
        )
        self._prob_panel._warning_label.setVisible(False)

        display = CLASS_DISPLAY.get(pred, pred) if pred else "--"
        self._replay_pred.setText(f"预测状态：{display}")

        self._label_progress.setText(
            f"{idx + 1} / {len(self._data)}  ({row.get('timestamp', '')}s)"
        )

    def update_state(self, state):
        """回放页面不消费 DashboardState 实时字段，空实现避免父类调用。"""
        pass
