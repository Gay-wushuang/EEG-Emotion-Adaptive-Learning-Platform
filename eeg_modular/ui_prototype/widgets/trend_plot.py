"""Attention / Meditation 趋势图（pyqtgraph）。"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QVBoxLayout, QWidget, QHBoxLayout, QLabel


class TrendPlotWidget(QWidget):
    """Attention + Meditation 双曲线趋势图。"""

    MAX_POINTS = 900  # 90秒 @ 10Hz

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # 图例
        legend_layout = QHBoxLayout()
        legend_layout.setSpacing(16)
        legend_layout.addStretch()

        att_dot = QLabel("●")
        att_dot.setStyleSheet("color: #4FC3F7; font-size: 14px;")
        att_label = QLabel("Attention")
        att_label.setStyleSheet("color: #4FC3F7; font-size: 12px;")
        legend_layout.addWidget(att_dot)
        legend_layout.addWidget(att_label)

        med_dot = QLabel("●")
        med_dot.setStyleSheet("color: #8EA3BF; font-size: 14px;")
        med_label = QLabel("Meditation")
        med_label.setStyleSheet("color: #8EA3BF; font-size: 12px;")
        legend_layout.addWidget(med_dot)
        legend_layout.addWidget(med_label)
        legend_layout.addStretch()
        layout.addLayout(legend_layout)

        self._plot = pg.PlotWidget()
        self._plot.setBackground(QColor("#1A1F2E"))
        self._plot.showGrid(x=False, y=True, alpha=0.15)
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.hideButtons()
        self._plot.setYRange(0, 100, padding=0.02)
        self._plot.setLabel("left", "数值", color="#6B7689", **{"font-size": "10px"})
        self._plot.setLabel("bottom", "时间 (秒)", color="#6B7689", **{"font-size": "10px"})

        for axis_name in ["left", "bottom"]:
            ax = self._plot.getAxis(axis_name)
            ax.setPen(QColor("#2D3548"))
            ax.setTextPen(QColor("#6B7689"))
            ax.setStyle(tickFont=QFont("Microsoft YaHei UI", 9))

        self._att_curve = self._plot.plot(pen=pg.mkPen(QColor("#4FC3F7"), width=1.8))
        self._med_curve = self._plot.plot(pen=pg.mkPen(QColor("#8EA3BF"), width=1.8))

        # 阈值参考线
        for y, label in [(30, None), (70, None)]:
            line = pg.InfiniteLine(
                pos=y, angle=0,
                pen=pg.mkPen(QColor("#2D3548"), width=1, style=Qt.DotLine)
            )
            self._plot.addItem(line)

        layout.addWidget(self._plot)

        self._att_data = np.full(self.MAX_POINTS, np.nan, dtype=np.float32)
        self._med_data = np.full(self.MAX_POINTS, np.nan, dtype=np.float32)
        self._x = np.linspace(-90, 0, self.MAX_POINTS, dtype=np.float32)

    def push_values(self, attention: int, meditation: int):
        self._att_data = np.roll(self._att_data, -1)
        self._att_data[-1] = attention
        self._med_data = np.roll(self._med_data, -1)
        self._med_data[-1] = meditation
        self._att_curve.setData(self._x, self._att_data)
        self._med_curve.setData(self._x, self._med_data)

    def reset(self):
        self._att_data = np.full(self.MAX_POINTS, np.nan, dtype=np.float32)
        self._med_data = np.full(self.MAX_POINTS, np.nan, dtype=np.float32)
        self._att_curve.setData(self._x, self._att_data)
        self._med_curve.setData(self._x, self._med_data)


class ProbabilityTrendWidget(QWidget):
    """三分类概率趋势堆叠图。"""

    MAX_POINTS = 450  # 90秒 @ 5Hz(inference 2s, but tick 5Hz)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        legend = QHBoxLayout()
        legend.setContentsMargins(0, 0, 0, 0)
        legend.setSpacing(14)
        legend.addStretch()
        for label, color in (
            ("积极", "#5B8DEF"),
            ("中性", "#A0AEC0"),
            ("消极", "#6F82A0"),
        ):
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 11px;")
            text = QLabel(label)
            text.setStyleSheet("color: #A5B0C0; font-size: 11px;")
            legend.addWidget(dot)
            legend.addWidget(text)
        legend.addStretch()
        layout.addLayout(legend)

        self._plot = pg.PlotWidget()
        self._plot.setBackground(QColor("#1A1F2E"))
        self._plot.showGrid(x=False, y=True, alpha=0.12)
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.hideButtons()
        self._plot.setYRange(0, 1.0, padding=0.02)
        self._plot.setLabel("left", "概率", color="#6B7689", **{"font-size": "10px"})
        self._plot.setLabel("bottom", "时间 (秒)", color="#6B7689", **{"font-size": "10px"})

        for axis_name in ["left", "bottom"]:
            ax = self._plot.getAxis(axis_name)
            ax.setPen(QColor("#2D3548"))
            ax.setTextPen(QColor("#6B7689"))
            ax.setStyle(tickFont=QFont("Microsoft YaHei UI", 9))

        self._pos_curve = self._plot.plot(
            pen=pg.mkPen(QColor("#5B8DEF"), width=1.5),
            fillLevel=0,
            fillBrush=pg.mkBrush(QColor(91, 141, 239, 30)),
        )
        self._neu_curve = self._plot.plot(
            pen=pg.mkPen(QColor("#A0AEC0"), width=1.5),
        )
        self._neg_curve = self._plot.plot(
            pen=pg.mkPen(QColor("#6F82A0"), width=1.5),
            fillLevel=0,
            fillBrush=pg.mkBrush(QColor(111, 130, 160, 24)),
        )

        # 阈值线
        thresh = pg.InfiniteLine(
            pos=0.60, angle=0,
            pen=pg.mkPen(QColor("#FBBF24"), width=1, style=Qt.DashLine)
        )
        self._plot.addItem(thresh)

        layout.addWidget(self._plot)

        self._pos_data = np.full(self.MAX_POINTS, np.nan, dtype=np.float32)
        self._neu_data = np.full(self.MAX_POINTS, np.nan, dtype=np.float32)
        self._neg_data = np.full(self.MAX_POINTS, np.nan, dtype=np.float32)
        self._x = np.linspace(-90, 0, self.MAX_POINTS, dtype=np.float32)

    def push_values(self, pos: float, neu: float, neg: float):
        self._pos_data = np.roll(self._pos_data, -1)
        self._pos_data[-1] = pos
        self._neu_data = np.roll(self._neu_data, -1)
        self._neu_data[-1] = neu
        self._neg_data = np.roll(self._neg_data, -1)
        self._neg_data[-1] = neg
        self._pos_curve.setData(self._x, self._pos_data)
        self._neu_curve.setData(self._x, self._neu_data)
        self._neg_curve.setData(self._x, self._neg_data)

    def reset(self):
        for arr_name in ["_pos_data", "_neu_data", "_neg_data"]:
            setattr(self, arr_name, np.full(self.MAX_POINTS, np.nan, dtype=np.float32))
        self._pos_curve.setData(self._x, self._pos_data)
        self._neu_curve.setData(self._x, self._neu_data)
        self._neg_curve.setData(self._x, self._neg_data)
