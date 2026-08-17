"""EEG实时曲线（pyqtgraph）。"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QVBoxLayout, QWidget


# 统一 pyqtgraph 暗色主题
pg.setConfigOptions(antialias=True, useNumba=False)


class EEGPlotWidget(QWidget):
    """Readable display-only view of the single-channel Raw EEG stream."""

    DISPLAY_SECONDS = 5
    DISPLAY_POINTS = 512

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot = pg.PlotWidget()
        self._plot.setBackground(QColor("#1A1F2E"))
        self._plot.showGrid(x=False, y=True, alpha=0.15)
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.hideButtons()
        self._plot.setLabel("left", "显示幅度（相对值）", color="#6B7689", **{"font-size": "10px"})
        self._plot.setLabel("bottom", "最近5秒", color="#6B7689", **{"font-size": "10px"})
        self._plot.setYRange(-250, 250, padding=0.02)

        # 坐标轴样式
        for axis_name in ["left", "bottom"]:
            ax = self._plot.getAxis(axis_name)
            ax.setPen(QColor("#2D3548"))
            ax.setTextPen(QColor("#6B7689"))
            ax.setStyle(tickFont=QFont("Microsoft YaHei UI", 9))

        self._curve = self._plot.plot(
            pen=pg.mkPen(QColor("#4FC3F7"), width=1.5),
        )
        # 零线
        self._zero_line = pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen(QColor("#2D3548"), width=1, style=Qt.DashLine)
        )
        self._plot.addItem(self._zero_line)

        layout.addWidget(self._plot)

        self._data = np.zeros(self.DISPLAY_POINTS, dtype=np.float32)
        self._x = np.linspace(-self.DISPLAY_SECONDS, 0, self.DISPLAY_POINTS)
        self._scale = 250.0

    def push_value(self, value: float):
        self._data = np.roll(self._data, -1)
        self._data[-1] = value
        self._curve.setData(self._x, self._data)

    def push_buffer(self, buffer):
        """直接用 deque/list 替换全部数据。"""
        arr = np.asarray(buffer, dtype=np.float32)[-(512 * self.DISPLAY_SECONDS):]
        if not arr.size:
            return
        # Display processing only: remove DC offset and average adjacent Raw
        # samples so a 512 Hz stream remains readable on a ~500 px chart.
        arr = arr - float(np.median(arr))
        block = max(1, int(np.ceil(arr.size / self.DISPLAY_POINTS)))
        usable = (arr.size // block) * block
        if usable:
            arr = arr[-usable:].reshape(-1, block).mean(axis=1)
        if arr.size > self.DISPLAY_POINTS:
            arr = arr[-self.DISPLAY_POINTS:]
        self._data = np.full(self.DISPLAY_POINTS, np.nan, dtype=np.float32)
        self._data[-arr.size:] = arr
        robust_peak = float(np.nanpercentile(np.abs(arr), 99)) if arr.size else 0.0
        target_scale = float(np.clip(robust_peak * 1.25, 100.0, 1000.0))
        self._scale = 0.90 * self._scale + 0.10 * target_scale
        self._plot.setYRange(-self._scale, self._scale, padding=0.02)
        self._curve.setData(self._x, self._data)

    def set_dimmed(self, dimmed: bool):
        color = QColor("#3A4458") if dimmed else QColor("#4FC3F7")
        self._curve.setPen(pg.mkPen(color, width=1.5))
