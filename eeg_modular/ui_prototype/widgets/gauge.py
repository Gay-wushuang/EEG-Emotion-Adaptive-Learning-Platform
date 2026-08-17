"""弧形仪表盘组件（QPainter自绘，非霓虹风格）。"""

from __future__ import annotations

import math
from PySide6.QtCore import Qt, QRectF, QSize
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QConicalGradient
from PySide6.QtWidgets import QWidget


class ArcGauge(QWidget):
    """270度弧形仪表，显示0-100数值。"""

    def __init__(self, title: str = "", color: str = "#4FC3F7", parent=None):
        super().__init__(parent)
        self._title = title
        self._color = QColor(color)
        self._value = 0
        self._max = 100
        self.setMinimumSize(82, 82)

    def set_value(self, value: int):
        self._value = max(0, min(self._max, int(value)))
        self.update()

    def set_color(self, color: str):
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        side = min(w, h)
        cx = w / 2
        cy = h / 2 + 3
        radius = max(18, side / 2 - 13)

        # 弧参数：从225度到-45度（270度跨度）
        start_angle = 225 * 16
        span_angle = 270 * 16

        # 背景轨道
        rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
        stroke = max(5, min(8, int(side / 18)))
        bg_pen = QPen(QColor("#2D3548"), stroke, Qt.SolidLine, Qt.RoundCap)
        p.setPen(bg_pen)
        p.drawArc(rect, start_angle, -span_angle)

        # 前景值弧
        ratio = self._value / self._max
        if ratio > 0:
            fg_pen = QPen(self._color, stroke, Qt.SolidLine, Qt.RoundCap)
            p.setPen(fg_pen)
            p.drawArc(rect, start_angle, -int(span_angle * ratio))

        # 数值
        p.setPen(QColor("#E8EDF3"))
        font = QFont("Microsoft YaHei UI", max(14, int(side / 6)), QFont.Bold)
        p.setFont(font)
        val_rect = QRectF(cx - radius, cy - radius + 20, radius * 2, radius * 2 - 40)
        p.drawText(val_rect, Qt.AlignCenter, str(self._value))

        # 标题
        if self._title:
            p.setPen(QColor("#8B95A7"))
            font = QFont("Microsoft YaHei UI", 8)
            p.setFont(font)
            title_rect = QRectF(0, h - 22, w, 18)
            p.drawText(title_rect, Qt.AlignCenter, self._title)
