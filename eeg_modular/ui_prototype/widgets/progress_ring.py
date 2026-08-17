"""圆形进度环组件。"""

from __future__ import annotations

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QColor, QPen, QFont
from PySide6.QtWidgets import QWidget


class ProgressRing(QWidget):
    """圆形进度环，支持百分比文本和状态色。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._progress = 0.0      # 0.0 ~ 1.0
        self._color = QColor("#4FC3F7")
        self._text = ""
        self._subtext = ""
        self.setMinimumSize(120, 120)

    def set_progress(self, progress: float):
        self._progress = max(0.0, min(1.0, progress))
        self.update()

    def set_color(self, color: str):
        self._color = QColor(color)
        self.update()

    def set_text(self, text: str):
        self._text = text
        self.update()

    def set_subtext(self, subtext: str):
        self._subtext = subtext
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        side = min(w, h)
        cx = w / 2
        cy = h / 2
        radius = side / 2 - 12

        rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)

        # 背景环
        bg_pen = QPen(QColor("#2D3548"), 6, Qt.SolidLine, Qt.RoundCap)
        p.setPen(bg_pen)
        p.drawArc(rect, 90 * 16, -360 * 16)

        # 进度环
        if self._progress > 0.001:
            fg_pen = QPen(self._color, 6, Qt.SolidLine, Qt.RoundCap)
            p.setPen(fg_pen)
            p.drawArc(rect, 90 * 16, -int(360 * 16 * self._progress))

        # 中心文本
        if self._text:
            p.setPen(QColor("#E8EDF3"))
            font = QFont("Microsoft YaHei UI", 16, QFont.Bold)
            p.setFont(font)
            text_rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
            p.drawText(text_rect, Qt.AlignCenter, self._text)

        # 子文本
        if self._subtext:
            p.setPen(QColor("#6B7689"))
            font = QFont("Microsoft YaHei UI", 9)
            p.setFont(font)
            sub_rect = QRectF(cx - radius, cy + radius * 0.15, radius * 2, radius * 0.5)
            p.drawText(sub_rect, Qt.AlignCenter, self._subtext)
