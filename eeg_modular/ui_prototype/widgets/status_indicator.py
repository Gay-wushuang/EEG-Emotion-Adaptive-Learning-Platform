"""状态指示器：彩色圆点 + 文本。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel


class StatusDot(QWidget):
    """纯色圆点。"""

    def __init__(self, color: str = "#6B7689", size: int = 10, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._size = size
        self.setFixedSize(size + 2, size + 2)

    def set_color(self, color: str):
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(self._color)
        p.setPen(Qt.NoPen)
        p.drawEllipse(1, 1, self._size, self._size)


class StatusIndicator(QWidget):
    """状态指示器：圆点 + 标签 + 状态文本。"""

    LEVEL_GOOD = ("#4ADE80", "StatusGood")
    LEVEL_WARN = ("#FBBF24", "StatusWarning")
    LEVEL_ERROR = ("#F87171", "StatusError")
    LEVEL_NEUTRAL = ("#6B7689", "StatusNeutral")

    def __init__(self, label: str = "", parent=None):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)

        self._dot = StatusDot("#6B7689")
        self._layout.addWidget(self._dot)

        if label:
            self._label = QLabel(label)
            self._label.setObjectName("DimLabel")
            self._label.setStyleSheet("font-size: 13px;")
            self._layout.addWidget(self._label)
        else:
            self._label = None

        self._value = QLabel("未连接")
        self._value.setObjectName("StatusNeutral")
        self._value.setStyleSheet("font-size: 13px; font-weight: 600;")
        self._layout.addWidget(self._value)
        self._layout.addStretch()

    def set_state(self, level: tuple, text: str):
        color, obj_name = level
        self._dot.set_color(color)
        self._value.setText(text)
        self._value.setObjectName(obj_name)
        self._value.style().unpolish(self._value)
        self._value.style().polish(self._value)

    def set_text(self, text: str):
        self._value.setText(text)
