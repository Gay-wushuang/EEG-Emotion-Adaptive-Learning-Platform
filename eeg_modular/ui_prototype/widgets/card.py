"""可复用卡片容器。"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QWidget, QHBoxLayout


class Card(QFrame):
    """统一卡片容器，带标题和内容区。"""

    def __init__(self, title: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("Card")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 10, 12, 10)
        self._layout.setSpacing(6)

        if title:
            self._title_label = QLabel(title)
            self._title_label.setObjectName("CardTitle")
            self._layout.addWidget(self._title_label)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(6)
        self._layout.addWidget(self._content, 1)

    @property
    def content_layout(self):
        return self._content_layout

    def add_widget(self, widget: QWidget):
        self._content_layout.addWidget(widget)

    def set_title(self, title: str):
        if hasattr(self, "_title_label"):
            self._title_label.setText(title)
