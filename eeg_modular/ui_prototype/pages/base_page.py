"""页面基类。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea,
)


class BasePage(QWidget):
    """所有页面的基类，提供标题栏和统一布局。

    Args:
        title: 页面标题（可选）。
        description: 页面副标题（可选）。
        parent: 父控件。
        scrollable: 是否使用 QScrollArea 包裹内容区域，
                    用于 1366×768 等低分辨率屏幕下的滚动支持。
    """

    def __init__(
        self,
        title: str = "",
        description: str = "",
        parent: QWidget | None = None,
        scrollable: bool = False,
    ):
        super().__init__(parent)
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(24, 20, 24, 20)
        self._main_layout.setSpacing(16)

        # 标题栏
        if title:
            header = QVBoxLayout()
            header.setSpacing(4)
            self._title_label = QLabel(title)
            self._title_label.setObjectName("PageTitle")
            header.addWidget(self._title_label)

            if description:
                self._desc_label = QLabel(description)
                self._desc_label.setObjectName("PageDescription")
                header.addWidget(self._desc_label)

            # 分隔线
            sep = QFrame()
            sep.setObjectName("HSeparator")
            sep.setFrameShape(QFrame.HLine)
            header.addSpacing(8)
            header.addWidget(sep)

            self._main_layout.addLayout(header)

        # 内容容器（可选滚动区域）
        if scrollable:
            self._scroll_area = QScrollArea()
            self._scroll_area.setWidgetResizable(True)
            self._scroll_area.setFrameShape(QFrame.NoFrame)
            self._scroll_area.setStyleSheet(
                "QScrollArea { border: none; background: transparent; }"
            )
            self._content = QWidget()
            self._content_layout = QVBoxLayout(self._content)
            self._content_layout.setContentsMargins(0, 0, 0, 0)
            self._content_layout.setSpacing(14)
            self._scroll_area.setWidget(self._content)
            self._main_layout.addWidget(self._scroll_area, 1)
        else:
            self._content = QWidget()
            self._content_layout = QVBoxLayout(self._content)
            self._content_layout.setContentsMargins(0, 0, 0, 0)
            self._content_layout.setSpacing(14)
            self._main_layout.addWidget(self._content, 1)

    @property
    def content_layout(self):
        return self._content_layout

    def update_state(self, state):
        """子类重写：根据 DashboardState 刷新页面。"""
        pass

    def on_show(self):
        """子类重写：页面被切到前台时调用。"""
        pass

    def on_hide(self):
        """子类重写：页面被切走时调用。"""
        pass
