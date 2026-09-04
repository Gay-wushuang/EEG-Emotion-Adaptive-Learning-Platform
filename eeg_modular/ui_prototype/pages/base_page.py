"""页面基类。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea,
)


# 统一页面背景色（与 theme.qss 中 RootWidget 保持一致）
_PAGE_BG = "#111722"


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
        self._role = self._normalize_role(
            getattr(getattr(self, "state", None), "current_role", "research")
        )

        # 确保 BasePage 自身使用深色背景
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(_PAGE_BG))
        self.setPalette(pal)

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
            # 滚动区域背景：防止 viewport 默认显示白色
            self._scroll_area.setStyleSheet(
                f"QScrollArea {{ border: none; background: {_PAGE_BG}; }}"
                f"QScrollArea > QWidget > QWidget {{ background: {_PAGE_BG}; }}"
            )
            self._content = QWidget()
            # 内容容器背景色
            self._content.setAutoFillBackground(True)
            pal_c = self._content.palette()
            pal_c.setColor(QPalette.Window, QColor(_PAGE_BG))
            self._content.setPalette(pal_c)

            self._content_layout = QVBoxLayout(self._content)
            self._content_layout.setContentsMargins(0, 0, 0, 0)
            self._content_layout.setSpacing(14)
            self._scroll_area.setWidget(self._content)
            self._main_layout.addWidget(self._scroll_area, 1)
        else:
            self._content = QWidget()
            # 非滚动模式同样设置深色背景
            self._content.setAutoFillBackground(True)
            pal_c = self._content.palette()
            pal_c.setColor(QPalette.Window, QColor(_PAGE_BG))
            self._content.setPalette(pal_c)

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

    @staticmethod
    def _normalize_role(role: str | None) -> str:
        """将登录层可能使用的角色别名归一为页面使用的三种角色。"""
        value = str(role or "research").strip().lower()
        if value in {"student", "learner", "learning", "学习端", "学生"}:
            return "student"
        if value in {"teacher", "teaching", "教学端", "教师"}:
            return "teacher"
        return "research"

    def set_role(self, role: str):
        """设置页面角色；具体页面可重写并更新角色化内容。"""
        self._role = self._normalize_role(role)

    def on_show(self):
        """子类重写：页面被切到前台时调用。"""
        pass

    def on_hide(self):
        """子类重写：页面被切走时调用。"""
        pass
