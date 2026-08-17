"""三分类概率条组件。

双层结果展示：
  - 模型三分类概率（positive/neutral/negative）通过彩色条形展示
  - 信号可信度独立指示，不合格时灰显并提示
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QRectF, QSize
from PySide6.QtGui import QPainter, QColor, QPen, QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel

from services.dashboard_state import CLASS_DISPLAY


class ProbabilityBar(QWidget):
    """单条水平概率条。"""

    COLORS = {
        "positive": QColor("#5B8DEF"),
        "neutral": QColor("#8EA3BF"),
        "negative": QColor("#6F82A0"),
    }
    LABELS = CLASS_DISPLAY

    def __init__(self, class_name: str, parent=None):
        super().__init__(parent)
        self._class_name = class_name
        self._value = 0.0
        self._dimmed = False
        # Keep the three-class panel usable on 1366x768 competition displays.
        self.setFixedHeight(28)
        self.setMinimumWidth(100)

    def set_value(self, value: float):
        self._value = max(0.0, min(1.0, value))
        self.update()

    def set_dimmed(self, dimmed: bool):
        self._dimmed = dimmed
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        bar_h = 8
        bar_y = h - bar_h - 2

        # 标签
        font = QFont("Microsoft YaHei UI", 10)
        p.setFont(font)
        color = self.COLORS[self._class_name]
        if self._dimmed:
            color = QColor("#4A5263")
        # Category identity is encoded by the bar fill, not by text color.
        # Keeping labels uniform prevents them looking like links/status alerts.
        p.setPen(QColor("#778397") if self._dimmed else QColor("#A5B0C0"))
        label = self.LABELS[self._class_name]
        p.drawText(QRectF(0, 0, 60, 18), Qt.AlignLeft | Qt.AlignVCenter, label)

        # 数值
        p.setPen(QColor("#778397") if self._dimmed else QColor("#E7ECF3"))
        val_text = f"{self._value * 100:.1f}%"
        p.drawText(QRectF(60, 0, w - 60, 18), Qt.AlignRight | Qt.AlignVCenter, val_text)

        # 背景轨道
        track_rect = QRectF(0, bar_y, w, bar_h)
        p.setBrush(QColor("#1A1F2E"))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(track_rect, 6, 6)

        # 填充
        if self._value > 0.001:
            fill_w = max(2, w * self._value)
            fill_rect = QRectF(0, bar_y, fill_w, bar_h)
            p.setBrush(color)
            p.drawRoundedRect(fill_rect, 6, 6)


class ProbabilityPanel(QWidget):
    """三分类概率面板 + 信号可信度独立指示。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        bars_row = QHBoxLayout()
        bars_row.setContentsMargins(0, 0, 0, 0)
        bars_row.setSpacing(10)
        self._bars = {}
        for cls in ["positive", "neutral", "negative"]:
            bar = ProbabilityBar(cls)
            bars_row.addWidget(bar, 1)
            self._bars[cls] = bar
        layout.addLayout(bars_row)

        # 信号可信度独立行
        self._confidence_label = QLabel("信号质量：--")
        self._confidence_label.setObjectName("DimLabel")
        self._confidence_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(self._confidence_label)

        # 不合格提示
        self._warning_label = QLabel("当前信号质量不足，暂不进行状态解释")
        self._warning_label.setObjectName("WarnLabel")
        self._warning_label.setStyleSheet(
            "font-size: 10px; padding: 2px 6px; border-radius: 4px; "
            "background-color: rgba(200,150,40,0.1);"
        )
        self._warning_label.setVisible(False)
        layout.addWidget(self._warning_label)

    def update_state(self, state):
        eligible = state.inference_eligible
        quality_level = state.quality_level
        quality_reasons = state.quality_reasons or []

        # 概率值可能为 None（尚未推理）
        self._bars["positive"].set_value(state.prob_positive or 0.0)
        self._bars["neutral"].set_value(state.prob_neutral or 0.0)
        self._bars["negative"].set_value(state.prob_negative or 0.0)

        for bar in self._bars.values():
            bar.set_dimmed(not eligible)

        # 信号质量文本（按 quality_level 决定颜色与文案）
        if quality_level == "trusted":
            conf_text = "信号质量：可信"
            self._confidence_label.setStyleSheet(
                "font-size: 10px; color: #8FA6C5;")
        elif quality_level == "warning":
            conf_text = "信号质量：警告"
            self._confidence_label.setStyleSheet(
                "font-size: 10px; color: #B9A77A;")
        else:  # rejected
            conf_text = "信号质量：暂不可用"
            self._confidence_label.setStyleSheet(
                "font-size: 10px; color: #8B98AA;")

        # 附加质量原因列表
        if quality_reasons:
            conf_text += f"（{', '.join(quality_reasons)}）"
        self._confidence_label.setText(conf_text)

        # 不合格提示：仅当质量被拒绝且预热完成时显示
        self._warning_label.setVisible(
            quality_level == "rejected" and state.warmup_complete
        )
