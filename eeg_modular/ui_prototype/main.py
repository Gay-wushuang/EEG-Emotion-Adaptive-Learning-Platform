"""智学脑机助手 - 桌面应用入口。

运行方式：
    cd ui_prototype
    python main.py

Mock模式默认启用，无需连接真实设备。
"""

from __future__ import annotations

import os
import sys

# 将当前目录加入 sys.path，确保包导入正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from main_window import MainWindow
from services.font_loader import ensure_chinese_font


def load_stylesheet(app: QApplication):
    qss_path = os.path.join(os.path.dirname(__file__), "resources", "theme.qss")
    if os.path.exists(qss_path):
        with open(qss_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())


def main():
    # 高DPI支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("智学脑机助手")

    # 显式注册中文字体，避免离屏测试或打包环境显示方框。
    ensure_chinese_font()

    load_stylesheet(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
