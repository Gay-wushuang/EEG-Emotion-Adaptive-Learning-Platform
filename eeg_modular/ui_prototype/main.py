"""智学脑机助手 - 桌面应用入口。

运行方式：
    cd ui_prototype
    python main.py                 # Mock 模式（默认）
    python main.py --mode mock     # 显式 Mock 模式
    python main.py --mode live     # Live 模式（真实 ThinkGear + Production Baseline v1）

Mock 模式默认启用，无需连接真实设备。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 将当前目录和项目根目录加入 sys.path，确保包导入正确
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)  # ui_prototype 的父目录即 eeg_modular
sys.path.insert(0, _HERE)
sys.path.insert(0, _PROJECT_ROOT)

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from main_window import MainWindow
from services.font_loader import ensure_chinese_font


def parse_args():
    parser = argparse.ArgumentParser(description="智学脑机助手 - 桌面应用")
    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        default="mock",
        help="运行模式：mock（模拟数据，默认）或 live（真实设备）",
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=None,
        help="Production Baseline v1 包路径（仅 live 模式需要）",
    )
    return parser.parse_args()


def load_stylesheet(app: QApplication):
    qss_path = os.path.join(os.path.dirname(__file__), "resources", "theme.qss")
    if os.path.exists(qss_path):
        with open(qss_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())


def main():
    args = parse_args()

    # 高DPI支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("智学脑机助手")

    # 显式注册中文字体，避免离屏测试或打包环境显示方框。
    ensure_chinese_font()

    load_stylesheet(app)

    # 根据模式创建主窗口
    if args.mode == "live":
        window = MainWindow(mode="live", package_dir=args.package_dir)
    else:
        window = MainWindow(mode="mock")

    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
