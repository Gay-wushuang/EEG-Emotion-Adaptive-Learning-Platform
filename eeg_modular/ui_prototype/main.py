"""智学脑机助手 - 桌面应用入口。

运行方式：
    cd ui_prototype
    python main.py                 # Mock 模式（默认）
    python main.py --mode mock     # 显式 Mock 模式
    python main.py --mode live     # Live 模式（真实 ThinkGear + Production Baseline v1）
    python main.py --skip-login --role student --user-id test_01 --auto-exit-ms 3000

Mock 模式默认启用，无需连接真实设备。
正常启动只显示本地账号登录，使用端由账号 ID 前缀固定；--skip-login 仅供测试/开发。
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

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QDialog

from login_dialog import LoginDialog
from main_window import MainWindow
from services.font_loader import ensure_chinese_font
from services.identity_store import (
    IdentityStore, ROLE_RESEARCH, VALID_ROLES, role_for_user_id,
)


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
    parser.add_argument(
        "--skip-login",
        action="store_true",
        help="跳过登录对话框（仅用于自动测试或本地开发）",
    )
    parser.add_argument(
        "--role",
        choices=VALID_ROLES,
        default=ROLE_RESEARCH,
        help="兼容参数；实际角色始终由 --user-id 前缀确定",
    )
    parser.add_argument(
        "--user-id",
        default="admin_test",
        help="--skip-login 时注入的本地用户 ID",
    )
    parser.add_argument(
        "--user-name",
        default="测试用户",
        help="--skip-login 时注入的显示名称",
    )
    parser.add_argument(
        "--auto-exit-ms",
        type=int,
        default=0,
        help="窗口启动后自动退出的毫秒数（仅用于自动验收）",
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

    identity_store = IdentityStore()
    if args.skip_login:
        try:
            user_id, user_name = identity_store.validate(args.user_id, args.user_name)
        except ValueError as exc:
            print(f"Invalid test identity: {exc}", file=sys.stderr)
            return 2
        identity_kwargs = {
            "user_id": user_id,
            "user_name": user_name,
            "role": role_for_user_id(user_id),
        }
    else:
        login = LoginDialog(identity_store)
        if login.exec() != QDialog.DialogCode.Accepted:
            return 0
        identity_kwargs = {
            "user_id": login.selected_user_id,
            "user_name": login.selected_user_name,
            "role": login.selected_role,
        }

    # 根据模式创建主窗口
    if args.mode == "live":
        window = MainWindow(
            mode="live",
            package_dir=args.package_dir,
            **identity_kwargs,
        )
    else:
        window = MainWindow(mode="mock", **identity_kwargs)

    def switch_identity():
        dialog = LoginDialog(identity_store, window)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            window.set_identity(
                dialog.selected_user_id,
                dialog.selected_user_name,
                dialog.selected_role,
            )

    window.identity_switch_requested.connect(switch_identity)

    window.show()
    if args.auto_exit_ms > 0:
        QTimer.singleShot(args.auto_exit_ms, app.quit)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
