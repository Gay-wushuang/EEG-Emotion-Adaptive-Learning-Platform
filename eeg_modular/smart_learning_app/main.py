"""Stable desktop entry point while the accepted UI is migrated into product code."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOTYPE = ROOT / "ui_prototype"


def main() -> None:
    # The UI prototype is an accepted presentation layer. Business integration
    # remains in smart_learning_app and must not import the prototype mock service.
    sys.path.insert(0, str(PROTOTYPE))
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from main import load_stylesheet
    from main_window import MainWindow
    from services.font_loader import ensure_chinese_font
    from smart_learning_app.live_service import LiveDataService

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("智学脑机助手")
    ensure_chinese_font()
    load_stylesheet(app)
    package_dir = ROOT / "production_baseline_v1"
    window = MainWindow(
        service_factory=lambda state: LiveDataService(state, package_dir)
    )
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
