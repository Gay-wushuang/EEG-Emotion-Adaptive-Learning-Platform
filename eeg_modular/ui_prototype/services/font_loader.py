"""Register a Chinese-capable Windows font for Qt, including offscreen tests."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


_REGISTERED_FAMILY: str | None = None


def ensure_chinese_font() -> str:
    """Return a usable Chinese font family and apply it to the application."""
    global _REGISTERED_FAMILY
    app = QApplication.instance()
    if app is None:
        raise RuntimeError("QApplication must exist before loading fonts")
    if _REGISTERED_FAMILY is None:
        candidates = (
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("C:/Windows/Fonts/simsun.ttc"),
        )
        for path in candidates:
            if not path.exists():
                continue
            font_id = QFontDatabase.addApplicationFont(str(path))
            if font_id < 0:
                continue
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                _REGISTERED_FAMILY = families[0]
                break
        if _REGISTERED_FAMILY is None:
            _REGISTERED_FAMILY = app.font().family()
    app.setFont(QFont(_REGISTERED_FAMILY, 10))
    return _REGISTERED_FAMILY
