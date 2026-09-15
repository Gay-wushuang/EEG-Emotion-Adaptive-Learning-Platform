"""事件详情弹窗深色主题修复测试。

验证：复用项目现有深色弹窗主题（FEEDBACK_DIALOG_STYLE）、字段分层着色
（字段名灰蓝 #94A3B8 / 内容主文字 #F3F6FA）、长备注自动换行、按钮可用、
Demo/Live/教师观察/AI 建议事件均可显示、零数据副作用。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui_prototype"))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QPushButton

from services.dashboard_state import DashboardState, EventMarker
from pages.task_page import TaskPage


class Service:
    def __init__(self, state):
        self.state = state


def _event(content, *, source="mock", note="", type_="marker"):
    return EventMarker(
        label=content, content=content, note=note,
        source=source, type=type_, timestamp=1000000.0,
        session_id="295ebe51b6fe", task_id="",
    )


class EventDetailThemeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.state = DashboardState(seed_demo_history=False)
        self.state.current_role = "research"
        self.page = TaskPage(self.state, Service(self.state))

    def _capture_dialog(self, events, row=0):
        """拦截 exec 返回对话框实例，供断言样式与内容。"""
        dialogs = []
        original = QDialog.exec
        try:
            QDialog.exec = lambda self_d: (dialogs.append(self_d), QDialog.Accepted)[1]
            self.page._refresh_table(events)
            self.page._show_event_detail(row, 4)
        finally:
            QDialog.exec = original
        self.assertEqual(len(dialogs), 1)
        return dialogs[0]

    def test_01_dialog_uses_dark_theme(self):
        events = [_event("事件0", source="mock")]
        dialog = self._capture_dialog(events)
        # QDialog 级样式表复用项目现有深色主题（非空且含深色背景）
        stylesheet = dialog.styleSheet()
        self.assertTrue(stylesheet)
        self.assertIn("QDialog { background: #161D2A", stylesheet)
        self.assertIn("QPushButton", stylesheet)

    def test_02_field_caption_secondary_color(self):
        events = [_event("事件0")]
        dialog = self._capture_dialog(events)
        labels = dialog.findChildren(QLabel)
        captions = [l for l in labels if l.text() == "时间"]
        self.assertEqual(len(captions), 1)
        self.assertIn("#94A3B8", captions[0].styleSheet())  # 次级灰蓝

    def test_03_field_value_primary_color(self):
        events = [_event("AI建议：降低后续学习负荷",
                         source="live",
                         note="建议调整后续难度或学习节奏")]
        dialog = self._capture_dialog(events)
        labels = dialog.findChildren(QLabel)
        values = [l for l in labels if l.text() == "实时采集"]
        self.assertEqual(len(values), 1)
        self.assertIn("#F3F6FA", values[0].styleSheet())  # 主文字色

    def test_04_long_note_wraps(self):
        long_note = "建议调整后续学习难度或学习节奏" * 20
        events = [_event("事件0", note=long_note)]
        dialog = self._capture_dialog(events)
        labels = dialog.findChildren(QLabel)
        notes = [l for l in labels if l.text() == long_note]
        self.assertEqual(len(notes), 1)
        self.assertTrue(notes[0].wordWrap())  # 自动换行
        self.assertTrue(notes[0].textInteractionFlags() & Qt.TextSelectableByMouse)

    def test_05_close_button_works(self):
        events = [_event("事件0")]
        dialog = self._capture_dialog(events)
        buttons = dialog.findChildren(QPushButton)
        self.assertEqual(len(buttons), 1)
        self.assertEqual(buttons[0].text(), "关闭")
        buttons[0].click()  # 不抛异常即视为正常工作

    def test_06_demo_live_teacher_ai_events_display(self):
        # 真实 AI 干预事件形态：source="system" + type="intervention"
        cases = [
            (_event("演示事件", source="mock"), "教学演示"),
            (_event("实时事件", source="live"), "实时采集"),
            (_event("教师观察：走神", source="teacher"), "教师观察"),
            (_event("AI建议：降低后续学习负荷",
                    source="system", type_="intervention",
                    note="未自动修改当前任务"),
             "系统记录"),
        ]
        for event, expected_source in cases:
            dialog = self._capture_dialog([event])
            texts = [l.text() for l in dialog.findChildren(QLabel)]
            self.assertIn(expected_source, texts)
            self.assertIn(event.label, texts)

    def test_07_no_event_data_modified(self):
        events = [_event("AI建议：降低后续学习负荷",
                         source="live", note="原始备注")]
        before = events[0].to_dict()
        count_before = len(self.state._events)
        self._capture_dialog(events)
        self.assertEqual(events[0].to_dict(), before)
        self.assertEqual(len(self.state._events), count_before)


if __name__ == "__main__":
    unittest.main()
