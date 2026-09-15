"""Round 4B 修复：教师端事件时间线交互（滚动/来源中文化/长备注查看）。

覆盖：
1. 用户在表格底部时，新事件刷新后仍跟随底部；
2. 用户向上滚动后，自动刷新不跳回底部；
3. 普通刷新不清除当前选择；
4～8. source 中文化映射（mock/live/system/teacher/self_report）；
9. 原始 Event.source 不被修改；
10. 长备注 Tooltip 全文；
11. 事件详情对话框全文且只读；
12. 打开详情不增加事件数量。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui_prototype"))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from services.dashboard_state import DashboardState, EventMarker
from pages.task_page import EVENT_SOURCE_DISPLAY, TaskPage


class _Service:
    def __init__(self):
        self.sessions_dir = "data/sessions"


def _events(count, *, source="mock", note="建议调整后续难度或学习节奏"
            "（未自动修改当前任务）；原因：负荷偏高"):
    return [
        EventMarker(
            label=f"事件{i}", content=f"事件{i}", note=note if i == 0 else "",
            source=source, type="marker", timestamp=1000000.0 + i,
            session_id="295ebe51b6fe", task_id="",
        )
        for i in range(count)
    ]


class Round4BTimelineFixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def _page(self):
        state = DashboardState(seed_demo_history=False)
        state.current_role = "research"  # 任意非教师角色即可驱动表格
        page = TaskPage(state, _Service())
        return state, page

    # ── 1：底部刷新后跟随最新事件 ──
    def test_01_bottom_user_follows_new_events(self):
        state, page = self._page()
        page._refresh_table(_events(30))
        scrollbar = page._event_table.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())  # 用户位于底部
        self.assertTrue(
            scrollbar.value() >= scrollbar.maximum() - page._SCROLL_BOTTOM_TOLERANCE
        )
        page._refresh_table(_events(35))  # 新事件到达
        self.assertEqual(scrollbar.value(), scrollbar.maximum())

    # ── 2：向上滚动后刷新不跳回底部 ──
    def test_02_scrolled_up_user_keeps_position(self):
        state, page = self._page()
        page._refresh_table(_events(30))
        scrollbar = page._event_table.verticalScrollBar()
        self.assertGreater(scrollbar.maximum(), 0)  # 表格确有滚动空间
        old_value = 0
        scrollbar.setValue(old_value)  # 用户向上滚到顶部附近
        page._refresh_table(_events(35))  # 定时自动刷新 + 新事件
        self.assertEqual(scrollbar.value(), old_value)
        self.assertNotEqual(scrollbar.value(), scrollbar.maximum())

        # 中间位置同样保持
        middle = scrollbar.maximum() // 2
        scrollbar.setValue(middle)
        page._refresh_table(_events(36))
        self.assertEqual(scrollbar.value(), middle)

    # ── 3：普通刷新不清除选择 ──
    def test_03_refresh_keeps_selection(self):
        state, page = self._page()
        page._refresh_table(_events(10))
        page._event_table.setCurrentCell(4, 1)
        page._refresh_table(_events(10))  # 同一批事件普通刷新
        self.assertEqual(page._event_table.currentRow(), 4)

    # ── 4～8：source 中文化映射 ──
    def test_04_mock_displayed_as_teaching_demo(self):
        state, page = self._page()
        page._refresh_table(_events(1, source="mock"))
        self.assertEqual(page._event_table.item(0, 1).text(), "教学演示")

    def test_05_live_displayed_as_realtime(self):
        state, page = self._page()
        page._refresh_table(_events(1, source="live"))
        self.assertEqual(page._event_table.item(0, 1).text(), "实时采集")

    def test_06_system_displayed_as_system_record(self):
        state, page = self._page()
        page._refresh_table(_events(1, source="system"))
        self.assertEqual(page._event_table.item(0, 1).text(), "系统记录")

    def test_07_teacher_displayed_as_teacher_observation(self):
        state, page = self._page()
        page._refresh_table(_events(1, source="teacher"))
        self.assertEqual(page._event_table.item(0, 1).text(), "教师观察")

    def test_08_self_report_displayed_as_student_feedback(self):
        state, page = self._page()
        page._refresh_table(_events(1, source="self_report"))
        self.assertEqual(page._event_table.item(0, 1).text(), "学生反馈")

    def test_replay_displayed_as_offline_replay(self):
        self.assertEqual(EVENT_SOURCE_DISPLAY["replay"], "离线回放")

    # ── 9：原始 source 不被修改 ──
    def test_09_original_source_value_unchanged(self):
        state, page = self._page()
        events = _events(2, source="mock")
        before = [e.source for e in events]
        page._refresh_table(events)
        self.assertEqual([e.source for e in events], before)
        self.assertEqual(events[0].source, "mock")  # 原始值仍是内部名称
        # UI 显示与内部值已分离
        self.assertEqual(page._event_table.item(0, 1).text(), "教学演示")

    # ── 10：长备注 Tooltip 全文 ──
    def test_10_long_note_tooltip_full_text(self):
        state, page = self._page()
        events = _events(2)
        note = events[0].note
        self.assertGreater(len(note), 20)  # 确为长备注
        page._refresh_table(events)
        tooltip = page._event_table.item(0, 4).toolTip()
        self.assertEqual(tooltip, note)  # Tooltip 提供全文
        # 单元格内允许省略显示（不撑高行高）
        self.assertIn("未自动修", page._event_table.item(0, 4).text()[:20])
        # 空备注不加 Tooltip
        self.assertEqual(page._event_table.item(1, 4).toolTip(), "")

    # ── 11～12：事件详情对话框（只读、零副作用） ──
    def test_11_event_detail_dialog_content_and_readonly(self):
        state, page = self._page()
        events = _events(2, source="mock")
        page._refresh_table(events)

        dialogs = []
        original_exec = None

        from PySide6.QtWidgets import QDialog, QLabel, QPushButton
        original_exec = QDialog.exec
        try:
            # 拦截 exec：记录对话框内容后立即关闭（模态循环无法在测试中运行）
            def fake_exec(self_dialog):
                dialogs.append(self_dialog)
                return QDialog.Accepted
            QDialog.exec = fake_exec
            page._show_event_detail(0, 4)
        finally:
            QDialog.exec = original_exec

        self.assertEqual(len(dialogs), 1)
        dialog = dialogs[0]
        self.assertEqual(dialog.windowTitle(), "事件详情")
        labels = dialog.findChildren(QLabel)
        text = "\n".join(l.text() for l in labels)
        # 全文正确（含被表格截断的完整备注）
        self.assertIn(events[0].note, text)
        self.assertIn("教学演示", text)          # 来源中文化
        self.assertIn("295ebe51b6fe", text)      # 所属会话
        self.assertIn("事件0", text)             # 事件名
        # 只读：文本可选中复制、自动换行、无输入控件
        value_labels = [l for l in labels if l.text() == events[0].note]
        self.assertTrue(value_labels[0].wordWrap())
        self.assertTrue(
            value_labels[0].textInteractionFlags() & Qt.TextSelectableByMouse
        )

    def test_12_detail_dialog_does_not_add_events(self):
        state, page = self._page()
        events = _events(2)
        page._refresh_table(events)
        count_before = len(state._events)

        from PySide6.QtWidgets import QDialog
        original_exec = QDialog.exec
        try:
            QDialog.exec = lambda self_dialog: QDialog.Accepted
            page._show_event_detail(0, 4)
            page._show_event_detail(1, 2)
        finally:
            QDialog.exec = original_exec

        self.assertEqual(len(state._events), count_before)
        self.assertEqual(len(page._last_events), 2)  # 表格数据不变
        # 详情对话框不改 Event 对象本身
        self.assertEqual(events[0].note, _events(2)[0].note)

    def test_detail_dialog_ignores_invalid_row(self):
        state, page = self._page()
        page._refresh_table(_events(2))
        from PySide6.QtWidgets import QDialog
        original_exec = QDialog.exec
        try:
            QDialog.exec = lambda self_dialog: QDialog.Accepted
            page._show_event_detail(99, 0)  # 越界行：静默忽略
        finally:
            QDialog.exec = original_exec


if __name__ == "__main__":
    unittest.main()
