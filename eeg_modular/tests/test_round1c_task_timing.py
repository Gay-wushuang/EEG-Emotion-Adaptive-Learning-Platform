"""Round 1C regressions for effective task time across session pauses."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui_prototype"))

from PySide6.QtWidgets import QApplication

from pages.task_page import TaskPage
from services.dashboard_state import DashboardState, SESSION_PAUSED, SESSION_RUNNING


class Round1CTaskTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_multiple_pauses_excluded_and_task_identity_events_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            state = DashboardState(
                sessions_dir=Path(directory), seed_demo_history=False,
            )
            state.begin_session(source="live", demo=False)
            task_id = state.begin_task("数学练习")
            task = state._tasks[0]
            task._running_since = 100.0
            state._session_running_since = 100.0
            state.add_event("暂停前事件", "user")

            with patch("services.dashboard_state.time.monotonic", return_value=105.0):
                state.set_session_paused(True)
            self.assertEqual(state.session_status, SESSION_PAUSED)
            self.assertTrue(state.task_running)
            self.assertEqual(state.current_task_id, task_id)
            self.assertAlmostEqual(state.current_task_elapsed_seconds, 5.0)

            state.add_event("暂停中事件", "user")
            with patch("services.dashboard_state.time.monotonic", return_value=999.0):
                self.assertAlmostEqual(state.current_task_elapsed_seconds, 5.0)

            with patch("services.dashboard_state.time.monotonic", return_value=200.0):
                state.set_session_paused(False)
            self.assertEqual(state.session_status, SESSION_RUNNING)
            self.assertEqual(state.current_task_id, task_id)

            with patch("services.dashboard_state.time.monotonic", return_value=207.0):
                state.set_session_paused(True)
            self.assertAlmostEqual(state.current_task_elapsed_seconds, 12.0)

            with patch("services.dashboard_state.time.monotonic", return_value=300.0):
                state.set_session_paused(False)
            state.add_event("恢复后事件", "user")
            with patch("services.dashboard_state.time.monotonic", return_value=303.0):
                state.end_task()

            self.assertEqual(len(state._tasks), 1)
            self.assertEqual(state._tasks[0].task_id, task_id)
            self.assertAlmostEqual(state._tasks[0].duration_seconds, 15.0)
            self.assertTrue(all(
                event.task_id == task_id
                for event in state._events
                if event.content in {"暂停前事件", "暂停中事件", "恢复后事件"}
            ))
            with patch("services.dashboard_state.time.monotonic", return_value=303.0):
                state.finalize_session(status="completed")
            history = state.reload_history()
            self.assertEqual(len(history), 1)
            self.assertEqual(len(history[0].tasks), 1)
            self.assertAlmostEqual(history[0].tasks[0].duration_seconds, 15.0)
            self.assertAlmostEqual(history[0].duration_seconds, 15.0)

    def test_task_page_timer_freezes_while_paused_and_resumes(self):
        state = DashboardState(seed_demo_history=False)
        state.begin_session(source="live", demo=False)
        state.begin_task("英语阅读")
        page = TaskPage(state, object())
        page._task_active = True
        task = state._tasks[0]
        task.duration_seconds = 323.0
        task._running_since = None
        state.set_session_paused(True)

        with patch("services.dashboard_state.time.monotonic", return_value=900.0):
            page.update_state(state)
        self.assertEqual(page._task_time.text(), "05:23")
        self.assertEqual(page._task_status.text(), "已暂停")

        with patch("services.dashboard_state.time.monotonic", return_value=1000.0):
            state.set_session_paused(False)
        with patch("services.dashboard_state.time.monotonic", return_value=1002.0):
            page.update_state(state)
        self.assertEqual(page._task_time.text(), "05:25")
        self.assertEqual(page._task_status.text(), "进行中")
        page.deleteLater()


if __name__ == "__main__":
    unittest.main()
