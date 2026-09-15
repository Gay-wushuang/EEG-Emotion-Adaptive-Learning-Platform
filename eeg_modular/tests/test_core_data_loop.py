"""Regression tests for the Session -> Task -> Event -> History lifecycle."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui_prototype"))

from services.dashboard_state import (
    DashboardState,
    MODEL_FAILED,
    MODEL_LOADING,
    MODEL_READY,
    SESSION_IDLE,
    SESSION_PAUSED,
    SESSION_RUNNING,
)


class CoreDataLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_task_and_event_survive_session_history_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            state = DashboardState(
                sessions_dir=Path(directory), seed_demo_history=False,
            )
            state.begin_session(source="live", demo=False)
            self.assertEqual(state.session_status, SESSION_RUNNING)

            task_id = state.begin_task("数学练习", "medium", "第一组")
            state.add_event("完成第 1 题", "user", task_id=task_id)
            state.end_task()
            metadata_path = state.finalize_session(status="completed")

            self.assertTrue(Path(metadata_path).is_file())
            self.assertEqual(state.session_status, SESSION_IDLE)
            history = state.reload_history()
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0].tasks[0].task_id, task_id)
            self.assertEqual(history[0].tasks[0].status, "completed")
            self.assertTrue(any(
                event.content == "完成第 1 题" and event.task_id == task_id
                for event in history[0].events
            ))

    def test_pausing_session_does_not_interrupt_current_task(self):
        with tempfile.TemporaryDirectory() as directory:
            state = DashboardState(
                sessions_dir=Path(directory), seed_demo_history=False,
            )
            state.begin_session(source="live", demo=False)
            task_id = state.begin_task("英语阅读")

            state.set_session_paused(True)
            self.assertTrue(state.session_active)
            self.assertEqual(state.session_status, SESSION_PAUSED)
            self.assertTrue(state.task_running)
            self.assertEqual(state.current_task_id, task_id)

            state.set_session_paused(False)
            self.assertEqual(state.session_status, SESSION_RUNNING)
            self.assertTrue(state.task_running)
            self.assertEqual(state.current_task_id, task_id)

    def test_model_failure_is_sticky_until_explicit_reload_or_ready(self):
        state = DashboardState(seed_demo_history=False)
        self.assertEqual(state.model_status, MODEL_LOADING)
        state.prob_positive = 0.8
        state.predicted_state = "positive"
        state.set_pipeline_error("模型不可用", "checksum mismatch")
        self.assertEqual(state.model_status, MODEL_FAILED)
        self.assertIsNone(state.prob_positive)
        self.assertIsNone(state.predicted_state)

        state.set_model_loading()
        self.assertEqual(state.model_status, MODEL_LOADING)
        state.set_model_ready()
        self.assertEqual(state.model_status, MODEL_READY)

    def test_task_auto_created_session_closes_and_refreshes_history(self):
        from PySide6.QtWidgets import QMessageBox
        from pages.task_page import TaskPage

        with tempfile.TemporaryDirectory() as directory:
            state = DashboardState(
                sessions_dir=Path(directory), seed_demo_history=False,
            )

            class Service:
                def start_session(self):
                    return state.begin_session(source="live", demo=False)

                def end_session(self, status="completed"):
                    return state.finalize_session(status=status)

            page = TaskPage(state, Service())
            with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
                page._on_task_start()
            task_id = state.current_task_id
            self.assertTrue(task_id)
            self.assertEqual(state.session_status, SESSION_RUNNING)
            time.sleep(0.02)

            page._on_task_stop()

            self.assertEqual(state.session_status, SESSION_IDLE)
            self.assertFalse(state.task_running)
            self.assertEqual(len(state.history_sessions), 1)
            record = state.history_sessions[0]
            self.assertEqual(record.status, "completed")
            self.assertGreater(record.duration_seconds, 0.0)
            self.assertEqual(record.tasks[0].task_id, task_id)
            self.assertEqual(record.tasks[0].status, "completed")
            self.assertTrue(any(event.type == "task_end" for event in record.events))
            self.assertTrue(any(event.type == "session_end" for event in record.events))
            page.deleteLater()

    def test_explicit_session_is_not_closed_when_one_task_ends(self):
        from pages.task_page import TaskPage

        with tempfile.TemporaryDirectory() as directory:
            state = DashboardState(
                sessions_dir=Path(directory), seed_demo_history=False,
            )

            class Service:
                def start_session(self):
                    return state.begin_session(source="live", demo=False)

                def end_session(self, status="completed"):
                    return state.finalize_session(status=status)

            page = TaskPage(state, Service())
            page._on_start_session()
            page._on_task_start()
            page._on_task_stop()

            self.assertEqual(state.session_status, SESSION_RUNNING)
            self.assertTrue(state.session_active)
            self.assertEqual(len(state.history_sessions), 0)
            state.finalize_session(status="completed")
            page.deleteLater()


if __name__ == "__main__":
    unittest.main()
