"""Round 2D SQLite runtime coordination and UI regressions."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui_prototype"))

from PySide6.QtWidgets import QApplication
from main_window import MainWindow
from pages.task_page import TaskPage
from services.dashboard_state import DashboardState
from services.identity_store import IdentityStore
from services.teaching_store import (
    AssignmentStore, StudentRuntimeRegistry, TeacherObserverService,
    TeacherStudentStore,
)


class Service:
    def __init__(self, state): self.state = state
    def start_session(self): return self.state.begin_session(source="live", demo=False)
    def end_session(self, status="completed"): return self.state.finalize_session(status=status)
    def pause_session(self): self.state.set_session_paused(True)
    def resume_session(self): self.state.set_session_paused(False)


class Round2DCrossProcessRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_separate_registry_instances_share_sqlite_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coordination.sqlite3"
            writer = StudentRuntimeRegistry(path)
            reader = StudentRuntimeRegistry(path)
            state = DashboardState(seed_demo_history=False)
            state._user_id, state._user_name = "st_001", "学生一"
            state.device_status = state.connector_status = "online"
            state.attention, state.meditation = 63, 41
            state.begin_session(source="live", demo=False)
            task_id = state.begin_task("英语阅读", assignment_id="A001")
            writer.publish(state, force=True)
            snapshot = reader.get("st_001")
            self.assertEqual(snapshot["task_id"], task_id)
            self.assertEqual(snapshot["assignment_id"], "A001")
            self.assertEqual(snapshot["attention"], 63)
            self.assertGreater(snapshot["updated_at"], 0)

    def test_students_are_isolated_and_stale_snapshot_becomes_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coordination.sqlite3"
            writer = StudentRuntimeRegistry(path, stale_seconds=0.03)
            reader = StudentRuntimeRegistry(path, stale_seconds=0.03)
            for uid, attention in (("st_001", 20), ("st_002", 80)):
                state = DashboardState(seed_demo_history=False)
                state._user_id, state.attention = uid, attention
                state.device_status = "online"
                writer.publish(state, force=True)
            self.assertEqual(reader.get("st_001")["attention"], 20)
            self.assertEqual(reader.get("st_002")["attention"], 80)
            time.sleep(0.05)
            stale = reader.get("st_001")
            self.assertTrue(stale["stale"])
            self.assertFalse(stale["online"])

    def test_cross_process_teacher_event_is_delivered_to_student_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coordination.sqlite3"
            student_registry = StudentRuntimeRegistry(path)
            teacher_registry = StudentRuntimeRegistry(path)
            state = DashboardState(seed_demo_history=False)
            state._user_id = "st_001"
            state.begin_session(source="live", demo=False)
            task_id = state.begin_task("数学练习")
            student_registry.publish(state, force=True)
            event_id = teacher_registry.add_teacher_event(
                "st_001", "teacher_01", "教师干预"
            )
            self.assertTrue(event_id)
            student_registry.publish(state, force=True)
            event = state._events[-1]
            self.assertEqual((event.observer_id, event.task_id), ("teacher_01", task_id))

    def test_student_assignment_combo_is_readable_and_hot_refreshes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identities = IdentityStore(root / "identities.json")
            identities.save_profile("st_001", "学生", make_current=False)
            bindings = TeacherStudentStore(root / "bindings.json", identity_store=identities)
            assignments = AssignmentStore(root / "assignments.json")
            registry = StudentRuntimeRegistry(root / "coordination.sqlite3")
            state = DashboardState(sessions_dir=root / "sessions", seed_demo_history=False)
            state._user_id, state.current_role = "st_001", "student"
            page = TaskPage(state, Service(state), identity_store=identities,
                            binding_store=bindings, assignment_store=assignments,
                            runtime_registry=registry)
            for task in ("物理复习", "英语阅读", "数学练习"):
                assignments.publish("teacher_01", "st_001", task, "medium")
            page.update_state(state)
            self.assertEqual(page._combo_assignment.count(), 3)
            self.assertGreaterEqual(page._combo_assignment.minimumHeight(), 42)
            self.assertGreaterEqual(page._combo_assignment.minimumWidth(), 320)

    def test_teacher_window_is_passive_and_hidden_event_button_is_not_window(self):
        window = MainWindow(
            mode="live", user_id="teacher_css", user_name="教师", role="teacher"
        )
        self.addCleanup(window.close)
        self.assertIsInstance(window.service, TeacherObserverService)
        dashboard = window._pages["dashboard"]
        self.assertFalse(dashboard._btn_event.isWindow())
        self.assertFalse(dashboard._btn_manage_students.isHidden())


if __name__ == "__main__": unittest.main()
