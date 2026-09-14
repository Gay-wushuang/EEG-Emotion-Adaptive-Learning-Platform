"""Role-oriented task/event workflow regressions."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui_prototype"))
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication
from pages.task_page import TaskPage
from services.dashboard_state import DashboardState, SESSION_IDLE, SESSION_PAUSED
from services.identity_store import IdentityStore
from services.teaching_store import AssignmentStore, StudentRuntimeRegistry, TeacherStudentStore
from smart_learning_app.live_service import LiveDataService


class StateService:
    def __init__(self, state):
        self.state = state

    def start_session(self):
        return self.state.begin_session(source="live", demo=False)

    def end_session(self, status="completed"):
        return self.state.finalize_session(status=status)

    def pause_session(self):
        self.state.set_session_paused(True)

    def resume_session(self):
        self.state.set_session_paused(False)


class RoleTaskEventWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_student_one_click_starts_session_task_and_system_event(self):
        with tempfile.TemporaryDirectory() as directory:
            state = DashboardState(sessions_dir=Path(directory), seed_demo_history=False)
            page = TaskPage(state, StateService(state))
            page.set_role("student")
            page._on_task_start()
            self.assertTrue(state.session_active)
            self.assertTrue(state.task_running)
            self.assertTrue(page._hierarchy_card.isHidden())
            event = next(e for e in state._events if e.type == "task_start")
            self.assertEqual(event.source, "system")
            page._on_task_stop()
            self.assertEqual(state.session_status, SESSION_IDLE)
            page.deleteLater()

    def test_student_pause_resume_keeps_task_and_adds_system_events(self):
        state = DashboardState(seed_demo_history=False)
        page = TaskPage(state, StateService(state))
        page.set_role("student")
        page._on_task_start()
        task_id = state.current_task_id
        page._on_task_pause_resume()
        self.assertEqual(state.session_status, SESSION_PAUSED)
        self.assertEqual(state.current_task_id, task_id)
        page._on_task_pause_resume()
        self.assertEqual(state.current_task_id, task_id)
        self.assertEqual(
            [e.source for e in state._events if e.type in {"pause", "resume"}],
            ["system", "system"],
        )
        page._on_task_stop()
        page.deleteLater()

    def test_self_report_is_typed_and_linked_to_current_task(self):
        state = DashboardState(seed_demo_history=False)
        page = TaskPage(state, StateService(state))
        page.set_role("student")
        page._on_task_start()
        task_id = state.current_task_id
        page.submit_self_report({"fatigue": "较高", "difficulty": "适中"})
        event = state._events[-1]
        self.assertEqual(event.source, "self_report")
        self.assertEqual(event.type, "self_report")
        self.assertEqual(event.student_id, state._user_id)
        self.assertEqual(event.task_id, task_id)
        page._on_task_stop()
        page.deleteLater()

    def test_teacher_observation_has_full_actor_and_scope_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = StudentRuntimeRegistry(root / "coordination.sqlite3")
            assignments = AssignmentStore(root / "assignments.json")
            identities = IdentityStore(root / "identities.json")
            identities.save_profile("st_01", "学生", make_current=False)
            bindings = TeacherStudentStore(
                root / "bindings.json", identity_store=identities
            )
            bindings.add("teacher_01", "st_01")
            student = DashboardState(seed_demo_history=False)
            student._user_id = "st_01"
            student.begin_session(source="live", demo=False)
            task_id = student.begin_task("数学练习")
            registry.publish(student)
            teacher = DashboardState(seed_demo_history=False)
            teacher._user_id, teacher.current_role = "teacher_01", "teacher"
            page = TaskPage(
                teacher, StateService(teacher), binding_store=bindings,
                assignment_store=assignments, runtime_registry=registry,
                identity_store=identities,
            )
            page._combo_student.setCurrentIndex(page._combo_student.findData("st_01"))
            page._add_quick_event("疑似走神", "teacher")
            event = student._events[-1]
            self.assertEqual(event.source, "teacher")
            self.assertEqual(event.observer_id, "teacher_01")
            self.assertEqual(event.student_id, "st_01")
            self.assertEqual(event.session_id, student.run_id)
            self.assertEqual(event.task_id, task_id)
            self.assertTrue(page._timer_card.isHidden())
            page.deleteLater()

    def test_live_status_transitions_create_signal_events_only_in_session(self):
        state = DashboardState(seed_demo_history=False)
        service = LiveDataService(state, ROOT / "production_baseline_v1")
        state.begin_session(source="live", demo=False)
        service._on_status({
            "connector_status": "online", "device_status": "online", "reason": ""
        })
        service._on_status({
            "connector_status": "online", "device_status": "offline", "reason": "lost"
        })
        service._on_status({
            "connector_status": "online", "device_status": "online", "reason": ""
        })
        transitions = [e for e in state._events if e.type.startswith("signal_")]
        self.assertEqual([e.type for e in transitions], ["signal_lost", "signal_recovered"])
        self.assertTrue(all(e.source == "system" for e in transitions))


if __name__ == "__main__":
    unittest.main()
