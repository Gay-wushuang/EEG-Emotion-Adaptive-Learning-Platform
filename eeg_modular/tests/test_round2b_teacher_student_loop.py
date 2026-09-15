"""Round 2B teacher binding, assignment and real runtime observation tests."""

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

from PySide6.QtWidgets import QApplication, QMessageBox
from pages.task_page import TaskPage
from services.dashboard_state import DashboardState
from services.identity_store import IdentityStore
from services.teaching_store import AssignmentStore, StudentRuntimeRegistry, TeacherStudentStore


class Service:
    def __init__(self, state): self.state = state
    def start_session(self): return self.state.begin_session(source="live", demo=False)
    def end_session(self, status="completed"): return self.state.finalize_session(status=status)
    def pause_session(self): self.state.set_session_paused(True)
    def resume_session(self): self.state.set_session_paused(False)


class Round2BTeacherStudentLoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.root = root
        self.assignments = AssignmentStore(root / "assignments.json")
        self.identities = IdentityStore(root / "identities.json")
        for uid, name in (("teacher_001", "教师一"), ("teacher_002", "教师二"),
                          ("st_001", "学生一"), ("st_002", "学生二")):
            self.identities.save_profile(uid, name, make_current=False)
        self.bindings = TeacherStudentStore(
            root / "bindings.json", identity_store=self.identities
        )
        self.registry = StudentRuntimeRegistry(root / "coordination.sqlite3")

    def tearDown(self): self.temp.cleanup()

    def page(self, user_id, role):
        state = DashboardState(
            sessions_dir=self.root / f"sessions-{user_id}", seed_demo_history=False
        )
        state._user_id, state._user_name = user_id, user_id
        state._user_role = state.current_role = role
        page = TaskPage(
            state, Service(state), binding_store=self.bindings,
            assignment_store=self.assignments, runtime_registry=self.registry,
            identity_store=self.identities,
        )
        return state, page

    def test_teacher_draft_survives_realtime_refresh(self):
        state, page = self.page("teacher_001", "teacher")
        page._combo_task.setCurrentText("英语阅读")
        page._combo_diff.setCurrentIndex(2)
        page._input_note.setText("第 3 章")
        for _ in range(3): page.update_state(state)
        self.assertEqual(page._combo_task.currentText(), "英语阅读")
        self.assertEqual(page._combo_diff.currentIndex(), 2)
        self.assertEqual(page._input_note.text(), "第 3 章")

    def test_bindings_are_private_and_removal_does_not_delete_identity(self):
        self.assertTrue(self.bindings.add("teacher_001", "st_001"))
        self.assertEqual(self.bindings.students_for("teacher_001"), ["st_001"])
        self.assertEqual(self.bindings.students_for("teacher_002"), [])
        self.assertTrue(self.bindings.remove("teacher_001", "st_001"))
        self.assertIsNotNone(self.identities.get_profile("st_001"))

    def test_publish_requires_selected_bound_student_and_stays_visible(self):
        state, page = self.page("teacher_001", "teacher")
        with patch.object(QMessageBox, "warning") as warning:
            page._publish_teacher_task()
        warning.assert_called_once()
        self.assertEqual(self.assignments.list_all(), [])
        self.bindings.add("teacher_001", "st_001")
        page._refresh_teacher_students()
        page._combo_student.setCurrentIndex(page._combo_student.findData("st_001"))
        page._combo_task.setCurrentText("英语阅读")
        page._combo_diff.setCurrentIndex(2)
        page._input_note.setText("完成阅读题")
        page._publish_teacher_task()
        shown = page._assignment_summary.text()
        page.update_state(state)
        self.assertEqual(page._assignment_summary.text(), shown)
        self.assertIn("st_001", shown)
        self.assertIn("等待学生开始", shown)

    def test_students_only_receive_their_assignments(self):
        item = self.assignments.publish("teacher_001", "st_001", "英语阅读", "hard", "阅读")
        state1, page1 = self.page("st_001", "student")
        state2, page2 = self.page("st_002", "student")
        self.assertEqual(page1._selected_assignment_id, item.assignment_id)
        self.assertEqual(state1.task_type, "英语阅读")
        self.assertEqual(page2._selected_assignment_id, "")

    def test_student_execution_links_assignment_and_updates_status(self):
        item = self.assignments.publish("teacher_001", "st_001", "数学练习", "medium")
        state, page = self.page("st_001", "student")
        page._on_task_start()
        task = state._task_by_id(state.current_task_id)
        self.assertEqual(task.assignment_id, item.assignment_id)
        self.assertEqual(self.assignments.for_student("st_001")[-1].status, "running")
        page._on_task_pause_resume()
        self.assertEqual(self.assignments.for_student("st_001")[-1].status, "paused")
        page._on_task_pause_resume()
        page._on_task_stop()
        self.assertEqual(self.assignments.for_student("st_001")[-1].status, "completed")
        self.assertEqual(
            state.history_sessions[-1].tasks[0].assignment_id,
            item.assignment_id,
        )

    def test_teacher_reads_real_student_runtime_and_writes_real_event(self):
        self.bindings.add("teacher_001", "st_001")
        student_state, student_page = self.page("st_001", "student")
        student_page._on_task_start()
        student_state.device_status = "online"
        student_state.quality_level = "trusted"
        student_state.attention, student_state.meditation = 61, 48
        self.registry.publish(student_state)
        teacher_state, teacher_page = self.page("teacher_001", "teacher")
        teacher_page._combo_student.setCurrentIndex(
            teacher_page._combo_student.findData("st_001")
        )
        teacher_page._refresh_teacher_observation()
        self.assertIn(student_state.current_task_id, self.registry.get("st_001")["task_id"])
        teacher_page._add_quick_event("疑似走神", "teacher")
        event = student_state._events[-1]
        self.assertEqual(event.student_id, "st_001")
        self.assertEqual(event.observer_id, "teacher_001")
        self.assertEqual(event.session_id, student_state.run_id)
        self.assertEqual(event.task_id, student_state.current_task_id)
        student_page._on_task_stop()

    def test_teacher_cannot_create_event_without_active_student_task(self):
        self.bindings.add("teacher_001", "st_001")
        student_state, _ = self.page("st_001", "student")
        self.registry.publish(student_state)
        _, teacher_page = self.page("teacher_001", "teacher")
        teacher_page._combo_student.setCurrentIndex(teacher_page._combo_student.findData("st_001"))
        before = len(student_state._events)
        teacher_page._add_quick_event("教师干预", "teacher")
        self.assertEqual(len(student_state._events), before)
        self.assertTrue(all(not button.isEnabled() for button in teacher_page._event_buttons))


if __name__ == "__main__":
    unittest.main()
