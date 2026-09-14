"""Round 2C data isolation, assignment queue and teacher dashboard tests."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui_prototype"))

from PySide6.QtWidgets import QApplication
from pages.dashboard_page import DashboardPage
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


class Round2CIsolationQueueDashboardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.identities = IdentityStore(root / "identities.json")
        for uid, name in (("teacher_01", "教师"), ("teacher_02", "教师二"),
                          ("st_001", "学生一"), ("st_002", "学生二")):
            self.identities.save_profile(uid, name, make_current=False)
        self.bindings = TeacherStudentStore(
            root / "bindings.json", identity_store=self.identities
        )
        self.assignments = AssignmentStore(root / "assignments.json")
        self.registry = StudentRuntimeRegistry(root / "coordination.sqlite3")
        self.root = root

    def tearDown(self): self.temp.cleanup()

    def state(self, uid, role):
        value = DashboardState(sessions_dir=self.root / f"sessions-{uid}", seed_demo_history=False)
        value._user_id = value._user_name = uid
        value._user_role = value.current_role = role
        return value

    def task_page(self, uid, role):
        state = self.state(uid, role)
        return state, TaskPage(
            state, Service(state), identity_store=self.identities,
            binding_store=self.bindings, assignment_store=self.assignments,
            runtime_registry=self.registry,
        )

    def test_binding_requires_real_student_identity_and_deduplicates(self):
        with self.assertRaisesRegex(ValueError, "未找到该学生账号"):
            self.bindings.add("teacher_01", "st_missing")
        with self.assertRaisesRegex(ValueError, "该账号不是学生身份"):
            self.bindings.add("teacher_01", "teacher_02")
        self.assertTrue(self.bindings.add("teacher_01", "st_001"))
        self.assertFalse(self.bindings.add("teacher_01", "st_001"))
        self.assertEqual(self.bindings.students_for("teacher_01"), ["st_001"])
        self.assertEqual(self.bindings.students_for("teacher_02"), [])

    def test_assignment_queue_is_isolated_and_does_not_overwrite(self):
        first = self.assignments.publish("teacher_01", "st_001", "物理复习", "medium")
        second = self.assignments.publish("teacher_01", "st_001", "英语阅读", "hard")
        self.assignments.publish("teacher_01", "st_002", "数学练习", "easy")
        student_one = self.assignments.for_student("st_001")
        self.assertEqual([item.assignment_id for item in student_one],
                         [first.assignment_id, second.assignment_id])
        self.assertEqual(len(self.assignments.for_student("st_002")), 1)

    def test_logged_in_student_hot_refreshes_only_assignment_list(self):
        state, page = self.task_page("st_001", "student")
        self.assertEqual(page._combo_assignment.count(), 0)
        self.assignments.publish("teacher_01", "st_001", "物理复习", "medium")
        self.assignments.publish("teacher_01", "st_001", "英语阅读", "hard")
        page.update_state(state)
        self.assertEqual(page._combo_assignment.count(), 2)
        page._combo_assignment.setCurrentIndex(0)
        self.assertEqual(state.task_type, "物理复习")

    def test_teacher_dashboard_switches_entire_snapshot_and_never_uses_own_eeg(self):
        self.bindings.add("teacher_01", "st_001")
        self.bindings.add("teacher_01", "st_002")
        for uid, attention, task in (("st_001", 21, "物理复习"),
                                     ("st_002", 82, "英语阅读")):
            state = self.state(uid, "student")
            state.attention, state.meditation = attention, 50
            state.task_type = task
            state.begin_session(source="live", demo=False)
            state.begin_task(task)
            self.registry.publish(state)
        teacher = self.state("teacher_01", "teacher")
        teacher.attention = 99  # must never be rendered as a student's value
        page = DashboardPage(
            teacher, Service(teacher), identity_store=self.identities,
            binding_store=self.bindings, runtime_registry=self.registry,
        )
        page._teacher_student_combo.setCurrentIndex(
            page._teacher_student_combo.findData("st_001")
        )
        page.update_state(teacher)
        self.assertEqual(page._teacher_snapshot["attention"], 21)
        self.assertIn("物理复习", page._analysis_label.text())
        page._teacher_student_combo.setCurrentIndex(
            page._teacher_student_combo.findData("st_002")
        )
        page.update_state(teacher)
        self.assertEqual(page._teacher_snapshot["attention"], 82)
        self.assertIn("英语阅读", page._analysis_label.text())
        self.assertNotEqual(page._teacher_snapshot["attention"], teacher.attention)

    def test_inactive_student_disables_events_and_active_uses_real_scope(self):
        self.bindings.add("teacher_01", "st_001")
        student, _ = self.task_page("st_001", "student")
        self.registry.publish(student)
        _, teacher_page = self.task_page("teacher_01", "teacher")
        teacher_page._combo_student.setCurrentIndex(teacher_page._combo_student.findData("st_001"))
        self.assertTrue(all(not button.isEnabled() for button in teacher_page._event_buttons))
        student.begin_session(source="live", demo=False)
        task_id = student.begin_task("数学练习")
        self.registry.publish(student)
        teacher_page._refresh_teacher_observation()
        teacher_page._add_quick_event("环境干扰", "teacher")
        event = student._events[-1]
        self.assertEqual((event.student_id, event.session_id, event.task_id),
                         ("st_001", student.run_id, task_id))


if __name__ == "__main__": unittest.main()
