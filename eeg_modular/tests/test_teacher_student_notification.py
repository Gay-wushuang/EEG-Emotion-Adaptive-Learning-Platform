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
from services.notification_service import NotificationService
from services.teaching_store import (
    AssignmentStore, StudentRuntimeRegistry, TeacherSelectionContext,
    TeacherStudentStore,
)


class Service:
    def __init__(self, state, sessions_dir):
        self.state = state
        self.sessions_dir = str(sessions_dir)


class TeacherStudentNotificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.identities = IdentityStore(self.root / "identities.json")
        for uid, name, role in (("teacher_01", "教师一", "teacher"),
                                ("st_001", "学生一", "student"),
                                ("st_002", "学生二", "student")):
            self.identities.save_profile(uid, name, role, make_current=False)
        self.bindings = TeacherStudentStore(self.root / "bindings.json", identity_store=self.identities)
        self.bindings.add("teacher_01", "st_001")
        self.bindings.add("teacher_01", "st_002")
        self.registry = StudentRuntimeRegistry(self.root / "coordination.sqlite3", stale_seconds=3600)
        self.selection = TeacherSelectionContext(); self.selection.set_teacher("teacher_01")

    def tearDown(self):
        self.temp.cleanup()

    def state(self, uid, role):
        state = DashboardState(sessions_dir=self.root / "sessions", seed_demo_history=False)
        state._user_id = uid; state._user_name = uid
        state.current_role = state._user_role = role
        return state

    def test_service_enforces_binding_and_student_isolation(self):
        service = NotificationService(self.registry.path, binding_store=self.bindings)
        service.publish(teacher_id="teacher_01", student_id="st_001", label="疑似走神")
        self.assertEqual(service.take_pending("st_002"), [])
        self.assertEqual([item.label for item in service.take_pending("st_001")], ["疑似走神"])
        self.assertEqual(service.take_pending("st_001"), [])
        with self.assertRaises(PermissionError):
            service.publish(teacher_id="teacher_02", student_id="st_001", label="教师干预")

    def test_teacher_quick_event_creates_student_notification(self):
        student = self.state("st_001", "student")
        student.begin_session(source="live", demo=False); student.begin_task("英语阅读")
        self.registry.publish(student, force=True)
        teacher = self.state("teacher_01", "teacher")
        page = TaskPage(teacher, Service(teacher, self.root / "sessions"),
                        binding_store=self.bindings,
                        assignment_store=AssignmentStore(self.root / "assignments.json"),
                        runtime_registry=self.registry, identity_store=self.identities,
                        selection_context=self.selection)
        self.selection.select("st_001")
        page._add_quick_event("环境干扰", "teacher")
        inbox = NotificationService(self.registry.path).take_pending("st_001")
        self.assertEqual([item.label for item in inbox], ["环境干扰"])
        self.assertEqual(inbox[0].teacher_id, "teacher_01")

    def test_student_dashboard_consumes_only_own_notification(self):
        service = NotificationService(self.registry.path, binding_store=self.bindings)
        service.publish(teacher_id="teacher_01", student_id="st_001", label="教师干预")
        service.publish(teacher_id="teacher_01", student_id="st_002", label="提问 / 课堂事件")
        student = self.state("st_002", "student")
        page = DashboardPage(student, Service(student, self.root / "sessions"),
                             runtime_registry=self.registry, identity_store=self.identities,
                             binding_store=self.bindings, selection_context=self.selection)
        page._notification_sound.play = lambda: None
        page._poll_student_notifications()
        self.assertIn("提问 / 课堂事件", page._student_teacher_reminder.text())
        self.assertNotIn("教师干预", page._student_teacher_reminder.text())
        self.assertEqual([item.label for item in service.take_pending("st_001")], ["教师干预"])


if __name__ == "__main__":
    unittest.main()
