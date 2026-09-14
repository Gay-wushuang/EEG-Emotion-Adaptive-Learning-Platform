"""Tests for local identity persistence and role-based UI navigation."""

from __future__ import annotations

import os
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "ui_prototype"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class IdentityStoreTest(unittest.TestCase):
    def test_account_prefix_fixes_role_and_rejects_role_change(self):
        from services.identity_store import (
            IdentityStore, ROLE_RESEARCH, ROLE_STUDENT, ROLE_TEACHER,
            role_for_user_id,
        )

        self.assertEqual(role_for_user_id("st_01"), ROLE_STUDENT)
        self.assertEqual(role_for_user_id("teacher_01"), ROLE_TEACHER)
        self.assertEqual(role_for_user_id("tc_01"), ROLE_TEACHER)
        self.assertEqual(role_for_user_id("admin_01"), ROLE_RESEARCH)
        with tempfile.TemporaryDirectory() as directory:
            store = IdentityStore(Path(directory) / "identities.json")
            with self.assertRaisesRegex(ValueError, "不能修改"):
                store.save_profile("st_01", "学生", ROLE_TEACHER)
        with self.assertRaisesRegex(ValueError, "必须以"):
            role_for_user_id("unknown_01")

    def test_noninteractive_startup_arguments(self):
        spec = importlib.util.spec_from_file_location(
            "ui_prototype_entry_for_test",
            UI_DIR / "main.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with patch.object(
            sys,
            "argv",
            [
                "main.py",
                "--skip-login",
                "--role",
                "teacher",
                "--user-id",
                "teacher_99",
                "--user-name",
                "验收教师",
                "--auto-exit-ms",
                "500",
            ],
        ):
            args = module.parse_args()
        self.assertTrue(args.skip_login)
        self.assertEqual(args.role, "teacher")
        self.assertEqual(args.user_id, "teacher_99")
        self.assertEqual(args.user_name, "验收教师")
        self.assertEqual(args.auto_exit_ms, 500)

    def test_create_reload_update_and_delete_profile(self):
        from services.identity_store import IdentityStore, ROLE_TEACHER

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identities.json"
            store = IdentityStore(path)
            store.save_profile("teacher_01", "张老师", ROLE_TEACHER)

            reloaded = IdentityStore(path)
            self.assertEqual(reloaded.last_user_id(), "teacher_01")
            self.assertEqual(
                reloaded.get_profile("teacher_01"),
                {
                    "user_id": "teacher_01",
                    "name": "张老师",
                    "last_role": ROLE_TEACHER,
                },
            )

            reloaded.save_profile("teacher_01", "李老师", ROLE_TEACHER)
            self.assertEqual(len(reloaded.list_profiles()), 1)
            self.assertEqual(reloaded.get_profile("teacher_01")["name"], "李老师")
            self.assertTrue(reloaded.delete_profile("teacher_01"))
            self.assertEqual(reloaded.list_profiles(), [])

    def test_corrupt_file_recovers_as_empty_store(self):
        from services.identity_store import IdentityStore

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identities.json"
            path.write_text("not valid json", encoding="utf-8")
            self.assertEqual(IdentityStore(path).list_profiles(), [])


class IdentityRoleUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as exc:
            raise unittest.SkipTest(f"PySide6 未安装：{exc}")
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def _window(self, role, user_id=None, name="测试用户"):
        from main_window import MainWindow
        from services.identity_store import ROLE_RESEARCH, ROLE_STUDENT

        if user_id is None:
            user_id = {
                ROLE_STUDENT: "st_test",
                "teacher": "teacher_test",
                ROLE_RESEARCH: "admin_test",
            }[role]

        window = MainWindow(
            mode="mock",
            user_id=user_id,
            user_name=name,
            role=role,
        )
        self.addCleanup(window.close)
        return window

    def test_login_dialog_derives_role_without_role_selection(self):
        from PySide6.QtWidgets import QDialog
        from login_dialog import LoginDialog
        from services.identity_store import IdentityStore, ROLE_TEACHER

        with tempfile.TemporaryDirectory() as directory:
            store = IdentityStore(Path(directory) / "identities.json")
            dialog = LoginDialog(store)
            self.addCleanup(dialog.close)
            dialog._input_user_id.setText("teacher_02")
            dialog._input_user_name.setText("王老师")
            dialog._finish_login()
            self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
            self.assertEqual(dialog.selected_role, ROLE_TEACHER)
            self.assertEqual(store.last_user_id(), "teacher_02")
            self.assertEqual(dialog._stack.count(), 1)

    def test_student_sees_complete_learning_flow_but_not_admin_pages(self):
        from services.identity_store import ROLE_STUDENT

        window = self._window(ROLE_STUDENT)
        visible = {key for key, button in window._nav_buttons.items() if not button.isHidden()}
        self.assertEqual(
            visible, {"welcome", "baseline", "dashboard", "task", "history"}
        )
        self.assertEqual(window.state._user_role, ROLE_STUDENT)

        before = window.stack.currentWidget()
        window._navigate_to("settings")
        self.assertIs(window.stack.currentWidget(), before)
        window._navigate_to("replay")
        self.assertIs(window.stack.currentWidget(), before)

    def test_teacher_sees_only_teaching_pages(self):
        from PySide6.QtWidgets import QMessageBox
        from services.identity_store import ROLE_TEACHER

        window = self._window(ROLE_TEACHER)
        visible = {key for key, button in window._nav_buttons.items() if not button.isHidden()}
        self.assertEqual(
            visible, {"welcome", "baseline", "dashboard", "task", "history"}
        )
        self.assertIs(window.stack.currentWidget(), window._pages["welcome"])
        before = window.stack.currentWidget()
        window._navigate_to("settings")
        self.assertIs(window.stack.currentWidget(), before)

        dashboard = window._pages["dashboard"]
        window._navigate_to("dashboard")
        dashboard._on_start()
        self.assertFalse(window.state.session_active)
        self.assertFalse(dashboard._btn_start.isVisible())

        baseline = window._pages["baseline"]
        window._navigate_to("baseline")
        baseline._start_baseline()
        self.assertFalse(baseline._baseline_active)
        self.assertFalse(baseline._btn_start.isVisible())

        task = window._pages["task"]
        window._navigate_to("task")
        task._on_start_session()
        task._on_task_start()
        self.assertFalse(window.state.session_active)
        self.assertFalse(window.state.task_running)
        self.assertFalse(task._btn_publish_task.isHidden())
        with patch.object(QMessageBox, "warning") as warning:
            task._publish_teacher_task()
        warning.assert_called_once()
        self.assertIsNone(task._teacher_assignment)

    def test_default_window_is_research_and_keeps_all_pages(self):
        from main_window import MainWindow
        from services.identity_store import ROLE_RESEARCH

        window = MainWindow(mode="mock")
        self.addCleanup(window.close)
        visible = {key for key, button in window._nav_buttons.items() if not button.isHidden()}
        self.assertEqual(visible, set(window._pages))
        self.assertEqual(window.state._user_role, ROLE_RESEARCH)

    def test_identity_can_switch_without_rebuilding_pages(self):
        from services.identity_store import ROLE_RESEARCH, ROLE_STUDENT

        window = self._window(ROLE_RESEARCH)
        original_pages = dict(window._pages)
        window.set_identity("st_03", "周同学", ROLE_STUDENT)

        self.assertEqual(window.state._user_id, "st_03")
        self.assertEqual(window.state._user_name, "周同学")
        self.assertEqual(window.state._user_role, ROLE_STUDENT)
        self.assertEqual(window._pages, original_pages)
        self.assertIs(window.stack.currentWidget(), window._pages["welcome"])

    def test_admin_student_admin_switch_recomputes_and_enforces_access(self):
        from services.identity_store import ROLE_RESEARCH, ROLE_STUDENT

        window = self._window(ROLE_RESEARCH, "admin_01", "管理员")
        window._navigate_to("settings")
        self.assertIs(window.stack.currentWidget(), window._pages["settings"])

        window.set_identity("st_01", "学生", ROLE_STUDENT)
        self.assertIs(window.stack.currentWidget(), window._pages["welcome"])
        self.assertTrue(window._nav_buttons["settings"].isHidden())
        window._navigate_to("settings")
        self.assertIs(window.stack.currentWidget(), window._pages["welcome"])

        window.set_identity("admin_01", "管理员", ROLE_RESEARCH)
        self.assertFalse(window._nav_buttons["settings"].isHidden())
        window._navigate_to("settings")
        self.assertIs(window.stack.currentWidget(), window._pages["settings"])

    def test_student_history_is_filtered_by_current_user(self):
        from services.dashboard_state import SessionRecord
        from services.identity_store import ROLE_RESEARCH, ROLE_STUDENT, ROLE_TEACHER

        window = self._window(ROLE_STUDENT, "st_01", "学生一")
        history = window._pages["history"]
        window.state._history_sessions = [
            SessionRecord(session_id="own", user_id="st_01"),
            SessionRecord(session_id="other", user_id="st_02"),
            SessionRecord(session_id="teacher", user_id="teacher_02"),
            SessionRecord(session_id="admin", user_id="admin_02"),
        ]
        self.assertEqual(
            [item.session_id for item in history._get_sorted_sessions()], ["own"]
        )

        window.set_identity("teacher_01", "教师", ROLE_TEACHER)
        self.assertEqual(
            {item.session_id for item in history._get_sorted_sessions()},
            {"own", "other"},
        )
        window.set_identity("admin_01", "管理员", ROLE_RESEARCH)
        self.assertEqual(
            {item.session_id for item in history._get_sorted_sessions()},
            {"own", "other", "teacher", "admin"},
        )


if __name__ == "__main__":
    unittest.main()
