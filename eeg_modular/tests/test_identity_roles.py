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
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class IdentityStoreTest(unittest.TestCase):
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

    def _window(self, role, user_id="local_01", name="测试用户"):
        from main_window import MainWindow

        window = MainWindow(
            mode="mock",
            user_id=user_id,
            user_name=name,
            role=role,
        )
        self.addCleanup(window.close)
        return window

    def test_login_dialog_requires_identity_then_selects_role(self):
        from PySide6.QtWidgets import QDialog
        from login_dialog import LoginDialog
        from services.identity_store import IdentityStore, ROLE_TEACHER

        with tempfile.TemporaryDirectory() as directory:
            store = IdentityStore(Path(directory) / "identities.json")
            dialog = LoginDialog(store)
            self.addCleanup(dialog.close)
            dialog._input_user_id.setText("teacher_02")
            dialog._input_user_name.setText("王老师")
            dialog._continue_to_roles()
            self.assertEqual(dialog._stack.currentIndex(), 1)

            dialog._role_buttons[ROLE_TEACHER].click()
            dialog._finish_login()
            self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
            self.assertEqual(dialog.selected_role, ROLE_TEACHER)
            self.assertEqual(store.last_user_id(), "teacher_02")

    def test_student_sees_only_three_learning_pages(self):
        from services.identity_store import ROLE_STUDENT

        window = self._window(ROLE_STUDENT)
        visible = {key for key, button in window._nav_buttons.items() if not button.isHidden()}
        self.assertEqual(visible, {"welcome", "baseline", "dashboard"})
        self.assertEqual(window.state._user_role, ROLE_STUDENT)

        before = window.stack.currentWidget()
        window._navigate_to("history")
        self.assertIs(window.stack.currentWidget(), before)

    def test_teacher_sees_only_teaching_pages(self):
        from services.identity_store import ROLE_TEACHER

        window = self._window(ROLE_TEACHER)
        visible = {key for key, button in window._nav_buttons.items() if not button.isHidden()}
        self.assertEqual(visible, {"baseline", "dashboard", "task", "history"})
        self.assertIs(window.stack.currentWidget(), window._pages["baseline"])

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
        window.set_identity("student_03", "周同学", ROLE_STUDENT)

        self.assertEqual(window.state._user_id, "student_03")
        self.assertEqual(window.state._user_name, "周同学")
        self.assertEqual(window.state._user_role, ROLE_STUDENT)
        self.assertEqual(window._pages, original_pages)
        self.assertIs(window.stack.currentWidget(), window._pages["welcome"])


if __name__ == "__main__":
    unittest.main()
