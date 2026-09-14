"""Round 3 teaching-demo and product-language acceptance tests."""

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

from main_window import MainWindow
from pages.dashboard_page import DashboardPage
from pages.settings_page import SettingsPage
from pages.welcome_page import WelcomePage
from services.dashboard_state import DashboardState
from services.eeg_acquisition import _MockSimState
from services.identity_store import IdentityStore
from services.teaching_store import StudentRuntimeRegistry, TeacherSelectionContext, TeacherStudentStore


class _Service:
    def __init__(self, state): self.state = state
    def stop_streaming(self): pass


class Round3ProductizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_demo_trajectory_is_deterministic_and_staged(self):
        first, second = _MockSimState(), _MockSimState()
        values = []
        for timestamp in (5, 20, 30, 40, 55, 70):
            first.advance(timestamp)
            second.advance(timestamp)
            self.assertEqual((first.attention, first.meditation, first.powers),
                             (second.attention, second.meditation, second.powers))
            values.append((first.attention, first.emotion_trend))
        self.assertGreater(values[1][0], values[0][0])
        self.assertEqual(values[3][1], 2)
        self.assertEqual(values[4][1], 2)
        self.assertGreater(values[5][0], values[4][0])

    def test_demo_mode_has_product_label_and_is_ready(self):
        window = MainWindow(mode="mock", user_id="st_001", user_name="学生", role="student")
        try:
            for _ in range(3): self.app.processEvents()
            self.assertIn("教学演示", window._demo_badge.text())
            self.assertNotIn("Mock", window._version_label.text())
            self.assertEqual(window.state.device_status, "online")
            self.assertTrue(window._pages["baseline"]._btn_start.isEnabled())
        finally:
            window.service.stop_streaming(); window.close(); window.deleteLater()

    def test_active_learning_blocks_mode_switch(self):
        window = MainWindow(mode="mock", user_id="st_001", user_name="学生", role="student")
        try:
            window.state.begin_session(source="mock", demo=True)
            window.state.begin_task("演示任务")
            with patch("main_window.QMessageBox.information") as info:
                self.assertFalse(window.switch_data_mode("live"))
            self.assertEqual(window._mode, "mock")
            self.assertIn("先结束任务", info.call_args.args[2])
        finally:
            window.service.stop_streaming(); window.close(); window.deleteLater()

    def test_student_home_recommends_next_business_step(self):
        state = DashboardState(seed_demo_history=False)
        state._user_id = "st_001"; state.current_role = "student"; state.mode = "mock"
        page = WelcomePage(state, _Service(state))
        page.set_role("student"); page.update_state(state)
        self.assertIn("完成基线采集", page._flow_action.text())
        state.baseline_status = "COMPLETED"; page.update_state(state)
        self.assertIn("学习任务", page._flow_action.text())

    def test_teacher_home_uses_business_language(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            identities = IdentityStore(root / "identities.json")
            identities.save_profile("teacher_01", "教师", make_current=False)
            identities.save_profile("st_001", "学生", make_current=False)
            bindings = TeacherStudentStore(root / "bindings.json", identities)
            bindings.add("teacher_01", "st_001")
            registry = StudentRuntimeRegistry(root / "runtime.sqlite3")
            student = DashboardState(seed_demo_history=False)
            student._user_id = "st_001"; student._user_name = "学生"
            student.quality_level = "trusted"; student.stable_state = "negative"
            student.attention = 38; student.adaptive_feedback_text = "建议调整学习节奏"
            registry.publish(student, force=True)
            teacher = DashboardState(seed_demo_history=False)
            teacher._user_id = "teacher_01"; teacher.current_role = "teacher"
            page = WelcomePage(teacher, _Service(teacher), identity_store=identities,
                               binding_store=bindings, runtime_registry=registry,
                               selection_context=TeacherSelectionContext())
            page.set_role("teacher"); page._refresh_teacher_workspace()
            text = page._teacher_detail.text()
            self.assertIn("学习负荷偏高", text)
            self.assertIn("建议调整学习节奏", text)
            self.assertNotIn("negative", text)
            self.assertNotIn("trusted", text)

    def test_model_error_is_layered_and_label_encoder_is_diagnostic_only(self):
        state = DashboardState(seed_demo_history=False)
        state.model_status = "FAILED"; state.pipeline_state = "error"
        state.model_error_user = "智能分析暂不可用，请联系管理员。"
        state.model_error_detail = "missing /secret/model.pt"
        dashboard = DashboardPage(state, _Service(state)); dashboard.set_role("student")
        kind, title, hint = dashboard._analysis_view(state)
        self.assertEqual(kind, "error")
        self.assertNotIn("/secret", title + hint)
        settings = SettingsPage(state, _Service(state)); settings.update_state(state)
        self.assertIn("/secret/model.pt", settings._diag_labels["model_detail"].text())
        self.assertIn("Production Baseline v1 不依赖", settings._asset_diagnostics.text())

    def test_1366_by_768_window_smoke(self):
        window = MainWindow(mode="mock", user_id="admin_demo", user_name="管理员", role="research")
        try:
            window.resize(1366, 768); window.show(); self.app.processEvents()
            self.assertEqual(window.size().width(), 1366)
            self.assertGreater(window.stack.width(), 900)
        finally:
            window.service.stop_streaming(); window.close(); window.deleteLater()


if __name__ == "__main__":
    unittest.main()
