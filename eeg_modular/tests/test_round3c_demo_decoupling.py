"""Round 3C demo decoupling and teacher presentation acceptance tests."""
from __future__ import annotations
import os, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui_prototype"))
from PySide6.QtWidgets import QApplication, QMessageBox
from main_window import MainWindow
from pages.task_page import TaskPage
from services.dashboard_state import DashboardState
from services.mock_data_service import MockDataService


class _SessionService:
    def __init__(self, state): self.state = state
    def start_session(self): return self.state.begin_session(source="live", demo=False)
    def end_session(self, status="completed"): return self.state.finalize_session(status=status)


class Round3CDemoDecouplingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_demo_has_no_inference_worker_and_is_immediately_ready(self):
        state = DashboardState(seed_demo_history=False)
        service = MockDataService(state); service.start_streaming()
        try:
            self.assertFalse(hasattr(service, "inf_worker"))
            self.assertEqual(state.pipeline_state, "ready")
            self.assertTrue(state.warmup_complete)
            self.assertEqual(state.quality_level, "trusted")
        finally: service.stop_streaming()

    def test_demo_probability_stages_cross_real_negative_threshold(self):
        early = MockDataService._demo_result(15)["probabilities"]
        burden = MockDataService._demo_result(45)["probabilities"]
        recovery = MockDataService._demo_result(72)["probabilities"]
        self.assertGreater(early[0], early[2])
        self.assertGreaterEqual(burden[2], .60)
        self.assertLess(recovery[2], burden[2])

    def test_live_guard_attempt_explains_and_creates_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            state = DashboardState(sessions_dir=Path(folder), seed_demo_history=False)
            state.current_role = "student"; state._user_id = "st_001"; state.mode = "live"
            page = TaskPage(state, _SessionService(state)); page.set_role("student"); page.show()
            self.app.processEvents()
            with patch.object(QMessageBox, "warning") as warning:
                page._on_task_start()
            self.assertIn("EEG 设备", warning.call_args.args[2])
            self.assertFalse(state.session_active); self.assertFalse(state.task_running)
            self.assertEqual(state._history_sessions, [])
            page.close(); page.deleteLater()

    def test_teacher_has_readonly_source_and_prominent_demo_badge(self):
        window = MainWindow(mode="mock", user_id="teacher_01", user_name="教师", role="teacher")
        try:
            self.assertFalse(window._mode_combo.isVisible())
            self.assertFalse(window._teacher_mode_source.isHidden())
            dashboard = window._pages["dashboard"]
            snapshot = {"student_id":"st_001", "student_name":"学生", "data_mode":"demo",
                        "online":True, "elapsed_seconds":1, "quality_level":"trusted",
                        "stable_state":"positive", "attention":60, "meditation":55,
                        "probabilities":{"positive":.7,"neutral":.2,"negative":.1},
                        "raw_eeg":list(range(128)), "attention_history":[], "meditation_history":[],
                        "baseline_status":"COMPLETED", "task_id":"t", "task_name":"练习", "updated_at":1}
            with patch.object(dashboard.runtime_registry, "get", return_value=snapshot):
                dashboard.selection_context.select("st_001"); dashboard._update_teacher_snapshot()
            self.assertIn("教学演示", dashboard._analysis_label.text())
            self.assertFalse(any(map(lambda value: value != value, dashboard._eeg_plot._data)))
        finally: window.close(); window.deleteLater()

    def test_asset_text_separates_production_and_legacy(self):
        window = MainWindow(mode="mock", user_id="admin_demo", user_name="管理员", role="research")
        try:
            text = window._pages["settings"]._asset_diagnostics.text()
            self.assertIn("生产运行资产完整", text)
            self.assertIn("兼容资产", text)
            self.assertIn("不影响正式推理", text)
        finally: window.close(); window.deleteLater()

    def test_demo_live_switch_reuses_cached_live_service_and_keeps_history(self):
        class CachedLive:
            def __init__(self, state): self.state = state; self.resumed = self.suspended = 0
            def resume_streaming(self): self.resumed += 1; self.state.mode = "live"
            def suspend_streaming(self): self.suspended += 1
            def stop_streaming(self): pass
        window = MainWindow(mode="mock", user_id="st_001", user_name="学生", role="student")
        live = CachedLive(window.state); window._live_service = live
        marker = object(); window.state._history_sessions = [marker]
        try:
            self.assertTrue(window.switch_data_mode("live"))
            self.assertIs(window.service, live); self.assertEqual(live.resumed, 1)
            self.assertEqual(window.state._history_sessions, [marker])
            self.assertTrue(window.switch_data_mode("mock"))
            self.assertEqual(live.suspended, 1)
            self.assertEqual(window.state.mode, "mock")
        finally: window.close(); window.deleteLater()


if __name__ == "__main__": unittest.main()
