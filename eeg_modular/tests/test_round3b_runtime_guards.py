"""Round 3B readiness, runtime source, history and navigation tests."""
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
from main_window import MainWindow, ROLE_NAV_LABELS
from pages.dashboard_page import DashboardPage
from pages.history_page import HistoryPage
from services.dashboard_state import DashboardState, SessionRecord
from services.learning_readiness import learning_start_block_reason
from services.teaching_store import StudentRuntimeRegistry, TeacherSelectionContext


class Service:
    def __init__(self, state): self.state = state; self.sessions_dir = Path("data/sessions")
    def stop_streaming(self): pass


class Round3BRuntimeGuardsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app = QApplication.instance() or QApplication(sys.argv)

    def live_state(self):
        state = DashboardState(seed_demo_history=False); state.mode = "live"
        state._user_id = "st_001"; state.current_role = "student"
        return state

    def test_live_start_reasons_cover_device_wait_warmup_signal_and_baseline(self):
        state = self.live_state()
        self.assertIn("连接", learning_start_block_reason(state))
        state.connector_status = "online"; state.device_status = "waiting_raw"
        self.assertIn("等待设备数据", learning_start_block_reason(state))
        state.device_status = "online"; state.poor_signal = 0; state.quality_level = "trusted"
        self.assertIn("预热", learning_start_block_reason(state))
        state.warmup_progress = 1.0; state.model_status = "READY"
        self.assertIn("基线", learning_start_block_reason(state))
        state.baseline_status = "COMPLETED"
        self.assertEqual(learning_start_block_reason(state), "")
        state.poor_signal = 200
        self.assertIn("信号质量", learning_start_block_reason(state))

    def test_demo_ignores_physical_readiness(self):
        state = self.live_state(); state.mode = "mock"
        state.connector_status = state.device_status = "offline"
        state.quality_level = "rejected"; state.model_status = "FAILED"
        self.assertEqual(learning_start_block_reason(state), "")

    def test_registry_persists_demo_mode_and_latest_quality(self):
        with tempfile.TemporaryDirectory() as folder:
            registry = StudentRuntimeRegistry(Path(folder) / "runtime.sqlite3")
            state = self.live_state(); state.mode = "mock"; state.quality_level = "rejected"
            registry.publish(state, force=True)
            self.assertEqual(registry.get("st_001")["data_mode"], "demo")
            state.mode = "live"; state.quality_level = "trusted"; state.stable_state = "positive"
            state.prob_positive = .8
            registry.publish(state, force=True)
            snap = registry.get("st_001")
            self.assertEqual(snap["data_mode"], "live")
            self.assertEqual(snap["quality_level"], "trusted")

    def test_teacher_snapshot_clears_rejected_message_after_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            registry = StudentRuntimeRegistry(Path(folder) / "runtime.sqlite3")
            selection = TeacherSelectionContext(); selection.select("st_001")
            student = self.live_state(); student.quality_level = "rejected"
            registry.publish(student, force=True)
            teacher = DashboardState(seed_demo_history=False); teacher.current_role = "teacher"
            page = DashboardPage(teacher, Service(teacher), runtime_registry=registry,
                                 selection_context=selection)
            page.set_role("teacher"); selection.select("st_001"); page._update_teacher_snapshot()
            self.assertIn("不可解释", page._pred_label.text())
            student.quality_level = "trusted"; student.stable_state = "positive"; student.prob_positive = .8
            registry.publish(student, force=True); page._update_teacher_snapshot()
            self.assertIn("积极", page._pred_label.text())
            self.assertNotIn("不可解释", page._pred_label.text())

    def test_history_source_filter_is_stable_when_selecting_detail(self):
        state = DashboardState(seed_demo_history=False); state.current_role = "student"; state._user_id = "st_001"
        common = dict(status="completed", user_id="st_001", start_time="2026-01-01", end_time="2026-01-01")
        state._history_sessions = [
            SessionRecord(session_id="live1", source="live", demo=False, **common),
            SessionRecord(session_id="demo1", source="mock", demo=True, **common),
        ]
        page = HistoryPage(state, Service(state)); page._combo_source.setCurrentIndex(2); page._refresh_table()
        self.assertEqual(page._table.rowCount(), 1)
        page._table.selectRow(0); page._on_select()
        self.assertEqual(page._table.rowCount(), 1)
        self.assertEqual(page._get_sorted_sessions()[0].session_id, "demo1")

    def test_role_navigation_labels(self):
        self.assertEqual(ROLE_NAV_LABELS["student"]["task"], "我的学习任务")
        self.assertEqual(ROLE_NAV_LABELS["teacher"]["dashboard"], "学生实时观察")
        self.assertEqual(ROLE_NAV_LABELS["research"]["dashboard"], "模型与信号诊断")
        self.assertNotIn("基线采集", ROLE_NAV_LABELS["research"].values())

    def test_admin_asset_diagnostics_exposes_expected_encoder_path(self):
        window = MainWindow(mode="mock", user_id="admin_demo", user_name="管理员", role="research")
        try:
            text = window._pages["settings"]._asset_diagnostics.text()
            self.assertIn("Legacy Label Encoder：未安装", text)
            self.assertIn("features", text)
            self.assertIn("coordination.sqlite3", text)
        finally:
            window.service.stop_streaming(); window.close(); window.deleteLater()


if __name__ == "__main__": unittest.main()
