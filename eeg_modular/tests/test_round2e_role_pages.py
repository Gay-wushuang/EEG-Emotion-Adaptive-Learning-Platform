"""Round 2E teacher page binding and role-semantics regressions."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui_prototype"))

from PySide6.QtWidgets import QApplication
from pages.dashboard_page import DashboardPage
from pages.task_page import TaskPage
from pages.welcome_page import WelcomePage
from pages.baseline_page import BaselinePage
from pages.replay_page import ReplayPage
from services.dashboard_state import DashboardState
from services.identity_store import IdentityStore
from services.teaching_store import (
    AssignmentStore, StudentRuntimeRegistry, TeacherObserverService,
    TeacherSelectionContext, TeacherStudentStore,
)
from services.mock_data_service import MockDataService
from widgets.trend_plot import TrendPlotWidget
from widgets.eeg_plot import EEGPlotWidget


class Service:
    def __init__(self, state): self.state = state
    def start_session(self): return self.state.begin_session(source="live", demo=False)
    def end_session(self, status="completed"): return self.state.finalize_session(status=status)
    def pause_session(self): self.state.set_session_paused(True)
    def resume_session(self): self.state.set_session_paused(False)


class Round2ERolePagesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.identities = IdentityStore(root / "identities.json")
        for uid in ("teacher_01", "st_001", "st_002"):
            self.identities.save_profile(uid, uid, make_current=False)
        self.bindings = TeacherStudentStore(root / "bindings.json", identity_store=self.identities)
        self.bindings.add("teacher_01", "st_001")
        self.bindings.add("teacher_01", "st_002")
        self.registry = StudentRuntimeRegistry(root / "coordination.sqlite3")
        self.assignments = AssignmentStore(root / "assignments.json")
        self.selection = TeacherSelectionContext()
        self.root = root

    def tearDown(self): self.temp.cleanup()

    def state(self, uid, role):
        state = DashboardState(sessions_dir=self.root / uid, seed_demo_history=False)
        state._user_id = state._user_name = uid
        state.current_role = state._user_role = role
        return state

    def test_mock_trend_timer_matches_the_ten_hz_chart_timebase(self):
        state = self.state("st_001", "student")
        service = MockDataService(state)
        self.assertEqual(service._tick_timer.interval(), 100)

    def test_partial_trend_uses_elapsed_range_then_caps_at_ninety_seconds(self):
        plot = TrendPlotWidget()
        for _ in range(300):
            plot.push_values(60, 50)
        start, end = plot._plot.viewRange()[0]
        self.assertAlmostEqual(start, -30, delta=1)
        self.assertAlmostEqual(end, 0, delta=1)
        for _ in range(600):
            plot.push_values(60, 50)
        start, end = plot._plot.viewRange()[0]
        self.assertAlmostEqual(start, -90, delta=1)
        self.assertAlmostEqual(end, 0, delta=1)

    def test_partial_eeg_buffer_uses_its_actual_duration(self):
        plot = EEGPlotWidget()
        plot.push_buffer([1] * 256)
        start, end = plot._plot.viewRange()[0]
        self.assertAlmostEqual(start, -0.5, delta=0.1)
        self.assertAlmostEqual(end, 0, delta=0.1)

    def test_mock_session_saves_replayable_csv(self):
        sessions = self.root / "mock_sessions"
        state = self.state("st_001", "student")
        state.configure_session_store(sessions, include_demo=True)
        service = MockDataService(state)
        service.sessions_dir = sessions
        state.reset_session()
        service.start_session()
        service._on_acq_data(SimpleNamespace(
            raw=123, attention=60, meditation=50, poor_signal=0,
            timestamp=time.time(),
        ))
        saved = service.end_session()
        csv_path = sessions / "_demo" / state.run_id / "session.csv"
        self.assertIsNotNone(saved)
        self.assertTrue(csv_path.is_file())
        self.assertIn("raw", csv_path.read_text(encoding="utf-8-sig").splitlines()[0])
        self.assertGreater(len(csv_path.read_text(encoding="utf-8-sig").splitlines()), 1)
        self.assertTrue(ReplayPage._csv_is_replayable(csv_path))

    def test_task_and_dashboard_share_selection_and_snapshot(self):
        student = self.state("st_001", "student")
        student.attention, student.meditation = 62, 44
        student.quality_level, student.stable_state = "trusted", "positive"
        student.begin_session(source="live", demo=False)
        student.begin_task("英语阅读", assignment_id="A001")
        self.registry.publish(student, force=True)
        teacher = self.state("teacher_01", "teacher")
        dashboard = DashboardPage(
            teacher, TeacherObserverService(), identity_store=self.identities,
            binding_store=self.bindings, runtime_registry=self.registry,
            selection_context=self.selection,
        )
        task = TaskPage(
            teacher, TeacherObserverService(), identity_store=self.identities,
            binding_store=self.bindings, assignment_store=self.assignments,
            runtime_registry=self.registry, selection_context=self.selection,
        )
        dashboard._teacher_student_combo.setCurrentIndex(
            dashboard._teacher_student_combo.findData("st_001")
        )
        task._poll_teacher_runtime()
        self.assertEqual(self.selection.selected_student_id, "st_001")
        self.assertIn("英语阅读", dashboard._analysis_label.text())
        self.assertEqual(dashboard._teacher_trend_card._title_label.text(), "实时学习趋势")
        self.assertNotIn("学习状态", dashboard._teacher_trend_hint.text())
        self.assertEqual(dashboard._teacher_main_row.minimumHeight(), 390)
        self.assertEqual(dashboard._teacher_main_row.maximumHeight(), 390)
        self.assertGreaterEqual(dashboard._teacher_eeg_stack.minimumHeight(), 96)
        self.assertIn("英语阅读", task._teacher_trend.text())
        self.assertIn("62/44", task._teacher_trend.text())

        student._attention_history.extend([30, 45, 62])
        student._meditation_history.extend([50, 48, 44])
        time.sleep(0.02)
        self.registry.publish(student, force=True)
        task._poll_teacher_runtime()
        self.assertAlmostEqual(float(task._teacher_trend_plot._att_data[-1]), 62)

    def test_teacher_runtime_keeps_full_ninety_second_trend_and_labels_idle_data(self):
        student = self.state("st_001", "student")
        student._attention_history.extend(range(900))
        student._meditation_history.extend(range(900))
        self.registry.publish(student, force=True)
        snapshot = self.registry.get("st_001")
        self.assertEqual(len(snapshot["attention_history"]), 900)
        self.assertEqual(len(snapshot["meditation_history"]), 900)

        teacher = self.state("teacher_01", "teacher")
        dashboard = DashboardPage(
            teacher, TeacherObserverService(), identity_store=self.identities,
            binding_store=self.bindings, runtime_registry=self.registry,
            selection_context=self.selection,
        )
        dashboard._teacher_student_combo.setCurrentIndex(
            dashboard._teacher_student_combo.findData("st_001")
        )
        self.assertEqual(dashboard._teacher_trend_card._title_label.text(), "实时状态趋势")
        self.assertIn("当前未进行学习任务", dashboard._teacher_trend_hint.text())

    def test_switching_student_changes_both_pages(self):
        for uid, task_name in (("st_001", "物理复习"), ("st_002", "数学练习")):
            student = self.state(uid, "student")
            student.begin_session(source="live", demo=False)
            student.begin_task(task_name)
            self.registry.publish(student, force=True)
        teacher = self.state("teacher_01", "teacher")
        dashboard = DashboardPage(
            teacher, TeacherObserverService(), identity_store=self.identities,
            binding_store=self.bindings, runtime_registry=self.registry,
            selection_context=self.selection,
        )
        task = TaskPage(
            teacher, TeacherObserverService(), identity_store=self.identities,
            binding_store=self.bindings, assignment_store=self.assignments,
            runtime_registry=self.registry, selection_context=self.selection,
        )
        dashboard._teacher_student_combo.setCurrentIndex(
            dashboard._teacher_student_combo.findData("st_002")
        )
        task._poll_teacher_runtime()
        self.assertIn("数学练习", dashboard._analysis_label.text())
        self.assertIn("数学练习", task._teacher_trend.text())

    def test_compatibility_event_button_hidden_and_feedback_style_readable(self):
        from main_window import MainWindow
        for uid, role in (("st_ui", "student"), ("admin_ui", "research")):
            window = MainWindow(mode="mock", user_id=uid, user_name=uid, role=role)
            self.addCleanup(window.close)
            self.assertTrue(window._pages["dashboard"]._btn_event.isHidden())
        self.assertIn("QLabel { color: #E8EDF3", TaskPage.FEEDBACK_DIALOG_STYLE)
        self.assertIn("QComboBox { background: #222B3A", TaskPage.FEEDBACK_DIALOG_STYLE)

    def test_teacher_home_and_baseline_share_selected_student_snapshot(self):
        student = self.state("st_001", "student")
        student.baseline_status = "COMPLETED"
        student._baseline_elapsed = 75
        student._baseline_samples = 300
        student._baseline_avg_attention = 58
        self.registry.publish(student, force=True)
        teacher = self.state("teacher_01", "teacher")
        kwargs = dict(
            identity_store=self.identities, binding_store=self.bindings,
            runtime_registry=self.registry, selection_context=self.selection,
        )
        home = WelcomePage(teacher, TeacherObserverService(), **kwargs)
        baseline = BaselinePage(teacher, TeacherObserverService(), **kwargs)
        home._teacher_selected.setCurrentIndex(home._teacher_selected.findData("st_001"))
        baseline._refresh_teacher_baseline()
        self.assertIn("st_001", home._teacher_detail.text())
        self.assertIn("已完成", baseline._teacher_baseline_detail.text())
        self.assertTrue(baseline._btn_start.isHidden())
        baseline._start_baseline()
        self.assertFalse(baseline._baseline_active)

    def test_admin_diagnostic_record_is_not_student_history(self):
        from main_window import MainWindow
        window = MainWindow(mode="mock", user_id="admin_ui", user_name="管理员", role="research")
        self.addCleanup(window.close)
        page = window._pages["dashboard"]
        self.assertEqual(page._page_title.text(), "本机设备与模型诊断")
        before = {item.session_id for item in window.state.history_sessions}
        page._on_start()
        with patch("pages.dashboard_page.QMessageBox.information"):
            page._on_end()
        created = [item for item in window.state.history_sessions if item.session_id not in before]
        self.assertTrue(created)
        self.assertTrue(all(item.user_id == "admin_ui" for item in created))


if __name__ == "__main__": unittest.main()
