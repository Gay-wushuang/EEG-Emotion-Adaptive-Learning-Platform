"""Round 2G event context, baseline isolation, admin semantics and dialog UI tests."""

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
from pages.baseline_page import BaselinePage
from pages.dashboard_page import DashboardPage
from pages.task_page import TaskPage
from services.dashboard_state import DashboardState
from services.identity_store import IdentityStore
from services.teaching_store import (
    AssignmentStore, BaselineResultStore, StudentRuntimeRegistry,
    TeacherObserverService, TeacherSelectionContext, TeacherStudentStore,
)


class Service:
    def __init__(self, state): self.state = state
    def start_session(self): return self.state.begin_session(source="live", demo=False)
    def end_session(self, status="completed"): return self.state.finalize_session(status=status)
    def pause_session(self): self.state.set_session_paused(True)
    def resume_session(self): self.state.set_session_paused(False)


class Round2GFinalCleanupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.identities = IdentityStore(self.root / "identities.json")
        for uid in ("teacher_01", "st_001", "st_002"):
            self.identities.save_profile(uid, uid, make_current=False)
        self.bindings = TeacherStudentStore(self.root / "bindings.json", self.identities)
        self.bindings.add("teacher_01", "st_001")
        self.bindings.add("teacher_01", "st_002")
        self.registry = StudentRuntimeRegistry(self.root / "coordination.sqlite3")
        self.baselines = BaselineResultStore(self.root / "baselines.json")
        self.selection = TeacherSelectionContext()

    def tearDown(self): self.temp.cleanup()

    def state(self, uid, role):
        state = DashboardState(sessions_dir=self.root / uid, seed_demo_history=False)
        state._user_id = state._user_name = uid
        state.current_role = state._user_role = role
        return state

    def teacher_page(self):
        return TaskPage(
            self.state("teacher_01", "teacher"), TeacherObserverService(),
            identity_store=self.identities, binding_store=self.bindings,
            assignment_store=AssignmentStore(self.root / "assignments.json"),
            runtime_registry=self.registry, selection_context=self.selection,
        )

    def test_completed_task_events_remain_and_new_task_switches_context(self):
        student = self.state("st_001", "student")
        student.begin_session(source="live", demo=False)
        first_task_id = student.begin_task("任务一")
        student.add_event("AI状态变化：积极", "system", source="system")
        student.add_event("疑似走神", "teacher", source="teacher")
        student.end_task()
        student.finalize_session()
        self.registry.publish(student, force=True)
        snapshot = self.registry.get("st_001")
        self.assertEqual(snapshot["last_completed_task_id"], first_task_id)
        labels = [event["label"] for event in snapshot["recent_events"]]
        for label in ("开始任务: 任务一", "AI状态变化：积极", "疑似走神", "结束任务"):
            self.assertIn(label, labels)
        page = self.teacher_page()
        self.selection.select("st_001")
        page._refresh_teacher_observation()
        table_labels = [page._event_table.item(row, 3).text()
                        for row in range(page._event_table.rowCount())]
        self.assertIn("疑似走神", table_labels)

        student.begin_session(source="live", demo=False)
        second_task_id = student.begin_task("任务二")
        student.add_event("新任务事件", "system", source="system")
        self.registry.publish(student, force=True)
        snapshot = self.registry.get("st_001")
        self.assertEqual(snapshot["active_task_id"], second_task_id)
        self.assertEqual(snapshot["displayed_task_id"], second_task_id)
        self.assertNotIn("疑似走神", [event["label"] for event in snapshot["recent_events"]])

    def test_event_context_is_isolated_when_switching_student(self):
        for uid, label in (("st_001", "学生一事件"), ("st_002", "学生二事件")):
            state = self.state(uid, "student")
            state.begin_session(source="live", demo=False)
            state.begin_task("任务")
            state.add_event(label, "system", source="system")
            self.registry.publish(state, force=True)
        page = self.teacher_page()
        self.selection.select("st_002")
        page._refresh_teacher_observation()
        labels = [page._event_table.item(row, 3).text()
                  for row in range(page._event_table.rowCount())]
        self.assertIn("学生二事件", labels)
        self.assertNotIn("学生一事件", labels)

    def test_baseline_account_switch_clears_and_reloads_per_user(self):
        self.baselines.save("st_001", {
            "status": "COMPLETED", "completed_at": "2026-09-13 20:00:00",
            "duration_seconds": 75, "avg_attention": 61,
            "avg_meditation": 49, "quality_rate": .9, "sample_count": 300,
        })
        state = self.state("st_001", "student")
        page = BaselinePage(state, Service(state), baseline_store=self.baselines)
        page.set_identity("st_001", "学生一")
        self.assertEqual(page._progress_bar.value(), 100)
        self.assertEqual(page._stat_att["label"].text(), "61")
        page.set_identity("st_002", "学生二")
        self.assertEqual(page._progress_bar.value(), 0)
        self.assertIn("未采集", page._label_status.text())
        self.assertEqual(page._stat_att["label"].text(), "--")
        self.assertEqual(page._label_samples.text(), "已采集样本：0")
        self.baselines.save("st_002", {
            "status": "COMPLETED", "completed_at": "2026-09-13 21:00:00",
            "duration_seconds": 60, "avg_attention": 72,
            "avg_meditation": 55, "quality_rate": .8, "sample_count": 240,
        })
        page.set_identity("st_002", "学生二")
        self.assertEqual(page._stat_att["label"].text(), "72")
        page.set_identity("st_001", "学生一")
        self.assertEqual(page._stat_att["label"].text(), "61")

    def test_admin_semantics_isolation_and_dialog_style(self):
        admin = self.state("admin_01", "research")
        page = TaskPage(admin, Service(admin))
        self.assertEqual(page._title_label.text(), "诊断记录")
        self.assertEqual(page._timer_card._title_label.text(), "监测时长")
        self.assertEqual(page._btn_start_session.text(), "开始本机监测")
        self.assertEqual(page._btn_task_start.text(), "开始监测")
        self.assertEqual(page._btn_task_stop.text(), "结束监测")
        self.assertEqual(page._btn_task_pause.text(), "暂停监测")
        self.assertTrue(all(button.isHidden() for button in page._teacher_event_buttons))
        page._on_task_start()
        page._on_task_stop()
        self.assertTrue(admin.history_sessions)
        self.assertTrue(all(record.user_id == "admin_01" for record in admin.history_sessions))
        self.assertTrue(all(not record.user_id.startswith("st_") for record in admin.history_sessions))
        style = DashboardPage.MANAGE_STUDENTS_DIALOG_STYLE
        self.assertIn("QDialog { background: #161D2A", style)
        self.assertIn("QListWidget::item:selected", style)
        self.assertIn("QLineEdit", style)


if __name__ == "__main__": unittest.main()
