"""Round 2F read-only baseline details and teacher event timeline regressions."""

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
from pages.task_page import TaskPage
from services.dashboard_state import DashboardState
from services.identity_store import IdentityStore
from services.teaching_store import (
    AssignmentStore, BaselineResultStore, StudentRuntimeRegistry,
    TeacherObserverService, TeacherSelectionContext, TeacherStudentStore,
)


class Round2FReadonlyDetailsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.identities = IdentityStore(self.root / "identities.json")
        for user_id in ("teacher_01", "st_001", "st_002"):
            self.identities.save_profile(user_id, user_id, make_current=False)
        self.bindings = TeacherStudentStore(
            self.root / "bindings.json", identity_store=self.identities
        )
        self.bindings.add("teacher_01", "st_001")
        self.bindings.add("teacher_01", "st_002")
        self.db = self.root / "coordination.sqlite3"
        self.student_registry = StudentRuntimeRegistry(self.db)
        self.teacher_registry = StudentRuntimeRegistry(self.db)
        self.baselines = BaselineResultStore(self.root / "baselines.json")
        self.selection = TeacherSelectionContext()

    def tearDown(self):
        self.temp.cleanup()

    def state(self, user_id, role):
        state = DashboardState(sessions_dir=self.root / user_id, seed_demo_history=False)
        state._user_id = state._user_name = user_id
        state.current_role = state._user_role = role
        return state

    def baseline_page(self):
        return BaselinePage(
            self.state("teacher_01", "teacher"), TeacherObserverService(),
            identity_store=self.identities, binding_store=self.bindings,
            runtime_registry=self.teacher_registry, selection_context=self.selection,
            baseline_store=self.baselines,
        )

    def test_completed_baseline_details_are_persisted_readonly_and_switch(self):
        self.baselines.save("st_001", {
            "status": "COMPLETED", "completed_at": "2026-09-13 22:50:00",
            "duration_seconds": 75, "avg_attention": 58.5,
            "avg_meditation": 47.5, "quality_rate": 0.92, "sample_count": 300,
        })
        self.baselines.save("st_002", {
            "status": "COMPLETED", "completed_at": "2026-09-13 23:00:00",
            "duration_seconds": 60, "avg_attention": 66,
            "avg_meditation": 51, "quality_rate": 0.88, "sample_count": 240,
        })
        page = self.baseline_page()
        self.selection.select("st_001")
        page._refresh_teacher_baseline()
        text = page._teacher_baseline_detail.text()
        for expected in ("st_001", "22:50:00", "75 秒", "58.5", "47.5", "92%", "300"):
            self.assertIn(expected, text)
        self.assertIn("未保存采样序列", text)
        page._start_baseline()
        page._stop_baseline()
        self.assertFalse(page._baseline_active)
        self.selection.select("st_002")
        page._refresh_teacher_baseline()
        self.assertIn("st_002", page._teacher_baseline_detail.text())
        self.assertIn("66", page._teacher_baseline_detail.text())
        self.assertNotIn("58.5", page._teacher_baseline_detail.text())

    def test_teacher_timeline_reads_student_snapshot_once_and_isolated(self):
        first = self.state("st_001", "student")
        first.begin_session(source="live", demo=False)
        first.begin_task("英语阅读")
        first.add_event("AI状态变化：positive", "system", source="system")
        first.add_event("疲劳程度：低", "self_report", source="self_report")
        self.student_registry.publish(first, force=True)

        teacher = self.state("teacher_01", "teacher")
        page = TaskPage(
            teacher, TeacherObserverService(), identity_store=self.identities,
            binding_store=self.bindings,
            assignment_store=AssignmentStore(self.root / "assignments.json"),
            runtime_registry=self.teacher_registry, selection_context=self.selection,
        )
        self.selection.select("st_001")
        before = len(first._events)
        event_id = self.teacher_registry.add_teacher_event(
            "st_001", "teacher_01", "疑似走神", "课堂观察"
        )
        self.assertIsNotNone(event_id)
        self.student_registry.publish(first, force=True)
        self.student_registry.publish(first, force=True)
        self.assertEqual(len(first._events), before + 1)
        self.assertEqual(sum(event.label == "疑似走神" for event in first._events), 1)

        page._refresh_teacher_observation()
        labels = [page._event_table.item(row, 3).text()
                  for row in range(page._event_table.rowCount())]
        sources = [page._event_table.item(row, 1).text()
                   for row in range(page._event_table.rowCount())]
        self.assertIn("AI状态变化：positive", labels)
        self.assertIn("疑似走神", labels)
        self.assertIn("系统记录", sources)
        self.assertIn("教师观察", sources)
        self.assertIn("学生反馈", sources)
        count_after_read = len(first._events)
        page._refresh_teacher_observation()
        self.assertEqual(len(first._events), count_after_read)

        second = self.state("st_002", "student")
        second.begin_session(source="live", demo=False)
        second.begin_task("数学练习")
        second.add_event("学生二事件", "system", source="system")
        self.student_registry.publish(second, force=True)
        self.selection.select("st_002")
        page._refresh_teacher_observation()
        switched = [page._event_table.item(row, 3).text()
                    for row in range(page._event_table.rowCount())]
        self.assertIn("学生二事件", switched)
        self.assertNotIn("疑似走神", switched)


if __name__ == "__main__":
    unittest.main()
