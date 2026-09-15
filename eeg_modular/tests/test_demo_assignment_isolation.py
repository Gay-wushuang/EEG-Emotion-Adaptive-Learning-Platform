"""Demo / Live 隔离修复：教学演示不消耗正式教师 Assignment。

规则：只有 data_mode == live 且真实完成任务时，Assignment 才允许
pending → running → completed 流转；Demo 模式只产生 Demo
TaskRecord / History，不修改正式 Assignment。
"""

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

from services.dashboard_state import DashboardState
from services.identity_store import IdentityStore
from services.session_store import SessionStore
from services.teaching_store import AssignmentStore, StudentRuntimeRegistry
from pages.task_page import TaskPage


class Service:
    def __init__(self, state):
        self.state = state

    def start_session(self):
        return self.state.begin_session(
            source="mock" if self.state.mode == "mock" else "live",
            demo=self.state.mode == "mock",
        )

    def end_session(self, status="completed"):
        return self.state.finalize_session(status=status)

    def pause_session(self):
        self.state.set_session_paused(True)

    def resume_session(self):
        self.state.set_session_paused(False)


class DemoAssignmentIsolationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.root = root
        self.assignments = AssignmentStore(root / "assignments.json")
        self.identities = IdentityStore(root / "identities.json")
        self.registry = StudentRuntimeRegistry(root / "coordination.sqlite3")
        self.state = DashboardState(
            sessions_dir=root / "sessions", seed_demo_history=False
        )
        self.state._user_id = "st_001"
        self.state._user_name = "学生一"
        self.state._user_role = self.state.current_role = "student"
        self.page = TaskPage(
            self.state, Service(self.state),
            assignment_store=self.assignments,
            runtime_registry=self.registry,
            identity_store=self.identities,
        )

    def tearDown(self):
        self.temp.cleanup()

    def _publish_assignment(self):
        return self.assignments.publish(
            "teacher_001", "st_001", "英语阅读", "medium", "第 3 章",
        )

    def _assignment(self, assignment_id):
        return next(
            a for a in self.assignments.list_all()
            if a.assignment_id == assignment_id
        )

    def _run_and_stop_task(self):
        """以当前 state.mode 开始并结束选中任务。"""
        self.page._on_task_start()
        self.assertTrue(self.page._task_active)
        self.page._on_task_stop()

    def _history_records(self):
        store = SessionStore(self.root / "sessions")
        return store.load(include_demo=True)

    # ── 1～3：Demo 模式执行教师任务，Assignment 不被消耗 ──

    def test_01_02_03_demo_completion_does_not_consume_assignment(self):
        assignment = self._publish_assignment()
        self.assertEqual(assignment.status, "pending")

        # 切到教学演示模式执行该任务
        self.state.mode = "mock"
        self.page.set_role("student")  # 触发任务列表刷新并选中
        self.assertEqual(self.page._selected_assignment_id, assignment.assignment_id)

        before = self._assignment(assignment.assignment_id)
        self._run_and_stop_task()

        after = self._assignment(assignment.assignment_id)
        # 状态未被消耗：仍是待开始
        self.assertEqual(after.status, "pending")
        # 整条正式记录逐字段不变（含真实完成时间相关字段）
        self.assertEqual(after, before)

    # ── 4：Demo 历史正确保存 ──

    def test_04_demo_history_saved(self):
        assignment = self._publish_assignment()
        self.state.mode = "mock"
        self.page.set_role("student")
        self._run_and_stop_task()

        records = self._history_records()
        self.assertEqual(len(records), 1)
        demo = records[0]
        self.assertTrue(demo["demo"])
        self.assertEqual(demo["source"], "mock")
        self.assertEqual(demo["user_id"], "st_001")
        tasks = demo["tasks"]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["name"], "英语阅读")
        self.assertEqual(tasks[0]["status"], "completed")
        self.assertEqual(tasks[0]["assignment_id"], assignment.assignment_id)

    # ── 5～6：切回 Live 后仍可开始；Live 完成才标记完成 ──

    def test_05_06_live_formal_completion_marks_assignment(self):
        assignment = self._publish_assignment()

        # 先在 Demo 模式跑一遍（不应消耗）
        self.state.mode = "mock"
        self.page.set_role("student")
        self._run_and_stop_task()
        self.assertEqual(
            self._assignment(assignment.assignment_id).status, "pending"
        )

        # 切回实时采集：任务仍在列表中（pending 会被列出）
        self.state.mode = "live"
        self.page._task_active = False
        self.page.set_role("student")
        self.page._refresh_student_assignments()
        self.assertEqual(self.page._selected_assignment_id, assignment.assignment_id)
        self.assertIn(
            "待开始", self.page._combo_assignment.currentText()
        )

        # Live 正式完成 → Assignment 才变完成
        self._run_and_stop_task()
        self.assertEqual(
            self._assignment(assignment.assignment_id).status, "completed"
        )

    # ── 7：Demo + Live 两条独立历史共存 ──

    def test_07_demo_and_live_histories_coexist(self):
        assignment = self._publish_assignment()

        self.state.mode = "mock"
        self.page.set_role("student")
        self._run_and_stop_task()

        self.state.mode = "live"
        self.page._task_active = False
        self.page.set_role("student")
        self.page._refresh_student_assignments()
        self._run_and_stop_task()

        records = self._history_records()
        self.assertEqual(len(records), 2)
        by_source = {r["source"]: r for r in records}
        self.assertIn("mock", by_source)
        self.assertIn("live", by_source)
        self.assertTrue(by_source["mock"]["demo"])
        self.assertFalse(by_source["live"]["demo"])
        for record in by_source.values():
            self.assertEqual(record["tasks"][0]["name"], "英语阅读")
            self.assertEqual(record["tasks"][0]["status"], "completed")
            self.assertEqual(record["tasks"][0]["assignment_id"], assignment.assignment_id)

    # ── 8～9：Demo 不写正式状态 / 完成时间 ──

    def test_08_09_demo_never_touches_formal_record(self):
        assignment = self._publish_assignment()
        before = self._assignment(assignment.assignment_id)

        self.state.mode = "mock"
        self.page.set_role("student")
        self._run_and_stop_task()
        # 暂停/恢复同样不得修改正式状态
        self.state.mode = "mock"
        self.page._task_active = True
        self.page._on_task_pause_resume()
        self.page._on_task_pause_resume()
        self.page._on_task_stop()

        after = self._assignment(assignment.assignment_id)
        self.assertEqual(after, before)  # 全字段未变（status / completed_at 均无写入）

    # ── 10：普通自主学习（无 Assignment）Demo 无副作用 ──

    def test_10_free_demo_task_no_assignment_side_effects(self):
        assignments_before = self.assignments.list_all()
        self.state.mode = "mock"
        self.page.set_role("student")
        self.assertEqual(self.page._selected_assignment_id, "")
        self.page._combo_task.setCurrentText("自由学习")
        self._run_and_stop_task()
        # 无 Assignment 关联：不产生任何正式任务状态变化
        self.assertEqual(self.assignments.list_all(), assignments_before)
        records = self._history_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["tasks"][0]["assignment_id"], "")

    # ── 补充：live 模式开始后状态为 running（正常流转不被误伤） ──

    def test_live_running_status_flows_normally(self):
        assignment = self._publish_assignment()
        self.state.mode = "live"
        self.page.set_role("student")
        self.page._on_task_start()
        self.assertEqual(
            self._assignment(assignment.assignment_id).status, "running"
        )
        # 暂停 → paused；恢复 → running（仅 live）
        self.page._on_task_pause_resume()
        self.assertEqual(
            self._assignment(assignment.assignment_id).status, "paused"
        )
        self.page._on_task_pause_resume()
        self.assertEqual(
            self._assignment(assignment.assignment_id).status, "running"
        )
        self.page._on_task_stop()
        self.assertEqual(
            self._assignment(assignment.assignment_id).status, "completed"
        )


if __name__ == "__main__":
    unittest.main()
