"""Round 2H AI advice must not mutate teacher-defined work."""

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
from pages.task_page import TaskPage
from services.adaptive_feedback_engine import AdaptiveDecision, apply_adaptive_decision
from services.dashboard_state import DashboardState, DIFFICULTY_EASY, DIFFICULTY_MEDIUM
from services.identity_store import IdentityStore
from services.teaching_store import (
    AssignmentStore, StudentRuntimeRegistry, TeacherObserverService,
    TeacherSelectionContext, TeacherStudentStore,
)


class Round2HAIBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.identities = IdentityStore(self.root / "identities.json")
        for uid in ("teacher_01", "st_001"):
            self.identities.save_profile(uid, uid, make_current=False)
        self.bindings = TeacherStudentStore(
            self.root / "bindings.json", self.identities
        )
        self.bindings.add("teacher_01", "st_001")
        self.assignments = AssignmentStore(self.root / "assignments.json")
        self.registry = StudentRuntimeRegistry(self.root / "runtime.sqlite3")
        self.selection = TeacherSelectionContext()

    def tearDown(self):
        self.temp.cleanup()

    def test_advice_does_not_mutate_assignment_task_or_history(self):
        assignment = self.assignments.publish(
            "teacher_01", "st_001", "数学练习", DIFFICULTY_MEDIUM, "完成第1至5题"
        )
        state = DashboardState(
            sessions_dir=self.root / "sessions", seed_demo_history=False
        )
        state._user_id = state._user_name = "st_001"
        state.current_role = state._user_role = "student"
        state.quality_level = "trusted"
        state.begin_session(source="live", demo=False)
        task_id = state.begin_task(
            assignment.task_type, assignment.difficulty, assignment.note,
            assignment_id=assignment.assignment_id,
        )
        decision = AdaptiveDecision.reduce_difficulty(
            DIFFICULTY_MEDIUM, DIFFICULTY_EASY,
            "持续负性状态趋势且 Attention 偏低",
        )
        apply_adaptive_decision(state, decision)

        stored_assignment = self.assignments.for_student("st_001")[0]
        task = state._task_by_id(task_id)
        self.assertEqual(stored_assignment.difficulty, DIFFICULTY_MEDIUM)
        self.assertEqual(task.difficulty, DIFFICULTY_MEDIUM)
        self.assertEqual(state.task_difficulty, DIFFICULTY_MEDIUM)
        self.assertIn("建议", state.adaptive_feedback_text)
        self.assertIn("保持不变", state.adaptive_feedback_text)
        self.assertNotIn("已从", state.adaptive_feedback_text)
        advice_events = [
            event for event in state._events if event.category == "intervention"
        ]
        self.assertEqual(len(advice_events), 1)
        self.assertIn("AI建议", advice_events[0].label)
        self.assertIn("未自动修改", advice_events[0].note)

        state.end_task()
        state.finalize_session()
        record = state.history_sessions[-1]
        saved_task = next(item for item in record.tasks if item.task_id == task_id)
        self.assertEqual(saved_task.difficulty, DIFFICULTY_MEDIUM)
        self.assertTrue(any("AI建议" in event.label for event in record.events))
        self.assertFalse(any(
            "已修改" in event.label or "调整为" in event.note
            for event in record.events
        ))

    def test_student_and_teacher_ui_share_advisory_wording(self):
        state = DashboardState(
            sessions_dir=self.root / "student", seed_demo_history=False
        )
        state._user_id = state._user_name = "st_001"
        state.current_role = state._user_role = "student"
        state.quality_level = "trusted"
        state.begin_session(source="live", demo=False)
        state.begin_task("数学练习", DIFFICULTY_MEDIUM)
        apply_adaptive_decision(state, AdaptiveDecision.reduce_difficulty(
            DIFFICULTY_MEDIUM, DIFFICULTY_EASY,
            "持续负性状态趋势且 Attention 偏低",
        ))
        student_page = TaskPage(
            state, TeacherObserverService(), identity_store=self.identities,
            binding_store=self.bindings, assignment_store=self.assignments,
            runtime_registry=self.registry, selection_context=self.selection,
        )
        student_page._refresh_adaptive_feedback(state)
        self.assertEqual(student_page._ai_strategy.text(), "建议降低学习负荷")
        self.assertNotIn("已调整", student_page._ai_suggestion.text())

        self.registry.publish(state, force=True)
        teacher = DashboardState(
            sessions_dir=self.root / "teacher", seed_demo_history=False
        )
        teacher._user_id = teacher._user_name = "teacher_01"
        teacher.current_role = teacher._user_role = "teacher"
        teacher_page = TaskPage(
            teacher, TeacherObserverService(), identity_store=self.identities,
            binding_store=self.bindings, assignment_store=self.assignments,
            runtime_registry=self.registry, selection_context=self.selection,
        )
        self.selection.select("st_001")
        teacher_page._refresh_teacher_observation()
        self.assertEqual(
            teacher_page._teacher_advice.text(),
            f"AI 建议：{state.adaptive_feedback_text}",
        )
        self.assertIn("建议", teacher_page._teacher_advice.text())
        self.assertNotIn("已调整", teacher_page._teacher_advice.text())


if __name__ == "__main__":
    unittest.main()
