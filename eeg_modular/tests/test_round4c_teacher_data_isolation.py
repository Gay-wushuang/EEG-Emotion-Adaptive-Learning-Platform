"""Round 4C：教师端学生数据隔离与当前学生同步回归测试。

规则：教师端所有"学生相关数据"只来自
TeacherSelectionContext.selected_student_id，不得混入其他学生、
demo_user、教师自身或管理员历史。

覆盖：
1. 教师选 st_001 → 历史只含 st_001
2. 教师选 st_002 → 历史只含 st_002
3. st_001 → st_002 → st_001 切换后历史立即同步（监听器驱动）
4. 切换学生后"教师端·学生过程概览"立即更新学生 ID
5. demo_user 不出现在教师普通学生历史
6. demo_user 不出现在教师学生选择列表（students_for / add 双重防线）
7. 学生端本人历史不受影响
8. 管理员本机诊断历史不受影响
9. 教学演示模式仍可正常使用（选中学生的 mock 记录仍可见）
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

from pages.dashboard_page import DashboardPage
from pages.history_page import HistoryPage
from services.dashboard_state import DashboardState, SessionRecord
from services.identity_store import IdentityStore
from services.teaching_store import (
    StudentRuntimeRegistry, TeacherSelectionContext, TeacherStudentStore,
)


class Service:
    def __init__(self, state, sessions_dir):
        self.state = state
        self.sessions_dir = str(sessions_dir)

    def start_session(self): pass
    def end_session(self, status="completed"): pass
    def pause_session(self): pass
    def resume_session(self): pass


def _record(session_id, user_id, *, source="live", demo=False,
            start="2026-09-14 10:00:00", attention=55.0):
    return SessionRecord(
        session_id=session_id, user_id=user_id, source=source, demo=demo,
        start_time=start, end_time=start, duration_seconds=600.0,
        signal_quality=0.9, avg_attention=attention, avg_meditation=45.0,
    )


def _mixed_records():
    """覆盖全部应被隔离的身份来源。"""
    return [
        _record("s_st1_live", "st_001"),
        _record("s_st1_mock", "st_001", source="mock", demo=True),
        _record("s_st2_live", "st_002", start="2026-09-14 11:00:00", attention=66.0),
        _record("s_st2_mock", "st_002", source="mock", demo=True),
        _record("s_demo_user", "demo_user", source="mock", demo=True),
        _record("s_demo_live", "demo_user"),  # 历史遗留的默认身份正式记录
        _record("s_teacher", "teacher_01"),
        _record("s_admin", "admin_01"),
    ]


class Round4CTeacherDataIsolationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.identities = IdentityStore(self.root / "identities.json")
        for uid, name in (("teacher_01", "教师一"), ("st_001", "学生一"),
                          ("st_002", "学生二"), ("admin_01", "管理员")):
            self.identities.save_profile(uid, name, make_current=False)
        self.bindings = TeacherStudentStore(
            self.root / "bindings.json", identity_store=self.identities
        )
        self.bindings.add("teacher_01", "st_001")
        self.bindings.add("teacher_01", "st_002")
        self.registry = StudentRuntimeRegistry(self.root / "coordination.sqlite3")
        self.selection = TeacherSelectionContext()
        self.selection.set_teacher("teacher_01")

    def tearDown(self):
        self.temp.cleanup()

    def _state(self, uid, role):
        state = DashboardState(
            sessions_dir=self.root / uid, seed_demo_history=False
        )
        state._user_id = state._user_name = uid
        state.current_role = state._user_role = role
        return state

    def _history_page(self, state):
        return HistoryPage(
            state, Service(state, self.root / "sessions"),
            selection_context=self.selection,
        )

    # ── 1 / 2 / 5：教师历史严格按当前选中学生过滤 ──

    def test_01_teacher_selecting_st_001_shows_only_st_001(self):
        state = self._state("teacher_01", "teacher")
        state._history_sessions = _mixed_records()
        page = self._history_page(state)
        self.selection.select("st_001")
        ids = [s.session_id for s in page._get_sorted_sessions()]
        self.assertEqual(ids, ["s_st1_live", "s_st1_mock"])
        self.assertNotIn("s_demo_user", ids)
        self.assertNotIn("s_demo_live", ids)

    def test_02_teacher_selecting_st_002_shows_only_st_002(self):
        state = self._state("teacher_01", "teacher")
        state._history_sessions = _mixed_records()
        page = self._history_page(state)
        self.selection.select("st_002")
        ids = [s.session_id for s in page._get_sorted_sessions()]
        self.assertEqual(ids, ["s_st2_live", "s_st2_mock"])

    def test_05_demo_user_never_enters_teacher_history(self):
        state = self._state("teacher_01", "teacher")
        state._history_sessions = _mixed_records()
        page = self._history_page(state)
        for student_id in ("st_001", "st_002"):
            self.selection.select(student_id)
            users = {s.user_id for s in page._get_sorted_sessions()}
            self.assertEqual(users, {student_id})
        # 未选择学生时为空，绝不回退到"全部学生"
        self.selection.select("")
        self.assertEqual(page._get_sorted_sessions(), [])

    # ── 3：切换学生后历史表立即同步（监听器驱动，无需重新进入页面）──

    def test_03_history_table_refreshes_immediately_on_switch(self):
        state = self._state("teacher_01", "teacher")
        records = _mixed_records()
        state._history_sessions = records
        # on_show 会 reload_history；测试中打桩避免读取磁盘替换内存记录。
        state.reload_history = lambda **kw: list(records)
        page = self._history_page(state)
        page.on_show()
        page._combo_source.setCurrentIndex(0)  # 全部来源
        # select() 本身应通过监听器触发页面刷新，不手动调用 _refresh_table
        self.selection.select("st_001")
        self.assertEqual(page._table.rowCount(), 2)
        self.selection.select("st_002")
        self.assertEqual(page._table.rowCount(), 2)
        users = {s.user_id for s in page._get_sorted_sessions()}
        self.assertEqual(users, {"st_002"})
        self.selection.select("st_001")
        users = {s.user_id for s in page._get_sorted_sessions()}
        self.assertEqual(users, {"st_001"})
        # 近期概览同样跟随当前学生（st_001 平均专注度 55.0 / st_002 为 66.0）
        self.assertIn("55.0", page._recent_body.text())
        self.assertNotIn("66.0", page._recent_body.text())
        self.selection.select("st_002")
        self.assertIn("66.0", page._recent_body.text())

    def test_03b_history_overview_prompts_when_no_student_selected(self):
        state = self._state("teacher_01", "teacher")
        state._history_sessions = _mixed_records()
        page = self._history_page(state)
        self.selection.select("")
        page._refresh_recent_overview()
        self.assertEqual(page._recent_body.text(),
                         "请先在教师工作台或学生实时观察页选择学生。")

    # ── 4：切换学生后"教师端·学生过程概览"立即更新 ──

    def test_04_process_overview_follows_selection_immediately(self):
        student = self._state("st_001", "student")
        student.attention, student.meditation = 62, 44
        student.quality_level, student.stable_state = "trusted", "positive"
        student.begin_session(source="live", demo=False)
        self.registry.publish(student, force=True)

        teacher = self._state("teacher_01", "teacher")
        page = DashboardPage(
            teacher, Service(teacher, self.root / "sessions"),
            identity_store=self.identities, binding_store=self.bindings,
            runtime_registry=self.registry, selection_context=self.selection,
        )
        page.set_role("teacher")

        # st_001 有实时快照 → 概览显示 st_001
        self.selection.select("st_001")
        self.assertIn("学生：st_001", page._focus_values[0].text())

        # 切到无快照的 st_002 → 概览必须立即跟随（不得残留 st_001）
        self.selection.select("st_002")
        text = page._focus_values[0].text()
        self.assertIn("st_002", text)
        self.assertNotIn("st_001", text)

        # 切回 st_001 → 立即恢复
        self.selection.select("st_001")
        self.assertIn("学生：st_001", page._focus_values[0].text())

    # ── 6：demo_user 不进入教师学生选择列表 ──

    def test_06_demo_user_not_in_teacher_student_list(self):
        # students_for 只返回 st_ 前缀绑定（即使历史数据被手写污染）
        raw = self.root / "polluted.json"
        raw.write_text(
            '{"version": 1, "bindings": {"teacher_01": '
            '["st_001", "demo_user", "st_002", "teacher_02"]}}',
            encoding="utf-8",
        )
        polluted = TeacherStudentStore(raw, identity_store=self.identities)
        self.assertEqual(polluted.students_for("teacher_01"), ["st_001", "st_002"])
        # demo_user 无有效学生身份，不能被绑定
        with self.assertRaises(ValueError):
            self.bindings.add("teacher_01", "demo_user")
        # 正常绑定不受影响
        self.assertIn("st_001", self.bindings.students_for("teacher_01"))

    # ── 7：学生端本人历史不受影响 ──

    def test_07_student_history_unchanged(self):
        state = self._state("st_001", "student")
        state._history_sessions = _mixed_records()
        page = self._history_page(state)
        ids = {s.session_id for s in page._get_sorted_sessions()}
        self.assertEqual(ids, {"s_st1_live", "s_st1_mock"})
        # 学生看得到自己的教学演示记录（演示模式不受教师过滤影响）
        page._combo_source.setCurrentIndex(2)  # 教学演示
        self.assertEqual(
            {s.session_id for s in page._get_sorted_sessions()}, {"s_st1_mock"}
        )

    # ── 8：管理员本机诊断历史不受影响 ──

    def test_08_admin_history_unchanged(self):
        state = self._state("admin_01", "research")
        state._history_sessions = _mixed_records()
        page = self._history_page(state)
        ids = {s.session_id for s in page._get_sorted_sessions()}
        self.assertEqual(ids, {r.session_id for r in _mixed_records()})

    # ── 9：教学演示模式仍可正常使用 ──

    def test_09_demo_mode_still_available_for_teacher(self):
        state = self._state("teacher_01", "teacher")
        state._history_sessions = _mixed_records()
        page = self._history_page(state)
        self.selection.select("st_001")
        # 教师主动切到"教学演示"来源 → 仍能看到该学生的演示记录
        page._combo_source.setCurrentIndex(2)
        ids = [s.session_id for s in page._get_sorted_sessions()]
        self.assertEqual(ids, ["s_st1_mock"])
        self.assertTrue(all(s.demo for s in page._get_sorted_sessions()))
        # 切回"实时采集" → 只剩正式记录
        page._combo_source.setCurrentIndex(1)
        self.assertEqual(
            [s.session_id for s in page._get_sorted_sessions()], ["s_st1_live"]
        )

    # ── 附：TeacherSelectionContext 监听机制 ──

    def test_10_selection_context_notifies_and_dedupes(self):
        class _Owner:
            def __init__(self):
                self.calls = []

            def on_change(self, student_id):
                self.calls.append(student_id)

        owner = _Owner()
        self.selection.add_listener(owner, "on_change")
        self.selection.select("st_001")
        self.selection.select("st_001")  # 重复选择不重复通知
        self.assertEqual(owner.calls, ["st_001"])
        self.selection.select("st_002")
        self.assertEqual(owner.calls, ["st_001", "st_002"])
        self.assertEqual(self.selection.selected_student_id, "st_002")
        # 换教师后选择复位
        self.selection.set_teacher("teacher_02")
        self.assertEqual(self.selection.selected_student_id, "")
        # 弱引用：持有者被回收后监听器自动移除，不抛错
        del owner
        self.selection.select("st_001")  # 不应抛 RuntimeError


if __name__ == "__main__":
    unittest.main()
