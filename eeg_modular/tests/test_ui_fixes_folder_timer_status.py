"""UI 预检修复：会话文件夹定位 / 跨身份计时隔离 / 任务状态同步。

BUG 1：打开会话文件夹必须复用 SessionStore 真实根目录。
BUG 2：管理员诊断或其他学生的任务计时不得串到新身份页面。
BUG 3：任务开始后任务选择区显示实际状态，不再停留"待开始"。
规则 4：结束后保留最终有效时长并明确标记完成。
"""

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

from services.dashboard_state import DashboardState
from services.identity_store import IdentityStore
from services.session_store import SessionStore, resolve_sessions_root
from services.teaching_store import AssignmentStore, StudentRuntimeRegistry
from pages.task_page import TaskPage

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class Service:
    def __init__(self, state, sessions_dir="data/sessions"):
        self.state = state
        self.sessions_dir = sessions_dir

    def start_session(self):
        return self.state.begin_session(source="live", demo=False)

    def end_session(self, status="completed"):
        return self.state.finalize_session(status=status)

    def pause_session(self):
        self.state.set_session_paused(True)

    def resume_session(self):
        self.state.set_session_paused(False)


# ════════════════════════════ BUG 1：会话文件夹 ════════════════════════════


class FolderRootTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _history_page(self, state):
        from pages.history_page import HistoryPage
        return HistoryPage(state, Service(state, sessions_dir=self.root))

    def test_01_open_folder_matches_session_store_root(self):
        state = DashboardState(seed_demo_history=False)
        # 模拟真实持久化位置：configure 后 store.root 即权威目录
        state.configure_session_store(self.root)
        self.assertEqual(state._session_store.root, self.root)

        page = self._history_page(state)
        with patch("pages.history_page.QDesktopServices.openUrl") as opener:
            page._open_sessions_folder()
        opened = Path(opener.call_args[0][0].toLocalFile())
        self.assertEqual(opened, self.root)
        # 与 SessionStore.root 完全一致
        self.assertEqual(opened, state._session_store.root)

    def test_02_open_folder_never_opens_ui_prototype_data_dir(self):
        wrong_dir = PACKAGE_ROOT / "ui_prototype" / "data" / "sessions"
        state = DashboardState(seed_demo_history=False)
        # 默认配置：真实根 = eeg_modular/data/sessions（非 CWD 解析）
        real_dir = state._sessions_dir
        self.assertEqual(real_dir, PACKAGE_ROOT / "data" / "sessions")
        self.assertNotEqual(real_dir, wrong_dir)

        # 即使进程 CWD 位于 ui_prototype，也不得解析到错误目录
        from pages.history_page import HistoryPage
        page = HistoryPage(state, Service(state, sessions_dir="data/sessions"))
        cwd = os.getcwd()
        try:
            os.chdir(PACKAGE_ROOT / "ui_prototype")
            with patch("pages.history_page.QDesktopServices.openUrl") as opener:
                page._open_sessions_folder()
            opened = Path(opener.call_args[0][0].toLocalFile())
        finally:
            os.chdir(cwd)
        self.assertNotEqual(opened, wrong_dir)
        self.assertEqual(opened, real_dir.resolve())

    def test_03_three_entries_resolved_unified(self):
        """学生(History)/管理员(Replay/Settings)三入口解析一致。"""
        from pages.history_page import HistoryPage
        from pages.replay_page import ReplayPage
        from pages.settings_page import SettingsPage

        state = DashboardState(seed_demo_history=False)
        state.configure_session_store(self.root)
        service = Service(state, sessions_dir=self.root)

        targets = []
        for page_cls in (HistoryPage, ReplayPage, SettingsPage):
            page = page_cls(state, service)
            module = f"pages.{page_cls.__module__.split('.')[-1]}"
            with patch(f"{module}.QDesktopServices.openUrl") as opener:
                page._open_sessions_folder()
            targets.append(Path(opener.call_args[0][0].toLocalFile()))
        self.assertEqual(len(set(targets)), 1)
        self.assertEqual(targets[0], self.root)

    def test_04_demo_live_storage_structure_unchanged(self):
        """修复只改打开目录，不改 Demo/Live 存储结构。"""
        store = SessionStore(self.root)
        formal = DashboardState(seed_demo_history=False)
        formal.begin_session(source="live", demo=False)
        formal.finalize_session(status="completed")
        record = formal._active_record or formal._history_sessions[-1]
        store.save(record)

        demo_state = DashboardState(
            sessions_dir=self.root, seed_demo_history=False
        )
        demo_state.begin_session(source="mock", demo=True)
        demo_state.finalize_session(status="completed")
        store.save(demo_state._active_record or demo_state._history_sessions[-1])

        self.assertEqual(store.demo_root, self.root / "_demo")
        formal_ids = {p["session_id"] for p in store.load(include_demo=False)}
        all_ids = {p["session_id"] for p in store.load(include_demo=True)}
        demo_ids = all_ids - formal_ids
        self.assertEqual(len(formal_ids), 1)
        self.assertEqual(len(demo_ids), 1)
        self.assertEqual(all_ids, formal_ids | demo_ids)


# ══════════════════════ BUG 2：跨身份计时隔离 ══════════════════════


class IdentitySwitchTimerTest(unittest.TestCase):
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

    def tearDown(self):
        self.temp.cleanup()

    def _page(self, user_id: str, role: str):
        state = DashboardState(
            sessions_dir=self.root / f"sessions-{user_id}",
            seed_demo_history=False,
        )
        state._user_id = user_id
        state._user_name = user_id
        state._user_role = state.current_role = role
        page = TaskPage(
            state, Service(state), assignment_store=self.assignments,
            runtime_registry=self.registry, identity_store=self.identities,
        )
        return state, page

    def _run_and_finish_task(self, page, seconds: float, name="本机诊断"):
        """跑一个任务并结束：时长固定为 seconds 秒。"""
        state = page.state
        state.begin_session(source="live", demo=False)
        page._task_owns_session = False
        page._task_active = True
        state.task_running = True
        state.begin_task(name, "medium", "")
        task = state._task_by_id(state.current_task_id)
        task._running_since = None
        task.duration_seconds = seconds  # 模拟已运行 seconds 秒
        page.update_state(state)  # 计时显示刷新
        page._on_task_stop()
        return task

    def test_01_admin_diagnostic_not_leaked_to_student(self):
        state, page = self._page("admin_001", "research")
        task = self._run_and_finish_task(page, 7.0)
        # 管理员页面保留最终时长
        self.assertEqual(page._task_time.text(), "00:07")
        self.assertEqual(task.duration_seconds, 7.0)

        # 切换为学生
        state._user_id = "st_001"
        state.current_role = "student"
        page.set_role("student")
        self.assertEqual(page._task_time.text(), "00:00")
        self.assertEqual(page._task_status.text(), "未开始")
        self.assertFalse(page._task_active)

    def test_02_student_a_timer_not_inherited_by_student_b(self):
        state, page = self._page("st_001", "student")
        task = self._run_and_finish_task(page, 7.0, name="英语阅读")
        self.assertEqual(page._task_time.text(), "00:07")

        # 切换到学生 B（同角色，set_identity 仍会调用 set_role）
        state._user_id = "st_002"
        state.current_role = "student"
        page.set_role("student")
        self.assertEqual(page._task_time.text(), "00:00")
        self.assertEqual(page._task_status.text(), "未开始")

    def test_03_finished_duration_retained_on_same_identity(self):
        state, page = self._page("st_001", "student")
        self._run_and_finish_task(page, 7.0, name="英语阅读")
        # 不切换身份：多次 update_state 后最终时长仍保留
        for _ in range(3):
            page.update_state(state)
        self.assertEqual(page._task_time.text(), "00:07")
        self.assertEqual(page._task_status.text(), "已完成")

    def test_04_new_task_starts_from_zero(self):
        state, page = self._page("st_001", "student")
        self._run_and_finish_task(page, 7.0, name="英语阅读")
        # 开始新任务：从 00:00 重新计时
        page._task_active = True
        page._task_owns_session = False
        state.task_running = True
        state.begin_task("物理复习", "medium", "")
        page.update_state(state)
        self.assertEqual(page._task_time.text(), "00:00")
        self.assertEqual(page._task_status.text(), "进行中")

    def test_05_history_duration_untouched_by_identity_switch(self):
        state, page = self._page("admin_001", "research")
        task = self._run_and_finish_task(page, 7.0)
        task_dict_before = task.to_dict()
        history_before = len(state._history_sessions)

        state._user_id = "st_001"
        state.current_role = "student"
        page.set_role("student")
        page.update_state(state)

        self.assertEqual(task.to_dict(), task_dict_before)
        self.assertEqual(task.duration_seconds, 7.0)
        self.assertEqual(len(state._history_sessions), history_before)

    def test_06_reset_restores_button_states(self):
        state, page = self._page("admin_001", "research")
        page._task_active = True
        page._btn_task_stop.setEnabled(True)
        page._btn_task_pause.setEnabled(True)
        state._user_id = "st_001"
        state.current_role = "student"
        page.set_role("student")
        self.assertTrue(page._btn_task_start.isEnabled())
        self.assertFalse(page._btn_task_stop.isEnabled())
        self.assertFalse(page._btn_task_pause.isEnabled())


# ══════════════ BUG 3 + 规则 4：任务状态显示同步 ══════════════


class TaskStatusDisplayTest(unittest.TestCase):
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
        # 发布一个教师任务，学生进入后自动选中
        self.assignment = self.assignments.publish(
            "teacher_001", "st_001", "英语阅读", "medium", "第 3 章",
        )
        self.page.set_role("student")

    def tearDown(self):
        self.temp.cleanup()

    def _start_task(self):
        self.page._on_task_start()

    def test_01_running_shown_immediately_after_start(self):
        self.assertIn("待开始", self.page._assignment_summary.text())
        self._start_task()
        # 立即（不等下一刷新周期）显示进行中
        self.assertIn("进行中", self.page._assignment_summary.text())
        self.assertIn("英语阅读", self.page._assignment_summary.text())
        self.assertIn("教师布置", self.page._assignment_summary.text())
        self.assertEqual(self.page._task_status.text(), "进行中")
        # 任务下拉框同步标识
        combo_text = self.page._combo_assignment.currentText()
        self.assertIn("进行中", combo_text)
        self.assertNotIn("待开始", combo_text)

    def test_02_pause_shows_paused(self):
        self._start_task()
        self.state.set_session_paused(True)
        self.page.update_state(self.state)
        self.assertIn("已暂停", self.page._assignment_summary.text())
        self.assertEqual(self.page._task_status.text(), "已暂停")

    def test_03_resume_shows_running_again(self):
        self._start_task()
        self.state.set_session_paused(True)
        self.page.update_state(self.state)
        self.state.set_session_paused(False)
        self.page.update_state(self.state)
        self.assertIn("进行中", self.page._assignment_summary.text())
        self.assertEqual(self.page._task_status.text(), "进行中")

    def test_04_finish_shows_completed_with_final_duration(self):
        self._start_task()
        task = self.state._task_by_id(self.state.current_task_id)
        task._running_since = None
        task.duration_seconds = 97.0
        self.page.update_state(self.state)  # 计时显示刷新为 01:37
        self.assertEqual(self.page._task_time.text(), "01:37")
        self.page._on_task_stop()
        # 规则 4：保留最终有效时长 + 明确标记完成
        self.assertEqual(self.page._task_status.text(), "已完成")
        self.assertEqual(self.page._task_time.text(), "01:37")
        self.assertEqual(task.status, "completed")
        self.assertEqual(self.assignments.list_all()[0].status, "completed")

    def test_05_status_consistent_after_page_switch(self):
        self._start_task()
        for _ in range(3):
            self.page.update_state(self.state)
            self.page.on_show()
        self.assertIn("进行中", self.page._assignment_summary.text())
        self.assertEqual(self.page._task_status.text(), "进行中")

    def test_06_no_task_shows_pending(self):
        # 无任务时任务状态为"未开始"，时长 00:00
        self.assertEqual(self.page._task_status.text(), "未开始")
        self.assertEqual(self.page._task_time.text(), "00:00")


if __name__ == "__main__":
    unittest.main()
