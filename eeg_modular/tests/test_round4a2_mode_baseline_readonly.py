"""Round 4A-2：模式切换刷新 / 基线推荐化 / 学生实时页只读化。

覆盖：
1. History 页面可见时 live→mock / mock→live 立即刷新筛选、列表与说明文字；
2. 模式切换后旧右侧详情被清除；
3. "全部"筛选仍可手动选择；
4. 无基线时"暂时跳过"可开始任务、"去采集基线"不开始任务；
5. 已有基线直接放行；无基线报告仍显示"暂无可用基线对比"；
6. 学生实时页无任务/难度编辑控件，只读展示任务/来源/难度/时长/建议；
7. 教师 Assignment 来源显示"教师布置"，自主学习显示"自主学习"。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui_prototype"))

from PySide6.QtWidgets import QApplication, QWidget

from services.dashboard_state import (
    DashboardState,
    SessionRecord,
    TaskRecord,
)
from services.learning_readiness import (
    baseline_advisory_confirmed,
    learning_start_block_reason,
)
from services.session_report_builder import MISSING_BASELINE, build_learning_report
from pages.history_page import HistoryPage
from pages.dashboard_page import DashboardPage


def _record(session_id, source):
    return SessionRecord(
        session_id=session_id, source=source, demo=(source == "mock"),
        user_id="st_001", user_name="学生一",
        start_time="2026-09-14 10:00:00", duration_seconds=120.0,
        tasks=[TaskRecord(name="英语阅读", difficulty="medium",
                          status="completed", duration_seconds=120.0)],
    )


def _ready_live_state():
    """live 模式、设备/信号/预热全部就绪、基线未完成。"""
    state = DashboardState(seed_demo_history=False)
    state.mode = "live"
    state._user_id = "st_001"
    state.current_role = "student"
    state.connector_status = "online"
    state.device_status = "online"
    state.poor_signal = 0
    state.quality_level = "trusted"
    state.warmup_progress = 1.0  # warmup_complete 为只读派生属性
    state.model_status = "READY"
    return state


class _StartService:
    """最小服务桩：start_session 触发真实 begin_session。"""

    def __init__(self, state):
        self.state = state

    def start_session(self):
        self.state.begin_session(source=self.state.mode)

    def end_session(self):
        pass

    def pause_session(self):
        pass

    def resume_session(self):
        pass


class _FakeWindow(QWidget):
    """带 _navigate_to 记录的假主窗口（parent.window() 返回自身）。"""

    def __init__(self):
        super().__init__()
        self.navigated = []

    def _navigate_to(self, key):
        self.navigated.append(key)


def _advisory_box(skip: bool):
    """构造可控制按钮选择的 QMessageBox 替身。"""
    box_cls = MagicMock()
    box = box_cls.return_value
    go_button, skip_button = MagicMock(), MagicMock()
    box.addButton.side_effect = (
        lambda text, role: go_button if text == "去采集基线" else skip_button
    )
    box.clickedButton.return_value = skip_button if skip else go_button
    return box_cls


class Round4A2HistoryModeRefreshTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def _make_page(self):
        state = DashboardState(seed_demo_history=False)
        state.mode = "live"
        state._history_sessions = [_record("live_1", "live"), _record("mock_1", "mock")]
        state.reload_history = lambda **kw: list(state._history_sessions)
        page = HistoryPage(state, SimpleNamespace(sessions_dir="data/sessions"))
        return state, page

    def test_01_live_to_mock_refreshes_immediately(self):
        state, page = self._make_page()
        page.on_show()  # live 模式默认筛选：实时采集
        self.assertEqual(page._combo_source.currentData(), "live")
        # 模式切换（用户停留在 History 页）→ 立即刷新
        state.mode = "mock"
        page.on_data_mode_changed("mock")
        self.assertEqual(page._combo_source.currentData(), "mock")
        self.assertEqual(
            page._demo_banner.text(),
            "当前显示教学演示记录，与正式学习记录分开保存。"
        )
        # 表格已无 session_id 列（6 列产品化列序），用数据源断言筛选结果
        ids = [s.session_id for s in page._get_sorted_sessions()]
        self.assertEqual(ids, ["mock_1"])

    def test_02_mock_to_live_refreshes_immediately(self):
        state, page = self._make_page()
        state.mode = "mock"
        page.on_show()  # mock 模式默认筛选：教学演示
        self.assertEqual(page._combo_source.currentData(), "mock")
        state.mode = "live"
        page.on_data_mode_changed("live")
        self.assertEqual(page._combo_source.currentData(), "live")
        self.assertEqual(page._demo_banner.text(), "当前显示实时采集学习记录。")
        ids = [s.session_id for s in page._get_sorted_sessions()]
        self.assertEqual(ids, ["live_1"])

    def test_03_mode_switch_clears_stale_detail(self):
        state, page = self._make_page()
        page.on_show()
        page._table.selectRow(0)
        self.assertNotEqual(page._detail_labels["session_id"].text(), "暂无数据")
        page.on_data_mode_changed("mock")
        for label in page._detail_labels.values():
            self.assertEqual(label.text(), "暂无数据")
        self.assertEqual(page._tasks_label.text(), "暂无数据")
        self.assertFalse(page._demo_badge.isVisible() and not page._demo_badge.isHidden())

    def test_03b_unselected_detail_shows_select_hint(self):
        """产品化空态：未选中显示提示语；选中后隐藏并填充详情。"""
        state, page = self._make_page()
        page.on_show()
        # 未选中：提示可见 + 字段为 暂无数据（不显示 -- / 0%）
        self.assertFalse(page._select_hint.isHidden())
        self.assertEqual(page._select_hint.text(),
                         "请选择一条历史学习记录查看详情。")
        self.assertEqual(page._detail_labels["start_time"].text(), "暂无数据")
        # 选中：提示隐藏
        page._table.selectRow(0)
        self.assertTrue(page._select_hint.isHidden())
        # 再次清空（模式切换）→ 提示恢复
        page.on_data_mode_changed("mock")
        self.assertFalse(page._select_hint.isHidden())

    def test_03c_empty_table_shows_placeholder(self):
        """无历史记录时表格切换到"暂无历史学习记录"占位页。"""
        state = DashboardState(seed_demo_history=False)
        state.mode = "live"
        state.current_role = "student"
        state._user_id = "st_001"
        state._history_sessions = []
        state.reload_history = lambda **kw: []
        page = HistoryPage(state, SimpleNamespace(sessions_dir="data/sessions"))
        page.on_show()
        self.assertEqual(page._table.rowCount(), 0)
        self.assertEqual(page._table.columnCount(), 6)  # 6 列产品化列序
        self.assertEqual(page._table_stack.currentIndex(), 1)
        self.assertEqual(page._table_empty_hint.text(), "暂无历史学习记录")

    def test_03d_table_has_no_session_id_column(self):
        """session_id 不再是表格首列：表头为产品化 6 列。"""
        state, page = self._make_page()
        page.on_show()
        headers = [page._table.horizontalHeaderItem(i).text()
                   for i in range(page._table.columnCount())]
        self.assertEqual(headers, ["开始时间", "任务", "有效时长", "信号质量",
                                   "事件数", "来源"])
        self.assertNotIn("会话ID", headers)
        # 第一列内容是开始时间而非 session_id
        self.assertNotEqual(page._table.item(0, 0).text(), "live_1")

    def test_04_all_filter_still_selectable(self):
        state, page = self._make_page()
        page.on_show()
        page._combo_source.setCurrentIndex(0)
        self.assertEqual(page._combo_source.currentData(), "all")
        self.assertEqual(page._table.rowCount(), 2)
        self.assertEqual(
            page._demo_banner.text(),
            "当前显示全部学习记录，教学演示与正式学习记录分别保存。"
        )

    def test_05_switch_data_mode_notifies_history_page(self):
        from main_window import MainWindow

        window = MainWindow(mode="mock", user_id="st_001",
                            user_name="学生", role="student")
        self.addCleanup(window.close)
        history = window._pages["history"]
        with patch.object(history, "on_data_mode_changed") as notified:
            self.assertTrue(window.switch_data_mode("live"))
            notified.assert_called_once_with("live")


class Round4A2BaselineAdvisoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_05_skip_allows_start(self):
        state = _ready_live_state()
        with patch("PySide6.QtWidgets.QMessageBox", _advisory_box(skip=True)):
            allowed = baseline_advisory_confirmed(state, None)
        self.assertTrue(allowed)
        self.assertTrue(state._baseline_skip_prompted)
        # 同一运行内不再重复打扰
        with patch("PySide6.QtWidgets.QMessageBox") as box:
            self.assertTrue(baseline_advisory_confirmed(state, None))
            box.assert_not_called()

    def test_06_go_baseline_does_not_start(self):
        state = _ready_live_state()
        parent = _FakeWindow()
        with patch("PySide6.QtWidgets.QMessageBox", _advisory_box(skip=False)):
            allowed = baseline_advisory_confirmed(state, parent)
        self.assertFalse(allowed)
        self.assertEqual(parent.navigated, ["baseline"])
        self.assertFalse(getattr(state, "_baseline_skip_prompted", False))

    def test_07_existing_baseline_allows_directly(self):
        state = _ready_live_state()
        state.baseline_status = "COMPLETED"
        with patch("PySide6.QtWidgets.QMessageBox") as box:
            self.assertTrue(baseline_advisory_confirmed(state, None))
            box.assert_not_called()

    def test_05b_dashboard_start_flow_with_skip(self):
        state = _ready_live_state()
        page = DashboardPage(state, _StartService(state))
        page.set_role("student")
        page.show()
        with patch("PySide6.QtWidgets.QMessageBox", _advisory_box(skip=True)):
            page._on_start()
        self.assertTrue(state.session_active)  # 暂时跳过 → 正常开始

    def test_06b_dashboard_start_flow_with_go_baseline(self):
        state = _ready_live_state()
        page = DashboardPage(state, _StartService(state))
        page.set_role("student")
        page.show()
        with patch("PySide6.QtWidgets.QMessageBox", _advisory_box(skip=False)):
            page._on_start()
        self.assertFalse(state.session_active)  # 去采集基线 → 不开始

    def test_08_report_without_baseline_keeps_placeholder(self):
        record = _record("live_1", "live")
        report = build_learning_report(record, None)
        self.assertFalse(report["baseline"]["available"])
        self.assertEqual(report["baseline"]["empty_text"], MISSING_BASELINE)

    def test_baseline_reason_still_available_for_legacy_callers(self):
        # learning_start_block_reason 默认语义不变（Round 3B 契约）
        state = _ready_live_state()
        self.assertIn("基线", learning_start_block_reason(state))
        self.assertEqual(
            learning_start_block_reason(state, require_baseline=False), ""
        )


class Round4A2StudentDashboardReadonlyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def _page_with_running_task(self, assignment_id=""):
        state = _ready_live_state()
        state.baseline_status = "COMPLETED"
        page = DashboardPage(state, _StartService(state))
        page.set_role("student")
        state.begin_session(source="live")
        state.begin_task(
            "英语阅读", "hard", assignment_id=assignment_id,
        )
        state.task_running = True
        page.update_state(state)
        return state, page

    def test_09_no_task_editing_widgets(self):
        state, page = self._page_with_running_task()
        # 任务下拉框 / 难度下拉框已删除，不存在任何编辑入口
        self.assertFalse(hasattr(page, "_learner_task"))
        self.assertFalse(hasattr(page, "_learner_difficulty"))
        self.assertFalse(hasattr(page, "_on_learner_task_changed"))
        self.assertFalse(hasattr(page, "_on_learner_difficulty_changed"))

    def test_10_update_state_does_not_modify_task_or_difficulty(self):
        state, page = self._page_with_running_task()
        before_task = state.task_type
        before_difficulty = state.task_difficulty
        before_events = len(state._events)
        page.update_state(state)
        page._refresh_role_focus(state)
        self.assertEqual(state.task_type, before_task)
        self.assertEqual(state.task_difficulty, before_difficulty)
        self.assertEqual(len(state._events), before_events)  # 不写事件

    def test_11_assignment_task_shows_name_difficulty_and_source(self):
        state, page = self._page_with_running_task(assignment_id="A001")
        text = page._learner_task_info.text()
        self.assertIn("任务：英语阅读", text)
        self.assertIn("难度：困难", text)
        self.assertIn("来源：教师布置", text)
        self.assertIn("有效学习时间：", text)
        self.assertIn("AI学习建议：", text)

    def test_12_free_task_shows_self_study_source(self):
        state, page = self._page_with_running_task(assignment_id="")
        text = page._learner_task_info.text()
        self.assertIn("来源：自主学习", text)
        self.assertNotIn("教师布置", text)

    def test_13_teacher_task_source_semantics(self):
        # assignment_id 非空 = 教师布置；该判断只读 TaskRecord，不改结构
        state, page = self._page_with_running_task(assignment_id="A009")
        task = page._current_task_record()
        self.assertEqual(task.assignment_id, "A009")
        self.assertIn("教师布置", page._learner_task_info.text())

    def test_no_task_shows_guidance(self):
        state = _ready_live_state()
        state.baseline_status = "COMPLETED"
        page = DashboardPage(state, _StartService(state))
        page.set_role("student")
        page.update_state(state)
        text = page._learner_task_info.text()
        self.assertIn("当前暂无进行中的学习任务", text)
        self.assertIn("请前往“我的学习任务”开始学习", text)


if __name__ == "__main__":
    unittest.main()
