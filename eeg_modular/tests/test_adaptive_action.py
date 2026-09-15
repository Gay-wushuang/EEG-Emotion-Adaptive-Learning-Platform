"""AdaptiveAction 行为测试 - 自适应学习场景接口。

测试覆盖（AGENTS.md §9 最小自动测试要求）：
  1. hard -> medium
  2. medium -> easy
  3. easy -> suggest_break
  4. rejected quality 不触发动作
  5. 一次动作只产生一次 intervention event
  6. reset_session 后 adaptive 状态正确复位
  7. TaskPage 能正确响应 DashboardState 的难度变化

运行方式（在 eeg_modular 目录下）：
    .venv\\Scripts\\python.exe -m unittest tests.test_adaptive_action -v
"""

import os
import sys
import time
import unittest

# 确保 ui_prototype 目录在 sys.path 中
_UI_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ui_prototype",
)
if _UI_DIR not in sys.path:
    sys.path.insert(0, _UI_DIR)

# 无显示器环境支持
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _import_qt():
    try:
        from PySide6.QtWidgets import QApplication
        return QApplication
    except ImportError as e:
        raise unittest.SkipTest(f"PySide6未安装，跳过测试: {e}")


class AdaptiveActionTest(unittest.TestCase):
    """AdaptiveAction 决策行为测试。"""

    @classmethod
    def setUpClass(cls):
        QApplication = _import_qt()
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def _make_state(self, difficulty="hard", quality_level="trusted"):
        """构造一个处于可推理状态的 DashboardState。

        使用正式字段而非内部簿记，避免依赖 EWMA 等时序逻辑。
        """
        from services.dashboard_state import (
            DashboardState,
            DIFFICULTY_HARD, DIFFICULTY_MEDIUM, DIFFICULTY_EASY,
            AdaptiveAction,
        )

        state = DashboardState()
        state.warmup_progress = 1.0           # 预热完成
        state.quality_level = quality_level  # trusted / warning / rejected
        state.poor_signal = 10 if quality_level != "rejected" else None
        state.task_difficulty = difficulty
        state.attention = 40.0  # 偏低，用于触发 "Attention 偏低" 原因
        state.meditation = 30.0
        # 让 inference_eligible 为 True（除非 quality_level == rejected）
        return state

    def _make_service(self, state):
        """构造 MockDataService 但不启动后台线程。

        测试只调用 _apply_adaptive_action()，不依赖采集/推理线程。
        """
        from services.mock_data_service import MockDataService
        svc = MockDataService(state)
        # 不调用 start_streaming，避免后台线程干扰
        return svc

    # ── 测试 1: hard -> medium ──
    def test_01_hard_to_medium(self):
        """hard 难度触发后只建议降低负荷，不修改正式难度。"""
        from services.dashboard_state import (
            DIFFICULTY_MEDIUM, AdaptiveAction,
        )
        state = self._make_state(difficulty="hard")
        svc = self._make_service(state)
        svc._apply_adaptive_action()

        self.assertEqual(state.task_difficulty, "hard")
        self.assertEqual(state.adaptive_action, AdaptiveAction.REDUCE_DIFFICULTY)
        self.assertTrue(state.adaptive_action_reason)
        self.assertIsNotNone(state.adaptive_action_time)
        self.assertIn("建议", state.adaptive_feedback_text)
        self.assertIn("保持不变", state.adaptive_feedback_text)

    # ── 测试 2: medium -> easy ──
    def test_02_medium_to_easy(self):
        """medium 难度触发后只建议降低负荷，不修改正式难度。"""
        from services.dashboard_state import (
            DIFFICULTY_EASY, AdaptiveAction,
        )
        state = self._make_state(difficulty="medium")
        svc = self._make_service(state)
        svc._apply_adaptive_action()

        self.assertEqual(state.task_difficulty, "medium")
        self.assertEqual(state.adaptive_action, AdaptiveAction.REDUCE_DIFFICULTY)
        self.assertIn("建议", state.adaptive_feedback_text)
        self.assertNotIn("已从", state.adaptive_feedback_text)

    # ── 测试 3: easy -> suggest_break ──
    def test_03_easy_to_suggest_break(self):
        """easy 难度触发后不应继续降低，应转为 suggest_break。"""
        from services.dashboard_state import (
            DIFFICULTY_EASY, AdaptiveAction,
        )
        state = self._make_state(difficulty="easy")
        svc = self._make_service(state)
        svc._apply_adaptive_action()

        # 难度保持 easy，不继续降
        self.assertEqual(state.task_difficulty, DIFFICULTY_EASY)
        self.assertEqual(state.adaptive_action, AdaptiveAction.SUGGEST_BREAK)
        self.assertIn("休息", state.adaptive_feedback_text)
        self.assertIn("最低难度", state.adaptive_action_reason)

    # ── 测试 4: rejected quality 不触发动作 ──
    def test_04_rejected_no_action(self):
        """quality_level=rejected 时不得改变难度或产生 intervention。"""
        from services.dashboard_state import AdaptiveAction
        state = self._make_state(difficulty="hard", quality_level="rejected")
        svc = self._make_service(state)
        svc._apply_adaptive_action()

        # 难度未变
        self.assertEqual(state.task_difficulty, "hard")
        # action 保持 NONE
        self.assertEqual(state.adaptive_action, AdaptiveAction.NONE)
        # 无 intervention 事件
        interventions = [e for e in state._events if e.category == "intervention"]
        self.assertEqual(len(interventions), 0)

    # ── 测试 5: 一次动作只产生一次 intervention event ──
    def test_05_one_event_per_decision(self):
        """_apply_adaptive_action 一次调用只能产生一次 intervention 事件。"""
        state = self._make_state(difficulty="hard")
        svc = self._make_service(state)

        before = len([e for e in state._events if e.category == "intervention"])
        svc._apply_adaptive_action()
        after = len([e for e in state._events if e.category == "intervention"])

        self.assertEqual(after - before, 1,
                         "一次决策应只产生一次 intervention event")

        # 再调用一次（同一决策）不应在冷却内重复触发
        # 这里直接调用 _apply_adaptive_action 模拟重复刷新
        svc._apply_adaptive_action()
        after2 = len([e for e in state._events if e.category == "intervention"])
        # _apply_adaptive_action 自身不防冷却（由 _update_stable_state 守护），
        # 但单次调用只 add_event 一次，所以 after2 - after == 1
        self.assertEqual(after2 - after, 1,
                         "_apply_adaptive_action 单次调用只 add_event 一次")

    # ── 测试 6: reset_session 后 adaptive 状态复位 ──
    def test_06_reset_session_adaptive(self):
        """reset_session 后 adaptive_* 必须复位，task_type/task_difficulty 保留。"""
        from services.dashboard_state import (
            DIFFICULTY_MEDIUM, AdaptiveAction,
        )
        state = self._make_state(difficulty="hard")
        svc = self._make_service(state)
        svc._apply_adaptive_action()

        # 触发后状态已变化
        self.assertEqual(state.task_difficulty, "hard")
        self.assertNotEqual(state.adaptive_action, AdaptiveAction.NONE)

        # reset_session
        state.reset_session()

        # adaptive_* 复位
        self.assertEqual(state.adaptive_action, AdaptiveAction.NONE)
        self.assertEqual(state.adaptive_action_reason, "")
        self.assertIsNone(state.adaptive_action_time)
        self.assertEqual(state.adaptive_feedback_text, "")
        # task_running 也复位
        self.assertFalse(state.task_running)
        # task_difficulty 保留用户选择（不应被 reset 清空）
        self.assertEqual(state.task_difficulty, "hard")

    # ── 测试 7: TaskPage 能正确响应 DashboardState 的难度变化 ──
    def test_07_taskpage_resyncs_difficulty(self):
        """state.task_difficulty 变化后，TaskPage._combo_diff 必须同步。"""
        from main_window import MainWindow
        from services.dashboard_state import (
            DIFFICULTY_EASY, DIFFICULTY_HARD, DIFFICULTY_MEDIUM,
        )

        window = MainWindow()
        try:
            window._navigate_to("task")
            task_page = window._pages["task"]
            state = window.state

            # 初始：state 默认 medium，combo 索引应为 1
            state.task_difficulty = DIFFICULTY_MEDIUM
            task_page.update_state(state)
            self.assertEqual(task_page._combo_diff.currentIndex(), 1)

            # 模拟 _apply_adaptive_action 把难度改为 easy
            state.task_difficulty = DIFFICULTY_EASY
            task_page.update_state(state)
            self.assertEqual(task_page._combo_diff.currentIndex(), 0,
                             "state 改为 easy 后 combo 应同步到 0")

            # 模拟改为 hard
            state.task_difficulty = DIFFICULTY_HARD
            task_page.update_state(state)
            self.assertEqual(task_page._combo_diff.currentIndex(), 2,
                             "state 改为 hard 后 combo 应同步到 2")
        finally:
            window.service.stop_streaming()
            window.close()
            window.deleteLater()
            self.app.processEvents()

    # ── 测试 8: rejected 时 TaskPage 反馈区不解释学习状态 ──
    def test_08_taskpage_rejected_no_interpretation(self):
        """quality_level=rejected 时 TaskPage 反馈区显示信号不足，不显示学习建议。"""
        from main_window import MainWindow

        window = MainWindow()
        try:
            window._navigate_to("task")
            task_page = window._pages["task"]
            state = window.state

            # 强制 rejected
            state.quality_level = "rejected"
            state.poor_signal = None
            state.adaptive_action = "reduce_difficulty"  # 即使有动作
            state.adaptive_feedback_text = "应被屏蔽的文本"
            task_page.update_state(state)

            sug_text = task_page._ai_suggestion.text()
            self.assertIn("信号质量不足", sug_text,
                          "rejected 时反馈区应显示信号不足，不显示学习状态解释")
            # 策略栏显示"暂不评估"
            self.assertEqual(task_page._ai_strategy.text(), "暂不评估")
        finally:
            window.service.stop_streaming()
            window.close()
            window.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main(verbosity=2)
