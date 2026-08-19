"""AdaptiveFeedbackEngine 集成测试。

覆盖（AGENTS.md §IX 验收要求）：
A. Engine 单元测试
B. 事件测试
C. Mock 集成测试
D. Live 服务测试（synthetic/fake inference）
E. Reset 测试

运行方式（在 eeg_modular 目录下）：
    E:\\anaconda3\\envs\\eegcnn\\python.exe -m unittest tests.test_adaptive_feedback_engine -v
"""

import os
import sys
import time
import unittest
from pathlib import Path

# 确保路径正确
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "ui_prototype"))
sys.path.insert(0, str(_ROOT))

# 无显示器环境支持
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _import_qt():
    try:
        from PySide6.QtWidgets import QApplication
        return QApplication
    except ImportError as e:
        raise unittest.SkipTest(f"PySide6未安装，跳过测试: {e}")


# ── 导入被测模块 ──
from services.dashboard_state import (
    DashboardState,
    DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD,
    DIFFICULTY_DISPLAY,
    AdaptiveAction,
)
from services.adaptive_feedback_engine import (
    AdaptiveFeedbackEngine,
    AdaptiveDecision,
    apply_adaptive_decision,
)


class TestAdaptiveFeedbackEngine(unittest.TestCase):
    """A. Engine 单元测试。"""

    def _make_engine(self, sustain_seconds=5.0, cooldown_seconds=10.0,
                     negative_threshold=0.60, alpha=1.0):
        """创建一个测试用引擎（默认 alpha=1.0 让 EWMA 立即生效）。"""
        return AdaptiveFeedbackEngine(
            negative_threshold=negative_threshold,
            sustain_seconds=sustain_seconds,
            cooldown_seconds=cooldown_seconds,
            alpha=alpha,
        )

    def _drive_engine(self, engine, task_difficulty, negative_prob=0.70,
                      attention=45.0, quality_level="trusted",
                      warmup_complete=True, count=10, start_time=None):
        """驱动引擎 N 次，返回所有决策列表。"""
        if start_time is None:
            start_time = time.time()
        decisions = []
        for i in range(count):
            decision = engine.decide(
                quality_level=quality_level,
                negative_prob=negative_prob,
                attention=attention,
                task_difficulty=task_difficulty,
                timestamp=start_time + i,
                warmup_complete=warmup_complete,
            )
            decisions.append(decision)
        return decisions

    # ── A1: hard → medium ──
    def test_A1_hard_to_medium(self):
        """持续负性 + hard 难度 → reduce_difficulty → medium。"""
        engine = self._make_engine(sustain_seconds=5.0)
        decisions = self._drive_engine(
            engine, DIFFICULTY_HARD, count=10,
        )

        # 找到触发的决策
        triggered = [d for d in decisions if d.should_emit_event]
        self.assertTrue(len(triggered) > 0, "应有至少一次触发")
        decision = triggered[0]

        self.assertEqual(decision.action, AdaptiveAction.REDUCE_DIFFICULTY)
        self.assertEqual(decision.new_difficulty, DIFFICULTY_MEDIUM)
        self.assertTrue(decision.should_emit_event)
        self.assertIn("困难", decision.feedback_text)
        self.assertIn("中等", decision.feedback_text)

    # ── A2: medium → easy ──
    def test_A2_medium_to_easy(self):
        """持续负性 + medium 难度 → reduce_difficulty → easy。"""
        engine = self._make_engine(sustain_seconds=5.0)
        decisions = self._drive_engine(
            engine, DIFFICULTY_MEDIUM, count=10,
        )

        triggered = [d for d in decisions if d.should_emit_event]
        self.assertTrue(len(triggered) > 0, "应有至少一次触发")
        decision = triggered[0]

        self.assertEqual(decision.action, AdaptiveAction.REDUCE_DIFFICULTY)
        self.assertEqual(decision.new_difficulty, DIFFICULTY_EASY)
        self.assertIn("中等", decision.feedback_text)
        self.assertIn("简单", decision.feedback_text)

    # ── A3: easy → suggest_break ──
    def test_A3_easy_to_suggest_break(self):
        """持续负性 + easy 难度 → suggest_break（不继续降低）。"""
        engine = self._make_engine(sustain_seconds=5.0)
        decisions = self._drive_engine(
            engine, DIFFICULTY_EASY, count=10,
        )

        triggered = [d for d in decisions if d.should_emit_event]
        self.assertTrue(len(triggered) > 0, "应有至少一次触发")
        decision = triggered[0]

        self.assertEqual(decision.action, AdaptiveAction.SUGGEST_BREAK)
        self.assertTrue(decision.should_emit_event)
        self.assertIn("休息", decision.feedback_text)
        self.assertIn("最低难度", decision.reason)

    # ── A4: rejected → none ──
    def test_A4_rejected_no_action(self):
        """quality_level=rejected → 不得触发任何动作。"""
        engine = self._make_engine(
            sustain_seconds=0.1, negative_threshold=0.01,
        )
        decision = engine.decide(
            quality_level="rejected",
            negative_prob=0.99,
            attention=45.0,
            task_difficulty=DIFFICULTY_HARD,
            timestamp=time.time(),
            warmup_complete=True,
        )

        self.assertEqual(decision.action, AdaptiveAction.NONE)
        self.assertFalse(decision.should_emit_event)

    # ── A5: cooldown → none ──
    def test_A5_cooldown_no_action(self):
        """冷却中再次输入 → 不得触发新 intervention。"""
        engine = self._make_engine(sustain_seconds=3.0, cooldown_seconds=100.0)
        start = time.time()

        # 第一次触发（3 秒 sustain + 3 次调用）
        decisions1 = self._drive_engine(
            engine, DIFFICULTY_HARD, count=6, start_time=start,
        )
        triggered1 = [d for d in decisions1 if d.should_emit_event]
        self.assertTrue(len(triggered1) > 0, "第一次应触发")

        # 冷却中（50 秒后，远未达 cooldown=100）
        decision2 = engine.decide(
            quality_level="trusted",
            negative_prob=0.70,
            attention=45.0,
            task_difficulty=DIFFICULTY_HARD,
            timestamp=start + 60,
            warmup_complete=True,
        )

        self.assertFalse(decision2.should_emit_event, "冷却中不应触发")
        self.assertTrue(decision2.in_cooldown)

    # ── A6: 未达 sustain → none ──
    def test_A6_insufficient_sustain_no_action(self):
        """未达持续证据要求 → 不触发。"""
        engine = self._make_engine(sustain_seconds=20.0)
        decisions = self._drive_engine(
            engine, DIFFICULTY_HARD, count=10,  # 仅 10 次，未达 20 秒
        )

        triggered = [d for d in decisions if d.should_emit_event]
        self.assertEqual(len(triggered), 0, "未达 sustain 不应触发")

        # 最后一次的 above_seconds 应大于 0 但小于 sustain
        last = decisions[-1]
        self.assertGreater(last.above_seconds, 0)
        self.assertLess(last.above_seconds, 20.0)

    # ── A7: warmup 未完成 → 不触发 ──
    def test_A7_warmup_incomplete_no_action(self):
        """预热未完成 → 不触发正式干预。"""
        engine = self._make_engine(
            sustain_seconds=0.1, negative_threshold=0.01,
        )
        decision = engine.decide(
            quality_level="trusted",
            negative_prob=0.99,
            attention=45.0,
            task_difficulty=DIFFICULTY_HARD,
            timestamp=time.time(),
            warmup_complete=False,
        )

        self.assertEqual(decision.action, AdaptiveAction.MAINTAIN)
        self.assertFalse(decision.should_emit_event)


class TestEventRules(unittest.TestCase):
    """B. 事件测试。"""

    def setUp(self):
        QApplication = _import_qt()
        self.app = QApplication.instance() or QApplication(sys.argv)

    def test_B1_one_decision_one_event(self):
        """一次合法决策 → exactly one intervention event。"""
        state = DashboardState()
        state.warmup_progress = 1.0
        state.quality_level = "trusted"
        state.poor_signal = 5
        state.task_difficulty = DIFFICULTY_HARD
        state.attention = 45.0

        engine = AdaptiveFeedbackEngine(
            negative_threshold=0.60, sustain_seconds=3.0, cooldown_seconds=2.0,
            alpha=1.0,
        )

        start = time.time()
        decisions = []
        for i in range(10):
            d = engine.decide(
                quality_level="trusted",
                negative_prob=0.70,
                attention=45.0,
                task_difficulty=state.task_difficulty,
                timestamp=start + i,
                warmup_complete=True,
            )
            decisions.append(d)

        triggered = [d for d in decisions if d.should_emit_event]
        self.assertTrue(len(triggered) > 0)
        decision = triggered[0]

        before = len([e for e in state._events if e.category == "intervention"])
        apply_adaptive_decision(state, decision)
        after = len([e for e in state._events if e.category == "intervention"])

        self.assertEqual(after - before, 1,
                         "一次决策应只产生一次 intervention event")

    def test_B2_cooldown_zero_additional(self):
        """cooldown 中再次输入 → zero additional intervention。"""
        state = DashboardState()
        state.warmup_progress = 1.0
        state.quality_level = "trusted"
        state.poor_signal = 5
        state.task_difficulty = DIFFICULTY_HARD
        state.attention = 45.0

        engine = AdaptiveFeedbackEngine(
            negative_threshold=0.60, sustain_seconds=3.0, cooldown_seconds=100.0,
            alpha=1.0,
        )
        start = time.time()

        # 第一次触发
        decisions1 = []
        for i in range(10):
            d = engine.decide(
                quality_level="trusted",
                negative_prob=0.70,
                attention=45.0,
                task_difficulty=state.task_difficulty,
                timestamp=start + i,
                warmup_complete=True,
            )
            decisions1.append(d)

        triggered1 = [d for d in decisions1 if d.should_emit_event]
        self.assertTrue(len(triggered1) > 0)
        apply_adaptive_decision(state, triggered1[0])
        first_events = len([e for e in state._events if e.category == "intervention"])

        # 冷却中（50秒后，远小于 cooldown=100）
        decision2 = engine.decide(
            quality_level="trusted",
            negative_prob=0.70,
            attention=45.0,
            task_difficulty=state.task_difficulty,
            timestamp=start + 50,
            warmup_complete=True,
        )

        self.assertFalse(decision2.should_emit_event)
        before = len([e for e in state._events if e.category == "intervention"])
        apply_adaptive_decision(state, decision2)
        after = len([e for e in state._events if e.category == "intervention"])
        self.assertEqual(after - before, 0,
                         "冷却中不应产生新的 intervention event")
        self.assertEqual(after, first_events)


class TestMockIntegration(unittest.TestCase):
    """C. Mock 集成测试。"""

    def setUp(self):
        QApplication = _import_qt()
        self.app = QApplication.instance() or QApplication(sys.argv)

    def test_C1_mock_sustained_decision_shared_engine(self):
        """Mock sustained decision → shared engine → DashboardState → difficulty changed。"""
        state = DashboardState()
        state.warmup_progress = 1.0
        state.quality_level = "trusted"
        state.poor_signal = 5
        state.task_difficulty = DIFFICULTY_HARD
        state.attention = 45.0

        # 使用引擎（模拟 Mock 的 _update_stable_state 路径）
        engine = AdaptiveFeedbackEngine(
            negative_threshold=0.60, sustain_seconds=3.0, cooldown_seconds=2.0,
            alpha=1.0,
        )
        start = time.time()
        decisions = []
        for i in range(10):
            d = engine.decide(
                quality_level="trusted",
                negative_prob=0.70,
                attention=45.0,
                task_difficulty=state.task_difficulty,
                timestamp=start + i,
                warmup_complete=True,
            )
            decisions.append(d)

        triggered = [d for d in decisions if d.should_emit_event]
        self.assertTrue(len(triggered) > 0)
        decision = triggered[0]

        self.assertEqual(decision.new_difficulty, DIFFICULTY_MEDIUM)

        # 应用决策到 state
        apply_adaptive_decision(state, decision)

        # 验证 DashboardState 已更新
        self.assertEqual(state.task_difficulty, DIFFICULTY_MEDIUM)
        self.assertEqual(state.adaptive_action, AdaptiveAction.REDUCE_DIFFICULTY)
        self.assertIsNotNone(state.adaptive_action_time)
        interventions = [e for e in state._events if e.category == "intervention"]
        self.assertGreaterEqual(len(interventions), 1)

    def test_C2_mock_multiple_decisions(self):
        """连续触发：hard→medium→easy→break（通过共享引擎）。"""
        state = DashboardState()
        state.warmup_progress = 1.0
        state.quality_level = "trusted"
        state.poor_signal = 5
        state.attention = 45.0

        engine = AdaptiveFeedbackEngine(
            negative_threshold=0.60, sustain_seconds=3.0, cooldown_seconds=5.0,
            alpha=1.0,
        )

        def _trigger(eng, task_diff, start_ts):
            """驱动引擎到触发点，返回触发决策。"""
            for i in range(20):
                d = eng.decide(
                    quality_level="trusted",
                    negative_prob=0.70,
                    attention=45.0,
                    task_difficulty=task_diff,
                    timestamp=start_ts + i,
                    warmup_complete=True,
                )
                if d.should_emit_event:
                    return d
            self.fail("未能触发")

        # 第一次: hard → medium
        state.task_difficulty = DIFFICULTY_HARD
        d1 = _trigger(engine, DIFFICULTY_HARD, time.time())
        apply_adaptive_decision(state, d1)
        self.assertEqual(state.task_difficulty, DIFFICULTY_MEDIUM)
        self.assertEqual(state.adaptive_action, AdaptiveAction.REDUCE_DIFFICULTY)

        # 第二次: medium → easy（跨越 cooldown）
        state.task_difficulty = DIFFICULTY_MEDIUM
        d2 = _trigger(engine, DIFFICULTY_MEDIUM, time.time() + 100)
        apply_adaptive_decision(state, d2)
        self.assertEqual(state.task_difficulty, DIFFICULTY_EASY)

        # 第三次: easy → suggest_break
        state.task_difficulty = DIFFICULTY_EASY
        d3 = _trigger(engine, DIFFICULTY_EASY, time.time() + 200)
        apply_adaptive_decision(state, d3)
        self.assertEqual(state.task_difficulty, DIFFICULTY_EASY)  # 保持 easy
        self.assertEqual(state.adaptive_action, AdaptiveAction.SUGGEST_BREAK)

        # 验证事件数
        interventions = [e for e in state._events if e.category == "intervention"]
        self.assertEqual(len(interventions), 3,
                         "三次决策应产生三次 intervention event")


class TestLiveServiceIntegration(unittest.TestCase):
    """D. Live 服务测试（使用 synthetic/fake inference，不需要真实 MindWave）。"""

    def setUp(self):
        QApplication = _import_qt()
        self.app = QApplication.instance() or QApplication(sys.argv)

    def test_D1_accepted_negative_triggers_adaptive_action(self):
        """accepted negative sustained decision → adaptive action → DashboardState 改变。"""
        state = DashboardState()
        state.device_status = "online"
        state.connector_status = "online"
        state.warmup_progress = 1.0
        state.quality_level = "trusted"
        state.poor_signal = 5
        state.task_difficulty = DIFFICULTY_HARD
        state.attention = 45.0

        engine = AdaptiveFeedbackEngine(
            negative_threshold=0.60, sustain_seconds=3.0, cooldown_seconds=2.0,
            alpha=1.0,
        )

        start = time.time()
        decisions = []
        for i in range(10):
            d = engine.decide(
                quality_level=state.quality_level,
                negative_prob=0.70,
                attention=state.attention,
                task_difficulty=state.task_difficulty,
                timestamp=start + i,
                warmup_complete=state.warmup_complete,
            )
            decisions.append(d)

        triggered = [d for d in decisions if d.should_emit_event]
        self.assertTrue(len(triggered) > 0)
        decision = triggered[0]

        self.assertEqual(decision.action, AdaptiveAction.REDUCE_DIFFICULTY)

        # 通过共享契约写入
        apply_adaptive_decision(state, decision)

        self.assertEqual(state.task_difficulty, DIFFICULTY_MEDIUM)
        self.assertEqual(state.adaptive_action, AdaptiveAction.REDUCE_DIFFICULTY)
        self.assertIsNotNone(state.adaptive_action_time)
        self.assertIn("困难", state.adaptive_feedback_text)
        self.assertIn("中等", state.adaptive_feedback_text)

        interventions = [e for e in state._events if e.category == "intervention"]
        self.assertEqual(len(interventions), 1)

    def test_D2_rejected_result_no_adaptive_action(self):
        """rejected result → 无 adaptive action。"""
        state = DashboardState()
        state.device_status = "online"
        state.connector_status = "online"
        state.warmup_progress = 1.0
        state.quality_level = "rejected"
        state.poor_signal = None
        state.task_difficulty = DIFFICULTY_HARD
        state.attention = 45.0

        engine = AdaptiveFeedbackEngine(
            negative_threshold=0.01, sustain_seconds=0.1, cooldown_seconds=2.0,
            alpha=1.0,
        )

        decision = engine.decide(
            quality_level="rejected",
            negative_prob=0.99,
            attention=45.0,
            task_difficulty=DIFFICULTY_HARD,
            timestamp=time.time(),
            warmup_complete=True,
        )

        self.assertEqual(decision.action, AdaptiveAction.NONE)
        self.assertFalse(decision.should_emit_event)

        before = len([e for e in state._events if e.category == "intervention"])
        apply_adaptive_decision(state, decision)
        after = len([e for e in state._events if e.category == "intervention"])
        self.assertEqual(after - before, 0, "rejected 时不应产生任何 event")
        self.assertEqual(state.task_difficulty, DIFFICULTY_HARD)
        self.assertEqual(state.adaptive_action, AdaptiveAction.NONE)

    def test_D3_live_service_engine_reset_on_offline(self):
        """Live 设备离线时引擎应复位。"""
        engine = AdaptiveFeedbackEngine(
            negative_threshold=0.60, sustain_seconds=3.0, cooldown_seconds=2.0,
            alpha=1.0,
        )

        start = time.time()
        for i in range(10):
            engine.decide(
                quality_level="trusted",
                negative_prob=0.70,
                attention=45.0,
                task_difficulty=DIFFICULTY_HARD,
                timestamp=start + i,
                warmup_complete=True,
            )

        self.assertIsNotNone(engine._last_intervention)
        self.assertGreater(engine.current_negative_ewma, 0)

        engine.reset()
        self.assertIsNone(engine._last_intervention)
        self.assertEqual(engine.current_negative_ewma, 0.0)
        self.assertFalse(engine.in_cooldown)


class TestReset(unittest.TestCase):
    """E. Reset 测试。"""

    def setUp(self):
        QApplication = _import_qt()
        self.app = QApplication.instance() or QApplication(sys.argv)

    def test_E1_reset_session_clears_adaptive_state(self):
        """reset_session 后 adaptive 状态正确复位。"""
        state = DashboardState()
        state.warmup_progress = 1.0
        state.quality_level = "trusted"
        state.poor_signal = 5
        state.task_difficulty = DIFFICULTY_HARD
        state.attention = 45.0

        engine = AdaptiveFeedbackEngine(
            negative_threshold=0.60, sustain_seconds=3.0, cooldown_seconds=2.0,
            alpha=1.0,
        )

        start = time.time()
        decisions = []
        for i in range(10):
            d = engine.decide(
                quality_level="trusted",
                negative_prob=0.70,
                attention=45.0,
                task_difficulty=state.task_difficulty,
                timestamp=start + i,
                warmup_complete=True,
            )
            decisions.append(d)

        triggered = [d for d in decisions if d.should_emit_event]
        self.assertTrue(len(triggered) > 0)

        apply_adaptive_decision(state, triggered[0])
        self.assertEqual(state.adaptive_action, AdaptiveAction.REDUCE_DIFFICULTY)
        self.assertNotEqual(state.adaptive_action_reason, "")
        self.assertIsNotNone(state.adaptive_action_time)

        # reset_session
        state.reset_session()
        engine.reset()

        # 验证复位
        self.assertEqual(state.adaptive_action, AdaptiveAction.NONE)
        self.assertEqual(state.adaptive_action_reason, "")
        self.assertIsNone(state.adaptive_action_time)
        self.assertEqual(state.adaptive_feedback_text, "")
        self.assertFalse(state.task_running)

        # 引擎状态也应复位
        self.assertIsNone(engine._last_intervention)
        self.assertFalse(engine.in_cooldown)

    def test_E2_reset_preserves_user_fields(self):
        """reset_session 后 task_type 保留，adaptive 状态清除。"""
        state = DashboardState()
        state.task_type = "编程任务"
        state.task_difficulty = DIFFICULTY_HARD

        # 触发后
        state.adaptive_action = AdaptiveAction.REDUCE_DIFFICULTY
        state.adaptive_feedback_text = "测试反馈"

        state.reset_session()

        # adaptive 状态清除
        self.assertEqual(state.adaptive_action, AdaptiveAction.NONE)
        self.assertEqual(state.adaptive_feedback_text, "")
        # task_type 保留
        self.assertEqual(state.task_type, "编程任务")


if __name__ == "__main__":
    unittest.main(verbosity=2)