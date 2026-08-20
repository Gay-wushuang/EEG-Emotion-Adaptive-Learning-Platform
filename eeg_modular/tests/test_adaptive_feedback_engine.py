"""AdaptiveFeedbackEngine 集成测试（Phase 2.1 修订版）。

覆盖（AGENTS.md §IX 验收要求）：
A. Engine 单元测试（使用新 API: probabilities + accepted）
B. 事件测试
C. Mock 集成测试（含通过 _update_stable_state 的正式路径）
D. Live 服务测试（真正实例化 LiveDataService + fake InferenceResult）
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
import numpy as np

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
from realtime_inference.src.decision import (
    EWMASustainedNegativeDecision,
    DecisionState,
)


# ═══════════════════════════════════════════════════════════════
#  A. Engine 单元测试
# ═══════════════════════════════════════════════════════════════

class TestAdaptiveFeedbackEngine(unittest.TestCase):
    """A. Engine 单元测试（使用新 API: probabilities + accepted）。"""

    def _make_engine(self, sustain_seconds=5.0, cooldown_seconds=10.0,
                     negative_threshold=0.60, alpha=1.0):
        """创建一个测试用引擎（默认 alpha=1.0 让 EWMA 立即生效）。"""
        return AdaptiveFeedbackEngine(
            negative_threshold=negative_threshold,
            sustain_seconds=sustain_seconds,
            cooldown_seconds=cooldown_seconds,
            alpha=alpha,
        )

    def _prob_array(self, neg=0.70):
        """构造三分类概率数组 [positive, neutral, negative]。"""
        return np.array([0.1, 1.0 - 0.1 - neg, neg], dtype=np.float64)

    def _drive_engine(self, engine, task_difficulty, neg_prob=0.70,
                      attention=45.0, quality_level="trusted",
                      warmup_complete=True, accepted=True,
                      count=10, start_time=None):
        """驱动引擎 N 次，返回所有决策列表。"""
        if start_time is None:
            start_time = time.time()
        decisions = []
        probs = self._prob_array(neg=neg_prob)
        for i in range(count):
            decision = engine.decide(
                probabilities=probs.copy(),
                accepted=accepted,
                quality_level=quality_level,
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
        decisions = self._drive_engine(engine, DIFFICULTY_HARD, count=10)

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
        decisions = self._drive_engine(engine, DIFFICULTY_MEDIUM, count=10)

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
        decisions = self._drive_engine(engine, DIFFICULTY_EASY, count=10)

        triggered = [d for d in decisions if d.should_emit_event]
        self.assertTrue(len(triggered) > 0, "应有至少一次触发")
        decision = triggered[0]

        self.assertEqual(decision.action, AdaptiveAction.SUGGEST_BREAK)
        self.assertTrue(decision.should_emit_event)
        self.assertIn("休息", decision.feedback_text)

    # ── A4: rejected → none ──
    def test_A4_rejected_no_action(self):
        """quality_level=rejected → 不得触发任何动作。"""
        engine = self._make_engine(
            sustain_seconds=0.1, negative_threshold=0.01,
        )
        decision = engine.decide(
            probabilities=self._prob_array(neg=0.99),
            accepted=True,
            quality_level="rejected",
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

        # 第一次触发
        decisions1 = self._drive_engine(
            engine, DIFFICULTY_HARD, count=6, start_time=start,
        )
        triggered1 = [d for d in decisions1 if d.should_emit_event]
        self.assertTrue(len(triggered1) > 0, "第一次应触发")

        # 冷却中（50 秒后，远未达 cooldown=100）
        decision2 = engine.decide(
            probabilities=self._prob_array(neg=0.70),
            accepted=True,
            quality_level="trusted",
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
            probabilities=self._prob_array(neg=0.99),
            accepted=True,
            quality_level="trusted",
            attention=45.0,
            task_difficulty=DIFFICULTY_HARD,
            timestamp=time.time(),
            warmup_complete=False,
        )

        self.assertEqual(decision.action, AdaptiveAction.MAINTAIN)
        self.assertFalse(decision.should_emit_event)

    # ── A8: accepted=False 时不贡献证据（安全护栏核心）──
    def test_A8_rejected_accepted_no_evidence(self):
        """accepted=False → 不计入情绪证据、不更新 EWMA、不累计 sustain。

        这是 Phase 2.1 修复的核心：Production Baseline 拒识的窗口
        不得累计负性证据，即使 negative_prob=0.99。
        """
        engine = self._make_engine(
            sustain_seconds=0.1, negative_threshold=0.01, alpha=1.0,
        )
        start = time.time()

        # 连续 20 次 accepted=False，negative 极高
        for i in range(20):
            decision = engine.decide(
                probabilities=self._prob_array(neg=0.99),
                accepted=False,  # Production 拒识
                quality_level="trusted",
                attention=45.0,
                task_difficulty=DIFFICULTY_HARD,
                timestamp=start + i,
                warmup_complete=True,
            )
            self.assertFalse(
                decision.should_emit_event,
                f"accepted=False 不应在第 {i} 次触发"
            )

        # 验证 EWMA 已被清空（accepted=False 必须清空旧证据）
        self.assertEqual(engine.current_negative_ewma, 0.0,
                         "accepted=False 时 EWMA 必须被清空")
        # 验证 above_since 未被设置
        self.assertIsNone(engine._policy.above_since,
                          "accepted=False 时 above_since 应为 None")
        # 验证 EWMA 已被清空（不是保留旧值）
        self.assertIsNone(engine._policy.ewma,
                          "accepted=False 后 EWMA 必须重置为 None")

    # ── A9: accepted=True 之后 accepted=False 重置持续计时 ──
    def test_A9_accepted_false_resets_sustain(self):
        """先 accepted=True 建立 sustain，然后 accepted=False 应重置。"""
        engine = self._make_engine(
            sustain_seconds=5.0, negative_threshold=0.60, alpha=1.0,
        )
        start = time.time()

        # 5 次 accepted=True → 应累计到 sustain_seconds
        for i in range(5):
            engine.decide(
                probabilities=self._prob_array(neg=0.70),
                accepted=True,
                quality_level="trusted",
                attention=45.0,
                task_difficulty=DIFFICULTY_HARD,
                timestamp=start + i,
                warmup_complete=True,
            )

        # 现在 above_since 应已设置
        self.assertIsNotNone(engine._policy.above_since)

        # 一次 accepted=False 应重置 above_since
        engine.decide(
            probabilities=self._prob_array(neg=0.70),
            accepted=False,
            quality_level="trusted",
            attention=45.0,
            task_difficulty=DIFFICULTY_HARD,
            timestamp=start + 6,
            warmup_complete=True,
        )
        self.assertIsNone(engine._policy.above_since,
                          "accepted=False 应重置 above_since")


# ═══════════════════════════════════════════════════════════════
#  B. 事件测试
# ═══════════════════════════════════════════════════════════════

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
                probabilities=np.array([0.1, 0.2, 0.7]),
                accepted=True,
                quality_level="trusted",
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
                probabilities=np.array([0.1, 0.2, 0.7]),
                accepted=True,
                quality_level="trusted",
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
            probabilities=np.array([0.1, 0.2, 0.7]),
            accepted=True,
            quality_level="trusted",
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


# ═══════════════════════════════════════════════════════════════
#  C. Mock 集成测试
# ═══════════════════════════════════════════════════════════════

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

        engine = AdaptiveFeedbackEngine(
            negative_threshold=0.60, sustain_seconds=3.0, cooldown_seconds=2.0,
            alpha=1.0,
        )
        start = time.time()
        decisions = []
        for i in range(10):
            d = engine.decide(
                probabilities=np.array([0.1, 0.2, 0.7]),
                accepted=True,
                quality_level="trusted",
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
                    probabilities=np.array([0.1, 0.2, 0.7]),
                    accepted=True,
                    quality_level="trusted",
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

    def test_C3_mock_via_update_stable_state(self):
        """Mock 正式路径：_update_stable_state → shared engine → DashboardState。

        不是直接调用引擎，而是通过 MockDataService 的内部方法，
        验证完整的 Mock 决策链路。
        """
        from services.mock_data_service import MockDataService

        state = DashboardState()
        state.warmup_progress = 1.0
        state.quality_level = "trusted"
        state.poor_signal = 5
        state.task_difficulty = DIFFICULTY_HARD
        state.attention = 45.0
        state.connector_status = "online"
        state.device_status = "online"

        service = MockDataService(state)

        # 调整引擎时间参数使测试在合理时间内完成
        service.engine._policy.sustain_seconds = 0.1
        service.engine._policy.cooldown_seconds = 0.5

        # 预填充 Temporal Policy 状态：模拟持续负性已建立 > sustain_seconds
        # 这样下一次调用就会触发 intervention
        service.engine._policy.ewma = np.array([0.10, 0.20, 0.70])
        service.engine._policy.above_since = time.time() - 1.0  # 1 秒前

        # 单次调用 _update_stable_state 就应触发
        raw_probs = np.array([0.10, 0.20, 0.70])
        state.prob_positive = 0.10
        state.prob_neutral = 0.20
        state.prob_negative = 0.70

        service._update_stable_state(raw_probs.copy())

        self.assertTrue(state._intervention_triggered,
                        "Mock 路径应通过 _update_stable_state 触发干预")
        self.assertEqual(state.task_difficulty, DIFFICULTY_MEDIUM,
                         "困难应降低为中等")
        self.assertEqual(state.adaptive_action, AdaptiveAction.REDUCE_DIFFICULTY)
        interventions = [e for e in state._events if e.category == "intervention"]
        self.assertGreaterEqual(len(interventions), 1,
                                 "应有至少一次 intervention event")

    def test_C4_mock_rejected_quality_blocks(self):
        """Mock: quality=rejected 时不应触发自适应动作。"""
        from services.mock_data_service import MockDataService

        state = DashboardState()
        state.warmup_progress = 1.0
        state.quality_level = "rejected"  # 关键：rejected
        state.poor_signal = 200
        state.task_difficulty = DIFFICULTY_HARD
        state.attention = None
        state.connector_status = "offline"
        state.device_status = "offline"

        service = MockDataService(state)

        # 调整引擎时间参数（即使时间短也不应触发，因为 quality=rejected）
        service.engine._policy.sustain_seconds = 0.1
        service.engine._policy.cooldown_seconds = 0.5

        # 尝试驱动
        raw_probs = np.array([0.10, 0.01, 0.89])
        for i in range(10):
            service._update_stable_state(raw_probs.copy())

        self.assertFalse(state._intervention_triggered,
                         "rejected 时不应触发干预")
        self.assertEqual(state.task_difficulty, DIFFICULTY_HARD,
                         "难度应保持不变")


# ═══════════════════════════════════════════════════════════════
#  D. Live 服务测试（真正实例化 LiveDataService）
# ═══════════════════════════════════════════════════════════════

class TestLiveServiceIntegration(unittest.TestCase):
    """D. Live 服务测试（使用 synthetic/fake inference，不需要真实 MindWave）。"""

    def setUp(self):
        QApplication = _import_qt()
        self.app = QApplication.instance() or QApplication(sys.argv)

        # 创建一个假的 package_dir（不需要真实模型，只测试 _on_result 逻辑）
        self._fake_pkg_dir = Path(__file__).parent / "_fake_production_pkg"

    def _make_fake_result(self, accepted=True, negative=0.70):
        """构造一个假的 InferenceResult。"""
        from smart_learning_app.inference_engine import InferenceResult

        pos = 0.10
        neg = negative
        neu = 1.0 - pos - neg
        return InferenceResult(
            probabilities=(pos, neu, neg),
            internal_class="sad",
            display_class="negative",
            confidence=neg,
            accepted=accepted,
            latency_ms=5.0,
        )

    def _setup_live_state(self, state):
        """为 Live 测试初始化 DashboardState。"""
        state.device_status = "online"
        state.connector_status = "online"
        state.warmup_progress = 1.0
        state.quality_level = "trusted"
        state.poor_signal = 5
        state.task_difficulty = DIFFICULTY_HARD
        state.attention = 45.0
        state.meditation = 50.0

    # ── D1: accepted=True, negative high → 真正经过 LiveDataService._on_result ──
    def test_D1_accepted_negative_triggers_via_live_service(self):
        """accepted negative sustained decision → LiveDataService._on_result() →
        adaptive action → DashboardState 改变 → exactly one intervention。"""
        from smart_learning_app.live_service import LiveDataService

        state = DashboardState()
        self._setup_live_state(state)

        # 创建 LiveDataService（不需要启动 worker，直接调 _on_result）
        service = LiveDataService(state, self._fake_pkg_dir)

        # 调整引擎时间参数使测试在合理时间内完成
        service.engine._policy.sustain_seconds = 0.1
        service.engine._policy.cooldown_seconds = 0.5

        # 预填充 Temporal Policy 状态：模拟持续负性已建立 > sustain_seconds
        service.engine._policy.ewma = np.array([0.10, 0.20, 0.70])
        service.engine._policy.above_since = time.time() - 1.0  # 1 秒前

        # 单次调用 _on_result 就应触发
        fake_result = self._make_fake_result(accepted=True, negative=0.70)
        service._on_result(fake_result)

        self.assertTrue(state._intervention_triggered,
                        "accepted=True + negative high 应触发干预")
        self.assertEqual(state.task_difficulty, DIFFICULTY_MEDIUM,
                         "困难应降低为中等")
        self.assertEqual(state.adaptive_action, AdaptiveAction.REDUCE_DIFFICULTY)
        self.assertIsNotNone(state.adaptive_action_time)
        self.assertIn("困难", state.adaptive_feedback_text)
        self.assertIn("中等", state.adaptive_feedback_text)

        interventions = [e for e in state._events if e.category == "intervention"]
        self.assertEqual(len(interventions), 1,
                         "应恰好产生 1 次 intervention event")

    # ── D2: accepted=False, negative=0.99 → 不触发（安全护栏核心）──
    def test_D2_rejected_accepted_no_trigger_via_live_service(self):
        """Fake InferenceResult(accepted=False, negative=0.99) 连续输入 →
        不累计 sustain → 不改变 task_difficulty → adaptive_action 不触发 →
        0 intervention events。"""
        from smart_learning_app.live_service import LiveDataService

        state = DashboardState()
        self._setup_live_state(state)

        service = LiveDataService(state, self._fake_pkg_dir)

        # 调整引擎时间参数
        service.engine._policy.sustain_seconds = 0.1
        service.engine._policy.cooldown_seconds = 0.5

        before_interventions = len(
            [e for e in state._events if e.category == "intervention"]
        )

        # 连续 20 次 accepted=False，negative 极高
        for i in range(20):
            fake_result = self._make_fake_result(
                accepted=False, negative=0.99
            )
            service._on_result(fake_result)

            # 每次都断言：不应触发干预
            self.assertFalse(
                state._intervention_triggered,
                f"accepted=False 不应在第 {i} 次触发"
            )

        # 最终验证
        self.assertEqual(state.task_difficulty, DIFFICULTY_HARD,
                         "accepted=False 时难度应保持不变")
        self.assertEqual(state.adaptive_action, AdaptiveAction.NONE,
                         "accepted=False 时 adaptive_action 应为 NONE")
        self.assertFalse(state._intervention_triggered)

        after_interventions = len(
            [e for e in state._events if e.category == "intervention"]
        )
        self.assertEqual(
            after_interventions - before_interventions, 0,
            "accepted=False 时不应产生任何 intervention event"
        )

    # ── D3: accepted=True 触发后，accepted=False 不应再触发 ──
    def test_D3_accepted_then_rejected_no_double_trigger(self):
        """先 accepted=True 触发，然后 accepted=False 连续输入 → 不产生新干预。"""
        from smart_learning_app.live_service import LiveDataService

        state = DashboardState()
        self._setup_live_state(state)

        service = LiveDataService(state, self._fake_pkg_dir)

        # 调整引擎时间参数
        service.engine._policy.sustain_seconds = 0.1
        service.engine._policy.cooldown_seconds = 0.5

        # 预填充 Temporal Policy 状态
        service.engine._policy.ewma = np.array([0.10, 0.20, 0.70])
        service.engine._policy.above_since = time.time() - 1.0

        # 第一阶段：accepted=True → 触发（单次调用）
        fake_result = self._make_fake_result(accepted=True, negative=0.70)
        service._on_result(fake_result)

        self.assertTrue(state._intervention_triggered,
                        "第一阶段应触发")
        first_interventions = len(
            [e for e in state._events if e.category == "intervention"]
        )

        # 第二阶段：accepted=False 连续输入（accepted=False 永远不会触发）
        for i in range(50):
            fake_result = self._make_fake_result(accepted=False, negative=0.99)
            service._on_result(fake_result)

        second_interventions = len(
            [e for e in state._events if e.category == "intervention"]
        )
        self.assertEqual(
            second_interventions, first_interventions,
            "accepted=False 不应产生新的 intervention"
        )

    # ── D4: 验证 accepted=False 时 EWMA 不被更新 ──
    def test_D4_rejected_ewma_not_updated(self):
        """accepted=False 时 Live 引擎的 EWMA 不应被更新。"""
        from smart_learning_app.live_service import LiveDataService

        state = DashboardState()
        self._setup_live_state(state)

        service = LiveDataService(state, self._fake_pkg_dir)

        # 调整引擎时间参数
        service.engine._policy.sustain_seconds = 0.1
        service.engine._policy.cooldown_seconds = 0.5

        # 确认初始 EWMA 为 0
        self.assertEqual(service.engine.current_negative_ewma, 0.0)

        # 连续 accepted=False
        for i in range(10):
            fake_result = self._make_fake_result(accepted=False, negative=0.99)
            service._on_result(fake_result)

        # EWMA 应仍为 0（因为 rejected windows 不作为情绪证据）
        self.assertEqual(service.engine.current_negative_ewma, 0.0,
                         "accepted=False 时 EWMA 不应被更新")
        self.assertIsNone(service.engine._policy.above_since,
                          "accepted=False 时 above_since 应为 None")


# ═══════════════════════════════════════════════════════════════
#  E. Reset 测试
# ═══════════════════════════════════════════════════════════════

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
                probabilities=np.array([0.1, 0.2, 0.7]),
                accepted=True,
                quality_level="trusted",
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

        # 引擎状态也应复位（通过 policy 访问）
        self.assertIsNone(engine._policy.last_intervention)
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

    def test_E3_engine_reset_clears_all_temporal_state(self):
        """引擎 reset 应清除所有时间相关状态。"""
        engine = AdaptiveFeedbackEngine(
            negative_threshold=0.60, sustain_seconds=3.0, cooldown_seconds=2.0,
            alpha=1.0,
        )

        # 先建立一些状态
        start = time.time()
        for i in range(10):
            engine.decide(
                probabilities=np.array([0.1, 0.2, 0.7]),
                accepted=True,
                quality_level="trusted",
                attention=45.0,
                task_difficulty=DIFFICULTY_HARD,
                timestamp=start + i,
                warmup_complete=True,
            )

        # 确认状态存在
        self.assertIsNotNone(engine._policy.last_intervention)
        self.assertGreater(engine.current_negative_ewma, 0.0)

        # reset
        engine.reset()

        # 验证所有状态清除
        self.assertIsNone(engine._policy.last_intervention)
        self.assertEqual(engine.current_negative_ewma, 0.0)
        self.assertFalse(engine.in_cooldown)
        self.assertIsNone(engine._policy.above_since)


# ═══════════════════════════════════════════════════════════════
#  F. Temporal Decision Policy 单元测试（EWMASustainedNegativeDecision）
# ═══════════════════════════════════════════════════════════════

class TestEWMASustainedNegativeDecision(unittest.TestCase):
    """F. 直接测试 Temporal Policy 的安全语义。"""

    def test_F1_eligible_false_no_ewma_update(self):
        """eligible=False → 不更新 EWMA、不累计 sustain。"""
        policy = EWMASustainedNegativeDecision(
            negative_index=2, alpha=1.0,
            negative_threshold=0.60, sustain_seconds=5.0, cooldown_seconds=10.0,
        )

        probs = np.array([0.1, 0.1, 0.8])

        # eligible=False，连续 10 次
        for i in range(10):
            state = policy.update(probs, time.time() + i, eligible=False)
            self.assertFalse(state.intervention_triggered)
            self.assertEqual(state.above_seconds, 0.0)

        # EWMA 应为 None（从未初始化）
        self.assertIsNone(policy.ewma)
        self.assertIsNone(policy.above_since)

    def test_F2_eligible_true_updates_ewma(self):
        """eligible=True → 正常更新 EWMA。"""
        policy = EWMASustainedNegativeDecision(
            negative_index=2, alpha=1.0,
            negative_threshold=0.60, sustain_seconds=5.0, cooldown_seconds=10.0,
        )

        probs = np.array([0.1, 0.1, 0.8])
        state = policy.update(probs, time.time(), eligible=True)

        self.assertIsNotNone(policy.ewma)
        self.assertAlmostEqual(policy.ewma[2], 0.8, places=5)

    def test_F3_eligible_false_resets_above_since(self):
        """先 eligible=True 建立 above_since，然后 eligible=False 应重置。"""
        policy = EWMASustainedNegativeDecision(
            negative_index=2, alpha=1.0,
            negative_threshold=0.60, sustain_seconds=5.0, cooldown_seconds=10.0,
        )

        probs = np.array([0.1, 0.1, 0.8])
        t0 = time.time()

        # 2 次 eligible=True → above_since 应已设置
        policy.update(probs, t0, eligible=True)
        policy.update(probs, t0 + 1, eligible=True)
        self.assertIsNotNone(policy.above_since)

        # 1 次 eligible=False → above_since 应重置
        state = policy.update(probs, t0 + 2, eligible=False)
        self.assertIsNone(policy.above_since)
        self.assertEqual(state.above_seconds, 0.0)


# ═══════════════════════════════════════════════════════════════
#  G. Mock / Live 共享 Temporal Decision 语义一致性测试
# ═══════════════════════════════════════════════════════════════

class TestMockLiveTemporalConsistency(unittest.TestCase):
    """G. 验证 Mock 和 Live 通过相同的 AdaptiveFeedbackEngine →
    EWMASustainedNegativeDecision 处理链路，
    从而保证两边 Temporal Decision 语义（EWMA、sustain、cooldown）完全一致。
    """

    def setUp(self):
        QApplication = _import_qt()
        self.app = QApplication.instance() or QApplication(sys.argv)

    def _make_engine(self):
        return AdaptiveFeedbackEngine(
            negative_threshold=0.60,
            sustain_seconds=3.0,
            cooldown_seconds=10.0,
            alpha=1.0,
        )

    def test_G1_mock_and_live_share_same_policy_class(self):
        """Mock 和 Live 使用的 Temporal Policy 必须是同一类。"""
        engine = self._make_engine()
        from services.mock_data_service import MockDataService
        from smart_learning_app.live_service import LiveDataService

        s_mock = DashboardState()
        s_live = DashboardState()
        s_live.connector_status = "online"
        s_live.device_status = "online"

        mock = MockDataService(s_mock)
        live = LiveDataService(s_live, Path(__file__).parent / "_fake_production_pkg")

        self.assertIsInstance(mock.engine._policy, EWMASustainedNegativeDecision)
        self.assertIsInstance(live.engine._policy, EWMASustainedNegativeDecision)
        self.assertEqual(type(mock.engine._policy), type(live.engine._policy))

    def test_G2_mock_and_live_identical_inputs_produce_identical_state(self):
        """相同原始概率序列 + 相同 accepted 序列 → Mock 和 Live
        产生相同的 EWMA、above_since、last_intervention。"""
        from services.mock_data_service import MockDataService
        from smart_learning_app.live_service import LiveDataService
        from smart_learning_app.inference_engine import InferenceResult

        s_mock = DashboardState()
        s_mock.warmup_progress = 1.0
        s_mock.quality_level = "trusted"
        s_mock.poor_signal = 5
        s_mock.task_difficulty = DIFFICULTY_HARD
        s_mock.attention = 45.0
        s_mock.connector_status = "online"
        s_mock.device_status = "online"

        s_live = DashboardState()
        s_live.warmup_progress = 1.0
        s_live.quality_level = "trusted"
        s_live.poor_signal = 5
        s_live.task_difficulty = DIFFICULTY_HARD
        s_live.attention = 45.0
        s_live.connector_status = "online"
        s_live.device_status = "online"

        mock = MockDataService(s_mock)
        live = LiveDataService(s_live, Path(__file__).parent / "_fake_production_pkg")

        # 统一时间参数
        for svc in (mock, live):
            svc.engine._policy.sustain_seconds = 0.5
            svc.engine._policy.cooldown_seconds = 50.0

        probs = np.array([0.10, 0.20, 0.70])
        start = time.time()

        for i in range(10):
            t = start + i
            # Mock 路径: _update_stable_state(raw_probs)
            s_mock.prob_positive = probs[0]
            s_mock.prob_neutral = probs[1]
            s_mock.prob_negative = probs[2]
            mock._update_stable_state(probs.copy())

            # Live 路径: _on_result(InferenceResult(accepted=True))
            live._on_result(InferenceResult(
                probabilities=tuple(probs.tolist()),
                internal_class="sad",
                display_class="negative",
                confidence=probs[2],
                accepted=True,
                latency_ms=1.0,
            ))

        # 验证两边 Temporal Policy 内部状态完全一致
        np.testing.assert_array_almost_equal(
            mock.engine._policy.ewma,
            live.engine._policy.ewma,
            decimal=6,
            err_msg="Mock 和 Live 的 EWMA 必须一致",
        )
        self.assertAlmostEqual(
            mock.engine._policy.above_since,
            live.engine._policy.above_since,
            delta=0.1,
            msg="Mock 和 Live 的 above_since 必须近似一致（允许 0.1s 时钟偏差）",
        )
        self.assertEqual(
            mock.engine._policy.last_intervention,
            live.engine._policy.last_intervention,
            "Mock 和 Live 的 last_intervention 必须一致",
        )

        # 验证两边 DashboardState 最终状态一致
        self.assertEqual(s_mock.task_difficulty, s_live.task_difficulty,
                         "Mock 和 Live 最终 task_difficulty 应一致")
        self.assertEqual(s_mock.adaptive_action, s_live.adaptive_action,
                         "Mock 和 Live 最终 adaptive_action 应一致")

    def test_G3_accepted_false_resets_policy_for_both_paths(self):
        """accepted=False 必须清空 Temporal Policy 的 EWMA 和 above_since。

        Mock 始终 accepted=True（无 Production Baseline 拒识机制），
        所以通过 engine 直接调用验证；Live 通过 _on_result 验证。
        这样保证两条路径在遇到 accepted=False 时语义一致。
        """
        from smart_learning_app.live_service import LiveDataService
        from smart_learning_app.inference_engine import InferenceResult

        # ── 先直接用 engine 建立证据 ──
        engine_mock = AdaptiveFeedbackEngine(
            negative_threshold=0.60, sustain_seconds=3.0,
            cooldown_seconds=10.0, alpha=1.0,
        )
        engine_live = AdaptiveFeedbackEngine(
            negative_threshold=0.60, sustain_seconds=3.0,
            cooldown_seconds=10.0, alpha=1.0,
        )

        probs_high = np.array([0.05, 0.05, 0.90])
        t = time.time()
        for i in range(5):
            engine_mock.decide(
                probabilities=probs_high.copy(),
                accepted=True, quality_level="trusted",
                attention=45.0, task_difficulty=DIFFICULTY_HARD,
                timestamp=t + i, warmup_complete=True,
            )
            engine_live.decide(
                probabilities=probs_high.copy(),
                accepted=True, quality_level="trusted",
                attention=45.0, task_difficulty=DIFFICULTY_HARD,
                timestamp=t + i, warmup_complete=True,
            )

        self.assertIsNotNone(engine_mock._policy.ewma, "Mock engine 已建立 EWMA")
        self.assertIsNotNone(engine_live._policy.ewma, "Live engine 已建立 EWMA")

        # ── accepted=False → 两边都必须清空 ──
        for i in range(5):
            engine_mock.decide(
                probabilities=probs_high.copy(),
                accepted=False,  # 关键：Production 拒识
                quality_level="trusted",
                attention=45.0, task_difficulty=DIFFICULTY_HARD,
                timestamp=t + 5 + i, warmup_complete=True,
            )
            engine_live.decide(
                probabilities=probs_high.copy(),
                accepted=False,
                quality_level="trusted",
                attention=45.0, task_difficulty=DIFFICULTY_HARD,
                timestamp=t + 5 + i, warmup_complete=True,
            )

        self.assertIsNone(engine_mock._policy.ewma,
                          "accepted=False 后 Mock engine EWMA 应为 None")
        self.assertIsNone(engine_live._policy.ewma,
                          "accepted=False 后 Live engine EWMA 应为 None")
        self.assertIsNone(engine_mock._policy.above_since)
        self.assertIsNone(engine_live._policy.above_since)

        # ── 最后验证 LiveDataService 真实路径也清空 ──
        s_live = DashboardState()
        s_live.warmup_progress = 1.0
        s_live.quality_level = "trusted"
        s_live.poor_signal = 5
        s_live.task_difficulty = DIFFICULTY_HARD
        s_live.attention = 45.0
        s_live.connector_status = "online"
        s_live.device_status = "online"

        live = LiveDataService(
            s_live, Path(__file__).parent / "_fake_production_pkg"
        )
        # 建立证据
        for i in range(5):
            live._on_result(InferenceResult(
                probabilities=tuple(probs_high.tolist()),
                internal_class="sad", display_class="negative",
                confidence=0.90, accepted=True, latency_ms=1.0,
            ))
        self.assertIsNotNone(live.engine._policy.ewma)
        # accepted=False → Live 路径清空
        for i in range(3):
            live._on_result(InferenceResult(
                probabilities=tuple(probs_high.tolist()),
                internal_class="sad", display_class="negative",
                confidence=0.05, accepted=False, latency_ms=1.0,
            ))
        self.assertIsNone(live.engine._policy.ewma,
                          "accepted=False 后 Live engine EWMA 应为 None")
        self.assertIsNone(live.engine._policy.above_since)


if __name__ == "__main__":
    unittest.main(verbosity=2)
