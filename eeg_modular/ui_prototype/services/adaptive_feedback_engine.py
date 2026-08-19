"""AdaptiveFeedbackEngine - 最小公共自适应反馈决策引擎。

职责：
    1. 质量门控（rejected → 无动作）
    2. 预热检查（warmup 未完成 → 无正式干预）
    3. 持续负性判定（EWMA + sustain_seconds + cooldown_seconds）
    4. AdaptiveAction 选择（hard→medium, medium→easy, easy→break）
    5. 生成原因和反馈文本

不得依赖：
    QWidget, TaskPage, QComboBox, ThinkGear socket, 模型权重。
    只接受已得到的状态信息。

安全护栏（AGENTS.md §IV）：
    - quality_level == rejected → 不得自适应改变任务、不得生成 intervention
    - warmup 未完成 → 不触发正式学习状态干预
    - 没达到持续证据要求 → 不触发 reduce_difficulty / suggest_break
    - cooldown 未结束 → 不得再次触发 intervention
    - hard → medium, medium → easy, easy → suggest_break
    - suggest_break：不关闭 EEG session、不结束学习 session、不伪造已休息
    - ATT/MED 仅作辅助依据；不加入 Production Baseline v1 情绪分类
    - negative 仅表述为"负性状态/趋势"
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

# 复用 DashboardState 中的正式枚举（不建立第二套枚举）
from services.dashboard_state import (
    DIFFICULTY_EASY,
    DIFFICULTY_MEDIUM,
    DIFFICULTY_HARD,
    DIFFICULTY_DISPLAY,
    AdaptiveAction,
)


@dataclass(frozen=True)
class AdaptiveDecision:
    """自适应决策结果（不可变，便于测试和审查）。

    Attributes:
        action: AdaptiveAction 枚举值
        reason: 触发原因（中文展示用）
        feedback_text: AI 反馈文本（中文）
        should_emit_event: 是否应写入 intervention 事件（一次决策只触发一次）
        new_difficulty: 若为 REDUCE_DIFFICULTY，这里给出新的难度等级
        negative_ewma: 当前负性 EWMA 值（供 UI 展示）
        above_seconds: 负性持续秒数（供 UI 展示）
        in_cooldown: 是否处于冷却期
    """
    action: str
    reason: str
    feedback_text: str
    should_emit_event: bool
    new_difficulty: Optional[str] = None
    negative_ewma: float = 0.0
    above_seconds: float = 0.0
    in_cooldown: bool = False

    # ── 工厂方法 ──

    @staticmethod
    def none() -> "AdaptiveDecision":
        return AdaptiveDecision(
            action=AdaptiveAction.NONE,
            reason="",
            feedback_text="",
            should_emit_event=False,
        )

    @staticmethod
    def maintain(feedback_text: str = "") -> "AdaptiveDecision":
        """维持当前策略，只更新反馈文本，不产生 intervention 事件。"""
        return AdaptiveDecision(
            action=AdaptiveAction.MAINTAIN,
            reason="",
            feedback_text=feedback_text,
            should_emit_event=False,
        )

    @staticmethod
    def reduce_difficulty(
        old_difficulty: str,
        new_difficulty: str,
        reason: str,
    ) -> "AdaptiveDecision":
        prev_disp = DIFFICULTY_DISPLAY.get(old_difficulty, old_difficulty)
        new_disp = DIFFICULTY_DISPLAY.get(new_difficulty, new_difficulty)
        return AdaptiveDecision(
            action=AdaptiveAction.REDUCE_DIFFICULTY,
            reason=reason,
            feedback_text=f"当前任务已从\"{prev_disp}\"调整为\"{new_disp}\"",
            should_emit_event=True,
            new_difficulty=new_difficulty,
        )

    @staticmethod
    def suggest_break(
        reason: str = "学习状态持续不佳，已为最低难度",
        feedback_text: str = "当前任务已为最低难度，建议短暂休息后继续",
    ) -> "AdaptiveDecision":
        return AdaptiveDecision(
            action=AdaptiveAction.SUGGEST_BREAK,
            reason=reason,
            feedback_text=feedback_text,
            should_emit_event=True,
        )


class AdaptiveFeedbackEngine:
    """最小公共自适应反馈决策引擎。

    输入：已得到的状态信息（quality_level, negative_prob, attention,
          task_difficulty, timestamp, warmup_complete, prev_ewma）
    输出：AdaptiveDecision（不可变决策结果）

    使用方法：
        engine = AdaptiveFeedbackEngine()
        decision = engine.decide(
            quality_level="trusted",
            negative_prob=0.65,
            attention=45.0,
            task_difficulty="hard",
            timestamp=time.time(),
            warmup_complete=True,
        )
        if decision.should_emit_event:
            # 写入 DashboardState
    """

    def __init__(
        self,
        negative_threshold: float = 0.60,
        sustain_seconds: float = 20.0,
        cooldown_seconds: float = 90.0,
        alpha: float = 0.2,
    ):
        self.negative_threshold = negative_threshold
        self.sustain_seconds = sustain_seconds
        self.cooldown_seconds = cooldown_seconds
        self.alpha = alpha

        # ── 内部状态（不可从外部直接操作）──
        self._above_since: Optional[float] = None
        self._last_intervention: Optional[float] = None
        self._negative_ewma: float = 0.0
        self._initialized: bool = False

    # ── 核心决策方法 ──

    def decide(
        self,
        quality_level: str,
        negative_prob: float,
        attention: Optional[float],
        task_difficulty: str,
        timestamp: float,
        warmup_complete: bool,
        prev_ewma: Optional[float] = None,
    ) -> AdaptiveDecision:
        """核心决策：输入状态，输出 AdaptiveDecision。

        Args:
            quality_level: "trusted" | "warning" | "rejected"
            negative_prob: 当前窗口的负性概率（原始，未经 EWMA）
            attention: 当前 Attention 值（None 表示设备离线）
            task_difficulty: "easy" | "medium" | "hard"
            timestamp: 当前时间戳（time.time()）
            warmup_complete: 预热是否完成
            prev_ewma: 上一次负性 EWMA（None 则用内部维护值）

        Returns:
            AdaptiveDecision: 决策结果（不可变）
        """
        # ── 安全护栏 1：质量 rejected → 无动作 ──
        if quality_level == "rejected":
            self._above_since = None
            return AdaptiveDecision.none()

        # ── 安全护栏 2：预热未完成 → 不触发正式干预 ──
        if not warmup_complete:
            self._above_since = None
            return AdaptiveDecision.maintain("预热进行中，暂不生成学习状态干预。")

        # ── EWMA 更新 ──
        if prev_ewma is not None:
            self._negative_ewma = (
                self.alpha * negative_prob
                + (1 - self.alpha) * prev_ewma
            )
        elif not self._initialized:
            self._negative_ewma = negative_prob
            self._initialized = True
        else:
            self._negative_ewma = (
                self.alpha * negative_prob
                + (1 - self.alpha) * self._negative_ewma
            )

        neg = self._negative_ewma

        # ── 安全护栏 3：未达负性阈值 → 重置持续计时 ──
        if neg < self.negative_threshold:
            self._above_since = None
            return AdaptiveDecision(
                action=AdaptiveAction.MAINTAIN,
                reason="",
                feedback_text="",
                should_emit_event=False,
                negative_ewma=neg,
                above_seconds=0.0,
            )

        # ── 开始 / 持续计时 ──
        if self._above_since is None:
            self._above_since = timestamp

        above_seconds = timestamp - self._above_since

        # ── 安全护栏 4：冷却期检查 ──
        cooled = (
            self._last_intervention is None
            or timestamp - self._last_intervention >= self.cooldown_seconds
        )

        in_cooldown = not cooled

        # ── 安全护栏 5：持续判定 + 冷却通过 → 触发 intervention ──
        if above_seconds >= self.sustain_seconds and cooled:
            self._last_intervention = timestamp
            self._above_since = None
            return self._select_action(
                task_difficulty, attention, neg, above_seconds
            )

        # ── 持续中但未达 sustain，或冷却中 ──
        if in_cooldown:
            return AdaptiveDecision(
                action=AdaptiveAction.MAINTAIN,
                reason="",
                feedback_text="",
                should_emit_event=False,
                negative_ewma=neg,
                above_seconds=above_seconds,
                in_cooldown=True,
            )

        # 持续中，尚未达 sustain_seconds
        return AdaptiveDecision(
            action=AdaptiveAction.MAINTAIN,
            reason="",
            feedback_text="",
            should_emit_event=False,
            negative_ewma=neg,
            above_seconds=above_seconds,
        )

    # ── 动作选择（公开方法，供直接调用使用）──

    def select_action(
        self,
        task_difficulty: str,
        attention: Optional[float],
    ) -> AdaptiveDecision:
        """根据当前任务难度选择 AdaptiveAction（无 sustain/cooldown 检查）。

        用于：
        1. 测试直接调用 _apply_adaptive_action 时的向后兼容
        2. 已知"允许触发"后的纯动作选择

        规则（AGENTS.md §IV）：
            hard → medium（reduce_difficulty）
            medium → easy（reduce_difficulty）
            easy → suggest_break（不继续降低）
        """
        att_low = attention is not None and attention < 50.0
        reason_neg = "持续负性状态趋势"
        reason_att = "且 Attention 偏低" if att_low else ""
        reason_full = f"{reason_neg}{reason_att}"

        if task_difficulty == DIFFICULTY_HARD:
            return AdaptiveDecision.reduce_difficulty(
                DIFFICULTY_HARD, DIFFICULTY_MEDIUM, reason_full
            )
        elif task_difficulty == DIFFICULTY_MEDIUM:
            return AdaptiveDecision.reduce_difficulty(
                DIFFICULTY_MEDIUM, DIFFICULTY_EASY, reason_full
            )
        else:  # easy 或其他
            return AdaptiveDecision.suggest_break()

    def _select_action(
        self,
        task_difficulty: str,
        attention: Optional[float],
        negative_ewma: float,
        above_seconds: float,
    ) -> AdaptiveDecision:
        """内部包装，调用 select_action 并附加状态信息。"""
        decision = self.select_action(task_difficulty, attention)
        # 附加当前 EWMA 和持续秒数（不可变对象需要重建）
        return AdaptiveDecision(
            action=decision.action,
            reason=decision.reason,
            feedback_text=decision.feedback_text,
            should_emit_event=decision.should_emit_event,
            new_difficulty=decision.new_difficulty,
            negative_ewma=negative_ewma,
            above_seconds=above_seconds,
        )

    # ── 状态管理 ──

    def reset(self) -> None:
        """复位所有内部状态（reset_session 时调用）。"""
        self._above_since = None
        self._last_intervention = None
        self._negative_ewma = 0.0
        self._initialized = False

    @property
    def in_cooldown(self) -> bool:
        """当前是否处于冷却期（供外部查询）。"""
        if self._last_intervention is None:
            return False
        return (time.time() - self._last_intervention) < self.cooldown_seconds

    @property
    def current_negative_ewma(self) -> float:
        """当前负性 EWMA 值。"""
        return self._negative_ewma

    @property
    def above_seconds(self) -> float:
        """当前负性持续秒数（未达阈值时为 0）。"""
        if self._above_since is None:
            return 0.0
        return time.time() - self._above_since


# ── 共享 DashboardState 写入契约 ──

def apply_adaptive_decision(state, decision: AdaptiveDecision) -> None:
    """将 AdaptiveDecision 写入 DashboardState 并生成 intervention 事件。

    这是 MockDataService 和 LiveDataService 共用的正式写入契约，
    确保两者在相同决策下产生完全一致的 DashboardState 变化和事件。

    安全护栏：
    - quality_level == rejected → 不得触发任何自适应动作
    - NONE / MAINTAIN → 不写 intervention event
    - 一次决策只产生一次 intervention event

    Args:
        state: DashboardState 实例
        decision: AdaptiveDecision 实例
    """
    # 安全护栏：信号 rejected 不得触发任何自适应动作
    if state.quality_level == "rejected":
        return

    if not decision.should_emit_event:
        return

    now = time.time()

    # ── 写入 DashboardState ──
    if decision.new_difficulty:
        state.task_difficulty = decision.new_difficulty

    state.adaptive_action = decision.action
    state.adaptive_action_reason = decision.reason
    state.adaptive_feedback_text = decision.feedback_text
    state.adaptive_action_time = now

    # ── 构造事件（一次决策只产生一次 intervention event）──
    if decision.action == AdaptiveAction.REDUCE_DIFFICULTY:
        if decision.new_difficulty:
            prev_diff = (
                DIFFICULTY_HARD
                if decision.new_difficulty == DIFFICULTY_MEDIUM
                else DIFFICULTY_MEDIUM
            )
            prev_disp = DIFFICULTY_DISPLAY.get(prev_diff, prev_diff)
            new_disp = DIFFICULTY_DISPLAY.get(
                decision.new_difficulty, decision.new_difficulty
            )
            label = "自适应降低任务难度"
            note = f"{prev_disp} -> {new_disp}；原因：{decision.reason}"
        else:
            label = "自适应降低任务难度"
            note = f"原因：{decision.reason}"
    elif decision.action == AdaptiveAction.SUGGEST_BREAK:
        label = "建议短暂休息"
        note = "当前已为最低任务难度，进入休息建议"
    else:
        return  # NONE / MAINTAIN 不写 intervention event

    state.add_event(label, "intervention", note)
