"""AdaptiveFeedbackEngine - 最小公共自适应反馈决策引擎。

职责：
    1. 质量门控（rejected → 无动作）
    2. 预热检查（warmup 未完成 → 无正式干预）
    3. 生产模型接受门控（accepted=False → 不计入情绪证据）
    4. 持续负性判定（复用 EWMASustainedNegativeDecision：EWMA + sustain + cooldown）
    5. AdaptiveAction 选择（hard→medium, medium→easy, easy→break）
    6. 生成原因和反馈文本

不得依赖：
    QWidget, TaskPage, QComboBox, ThinkGear socket, 模型权重。
    只接受已得到的状态信息。

时间策略（单一来源）：
    原始 probability → 一次 EWMA → 持续 20s → 冷却 90s → 动作选择。
    不得 Mock 和 Live 使用不同的隐式平滑层数。

安全护栏（AGENTS.md §IV）：
    - quality_level == rejected → 不得自适应改变任务、不得生成 intervention
    - warmup 未完成 → 不触发正式学习状态干预
    - accepted == False（Production Baseline 拒识）→ 不计入情绪证据、不更新 EWMA、不累计 sustain
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

import numpy as np

# 复用 DashboardState 中的正式枚举（不建立第二套枚举）
from services.dashboard_state import (
    DIFFICULTY_EASY,
    DIFFICULTY_MEDIUM,
    DIFFICULTY_HARD,
    DIFFICULTY_DISPLAY,
    AdaptiveAction,
)

# 复用仓库已有的 Temporal Decision Policy
# realtime_inference/src/decision.py 中的 EWMASustainedNegativeDecision
# 已正确实现：eligible=False 时不更新 EWMA、重置持续计时
from realtime_inference.src.decision import (
    EWMASustainedNegativeDecision,
    DecisionState,
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
        return AdaptiveDecision(
            action=AdaptiveAction.REDUCE_DIFFICULTY,
            reason=reason,
            feedback_text=(
                "检测到持续低专注/负性状态趋势，建议适当降低后续学习难度"
                "或调整学习节奏。当前任务难度保持不变。"
            ),
            should_emit_event=True,
            # Advisory hint only; applying the decision never changes the task.
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

    Temporal Decision（EWMA + sustain + cooldown）委托给
    仓库已有的 EWMASustainedNegativeDecision，确保 Mock 和 Live
    使用完全相同的时间平滑策略。

    输入：原始概率数组 + 接受标志 + 质量/预热/注意力等
    输出：AdaptiveDecision（不可变决策结果）

    使用方法：
        engine = AdaptiveFeedbackEngine()
        decision = engine.decide(
            probabilities=np.array([0.1, 0.2, 0.7]),
            accepted=True,
            quality_level="trusted",
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
        # ── Temporal Decision Policy（单一 EWMA 来源）──
        self._policy = EWMASustainedNegativeDecision(
            negative_index=2,  # [positive, neutral, negative] → negative 是 index 2
            alpha=alpha,
            negative_threshold=negative_threshold,
            sustain_seconds=sustain_seconds,
            cooldown_seconds=cooldown_seconds,
        )

        # ── 参数镜像（供外部查询）──
        self.negative_threshold = negative_threshold
        self.sustain_seconds = sustain_seconds
        self.cooldown_seconds = cooldown_seconds
        self.alpha = alpha

    # ── 核心决策方法 ──

    def decide(
        self,
        probabilities: np.ndarray,
        accepted: bool,
        quality_level: str,
        attention: Optional[float],
        task_difficulty: str,
        timestamp: float,
        warmup_complete: bool,
    ) -> AdaptiveDecision:
        """核心决策：输入原始概率 + 状态信息，输出 AdaptiveDecision。

        Args:
            probabilities: 三分类概率数组 [P(positive), P(neutral), P(negative)]
                          原始值，未经二次 EWMA
            accepted: Production Baseline 置信度是否通过拒识阈值
            quality_level: "trusted" | "warning" | "rejected"
            attention: 当前 Attention 值（None 表示设备离线）
            task_difficulty: "easy" | "medium" | "hard"
            timestamp: 当前时间戳（time.time()）
            warmup_complete: 预热是否完成

        Returns:
            AdaptiveDecision: 决策结果（不可变）
        """
        # ── 安全护栏 1：质量 rejected → 无动作，重置计时 ──
        if quality_level == "rejected":
            self._policy.above_since = None
            return AdaptiveDecision.none()

        # ── 安全护栏 2：预热未完成 → 不触发正式干预 ──
        if not warmup_complete:
            self._policy.above_since = None
            return AdaptiveDecision.maintain("预热进行中，暂不生成学习状态干预。")

        # ── 安全护栏 3：accepted=False（Production 拒识）→ 不计入情绪证据 ──
        # 当 accepted=False 时：
        #   - 不更新 Temporal Policy 的 EWMA（不保留被拒识窗口的证据）
        #   - 重置 above_since（不累计 sustain）
        #   - 后续 accepted=True 时必须重新累计证据，不可复用旧 EWMA
        if not accepted:
            self._policy.ewma = None
            self._policy.above_since = None

        eligible = bool(accepted and quality_level != "rejected" and warmup_complete)

        # ── 委托 Temporal Decision Policy ──
        state: DecisionState = self._policy.update(
            np.asarray(probabilities, dtype=np.float64),
            timestamp,
            eligible,
        )

        # ── Temporal Policy 触发 → 选择 AdaptiveAction ──
        if state.intervention_triggered:
            return self._select_action(
                task_difficulty,
                attention,
                state.negative_ewma if state.negative_ewma is not None else 0.0,
                state.above_seconds,
            )

        # ── 未触发：返回 MAINTAIN（带状态信息，供 UI 展示）──
        in_cooldown = self._compute_cooldown_status(timestamp)

        return AdaptiveDecision(
            action=AdaptiveAction.MAINTAIN,
            reason="",
            feedback_text="",
            should_emit_event=False,
            negative_ewma=state.negative_ewma if state.negative_ewma is not None else 0.0,
            above_seconds=state.above_seconds,
            in_cooldown=in_cooldown,
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
        """内部包装，调用 select_action 并附加 Temporal 状态信息。"""
        decision = self.select_action(task_difficulty, attention)
        return AdaptiveDecision(
            action=decision.action,
            reason=decision.reason,
            feedback_text=decision.feedback_text,
            should_emit_event=decision.should_emit_event,
            new_difficulty=decision.new_difficulty,
            negative_ewma=negative_ewma,
            above_seconds=above_seconds,
        )

    def _compute_cooldown_status(self, timestamp: float) -> bool:
        """计算当前是否处于冷却期。"""
        if self._policy.last_intervention is None:
            return False
        return (timestamp - self._policy.last_intervention) < self._policy.cooldown_seconds

    # ── 状态管理 ──

    def reset(self) -> None:
        """复位所有内部状态（reset_session 或设备离线时调用）。"""
        self._policy.ewma = None
        self._policy.above_since = None
        self._policy.last_intervention = None

    @property
    def in_cooldown(self) -> bool:
        """当前是否处于冷却期（供外部查询）。"""
        if self._policy.last_intervention is None:
            return False
        return (time.time() - self._policy.last_intervention) < self._policy.cooldown_seconds

    @property
    def current_negative_ewma(self) -> float:
        """当前负性 EWMA 值。"""
        if self._policy.ewma is None:
            return 0.0
        return float(self._policy.ewma[self._policy.negative_index])

    @property
    def above_seconds(self) -> float:
        """当前负性持续秒数（未达阈值时为 0）。"""
        if self._policy.above_since is None:
            return 0.0
        return time.time() - self._policy.above_since


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

    # Decision support only: Assignment, TaskRecord and runtime difficulty
    # remain teacher-controlled. new_difficulty is advisory metadata.
    state.adaptive_action = decision.action
    state.adaptive_action_reason = decision.reason
    state.adaptive_feedback_text = decision.feedback_text
    state.adaptive_action_time = now

    # ── 构造事件（一次决策只产生一次 intervention event）──
    if decision.action == AdaptiveAction.REDUCE_DIFFICULTY:
        label = "AI建议：降低后续学习负荷"
        note = f"建议调整后续难度或学习节奏（未自动修改当前任务）；原因：{decision.reason}"
    elif decision.action == AdaptiveAction.SUGGEST_BREAK:
        label = "建议短暂休息"
        note = "当前已为最低任务难度，进入休息建议"
    else:
        return  # NONE / MAINTAIN 不写 intervention event

    state.add_event(label, "intervention", note)
