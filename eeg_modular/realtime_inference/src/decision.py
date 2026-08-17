"""Safety-oriented temporal decision policy for overlapping EEG windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DecisionState:
    updated: bool
    negative_ewma: float | None
    above_seconds: float
    intervention_triggered: bool


class EWMASustainedNegativeDecision:
    """EWMA probabilities; rejected windows never count as emotional evidence."""

    def __init__(
        self,
        negative_index: int,
        alpha: float = 0.2,
        negative_threshold: float = 0.60,
        sustain_seconds: float = 20.0,
        cooldown_seconds: float = 90.0,
    ):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.negative_index = negative_index
        self.alpha = alpha
        self.negative_threshold = negative_threshold
        self.sustain_seconds = sustain_seconds
        self.cooldown_seconds = cooldown_seconds
        self.ewma: np.ndarray | None = None
        self.above_since: float | None = None
        self.last_intervention: float | None = None

    def update(
        self, probabilities: np.ndarray, timestamp: float, eligible: bool
    ) -> DecisionState:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim != 1 or self.negative_index >= len(probabilities):
            raise ValueError("Invalid probability vector")
        if not eligible:
            self.above_since = None
            return DecisionState(
                updated=False,
                negative_ewma=(
                    None if self.ewma is None else float(self.ewma[self.negative_index])
                ),
                above_seconds=0.0,
                intervention_triggered=False,
            )
        if self.ewma is None:
            self.ewma = probabilities.copy()
        else:
            self.ewma = self.alpha * probabilities + (1 - self.alpha) * self.ewma
        negative = float(self.ewma[self.negative_index])
        if negative < self.negative_threshold:
            self.above_since = None
            return DecisionState(True, negative, 0.0, False)
        if self.above_since is None:
            self.above_since = timestamp
        above_seconds = timestamp - self.above_since
        cooled_down = (
            self.last_intervention is None
            or timestamp - self.last_intervention >= self.cooldown_seconds
        )
        triggered = above_seconds >= self.sustain_seconds and cooled_down
        if triggered:
            self.last_intervention = timestamp
            self.above_since = None
        return DecisionState(True, negative, above_seconds, triggered)

