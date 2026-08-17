import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "realtime_inference"))

from src.decision import EWMASustainedNegativeDecision


class EWMADecisionTest(unittest.TestCase):
    def test_requires_sustained_eligible_evidence(self):
        policy = EWMASustainedNegativeDecision(2, alpha=1.0, sustain_seconds=20)
        for timestamp in range(0, 20, 2):
            self.assertFalse(policy.update(np.array([0.1, 0.1, 0.8]), timestamp, True).intervention_triggered)
        self.assertTrue(
            policy.update(np.array([0.1, 0.1, 0.8]), 20, True).intervention_triggered
        )

    def test_rejection_breaks_continuity_and_does_not_update(self):
        policy = EWMASustainedNegativeDecision(2, alpha=0.5, sustain_seconds=4)
        policy.update(np.array([0.1, 0.1, 0.8]), 0, True)
        before = policy.ewma.copy()
        state = policy.update(np.array([0.8, 0.1, 0.1]), 2, False)
        np.testing.assert_allclose(policy.ewma, before)
        self.assertFalse(state.updated)
        self.assertFalse(
            policy.update(np.array([0.1, 0.1, 0.8]), 4, True).intervention_triggered
        )


if __name__ == "__main__":
    unittest.main()
