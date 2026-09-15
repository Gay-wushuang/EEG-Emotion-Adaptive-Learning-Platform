"""Focused regressions for Round 1B baseline and stale-analysis bugs."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui_prototype"))
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication

from services.dashboard_state import (
    AdaptiveAction,
    BASELINE_COLLECTING,
    BASELINE_FAILED,
    DashboardState,
)
from pages.baseline_page import BaselinePage
from smart_learning_app.live_service import (
    RAW_DATA_TIMEOUT_SECONDS,
    WINDOW_SAMPLES,
    LiveDataService,
    ThinkGearLiveWorker,
)


class Round1BRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _result(label="positive"):
        probabilities = {
            "positive": (0.8, 0.1, 0.1),
            "neutral": (0.1, 0.8, 0.1),
        }[label]
        return SimpleNamespace(
            probabilities=probabilities,
            display_class=label,
            confidence=0.8,
            accepted=True,
        )

    def test_baseline_normal_collection_and_confirmed_disconnect(self):
        state = DashboardState(seed_demo_history=False)
        state.connector_status = "online"
        state.device_status = "online"
        page = BaselinePage(state, SimpleNamespace())

        page._start_baseline()
        page.update_state(state)
        self.assertTrue(page._baseline_active)
        self.assertEqual(state.baseline_status, BASELINE_COLLECTING)

        state.device_status = "waiting_raw"
        page.update_state(state)
        self.assertFalse(page._baseline_active)
        self.assertFalse(page._baseline_done)
        self.assertEqual(state.baseline_status, BASELINE_FAILED)
        self.assertFalse(page._btn_next.isEnabled())
        self.assertIn("设备连接已中断", page._label_status.text())

        state.device_status = "online"
        page.update_state(state)
        self.assertTrue(page._btn_start.isEnabled())
        page.deleteLater()

    def test_raw_timeout_uses_existing_receive_clock_and_recovers(self):
        worker = ThinkGearLiveWorker(port=1)
        statuses = []
        worker.status_changed.connect(statuses.append)
        worker._consume_packet({"rawEeg": 1})
        last_raw = worker._last_raw_monotonic

        worker._check_raw_timeout(last_raw + RAW_DATA_TIMEOUT_SECONDS - 0.01)
        self.assertEqual(statuses[-1]["device_status"], "online")
        worker._check_raw_timeout(last_raw + RAW_DATA_TIMEOUT_SECONDS)
        self.assertEqual(statuses[-1]["device_status"], "waiting_raw")

        worker._consume_packet({"rawEeg": 2})
        self.assertEqual(statuses[-1]["device_status"], "online")

    def test_rejected_signal_clears_old_analysis_and_new_result_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            state = DashboardState(seed_demo_history=False)
            service = LiveDataService(state, ROOT / "production_baseline_v1")
            service.sessions_dir = Path(directory)
            state.connector_status = "online"
            state.device_status = "online"
            state.poor_signal = 0
            state.warmup_progress = 1.0
            state.quality_level = "trusted"

            service._on_result(self._result("positive"))
            state.adaptive_feedback_text = "降低任务难度"
            state.adaptive_action = AdaptiveAction.REDUCE_DIFFICULTY
            self.assertEqual(state.predicted_state, "positive")
            self.assertIsNotNone(state.prob_positive)

            service._on_batch({
                "raw": [], "attention": 50, "meditation": 50,
                "poor_signal": 200, "raw_count": WINDOW_SAMPLES,
                "buffer_samples": WINDOW_SAMPLES, "sample_rate_hz": 512,
            })
            self.assertIsNone(state.predicted_state)
            self.assertIsNone(state.prob_positive)
            self.assertIsNone(state.stable_state)
            self.assertEqual(state.adaptive_feedback_text, "")
            self.assertEqual(state.adaptive_action, AdaptiveAction.NONE)
            self.assertIn("当前信号不可解释", state.feedback_text)

            # A delayed result from before rejection must not restore old UI.
            service._on_result(self._result("positive"))
            self.assertIsNone(state.predicted_state)

            state.poor_signal = 0
            service._on_batch({
                "raw": [], "attention": 55, "meditation": 48,
                "poor_signal": 0, "raw_count": WINDOW_SAMPLES,
                "buffer_samples": WINDOW_SAMPLES, "sample_rate_hz": 512,
            })
            service._on_result(self._result("neutral"))
            self.assertEqual(state.predicted_state, "neutral")
            self.assertAlmostEqual(state.prob_neutral, 0.8)
            self.assertEqual(state.stable_state, "neutral")


if __name__ == "__main__":
    unittest.main()
