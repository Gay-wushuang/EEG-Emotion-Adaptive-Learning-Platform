import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui_prototype"))
sys.path.insert(0, str(ROOT))

from smart_learning_app.live_service import (
    INFERENCE_STEP_SAMPLES,
    LiveDataService,
    WINDOW_SAMPLES,
    ProductionInferenceWorker,
    ThinkGearLiveWorker,
)
from services.dashboard_state import DashboardState


class LiveServiceTest(unittest.TestCase):
    def test_first_raw_and_window_are_sample_index_driven(self):
        worker = ThinkGearLiveWorker(port=1)
        windows = []
        worker.window_ready.connect(windows.append)
        worker._consume_packet({
            "poorSignalLevel": 0,
            "eSense": {"attention": 55, "meditation": 48},
        })
        for index in range(WINDOW_SAMPLES + INFERENCE_STEP_SAMPLES):
            worker._consume_packet({"rawEeg": index % 100})
        # One result as soon as 30 s is ready, then one more after the 2 s step.
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[-1]["raw"].shape, (WINDOW_SAMPLES,))
        self.assertTrue(np.isfinite(windows[-1]["attention"]).all())
        self.assertEqual(windows[-1]["poor_signal"], 0)

    def test_auxiliary_fill_rejects_fully_missing_values(self):
        with self.assertRaisesRegex(ValueError, "Attention/Meditation"):
            ProductionInferenceWorker._fill(np.full(10, np.nan))

    def test_reconnect_clears_stale_window(self):
        worker = ThinkGearLiveWorker(port=1)
        worker._consume_packet({"rawEeg": 1})
        self.assertEqual(len(worker._raw), 1)
        worker._reset_stream()
        self.assertEqual(len(worker._raw), 0)
        self.assertIsNone(worker._first_raw_monotonic)

    def test_session_csv_contains_capture_quality_and_prediction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = DashboardState()
            state.device_status = "online"
            state.poor_signal = 0
            state.warmup_progress = 1.0
            state.prob_positive = 0.7
            state.prob_neutral = 0.2
            state.prob_negative = 0.1
            state.predicted_state = "positive"
            state.confidence = 0.7
            service = LiveDataService(state, ROOT / "production_baseline_v1")
            service.sessions_dir = Path(temp_dir)
            service.start_session()
            service._on_batch({
                "raw": [10, 11], "attention": 60, "meditation": 50,
                "poor_signal": 0, "raw_count": 2, "buffer_samples": 2,
                "sample_rate_hz": 512,
            })
            saved = service.end_session()
            self.assertIsNotNone(saved)
            with Path(saved).open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["raw"], "10")
            self.assertEqual(rows[0]["predicted_class"], "positive")
            self.assertIn("quality_level", rows[0])

    def test_dominant_state_is_mode_of_accepted_predictions(self):
        state = DashboardState()
        service = LiveDataService(state, ROOT / "production_baseline_v1")
        def result(label, accepted=True):
            probs = {
                "positive": [0.7, 0.2, 0.1],
                "neutral": [0.2, 0.7, 0.1],
                "negative": [0.1, 0.2, 0.7],
            }[label]
            return SimpleNamespace(
                probabilities=probs, display_class=label,
                confidence=max(probs), accepted=accepted,
            )
        for label in ("positive", "neutral", "positive"):
            service._on_result(result(label))
        self.assertEqual(state.stable_state, "positive")
        service._on_result(result("negative", accepted=False))
        self.assertEqual(state.stable_state, "positive")


if __name__ == "__main__":
    unittest.main()
