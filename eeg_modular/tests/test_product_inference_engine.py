import unittest
from pathlib import Path

import numpy as np

from smart_learning_app.inference_engine import ProductionInferenceEngine


ROOT = Path(__file__).resolve().parents[1]


class ProductInferenceEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ProductionInferenceEngine(ROOT / "production_baseline_v1")

    def test_package_contract_and_feature_inference(self):
        fixture = np.load(
            ROOT / "production_baseline_v1" / "golden_inference_fixture.npz"
        )
        result = self.engine.infer_features(
            {"filtered": fixture["filtered"][0], "bandpower": fixture["bandpower"][0]}
        )
        np.testing.assert_allclose(
            result.probabilities,
            fixture["expected_probabilities"][0],
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertIn(result.internal_class, ("happy", "normal", "sad"))
        self.assertIn(result.display_class, ("positive", "neutral", "negative"))

    def test_auxiliary_modalities_cannot_enter_classifier(self):
        zeros = np.zeros((10, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "ATT/MED"):
            self.engine.infer_features(
                {"filtered": zeros, "bandpower": zeros, "att": zeros}
            )

    def test_raw_window_uses_canonical_contract(self):
        count = self.engine.required_samples
        time = np.arange(count, dtype=np.float64) / 512.0
        raw = 20.0 * np.sin(2.0 * np.pi * 10.0 * time)
        auxiliary = np.full(count, 50.0)
        result = self.engine.infer_window(raw, auxiliary, auxiliary)
        self.assertAlmostEqual(sum(result.probabilities), 1.0, places=6)
        self.assertGreater(result.latency_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
