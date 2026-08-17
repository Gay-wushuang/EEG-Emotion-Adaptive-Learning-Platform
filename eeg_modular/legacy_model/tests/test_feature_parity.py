"""Golden parity test for the recoverable training feature contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "time_data_preprocess"))
sys.path.insert(0, str(ROOT / "eeg_modular"))
sys.path.insert(0, str(ROOT / "eeg_modular" / "realtime_inference"))

from preprocess.feature_extraction import extract_time_features
from preprocess.filters import bandpass_filter
from src.model import EmotionInferenceModel, InferenceConfig


class FeatureParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture = Path(__file__).parent / "fixtures" / "training_golden_sample.npz"
        cls.data = np.load(fixture)

    def test_all_modalities_match_training_assets(self) -> None:
        for modality in ("filtered", "powerspec", "att", "med"):
            with self.subTest(modality=modality):
                aligned = self.data[f"aligned_{modality}"]
                processed = (
                    bandpass_filter(aligned).astype(np.float32)
                    if modality == "filtered"
                    else aligned
                )
                unscaled = extract_time_features(pd.DataFrame(processed), time_steps=10)
                scaled = (
                    unscaled - self.data[f"scaler_mean_{modality}"]
                ) / self.data[f"scaler_scale_{modality}"]
                np.testing.assert_allclose(
                    unscaled,
                    self.data[f"expected_unscaled_{modality}"],
                    rtol=1e-5,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    scaled,
                    self.data[f"expected_scaled_{modality}"],
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_model_probability_matches_golden_fixture(self) -> None:
        modalities = ("filtered", "powerspec", "att", "med")
        model = EmotionInferenceModel(
            InferenceConfig(
                model_path=ROOT / "eeg_modular" / "outputs" / "CNN" / "models" / "best_fold4.pt",
                modalities=modalities,
                scalers_dir=ROOT / "eeg_modular" / "features",
                skip_scaling=True,
            )
        )
        predicted, probabilities = model.predict(
            {
                modality: self.data[f"expected_scaled_{modality}"]
                for modality in modalities
            }
        )
        expected_names = [str(name) for name in self.data["class_names"]]
        expected_class = expected_names[int(self.data["expected_class_index"][0])]
        self.assertEqual(predicted, expected_class)
        np.testing.assert_allclose(
            probabilities,
            self.data["expected_probabilities"],
            rtol=1e-4,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
