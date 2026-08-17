"""Training/deployment parity test for the v2 canonical feature extractor."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eeg_modular"))

from eeg_emotion.features.canonical import CanonicalFeatureConfig, extract_canonical_features


class CanonicalFeatureParityTest(unittest.TestCase):
    def test_anonymous_golden_sample(self) -> None:
        fixture = np.load(
            Path(__file__).parent / "fixtures" / "canonical_golden_sample.npz"
        )
        actual = extract_canonical_features(
            fixture["raw"], fixture["att"], fixture["med"], CanonicalFeatureConfig()
        )
        for modality in ("filtered", "bandpower", "att", "med"):
            with self.subTest(modality=modality):
                self.assertEqual(actual[modality].shape, (10, 4))
                np.testing.assert_allclose(
                    actual[modality],
                    fixture[f"expected_{modality}"],
                    rtol=1e-5,
                    atol=1e-6,
                )


if __name__ == "__main__":
    unittest.main()
