"""Build anonymous Raw/ATT/MED canonical feature fixture."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eeg_modular"))

from eeg_emotion.features.canonical import CanonicalFeatureConfig, extract_canonical_features
from eeg_emotion.features.canonical_io import load_training_window


def main() -> None:
    cfg = CanonicalFeatureConfig()
    sample = ROOT / "time_data_preprocess" / "data" / "happy" / "sample1"
    raw_count = sum(1 for _ in (sample / "rawwave.csv").open(encoding="utf-8-sig")) - 1
    crop_start = (raw_count - cfg.window_samples) // 2
    raw, att, med, metadata = load_training_window(sample, crop_start, cfg)
    features = extract_canonical_features(raw, att, med, cfg)
    output = Path(__file__).parent / "fixtures" / "canonical_golden_sample.npz"
    np.savez_compressed(
        output,
        raw=raw,
        att=att,
        med=med,
        **{f"expected_{name}": value for name, value in features.items()},
        **{name: np.asarray([value]) for name, value in metadata.items()},
    )
    print(output)


if __name__ == "__main__":
    main()
