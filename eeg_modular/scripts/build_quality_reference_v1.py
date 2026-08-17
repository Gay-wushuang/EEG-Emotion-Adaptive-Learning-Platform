"""Fit frozen quality warning/OOD limits from the 277 training windows only."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eeg_emotion.features.canonical import CanonicalFeatureConfig
from eeg_emotion.features.canonical_io import read_time_value_csv
from eeg_emotion.features.quality import (
    QUALITY_FEATURES,
    QUALITY_METRIC_VERSION,
    compute_quality_metrics,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    manifest = pd.read_csv(ROOT / "features_v2" / "manifest.csv")
    source = ROOT.parent / "time_data_preprocess" / "data"
    cfg = CanonicalFeatureConfig()
    rows = []
    for item in manifest.itertuples(index=False):
        sample_dir = source / item.sample_id
        _, raw = read_time_value_csv(sample_dir / "rawwave.csv")
        start = int(item.crop_start_sample)
        stop = start + cfg.window_samples
        _, sig_values = read_time_value_csv(sample_dir / "sigqual.csv")
        poor_full = np.interp(
            np.arange(len(raw), dtype=np.float64),
            np.linspace(0, len(raw) - 1, len(sig_values)),
            sig_values,
        )
        poor = poor_full[start:stop]
        rows.append(
            {
                "sample_id": item.sample_id,
                "subject_id": item.subject_id,
                **compute_quality_metrics(raw[start:stop], poor, cfg.sample_rate),
            }
        )
    reference = pd.DataFrame(rows)
    output = ROOT / "quality_reference_v1"
    output.mkdir(exist_ok=True)
    reference.to_csv(output / "training_quality_277.csv", index=False, encoding="utf-8-sig")

    high_metrics = [
        name for name in QUALITY_FEATURES
        if name not in (
            "poor_signal_mean",
            "poor_signal_bad_fraction",
            "poor_signal_change_count",
        )
    ]
    limits = {}
    for name in high_metrics:
        limits[name] = {
            "warning_high": float(reference[name].quantile(0.95)),
            "ood_high": float(reference[name].quantile(0.995)),
        }
    for name in ("raw_rms", "first_difference_rms"):
        limits[name].update(
            {
                "warning_low": float(reference[name].quantile(0.05)),
                "ood_low": float(reference[name].quantile(0.005)),
            }
        )
    policy = {
        "version": "Quality Gate v1",
        "metric_version": QUALITY_METRIC_VERSION,
        "fit_dataset": "277 canonical training windows only",
        "fit_subjects": int(manifest["subject_id"].nunique()),
        "percentile_policy": {
            "warning": "outside training 5th/95th percentile where applicable",
            "low_ood": "outside training 0.5th/99.5th percentile where applicable",
        },
        "poor_signal_rules": {
            "warning_bad_fraction_greater_than": 0.0,
            "low_ood_bad_fraction_greater_than": 0.20,
            "low_ood_mean_at_least": 50.0,
        },
        "limits": limits,
        "aggregation": (
            "low_ood if any OOD rule fires; warning if no OOD and any warning fires; "
            "otherwise trusted"
        ),
        "emotion_independent": True,
        "frozen_before_repeat_capture": True,
        "source_sha256": {
            "quality_algorithm": sha256(ROOT / "eeg_emotion" / "features" / "quality.py"),
            "builder": sha256(Path(__file__)),
            "canonical_config": sha256(ROOT / "features_v2" / "canonical_config.json"),
        },
    }
    (output / "quality_gate_policy.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(policy, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
