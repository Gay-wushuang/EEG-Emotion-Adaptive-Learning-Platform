"""Offline diagnostic of Production Baseline v1 on a canonical capture tensor."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eeg_emotion.models.production_baseline import (  # noqa: E402
    CLASS_NAMES,
    MODALITIES,
    load_production_package,
    predict_probabilities,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--capture-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model, scalers, contract = load_production_package(ROOT / "production_baseline_v1")
    fixture = np.load(args.features)
    arrays = {name: fixture[name] for name in MODALITIES}
    starts = fixture["window_end_sample"] - (30 * 512 - 1)
    stops = fixture["window_end_sample"] + 1
    if any(value.shape != (len(starts), 10, 4) for value in arrays.values()):
        raise RuntimeError("Canonical capture tensor shape mismatch")

    scaled_stats = {}
    for name, values in arrays.items():
        scaled = scalers[name].transform(values.reshape(-1, 4)).reshape(values.shape)
        scaled_stats[name] = {
            "min": float(np.min(scaled)),
            "max": float(np.max(scaled)),
            "mean": float(np.mean(scaled)),
            "std": float(np.std(scaled)),
            "abs_gt_5_fraction": float(np.mean(np.abs(scaled) > 5)),
            "abs_gt_10_fraction": float(np.mean(np.abs(scaled) > 10)),
            "nonfinite_count": int(np.size(scaled) - np.isfinite(scaled).sum()),
        }

    repetitions = 100
    started = time.perf_counter()
    for _ in range(repetitions):
        probabilities = predict_probabilities(model, scalers, arrays)
    latency_ms = 1000 * (time.perf_counter() - started) / (repetitions * len(starts))

    rows = list(csv.DictReader(args.capture_csv.open("r", encoding="utf-8-sig", newline="")))
    poor = np.asarray(
        [float(row["poor_signal"]) if row["poor_signal"] else np.nan for row in rows],
        dtype=np.float64,
    )
    for index in range(1, len(poor)):
        if not np.isfinite(poor[index]):
            poor[index] = poor[index - 1]
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    output_rows = []
    for index, (start, stop) in enumerate(zip(starts, stops)):
        quality = poor[int(start):int(stop)]
        good_fraction = float(np.mean(np.isfinite(quality) & (quality < 50)))
        reasons = []
        if good_fraction < 0.8:
            reasons.append("poor_signal")
        if confidence[index] < 0.60:
            reasons.append("low_confidence")
        output_rows.append(
            {
                "window_index": index,
                "signal_start_seconds": float(start / 512),
                "signal_end_seconds": float(stop / 512),
                "predicted_class": CLASS_NAMES[int(predictions[index])],
                "prob_happy": float(probabilities[index, 0]),
                "prob_normal": float(probabilities[index, 1]),
                "prob_sad": float(probabilities[index, 2]),
                "confidence": float(confidence[index]),
                "signal_good_fraction": good_fraction,
                "rejected": bool(reasons),
                "rejection_reason": ",".join(reasons),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "production_baseline_v1_capture_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    class_counts = {
        name: int(np.sum(predictions == index)) for index, name in enumerate(CLASS_NAMES)
    }
    accepted = [row for row in output_rows if not row["rejected"]]
    report = {
        "model": contract["name"],
        "performance_attribution": "grouped_cross_validation_not_this_capture",
        "window_count": len(output_rows),
        "class_counts": class_counts,
        "confidence": {
            "min": float(np.min(confidence)),
            "mean": float(np.mean(confidence)),
            "max": float(np.max(confidence)),
            "std": float(np.std(confidence)),
        },
        "fixed_single_class_near_100_percent": bool(
            len([count for count in class_counts.values() if count]) == 1
            and float(np.min(confidence)) >= 0.99
        ),
        "scaled_feature_diagnostics": scaled_stats,
        "rejection_rule": "poor_signal good fraction < 0.8 or confidence < 0.60",
        "accepted_windows": len(accepted),
        "rejected_windows": len(output_rows) - len(accepted),
        "mean_inference_latency_ms_per_window": latency_ms,
    }
    (args.output_dir / "production_baseline_v1_capture_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
