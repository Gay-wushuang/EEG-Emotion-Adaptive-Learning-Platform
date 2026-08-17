"""Run the deployed model over pre-scaled sliding-window NPZ features."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import EmotionInferenceModel, InferenceConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scalers-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    parser.add_argument("--sample-rate", type=float, default=512.0)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    args = parser.parse_args()

    archive = np.load(args.features)
    modalities = ("filtered", "powerspec", "att", "med")
    count = archive[modalities[0]].shape[0]
    for modality in modalities:
        if archive[modality].shape != (count, 10, 4):
            raise ValueError(f"Invalid {modality} shape: {archive[modality].shape}")

    model = EmotionInferenceModel(
        InferenceConfig(
            model_path=args.model,
            modalities=modalities,
            scalers_dir=args.scalers_dir,
            skip_scaling=True,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(count):
        sample = {modality: archive[modality][index] for modality in modalities}
        started = time.perf_counter()
        predicted, probabilities = model.predict(sample)
        latency_ms = (time.perf_counter() - started) * 1000.0
        by_class = dict(zip(model.class_names, probabilities.tolist()))
        confidence = float(np.max(probabilities))
        signal_good = (
            float(archive["signal_good_fraction"][index]) >= 0.8
            and float(archive["poor_signal"][index]) <= 50
        )
        output_class = predicted
        if not signal_good:
            output_class = "poor_signal"
        elif confidence < args.confidence_threshold:
            output_class = "uncertain"
        rows.append(
            {
                "window_index": index,
                "signal_start_seconds": index * args.stride_seconds,
                "signal_end_seconds": index * args.stride_seconds + args.window_seconds,
                "predicted_class": output_class,
                "raw_class": predicted,
                "prob_positive": by_class["happy"],
                "prob_neutral": by_class["normal"],
                "prob_negative": by_class["sad"],
                "confidence": confidence,
                "attention": float(archive["attention"][index]),
                "meditation": float(archive["meditation"][index]),
                "poor_signal": float(archive["poor_signal"][index]),
                "signal_good_fraction": float(archive["signal_good_fraction"][index]),
                "inference_latency_ms": latency_ms,
            }
        )

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    probabilities = np.asarray(
        [[row["prob_positive"], row["prob_neutral"], row["prob_negative"]] for row in rows]
    )
    raw_classes = [row["raw_class"] for row in rows]
    transitions = sum(a != b for a, b in zip(raw_classes, raw_classes[1:]))
    print(
        {
            "windows": count,
            "raw_class_counts": {name: raw_classes.count(name) for name in model.class_names},
            "rejected": sum(row["predicted_class"] in {"poor_signal", "uncertain"} for row in rows),
            "confidence_mean": float(np.max(probabilities, axis=1).mean()),
            "confidence_std": float(np.max(probabilities, axis=1).std()),
            "probability_std": probabilities.std(axis=0).tolist(),
            "adjacent_class_transitions": transitions,
            "latency_mean_ms": float(np.mean([row["inference_latency_ms"] for row in rows])),
            "latency_max_ms": float(np.max([row["inference_latency_ms"] for row in rows])),
        }
    )


if __name__ == "__main__":
    main()
