"""Benchmark CPU computation from a 30-second Raw window to logged decision."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "realtime_inference"))

from eeg_emotion.features.canonical import CanonicalFeatureConfig, extract_canonical_features
from eeg_emotion.models.production_baseline import MODALITIES, load_production_package
from realtime_inference.tools.convert_canonical_capture import read_capture
from src.decision import EWMASustainedNegativeDecision


def summarize(values):
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": float(np.percentile(values, 95)),
    }


def main() -> None:
    capture = ROOT / "realtime_inference" / "captures" / "baseline_120s.csv"
    package = ROOT / "production_baseline_v1"
    output = ROOT / "outputs_v2" / "production_pipeline_benchmark.json"
    data = read_capture(capture)
    cfg = CanonicalFeatureConfig()
    stop = cfg.window_samples
    raw, att, med = data["raw"][:stop], data["att"][:stop], data["med"][:stop]
    model, scalers, _ = load_production_package(package, device="cpu")
    decision = EWMASustainedNegativeDecision(2)
    repetitions = 100
    timings = {name: [] for name in ("feature", "scaler", "model", "decision_log", "end_to_end")}

    for index in range(repetitions + 5):
        total_start = time.perf_counter()
        started = time.perf_counter()
        features = extract_canonical_features(raw, att, med, cfg)
        feature_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        tensors = {}
        for modality in MODALITIES:
            value = features[modality][None]
            scaled = scalers[modality].transform(value.reshape(-1, 4)).reshape(value.shape)
            tensors[modality] = torch.tensor(scaled, dtype=torch.float32)
        scaler_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        with torch.no_grad():
            probabilities = torch.softmax(model(tensors), dim=1).numpy()[0]
        model_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        state = decision.update(probabilities, timestamp=float(index * 2), eligible=True)
        json.dumps(
            {
                "probabilities": probabilities.tolist(),
                "confidence": float(probabilities.max()),
                "negative_ewma": state.negative_ewma,
                "intervention": state.intervention_triggered,
            }
        )
        decision_ms = (time.perf_counter() - started) * 1000
        total_ms = (time.perf_counter() - total_start) * 1000
        if index >= 5:
            for name, value in (
                ("feature", feature_ms),
                ("scaler", scaler_ms),
                ("model", model_ms),
                ("decision_log", decision_ms),
                ("end_to_end", total_ms),
            ):
                timings[name].append(value)

    report = {
        "device": "cpu",
        "repetitions_after_warmup": repetitions,
        "observation_window_seconds": 30,
        "update_interval_seconds": 2,
        "timings": {name: summarize(values) for name, values in timings.items()},
        "wording": "First result requires 30 s warm-up; thereafter updates every 2 s.",
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

