"""Analyze seven-stage capture without rejection and with branch logit diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eeg_emotion.models.production_baseline import (  # noqa: E402
    CLASS_NAMES,
    MODALITIES,
    load_production_package,
)
from eeg_emotion.features.quality import QUALITY_FEATURES, compute_quality_metrics
from eeg_emotion.features.quality_gate import evaluate_quality


def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model, scalers, _ = load_production_package(ROOT / "production_baseline_v1")
    feature_file = np.load(args.features)
    tensors = {}
    for modality in MODALITIES:
        values = feature_file[modality]
        scaled = scalers[modality].transform(values.reshape(-1, 4)).reshape(values.shape)
        tensors[modality] = torch.tensor(scaled, dtype=torch.float32)
    with torch.no_grad():
        diagnostics = {
            name: value.cpu().numpy() for name, value in model.diagnostic_outputs(tensors).items()
        }
    fusion_probabilities = softmax(diagnostics["logits_fusion"])
    filtered_zero_probabilities = softmax(diagnostics["logits_filtered_only"])
    bandpower_zero_probabilities = softmax(diagnostics["logits_bandpower_only"])

    capture = pd.read_csv(
        args.capture,
        usecols=[
            "timestamp_unix", "packet_received_unix", "sample_index",
            "raw_eeg", "poor_signal",
        ],
    )
    sample_index = capture["sample_index"].to_numpy(dtype=np.float64)
    packet_time = capture["packet_received_unix"].to_numpy(dtype=np.float64)
    slope, intercept = np.polyfit(sample_index[::128], packet_time[::128], 1)
    fitted_sample_rate = 1.0 / slope
    starts = feature_file["window_end_sample"] - (30 * 512 - 1)
    stops = feature_file["window_end_sample"] + 1
    # Features remain fixed at 15360 nominal samples. Wall-clock stage alignment uses
    # an affine sample-index/packet-time fit to avoid cumulative device-clock drift.
    window_start = intercept + starts * slope
    window_end = intercept + stops * slope

    events = pd.read_csv(args.events)
    stage_ranges = {}
    planned_ranges = {}
    for stage, group in events.groupby("stage"):
        starts_event = group[group["event"] == "stage_start"]["timestamp_unix"]
        ends_event = group[group["event"] == "stage_end_self_report"]["timestamp_unix"]
        if len(starts_event) == 1 and len(ends_event) == 1:
            stage_start = float(starts_event.iloc[0])
            planned_seconds = float(
                group[group["event"] == "stage_start"]["planned_seconds"].iloc[0]
            )
            stage_ranges[stage] = (stage_start, stage_start + planned_seconds)
            planned_ranges[stage] = {
                "planned_start": stage_start,
                "planned_end": stage_start + planned_seconds,
                "self_report_recorded_at": float(ends_event.iloc[0]),
            }

    assigned_stage, transition = [], []
    for start, end in zip(window_start, window_end):
        stage_name, is_transition = "unassigned", True
        for stage, (stage_start, stage_end) in stage_ranges.items():
            if end > stage_start and end <= stage_end:
                stage_name = stage
                is_transition = start < stage_start
                break
        assigned_stage.append(stage_name)
        transition.append(is_transition)

    rows = pd.DataFrame(
        {
            "window_index": np.arange(len(starts)),
            "signal_start_unix": window_start,
            "signal_end_unix": window_end,
            "stage": assigned_stage,
            "transition_window": transition,
            "predicted_class": [CLASS_NAMES[index] for index in fusion_probabilities.argmax(axis=1)],
        }
    )
    raw = capture["raw_eeg"].to_numpy(dtype=np.float64)
    poor_signal = capture["poor_signal"].ffill().bfill().to_numpy(dtype=np.float64)
    quality = {name: [] for name in QUALITY_FEATURES}
    for start, stop in zip(starts.astype(int), stops.astype(int)):
        current = compute_quality_metrics(
            raw[start:stop], poor_signal[start:stop], sample_rate=512
        )
        for name, value in current.items():
            quality[name].append(value)
    for name, values in quality.items():
        rows[name] = values
    quality_policy = json.loads(
        (ROOT / "quality_reference_v1" / "quality_gate_policy.json").read_text(
            encoding="utf-8"
        )
    )
    gate_results = [
        evaluate_quality(
            {name: float(row[name]) for name in QUALITY_FEATURES},
            quality_policy,
        )
        for _, row in rows.iterrows()
    ]
    rows["quality_level"] = [item["quality_level"] for item in gate_results]
    rows["emotion_interpretation_allowed"] = [
        item["emotion_interpretation_allowed"] for item in gate_results
    ]
    rows["quality_warning_reasons"] = [
        "|".join(item["warning_reasons"]) for item in gate_results
    ]
    rows["quality_ood_reasons"] = [
        "|".join(item["ood_reasons"]) for item in gate_results
    ]
    for index, name in enumerate(CLASS_NAMES):
        rows[f"prob_{name}"] = fusion_probabilities[:, index]
        rows[f"logit_{name}"] = diagnostics["logits_fusion"][:, index]
        rows[f"filtered_zero_other_prob_{name}"] = filtered_zero_probabilities[:, index]
        rows[f"bandpower_zero_other_prob_{name}"] = bandpower_zero_probabilities[:, index]
        rows[f"filtered_logit_contribution_{name}"] = diagnostics[
            "logit_contribution_filtered"
        ][:, index]
        rows[f"bandpower_logit_contribution_{name}"] = diagnostics[
            "logit_contribution_bandpower"
        ][:, index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "diagnostic_window_predictions.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(args.output_dir / "diagnostic_embeddings_logits.npz", **diagnostics)

    stable = rows[(~rows["transition_window"]) & (rows["stage"] != "unassigned")]
    summary_columns = [
        column for column in rows.columns
        if column.startswith(("prob_", "logit_", "filtered_", "bandpower_"))
    ]
    summary_columns += list(quality)
    flat_rows = []
    for stage, group in stable.groupby("stage"):
        record = {"stage": stage, "window_count": len(group)}
        for column in summary_columns:
            record[f"{column}_mean"] = float(group[column].mean())
            record[f"{column}_std"] = float(group[column].std())
            record[f"{column}_count"] = int(group[column].count())
        flat_rows.append(record)
    pd.DataFrame(flat_rows).to_csv(
        args.output_dir / "diagnostic_stage_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    half_rows = []
    for stage, group in stable.groupby("stage"):
        ordered = group.sort_values("signal_end_unix")
        for half_name, half in (
            ("first_half", ordered.iloc[: len(ordered) // 2]),
            ("second_half", ordered.iloc[len(ordered) // 2 :]),
        ):
            half_rows.append(
                {
                    "stage": stage,
                    "half": half_name,
                    "window_count": len(half),
                    **{
                        f"prob_{name}_mean": float(half[f"prob_{name}"].mean())
                        for name in CLASS_NAMES
                    },
                    **{
                        name: float(half[name].mean()) for name in quality
                    },
                }
            )
    pd.DataFrame(half_rows).to_csv(
        args.output_dir / "diagnostic_stage_halves.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = {
        "total_windows": len(rows),
        "stable_stage_windows": len(stable),
        "transition_or_unassigned_windows": len(rows) - len(stable),
        "automatic_intervention": False,
        "confidence_rejection_applied_to_stage_analysis": False,
        "branch_diagnostic_note": (
            "filtered/bandpower zero-other probabilities are counterfactual branch diagnostics "
            "inside the frozen fusion model, not separately trained classifiers"
        ),
        "stage_ranges": planned_ranges,
        "last_available_signal_time": float(window_end.max()),
        "timeline_audit": {
            "raw_samples": len(capture),
            "sample_index_continuous": bool(
                np.array_equal(
                    capture["sample_index"].to_numpy(), np.arange(len(capture))
                )
            ),
            "nominal_512hz_signal_span_seconds": float(
                (len(capture) - 1) / 512.0
            ),
            "packet_receive_span_seconds": float(packet_time[-1] - packet_time[0]),
            "fitted_wall_clock_sample_rate_hz": float(fitted_sample_rate),
            "alignment": "affine sample_index to packet_received_unix fit",
        },
        "truncated_stages": [
            stage
            for stage, bounds in planned_ranges.items()
            if bounds["planned_end"] > float(window_end.max())
        ],
        "stage_mean_fusion_probabilities": {
            stage: {
                name: float(group[f"prob_{name}"].mean()) for name in CLASS_NAMES
            }
            for stage, group in stable.groupby("stage")
        },
        "stage_quality_levels": {
            stage: {
                level: int(count)
                for level, count in group["quality_level"].value_counts().items()
            }
            for stage, group in stable.groupby("stage")
        },
    }
    (args.output_dir / "diagnostic_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
