"""Descriptive optimal quality matching for P(negative); no window-level p-values."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "outputs_v2" / "diagnostic_personal" / "diagnostic_window_predictions.csv"
OUTPUT = ROOT / "outputs_v2" / "diagnostic_personal"
QUALITY = [
    "raw_rms",
    "raw_peak_to_peak",
    "low_0_5_4hz_energy_ratio",
    "abnormal_peak_count",
]


def split_half(frame: pd.DataFrame, stage: str, half: str) -> pd.DataFrame:
    selected = frame[frame["stage"] == stage].sort_values("signal_end_unix")
    midpoint = len(selected) // 2
    return selected.iloc[:midpoint] if half == "first" else selected.iloc[midpoint:]


def optimal_match(left: pd.DataFrame, right: pd.DataFrame, caliper: float = 0.5):
    pooled = pd.concat([left[QUALITY], right[QUALITY]])
    scale = pooled.std().replace(0, 1)
    center = pooled.mean()
    left_z = ((left[QUALITY] - center) / scale).to_numpy()
    right_z = ((right[QUALITY] - center) / scale).to_numpy()
    distances = np.sqrt(((left_z[:, None, :] - right_z[None, :, :]) ** 2).mean(axis=2))
    left_index, right_index = linear_sum_assignment(distances)
    keep = distances[left_index, right_index] <= caliper
    return (
        left.iloc[left_index[keep]].reset_index(drop=True),
        right.iloc[right_index[keep]].reset_index(drop=True),
        distances[left_index[keep], right_index[keep]],
    )


def compare(frame, left_name, left, right_name, right):
    matched_left, matched_right, distance = optimal_match(left, right)
    before = float(right["prob_sad"].mean() - left["prob_sad"].mean())
    after = float(
        matched_right["prob_sad"].mean() - matched_left["prob_sad"].mean()
    ) if len(matched_left) else None
    rows = []
    for index in range(len(matched_left)):
        rows.append(
            {
                "comparison": f"{right_name}_minus_{left_name}",
                "pair": index,
                "quality_distance": distance[index],
                "left_window_index": matched_left.loc[index, "window_index"],
                "right_window_index": matched_right.loc[index, "window_index"],
                "left_prob_negative": matched_left.loc[index, "prob_sad"],
                "right_prob_negative": matched_right.loc[index, "prob_sad"],
                "prob_negative_difference": (
                    matched_right.loc[index, "prob_sad"]
                    - matched_left.loc[index, "prob_sad"]
                ),
            }
        )
    quality_balance = {}
    for name in QUALITY:
        pooled_std = pd.concat([matched_left[name], matched_right[name]]).std()
        quality_balance[name] = (
            float((matched_right[name].mean() - matched_left[name].mean()) / pooled_std)
            if len(matched_left) and pooled_std > 0
            else None
        )
    poor_balance = any(
        value is not None and abs(value) > 0.25 for value in quality_balance.values()
    )
    result = {
        "comparison": f"{right_name}_minus_{left_name}",
        "left_windows": len(left),
        "right_windows": len(right),
        "matched_pairs": len(matched_left),
        "caliper_standardized_rms_distance": 0.5,
        "unmatched_mean_prob_negative_difference": before,
        "matched_mean_prob_negative_difference": after,
        "matched_left_prob_negative_mean": (
            float(matched_left["prob_sad"].mean()) if len(matched_left) else None
        ),
        "matched_right_prob_negative_mean": (
            float(matched_right["prob_sad"].mean()) if len(matched_right) else None
        ),
        "post_match_standardized_quality_differences": quality_balance,
        "interpretation_limit": "Overlapping windows; descriptive effect only, no p-value.",
        "common_support_warning": (
            "No adequate quality-matched common support."
            if len(matched_left) == 0
            else (
                "Residual quality imbalance remains after matching."
                if poor_balance
                else None
            )
        ),
    }
    return result, rows


def main():
    frame = pd.read_csv(WINDOWS)
    stable = frame[(~frame["transition_window"]) & frame["stage"].ne("unassigned")]
    normal_first = split_half(stable, "normal_task", "first")
    normal_second = split_half(stable, "normal_task", "second")
    washout_second = stable[stable["stage"] == "washout_2"]
    comparisons = (
        ("normal_first", normal_first, "normal_second", normal_second),
        ("normal_first", normal_first, "washout_2", washout_second),
        ("normal_second", normal_second, "washout_2", washout_second),
    )
    results, pair_rows = [], []
    for left_name, left, right_name, right in comparisons:
        result, rows = compare(stable, left_name, left, right_name, right)
        results.append(result)
        pair_rows.extend(rows)
    pd.DataFrame(pair_rows).to_csv(
        OUTPUT / "quality_matched_pairs.csv", index=False, encoding="utf-8-sig"
    )
    report = {
        "matching_features": QUALITY,
        "method": "Hungarian optimal one-to-one matching after pooled z-scoring",
        "comparisons": results,
        "global_limit": (
            "Windows overlap by 93.3%; matching is diagnostic and cannot establish "
            "independent-sample significance or causal state effects."
        ),
    }
    (OUTPUT / "quality_matched_probability_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
