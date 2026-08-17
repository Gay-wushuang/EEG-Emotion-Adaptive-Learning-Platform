"""Summarize grouped CVAE augmentation against frozen no-CVAE baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import classification_report


def aggregate(metrics: pd.DataFrame) -> dict:
    result = {}
    for name in ("accuracy", "macro_f1"):
        values = metrics[name].to_numpy(float)
        result[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "ci95": [
                float(values.mean() - 1.96 * values.std(ddof=1) / np.sqrt(len(values))),
                float(values.mean() + 1.96 * values.std(ddof=1) / np.sqrt(len(values))),
            ],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs_v2/ablations"))
    parser.add_argument("--cvae-dir", type=Path, default=Path("outputs_v2/cvae_grouped_v1"))
    args = parser.parse_args()
    baseline_metrics = pd.read_csv(args.baseline_dir / "fold_metrics.csv")
    baseline_metrics = baseline_metrics[
        baseline_metrics["experiment"] == "D_filtered_bandpower"
    ].copy()
    cvae_metrics = pd.read_csv(args.cvae_dir / "fold_metrics.csv")
    baseline_predictions = pd.read_csv(args.baseline_dir / "predictions.csv")
    baseline_predictions = baseline_predictions[
        baseline_predictions["experiment"] == "D_filtered_bandpower"
    ].copy()
    cvae_predictions = pd.read_csv(args.cvae_dir / "predictions.csv")
    class_names = ["happy", "normal", "sad"]
    reports = {}
    for name, predictions in (
        ("baseline", baseline_predictions), ("cvae", cvae_predictions)
    ):
        reports[name] = classification_report(
            predictions["true_label"], predictions["predicted_label"],
            labels=class_names, output_dict=True, zero_division=0,
        )
    subject_rows = []
    for name, predictions in (
        ("baseline", baseline_predictions), ("cvae", cvae_predictions)
    ):
        predictions = predictions.assign(
            correct=predictions["true_label"] == predictions["predicted_label"]
        )
        for subject, group in predictions.groupby("subject_id"):
            subject_rows.append(
                {"model": name, "subject_id": int(subject), "accuracy": group["correct"].mean()}
            )
    subjects = pd.DataFrame(subject_rows)
    paired = subjects.pivot(index="subject_id", columns="model", values="accuracy").dropna()
    differences = paired["cvae"] - paired["baseline"]
    rng = np.random.default_rng(42)
    bootstrap = np.asarray([
        rng.choice(differences.to_numpy(), size=len(differences), replace=True).mean()
        for _ in range(10000)
    ])
    payload = {
        "baseline": aggregate(baseline_metrics),
        "cvae": aggregate(cvae_metrics),
        "pooled_class_f1": {
            model: {label: float(report[label]["f1-score"]) for label in class_names}
            for model, report in reports.items()
        },
        "paired_subject_comparison": {
            "subjects": len(paired),
            "mean_accuracy_difference_cvae_minus_baseline": float(differences.mean()),
            "bootstrap_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "wilcoxon_p": float(wilcoxon(differences).pvalue),
        },
        "decision": "reject_cvae_for_production" if differences.mean() <= 0 else "candidate",
    }
    (args.cvae_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paired.assign(difference=differences).to_csv(
        args.cvae_dir / "subject_comparison.csv", encoding="utf-8-sig"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
