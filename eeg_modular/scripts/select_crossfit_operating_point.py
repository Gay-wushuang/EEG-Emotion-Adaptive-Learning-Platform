"""Select a confidence operating point without tuning on the evaluated subject."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("outputs_v2/confidence_calibration_v1/oof_predictions_277.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs_v2/confidence_calibration_v1/crossfit_90pct_operating_point.json"))
    parser.add_argument("--target-accepted-accuracy", type=float, default=0.90)
    parser.add_argument("--minimum-selection-samples", type=int, default=15)
    args = parser.parse_args()

    data = pd.read_csv(args.predictions)
    probabilities = data[["calibrated_prob_happy", "calibrated_prob_normal", "calibrated_prob_sad"]].to_numpy()
    truth = data["label_index"].to_numpy()
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    subjects = data["subject_id"].to_numpy()
    accepted = np.zeros(len(data), dtype=bool)
    subject_rows = []
    for subject in np.unique(subjects):
        selection = subjects != subject
        threshold = 0.995
        for candidate in np.arange(0.34, 0.996, 0.005):
            mask = selection & (confidence >= candidate)
            if mask.sum() < args.minimum_selection_samples:
                continue
            if (prediction[mask] == truth[mask]).mean() >= args.target_accepted_accuracy:
                threshold = float(candidate)
                break
        evaluation = (subjects == subject) & (confidence >= threshold)
        accepted |= evaluation
        subject_rows.append({
            "subject_id": int(subject), "selected_threshold": threshold,
            "accepted": int(evaluation.sum()),
            "correct": int((prediction[evaluation] == truth[evaluation]).sum()),
        })

    rng = np.random.default_rng(42)
    unique_subjects = np.unique(subjects)
    bootstrap_accuracy = []
    for _ in range(20000):
        sampled = rng.choice(unique_subjects, len(unique_subjects), replace=True)
        indices = np.concatenate([np.flatnonzero((subjects == subject) & accepted) for subject in sampled])
        if len(indices):
            bootstrap_accuracy.append(float((prediction[indices] == truth[indices]).mean()))
    report = {
        "selection": "leave_one_subject_out_threshold_crossfitting",
        "target_accepted_accuracy_on_other_subjects": args.target_accepted_accuracy,
        "accepted_count": int(accepted.sum()),
        "total_count": int(len(data)),
        "coverage": float(accepted.mean()),
        "accepted_accuracy": float((prediction[accepted] == truth[accepted]).mean()),
        "accepted_accuracy_subject_bootstrap_ci95": [float(x) for x in np.quantile(bootstrap_accuracy, [0.025, 0.975])],
        "accepted_macro_f1": float(f1_score(truth[accepted], prediction[accepted], average="macro")),
        "confusion_matrix": confusion_matrix(truth[accepted], prediction[accepted], labels=[0, 1, 2]).tolist(),
        "threshold_median": float(np.median([row["selected_threshold"] for row in subject_rows])),
        "subject_results": subject_rows,
        "required_wording": "High-confidence accepted-window accuracy; rejected windows return uncertain. Not full-coverage three-class accuracy.",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "subject_results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
