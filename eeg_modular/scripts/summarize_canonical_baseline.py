"""Aggregate grouped-CV baseline predictions into class and subject metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.predictions.open(encoding="utf-8")))
    class_names = ["happy", "normal", "sad"]
    class_to_index = {name: index for index, name in enumerate(class_names)}

    seed_metrics = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        current = [row for row in rows if int(row["seed"]) == seed]
        truth = np.asarray([class_to_index[row["true_label"]] for row in current])
        prediction = np.asarray([class_to_index[row["predicted_label"]] for row in current])
        seed_metrics.append(
            {
                "seed": seed,
                "accuracy": accuracy_score(truth, prediction),
                "macro_f1": f1_score(truth, prediction, average="macro"),
                "classification_report": classification_report(
                    truth,
                    prediction,
                    labels=np.arange(3),
                    target_names=class_names,
                    output_dict=True,
                    zero_division=0,
                ),
                "confusion_matrix": confusion_matrix(
                    truth, prediction, labels=np.arange(3)
                ).tolist(),
            }
        )

    truth = np.asarray([class_to_index[row["true_label"]] for row in rows])
    prediction = np.asarray([class_to_index[row["predicted_label"]] for row in rows])
    pooled_matrix = confusion_matrix(truth, prediction, labels=np.arange(3))
    subject_results = []
    for subject in sorted({int(row["subject_id"]) for row in rows}):
        current = [row for row in rows if int(row["subject_id"]) == subject]
        subject_truth = [class_to_index[row["true_label"]] for row in current]
        subject_prediction = [class_to_index[row["predicted_label"]] for row in current]
        subject_results.append(
            {
                "subject_id": f"subject_{subject + 1:03d}",
                "samples_across_seeds": len(current),
                "accuracy": accuracy_score(subject_truth, subject_prediction),
                "macro_f1": f1_score(
                    subject_truth, subject_prediction, average="macro", zero_division=0
                ),
            }
        )

    class_summary = {}
    for class_name in class_names:
        class_summary[class_name] = {}
        for metric in ("precision", "recall", "f1-score"):
            values = [
                seed["classification_report"][class_name][metric] for seed in seed_metrics
            ]
            class_summary[class_name][f"{metric}_mean"] = float(np.mean(values))
            class_summary[class_name][f"{metric}_std"] = float(np.std(values, ddof=1))

    output = {
        "seed_metrics": seed_metrics,
        "per_class_across_seeds": class_summary,
        "pooled_confusion_matrix_three_seeds": pooled_matrix.tolist(),
        "per_subject": subject_results,
        "subject_accuracy_mean": float(np.mean([row["accuracy"] for row in subject_results])),
        "subject_accuracy_std": float(np.std([row["accuracy"] for row in subject_results], ddof=1)),
        "subject_accuracy_min": float(np.min([row["accuracy"] for row in subject_results])),
        "subject_accuracy_max": float(np.max([row["accuracy"] for row in subject_results])),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aggregate_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(pooled_matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axis.text(column, row, int(pooled_matrix[row, column]), ha="center", va="center")
    axis.set_xticks(range(3), class_names)
    axis.set_yticks(range(3), class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Canonical baseline: pooled grouped-CV confusion matrix")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(args.output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
