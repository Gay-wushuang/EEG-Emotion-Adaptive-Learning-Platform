"""Summarize ablations and perform subject-level paired statistics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, t, wilcoxon
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

EXPERIMENT_ORDER = [
    "A_filtered",
    "B_bandpower",
    "C_att_med",
    "D_filtered_bandpower",
    "E_all",
]
CLASS_NAMES = ["happy", "normal", "sad"]


def read_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def ci95(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean
    margin = float(t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean - margin, mean + margin


def subject_metrics(rows: list[dict]) -> list[dict]:
    output = []
    for experiment in EXPERIMENT_ORDER:
        experiment_rows = [row for row in rows if row["experiment"] == experiment]
        for subject in sorted({int(row["subject_id"]) for row in experiment_rows}):
            current = [row for row in experiment_rows if int(row["subject_id"]) == subject]
            truth = [CLASS_NAMES.index(row["true_label"]) for row in current]
            prediction = [CLASS_NAMES.index(row["predicted_label"]) for row in current]
            output.append(
                {
                    "experiment": experiment,
                    "subject_id": f"subject_{subject + 1:03d}",
                    "accuracy": accuracy_score(truth, prediction),
                    "macro_f1": f1_score(
                        truth, prediction, average="macro", zero_division=0
                    ),
                    "predictions_across_seeds": len(current),
                }
            )
    return output


def paired_statistics(subject_rows: list[dict], bootstrap_iterations: int = 10000) -> dict:
    rng = np.random.default_rng(20260726)
    by_experiment = {
        experiment: {
            row["subject_id"]: float(row["accuracy"])
            for row in subject_rows
            if row["experiment"] == experiment
        }
        for experiment in EXPERIMENT_ORDER
    }
    results = {}
    reference = by_experiment["E_all"]
    for comparator in EXPERIMENT_ORDER[:-1]:
        subjects = sorted(set(reference) & set(by_experiment[comparator]))
        e_values = np.asarray([reference[subject] for subject in subjects])
        c_values = np.asarray([by_experiment[comparator][subject] for subject in subjects])
        differences = e_values - c_values
        bootstrap_means = np.asarray(
            [
                np.mean(rng.choice(differences, size=len(differences), replace=True))
                for _ in range(bootstrap_iterations)
            ]
        )
        nonzero = differences[differences != 0]
        if nonzero.size:
            ranks = rankdata(np.abs(nonzero))
            rank_biserial = float(
                (np.sum(ranks[nonzero > 0]) - np.sum(ranks[nonzero < 0])) / np.sum(ranks)
            )
            statistic, p_value = wilcoxon(differences, zero_method="wilcox")
        else:
            rank_biserial, statistic, p_value = 0.0, 0.0, 1.0
        results[f"E_all_vs_{comparator}"] = {
            "subjects": len(subjects),
            "mean_accuracy_difference": float(np.mean(differences)),
            "bootstrap_95_ci": [
                float(np.quantile(bootstrap_means, 0.025)),
                float(np.quantile(bootstrap_means, 0.975)),
            ],
            "wilcoxon_statistic": float(statistic),
            "wilcoxon_p_value": float(p_value),
            "rank_biserial_effect": rank_biserial,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--fold-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prediction_rows = read_rows(args.predictions)
    fold_rows = read_rows(args.fold_metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows, details = [], {}
    for experiment in EXPERIMENT_ORDER:
        folds = [row for row in fold_rows if row["experiment"] == experiment]
        accuracy = np.asarray([float(row["accuracy"]) for row in folds])
        macro_f1 = np.asarray([float(row["macro_f1"]) for row in folds])
        current = [row for row in prediction_rows if row["experiment"] == experiment]
        seed_reports = []
        matrices = []
        for seed in sorted({int(row["seed"]) for row in current}):
            seed_rows = [row for row in current if int(row["seed"]) == seed]
            truth = np.asarray([CLASS_NAMES.index(row["true_label"]) for row in seed_rows])
            prediction = np.asarray(
                [CLASS_NAMES.index(row["predicted_label"]) for row in seed_rows]
            )
            seed_reports.append(
                classification_report(
                    truth,
                    prediction,
                    labels=np.arange(3),
                    target_names=CLASS_NAMES,
                    output_dict=True,
                    zero_division=0,
                )
            )
            matrices.append(confusion_matrix(truth, prediction, labels=np.arange(3)))
        class_f1 = {
            class_name: float(
                np.mean([report[class_name]["f1-score"] for report in seed_reports])
            )
            for class_name in CLASS_NAMES
        }
        accuracy_ci = ci95(accuracy)
        f1_ci = ci95(macro_f1)
        summary_rows.append(
            {
                "experiment": experiment,
                "accuracy_mean": float(np.mean(accuracy)),
                "accuracy_std": float(np.std(accuracy, ddof=1)),
                "accuracy_ci95_low": accuracy_ci[0],
                "accuracy_ci95_high": accuracy_ci[1],
                "macro_f1_mean": float(np.mean(macro_f1)),
                "macro_f1_std": float(np.std(macro_f1, ddof=1)),
                "macro_f1_ci95_low": f1_ci[0],
                "macro_f1_ci95_high": f1_ci[1],
                "happy_f1": class_f1["happy"],
                "normal_f1": class_f1["normal"],
                "sad_f1": class_f1["sad"],
                "parameter_count": int(folds[0]["parameter_count"]),
                "training_seconds_mean": float(
                    np.mean([float(row["training_seconds"]) for row in folds])
                ),
                "inference_latency_ms_mean": float(
                    np.mean([float(row["inference_latency_ms_per_sample"]) for row in folds])
                ),
            }
        )
        pooled_matrix = np.sum(matrices, axis=0)
        details[experiment] = {
            "seed_classification_reports": seed_reports,
            "pooled_confusion_matrix": pooled_matrix.tolist(),
        }
        figure, axis = plt.subplots(figsize=(5.5, 4.8))
        image = axis.imshow(pooled_matrix, cmap="Blues")
        for row in range(3):
            for column in range(3):
                axis.text(column, row, int(pooled_matrix[row, column]), ha="center", va="center")
        axis.set_xticks(range(3), CLASS_NAMES)
        axis.set_yticks(range(3), CLASS_NAMES)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(experiment)
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(args.output_dir / f"{experiment}_confusion_matrix.png", dpi=180)
        plt.close(figure)

    with (args.output_dir / "ablation_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    subjects = subject_metrics(prediction_rows)
    with (args.output_dir / "subject_by_experiment.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subjects[0]))
        writer.writeheader()
        writer.writerows(subjects)
    statistics = paired_statistics(subjects)
    (args.output_dir / "ablation_details.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    (args.output_dir / "paired_subject_statistics.json").write_text(
        json.dumps(statistics, indent=2), encoding="utf-8"
    )
    print(json.dumps({"summary": summary_rows, "paired_statistics": statistics}, indent=2))


if __name__ == "__main__":
    main()
