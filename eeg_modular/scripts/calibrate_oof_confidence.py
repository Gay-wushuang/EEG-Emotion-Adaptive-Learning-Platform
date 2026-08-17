"""Cross-fit temperature scaling and selective-risk analysis for pure-EEG OOF predictions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score, log_loss


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs_v2" / "ablations" / "predictions.csv"
OUTPUT = ROOT / "outputs_v2" / "confidence_calibration_v1"
CLASSES = ("happy", "normal", "sad")
THRESHOLDS = np.arange(0.40, 0.651, 0.05)


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    objective = lambda log_t: log_loss(  # noqa: E731
        labels, temperature_scale(probabilities, float(np.exp(log_t))), labels=np.arange(3)
    )
    result = minimize_scalar(objective, bounds=(np.log(0.05), np.log(10.0)), method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature fit failed: {result.message}")
    return float(np.exp(result.x))


def brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    one_hot = np.eye(3)[labels]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def reliability(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10):
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows, ece = [], 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        count = int(mask.sum())
        accuracy = float(np.mean(prediction[mask] == labels[mask])) if count else np.nan
        mean_confidence = float(np.mean(confidence[mask])) if count else np.nan
        if count:
            ece += count / len(labels) * abs(accuracy - mean_confidence)
        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
            }
        )
    return rows, float(ece)


def metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    _, ece = reliability(probabilities, labels)
    return {
        "nll": float(log_loss(labels, probabilities, labels=np.arange(3))),
        "brier_score": brier(probabilities, labels),
        "ece_10_bins": ece,
        "accuracy": float(accuracy_score(labels, probabilities.argmax(axis=1))),
        "macro_f1": float(f1_score(labels, probabilities.argmax(axis=1), average="macro")),
    }


def threshold_rows(probabilities: np.ndarray, labels: np.ndarray):
    confidence, prediction = probabilities.max(axis=1), probabilities.argmax(axis=1)
    rows = []
    for threshold in THRESHOLDS:
        accepted = confidence >= threshold
        count = int(accepted.sum())
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "accepted_count": count,
                "coverage": float(np.mean(accepted)),
                "accepted_accuracy": (
                    float(accuracy_score(labels[accepted], prediction[accepted])) if count else np.nan
                ),
                "accepted_macro_f1": (
                    float(f1_score(labels[accepted], prediction[accepted], average="macro"))
                    if count else np.nan
                ),
                "error_rate": (
                    float(1 - accuracy_score(labels[accepted], prediction[accepted]))
                    if count else np.nan
                ),
            }
        )
    return rows


def risk_coverage(probabilities: np.ndarray, labels: np.ndarray):
    confidence, prediction = probabilities.max(axis=1), probabilities.argmax(axis=1)
    order = np.argsort(-confidence)
    correct = (prediction[order] == labels[order]).astype(float)
    coverage = np.arange(1, len(labels) + 1) / len(labels)
    risk = 1.0 - np.cumsum(correct) / np.arange(1, len(labels) + 1)
    return pd.DataFrame(
        {
            "accepted_count": np.arange(1, len(labels) + 1),
            "coverage": coverage,
            "risk": risk,
            "minimum_confidence": confidence[order],
        }
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE)
    frame = frame[frame["experiment"] == "D_filtered_bandpower"].copy()
    if len(frame) != 831:
        raise RuntimeError(f"Expected 831 OOF prediction rows, got {len(frame)}")
    label_map = {name: index for index, name in enumerate(CLASSES)}
    frame["label_index"] = frame["true_label"].map(label_map)
    probability_columns = ["prob_happy", "prob_normal", "prob_sad"]
    raw = frame[probability_columns].to_numpy()

    calibrated = np.empty_like(raw)
    temperature_records = []
    for seed in sorted(frame["seed"].unique()):
        for fold in sorted(frame["fold"].unique()):
            evaluate = (frame["seed"] == seed) & (frame["fold"] == fold)
            calibrate = (frame["seed"] == seed) & (frame["fold"] != fold)
            train_subjects = set(frame.loc[calibrate, "subject_id"])
            test_subjects = set(frame.loc[evaluate, "subject_id"])
            if train_subjects & test_subjects:
                raise RuntimeError("Subject leakage in calibration split")
            temperature = fit_temperature(raw[calibrate], frame.loc[calibrate, "label_index"].to_numpy())
            calibrated[evaluate] = temperature_scale(raw[evaluate], temperature)
            temperature_records.append(
                {
                    "seed": int(seed),
                    "evaluated_fold": int(fold),
                    "temperature": temperature,
                    "calibration_windows": int(calibrate.sum()),
                    "evaluation_windows": int(evaluate.sum()),
                }
            )
    for index, name in enumerate(CLASSES):
        frame[f"calibrated_prob_{name}"] = calibrated[:, index]
    frame.to_csv(OUTPUT / "crossfit_oof_predictions_831.csv", index=False, encoding="utf-8-sig")

    aggregate = (
        frame.groupby(["sample_id", "subject_id", "true_label", "label_index"], as_index=False)
        [[*probability_columns, *[f"calibrated_prob_{name}" for name in CLASSES]]]
        .mean()
    )
    aggregate_raw = aggregate[probability_columns].to_numpy()
    aggregate_calibrated = aggregate[
        [f"calibrated_prob_{name}" for name in CLASSES]
    ].to_numpy()
    aggregate_raw /= aggregate_raw.sum(axis=1, keepdims=True)
    aggregate_calibrated /= aggregate_calibrated.sum(axis=1, keepdims=True)
    labels = aggregate["label_index"].to_numpy(dtype=int)
    aggregate["raw_predicted"] = [CLASSES[index] for index in aggregate_raw.argmax(axis=1)]
    aggregate["calibrated_predicted"] = [
        CLASSES[index] for index in aggregate_calibrated.argmax(axis=1)
    ]
    aggregate["calibrated_confidence"] = aggregate_calibrated.max(axis=1)
    aggregate.to_csv(OUTPUT / "oof_predictions_277.csv", index=False, encoding="utf-8-sig")

    raw_rel, _ = reliability(aggregate_raw, labels)
    cal_rel, _ = reliability(aggregate_calibrated, labels)
    pd.DataFrame(raw_rel).to_csv(OUTPUT / "reliability_raw.csv", index=False)
    pd.DataFrame(cal_rel).to_csv(OUTPUT / "reliability_calibrated.csv", index=False)
    threshold_raw_ensemble = pd.DataFrame(threshold_rows(aggregate_raw, labels))
    threshold_calibrated_ensemble = pd.DataFrame(
        threshold_rows(aggregate_calibrated, labels)
    )
    threshold_raw_ensemble.to_csv(
        OUTPUT / "threshold_operating_points_raw_ensemble.csv", index=False
    )
    threshold_calibrated_ensemble.to_csv(
        OUTPUT / "threshold_operating_points_calibrated_ensemble.csv", index=False
    )
    row_labels = frame["label_index"].to_numpy(dtype=int)
    single_model_thresholds = []
    per_seed_metrics = {}
    for seed in sorted(frame["seed"].unique()):
        selected = frame["seed"] == seed
        seed_raw = raw[selected]
        seed_calibrated = calibrated[selected]
        seed_labels = row_labels[selected]
        per_seed_metrics[str(seed)] = {
            "raw": metrics(seed_raw, seed_labels),
            "crossfit_temperature": metrics(seed_calibrated, seed_labels),
        }
        for calibration, probabilities in (
            ("raw", seed_raw),
            ("crossfit_temperature", seed_calibrated),
        ):
            for row in threshold_rows(probabilities, seed_labels):
                single_model_thresholds.append(
                    {"seed": int(seed), "calibration": calibration, **row}
                )
    single_model_thresholds = pd.DataFrame(single_model_thresholds)
    single_model_thresholds.to_csv(
        OUTPUT / "threshold_operating_points_by_seed.csv", index=False
    )
    single_model_summary = (
        single_model_thresholds.groupby(["calibration", "threshold"], as_index=False)
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_min=("coverage", "min"),
            accepted_accuracy_mean=("accepted_accuracy", "mean"),
            accepted_accuracy_min=("accepted_accuracy", "min"),
            accepted_macro_f1_mean=("accepted_macro_f1", "mean"),
            error_rate_mean=("error_rate", "mean"),
        )
    )
    single_model_summary.to_csv(
        OUTPUT / "threshold_operating_points_single_model_summary.csv", index=False
    )
    risk = risk_coverage(aggregate_calibrated, labels)
    risk.to_csv(OUTPUT / "risk_coverage.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="ideal")
    for rows, label in ((raw_rel, "raw"), (cal_rel, "cross-fit temperature")):
        valid = [row for row in rows if row["count"]]
        axes[0].plot(
            [row["mean_confidence"] for row in valid],
            [row["accuracy"] for row in valid],
            marker="o",
            label=label,
        )
    axes[0].set(xlabel="Confidence", ylabel="Accuracy", title="Reliability")
    axes[0].legend()
    axes[1].plot(risk["coverage"], risk["risk"])
    axes[1].set(xlabel="Coverage", ylabel="Selective risk", title="Risk–Coverage")
    fig.tight_layout()
    fig.savefig(OUTPUT / "calibration_and_risk_coverage.png", dpi=180)
    plt.close(fig)

    report = {
        "experiment": "D_filtered_bandpower",
        "crossfit_method": "within each seed, fit temperature on the other four subject-disjoint folds",
        "aggregation": "average three cross-fit calibrated OOF probabilities per canonical window",
        "unique_windows": len(aggregate),
        "unique_subjects": int(aggregate["subject_id"].nunique()),
        "raw_metrics": metrics(aggregate_raw, labels),
        "crossfit_temperature_metrics": metrics(aggregate_calibrated, labels),
        "per_seed_single_model_metrics": per_seed_metrics,
        "temperatures": temperature_records,
        "single_model_threshold_summary": single_model_summary.to_dict(orient="records"),
        "raw_ensemble_threshold_operating_points": threshold_raw_ensemble.to_dict(orient="records"),
        "calibrated_ensemble_threshold_operating_points": threshold_calibrated_ensemble.to_dict(
            orient="records"
        ),
    }
    (OUTPUT / "calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
