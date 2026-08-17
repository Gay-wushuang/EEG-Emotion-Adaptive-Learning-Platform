"""Relate subject-level ablation performance to data-quality indicators."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "eeg_modular"
FEATURES = MODULE / "features_v2"
SUMMARY = MODULE / "outputs_v2" / "ablations" / "summary"
SOURCE = ROOT / "time_data_preprocess" / "data"


def safe_spearman(x: pd.Series, y: pd.Series) -> dict[str, float | None]:
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return {"rho": None, "p_value": None, "n": int(mask.sum())}
    result = spearmanr(x[mask], y[mask])
    return {"rho": float(result.statistic), "p_value": float(result.pvalue), "n": int(mask.sum())}


def main() -> None:
    manifest = pd.read_csv(FEATURES / "manifest.csv")
    performance = pd.read_csv(SUMMARY / "subject_by_experiment.csv")
    arrays = {
        name: np.load(FEATURES / f"X_{name}.npy")
        for name in ("filtered", "bandpower", "att", "med")
    }

    sample_rows: list[dict[str, object]] = []
    for index, row in manifest.iterrows():
        sig_path = SOURCE / row["sample_id"] / "sigqual.csv"
        sig_frame = pd.read_csv(sig_path)
        sig_frame.columns = sig_frame.columns.str.strip()
        sig = sig_frame["Value"].apply(pd.to_numeric, errors="coerce").dropna()
        record: dict[str, object] = {
            **row.to_dict(),
            "poor_signal_mean": float(sig.mean()) if len(sig) else np.nan,
            "poor_signal_bad_fraction": float((sig >= 50).mean()) if len(sig) else np.nan,
            "poor_signal_200_fraction": float((sig >= 200).mean()) if len(sig) else np.nan,
        }
        for name, values in arrays.items():
            value = values[index]
            record[f"{name}_mean"] = float(np.mean(value))
            record[f"{name}_std"] = float(np.std(value))
            record[f"{name}_abs_max"] = float(np.max(np.abs(value)))
        sample_rows.append(record)

    samples = pd.DataFrame(sample_rows)
    subject_rows: list[dict[str, object]] = []
    for subject_id, group in samples.groupby("subject_id", sort=True):
        counts = group["label"].value_counts()
        item: dict[str, object] = {
            "subject_id": subject_id,
            "window_count": int(len(group)),
            "happy_windows": int(counts.get("happy", 0)),
            "normal_windows": int(counts.get("normal", 0)),
            "sad_windows": int(counts.get("sad", 0)),
            "class_count": int(group["label"].nunique()),
        }
        numeric_columns = [
            column for column in samples.columns
            if column.endswith(("_ratio", "_mean", "_std", "_abs_max", "_fraction"))
        ]
        for column in numeric_columns:
            item[column] = float(group[column].mean())
        subject_rows.append(item)

    quality = pd.DataFrame(subject_rows)
    wide = performance.pivot(index="subject_id", columns="experiment", values=["accuracy", "macro_f1"])
    wide.columns = [f"{metric}_{experiment}" for metric, experiment in wide.columns]
    quality = quality.merge(wide.reset_index(), on="subject_id", how="left")
    quality.to_csv(SUMMARY / "subject_quality.csv", index=False, encoding="utf-8-sig")

    target = quality["accuracy_D_filtered_bandpower"]
    predictors = [
        "window_count", "class_count", "poor_signal_mean", "poor_signal_bad_fraction",
        "att_coverage_missing_ratio", "med_coverage_missing_ratio",
        "att_interpolation_ratio", "med_interpolation_ratio",
        "filtered_std", "filtered_abs_max", "bandpower_std",
        "att_std", "med_std",
    ]
    correlations = {name: safe_spearman(quality[name], target) for name in predictors}
    low = quality.nsmallest(5, "accuracy_D_filtered_bandpower")
    result = {
        "target": "subject-level accuracy of D_filtered_bandpower",
        "subject_count": int(len(quality)),
        "correlations": correlations,
        "lowest_five_subjects": low[
            [
                "subject_id", "accuracy_D_filtered_bandpower", "window_count",
                "class_count", "poor_signal_mean", "poor_signal_bad_fraction",
                "filtered_std", "att_std", "med_std",
            ]
        ].to_dict(orient="records"),
        "note": "Exploratory only; n=26 and multiple uncorrected comparisons.",
    }
    (SUMMARY / "subject_quality_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
