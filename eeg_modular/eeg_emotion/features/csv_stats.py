from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_CSV_FILES: Tuple[str, ...] = (
    "att.csv",
    "med.csv",
    "powerspec.csv",
    "combined.csv",
    "filtered.csv",
    "rawwave.csv",
    "sigqual.csv",
)


def _safe_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return df.select_dtypes(include=[np.number])


def extract_powerspec_features(df: pd.DataFrame) -> List[float]:
    numeric_df = _safe_numeric_df(df)
    stats: List[float] = []
    for col in numeric_df.columns:
        data = numeric_df[col].dropna()
        if len(data) == 0:
            stats.extend([0.0, 0.0, 0.0, 0.0])
        else:
            stats.extend([float(data.mean()), float(data.var()), float(data.max()), float(data.min())])
    return stats


def extract_features_from_df(df: pd.DataFrame, file_type: str) -> List[float]:
    """Mirror your current per-file stats strategy, but as a reusable function."""
    if file_type == "powerspec":
        return extract_powerspec_features(df)

    df = df.copy()
    df.dropna(how="all", inplace=True)

    # If a row contains NaN, treat it as end-of-stream (same as your script logic)
    nan_row_indices = df[df.isna().any(axis=1)].index
    if len(nan_row_indices) > 0:
        df = df.iloc[: int(nan_row_indices[0])]

    if df.empty:
        return []

    numeric_df = _safe_numeric_df(df)
    stats: List[float] = []

    if file_type in ("att", "med", "sigqual"):
        for col in numeric_df.columns:
            data = numeric_df[col].dropna()
            if len(data) == 0:
                stats.extend([0.0] * 5)
            else:
                stats.extend([float(data.mean()), float(data.std()), float(data.max()), float(data.min()), float(data.median())])
    elif file_type == "rawwave":
        raw = numeric_df.values.flatten()
        if len(raw) == 0:
            stats.extend([0.0, 0.0, 0.0])
        else:
            stats.append(float(np.max(raw) - np.min(raw)))
            stats.append(float(np.std(raw)))
            stats.append(float(np.mean(np.abs(np.diff(raw)))))
    else:
        for col in numeric_df.columns:
            data = numeric_df[col].dropna()
            if len(data) == 0:
                stats.extend([0.0] * 4)
            else:
                stats.extend([float(data.mean()), float(data.std()), float(data.max()), float(data.min())])
    return stats


def expected_feature_count(df: pd.DataFrame, file_type: str) -> int:
    numeric_df = _safe_numeric_df(df)
    n_cols = int(numeric_df.shape[1])
    if file_type == "powerspec":
        return n_cols * 4
    if file_type in ("att", "med", "sigqual"):
        return n_cols * 5
    if file_type == "rawwave":
        return 3
    return n_cols * 4


@dataclass
class SampleResult:
    features: List[float]
    ok: bool
    reason: str = ""


def extract_sample_features(sample_dir: str, csv_files: Sequence[str] = DEFAULT_CSV_FILES) -> SampleResult:
    """Extract a single sample's feature vector from a sample folder."""
    feature_vector: List[float] = []
    total_expected = 0

    for csv_file in csv_files:
        file_path = os.path.join(sample_dir, csv_file)
        file_type = os.path.splitext(csv_file)[0]

        if not os.path.exists(file_path):
            return SampleResult([], False, f"missing file: {csv_file}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            return SampleResult([], False, f"read failed {csv_file}: {e}")

        stats = extract_features_from_df(df, file_type)
        feature_vector.extend(stats)
        total_expected += expected_feature_count(df, file_type)

    if len(feature_vector) != total_expected:
        return SampleResult([], False, f"feature length mismatch: got {len(feature_vector)}, expected {total_expected}")

    arr = np.asarray(feature_vector, dtype=float)
    if np.isnan(arr).any():
        return SampleResult([], False, "NaN in features")

    return SampleResult(feature_vector, True, "")


def build_tabular_dataset(
    data_dir: str,
    emotions: Sequence[str],
    csv_files: Sequence[str] = DEFAULT_CSV_FILES,
    sample_prefix: str = "sample",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build (X, y) from a folder layout like:
       data_dir/<emotion>/sampleXXX/*.csv
    """
    features: List[List[float]] = []
    labels: List[int] = []
    skipped: List[str] = []

    for label, emotion in enumerate(emotions):
        emotion_path = os.path.join(data_dir, emotion)
        if not os.path.isdir(emotion_path):
            skipped.append(f"{emotion_path} (missing emotion folder)")
            continue

        for name in os.listdir(emotion_path):
            sample_path = os.path.join(emotion_path, name)
            if not (os.path.isdir(sample_path) and name.startswith(sample_prefix)):
                continue

            res = extract_sample_features(sample_path, csv_files=csv_files)
            if res.ok:
                features.append(res.features)
                labels.append(label)
            else:
                skipped.append(f"{sample_path} ({res.reason})")

    if len(features) == 0:
        raise RuntimeError("No valid samples found. Check your data_dir and CSV files.")

    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    return X, y, skipped
