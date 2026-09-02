"""Canonical loading/alignment helpers for legacy MindWave CSV recordings."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .canonical import CanonicalFeatureConfig


def _clock_seconds(value: str) -> float:
    hours, minutes, seconds = value.strip().split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def read_time_value_csv(path: Path, value_column: str = "Value") -> Tuple[np.ndarray, np.ndarray]:
    times, values = [], []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        field_map = {name.strip(): name for name in (reader.fieldnames or [])}
        time_key = field_map.get("Time")
        value_key = field_map.get(value_column)
        if time_key is None or value_key is None:
            raise ValueError(f"{path} requires Time and {value_column} columns")
        for row in reader:
            times.append(_clock_seconds(row[time_key]))
            values.append(float(row[value_key]))
    if not values:
        raise ValueError(f"No values in {path}")
    time_array = np.asarray(times, dtype=np.float64)
    for index in range(1, len(time_array)):
        if time_array[index] < time_array[index - 1] - 12 * 3600:
            time_array[index:] += 24 * 3600
    return time_array, np.asarray(values, dtype=np.float32)


def interpolate_auxiliary(
    raw_start_clock: float,
    crop_start_sample: int,
    aux_times: np.ndarray,
    aux_values: np.ndarray,
    cfg: CanonicalFeatureConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    target_times = (
        raw_start_clock
        + (crop_start_sample + np.arange(cfg.window_samples, dtype=np.float64)) / cfg.sample_rate
    )
    outside = (target_times < aux_times[0]) | (target_times > aux_times[-1])
    missing_ratio = float(np.mean(outside))
    if missing_ratio > cfg.max_aux_missing_ratio:
        raise ValueError(f"Auxiliary coverage missing ratio {missing_ratio:.3f} exceeds threshold")
    interpolated = np.interp(target_times, aux_times, aux_values).astype(np.float32)
    exact_updates = np.isclose(
        target_times[:, None],
        aux_times[None, :],
        rtol=0.0,
        atol=0.5 / cfg.sample_rate,
    ).any(axis=1)
    return interpolated, {
        "coverage_missing_ratio": missing_ratio,
        "interpolation_ratio": float(1.0 - np.mean(exact_updates)),
    }


def load_training_window(
    sample_dir: Path,
    crop_start_sample: int,
    cfg: CanonicalFeatureConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    raw_times, raw = read_time_value_csv(sample_dir / "rawwave.csv")
    stop = crop_start_sample + cfg.window_samples
    if stop > raw.size:
        raise ValueError(f"Insufficient Raw EEG in {sample_dir}")
    raw_window = raw[crop_start_sample:stop]
    att_times, att_values = read_time_value_csv(sample_dir / "att.csv")
    med_times, med_values = read_time_value_csv(sample_dir / "med.csv")
    att, att_meta = interpolate_auxiliary(
        raw_times[0], crop_start_sample, att_times, att_values, cfg
    )
    med, med_meta = interpolate_auxiliary(
        raw_times[0], crop_start_sample, med_times, med_values, cfg
    )
    metadata = {
        "att_coverage_missing_ratio": att_meta["coverage_missing_ratio"],
        "att_interpolation_ratio": att_meta["interpolation_ratio"],
        "med_coverage_missing_ratio": med_meta["coverage_missing_ratio"],
        "med_interpolation_ratio": med_meta["interpolation_ratio"],
    }
    return raw_window, att, med, metadata
