"""Frozen v1 signal-quality metrics, independent of emotion-model outputs."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis


QUALITY_METRIC_VERSION = "signal_quality_v1"
QUALITY_FEATURES = (
    "poor_signal_mean",
    "poor_signal_bad_fraction",
    "poor_signal_change_count",
    "raw_rms",
    "raw_peak_to_peak",
    "line_50hz_energy_ratio",
    "low_0_5_4hz_energy_ratio",
    "high_20_45hz_energy_ratio",
    "first_difference_rms",
    "raw_kurtosis",
    "flatline_fraction",
    "max_consecutive_jump",
    "abnormal_peak_count",
)


def _band_integral(frequencies, psd, low, high):
    mask = (frequencies >= low) & (frequencies <= high)
    return float(np.trapezoid(psd[mask], frequencies[mask]))


def compute_quality_metrics(
    raw: np.ndarray, poor_signal: np.ndarray | None = None, sample_rate: int = 512
) -> dict[str, float]:
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 1 or len(raw) == 0:
        raise ValueError("raw must be a non-empty 1D array")
    centered = raw - np.mean(raw)
    difference = np.diff(raw)
    frequencies, psd = welch(
        centered,
        fs=sample_rate,
        window="hann",
        nperseg=min(1024, len(raw)),
        noverlap=min(512, max(0, len(raw) // 2)),
        nfft=1024,
    )
    total_0_5_45 = _band_integral(frequencies, psd, 0.5, 45)
    total_0_5_60 = _band_integral(frequencies, psd, 0.5, 60)
    median = np.median(centered)
    robust_sigma = 1.4826 * np.median(np.abs(centered - median))
    if poor_signal is None:
        poor = np.zeros(len(raw), dtype=np.float64)
    else:
        poor = np.asarray(poor_signal, dtype=np.float64)
        if poor.shape != raw.shape:
            raise ValueError("poor_signal must match raw shape")
    return {
        "poor_signal_mean": float(np.mean(poor)),
        "poor_signal_bad_fraction": float(np.mean(poor >= 50)),
        "poor_signal_change_count": float(np.count_nonzero(np.diff(poor))),
        "raw_rms": float(np.sqrt(np.mean(centered ** 2))),
        "raw_peak_to_peak": float(np.ptp(raw)),
        "line_50hz_energy_ratio": _band_integral(frequencies, psd, 49, 51)
        / max(total_0_5_60, 1e-12),
        "low_0_5_4hz_energy_ratio": _band_integral(frequencies, psd, 0.5, 4)
        / max(total_0_5_45, 1e-12),
        "high_20_45hz_energy_ratio": _band_integral(frequencies, psd, 20, 45)
        / max(total_0_5_45, 1e-12),
        "first_difference_rms": float(np.sqrt(np.mean(difference ** 2))),
        "raw_kurtosis": float(kurtosis(centered, fisher=False, bias=False)),
        "flatline_fraction": float(np.mean(difference == 0)),
        "max_consecutive_jump": float(np.max(np.abs(difference))),
        "abnormal_peak_count": float(
            np.sum(np.abs(centered - median) > max(6 * robust_sigma, 1e-12))
        ),
    }

