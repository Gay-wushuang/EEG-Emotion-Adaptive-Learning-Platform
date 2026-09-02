"""Single Raw-to-model feature contract shared by training and deployment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.signal import butter, detrend, iirnotch, sosfiltfilt, tf2sos, welch


@dataclass(frozen=True)
class CanonicalFeatureConfig:
    sample_rate: int = 512
    window_seconds: float = 30.0
    time_steps: int = 10
    notch_hz: float = 50.0
    notch_q: float = 30.0
    low_hz: float = 0.5
    high_hz: float = 45.0
    bandpass_order: int = 4
    welch_window: str = "hann"
    welch_nperseg: int = 512
    welch_noverlap: int = 256
    welch_nfft: int = 512
    bandpower_mode: str = "relative"
    epsilon: float = 1e-12
    max_aux_missing_ratio: float = 0.20

    @property
    def window_samples(self) -> int:
        return int(round(self.sample_rate * self.window_seconds))

    @property
    def samples_per_step(self) -> int:
        if self.window_samples % self.time_steps:
            raise ValueError("window_samples must be divisible by time_steps")
        return self.window_samples // self.time_steps

    def to_dict(self) -> dict:
        return asdict(self)


BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
)


def _require_window(values: np.ndarray, cfg: CanonicalFeatureConfig, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != cfg.window_samples:
        raise ValueError(f"{name} requires {cfg.window_samples} points, got {array.size}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def filter_raw(raw_eeg: np.ndarray, cfg: CanonicalFeatureConfig) -> np.ndarray:
    """Detrend, 50 Hz notch, then 0.5-45 Hz zero-phase SOS bandpass."""
    raw = _require_window(raw_eeg, cfg, "raw_eeg")
    centered = detrend(raw, type="constant")
    notch_b, notch_a = iirnotch(cfg.notch_hz, cfg.notch_q, fs=cfg.sample_rate)
    notch_sos = tf2sos(notch_b, notch_a)
    notched = sosfiltfilt(notch_sos, centered)
    band_sos = butter(
        cfg.bandpass_order,
        [cfg.low_hz, cfg.high_hz],
        btype="bandpass",
        fs=cfg.sample_rate,
        output="sos",
    )
    return sosfiltfilt(band_sos, notched).astype(np.float32)


def segment_stats(values: np.ndarray, cfg: CanonicalFeatureConfig) -> np.ndarray:
    array = _require_window(values, cfg, "segment_stats")
    segments = array.reshape(cfg.time_steps, cfg.samples_per_step)
    return np.stack(
        (
            np.mean(segments, axis=1),
            np.std(segments, axis=1),
            np.max(segments, axis=1),
            np.min(segments, axis=1),
        ),
        axis=1,
    ).astype(np.float32)


def relative_bandpower(filtered: np.ndarray, cfg: CanonicalFeatureConfig) -> np.ndarray:
    signal = _require_window(filtered, cfg, "filtered")
    segments = signal.reshape(cfg.time_steps, cfg.samples_per_step)
    output = np.zeros((cfg.time_steps, len(BANDS)), dtype=np.float32)
    for index, segment in enumerate(segments):
        frequencies, psd = welch(
            segment,
            fs=cfg.sample_rate,
            window=cfg.welch_window,
            nperseg=cfg.welch_nperseg,
            noverlap=cfg.welch_noverlap,
            nfft=cfg.welch_nfft,
            detrend="constant",
            return_onesided=True,
            scaling="density",
        )
        total_mask = (frequencies >= cfg.low_hz) & (frequencies <= cfg.high_hz)
        total_power = np.trapezoid(psd[total_mask], frequencies[total_mask])
        for band_index, (_, low, high) in enumerate(BANDS):
            mask = (frequencies >= low) & (frequencies < high)
            power = np.trapezoid(psd[mask], frequencies[mask])
            if cfg.bandpower_mode == "relative":
                output[index, band_index] = power / (total_power + cfg.epsilon)
            elif cfg.bandpower_mode == "log":
                output[index, band_index] = np.log(power + cfg.epsilon)
            else:
                raise ValueError(f"Unknown bandpower_mode={cfg.bandpower_mode}")
    return output


def extract_canonical_features(
    raw_eeg: np.ndarray,
    attention: np.ndarray,
    meditation: np.ndarray,
    cfg: CanonicalFeatureConfig | None = None,
) -> Dict[str, np.ndarray]:
    cfg = cfg or CanonicalFeatureConfig()
    filtered_signal = filter_raw(raw_eeg, cfg)
    attention = _require_window(attention, cfg, "attention")
    meditation = _require_window(meditation, cfg, "meditation")
    return {
        "filtered": segment_stats(filtered_signal, cfg),
        "bandpower": relative_bandpower(filtered_signal, cfg),
        "att": segment_stats(attention, cfg),
        "med": segment_stats(meditation, cfg),
    }

