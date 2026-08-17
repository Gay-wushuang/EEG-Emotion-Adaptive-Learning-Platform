"""Build DE and multitaper frequency variants with canonical sample identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import dpss

from eeg_emotion.features.canonical import BANDS, CanonicalFeatureConfig, filter_raw
from eeg_emotion.features.canonical_io import load_training_window


def differential_entropy(filtered: np.ndarray, cfg: CanonicalFeatureConfig) -> np.ndarray:
    segments = filtered.reshape(cfg.time_steps, cfg.samples_per_step)
    result = np.empty((cfg.time_steps, len(BANDS)), dtype=np.float32)
    for band_index, (_, low, high) in enumerate(BANDS):
        sos = butter(4, [low, high], btype="bandpass", fs=cfg.sample_rate, output="sos")
        for step, segment in enumerate(segments):
            signal = sosfiltfilt(sos, segment)
            variance = np.var(signal, ddof=1)
            result[step, band_index] = 0.5 * np.log(
                2.0 * np.pi * np.e * variance + cfg.epsilon
            )
    return result


def multitaper_log_bandpower(
    filtered: np.ndarray, cfg: CanonicalFeatureConfig
) -> np.ndarray:
    segments = filtered.reshape(cfg.time_steps, cfg.samples_per_step)
    tapers = dpss(cfg.samples_per_step, NW=3.0, Kmax=5, sym=False)
    frequencies = np.fft.rfftfreq(cfg.samples_per_step, 1.0 / cfg.sample_rate)
    result = np.empty((cfg.time_steps, len(BANDS)), dtype=np.float32)
    for step, segment in enumerate(segments):
        centered = segment - np.mean(segment)
        spectra = np.fft.rfft(tapers * centered[None, :], axis=1)
        psd = np.mean(np.abs(spectra) ** 2, axis=0) / cfg.sample_rate
        for band_index, (_, low, high) in enumerate(BANDS):
            mask = (frequencies >= low) & (frequencies < high)
            power = np.trapezoid(psd[mask], frequencies[mask])
            result[step, band_index] = np.log(power + cfg.epsilon)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("../time_data_preprocess/data"))
    parser.add_argument("--canonical-dir", type=Path, default=Path("features_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("features_v2_frequency_variants"))
    args = parser.parse_args()
    cfg = CanonicalFeatureConfig()
    manifest = pd.read_csv(args.canonical_dir / "manifest.csv")
    de_values, multitaper_values = [], []
    for row in manifest.itertuples(index=False):
        sample_dir = args.data_dir / row.sample_id
        raw, _, _, _ = load_training_window(sample_dir, int(row.crop_start_sample), cfg)
        filtered = filter_raw(raw, cfg)
        de_values.append(differential_entropy(filtered, cfg))
        multitaper_values.append(multitaper_log_bandpower(filtered, cfg))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "X_de.npy", np.stack(de_values).astype(np.float32))
    np.save(args.output_dir / "X_multitaper.npy", np.stack(multitaper_values).astype(np.float32))
    for name in ("X_filtered.npy", "y.npy", "groups.npy", "sample_ids.npy"):
        np.save(args.output_dir / name, np.load(args.canonical_dir / name))
    (args.output_dir / "contract.json").write_text(
        json.dumps(
            {
                "sample_identity": "exact features_v2 manifest order",
                "de": "0.5*log(2*pi*e*bandpassed_variance), four bands per 3s segment",
                "multitaper": "DPSS NW=3 K=5 log absolute bandpower, four bands per 3s segment",
                "bands": list(BANDS),
                "shape": [len(manifest), 10, 4],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print({"samples": len(manifest), "shape": np.stack(de_values).shape})


if __name__ == "__main__":
    main()
