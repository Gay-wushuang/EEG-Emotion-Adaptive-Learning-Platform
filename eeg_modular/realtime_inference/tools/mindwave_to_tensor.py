"""Convert a timestamped MindWave CSV into the deployed model's input tensors."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

MODALITIES = ("filtered", "powerspec", "att", "med")


def read_capture(path: Path) -> Dict[str, np.ndarray]:
    columns = {name: [] for name in ("raw_eeg", "attention", "meditation", "poor_signal")}
    update_flags = {name: [] for name in ("attention", "meditation")}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            columns["raw_eeg"].append(float(row["raw_eeg"]))
            for name in ("attention", "meditation", "poor_signal"):
                value = row.get(name, "")
                columns[name].append(float(value) if value not in ("", None) else np.nan)
            for name in update_flags:
                flag = row.get(f"{name}_updated", "")
                update_flags[name].append(flag == "1" if flag != "" else False)
    if not columns["raw_eeg"]:
        raise ValueError(f"No raw EEG rows in {path}")
    output = {name: np.asarray(values, dtype=np.float32) for name, values in columns.items()}
    indices = np.arange(output["raw_eeg"].size)
    for name in ("attention", "meditation"):
        values = output[name]
        flags = np.asarray(update_flags[name], dtype=bool)
        if np.any(flags):
            known = flags & np.isfinite(values)
            if not np.any(known):
                raise ValueError(f"No valid {name} updates in capture")
            output[name] = np.interp(indices, indices[known], values[known]).astype(np.float32)
        elif not np.any(np.isfinite(values)):
            raise ValueError(f"No valid {name} values in capture")
        else:
            known = np.isfinite(values)
            output[name] = np.interp(indices, indices[known], values[known]).astype(np.float32)
    poor = output["poor_signal"]
    if not np.any(np.isfinite(poor)):
        raise ValueError("No Poor Signal values in capture")
    known = np.isfinite(poor)
    output["poor_signal"] = np.interp(indices, indices[known], poor[known]).astype(np.float32)
    return output


def fft_bandpass(signal: np.ndarray, sample_rate: float, low: float = 1.0, high: float = 45.0) -> np.ndarray:
    centered = signal.astype(np.float64) - float(np.mean(signal))
    spectrum = np.fft.rfft(centered)
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate)
    spectrum[(frequencies < low) | (frequencies > high)] = 0
    return np.fft.irfft(spectrum, n=centered.size).astype(np.float32)


def four_stats(values: np.ndarray, time_steps: int) -> np.ndarray:
    result = np.zeros((time_steps, 4), dtype=np.float32)
    for index, segment in enumerate(np.array_split(values, time_steps)):
        if segment.size:
            result[index] = (np.mean(segment), np.std(segment), np.max(segment), np.min(segment))
    return result


def power_envelope(signal: np.ndarray, sample_rate: float, time_steps: int) -> np.ndarray:
    """Return one broadband PSD scalar per subwindow, then four temporal statistics."""
    powers = []
    for segment in np.array_split(signal, time_steps * 4):
        if segment.size < 4:
            powers.append(0.0)
            continue
        windowed = (segment - np.mean(segment)) * np.hanning(segment.size)
        spectrum = np.fft.rfft(windowed)
        frequencies = np.fft.rfftfreq(segment.size, d=1.0 / sample_rate)
        mask = (frequencies >= 1.0) & (frequencies <= 45.0)
        powers.append(float(np.mean(np.abs(spectrum[mask]) ** 2)) if np.any(mask) else 0.0)
    return four_stats(np.log1p(np.asarray(powers, dtype=np.float32)), time_steps)


def apply_saved_scalers(features: Dict[str, np.ndarray], scalers_dir: Path) -> Dict[str, np.ndarray]:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib/scikit-learn is required to apply the training scalers") from exc
    output = {}
    for modality, values in features.items():
        scaler_path = scalers_dir / f"scaler_{modality}.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError(scaler_path)
        scaler = joblib.load(scaler_path)
        if int(getattr(scaler, "n_features_in_", -1)) != 4:
            raise ValueError(f"{scaler_path} does not accept [mean,std,max,min]")
        original_shape = values.shape
        scaled = scaler.transform(values.reshape(-1, 4)).astype(np.float32)
        output[modality] = scaled.reshape(original_shape)
    return output


def window_features(data: Dict[str, np.ndarray], sample_rate: float, time_steps: int) -> Dict[str, np.ndarray]:
    filtered = fft_bandpass(data["raw_eeg"], sample_rate)
    return {
        "filtered": four_stats(filtered, time_steps),
        "powerspec": power_envelope(filtered, sample_rate, time_steps),
        "att": four_stats(data["attention"], time_steps),
        "med": four_stats(data["meditation"], time_steps),
    }


def convert(
    capture_path: Path,
    sample_rate: float = 512.0,
    window_seconds: float = 30.0,
    stride_seconds: float = 2.0,
    time_steps: int = 10,
    scalers_dir: Path | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    data = read_capture(capture_path)
    needed = round(sample_rate * window_seconds)
    if data["raw_eeg"].size < needed:
        raise ValueError(f"Need at least {needed} raw samples, got {data['raw_eeg'].size}")
    stride = round(sample_rate * stride_seconds)
    if stride <= 0:
        raise ValueError("stride_seconds must be positive")

    starts = range(0, data["raw_eeg"].size - needed + 1, stride)
    windows = {name: [] for name in MODALITIES}
    metadata = {
        "window_end_sample": [],
        "attention": [],
        "meditation": [],
        "poor_signal": [],
        "signal_good_fraction": [],
    }
    for start in starts:
        stop = start + needed
        current = {name: values[start:stop] for name, values in data.items()}
        current_features = window_features(current, sample_rate, time_steps)
        for modality in MODALITIES:
            windows[modality].append(current_features[modality])
        metadata["window_end_sample"].append(stop - 1)
        metadata["attention"].append(current["attention"][-1])
        metadata["meditation"].append(current["meditation"][-1])
        metadata["poor_signal"].append(current["poor_signal"][-1])
        metadata["signal_good_fraction"].append(float(np.mean(current["poor_signal"] < 200)))

    raw_features = {
        name: np.stack(values).astype(np.float32) for name, values in windows.items()
    }
    scaled_features = raw_features
    if scalers_dir is not None:
        scaled_features = apply_saved_scalers(raw_features, scalers_dir)
    metadata_arrays = {
        name: np.asarray(values, dtype=np.float64) for name, values in metadata.items()
    }
    return scaled_features, raw_features, metadata_arrays


def diagnostic_rows(
    raw_features: Dict[str, np.ndarray], scaled_features: Dict[str, np.ndarray]
) -> list[tuple[str, str, str, int, str]]:
    rows = []
    for modality in MODALITIES:
        raw = raw_features[modality]
        scaled = scaled_features[modality]
        non_finite = int(np.size(scaled) - np.isfinite(scaled).sum())
        state = "OK"
        if non_finite or np.std(scaled) == 0:
            state = "FAIL"
        elif np.max(np.abs(scaled)) > 10:
            state = "CHECK"
        rows.append(
            (
                modality,
                f"{np.min(raw):.4g}..{np.max(raw):.4g}",
                f"{np.min(scaled):.4g}..{np.max(scaled):.4g}",
                non_finite,
                state,
            )
        )
    return rows


def print_diagnostics(
    raw_features: Dict[str, np.ndarray], scaled_features: Dict[str, np.ndarray]
) -> None:
    print(f"{'modality':<12}{'raw range':<24}{'scaled range':<24}{'NaN/Inf':<10}status")
    for modality, raw_range, scaled_range, non_finite, state in diagnostic_rows(
        raw_features, scaled_features
    ):
        print(f"{modality:<12}{raw_range:<24}{scaled_range:<24}{non_finite:<10}{state}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=float, default=512.0)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--scalers-dir", type=Path, required=True)
    args = parser.parse_args()

    tensors, raw_features, metadata = convert(
        args.capture,
        sample_rate=args.sample_rate,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        time_steps=args.time_steps,
        scalers_dir=args.scalers_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **tensors, **metadata)
    print_diagnostics(raw_features, tensors)
    print({name: tuple(value.shape) for name, value in tensors.items()})


if __name__ == "__main__":
    main()
