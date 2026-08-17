"""Convert MindWave capture CSV into canonical unscaled sliding-window features."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eeg_emotion.features.canonical import CanonicalFeatureConfig, extract_canonical_features


def read_capture(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8-sig")))
    if not rows:
        raise ValueError("Empty capture")
    raw = np.asarray([float(row["raw_eeg"]) for row in rows], dtype=np.float32)
    sample_index = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
    if not np.array_equal(sample_index, np.arange(len(rows))):
        raise ValueError("Raw sample_index must be continuous from zero")
    output = {"raw": raw}
    indices = np.arange(len(rows), dtype=np.float64)
    for modality, column in (("att", "attention"), ("med", "meditation")):
        values = np.asarray(
            [float(row[column]) if row.get(column, "") != "" else np.nan for row in rows]
        )
        flags = np.asarray([row.get(f"{column}_updated", "") == "1" for row in rows])
        known = flags & np.isfinite(values)
        if not np.any(known):
            known = np.isfinite(values)
        if not np.any(known):
            raise ValueError(f"No {column} updates")
        output[modality] = np.interp(indices, indices[known], values[known]).astype(np.float32)
        output[f"{modality}_interpolation_ratio"] = np.asarray(
            [1.0 - float(np.mean(known))], dtype=np.float32
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    args = parser.parse_args()

    cfg = CanonicalFeatureConfig()
    data = read_capture(args.capture)
    stride = int(round(args.stride_seconds * cfg.sample_rate))
    starts = range(0, len(data["raw"]) - cfg.window_samples + 1, stride)
    features = {name: [] for name in ("filtered", "bandpower", "att", "med")}
    end_samples = []
    for start in starts:
        stop = start + cfg.window_samples
        current = extract_canonical_features(
            data["raw"][start:stop],
            data["att"][start:stop],
            data["med"][start:stop],
            cfg,
        )
        for modality, values in current.items():
            features[modality].append(values)
        end_samples.append(stop - 1)
    if not end_samples:
        raise ValueError("Capture is shorter than one canonical 30-second window")
    payload = {
        modality: np.stack(values).astype(np.float32)
        for modality, values in features.items()
    }
    payload["window_end_sample"] = np.asarray(end_samples, dtype=np.int64)
    payload["att_interpolation_ratio"] = data["att_interpolation_ratio"]
    payload["med_interpolation_ratio"] = data["med_interpolation_ratio"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print({name: value.shape for name, value in payload.items()})


if __name__ == "__main__":
    main()
