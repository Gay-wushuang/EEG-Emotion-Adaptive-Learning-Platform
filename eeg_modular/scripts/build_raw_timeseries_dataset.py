"""Build fixed-length Raw EEG arrays matching an existing canonical dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eeg_emotion.features.canonical_io import read_time_value_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=Path("features_v3_12000"))
    parser.add_argument("--output-dir", type=Path, default=Path("raw_v3_12000"))
    parser.add_argument("--window-samples", type=int, default=12000)
    args = parser.parse_args()

    sample_ids = np.load(args.reference_dir / "sample_ids.npy")
    arrays = []
    for sample_id in sample_ids:
        _, raw = read_time_value_csv(args.source_dir / str(sample_id) / "rawwave.csv")
        if len(raw) < args.window_samples:
            raise RuntimeError(f"{sample_id}: {len(raw)} < {args.window_samples}")
        start = (len(raw) - args.window_samples) // 2
        window = raw[start:start + args.window_samples]
        arrays.append(window.astype(np.float32))
    output = np.stack(arrays)[:, None, :]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "X_raw.npy", output)
    for name in ("y.npy", "groups.npy", "sample_ids.npy"):
        np.save(args.output_dir / name, np.load(args.reference_dir / name))
    metadata = {
        "shape": list(output.shape), "sample_rate": 512,
        "window_samples": args.window_samples,
        "crop": "center", "reference_dir": str(args.reference_dir),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metadata)


if __name__ == "__main__":
    main()
