"""Report sampling, timing, update, missing-value, and signal-quality health."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--expected-rate", type=float, default=512.0)
    args = parser.parse_args()

    with args.capture.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Capture contains no samples")

    timestamps = np.asarray([float(row["timestamp_unix"]) for row in rows])
    raw = np.asarray([float(row["raw_eeg"]) for row in rows])
    duration = timestamps[-1] - timestamps[0] + 1.0 / args.expected_rate
    receive_times = np.asarray(
        [float(row["packet_received_unix"]) for row in rows if row.get("packet_received_unix")]
    )
    receive_duration = (
        receive_times[-1] - receive_times[0] if receive_times.size > 1 else float("nan")
    )
    intervals = np.diff(timestamps)

    def values(name: str) -> np.ndarray:
        return np.asarray(
            [float(row[name]) if row.get(name, "") != "" else np.nan for row in rows]
        )

    print(f"samples: {len(rows)}")
    print(f"duration_seconds: {duration:.3f}")
    print(f"nominal_reconstructed_sample_rate_hz: {len(rows) / duration:.3f}")
    if np.isfinite(receive_duration) and receive_duration > 0:
        print(f"receive_span_seconds: {receive_duration:.3f}")
        print(f"observed_receive_rate_hz: {(len(rows) - 1) / receive_duration:.3f}")
    print(f"timestamp_nonpositive_steps: {int(np.sum(intervals <= 0))}")
    print(f"timestamp_gaps_over_1.5x: {int(np.sum(intervals > 1.5 / args.expected_rate))}")
    print(f"raw_missing: {int(np.sum(~np.isfinite(raw)))}")
    print(f"raw_consecutive_duplicates: {int(np.sum(np.diff(raw) == 0))}")
    print(f"raw_std: {np.std(raw):.6g}")

    for name in ("attention", "meditation", "poor_signal"):
        array = values(name)
        finite = array[np.isfinite(array)]
        changes = int(np.sum(np.diff(finite) != 0)) if finite.size > 1 else 0
        print(
            f"{name}: missing={int(np.sum(~np.isfinite(array)))}, "
            f"updates_or_changes={changes}, "
            f"range={np.min(finite) if finite.size else 'NA'}.."
            f"{np.max(finite) if finite.size else 'NA'}"
        )

    poor = values("poor_signal")
    finite_poor = poor[np.isfinite(poor)]
    if finite_poor.size:
        print(f"poor_signal_good_fraction_lt_200: {np.mean(finite_poor < 200):.4f}")
        print(f"poor_signal_strict_good_fraction_le_50: {np.mean(finite_poor <= 50):.4f}")


if __name__ == "__main__":
    main()
