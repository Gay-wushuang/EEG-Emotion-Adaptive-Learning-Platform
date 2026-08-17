"""Build the v2 dataset exclusively from Raw EEG, ATT, and MED."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from eeg_emotion.features.canonical import CanonicalFeatureConfig, extract_canonical_features
from eeg_emotion.features.canonical_io import load_training_window, read_time_value_csv

MODALITIES = ("filtered", "bandpower", "att", "med")


def sample_number(path: Path) -> int:
    match = re.fullmatch(r"sample(\d+)", path.name)
    if not match:
        raise ValueError(f"Unexpected sample directory: {path}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("../time_data_preprocess/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("features_v2"))
    parser.add_argument(
        "--window-samples", type=int, default=15360,
        help="Canonical window length. Must be divisible by 10; 15360 preserves the v2 contract.",
    )
    args = parser.parse_args()

    if args.window_samples <= 0 or args.window_samples % 10:
        raise ValueError("--window-samples must be positive and divisible by 10")
    cfg = CanonicalFeatureConfig(window_seconds=args.window_samples / 512.0)
    arrays = {name: [] for name in MODALITIES}
    labels, groups, sample_ids, manifest = [], [], [], []
    excluded = []

    class_names = ["happy", "normal", "sad"]
    for label_index, label in enumerate(class_names):
        label_dir = args.data_dir / label
        sample_dirs = sorted(label_dir.glob("sample*"), key=sample_number)
        for sample_dir in sample_dirs:
            number = sample_number(sample_dir)
            if number >= 131:
                excluded.append(
                    {
                        "sample_id": f"{label}/sample{number}",
                        "reason": "subject_mapping_ambiguous_for_sample_131_to_135",
                        "raw_samples": None,
                    }
                )
                continue
            subject_id = (number - 1) // 5
            _, raw = read_time_value_csv(sample_dir / "rawwave.csv")
            if raw.size < cfg.window_samples:
                excluded.append(
                    {
                        "sample_id": f"{label}/sample{number}",
                        "reason": "raw_shorter_than_30s",
                        "raw_samples": int(raw.size),
                    }
                )
                continue
            crop_start = (raw.size - cfg.window_samples) // 2
            try:
                raw_window, att, med, aux_meta = load_training_window(
                    sample_dir, crop_start, cfg
                )
                features = extract_canonical_features(raw_window, att, med, cfg)
            except Exception as exc:
                excluded.append(
                    {
                        "sample_id": f"{label}/sample{number}",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "raw_samples": int(raw.size),
                    }
                )
                continue
            for modality in MODALITIES:
                arrays[modality].append(features[modality])
            labels.append(label_index)
            groups.append(subject_id)
            sample_ids.append(f"{label}/sample{number}")
            manifest.append(
                {
                    "sample_id": sample_ids[-1],
                    "subject_id": f"subject_{subject_id + 1:03d}",
                    "label": label,
                    "raw_samples": int(raw.size),
                    "crop_start_sample": crop_start,
                    **aux_meta,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for modality in MODALITIES:
        np.save(
            args.output_dir / f"X_{modality}.npy",
            np.stack(arrays[modality]).astype(np.float32),
        )
    np.save(args.output_dir / "y.npy", np.asarray(labels, dtype=np.int64))
    np.save(args.output_dir / "groups.npy", np.asarray(groups, dtype=np.int64))
    np.save(args.output_dir / "sample_ids.npy", np.asarray(sample_ids))
    (args.output_dir / "class_names.json").write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "canonical_config.json").write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    with (args.output_dir / "excluded.json").open("w", encoding="utf-8") as handle:
        json.dump(excluded, handle, ensure_ascii=False, indent=2)

    print(
        {
            "included": len(labels),
            "excluded": len(excluded),
            "class_counts": dict(Counter(class_names[index] for index in labels)),
            "subjects": len(set(groups)),
            "shapes": {name: np.stack(arrays[name]).shape for name in MODALITIES},
        }
    )


if __name__ == "__main__":
    main()
