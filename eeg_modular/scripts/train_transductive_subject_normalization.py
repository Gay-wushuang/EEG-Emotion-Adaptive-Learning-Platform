"""Evaluate unlabeled subject-wise normalization on the immutable grouped folds.

This is a transductive diagnostic: statistics for each held-out subject use all of
that subject's unlabeled windows.  It is not a baseline-only calibration claim.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from train_canonical_ablations import train_fold


def subject_normalize(values: np.ndarray, groups: np.ndarray, robust: bool) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float32)
    for subject in np.unique(groups):
        mask = groups == subject
        block = values[mask]
        flattened = block.reshape(-1, block.shape[-1])
        if robust:
            center = np.median(flattened, axis=0)
            scale = 1.4826 * np.median(np.abs(flattened - center), axis=0)
        else:
            center = flattened.mean(axis=0)
            scale = flattened.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        output[mask] = ((block - center) / scale).astype(np.float32)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("features_v2"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_v2/transductive_subject_norm_v1"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    labels = np.load(args.data_dir / "y.npy")
    groups = np.load(args.data_dir / "groups.npy")
    originals = {
        "filtered": np.load(args.data_dir / "X_filtered.npy"),
        "bandpower": np.load(args.data_dir / "X_bandpower.npy"),
    }
    manifest = json.loads((args.splits_dir / "manifest.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, robust in (("subject_z", False), ("subject_robust", True)):
        arrays = {name: subject_normalize(x, groups, robust) for name, x in originals.items()}
        for split_name in manifest["files"]:
            split = json.loads((args.splits_dir / split_name).read_text(encoding="utf-8"))
            train = np.asarray(split["train_indices"], dtype=np.int64)
            validation = np.asarray(split["val_indices"], dtype=np.int64)
            test = np.asarray(split["test_indices"], dtype=np.int64)
            result = train_fold(
                arrays, labels, train, validation, test,
                ("filtered", "bandpower"), int(split["seed"]) * 100 + int(split["fold"]),
                args.epochs, args.patience, 32, device,
            )
            prediction = result[3].argmax(axis=1)
            row = {
                "method": method, "seed": split["seed"], "fold": split["fold"],
                "accuracy": accuracy_score(result[2], prediction),
                "macro_f1": f1_score(result[2], prediction, average="macro"),
                "best_epoch": result[5], "training_seconds": result[7],
            }
            rows.append(row)
            print(f"{method} seed={row['seed']} fold={row['fold']} acc={row['accuracy']:.4f} f1={row['macro_f1']:.4f}", flush=True)
    with (args.output_dir / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metadata = {
        "evaluation_type": "transductive_unlabeled_subject_normalization",
        "warning": "Held-out subject statistics use all unlabeled target windows; not equivalent to 60-90 s independent resting-baseline calibration.",
        "modalities": ["filtered", "bandpower"],
    }
    (args.output_dir / "methodology.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
