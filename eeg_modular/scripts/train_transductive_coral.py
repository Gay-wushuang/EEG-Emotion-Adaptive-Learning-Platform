"""Evaluate regularized, unlabeled target-to-source CORAL on immutable folds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from train_canonical_ablations import train_fold


def symmetric_power(matrix: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, 1e-6)
    return (vectors * np.power(values, power)) @ vectors.T


def align_target_to_source(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_cov = LedoitWolf().fit(source).covariance_
    target_cov = LedoitWolf().fit(target).covariance_
    transform = symmetric_power(target_cov, -0.5) @ symmetric_power(source_cov, 0.5)
    return (target - target_mean) @ transform + source_mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("features_v2"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_v2/transductive_coral_v1"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    labels = np.load(args.data_dir / "y.npy")
    groups = np.load(args.data_dir / "groups.npy")
    filtered = np.load(args.data_dir / "X_filtered.npy")
    bandpower = np.load(args.data_dir / "X_bandpower.npy")
    flat = np.concatenate([filtered.reshape(len(labels), -1), bandpower.reshape(len(labels), -1)], axis=1)
    manifest = json.loads((args.splits_dir / "manifest.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_name in manifest["files"]:
        split = json.loads((args.splits_dir / split_name).read_text(encoding="utf-8"))
        train = np.asarray(split["train_indices"], dtype=np.int64)
        validation = np.asarray(split["val_indices"], dtype=np.int64)
        test = np.asarray(split["test_indices"], dtype=np.int64)
        scaler = StandardScaler().fit(flat[train])
        transformed = scaler.transform(flat).astype(np.float64)
        source = transformed[train]
        for indices in (validation, test):
            for subject in np.unique(groups[indices]):
                subject_indices = indices[groups[indices] == subject]
                if len(subject_indices) >= 3:
                    transformed[subject_indices] = align_target_to_source(transformed[subject_indices], source)
        arrays = {
            "filtered": transformed[:, :40].reshape(-1, 10, 4).astype(np.float32),
            "bandpower": transformed[:, 40:].reshape(-1, 10, 4).astype(np.float32),
        }
        result = train_fold(
            arrays, labels, train, validation, test, ("filtered", "bandpower"),
            int(split["seed"]) * 100 + int(split["fold"]), args.epochs, args.patience, 32, device,
        )
        prediction = result[3].argmax(axis=1)
        row = {
            "seed": split["seed"], "fold": split["fold"],
            "accuracy": accuracy_score(result[2], prediction),
            "macro_f1": f1_score(result[2], prediction, average="macro"),
            "best_epoch": result[5], "training_seconds": result[7],
        }
        rows.append(row)
        print(f"seed={row['seed']} fold={row['fold']} acc={row['accuracy']:.4f} f1={row['macro_f1']:.4f}", flush=True)
    with (args.output_dir / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "methodology.json").write_text(json.dumps({
        "evaluation_type": "transductive_unlabeled_target_to_source_coral",
        "covariance": "LedoitWolf",
        "warning": "All unlabeled held-out subject windows estimate target covariance; this is not independent resting-baseline calibration."
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
