"""Train DE and multitaper variants on the immutable grouped split indices."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from train_canonical_ablations import train_fold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("features_v2_frequency_variants"))
    parser.add_argument("--reference-dir", type=Path, default=Path("features_v2"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_v2/frequency_variants"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    for name in ("y.npy", "groups.npy", "sample_ids.npy"):
        if not np.array_equal(np.load(args.data_dir / name), np.load(args.reference_dir / name)):
            raise RuntimeError(f"Canonical identity mismatch: {name}")
    labels = np.load(args.data_dir / "y.npy")
    manifest = json.loads((args.splits_dir / "manifest.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in ("de", "multitaper"):
        arrays = {
            "filtered": np.load(args.data_dir / "X_filtered.npy"),
            variant: np.load(args.data_dir / f"X_{variant}.npy"),
        }
        for split_name in manifest["files"]:
            split = json.loads((args.splits_dir / split_name).read_text(encoding="utf-8"))
            train = np.asarray(split["train_indices"], dtype=np.int64)
            validation = np.asarray(split["val_indices"], dtype=np.int64)
            test = np.asarray(split["test_indices"], dtype=np.int64)
            result = train_fold(
                arrays, labels, train, validation, test,
                ("filtered", variant), int(split["seed"]) * 100 + int(split["fold"]),
                args.epochs, args.patience, 32, device,
            )
            truth, probabilities = result[2], result[3]
            rows.append(
                {
                    "variant": variant, "seed": split["seed"], "fold": split["fold"],
                    "accuracy": accuracy_score(truth, probabilities.argmax(axis=1)),
                    "macro_f1": f1_score(truth, probabilities.argmax(axis=1), average="macro"),
                    "best_epoch": result[5], "training_seconds": result[7],
                }
            )
            print(
                f"{variant} seed={split['seed']} fold={split['fold']} "
                f"acc={rows[-1]['accuracy']:.4f} f1={rows[-1]['macro_f1']:.4f}",
                flush=True,
            )
    with (args.output_dir / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
