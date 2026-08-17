"""Create immutable subject-grouped train/validation/test split files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def dataset_fingerprint(data_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("y.npy", "groups.npy", "sample_ids.npy", "canonical_config.json"):
        digest.update(name.encode())
        digest.update((data_dir / name).read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("features_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("splits_v2"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    labels = np.load(args.data_dir / "y.npy")
    groups = np.load(args.data_dir / "groups.npy")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = dataset_fingerprint(args.data_dir)
    files = []

    for seed in args.seeds:
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold_index, (development, test) in enumerate(
            outer.split(np.zeros(len(labels)), labels, groups), start=1
        ):
            inner = StratifiedGroupKFold(
                n_splits=4, shuffle=True, random_state=seed + fold_index - 1
            )
            local_train, local_val = next(
                inner.split(
                    np.zeros(len(development)),
                    labels[development],
                    groups[development],
                )
            )
            train = development[local_train]
            val = development[local_val]
            split = {
                "dataset_sha256": fingerprint,
                "seed": seed,
                "fold": fold_index,
                "train_indices": train.tolist(),
                "val_indices": val.tolist(),
                "test_indices": test.tolist(),
                "train_subjects": sorted(map(int, set(groups[train]))),
                "val_subjects": sorted(map(int, set(groups[val]))),
                "test_subjects": sorted(map(int, set(groups[test]))),
                "train_class_counts": np.bincount(labels[train], minlength=3).tolist(),
                "val_class_counts": np.bincount(labels[val], minlength=3).tolist(),
                "test_class_counts": np.bincount(labels[test], minlength=3).tolist(),
            }
            subject_sets = [
                set(split["train_subjects"]),
                set(split["val_subjects"]),
                set(split["test_subjects"]),
            ]
            if any(subject_sets[i] & subject_sets[j] for i in range(3) for j in range(i + 1, 3)):
                raise RuntimeError("Subject leakage detected")
            path = args.output_dir / f"seed_{seed}_fold_{fold_index}.json"
            path.write_text(json.dumps(split, indent=2), encoding="utf-8")
            files.append(path.name)

    manifest = {
        "dataset_sha256": fingerprint,
        "seeds": args.seeds,
        "folds_per_seed": 5,
        "files": files,
        "contract": "All ablations must reuse these exact indices.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(manifest)


if __name__ == "__main__":
    main()
