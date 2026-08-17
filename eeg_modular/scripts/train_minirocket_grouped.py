"""Strict grouped three-class MiniRocket evaluation on fixed-length Raw EEG."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from aeon.classification.convolution_based import MiniRocketClassifier
from scipy.signal import resample_poly
from sklearn.metrics import accuracy_score, f1_score

from eeg_emotion.features.canonical import CanonicalFeatureConfig, filter_raw


def variants(raw: np.ndarray) -> dict[str, np.ndarray]:
    cfg = CanonicalFeatureConfig(window_seconds=raw.shape[-1] / 512.0)
    filtered_512 = np.stack([filter_raw(row[0], cfg) for row in raw])
    # The canonical passband ends at 45 Hz, so 128 Hz retains it while reducing
    # redundant temporal samples by 4x for the random convolution transform.
    filtered = resample_poly(filtered_512, up=1, down=4, axis=-1)[:, None, :]
    centered = filtered - filtered.mean(axis=-1, keepdims=True)
    scale = centered.std(axis=-1, keepdims=True)
    return {
        "filtered": filtered.astype(np.float32),
        "filtered_window_z": (centered / np.maximum(scale, 1e-6)).astype(np.float32),
    }


def make_model(kernels: int, seed: int, jobs: int) -> MiniRocketClassifier:
    return MiniRocketClassifier(
        n_kernels=kernels, class_weight="balanced", n_jobs=jobs, random_state=seed
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("raw_v3_12000"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits_v3_12000"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_v3/minirocket_v1"))
    parser.add_argument("--kernels", type=int, nargs="+", default=[2000, 5000])
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    raw = np.load(args.raw_dir / "X_raw.npy")
    labels = np.load(args.raw_dir / "y.npy")
    groups = np.load(args.raw_dir / "groups.npy")
    sample_ids = np.load(args.raw_dir / "sample_ids.npy")
    inputs = variants(raw)
    manifest = json.loads((args.splits_dir / "manifest.json").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows, prediction_rows, selection_rows = [], [], []
    for split_name in manifest["files"]:
        split = json.loads((args.splits_dir / split_name).read_text(encoding="utf-8"))
        train = np.asarray(split["train_indices"], dtype=np.int64)
        validation = np.asarray(split["val_indices"], dtype=np.int64)
        test = np.asarray(split["test_indices"], dtype=np.int64)
        seed = int(split["seed"]) * 100 + int(split["fold"])
        candidates = []
        for variant_name, values in inputs.items():
            for kernels in args.kernels:
                model = make_model(kernels, seed, args.n_jobs)
                model.fit(values[train], labels[train])
                val_prediction = model.predict(values[validation])
                score = f1_score(labels[validation], val_prediction, average="macro")
                candidates.append((score, variant_name, kernels))
                selection_rows.append({
                    "seed": split["seed"], "fold": split["fold"],
                    "variant": variant_name, "kernels": kernels, "validation_macro_f1": score,
                })
        _, selected_variant, selected_kernels = max(candidates, key=lambda item: (item[0], -item[2]))
        development = np.concatenate([train, validation])
        model = make_model(selected_kernels, seed, args.n_jobs)
        started = time.perf_counter()
        model.fit(inputs[selected_variant][development], labels[development])
        training_seconds = time.perf_counter() - started
        probabilities = model.predict_proba(inputs[selected_variant][test])
        prediction = probabilities.argmax(axis=1)
        metric_rows.append({
            "seed": split["seed"], "fold": split["fold"],
            "variant": selected_variant, "kernels": selected_kernels,
            "accuracy": accuracy_score(labels[test], prediction),
            "macro_f1": f1_score(labels[test], prediction, average="macro"),
            "training_seconds": training_seconds,
        })
        for local, index in enumerate(test):
            prediction_rows.append({
                "seed": split["seed"], "fold": split["fold"],
                "sample_id": str(sample_ids[index]), "subject_id": int(groups[index]),
                "true_label": int(labels[index]), "predicted_label": int(prediction[local]),
                "prob_happy": probabilities[local, 0], "prob_normal": probabilities[local, 1],
                "prob_sad": probabilities[local, 2],
            })
        print(f"seed={split['seed']} fold={split['fold']} {selected_variant} k={selected_kernels} acc={metric_rows[-1]['accuracy']:.4f} f1={metric_rows[-1]['macro_f1']:.4f}", flush=True)
    for filename, rows in (("fold_metrics.csv", metric_rows), ("predictions.csv", prediction_rows), ("validation_selection.csv", selection_rows)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
