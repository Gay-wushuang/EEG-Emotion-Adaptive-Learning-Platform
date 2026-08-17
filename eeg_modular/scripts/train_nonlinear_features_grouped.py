"""Three-class grouped evaluation using richer time/frequency/nonlinear EEG features."""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score

from eeg_emotion.features.canonical import CanonicalFeatureConfig, filter_raw

warnings.filterwarnings("ignore", category=UserWarning)

BANDS = ((0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45))


def segment_features(segment: np.ndarray) -> np.ndarray:
    diff = np.diff(segment)
    diff2 = np.diff(diff)
    variance = np.var(segment) + 1e-12
    diff_variance = np.var(diff) + 1e-12
    mobility = np.sqrt(diff_variance / variance)
    complexity = np.sqrt((np.var(diff2) + 1e-12) / diff_variance) / max(mobility, 1e-12)
    frequencies, psd = welch(segment, fs=512, nperseg=512, noverlap=256, nfft=512)
    useful = (frequencies >= 0.5) & (frequencies <= 45)
    normalized = psd[useful] / (psd[useful].sum() + 1e-12)
    spectral_entropy = -np.sum(normalized * np.log(normalized + 1e-12)) / np.log(len(normalized))
    powers = []
    for low, high in BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        powers.append(np.trapezoid(psd[mask], frequencies[mask]) + 1e-12)
    powers = np.asarray(powers)
    relative = powers / powers.sum()
    ratios = np.asarray([
        powers[1] / powers[3], powers[2] / powers[3],
        (powers[1] + powers[2]) / (powers[3] + powers[4]),
    ])
    base = np.asarray([
        np.mean(segment), np.std(segment), np.min(segment), np.max(segment),
        np.ptp(segment), skew(segment), kurtosis(segment),
        np.mean(np.abs(diff)), np.sqrt(np.mean(diff ** 2)),
        np.mean(np.signbit(segment[:-1]) != np.signbit(segment[1:])),
        mobility, complexity, spectral_entropy,
        np.quantile(segment, 0.75) - np.quantile(segment, 0.25),
        np.median(np.abs(segment - np.median(segment))) * 1.4826,
    ])
    return np.concatenate([base, np.log(powers), relative, np.log(ratios + 1e-12)])


def extract(raw: np.ndarray) -> np.ndarray:
    cfg = CanonicalFeatureConfig(window_seconds=raw.shape[-1] / 512.0)
    rows = []
    for item in raw:
        filtered = filter_raw(item[0], cfg).reshape(10, -1)
        rows.append(np.concatenate([segment_features(segment) for segment in filtered]))
    return np.asarray(rows, dtype=np.float32)


def candidates(seed: int):
    yield "xgboost", xgb.XGBClassifier(
        n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=3.0, objective="multi:softprob",
        eval_metric="mlogloss", random_state=seed, n_jobs=-1,
    )
    yield "xgboost_depth3", xgb.XGBClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=5.0, objective="multi:softprob",
        eval_metric="mlogloss", random_state=seed, n_jobs=-1,
    )
    yield "lightgbm", lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=7, max_depth=3,
        min_child_samples=12, reg_lambda=3.0, subsample=0.8,
        colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbosity=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("raw_v3_12000"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits_v3_12000"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_v3/nonlinear_tree_v1"))
    args = parser.parse_args()
    raw = np.load(args.raw_dir / "X_raw.npy")
    labels = np.load(args.raw_dir / "y.npy")
    groups = np.load(args.raw_dir / "groups.npy")
    sample_ids = np.load(args.raw_dir / "sample_ids.npy")
    features = extract(raw)
    manifest = json.loads((args.splits_dir / "manifest.json").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "X_nonlinear.npy", features)
    metrics, predictions, selections = [], [], []
    for split_name in manifest["files"]:
        split = json.loads((args.splits_dir / split_name).read_text(encoding="utf-8"))
        train = np.asarray(split["train_indices"], dtype=int)
        validation = np.asarray(split["val_indices"], dtype=int)
        test = np.asarray(split["test_indices"], dtype=int)
        seed = int(split["seed"]) * 100 + int(split["fold"])
        options = []
        for k in (40, 80, "all"):
            selector = SelectKBest(f_classif, k=k).fit(features[train], labels[train])
            train_x = selector.transform(features[train])
            val_x = selector.transform(features[validation])
            for name, model in candidates(seed):
                model.fit(train_x, labels[train])
                score = f1_score(labels[validation], model.predict(val_x), average="macro")
                options.append((score, str(k), name))
                selections.append({"seed": split["seed"], "fold": split["fold"], "k": k, "model": name, "validation_macro_f1": score})
        _, selected_k, selected_name = max(options, key=lambda item: item[0])
        selected_k_value = "all" if selected_k == "all" else int(selected_k)
        development = np.concatenate([train, validation])
        selector = SelectKBest(f_classif, k=selected_k_value).fit(features[development], labels[development])
        model = dict(candidates(seed))[selected_name]
        model.fit(selector.transform(features[development]), labels[development])
        probabilities = model.predict_proba(selector.transform(features[test]))
        predicted = probabilities.argmax(axis=1)
        metrics.append({
            "seed": split["seed"], "fold": split["fold"], "k": selected_k,
            "model": selected_name, "accuracy": accuracy_score(labels[test], predicted),
            "macro_f1": f1_score(labels[test], predicted, average="macro"),
        })
        for local, index in enumerate(test):
            predictions.append({
                "seed": split["seed"], "fold": split["fold"], "sample_id": str(sample_ids[index]),
                "subject_id": int(groups[index]), "true_label": int(labels[index]),
                "predicted_label": int(predicted[local]), "prob_happy": probabilities[local, 0],
                "prob_normal": probabilities[local, 1], "prob_sad": probabilities[local, 2],
            })
        print(f"seed={split['seed']} fold={split['fold']} {selected_name} k={selected_k} acc={metrics[-1]['accuracy']:.4f} f1={metrics[-1]['macro_f1']:.4f}", flush=True)
    for filename, rows in (("fold_metrics.csv", metrics), ("predictions.csv", predictions), ("validation_selection.csv", selections)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
