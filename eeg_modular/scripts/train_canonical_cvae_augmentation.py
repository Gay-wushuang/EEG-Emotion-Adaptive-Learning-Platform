"""Evaluate train-fold-only CVAE augmentation on immutable grouped splits."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

from create_grouped_splits import dataset_fingerprint
from train_canonical_ablations import (
    ArrayDataset,
    EqualBranchCNN,
    fit_and_transform,
    predict,
    set_seed,
)

from eeg_emotion.models.torch.cvae_model import CVAE


MODALITIES = ("filtered", "bandpower")


def train_cvae(
    flattened_train: np.ndarray,
    labels_train: np.ndarray,
    run_seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> CVAE:
    set_seed(run_seed)
    model = CVAE(
        input_dim=flattened_train.shape[1],
        num_classes=3,
        latent_dim=32,
        hidden_dim=128,
    ).to(device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(flattened_train).float(),
            torch.from_numpy(labels_train).long(),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for values, targets in loader:
            values, targets = values.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mean, log_variance = model(values, targets)
            loss, _, _ = model.loss_function(
                reconstruction, values, mean, log_variance, beta=0.1
            )
            loss.backward()
            optimizer.step()
    return model


def generate_train_only(
    model: CVAE,
    flattened_train: np.ndarray,
    labels_train: np.ndarray,
    ratio: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.concatenate(
        [
            np.full(max(1, round((labels_train == label).sum() * ratio)), label)
            for label in range(3)
        ]
    ).astype(np.int64)
    generated = model.generate(torch.from_numpy(labels).to(device), len(labels))
    lower = np.quantile(flattened_train, 0.01, axis=0)
    upper = np.quantile(flattened_train, 0.99, axis=0)
    return np.clip(generated, lower, upper).astype(np.float32), labels


def augmented_arrays(
    generated: np.ndarray,
) -> dict[str, np.ndarray]:
    split = 10 * 4
    return {
        "filtered": generated[:, :split].reshape(-1, 10, 4),
        "bandpower": generated[:, split:].reshape(-1, 10, 4),
    }


def train_classifier(
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    synthetic: dict[str, np.ndarray],
    synthetic_labels: np.ndarray,
    run_seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
):
    set_seed(run_seed)
    train_arrays = {
        modality: np.concatenate([arrays[modality][train_indices], synthetic[modality]])
        for modality in MODALITIES
    }
    train_labels = np.concatenate([labels[train_indices], synthetic_labels])
    loaders = {
        "train": DataLoader(
            ArrayDataset(train_arrays, train_labels),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        ),
        "val": DataLoader(
            ArrayDataset(
                {m: arrays[m][val_indices] for m in MODALITIES}, labels[val_indices]
            ),
            batch_size=batch_size,
            shuffle=False,
        ),
        "test": DataLoader(
            ArrayDataset(
                {m: arrays[m][test_indices] for m in MODALITIES}, labels[test_indices]
            ),
            batch_size=batch_size,
            shuffle=False,
        ),
    }
    model = EqualBranchCNN(MODALITIES).to(device)
    counts = np.bincount(labels[train_indices], minlength=3)
    weights = counts.sum() / np.maximum(counts, 1)
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best_state, best_f1, best_epoch, stale = None, -1.0, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, targets in loaders["train"]:
            inputs = {name: value.to(device) for name, value in inputs.items()}
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
        truth, probabilities, _ = predict(model, loaders["val"], device)
        score = f1_score(truth, probabilities.argmax(axis=1), average="macro")
        if score > best_f1 + 1e-5:
            best_state = copy.deepcopy(model.state_dict())
            best_f1, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    truth, probabilities, latency = predict(model, loaders["test"], device)
    return model, truth, probabilities, best_f1, best_epoch, latency


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("features_v2"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_v2/cvae_grouped"))
    parser.add_argument("--cvae-epochs", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--augmentation-ratio", type=float, default=1.0)
    parser.add_argument("--splits-limit", type=int)
    args = parser.parse_args()

    manifest = json.loads((args.splits_dir / "manifest.json").read_text(encoding="utf-8"))
    fingerprint = dataset_fingerprint(args.data_dir)
    if fingerprint != manifest["dataset_sha256"]:
        raise RuntimeError("Dataset fingerprint differs from immutable split manifest")
    labels = np.load(args.data_dir / "y.npy")
    groups = np.load(args.data_dir / "groups.npy")
    sample_ids = np.load(args.data_dir / "sample_ids.npy")
    class_names = json.loads((args.data_dir / "class_names.json").read_text(encoding="utf-8"))
    raw_arrays = {m: np.load(args.data_dir / f"X_{m}.npy") for m in MODALITIES}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows, prediction_rows = [], []
    split_files = manifest["files"][: args.splits_limit]
    for split_name in split_files:
        split = json.loads((args.splits_dir / split_name).read_text(encoding="utf-8"))
        train_indices = np.asarray(split["train_indices"], dtype=np.int64)
        val_indices = np.asarray(split["val_indices"], dtype=np.int64)
        test_indices = np.asarray(split["test_indices"], dtype=np.int64)
        if set(groups[train_indices]) & set(groups[val_indices]) or set(groups[train_indices]) & set(groups[test_indices]):
            raise RuntimeError("Subject leakage detected")
        run_seed = int(split["seed"]) * 100 + int(split["fold"])
        arrays, _ = fit_and_transform(raw_arrays, train_indices)
        flat_train = np.concatenate(
            [arrays[m][train_indices].reshape(len(train_indices), -1) for m in MODALITIES],
            axis=1,
        )
        started = time.perf_counter()
        cvae = train_cvae(
            flat_train, labels[train_indices], run_seed, args.cvae_epochs,
            args.batch_size, device,
        )
        generated, generated_labels = generate_train_only(
            cvae, flat_train, labels[train_indices], args.augmentation_ratio, device
        )
        model, truth, probabilities, best_f1, best_epoch, latency = train_classifier(
            arrays, labels, train_indices, val_indices, test_indices,
            augmented_arrays(generated), generated_labels, run_seed,
            args.epochs, args.patience, args.batch_size, device,
        )
        predictions = probabilities.argmax(axis=1)
        metrics = {
            "seed": split["seed"],
            "fold": split["fold"],
            "accuracy": accuracy_score(truth, predictions),
            "macro_f1": f1_score(truth, predictions, average="macro"),
            "best_val_macro_f1": best_f1,
            "best_epoch": best_epoch,
            "real_train_samples": len(train_indices),
            "synthetic_train_samples": len(generated_labels),
            "training_seconds": time.perf_counter() - started,
            "inference_latency_ms_per_sample": latency,
        }
        run_dir = args.output_dir / f"seed_{split['seed']}_fold_{split['fold']}"
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "modalities": MODALITIES}, run_dir / "model.pt")
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metric_rows.append(metrics)
        report = classification_report(
            truth, predictions, labels=np.arange(3), target_names=class_names,
            output_dict=True, zero_division=0,
        )
        (run_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        for local, index in enumerate(test_indices):
            prediction_rows.append({
                "seed": split["seed"], "fold": split["fold"],
                "sample_id": str(sample_ids[index]), "subject_id": int(groups[index]),
                "true_label": class_names[int(truth[local])],
                "predicted_label": class_names[int(predictions[local])],
                "prob_happy": float(probabilities[local, 0]),
                "prob_normal": float(probabilities[local, 1]),
                "prob_sad": float(probabilities[local, 2]),
            })
        print(
            f"seed={split['seed']} fold={split['fold']} "
            f"acc={metrics['accuracy']:.4f} f1={metrics['macro_f1']:.4f}",
            flush=True,
        )
    for filename, rows in (("fold_metrics.csv", metric_rows), ("predictions.csv", prediction_rows)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "experiment_contract.json").write_text(
        json.dumps({
            "dataset_sha256": fingerprint,
            "split_files": split_files,
            "modalities": MODALITIES,
            "cvae_scope": "train_indices_only_per_fold",
            "scaler_scope": "train_indices_only_per_fold",
            "validation_and_test_generation": False,
            "cvae_epochs": args.cvae_epochs,
            "cvae_latent_dim": 32,
            "cvae_beta": 0.1,
            "augmentation_ratio": args.augmentation_ratio,
            "synthetic_clip": "per-feature train 1st to 99th percentile",
            "classifier_epochs": args.epochs,
            "patience": args.patience,
            "device": str(device),
        }, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
