"""Run five no-CVAE modality ablations on immutable grouped split files."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from create_grouped_splits import dataset_fingerprint

EXPERIMENTS = {
    "A_filtered": ("filtered",),
    "B_bandpower": ("bandpower",),
    "C_att_med": ("att", "med"),
    "D_filtered_bandpower": ("filtered", "bandpower"),
    "E_all": ("filtered", "bandpower", "att", "med"),
}


class ArrayDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], labels: np.ndarray):
        self.arrays = arrays
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            {name: torch.from_numpy(values[index]).float() for name, values in self.arrays.items()},
            torch.tensor(self.labels[index], dtype=torch.long),
        )


class EqualBranchCNN(nn.Module):
    """Every participating modality uses the exact same 32-dimensional branch."""

    def __init__(self, modalities: tuple[str, ...], dropout: float = 0.3):
        super().__init__()
        self.modalities = modalities
        self.branches = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Conv1d(4, 32, kernel_size=3, padding=1),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Conv1d(32, 32, kernel_size=3, padding=1),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                )
                for modality in modalities
            }
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * len(modalities), 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        features = [
            self.branches[modality](inputs[modality].transpose(1, 2))
            for modality in self.modalities
        ]
        return self.classifier(torch.cat(features, dim=1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fit_and_transform(
    arrays: dict[str, np.ndarray], train_indices: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, StandardScaler]]:
    scaled, scalers = {}, {}
    for modality, values in arrays.items():
        scaler = StandardScaler().fit(values[train_indices].reshape(-1, 4))
        scaled[modality] = scaler.transform(values.reshape(-1, 4)).reshape(values.shape).astype(np.float32)
        scalers[modality] = scaler
    return scaled, scalers


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    truths, probabilities = [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for inputs, labels in loader:
        inputs = {name: value.to(device) for name, value in inputs.items()}
        probabilities.append(torch.softmax(model(inputs), dim=1).cpu().numpy())
        truths.append(labels.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    truth = np.concatenate(truths)
    return truth, np.concatenate(probabilities), 1000.0 * elapsed / len(truth)


def train_fold(
    raw_arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    modalities: tuple[str, ...],
    run_seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
):
    set_seed(run_seed)
    arrays, scalers = fit_and_transform(raw_arrays, train_indices)
    loaders = {}
    for name, indices, shuffle in (
        ("train", train_indices, True),
        ("val", val_indices, False),
        ("test", test_indices, False),
    ):
        loaders[name] = DataLoader(
            ArrayDataset({m: arrays[m][indices] for m in modalities}, labels[indices]),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
        )
    model = EqualBranchCNN(modalities).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    counts = np.bincount(labels[train_indices], minlength=3)
    weights = counts.sum() / np.maximum(counts, 1)
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best_state, best_f1, stale, best_epoch = None, -1.0, 0, 0
    train_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, targets in loaders["train"]:
            inputs = {name: value.to(device) for name, value in inputs.items()}
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
        val_truth, val_probs, _ = predict(model, loaders["val"], device)
        val_f1 = f1_score(val_truth, np.argmax(val_probs, axis=1), average="macro")
        if val_f1 > best_f1 + 1e-5:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    training_seconds = time.perf_counter() - train_started
    model.load_state_dict(best_state)
    truth, probabilities, latency_ms = predict(model, loaders["test"], device)
    return (
        model,
        scalers,
        truth,
        probabilities,
        best_f1,
        best_epoch,
        parameter_count,
        training_seconds,
        latency_ms,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("features_v2"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_v2/ablations"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--experiments", nargs="+", choices=tuple(EXPERIMENTS), default=list(EXPERIMENTS),
        help="Subset of ablations to run; defaults to all five.",
    )
    args = parser.parse_args()

    manifest = json.loads((args.splits_dir / "manifest.json").read_text(encoding="utf-8"))
    current_fingerprint = dataset_fingerprint(args.data_dir)
    if current_fingerprint != manifest["dataset_sha256"]:
        raise RuntimeError("Dataset fingerprint differs from immutable split manifest")
    labels = np.load(args.data_dir / "y.npy")
    groups = np.load(args.data_dir / "groups.npy")
    sample_ids = np.load(args.data_dir / "sample_ids.npy")
    class_names = json.loads((args.data_dir / "class_names.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows, metric_rows = [], []
    for experiment in args.experiments:
        modalities = EXPERIMENTS[experiment]
        raw_arrays = {
            modality: np.load(args.data_dir / f"X_{modality}.npy")
            for modality in modalities
        }
        for split_name in manifest["files"]:
            split = json.loads((args.splits_dir / split_name).read_text(encoding="utf-8"))
            train_indices = np.asarray(split["train_indices"], dtype=np.int64)
            val_indices = np.asarray(split["val_indices"], dtype=np.int64)
            test_indices = np.asarray(split["test_indices"], dtype=np.int64)
            run_seed = int(split["seed"]) * 100 + int(split["fold"])
            result = train_fold(
                raw_arrays,
                labels,
                train_indices,
                val_indices,
                test_indices,
                modalities,
                run_seed,
                args.epochs,
                args.patience,
                args.batch_size,
                device,
            )
            (
                model,
                scalers,
                truth,
                probabilities,
                val_f1,
                best_epoch,
                parameter_count,
                training_seconds,
                latency_ms,
            ) = result
            prediction = np.argmax(probabilities, axis=1)
            accuracy = accuracy_score(truth, prediction)
            macro_f1 = f1_score(truth, prediction, average="macro")
            report = classification_report(
                truth,
                prediction,
                labels=np.arange(3),
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
            matrix = confusion_matrix(truth, prediction, labels=np.arange(3))
            run_dir = (
                args.output_dir
                / experiment
                / f"seed_{split['seed']}_fold_{split['fold']}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "modalities": modalities}, run_dir / "model.pt")
            for modality, scaler in scalers.items():
                joblib.dump(scaler, run_dir / f"scaler_{modality}.joblib")
            metrics = {
                "experiment": experiment,
                "modalities": modalities,
                "seed": split["seed"],
                "fold": split["fold"],
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "best_val_macro_f1": val_f1,
                "best_epoch": best_epoch,
                "parameter_count": parameter_count,
                "training_seconds": training_seconds,
                "inference_latency_ms_per_sample": latency_ms,
                "classification_report": report,
                "confusion_matrix": matrix.tolist(),
                "dataset_sha256": current_fingerprint,
                "split_file": split_name,
            }
            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            metric_rows.append(
                {key: metrics[key] for key in (
                    "experiment", "seed", "fold", "accuracy", "macro_f1",
                    "parameter_count", "training_seconds", "inference_latency_ms_per_sample"
                )}
            )
            for local_index, dataset_index in enumerate(test_indices):
                prediction_rows.append(
                    {
                        "experiment": experiment,
                        "seed": split["seed"],
                        "fold": split["fold"],
                        "sample_id": str(sample_ids[dataset_index]),
                        "subject_id": int(groups[dataset_index]),
                        "true_label": class_names[int(truth[local_index])],
                        "predicted_label": class_names[int(prediction[local_index])],
                        "prob_happy": float(probabilities[local_index, 0]),
                        "prob_normal": float(probabilities[local_index, 1]),
                        "prob_sad": float(probabilities[local_index, 2]),
                    }
                )
            print(
                f"{experiment} seed={split['seed']} fold={split['fold']} "
                f"acc={accuracy:.4f} f1={macro_f1:.4f} params={parameter_count}"
            )

    for filename, rows in (("fold_metrics.csv", metric_rows), ("predictions.csv", prediction_rows)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "experiment_contract.json").write_text(
        json.dumps(
            {
                "dataset_sha256": current_fingerprint,
                "split_manifest": str(args.splits_dir / "manifest.json"),
                "experiments": {name: list(value) for name, value in EXPERIMENTS.items()},
                "seeds": manifest["seeds"],
                "folds_per_seed": manifest["folds_per_seed"],
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "learning_rate": 1e-3,
                "weight_decay": 1e-3,
                "selection_metric": "validation macro-F1",
                "class_weight": "inverse frequency fitted from train indices only",
                "deterministic_algorithms": True,
                "cvae": False,
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
