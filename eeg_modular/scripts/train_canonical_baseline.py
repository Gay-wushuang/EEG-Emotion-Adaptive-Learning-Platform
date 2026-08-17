"""Train the no-CVAE four-modality CNN with subject-grouped evaluation."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from eeg_emotion.models.torch.multimodal_cvae_cnn import (
    MultiModalCVAECNN,
    MultiModalCVAECNNConfig,
)

MODALITIES = ("filtered", "bandpower", "att", "med")


class MultiModalDataset(Dataset):
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_scalers(
    arrays: dict[str, np.ndarray], train_indices: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, StandardScaler]]:
    scaled, scalers = {}, {}
    for modality, values in arrays.items():
        scaler = StandardScaler()
        scaler.fit(values[train_indices].reshape(-1, 4))
        scaled[modality] = scaler.transform(values.reshape(-1, 4)).reshape(values.shape).astype(np.float32)
        scalers[modality] = scaler
    return scaled, scalers


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    truths, probabilities = [], []
    for inputs, labels in loader:
        inputs = {name: value.to(device) for name, value in inputs.items()}
        logits = model(inputs)
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        truths.append(labels.numpy())
    return np.concatenate(truths), np.concatenate(probabilities)


def train_one_fold(
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
):
    scaled, scalers = fit_scalers(arrays, train_indices)
    train_loader = DataLoader(
        MultiModalDataset({m: scaled[m][train_indices] for m in MODALITIES}, labels[train_indices]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        MultiModalDataset({m: scaled[m][val_indices] for m in MODALITIES}, labels[val_indices]),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        MultiModalDataset({m: scaled[m][test_indices] for m in MODALITIES}, labels[test_indices]),
        batch_size=batch_size,
        shuffle=False,
    )

    model = MultiModalCVAECNN(
        num_classes=3,
        cfg=MultiModalCVAECNNConfig(
            modalities=list(MODALITIES),
            signal_modalities=["filtered", "bandpower"],
            scalar_modalities=["att", "med"],
            dropout=0.3,
            use_cvae=False,
        ),
        cvae_model=None,
    ).to(device)
    counts = np.bincount(labels[train_indices], minlength=3)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    best_state, best_val_f1, stale = None, -1.0, 0
    for _ in range(epochs):
        model.train()
        for inputs, target in train_loader:
            inputs = {name: value.to(device) for name, value in inputs.items()}
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), target)
            loss.backward()
            optimizer.step()
        val_true, val_probs = evaluate(model, val_loader, device)
        val_f1 = f1_score(val_true, np.argmax(val_probs, axis=1), average="macro")
        if val_f1 > best_val_f1 + 1e-5:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    test_true, test_probs = evaluate(model, test_loader, device)
    return model, scalers, test_true, test_probs, best_val_f1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("features_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_v2/canonical_baseline"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    arrays = {name: np.load(args.data_dir / f"X_{name}.npy") for name in MODALITIES}
    labels = np.load(args.data_dir / "y.npy")
    groups = np.load(args.data_dir / "groups.npy")
    sample_ids = np.load(args.data_dir / "sample_ids.npy")
    class_names = json.loads((args.data_dir / "class_names.json").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_rows, fold_metrics = [], []
    for seed in args.seeds:
        set_seed(seed)
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (development_indices, test_indices) in enumerate(
            outer.split(np.zeros(len(labels)), labels, groups)
        ):
            inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + fold)
            inner_train, inner_val = next(
                inner.split(
                    np.zeros(len(development_indices)),
                    labels[development_indices],
                    groups[development_indices],
                )
            )
            train_indices = development_indices[inner_train]
            val_indices = development_indices[inner_val]
            if set(groups[train_indices]) & set(groups[val_indices]):
                raise RuntimeError("Subject leakage between train and validation")
            if set(groups[train_indices]) & set(groups[test_indices]):
                raise RuntimeError("Subject leakage between train and test")
            if set(groups[val_indices]) & set(groups[test_indices]):
                raise RuntimeError("Subject leakage between validation and test")

            model, scalers, truth, probs, val_f1 = train_one_fold(
                arrays,
                labels,
                train_indices,
                val_indices,
                test_indices,
                seed,
                args.epochs,
                args.patience,
                args.batch_size,
                device,
            )
            prediction = np.argmax(probs, axis=1)
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
            fold_dir = args.output_dir / f"seed_{seed}" / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict()}, fold_dir / "model.pt")
            for modality, scaler in scalers.items():
                joblib.dump(scaler, fold_dir / f"scaler_{modality}.joblib")
            (fold_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "fold": fold,
                        "accuracy": accuracy,
                        "macro_f1": macro_f1,
                        "best_val_macro_f1": val_f1,
                        "classification_report": report,
                        "confusion_matrix": matrix.tolist(),
                        "train_subjects": sorted(map(int, set(groups[train_indices]))),
                        "val_subjects": sorted(map(int, set(groups[val_indices]))),
                        "test_subjects": sorted(map(int, set(groups[test_indices]))),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            fold_metrics.append(
                {"seed": seed, "fold": fold, "accuracy": accuracy, "macro_f1": macro_f1}
            )
            for local_index, dataset_index in enumerate(test_indices):
                all_rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "sample_id": str(sample_ids[dataset_index]),
                        "subject_id": int(groups[dataset_index]),
                        "true_label": class_names[int(truth[local_index])],
                        "predicted_label": class_names[int(prediction[local_index])],
                        "prob_happy": float(probs[local_index, 0]),
                        "prob_normal": float(probs[local_index, 1]),
                        "prob_sad": float(probs[local_index, 2]),
                    }
                )
            print(
                f"seed={seed} fold={fold} accuracy={accuracy:.4f} "
                f"macro_f1={macro_f1:.4f} val_f1={val_f1:.4f}"
            )

    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    accuracy_values = np.asarray([row["accuracy"] for row in fold_metrics])
    f1_values = np.asarray([row["macro_f1"] for row in fold_metrics])
    summary = {
        "device": str(device),
        "seeds": args.seeds,
        "folds_per_seed": 5,
        "accuracy_mean": float(accuracy_values.mean()),
        "accuracy_std": float(accuracy_values.std(ddof=1)),
        "macro_f1_mean": float(f1_values.mean()),
        "macro_f1_std": float(f1_values.std(ddof=1)),
        "fold_metrics": fold_metrics,
        "leakage_check": "passed: train/val/test subject sets disjoint in every fold",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(summary)


if __name__ == "__main__":
    main()
