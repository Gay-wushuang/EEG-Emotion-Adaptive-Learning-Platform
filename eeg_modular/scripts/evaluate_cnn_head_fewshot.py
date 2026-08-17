"""Evaluate validation-tuned last-layer CNN few-shot calibration."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from train_canonical_ablations import fit_and_transform, set_seed, train_fold


MODALITIES = ("filtered", "bandpower")
CANDIDATES = ((1e-3, 20), (1e-3, 50), (5e-3, 20), (5e-3, 50), (1e-2, 20), (1e-2, 50))


def tensors(arrays: dict[str, np.ndarray], indices: np.ndarray, device):
    return {name: torch.from_numpy(values[indices]).float().to(device) for name, values in arrays.items()}


def adapt(model, arrays, labels, calibration, lr, steps, device):
    adapted = copy.deepcopy(model).to(device)
    adapted.eval()
    for parameter in adapted.parameters():
        parameter.requires_grad = False
    layer = adapted.classifier[-1]
    for parameter in layer.parameters():
        parameter.requires_grad = True
    anchor_weight = layer.weight.detach().clone()
    anchor_bias = layer.bias.detach().clone()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=lr, weight_decay=0.0)
    inputs = tensors(arrays, calibration, device)
    targets = torch.from_numpy(labels[calibration]).long().to(device)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = adapted(inputs)
        anchor = (layer.weight - anchor_weight).pow(2).mean() + (layer.bias - anchor_bias).pow(2).mean()
        loss = F.cross_entropy(logits, targets) + anchor
        loss.backward(); optimizer.step()
    return adapted


@torch.no_grad()
def predict(model, arrays, indices, device):
    model.eval()
    return model(tensors(arrays, indices, device)).argmax(dim=1).cpu().numpy()


def split_subject_samples(indices, labels, groups, subject, shots, seed):
    current = indices[groups[indices] == subject]
    if any((labels[current] == label).sum() < shots + 1 for label in range(3)):
        return None
    rng = np.random.default_rng(seed * 10000 + int(subject) * 10 + shots)
    calibration = np.concatenate([
        rng.choice(current[labels[current] == label], shots, replace=False)
        for label in range(3)
    ])
    return calibration, np.setdiff1d(current, calibration)


def choose_candidate(model, arrays, labels, groups, validation, run_seed, device):
    scored = []
    for lr, steps in CANDIDATES:
        values = []
        for subject in np.unique(groups[validation]):
            selection = split_subject_samples(validation, labels, groups, subject, 2, run_seed)
            if selection is None:
                continue
            calibration, evaluation = selection
            adapted = adapt(model, arrays, labels, calibration, lr, steps, device)
            values.append(f1_score(labels[evaluation], predict(adapted, arrays, evaluation, device), labels=np.arange(3), average="macro", zero_division=0))
        scored.append((float(np.mean(values)) if values else -1.0, lr, steps))
    return max(scored, key=lambda item: (item[0], -item[1], -item[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("features_v2"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_v2/cnn_head_fewshot"))
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    labels = np.load(args.data_dir / "y.npy")
    groups = np.load(args.data_dir / "groups.npy")
    raw = {name: np.load(args.data_dir / f"X_{name}.npy") for name in MODALITIES}
    manifest = json.loads((args.splits_dir / "manifest.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, selections = [], []
    for split_name in manifest["files"]:
        split = json.loads((args.splits_dir / split_name).read_text(encoding="utf-8"))
        train = np.asarray(split["train_indices"], dtype=np.int64)
        validation = np.asarray(split["val_indices"], dtype=np.int64)
        test = np.asarray(split["test_indices"], dtype=np.int64)
        run_seed = int(split["seed"]) * 100 + int(split["fold"])
        result = train_fold(raw, labels, train, validation, test, MODALITIES, run_seed, 120, 15, 32, device)
        model, scalers = result[0], result[1]
        arrays = {name: scalers[name].transform(raw[name].reshape(-1, 4)).reshape(raw[name].shape).astype(np.float32) for name in MODALITIES}
        validation_score, lr, steps = choose_candidate(model, arrays, labels, groups, validation, run_seed, device)
        selections.append({"seed": split["seed"], "fold": split["fold"], "lr": lr, "steps": steps, "validation_macro_f1": validation_score})
        zero_predictions = {int(index): int(value) for index, value in zip(test, predict(model, arrays, test, device))}
        for subject in np.unique(groups[test]):
            for shots in args.shots:
                selection = split_subject_samples(test, labels, groups, subject, shots, run_seed)
                if selection is None:
                    continue
                calibration, evaluation = selection
                adapted = adapt(model, arrays, labels, calibration, lr, steps, device)
                calibrated = predict(adapted, arrays, evaluation, device)
                zero = np.asarray([zero_predictions[int(index)] for index in evaluation])
                rows.append({
                    "seed": split["seed"], "fold": split["fold"], "subject_id": int(subject), "shots_per_class": shots,
                    "evaluation_samples": len(evaluation),
                    "zero_shot_accuracy": accuracy_score(labels[evaluation], zero),
                    "calibrated_accuracy": accuracy_score(labels[evaluation], calibrated),
                    "zero_shot_macro_f1": f1_score(labels[evaluation], zero, labels=np.arange(3), average="macro", zero_division=0),
                    "calibrated_macro_f1": f1_score(labels[evaluation], calibrated, labels=np.arange(3), average="macro", zero_division=0),
                })
        print(f"seed={split['seed']} fold={split['fold']} lr={lr} steps={steps}", flush=True)
    for filename, data in (("fewshot_subject_metrics.csv", rows), ("selected_hyperparameters.csv", selections)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)


if __name__ == "__main__":
    main()
