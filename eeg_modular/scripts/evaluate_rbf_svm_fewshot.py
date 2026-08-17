"""Evaluate RBF-SVM and exact per-class few-shot calibration on grouped splits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from create_grouped_splits import dataset_fingerprint


PARAMETERS = tuple(
    (c, gamma)
    for c in (0.1, 1.0, 10.0, 100.0)
    for gamma in ("scale", 0.01, 0.1, 1.0)
)


def flat_features(data_dir: Path) -> np.ndarray:
    return np.concatenate(
        [
            np.load(data_dir / "X_filtered.npy").reshape(-1, 40),
            np.load(data_dir / "X_bandpower.npy").reshape(-1, 40),
        ],
        axis=1,
    )


def select_parameters(
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
) -> tuple[float, str | float, float]:
    scaler = StandardScaler().fit(features[train])
    train_x = scaler.transform(features[train])
    validation_x = scaler.transform(features[validation])
    best = (-1.0, PARAMETERS[0])
    for c, gamma in PARAMETERS:
        model = SVC(C=c, gamma=gamma, kernel="rbf", class_weight="balanced")
        model.fit(train_x, labels[train])
        score = f1_score(
            labels[validation], model.predict(validation_x), average="macro"
        )
        if score > best[0] + 1e-12:
            best = (score, (c, gamma))
    return best[1][0], best[1][1], best[0]


def fit_model(
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    c: float,
    gamma: str | float,
) -> tuple[StandardScaler, SVC]:
    scaler = StandardScaler().fit(features[indices])
    model = SVC(C=c, gamma=gamma, kernel="rbf", class_weight="balanced")
    model.fit(scaler.transform(features[indices]), labels[indices])
    return scaler, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("features_v2"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_v2/rbf_fewshot"))
    args = parser.parse_args()
    manifest = json.loads((args.splits_dir / "manifest.json").read_text(encoding="utf-8"))
    fingerprint = dataset_fingerprint(args.data_dir)
    if fingerprint != manifest["dataset_sha256"]:
        raise RuntimeError("Dataset fingerprint differs from immutable split manifest")
    features = flat_features(args.data_dir)
    labels = np.load(args.data_dir / "y.npy")
    groups = np.load(args.data_dir / "groups.npy")
    sample_ids = np.load(args.data_dir / "sample_ids.npy")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows, prediction_rows, calibration_rows = [], [], []

    for split_name in manifest["files"]:
        split = json.loads((args.splits_dir / split_name).read_text(encoding="utf-8"))
        train = np.asarray(split["train_indices"], dtype=np.int64)
        validation = np.asarray(split["val_indices"], dtype=np.int64)
        test = np.asarray(split["test_indices"], dtype=np.int64)
        c, gamma, validation_f1 = select_parameters(
            features, labels, train, validation
        )
        scaler, model = fit_model(features, labels, train, c, gamma)
        test_predictions = model.predict(scaler.transform(features[test]))
        fold_rows.append(
            {
                "seed": split["seed"], "fold": split["fold"],
                "accuracy": accuracy_score(labels[test], test_predictions),
                "macro_f1": f1_score(labels[test], test_predictions, average="macro"),
                "selected_c": c, "selected_gamma": gamma,
                "validation_macro_f1": validation_f1,
            }
        )
        prediction_lookup = dict(zip(test.tolist(), test_predictions.tolist()))
        for index in test:
            prediction_rows.append(
                {
                    "seed": split["seed"], "fold": split["fold"],
                    "sample_id": str(sample_ids[index]),
                    "subject_id": int(groups[index]),
                    "true_label": int(labels[index]),
                    "predicted_label": int(prediction_lookup[index]),
                }
            )
        run_seed = int(split["seed"]) * 100 + int(split["fold"])
        for subject in np.unique(groups[test]):
            subject_test = test[groups[test] == subject]
            for shots in (1, 2, 3):
                if any((labels[subject_test] == label).sum() < shots + 1 for label in range(3)):
                    continue
                rng = np.random.default_rng(run_seed * 10000 + int(subject) * 10 + shots)
                calibration = np.concatenate(
                    [
                        rng.choice(
                            subject_test[labels[subject_test] == label],
                            size=shots,
                            replace=False,
                        )
                        for label in range(3)
                    ]
                )
                evaluation = np.setdiff1d(subject_test, calibration)
                adapted_scaler, adapted_model = fit_model(
                    features,
                    labels,
                    np.concatenate([train, calibration]),
                    c,
                    gamma,
                )
                adapted = adapted_model.predict(
                    adapted_scaler.transform(features[evaluation])
                )
                zero = np.asarray([prediction_lookup[index] for index in evaluation])
                calibration_rows.append(
                    {
                        "seed": split["seed"], "fold": split["fold"],
                        "subject_id": int(subject), "shots_per_class": shots,
                        "calibration_samples": len(calibration),
                        "evaluation_samples": len(evaluation),
                        "zero_shot_accuracy": accuracy_score(labels[evaluation], zero),
                        "calibrated_accuracy": accuracy_score(labels[evaluation], adapted),
                        "zero_shot_macro_f1": f1_score(
                            labels[evaluation], zero, labels=np.arange(3),
                            average="macro", zero_division=0,
                        ),
                        "calibrated_macro_f1": f1_score(
                            labels[evaluation], adapted, labels=np.arange(3),
                            average="macro", zero_division=0,
                        ),
                    }
                )
        print(
            f"seed={split['seed']} fold={split['fold']} C={c} gamma={gamma} "
            f"acc={fold_rows[-1]['accuracy']:.4f} f1={fold_rows[-1]['macro_f1']:.4f}",
            flush=True,
        )

    for filename, rows in (
        ("fold_metrics.csv", fold_rows),
        ("predictions.csv", prediction_rows),
        ("fewshot_subject_metrics.csv", calibration_rows),
    ):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "experiment_contract.json").write_text(
        json.dumps(
            {
                "dataset_sha256": fingerprint,
                "split_files": manifest["files"],
                "features": "filtered+bandpower flattened to 80 dimensions",
                "parameter_selection": "validation subjects only",
                "grid": [{"C": c, "gamma": gamma} for c, gamma in PARAMETERS],
                "fewshot_rule": "exact k/class; calibration removed from evaluation",
                "fewshot_eligibility": "at least k+1 samples in every class",
                "calibration_scaler_scope": "train indices plus current-user calibration only",
                "test_subjects_used_for_general_hyperparameters": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
