"""Train and package Production Baseline v1 from scratch on all canonical windows."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib
import numpy as np
import scipy
import sklearn
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eeg_emotion.models.production_baseline import (  # noqa: E402
    CLASS_NAMES,
    MODALITIES,
    VERSION,
    ProductionBaselineV1,
    sha256,
)


class FullDataset(Dataset):
    def __init__(self, arrays, labels):
        self.arrays, self.labels = arrays, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            {name: torch.from_numpy(value[index]).float() for name, value in self.arrays.items()},
            torch.tensor(self.labels[index], dtype=torch.long),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def source_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "git_unavailable_or_not_a_repository"


def main() -> None:
    data_dir = ROOT / "features_v2"
    cv_dir = ROOT / "outputs_v2" / "ablations" / "D_filtered_bandpower"
    package = ROOT / "production_baseline_v1"
    seed = 42

    cv_metrics = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(cv_dir.glob("seed_*_fold_*/metrics.json"))
    ]
    if len(cv_metrics) != 15:
        raise RuntimeError(f"Expected 15 CV results, found {len(cv_metrics)}")
    best_epochs = [int(item["best_epoch"]) for item in cv_metrics]
    epochs = int(np.median(best_epochs))
    if epochs != 20:
        raise RuntimeError(f"Frozen median epoch expected 20, got {epochs}")

    labels = np.load(data_dir / "y.npy")
    groups = np.load(data_dir / "groups.npy")
    sample_ids = np.load(data_dir / "sample_ids.npy")
    if len(labels) != 277 or len(np.unique(groups)) != 26:
        raise RuntimeError("Canonical production dataset identity/count mismatch")
    arrays, scalers = {}, {}
    for modality in MODALITIES:
        raw = np.load(data_dir / f"X_{modality}.npy")
        scaler = StandardScaler().fit(raw.reshape(-1, 4))
        arrays[modality] = scaler.transform(raw.reshape(-1, 4)).reshape(raw.shape).astype(np.float32)
        scalers[modality] = scaler

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProductionBaselineV1().to(device)
    counts = np.bincount(labels, minlength=3)
    weights = counts.sum() / np.maximum(counts, 1)
    weights /= weights.mean()
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loader = DataLoader(FullDataset(arrays, labels), batch_size=32, shuffle=True, num_workers=0)
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for inputs, target in loader:
            inputs = {name: value.to(device) for name, value in inputs.items()}
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), target)
            loss.backward()
            optimizer.step()
    training_seconds = time.perf_counter() - started
    model.eval()

    package.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": VERSION,
            "modalities": MODALITIES,
            "class_names": CLASS_NAMES,
            "production_seed": seed,
            "production_epochs": epochs,
            "model_state_dict": model.state_dict(),
        },
        package / "model.pt",
    )
    for modality, scaler in scalers.items():
        joblib.dump(scaler, package / f"scaler_{modality}.joblib")
    shutil.copy2(data_dir / "canonical_config.json", package / "canonical_feature_config.json")
    (package / "class_mapping.json").write_text(
        json.dumps({"happy": 0, "normal": 1, "sad": 2}, indent=2), encoding="utf-8"
    )

    golden = np.load(ROOT / "tests" / "fixtures" / "canonical_golden_sample.npz")
    golden_inputs, golden_saved = {}, {}
    for modality in MODALITIES:
        value = golden[f"expected_{modality}"][None].astype(np.float32)
        golden_saved[modality] = value
        scaled = scalers[modality].transform(value.reshape(-1, 4)).reshape(value.shape)
        golden_inputs[modality] = torch.tensor(scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        golden_saved["expected_probabilities"] = torch.softmax(
            model(golden_inputs), dim=1
        ).cpu().numpy()
    np.savez_compressed(package / "golden_inference_fixture.npz", **golden_saved)

    excluded = json.loads((data_dir / "excluded.json").read_text(encoding="utf-8"))
    reasons = Counter(item["reason"] for item in excluded)
    versions = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    lock_names = {"pytorch": "torch", "scikit_learn": "scikit-learn"}
    (package / "environment.lock").write_text(
        "\n".join(f"{lock_names.get(name, name)}=={version}" for name, version in versions.items()) + "\n",
        encoding="utf-8",
    )
    asset_hashes = {
        name: sha256(package / name)
        for name in (
            "model.pt", "scaler_filtered.joblib", "scaler_bandpower.joblib",
            "canonical_feature_config.json", "class_mapping.json",
            "golden_inference_fixture.npz",
        )
    }
    manifest = {
        "name": VERSION,
        "purpose": "deployment_only_performance_is_estimated_by_grouped_cross_validation",
        "training_window_count": int(len(labels)),
        "training_window_ids": sample_ids.tolist(),
        "subject_count": int(len(np.unique(groups))),
        "class_counts": {
            CLASS_NAMES[index]: int(count) for index, count in enumerate(counts)
        },
        "excluded_count": len(excluded),
        "excluded_reason_summary": dict(sorted(reasons.items())),
        "model_class": "ProductionBaselineV1",
        "model_modalities": list(MODALITIES),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "production_seed": seed,
        "production_epochs": epochs,
        "epoch_selection": "median_best_epoch_from_15_grouped_cv_runs",
        "cv_best_epochs": best_epochs,
        "source_revision": source_revision(),
        "source_sha256": {
            "model_definition": sha256(ROOT / "eeg_emotion" / "models" / "production_baseline.py"),
            "training_script": sha256(Path(__file__)),
        },
        "environment": versions,
        "training_device": str(device),
        "training_seconds": training_seconds,
        "asset_sha256": asset_hashes,
    }
    (package / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config_hash = sha256(package / "canonical_feature_config.json")
    contract_path = package / "baseline_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract.update(
        {
            "status": "production_model_trained",
            "class_names": list(CLASS_NAMES),
            "production_seed": seed,
            "production_epochs": epochs,
            "epoch_selection": "median_best_epoch_from_15_grouped_cv_runs",
            "canonical_feature_config_sha256": config_hash,
            "confidence_policy": "confidence_policy.json",
            "rejection_threshold": 0.60,
            "probability_calibration": "none",
            "deployment_weight_status": "trained_from_scratch_on_all_277_windows",
        }
    )
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")

    checksum_files = [
        "BASELINE_CARD.md", "baseline_contract.json", "model.pt",
        "scaler_filtered.joblib", "scaler_bandpower.joblib",
        "canonical_feature_config.json", "class_mapping.json",
        "training_manifest.json", "environment.lock", "golden_inference_fixture.npz",
        "confidence_policy.json",
    ]
    (package / "checksums.sha256").write_text(
        "".join(f"{sha256(package / name)}  {name}\n" for name in checksum_files),
        encoding="ascii",
    )
    print(json.dumps({"epochs": epochs, "training_seconds": training_seconds, "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
