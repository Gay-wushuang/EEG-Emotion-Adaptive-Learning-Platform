"""Frozen architecture and strict loader for Production Baseline v1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn


VERSION = "Production Baseline v1"
MODALITIES = ("filtered", "bandpower")
CLASS_NAMES = ("happy", "normal", "sad")
DISPLAY_CLASS_NAMES = ("positive", "neutral", "negative")
DISPLAY_NAME_BY_INTERNAL = dict(zip(CLASS_NAMES, DISPLAY_CLASS_NAMES))


class ProductionBaselineV1(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.modalities = MODALITIES
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
                for modality in MODALITIES
            }
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.logits_from_embeddings(self.branch_embeddings(inputs))

    def branch_embeddings(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if tuple(inputs.keys()) != MODALITIES:
            raise ValueError(f"Expected exactly {MODALITIES}, got {tuple(inputs.keys())}")
        return {
            name: self.branches[name](inputs[name].transpose(1, 2))
            for name in MODALITIES
        }

    def logits_from_embeddings(
        self, embeddings: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.classifier(
            torch.cat([embeddings[name] for name in MODALITIES], dim=1)
        )

    def diagnostic_outputs(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Two-branch Shapley decomposition in logit space using zero embeddings."""
        embeddings = self.branch_embeddings(inputs)
        zero = {name: torch.zeros_like(value) for name, value in embeddings.items()}
        logits_none = self.logits_from_embeddings(zero)
        logits_filtered = self.logits_from_embeddings(
            {"filtered": embeddings["filtered"], "bandpower": zero["bandpower"]}
        )
        logits_bandpower = self.logits_from_embeddings(
            {"filtered": zero["filtered"], "bandpower": embeddings["bandpower"]}
        )
        logits_fusion = self.logits_from_embeddings(embeddings)
        return {
            "embedding_filtered": embeddings["filtered"],
            "embedding_bandpower": embeddings["bandpower"],
            "logits_none": logits_none,
            "logits_filtered_only": logits_filtered,
            "logits_bandpower_only": logits_bandpower,
            "logits_fusion": logits_fusion,
            "logit_contribution_filtered": 0.5
            * ((logits_filtered - logits_none) + (logits_fusion - logits_bandpower)),
            "logit_contribution_bandpower": 0.5
            * ((logits_bandpower - logits_none) + (logits_fusion - logits_filtered)),
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_production_package(package_dir: Path, device: str = "cpu"):
    package_dir = Path(package_dir)
    checksum_path = package_dir / "checksums.sha256"
    if not checksum_path.exists():
        raise RuntimeError("Missing checksums.sha256")
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        expected, filename = line.split("  ", 1)
        target = package_dir / filename
        if not target.is_file() or sha256(target) != expected:
            raise RuntimeError(f"Package checksum mismatch: {filename}")
    contract = json.loads((package_dir / "baseline_contract.json").read_text(encoding="utf-8"))
    if contract["name"] != VERSION or contract["status"] != "production_model_trained":
        raise RuntimeError("Production model version/status mismatch")
    if tuple(contract["emotion_classifier_modalities"]) != MODALITIES:
        raise RuntimeError("Classifier modalities must be filtered + bandpower only")
    if tuple(contract["class_names"]) != CLASS_NAMES:
        raise RuntimeError("Class order mismatch")
    class_mapping = json.loads((package_dir / "class_mapping.json").read_text(encoding="utf-8"))
    if class_mapping != {name: index for index, name in enumerate(CLASS_NAMES)}:
        raise RuntimeError("class_mapping.json mismatch")
    confidence_policy = json.loads(
        (package_dir / "confidence_policy.json").read_text(encoding="utf-8")
    )
    if confidence_policy["probability_calibration"] != "none":
        raise RuntimeError("Production v1 must not apply unvalidated probability calibration")
    if float(confidence_policy["rejection_threshold"]) != 0.60:
        raise RuntimeError("Production v1 rejection threshold mismatch")
    config_path = package_dir / "canonical_feature_config.json"
    if sha256(config_path) != contract["canonical_feature_config_sha256"]:
        raise RuntimeError("Canonical feature config hash mismatch")

    scalers = {}
    for modality in MODALITIES:
        scaler_path = package_dir / f"scaler_{modality}.joblib"
        if not scaler_path.exists():
            raise RuntimeError(f"Missing scaler: {scaler_path.name}")
        scaler = joblib.load(scaler_path)
        if getattr(scaler, "n_features_in_", None) != 4:
            raise RuntimeError(f"{modality} scaler must have exactly 4 features")
        scalers[modality] = scaler

    checkpoint = torch.load(package_dir / "model.pt", map_location=device, weights_only=True)
    if checkpoint["version"] != VERSION or tuple(checkpoint["modalities"]) != MODALITIES:
        raise RuntimeError("Checkpoint version/modalities mismatch")
    model = ProductionBaselineV1().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    fixture = np.load(package_dir / "golden_inference_fixture.npz")
    inputs = {}
    for modality in MODALITIES:
        value = fixture[modality]
        if value.shape != (1, 10, 4):
            raise RuntimeError(f"Golden {modality} shape mismatch: {value.shape}")
        scaled = scalers[modality].transform(value.reshape(-1, 4)).reshape(value.shape)
        inputs[modality] = torch.tensor(scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        actual = torch.softmax(model(inputs), dim=1).cpu().numpy()
    np.testing.assert_allclose(actual, fixture["expected_probabilities"], rtol=1e-5, atol=1e-6)
    return model, scalers, contract


def predict_probabilities(model, scalers, arrays: dict[str, np.ndarray], device: str = "cpu"):
    if tuple(arrays.keys()) != MODALITIES:
        raise ValueError(f"Expected exactly {MODALITIES}; ATT/MED must not enter classifier")
    tensors = {}
    for modality in MODALITIES:
        values = np.asarray(arrays[modality])
        if values.ndim != 3 or values.shape[1:] != (10, 4):
            raise ValueError(f"{modality} must have shape (N,10,4), got {values.shape}")
        scaled = scalers[modality].transform(values.reshape(-1, 4)).reshape(values.shape)
        tensors[modality] = torch.tensor(scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        return torch.softmax(model(tensors), dim=1).cpu().numpy()
