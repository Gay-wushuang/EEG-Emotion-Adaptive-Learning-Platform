"""Build an anonymous feature-parity fixture from one mapped training sample."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = ROOT / "time_data_preprocess"
sys.path.insert(0, str(TRAINING_ROOT))
sys.path.insert(0, str(ROOT / "eeg_modular"))
sys.path.insert(0, str(ROOT / "eeg_modular" / "realtime_inference"))

from preprocess.alignment import load_and_align_csv
from preprocess.feature_extraction import extract_time_features
from preprocess.filters import bandpass_filter
from src.model import EmotionInferenceModel, InferenceConfig

MODALITIES = ("filtered", "powerspec", "att", "med")


def numeric_values(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    return frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)


def main() -> None:
    sample = TRAINING_ROOT / "data" / "happy" / "sample1"
    features_dir = ROOT / "eeg_modular" / "features"
    output = Path(__file__).parent / "fixtures" / "training_golden_sample.npz"
    output.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray] = {
        "raw_eeg": numeric_values(sample / "rawwave.csv").ravel(),
        "filtered_source": numeric_values(sample / "filtered.csv").ravel(),
        "powerspec_source": numeric_values(sample / "powerspec.csv"),
        "att_source": numeric_values(sample / "att.csv").ravel(),
        "med_source": numeric_values(sample / "med.csv").ravel(),
    }

    for modality in MODALITIES:
        aligned = load_and_align_csv(str(sample / f"{modality}.csv"), freq="100ms")
        aligned_values = aligned.to_numpy(dtype=np.float32)
        payload[f"aligned_{modality}"] = aligned_values
        processed = (
            bandpass_filter(aligned_values).astype(np.float32)
            if modality == "filtered"
            else aligned_values
        )
        payload[f"processed_{modality}"] = processed
        unscaled = extract_time_features(pd.DataFrame(processed), time_steps=10)
        scaler = joblib.load(features_dir / f"scaler_{modality}.joblib")
        scaled = scaler.transform(unscaled).astype(np.float32)
        payload[f"expected_unscaled_{modality}"] = unscaled
        payload[f"expected_scaled_{modality}"] = scaled
        payload[f"scaler_mean_{modality}"] = scaler.mean_.astype(np.float64)
        payload[f"scaler_scale_{modality}"] = scaler.scale_.astype(np.float64)

    model = EmotionInferenceModel(
        InferenceConfig(
            model_path=ROOT / "eeg_modular" / "outputs" / "CNN" / "models" / "best_fold4.pt",
            modalities=MODALITIES,
            scalers_dir=features_dir,
            skip_scaling=True,
        )
    )
    predicted, probabilities = model.predict(
        {modality: payload[f"expected_scaled_{modality}"] for modality in MODALITIES}
    )
    payload["expected_probabilities"] = probabilities.astype(np.float32)
    payload["expected_class_index"] = np.asarray(
        [model.class_names.index(predicted)], dtype=np.int64
    )
    payload["class_names"] = np.asarray(model.class_names)

    np.savez_compressed(output, **payload)
    print(output)


if __name__ == "__main__":
    main()
