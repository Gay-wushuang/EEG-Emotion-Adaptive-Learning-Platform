"""Diagnostic-only modality ablation on pre-scaled real capture windows."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import EmotionInferenceModel, InferenceConfig

MODALITIES = ("filtered", "powerspec", "att", "med")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scalers-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    archive = np.load(args.features)
    real = {name: archive[name] for name in MODALITIES}
    zeros = {name: np.zeros_like(real[name]) for name in MODALITIES}
    variants = {
        "all_real": set(MODALITIES),
        "filtered_to_train_mean": set(MODALITIES) - {"filtered"},
        "powerspec_to_train_mean": set(MODALITIES) - {"powerspec"},
        "filtered_powerspec_to_train_mean": {"att", "med"},
        "att_med_to_train_mean": {"filtered", "powerspec"},
        "filtered_only": {"filtered"},
        "powerspec_only": {"powerspec"},
        "att_only": {"att"},
        "med_only": {"med"},
    }
    model = EmotionInferenceModel(
        InferenceConfig(
            model_path=args.model,
            modalities=MODALITIES,
            scalers_dir=args.scalers_dir,
            skip_scaling=True,
        )
    )

    rows = []
    count = real["filtered"].shape[0]
    for variant, real_modalities in variants.items():
        probabilities = []
        for index in range(count):
            sample = {
                name: (real[name][index] if name in real_modalities else zeros[name][index])
                for name in MODALITIES
            }
            _, probs = model.predict(sample)
            probabilities.append(probs)
        mean_probs = np.mean(np.stack(probabilities), axis=0)
        by_class = dict(zip(model.class_names, mean_probs.tolist()))
        rows.append(
            {
                "variant": variant,
                "happy": by_class["happy"],
                "normal": by_class["normal"],
                "sad": by_class["sad"],
                "predicted_class": model.class_names[int(np.argmax(mean_probs))],
                "mean_confidence": float(np.mean(np.max(probabilities, axis=1))),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
