"""Fail-closed startup verification for Production Baseline v1."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eeg_emotion.models.production_baseline import load_production_package


if __name__ == "__main__":
    model, scalers, contract = load_production_package(ROOT / "production_baseline_v1")
    print(
        f"OK: {contract['name']}; modalities={tuple(scalers)}; "
        f"parameters={sum(p.numel() for p in model.parameters())}"
    )
