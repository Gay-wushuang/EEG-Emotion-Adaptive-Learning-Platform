"""Write the immutable pre-capture contract for the original-order repeat."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

from mark_five_stage_experiment import ORIGINAL_STAGES


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "git_unavailable_or_not_a_repository"


def main():
    tracked = {
        "model": ROOT / "production_baseline_v1" / "model.pt",
        "scaler_filtered": ROOT / "production_baseline_v1" / "scaler_filtered.joblib",
        "scaler_bandpower": ROOT / "production_baseline_v1" / "scaler_bandpower.joblib",
        "canonical_feature_config": ROOT / "production_baseline_v1" / "canonical_feature_config.json",
        "class_mapping": ROOT / "production_baseline_v1" / "class_mapping.json",
        "label_semantics": ROOT / "LABEL_SEMANTICS.md",
        "quality_algorithm": ROOT / "eeg_emotion" / "features" / "quality.py",
        "quality_gate": ROOT / "eeg_emotion" / "features" / "quality_gate.py",
        "quality_policy": ROOT / "quality_reference_v1" / "quality_gate_policy.json",
        "event_script": ROOT / "scripts" / "mark_five_stage_experiment.py",
        "converter": ROOT / "realtime_inference" / "tools" / "convert_canonical_capture.py",
        "analysis_script": ROOT / "scripts" / "analyze_diagnostic_experiment.py",
    }
    contract = {
        "name": "Original-order personal repeat v1",
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "frozen_before_capture",
        "production_model": "Production Baseline v1",
        "model_modalities": ["filtered", "bandpower"],
        "sampling": {
            "nominal_sample_rate_hz": 512,
            "signal_time_rule": "sample index at nominal 512 Hz for feature extraction",
            "event_alignment_rule": "affine sample_index to packet_received_unix fit",
            "window_seconds": 30,
            "window_samples": 15360,
            "stride_seconds": 2,
            "stride_samples": 1024,
        },
        "labels": {
            "internal": ["happy", "normal", "sad"],
            "display": ["positive", "neutral", "negative"],
            "mapping": {
                "happy": "positive",
                "normal": "neutral",
                "sad": "negative",
            },
        },
        "stage_schedule": [
            {"stage": stage, "seconds": seconds, "description": description}
            for stage, seconds, description in ORIGINAL_STAGES
        ],
        "automatic_intervention": False,
        "confidence_rejection_for_primary_stage_analysis": False,
        "quality_gate": {
            "policy": "Quality Gate v1",
            "fitted_on": "277 training windows, 26 subjects",
            "evaluation_order": "before emotion probabilities",
            "levels": ["trusted", "warning", "low_ood"],
            "interpretation_rule": "only trusted windows allow limited state interpretation",
        },
        "analysis_order": [
            "data_integrity",
            "timeline_and_event_alignment",
            "quality_metrics",
            "quality_gate",
            "frozen_model_inference",
            "stage_and_half_summaries",
            "session_level_comparison",
        ],
        "statistics": {
            "independent_unit": "experiment session",
            "supporting_units": ["non-overlapping 30-second blocks", "stage median"],
            "forbidden": "window-level independent-sample significance",
        },
        "git_revision": git_revision(),
        "sha256": {name: sha256(path) for name, path in tracked.items()},
    }
    output = ROOT / "repeat_protocol_v1"
    output.mkdir(exist_ok=True)
    path = output / "ORIGINAL_REPEAT_FROZEN.json"
    path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "ORIGINAL_REPEAT_FROZEN.sha256").write_text(
        f"{sha256(path)}  {path.name}\n", encoding="ascii"
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
