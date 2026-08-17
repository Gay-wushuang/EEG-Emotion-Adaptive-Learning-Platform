from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


capture = load_module(
    "capture_mindwave_csv",
    ROOT / "realtime_inference/tools/capture_mindwave_csv.py",
)
validator = load_module(
    "validate_experiment_run",
    ROOT / "scripts/validate_experiment_run.py",
)
marker = load_module(
    "mark_five_stage_experiment",
    ROOT / "scripts/mark_five_stage_experiment.py",
)


def test_control_event_is_scoped_to_run_id(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "event"])
        writer.writeheader()
        writer.writerow({"run_id": "other", "event": "experiment_end"})
        writer.writerow({"run_id": "wanted", "event": "experiment_end"})
    assert capture.control_has_experiment_end(path, "wanted")
    assert not capture.control_has_experiment_end(path, "absent")


def test_validator_rejects_missing_stage_eeg(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.csv"
    with capture_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["packet_received_unix"])
        writer.writeheader()
        writer.writerow({"packet_received_unix": 100})
        writer.writerow({"packet_received_unix": 150})
    events_path = tmp_path / "events.csv"
    with events_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp_unix", "event", "stage"],
        )
        writer.writeheader()
        writer.writerow(
            {"timestamp_unix": 110, "event": "stage_start", "stage": "rest_baseline"}
        )
        writer.writerow(
            {
                "timestamp_unix": 140,
                "event": "stage_end_self_report",
                "stage": "rest_baseline",
            }
        )
        writer.writerow({"timestamp_unix": 200, "event": "experiment_end", "stage": ""})
    report = validator.validate(capture_path, events_path)
    assert report["run_status"] == "incomplete"
    assert report["stage_coverage"]["rest_baseline"]["state"] == "complete"
    assert report["stage_coverage"]["recovery"]["state"] == "missing_events"


def test_validator_requires_event_driven_termination(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture.csv"
    with capture_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["packet_received_unix"])
        writer.writeheader()
        writer.writerow({"packet_received_unix": 90})
        writer.writerow({"packet_received_unix": 300})
    events_path = tmp_path / "events.csv"
    with events_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["timestamp_unix", "event", "stage"]
        )
        writer.writeheader()
        for index, stage in enumerate(validator.EXPECTED_STAGES):
            writer.writerow(
                {
                    "timestamp_unix": 100 + index * 20,
                    "event": "stage_start",
                    "stage": stage,
                }
            )
            writer.writerow(
                {
                    "timestamp_unix": 110 + index * 20,
                    "event": "stage_end_self_report",
                    "stage": stage,
                }
            )
        writer.writerow(
            {"timestamp_unix": 250, "event": "experiment_end", "stage": ""}
        )
    diagnostics = tmp_path / "diagnostics.json"
    diagnostics.write_text(
        '{"stop_reason":"safety_timeout_reached","failure":null}',
        encoding="utf-8",
    )
    report = validator.validate(capture_path, events_path, diagnostics)
    assert report["run_status"] == "incomplete"
    assert report["termination_reason"] == "safety_timeout_reached"


def test_capture_failure_is_not_silent(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    status.write_text(
        json.dumps(
            {
                "run_id": "failed-run",
                "state": "capture_failed",
                "failure": "OSError: disk full",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="disk full"):
        marker.wait_for_capture_saved(status, "failed-run", 0.2)


def test_missing_capture_ack_times_out(tmp_path: Path) -> None:
    with pytest.raises(TimeoutError, match="CAPTURE_SAVED"):
        marker.wait_for_capture_saved(tmp_path / "missing.json", "run", 0.05)
