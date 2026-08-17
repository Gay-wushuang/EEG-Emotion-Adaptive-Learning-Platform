"""Validate EEG coverage and event completeness for a diagnostic experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_STAGES = (
    "rest_baseline",
    "positive_task",
    "washout_1",
    "normal_task",
    "washout_2",
    "frustration_task",
    "recovery",
)


def capture_bounds(path: Path) -> tuple[float, float, int]:
    first = last = None
    count = 0
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            value = float(row["packet_received_unix"])
            first = value if first is None else first
            last = value
            count += 1
    if first is None or last is None:
        raise ValueError("Capture contains no Raw samples")
    return first, last, count


def validate(
    capture: Path, events: Path, diagnostics: Path | None = None
) -> dict:
    eeg_start, eeg_end, raw_samples = capture_bounds(capture)
    with events.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    starts = {
        row["stage"]: float(row["timestamp_unix"])
        for row in rows
        if row.get("event") == "stage_start"
    }
    task_ends = {
        row["stage"]: float(row["timestamp_unix"])
        for row in rows
        if row.get("event") == "stage_task_end"
    }
    report_ends = {
        row["stage"]: float(row["timestamp_unix"])
        for row in rows
        if row.get("event") == "stage_end_self_report"
    }
    experiment_end = next(
        (
            float(row["timestamp_unix"])
            for row in rows
            if row.get("event") == "experiment_end"
        ),
        None,
    )
    coverage = {}
    for stage in EXPECTED_STAGES:
        start = starts.get(stage)
        end = task_ends.get(stage)
        if end is None and start is not None:
            matching = next(
                (
                    row for row in rows
                    if row.get("event") == "stage_start" and row.get("stage") == stage
                ),
                None,
            )
            if matching and matching.get("planned_seconds"):
                end = start + float(matching["planned_seconds"])
        report_end = report_ends.get(stage)
        if end is None:
            end = report_end
        if start is None or end is None:
            state = "missing_events"
        elif eeg_start <= start and eeg_end >= end:
            state = "complete"
        elif eeg_end >= start and eeg_start <= end:
            state = "partial"
        else:
            state = "missing_eeg"
        coverage[stage] = {
            "state": state,
            "start_unix": start,
            "end_unix": end,
            "self_report_end_unix": report_end,
        }
    complete = (
        experiment_end is not None
        and eeg_end >= experiment_end
        and all(item["state"] == "complete" for item in coverage.values())
    )
    diagnostic_data = {}
    if diagnostics and diagnostics.exists():
        diagnostic_data = json.loads(diagnostics.read_text(encoding="utf-8"))
        complete = complete and (
            diagnostic_data.get("stop_reason") == "experiment_end_received"
            and diagnostic_data.get("failure") is None
        )
    missing_stages = [
        stage for stage, item in coverage.items() if item["state"] == "missing_eeg"
    ]
    partially_covered_stages = [
        stage for stage, item in coverage.items() if item["state"] == "partial"
    ]
    missing_event_stages = [
        stage for stage, item in coverage.items() if item["state"] == "missing_events"
    ]
    return {
        "run_status": "complete" if complete else "incomplete",
        "termination_reason": diagnostic_data.get("stop_reason"),
        "raw_samples": raw_samples,
        "eeg_start_unix": eeg_start,
        "eeg_end_unix": eeg_end,
        "experiment_end_unix": experiment_end,
        "capture_saved_after_experiment_end": (
            experiment_end is not None and eeg_end >= experiment_end
        ),
        "tail_buffer_seconds": (
            eeg_end - experiment_end if experiment_end is not None else None
        ),
        "missing_stages": missing_stages,
        "partially_covered_stages": partially_covered_stages,
        "missing_event_stages": missing_event_stages,
        "stage_coverage": coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.capture, args.events, args.diagnostics)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    raise SystemExit(0 if report["run_status"] == "complete" else 2)


if __name__ == "__main__":
    main()
