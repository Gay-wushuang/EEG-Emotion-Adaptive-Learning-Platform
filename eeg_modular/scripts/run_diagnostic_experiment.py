"""Run capture, protocol, save handshake, and coverage validation as one unit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from mark_five_stage_experiment import CHANGED_ORDER_STAGES, ORIGINAL_STAGES


def read_status(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def wait_ready(path: Path, run_id: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = read_status(path)
        if status.get("run_id") == run_id and status.get("state") == "capture_ready":
            print("Capture ready")
            return
        if status.get("run_id") == run_id and status.get("state") == "capture_failed":
            raise RuntimeError(status.get("failure", "Capture failed"))
        time.sleep(0.25)
    raise TimeoutError("Capture did not become ready")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", choices=("original", "changed"), default="original")
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--smoke-stage-seconds", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir", type=Path, default=Path("realtime_inference/captures"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=13854)
    parser.add_argument("--post-end-buffer", type=float, default=30.0)
    parser.add_argument("--self-report-allowance", type=float, default=120.0)
    parser.add_argument("--insurance-seconds", type=float, default=600.0)
    parser.add_argument("--wait-for-raw-timeout", type=float, default=30.0)
    args = parser.parse_args()

    stages = ORIGINAL_STAGES if args.order == "original" else CHANGED_ORDER_STAGES
    protocol_seconds = (
        len(stages) * args.smoke_stage_seconds
        if args.smoke_stage_seconds is not None
        else sum(max(1, round(s[1] * args.duration_scale)) for s in stages)
    )
    max_duration = (
        protocol_seconds
        + len(stages) * args.self_report_allowance
        + args.post_end_buffer
        + args.insurance_seconds
    )
    run_id = args.run_id or f"diagnostic_{args.order}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = args.output_dir / run_id
    capture = base.with_suffix(".csv")
    events = Path(f"{base}_events.csv")
    status = Path(f"{base}.status.json")
    diagnostics = capture.with_suffix(".csv.diagnostics.json")
    coverage = Path(f"{base}.coverage.json")

    capture_command = [
        sys.executable,
        "realtime_inference/tools/capture_mindwave_csv.py",
        "--host", args.host,
        "--port", str(args.port),
        "--wait-for-raw-timeout", str(args.wait_for_raw_timeout),
        "--output", str(capture),
        "--control-file", str(events),
        "--run-id", run_id,
        "--post-end-buffer", str(args.post_end_buffer),
        "--max-duration", str(max_duration),
        "--status-file", str(status),
    ]
    print(f"run_id={run_id}")
    print(f"dynamic_max_duration_seconds={max_duration:.1f}")
    capture_process = subprocess.Popen(capture_command)
    try:
        wait_ready(status, run_id, args.wait_for_raw_timeout + 10)
        marker_command = [
            sys.executable,
            "scripts/mark_five_stage_experiment.py",
            "--order", args.order,
            "--output", str(events),
            "--run-id", run_id,
            "--capture-status-file", str(status),
            "--duration-scale", str(args.duration_scale),
        ]
        if args.smoke_stage_seconds is not None:
            marker_command.extend(
                ["--smoke-stage-seconds", str(args.smoke_stage_seconds)]
            )
        marker_result = subprocess.run(marker_command, check=False)
        if marker_result.returncode != 0:
            raise RuntimeError(
                f"Experiment program exited with code {marker_result.returncode}"
            )
        capture_code = capture_process.wait(timeout=args.post_end_buffer + 90)
        if capture_code != 0:
            raise RuntimeError(f"Capture exited with code {capture_code}")
        validation = subprocess.run(
            [
                sys.executable,
                "scripts/validate_experiment_run.py",
                "--capture", str(capture),
                "--events", str(events),
                "--diagnostics", str(diagnostics),
                "--output", str(coverage),
            ],
            check=False,
        )
        if validation.returncode != 0:
            raise RuntimeError("Coverage validation: incomplete")
        print("Coverage validation: complete")
    finally:
        if capture_process.poll() is None:
            capture_process.terminate()


if __name__ == "__main__":
    main()
