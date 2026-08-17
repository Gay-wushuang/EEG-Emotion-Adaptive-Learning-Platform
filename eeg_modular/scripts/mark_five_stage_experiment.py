"""Interactive event and self-report logger for the seven-stage experiment."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path

try:
    import winsound
except ImportError:  # pragma: no cover - non-Windows development environments
    winsound = None


ORIGINAL_STAGES = (
    ("rest_baseline", 120, "睁眼放松"),
    ("positive_task", 240, "简单题＋即时正反馈"),
    ("washout_1", 60, "放松"),
    ("normal_task", 240, "中等难度学习"),
    ("washout_2", 60, "放松"),
    ("frustration_task", 240, "高难限时任务"),
    ("recovery", 120, "停止任务、自然休息"),
)
CHANGED_ORDER_STAGES = (
    ("rest_baseline", 120, "睁眼放松"),
    ("normal_task", 240, "中等难度学习"),
    ("washout_1", 60, "放松"),
    ("positive_task", 240, "简单题＋即时正反馈"),
    ("washout_2", 60, "放松"),
    ("frustration_task", 240, "高难限时任务"),
    ("recovery", 120, "停止任务、自然休息"),
)


def now() -> tuple[float, str]:
    current = dt.datetime.now(dt.timezone.utc)
    return current.timestamp(), current.isoformat()


def rating(name: str) -> int:
    while True:
        value = input(f"{name}（1-5）：").strip()
        if value in {"1", "2", "3", "4", "5"}:
            return int(value)


def notify_stage_complete() -> None:
    if winsound is not None:
        for _ in range(3):
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            time.sleep(0.25)


def wait_for_capture_saved(
    status_file: Path, run_id: str, timeout: float
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status = json.loads(status_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            time.sleep(0.25)
            continue
        if status.get("run_id") == run_id and status.get("state") == "capture_saved":
            print("CAPTURE_SAVED acknowledged")
            return status
        if status.get("run_id") == run_id and status.get("state") == "capture_failed":
            raise RuntimeError(
                f"Capture failed: {status.get('failure', 'unknown error')}"
            )
        time.sleep(0.25)
    raise TimeoutError(f"CAPTURE_SAVED not received within {timeout:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", choices=("original", "changed"), default="original")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("realtime_inference/captures/diagnostic_event_markers.csv"),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--capture-status-file", type=Path)
    parser.add_argument("--capture-ack-timeout", type=float, default=90.0)
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=1.0,
        help="Smoke test only: multiply all stage durations by this value",
    )
    parser.add_argument(
        "--smoke-stage-seconds",
        type=int,
        help="Use the same protocol path with every stage shortened to this duration",
    )
    args = parser.parse_args()
    if bool(args.run_id) != bool(args.capture_status_file):
        parser.error("--run-id and --capture-status-file must be supplied together")
    if args.duration_scale <= 0:
        parser.error("--duration-scale must be positive")
    if args.smoke_stage_seconds is not None and args.smoke_stage_seconds <= 0:
        parser.error("--smoke-stage-seconds must be positive")

    stages = ORIGINAL_STAGES if args.order == "original" else CHANGED_ORDER_STAGES
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "timestamp_unix",
        "timestamp_iso_utc",
        "event",
        "stage",
        "planned_seconds",
        "valence",
        "stress",
        "attention",
        "fatigue",
        "self_label",
        "artifact_note",
    ]
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        input("确认Raw采集稳定后，按Enter开始实验：")
        unix, iso = now()
        writer.writerow(
            {
                "run_id": args.run_id or "",
                "timestamp_unix": unix,
                "timestamp_iso_utc": iso,
                "event": "experiment_start",
            }
        )
        stream.flush()

        for stage, base_duration, description in stages:
            duration = (
                args.smoke_stage_seconds
                if args.smoke_stage_seconds is not None
                else max(1, round(base_duration * args.duration_scale))
            )
            unix, iso = now()
            writer.writerow(
                {
                    "run_id": args.run_id or "",
                    "timestamp_unix": unix,
                    "timestamp_iso_utc": iso,
                    "event": "stage_start",
                    "stage": stage,
                    "planned_seconds": duration,
                }
            )
            stream.flush()
            print(f"\n开始：{description}，计划{duration}秒。")
            started = time.monotonic()
            while True:
                remaining = duration - (time.monotonic() - started)
                if remaining <= 0:
                    break
                print(f"\r剩余 {int(remaining):3d} 秒", end="", flush=True)
                time.sleep(min(1.0, remaining))
            print("\r阶段计时完成。       ")
            unix, iso = now()
            writer.writerow(
                {
                    "run_id": args.run_id or "",
                    "timestamp_unix": unix,
                    "timestamp_iso_utc": iso,
                    "event": "stage_task_end",
                    "stage": stage,
                    "planned_seconds": duration,
                }
            )
            stream.flush()
            notify_stage_complete()
            artifact = input(
                "是否有明显眨眼、动作或佩戴不适？无则回车："
            ).strip()
            self_label = input(
                "最接近 positive / neutral / negative："
            ).strip().lower()
            if self_label not in {"positive", "neutral", "negative"}:
                self_label = "unspecified"
            report = {
                "valence": rating("愉悦度"),
                "stress": rating("压力"),
                "attention": rating("专注度"),
                "fatigue": rating("疲劳度"),
            }
            unix, iso = now()
            writer.writerow(
                {
                    "run_id": args.run_id or "",
                    "timestamp_unix": unix,
                    "timestamp_iso_utc": iso,
                    "event": "stage_end_self_report",
                    "stage": stage,
                    "planned_seconds": duration,
                    "self_label": self_label,
                    "artifact_note": artifact,
                    **report,
                }
            )
            stream.flush()

        unix, iso = now()
        writer.writerow(
            {
                "run_id": args.run_id or "",
                "timestamp_unix": unix,
                "timestamp_iso_utc": iso,
                "event": "experiment_end",
            }
        )
        stream.flush()
        if args.capture_status_file:
            print("已发送EXPERIMENT_END，等待采集器保存确认……")
            wait_for_capture_saved(
                args.capture_status_file,
                args.run_id,
                args.capture_ack_timeout,
            )
    print(f"事件与自评已保存：{args.output.resolve()}")


if __name__ == "__main__":
    main()
