"""Capture timestamped MindWave packets from ThinkGear Connector into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def control_has_experiment_end(path: Path, run_id: str) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return any(
                row.get("event") == "experiment_end"
                and row.get("run_id") == run_id
                for row in csv.DictReader(handle)
            )
    except (OSError, csv.Error):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=13854)
    parser.add_argument("--duration", type=float, default=10.0, help="Capture duration in seconds")
    parser.add_argument(
        "--wait-for-raw-timeout",
        type=float,
        default=30.0,
        help="Maximum seconds to wait for the first Raw EEG sample",
    )
    parser.add_argument("--sample-rate", type=float, default=512.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-file", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--post-end-buffer", type=float, default=30.0)
    parser.add_argument("--max-duration", type=float)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        help="Defaults to <output>.diagnostics.json and is written even when Raw EEG is absent",
    )
    return parser.parse_args()


def iso_utc(unix_seconds: float) -> str:
    return datetime.fromtimestamp(unix_seconds, timezone.utc).isoformat(timespec="milliseconds")


def main() -> None:
    args = parse_args()
    if args.control_file and (not args.run_id or args.max_duration is None):
        raise ValueError("--control-file requires --run-id and --max-duration")
    if args.post_end_buffer < 0:
        raise ValueError("--post-end-buffer must be non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = args.output.with_suffix(args.output.suffix + ".partial")
    diagnostics_output = args.diagnostics_output or args.output.with_suffix(
        args.output.suffix + ".diagnostics.json"
    )
    diagnostics_output.parent.mkdir(parents=True, exist_ok=True)

    attention = ""
    meditation = ""
    poor_signal = ""
    attention_updated = False
    meditation_updated = False
    poor_signal_updated = False
    sample_index = 0
    first_sample_unix: float | None = None
    connected_unix: float | None = None
    first_raw_received_unix: float | None = None
    last_message_unix: float | None = None
    last_raw_received_unix: float | None = None
    max_raw_gap_seconds = 0.0
    max_active_raw_gap_seconds = 0.0
    started = time.monotonic()
    active_started: float | None = None
    experiment_end_seen: float | None = None
    experiment_end_detected_unix: float | None = None
    stop_reason: str | None = None
    raw_counts_per_second: dict[int, int] = {}
    counters = {
        "tcp_connected": False,
        "configuration_sent": False,
        "total_bytes": 0,
        "json_messages": 0,
        "raw_eeg_messages": 0,
        "esense_messages": 0,
        "poor_signal_messages": 0,
        "blink_strength_messages": 0,
        "json_parse_failures": 0,
        "socket_timeouts": 0,
        "socket_eof": 0,
        "max_buffer_chars": 0,
    }
    failure: str | None = None

    try:
        with socket.create_connection((args.host, args.port), timeout=5.0) as sock:
            counters["tcp_connected"] = True
            connected_unix = time.time()
            request = json.dumps(
                {"enableRawOutput": True, "format": "Json"}, separators=(",", ":")
            ) + "\r"
            sock.sendall(request.encode("utf-8"))
            counters["configuration_sent"] = True
            sock.settimeout(0.25)
            buffer = ""

            with partial_output.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "timestamp_unix",
                        "timestamp_iso_utc",
                        "packet_received_unix",
                        "sample_index",
                        "raw_eeg",
                        "attention",
                        "meditation",
                        "poor_signal",
                        "attention_updated",
                        "meditation_updated",
                        "poor_signal_updated",
                    ],
                )
                writer.writeheader()

                while True:
                    now_monotonic = time.monotonic()
                    if active_started is None:
                        if now_monotonic - started >= args.wait_for_raw_timeout:
                            raise TimeoutError(
                                f"No Raw EEG within {args.wait_for_raw_timeout:.1f}s"
                            )
                    elif args.control_file:
                        if (
                            experiment_end_seen is None
                            and control_has_experiment_end(args.control_file, args.run_id)
                        ):
                            experiment_end_seen = now_monotonic
                            experiment_end_detected_unix = time.time()
                            print("EXPERIMENT_END received", flush=True)
                        if (
                            experiment_end_seen is not None
                            and now_monotonic - experiment_end_seen >= args.post_end_buffer
                        ):
                            print("Tail buffer completed", flush=True)
                            stop_reason = "experiment_end_received"
                            break
                        if now_monotonic - active_started >= args.max_duration:
                            stop_reason = "safety_timeout_reached"
                            break
                    elif now_monotonic - active_started >= args.duration:
                        stop_reason = "fixed_duration_reached"
                        break
                    try:
                        chunk = sock.recv(8192)
                    except socket.timeout:
                        counters["socket_timeouts"] += 1
                        continue
                    if not chunk:
                        counters["socket_eof"] += 1
                        stop_reason = "socket_eof"
                        break
                    counters["total_bytes"] += len(chunk)
                    buffer += chunk.decode("utf-8", errors="replace")
                    buffer = buffer.replace("\r\n", "\n").replace("\r", "\n")
                    counters["max_buffer_chars"] = max(counters["max_buffer_chars"], len(buffer))

                    while "\n" in buffer:
                        packet_text, buffer = buffer.split("\n", 1)
                        if not packet_text.strip():
                            continue
                        try:
                            packet = json.loads(packet_text)
                        except json.JSONDecodeError:
                            counters["json_parse_failures"] += 1
                            continue
                        counters["json_messages"] += 1
                        last_message_unix = time.time()

                        if "poorSignalLevel" in packet:
                            counters["poor_signal_messages"] += 1
                            poor_signal = int(packet["poorSignalLevel"])
                            poor_signal_updated = True
                        esense = packet.get("eSense") or {}
                        if esense or "attention" in packet or "meditation" in packet:
                            counters["esense_messages"] += 1
                        if "blinkStrength" in packet:
                            counters["blink_strength_messages"] += 1
                        if "attention" in esense:
                            attention = int(esense["attention"])
                            attention_updated = True
                        if "meditation" in esense:
                            meditation = int(esense["meditation"])
                            meditation_updated = True
                        if "attention" in packet:
                            attention = int(packet["attention"])
                            attention_updated = True
                        if "meditation" in packet:
                            meditation = int(packet["meditation"])
                            meditation_updated = True

                        if "rawEeg" not in packet:
                            continue
                        counters["raw_eeg_messages"] += 1

                        if first_sample_unix is None:
                            first_sample_unix = time.time()
                            active_started = time.monotonic()
                            if args.status_file:
                                write_json_atomic(
                                    args.status_file,
                                    {
                                        "run_id": args.run_id,
                                        "state": "capture_ready",
                                        "timestamp_unix": first_sample_unix,
                                        "output": str(args.output),
                                    },
                                )
                        packet_received_unix = time.time()
                        if first_raw_received_unix is None:
                            first_raw_received_unix = packet_received_unix
                            if connected_unix is not None:
                                max_raw_gap_seconds = max(
                                    max_raw_gap_seconds,
                                    first_raw_received_unix - connected_unix,
                                )
                        if last_raw_received_unix is not None:
                            active_gap = packet_received_unix - last_raw_received_unix
                            max_active_raw_gap_seconds = max(
                                max_active_raw_gap_seconds, active_gap
                            )
                            max_raw_gap_seconds = max(
                                max_raw_gap_seconds,
                                active_gap,
                            )
                        last_raw_received_unix = packet_received_unix
                        active_second = int(time.monotonic() - active_started)
                        raw_counts_per_second[active_second] = (
                            raw_counts_per_second.get(active_second, 0) + 1
                        )
                        sample_time = first_sample_unix + sample_index / args.sample_rate
                        writer.writerow(
                            {
                                "timestamp_unix": f"{sample_time:.6f}",
                                "timestamp_iso_utc": iso_utc(sample_time),
                                "packet_received_unix": f"{packet_received_unix:.6f}",
                                "sample_index": sample_index,
                                "raw_eeg": int(packet["rawEeg"]),
                                "attention": attention,
                                "meditation": meditation,
                                "poor_signal": poor_signal,
                                "attention_updated": int(attention_updated),
                                "meditation_updated": int(meditation_updated),
                                "poor_signal_updated": int(poor_signal_updated),
                            }
                        )
                        sample_index += 1
                        attention_updated = False
                        meditation_updated = False
                        poor_signal_updated = False
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        ended_unix = time.time()
        if last_raw_received_unix is not None:
            max_raw_gap_seconds = max(max_raw_gap_seconds, ended_unix - last_raw_received_unix)
        diagnostics = {
            **counters,
            "requested_duration_seconds": args.duration,
            "event_driven": bool(args.control_file),
            "run_id": args.run_id,
            "post_end_buffer_seconds": args.post_end_buffer,
            "max_duration_seconds": args.max_duration,
            "experiment_end_received": experiment_end_seen is not None,
            "experiment_end_detected_unix": experiment_end_detected_unix,
            "stop_reason": stop_reason,
            "wait_for_raw_timeout_seconds": args.wait_for_raw_timeout,
            "wall_elapsed_seconds": time.monotonic() - started,
            "active_capture_elapsed_seconds": (
                time.monotonic() - active_started if active_started is not None else None
            ),
            "expected_sample_rate_hz": args.sample_rate,
            "observed_raw_samples": sample_index,
            "observed_average_raw_rate_hz": (
                sample_index / max(time.monotonic() - started, 1e-9)
            ),
            "observed_active_window_raw_rate_hz": (
                sample_index / max(time.monotonic() - active_started, 1e-9)
                if active_started is not None
                else None
            ),
            "first_raw_delay_seconds": (
                first_raw_received_unix - connected_unix
                if first_raw_received_unix is not None and connected_unix is not None
                else None
            ),
            "raw_receive_span_seconds": (
                last_raw_received_unix - first_raw_received_unix
                if last_raw_received_unix is not None and first_raw_received_unix is not None
                else None
            ),
            "raw_counts_per_active_second": {
                str(second + 1): raw_counts_per_second.get(second, 0)
                for second in range(
                    max(
                        int(
                            time.monotonic() - active_started
                            if active_started is not None
                            else args.duration
                        ),
                        max(raw_counts_per_second, default=-1) + 1,
                    )
                )
            },
            "max_raw_gap_seconds": max_raw_gap_seconds,
            "max_active_raw_gap_seconds": max_active_raw_gap_seconds,
            "last_message_unix": last_message_unix,
            "last_message_iso_utc": iso_utc(last_message_unix) if last_message_unix else None,
            "failure": failure,
        }
        write_json_atomic(diagnostics_output, diagnostics)
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2))

    if failure:
        if args.status_file:
            write_json_atomic(
                args.status_file,
                {
                    "run_id": args.run_id,
                    "state": "capture_failed",
                    "timestamp_unix": time.time(),
                    "failure": failure,
                    "stop_reason": stop_reason,
                },
            )
        partial_output.unlink(missing_ok=True)
        raise RuntimeError(failure)
    if sample_index == 0:
        partial_output.unlink(missing_ok=True)
        raise RuntimeError("No rawEeg samples received; see diagnostics JSON.")
    partial_output.replace(args.output)
    if args.status_file:
        write_json_atomic(
            args.status_file,
            {
                "run_id": args.run_id,
                "state": "capture_saved",
                "timestamp_unix": time.time(),
                "output": str(args.output),
                "observed_raw_samples": sample_index,
                "stop_reason": stop_reason,
            },
        )
    print("Capture saved", flush=True)
    print(f"Saved {sample_index} raw samples to {args.output}")


if __name__ == "__main__":
    main()
