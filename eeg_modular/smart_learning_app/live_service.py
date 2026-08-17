"""Real ThinkGear acquisition and Production Baseline v1 orchestration."""

from __future__ import annotations

import csv
import json
import queue
import socket
import threading
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Signal

from services.dashboard_state import DashboardState, MAX_POOR_SIGNAL
from smart_learning_app.inference_engine import ProductionInferenceEngine


SAMPLE_RATE = 512
WINDOW_SAMPLES = 30 * SAMPLE_RATE
INFERENCE_STEP_SAMPLES = 2 * SAMPLE_RATE


class SessionCsvWriter:
    """Write a replay-ready session CSV without blocking the Qt UI thread."""

    FIELDS = (
        "timestamp_unix", "signal_time_seconds", "raw_sample_index", "raw",
        "attention", "meditation", "poor_signal", "prob_positive",
        "prob_neutral", "prob_negative", "predicted_class", "confidence",
        "quality_level", "inference_index",
    )

    def __init__(self, path: Path):
        self.path = Path(path)
        self.error: str | None = None
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def append(self, rows: list[dict]) -> None:
        if rows:
            self._queue.put(rows)

    def close(self, timeout: float = 10.0) -> bool:
        self._queue.put(None)
        self._thread.join(timeout)
        if self._thread.is_alive():
            self.error = "会话CSV保存超时"
        return self.error is None

    def _run(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
                writer.writeheader()
                while True:
                    rows = self._queue.get()
                    if rows is None:
                        break
                    writer.writerows(rows)
                handle.flush()
        except Exception as exc:
            self.error = str(exc)


class ThinkGearLiveWorker(QThread):
    batch_ready = Signal(object)
    status_changed = Signal(object)
    window_ready = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, host: str = "127.0.0.1", port: int = 13854):
        super().__init__()
        self.host = host
        self.port = port
        self._running = False
        self._raw = deque(maxlen=WINDOW_SAMPLES)
        self._att = deque(maxlen=WINDOW_SAMPLES)
        self._med = deque(maxlen=WINDOW_SAMPLES)
        self._pending_raw: list[int] = []
        self._attention: float | None = None
        self._meditation: float | None = None
        self._poor_signal: int | None = None
        self._since_inference = 0
        self._raw_count = 0
        self._first_raw_monotonic: float | None = None
        self._last_raw_monotonic: float | None = None

    def stop(self) -> None:
        self._running = False
        self.wait(3000)

    def run(self) -> None:
        self._running = True
        while self._running:
            try:
                self._stream_once()
            except Exception as exc:
                self.error_occurred.emit(str(exc))
                self.status_changed.emit({
                    "connector_status": "offline",
                    "device_status": "offline",
                    "reason": "ThinkGear Connector 连接失败",
                })
                for _ in range(20):
                    if not self._running:
                        return
                    self.msleep(100)

    def _stream_once(self) -> None:
        self._reset_stream()
        self.status_changed.emit({
            "connector_status": "connecting",
            "device_status": "offline",
        })
        with socket.create_connection((self.host, self.port), timeout=5.0) as sock:
            sock.settimeout(0.2)
            request = {"enableRawOutput": True, "format": "Json"}
            sock.sendall((json.dumps(request) + "\r").encode("utf-8"))
            self.status_changed.emit({
                "connector_status": "online",
                "device_status": "waiting_raw",
            })
            buffer = ""
            last_emit = time.monotonic()
            while self._running:
                try:
                    chunk = sock.recv(8192)
                    if not chunk:
                        raise ConnectionError("ThinkGear Connector 已关闭连接")
                    buffer += chunk.decode("utf-8", errors="replace")
                    buffer = buffer.replace("\n", "\r")
                    while "\r" in buffer:
                        line, buffer = buffer.split("\r", 1)
                        if line.strip():
                            try:
                                self._consume_packet(json.loads(line))
                            except (json.JSONDecodeError, TypeError, ValueError):
                                # A malformed packet must not terminate acquisition.
                                continue
                except socket.timeout:
                    pass
                now = time.monotonic()
                if now - last_emit >= 0.1:
                    self._emit_batch(now)
                    last_emit = now

    def _reset_stream(self) -> None:
        self._raw.clear()
        self._att.clear()
        self._med.clear()
        self._pending_raw.clear()
        self._attention = None
        self._meditation = None
        self._poor_signal = None
        self._since_inference = 0
        self._raw_count = 0
        self._first_raw_monotonic = None
        self._last_raw_monotonic = None

    def _consume_packet(self, packet: dict) -> None:
        if "poorSignalLevel" in packet:
            self._poor_signal = int(packet["poorSignalLevel"])
        esense = packet.get("eSense")
        if isinstance(esense, dict):
            if "attention" in esense:
                self._attention = float(esense["attention"])
            if "meditation" in esense:
                self._meditation = float(esense["meditation"])
        if "rawEeg" not in packet:
            return
        value = int(packet["rawEeg"])
        now = time.monotonic()
        if self._first_raw_monotonic is None:
            self._first_raw_monotonic = now
            self.status_changed.emit({
                "connector_status": "online",
                "device_status": "online",
            })
        self._last_raw_monotonic = now
        self._raw_count += 1
        self._raw.append(value)
        self._att.append(np.nan if self._attention is None else self._attention)
        self._med.append(np.nan if self._meditation is None else self._meditation)
        self._pending_raw.append(value)
        self._since_inference += 1
        if len(self._raw) == WINDOW_SAMPLES and self._since_inference >= INFERENCE_STEP_SAMPLES:
            self._since_inference = 0
            self.window_ready.emit({
                "raw": np.asarray(self._raw, dtype=np.float64),
                "attention": np.asarray(self._att, dtype=np.float64),
                "meditation": np.asarray(self._med, dtype=np.float64),
                "poor_signal": self._poor_signal,
            })

    def _emit_batch(self, now: float) -> None:
        batch = self._pending_raw
        self._pending_raw = []
        elapsed = (
            now - self._first_raw_monotonic
            if self._first_raw_monotonic is not None else 0.0
        )
        self.batch_ready.emit({
            "raw": batch,
            "attention": self._attention,
            "meditation": self._meditation,
            "poor_signal": self._poor_signal,
            "raw_count": self._raw_count,
            "buffer_samples": len(self._raw),
            "sample_rate_hz": self._raw_count / elapsed if elapsed > 1.0 else None,
        })


class ProductionInferenceWorker(QThread):
    result_ready = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, package_dir: Path):
        super().__init__()
        self.package_dir = Path(package_dir)
        self._running = False
        self._queue: queue.Queue = queue.Queue(maxsize=1)

    def submit(self, window: dict) -> None:
        try:
            self._queue.put_nowait(window)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(window)

    def stop(self) -> None:
        self._running = False
        self.wait(5000)

    @staticmethod
    def _fill(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).copy()
        valid = np.isfinite(values)
        if not valid.any():
            raise ValueError("尚未收到 Attention/Meditation 数据")
        indices = np.arange(values.size)
        values[~valid] = np.interp(indices[~valid], indices[valid], values[valid])
        return values

    def run(self) -> None:
        self._running = True
        try:
            engine = ProductionInferenceEngine(self.package_dir)
        except Exception as exc:
            self.error_occurred.emit(f"生产模型自检失败：{exc}")
            return
        while self._running:
            try:
                window = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                result = engine.infer_window(
                    window["raw"], self._fill(window["attention"]),
                    self._fill(window["meditation"]),
                )
                self.result_ready.emit(result)
            except Exception as exc:
                self.error_occurred.emit(f"实时推理失败：{exc}")


class LiveDataService(QObject):
    """Main-thread adapter from real workers to DashboardState."""

    def __init__(self, state: DashboardState, package_dir: Path, parent=None):
        super().__init__(parent)
        self.state = state
        self.acquisition = ThinkGearLiveWorker()
        self.inference = ProductionInferenceWorker(package_dir)
        self.acquisition.status_changed.connect(self._on_status)
        self.acquisition.batch_ready.connect(self._on_batch)
        self.acquisition.window_ready.connect(self._on_window)
        self.acquisition.error_occurred.connect(self._on_error)
        self.inference.result_ready.connect(self._on_result)
        self.inference.error_occurred.connect(self._on_error)
        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._tick)
        self._last_tick = time.monotonic()
        self._session_running = False
        self._ewma: np.ndarray | None = None
        self.sessions_dir = Path(package_dir).parent / "data" / "sessions"
        self._session_writer: SessionCsvWriter | None = None
        self._session_csv_path: Path | None = None
        self._session_raw_origin = 0
        self._accepted_states = deque(maxlen=45)  # 90 s at a 2 s inference step
        self._inference_index = 0

    def start_streaming(self) -> None:
        if not self.inference.isRunning():
            self.inference.start()
        if not self.acquisition.isRunning():
            self.acquisition.start()
        self._timer.start()

    def stop_streaming(self) -> None:
        self._timer.stop()
        if self._session_writer is not None:
            self.end_session()
        self.acquisition.stop()
        self.inference.stop()

    def start_session(self) -> str:
        if self._session_writer is not None:
            self._session_writer.close()
        self._session_csv_path = self.sessions_dir / self.state.run_id / "session.csv"
        self._session_writer = SessionCsvWriter(self._session_csv_path)
        self._session_writer.start()
        self._session_raw_origin = int(self.state._raw_sample_count)
        self._accepted_states.clear()
        self._inference_index = 0
        self.state._session_active = True
        self._session_running = True
        return str(self._session_csv_path)

    def pause_session(self) -> None:
        self.state._session_active = False
        self._session_running = False

    def resume_session(self) -> None:
        self.state._session_active = True
        self._session_running = True

    def end_session(self) -> str | None:
        self.state._session_active = False
        self._session_running = False
        writer, self._session_writer = self._session_writer, None
        if writer is None:
            return None
        if not writer.close():
            self._on_error(f"会话CSV保存失败：{writer.error}")
            return None
        return str(writer.path)

    def _on_status(self, status: dict) -> None:
        self.state.mode = "live"
        self.state.connector_status = status["connector_status"]
        self.state.device_status = status["device_status"]
        if status["device_status"] != "online":
            self.state.poor_signal = None
            self.state.attention = None
            self.state.meditation = None
            self.state.warmup_progress = 0.0
            self.state._eeg_raw_buffer.clear()
            self.state.quality_level = "rejected"
            self.state.quality_reasons = [status.get("reason", "等待设备数据")]
            self._accepted_states.clear()
            self._ewma = None
            self.state.stable_state = None
        self.state.emit_update()

    def _on_batch(self, batch: dict) -> None:
        s = self.state
        for value in batch["raw"]:
            s._eeg_raw_buffer.append(value)
        s.attention = batch["attention"]
        s.meditation = batch["meditation"]
        if s.attention is not None:
            s._attention_history.append(s.attention)
        if s.meditation is not None:
            s._meditation_history.append(s.meditation)
        s.poor_signal = batch["poor_signal"]
        s.sample_rate_hz = SAMPLE_RATE if batch["raw_count"] else None
        s._raw_sample_count = int(batch["raw_count"])
        s.warmup_progress = min(1.0, batch["buffer_samples"] / WINDOW_SAMPLES)
        self._update_quality()
        self._record_batch(batch)
        s.emit_update()

    def _record_batch(self, batch: dict) -> None:
        if not self._session_running or self._session_writer is None or not batch["raw"]:
            return
        raw_values = batch["raw"]
        global_start = int(batch["raw_count"]) - len(raw_values)
        received_at = time.time()
        s = self.state
        rows = []
        for offset, value in enumerate(raw_values):
            global_index = global_start + offset
            sample_index = global_index - self._session_raw_origin
            if sample_index < 0:
                continue
            rows.append({
                "timestamp_unix": f"{received_at - (len(raw_values) - 1 - offset) / SAMPLE_RATE:.6f}",
                "signal_time_seconds": f"{sample_index / SAMPLE_RATE:.6f}",
                "raw_sample_index": sample_index,
                "raw": value,
                "attention": "" if s.attention is None else s.attention,
                "meditation": "" if s.meditation is None else s.meditation,
                "poor_signal": "" if s.poor_signal is None else s.poor_signal,
                "prob_positive": "" if s.prob_positive is None else s.prob_positive,
                "prob_neutral": "" if s.prob_neutral is None else s.prob_neutral,
                "prob_negative": "" if s.prob_negative is None else s.prob_negative,
                "predicted_class": s.predicted_state or "",
                "confidence": "" if s.confidence is None else s.confidence,
                "quality_level": s.quality_level,
                "inference_index": self._inference_index,
            })
        self._session_writer.append(rows)

    def _update_quality(self) -> None:
        s = self.state
        if s.device_status != "online" or s.poor_signal is None:
            s.quality_level, s.quality_reasons = "rejected", ["设备数据不可用"]
        elif s.poor_signal >= 100:
            s.quality_level, s.quality_reasons = "rejected", ["电极接触质量过低"]
        elif s.poor_signal >= MAX_POOR_SIGNAL:
            s.quality_level, s.quality_reasons = "warning", ["请调整电极接触"]
        elif not s.warmup_complete:
            s.quality_level, s.quality_reasons = "warning", ["正在填充30秒分析窗口"]
        else:
            s.quality_level, s.quality_reasons = "trusted", []

    def _on_window(self, window: dict) -> None:
        if self.state.quality_level != "rejected":
            self.inference.submit(window)

    def _on_result(self, result) -> None:
        s = self.state
        probs = np.asarray(result.probabilities, dtype=float)
        s.prob_positive, s.prob_neutral, s.prob_negative = probs.tolist()
        s.predicted_state = result.display_class
        s.confidence = result.confidence
        s._prob_history.append((time.time(), *probs.tolist()))
        self._ewma = probs if self._ewma is None else 0.2 * probs + 0.8 * self._ewma
        self._inference_index += 1
        if result.accepted:
            self._accepted_states.append(result.display_class)
            counts = Counter(self._accepted_states)
            highest = max(counts.values())
            tied = {name for name, count in counts.items() if count == highest}
            s.stable_state = next(
                name for name in reversed(self._accepted_states) if name in tied
            )
            s.feedback_text = self._feedback(s.stable_state)
        else:
            # Rejected predictions are not votes and do not erase prior valid evidence.
            s.stable_state = s.stable_state if self._accepted_states else None
            s.feedback_text = "当前状态置信度不足，继续观察后再提供学习建议。"
        s.emit_update()

    def _on_error(self, message: str) -> None:
        self.state.feedback_text = message
        self.state.emit_update()

    def _tick(self) -> None:
        now = time.monotonic()
        dt, self._last_tick = now - self._last_tick, now
        if self._session_running:
            self.state.session_seconds += dt
        self.state.emit_update()

    @staticmethod
    def _feedback(state: str) -> str:
        return {
            "positive": "当前状态较积极，可保持现有学习节奏。",
            "neutral": "当前状态相对平稳，建议继续当前任务并关注专注趋势。",
            "negative": "检测到负性状态倾向，建议降低任务强度并短暂休息。",
        }[state]
