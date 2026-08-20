"""采样率诊断脚本 —— 只报告事实，不修改任何数据。

诊断目标：
    验证 MindWave 输出的 rawEeg 实际采样率是否符合 512 Hz 契约。

诊断方法（Phase 2.2 修复）：
    1. TCP 建连后先等待第一条 rawEeg（记录 startup delay）
    2. 从第一条 rawEeg 开始计时 duration 秒（有效采集时长）
    3. 使用跨 TCP chunk 的 remainder buffer 进行可靠 JSON 解析
    4. 统计 rawEeg / eSense / eegPower / poorSignal 包数
    5. 计算有效 raw 采样率并分类报告

绝对禁止：
    - 重采样
    - 插值
    - 修改任何数据
    - 修改 LiveDataService
    - 修改 SAMPLE_RATE (512)
    - 修改 Production Baseline 模型、Scaler、canonical feature、bandpower

仅报告事实。若采样率偏离 512 Hz，只记录结果并退出，
不触发任何代码修改。

使用方法：
    E:\\anaconda3\\envs\\eegcnn\\python.exe smart_learning_app/diagnose_sample_rate.py
    E:\\anaconda3\\envs\\eegcnn\\python.exe smart_learning_app/diagnose_sample_rate.py --duration 30
    E:\\anaconda3\\envs\\eegcnn\\python.exe smart_learning_app/diagnose_sample_rate.py --host 127.0.0.1 --port 13854
"""

from __future__ import annotations

import json
import socket
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

# ── 只读常量（不修改） ──
TARGET_SAMPLE_RATE = 512
RATE_LOW = 450       # 450 Hz 以下判为低于 512
RATE_HIGH = 570      # 570 Hz 以上判为高于 512
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 13854
DEFAULT_DURATION = 30.0   # 有效 raw 采集秒数（不含 startup delay）
MIN_ACTIVE_SECONDS = 10.0  # 最少有效采集时长
STARTUP_TIMEOUT = 15.0     # 等待第一条 rawEeg 的最大秒数


@dataclass
class SampleRateReport:
    """采样率诊断报告（只读）。"""
    startup_delay_seconds: float            # TCP 建连到第一条 rawEeg 的等待时间
    active_duration_seconds: float          # 有效 raw 采集时长（不含 startup delay）
    total_raw_packets: int                  # rawEeg 总包数
    raw_count: int                          # rawEeg 有效计数
    active_raw_rate_hz: float               # 有效 raw 采样率
    esense_count: int                       # eSense 包计数
    eegpower_count: int                     # eegPower 包计数
    poorsignal_count: int                   # poorSignal 包计数
    samples_per_second: List[float] = field(default_factory=list)  # 每秒 rawEeg 计数
    min_rate: float = 0.0
    max_rate: float = 0.0
    mean_rate: float = 0.0
    median_rate: float = 0.0
    std_rate: float = 0.0
    warnings: List[str] = field(default_factory=list)
    passed_threshold: bool = False

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  MindWave 采样率诊断报告（只读）",
            "=" * 60,
            f"  目标采样率: {TARGET_SAMPLE_RATE} Hz",
            f"  Startup Delay: {self.startup_delay_seconds:.1f} 秒",
            f"  有效采集时长: {self.active_duration_seconds:.1f} 秒",
            f"  rawEeg 包数: {self.raw_count}",
            f"  rawEeg 总包数: {self.total_raw_packets}",
            f"  eSense 包数: {self.esense_count}",
            f"  eegPower 包数: {self.eegpower_count}",
            f"  poorSignal 包数: {self.poorsignal_count}",
            f"  有效 raw 采样率: {self.active_raw_rate_hz:.1f} Hz",
        ]
        if self.samples_per_second:
            lines.append(f"  每秒 rawEeg 计数:")
            lines.append(f"    min    = {self.min_rate:.1f} Hz")
            lines.append(f"    max    = {self.max_rate:.1f} Hz")
            lines.append(f"    mean   = {self.mean_rate:.1f} Hz")
            lines.append(f"    median = {self.median_rate:.1f} Hz")
            lines.append(f"    std    = {self.std_rate:.1f} Hz")
        lines.append(f"  阈值合格: {'是' if self.passed_threshold else '否'}")
        if self.warnings:
            lines.append("  警告:")
            for w in self.warnings:
                lines.append(f"    ⚠ {w}")
        lines.append("=" * 60)
        return "\n".join(lines)


def classify_rate(rate: float) -> str:
    """将采样率分类（仅报告，不改变行为）。

    450~570 Hz: approximately 512 Hz
    220~300 Hz: approximately 256 Hz
    其他: unexpected raw sample rate
    """
    if RATE_LOW <= rate <= RATE_HIGH:
        return f"OK: approximately 512 Hz (实际 {rate:.0f} Hz)"
    elif 220 <= rate <= 300:
        return f"WARNING: approximately 256 Hz (实际 {rate:.0f} Hz)"
    else:
        return f"WARNING: unexpected raw sample rate (实际 {rate:.0f} Hz)"


def diagnose(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    duration: float = DEFAULT_DURATION,
) -> Optional[SampleRateReport]:
    """连接 ThinkGear Connector 并统计采样率（只读）。

    Phase 2.2 修复：
    - TCP 建连后先等待第一条 rawEeg，记录 startup delay
    - duration 表示有效 raw 采集时长，不含 startup delay
    - 使用跨 TCP chunk 的 remainder buffer 进行可靠 JSON 解析
    - 不足 MIN_ACTIVE_SECONDS 时报告样本不足

    Returns:
        SampleRateReport 或 None（连接失败时返回 None）
    """
    # ── 连接 ThinkGear Connector ──
    try:
        sock = socket.create_connection((host, port), timeout=5.0)
    except (socket.timeout, ConnectionError, OSError) as e:
        print(f"[ERROR] 无法连接 ThinkGear Connector ({host}:{port}): {e}")
        print("请确保 ThinkGear Connector 已启动并连接 MindWave 设备。")
        return None

    # 请求 raw 输出
    request = json.dumps({"enableRawOutput": True, "format": "Json"})
    sock.sendall((request + "\r").encode("utf-8"))
    sock.settimeout(0.5)

    print(f"[INFO] 已连接 ThinkGear Connector，等待第一条 rawEeg...")

    # ── Phase 1: 等待第一条 rawEeg（记录 startup delay）──
    remainder = ""
    first_raw_received = False
    first_raw_time: Optional[float] = None
    startup_start = time.time()
    startup_delay = 0.0

    while not first_raw_received:
        try:
            chunk = sock.recv(8192)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            text = text.replace("\n", "\r")
            remainder += text

            # 使用 "\r" 分割，保留最后可能不完整的一条
            while "\r" in remainder:
                line, remainder = remainder.split("\r", 1)
                if not line.strip():
                    continue
                try:
                    packet = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if "rawEeg" in packet:
                    first_raw_received = True
                    first_raw_time = time.time()
                    startup_delay = first_raw_time - startup_start
                    break
        except socket.timeout:
            pass

        if time.time() - startup_start > STARTUP_TIMEOUT:
            print(f"[ERROR] 等待第一条 rawEeg 超时 ({STARTUP_TIMEOUT:.0f} 秒)。")
            print("请检查设备是否已连接并正常输出数据。")
            sock.close()
            return None

    # 检查 TCP 是否在第一条 rawEeg 到来前提前关闭
    if not first_raw_received or first_raw_time is None:
        print("[ERROR] TCP 连接在第一条 rawEeg 到达前已关闭。")
        print("请检查 ThinkGear Connector 和设备连接状态。")
        try:
            sock.close()
        except Exception:
            pass
        return None

    print(f"[INFO] 第一条 rawEeg 到达 (startup delay: {startup_delay:.1f}s)")
    print(f"[INFO] 开始有效采集 {duration:.1f} 秒...")

    # ── Phase 2: 有效 raw 采集 ──
    per_second_counts: deque[int] = deque()
    current_second = int(first_raw_time)
    current_count = 0
    total_raw = 0
    raw_count = 0
    esense_count = 0
    eegpower_count = 0
    poorsignal_count = 0
    active_start = first_raw_time
    last_data_time = first_raw_time

    try:
        while time.time() - active_start < duration:
            try:
                chunk = sock.recv(8192)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                text = text.replace("\n", "\r")
                remainder += text

                # 使用 "\r" 分割，保留最后可能不完整的一条
                while "\r" in remainder:
                    line, remainder = remainder.split("\r", 1)
                    if not line.strip():
                        continue
                    try:
                        packet = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue

                    last_data_time = time.time()

                    if "rawEeg" in packet:
                        total_raw += 1
                        raw_count += 1
                        current_count += 1

                        # 跨秒保存
                        now_sec = int(time.time())
                        if now_sec != current_second:
                            per_second_counts.append(current_count)
                            current_count = 0
                            current_second = now_sec

                    if "eSense" in packet:
                        esense_count += 1
                    if "eegPower" in packet:
                        eegpower_count += 1
                    if "poorSignalLevel" in packet:
                        poorsignal_count += 1
            except socket.timeout:
                pass
    except KeyboardInterrupt:
        print("[INFO] 用户中断诊断。")
    finally:
        # 保存最后一秒的计数
        if current_count > 0:
            per_second_counts.append(current_count)
        try:
            sock.close()
        except Exception:
            pass

    active_duration = last_data_time - active_start

    # ── 检查有效采集时长 ──
    if not per_second_counts:
        print("[ERROR] 未收到任何 rawEeg 数据。请检查设备连接。")
        return None

    if active_duration < MIN_ACTIVE_SECONDS:
        print(f"[WARN] 有效采集时长仅 {active_duration:.1f}s，"
              f"不足 {MIN_ACTIVE_SECONDS:.0f}s，采样率可能不稳定。")

    # ── 统计 ──
    counts_list = [c for c in per_second_counts if c > 0]
    if not counts_list:
        print("[ERROR] 所有秒的 rawEeg 计数为 0。")
        return None

    mean_rate = statistics.mean(counts_list)
    median_rate = statistics.median(counts_list)
    min_rate = min(counts_list)
    max_rate = max(counts_list)
    std_rate = statistics.stdev(counts_list) if len(counts_list) > 1 else 0.0
    active_raw_rate_hz = raw_count / active_duration if active_duration > 0 else 0.0

    # 分类报告
    warnings: List[str] = []
    for i, rate in enumerate(counts_list):
        if rate < RATE_LOW or rate > RATE_HIGH:
            warnings.append(f"第 {i+1} 秒: {classify_rate(rate)}")

    # 最终通过标准：以 active_raw_rate_hz 为主判定值
    # active_raw_rate_hz = raw_count / active_duration，不受每秒 bucket 波动影响
    passed = RATE_LOW <= active_raw_rate_hz <= RATE_HIGH

    report = SampleRateReport(
        startup_delay_seconds=startup_delay,
        active_duration_seconds=active_duration,
        total_raw_packets=total_raw,
        raw_count=raw_count,
        active_raw_rate_hz=active_raw_rate_hz,
        esense_count=esense_count,
        eegpower_count=eegpower_count,
        poorsignal_count=poorsignal_count,
        samples_per_second=counts_list,
        min_rate=min_rate,
        max_rate=max_rate,
        mean_rate=mean_rate,
        median_rate=median_rate,
        std_rate=std_rate,
        warnings=warnings,
        passed_threshold=passed,
    )

    print(str(report))

    # 分类输出
    classification = classify_rate(active_raw_rate_hz)
    print(f"\n[分类] {classification}")

    if not passed:
        print("\n[建议] 采样率偏离 512 Hz 契约。")
        print("  本诊断脚本不会修改任何代码或数据。")
        print("  请将本报告提交给负责采集/模型兼容性的团队成员处理。")
        print("  绝对禁止: 修改 SAMPLE_RATE、修改模型、重采样、插值。")

    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="MindWave 采样率诊断（只读，不修改任何数据）"
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST,
        help=f"ThinkGear Connector 主机（默认 {DEFAULT_HOST}）",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"ThinkGear Connector 端口（默认 {DEFAULT_PORT}）",
    )
    parser.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION,
        help=f"有效 raw 采集时长（秒，不含 startup delay，默认 {DEFAULT_DURATION}）",
    )
    args = parser.parse_args()

    report = diagnose(
        host=args.host,
        port=args.port,
        duration=args.duration,
    )

    if report is None:
        sys.exit(1)

    # 退出码：通过为 0，不通过为 2
    sys.exit(0 if report.passed_threshold else 2)


if __name__ == "__main__":
    main()