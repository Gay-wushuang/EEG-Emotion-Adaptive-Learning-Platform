"""采样率诊断脚本 —— 只报告事实，不修改任何数据。

诊断目标：
    验证 MindWave 输出的 rawEeg 实际采样率是否符合 512 Hz 契约。

诊断方法：
    从 ThinkGear 协议的 rawEeg 包中统计每秒的样本数，
    给出 min / max / mean / median，并分类报告。

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
from dataclasses import dataclass
from typing import List, Optional

# ── 只读常量（不修改） ──
TARGET_SAMPLE_RATE = 512
RATE_TOLERANCE_HZ = 20   # 允许 ±20 Hz 浮动
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 13854
DEFAULT_DURATION = 30.0   # 采样秒数


@dataclass
class SampleRateReport:
    """采样率诊断报告（只读）。"""
    duration_seconds: float
    total_raw_packets: int
    samples_per_second: List[float]       # 每秒样本数列表
    min_rate: float
    max_rate: float
    mean_rate: float
    median_rate: float
    std_rate: float
    below_min_seconds: float              # 每秒样本数低于阈值的总时长
    warnings: List[str]                   # 警告消息（只读）
    passed_threshold: bool                # 是否符合 512 ± tolerance

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  MindWave 采样率诊断报告（只读）",
            "=" * 60,
            f"  目标采样率: {TARGET_SAMPLE_RATE} Hz",
            f"  诊断时长: {self.duration_seconds:.1f} 秒",
            f"  总 rawEeg 包数: {self.total_raw_packets}",
            f"  每秒样本数统计:",
            f"    min    = {self.min_rate:.1f} Hz",
            f"    max    = {self.max_rate:.1f} Hz",
            f"    mean   = {self.mean_rate:.1f} Hz",
            f"    median = {self.median_rate:.1f} Hz",
            f"    std    = {self.std_rate:.1f} Hz",
            f"  阈值合格: {'是' if self.passed_threshold else '否'}",
        ]
        if self.below_min_seconds > 0:
            lines.append(f"  低于阈值时长: {self.below_min_seconds:.1f} 秒")
        if self.warnings:
            lines.append("  警告:")
            for w in self.warnings:
                lines.append(f"    ⚠ {w}")
        lines.append("=" * 60)
        return "\n".join(lines)


def classify_rate(rate: float) -> str:
    """将采样率分类（仅报告，不改变行为）。"""
    if 220 <= rate <= 300:
        return f"WARNING: approximately 256 Hz (实际 {rate:.0f} Hz)"
    elif 480 <= rate <= 540:
        return f"OK: approximately 512 Hz (实际 {rate:.0f} Hz)"
    else:
        return f"WARNING: unexpected raw sample rate (实际 {rate:.0f} Hz)"


def diagnose(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    duration: float = DEFAULT_DURATION,
) -> Optional[SampleRateReport]:
    """连接 ThinkGear Connector 并统计采样率（只读）。

    Returns:
        SampleRateReport 或 None（连接失败时返回 None）
    """
    # 连接 ThinkGear Connector
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

    print(f"[INFO] 已连接 ThinkGear Connector，开始诊断 {duration:.1f} 秒...")

    # 统计每秒样本数
    per_second_counts: deque[int] = deque()
    current_second = int(time.time())
    current_count = 0
    total_raw = 0
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            try:
                chunk = sock.recv(8192)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                text = text.replace("\n", "\r")
                for line in text.split("\r"):
                    if not line.strip():
                        continue
                    try:
                        packet = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if "rawEeg" not in packet:
                        continue
                    total_raw += 1
                    current_count += 1

                    # 若跨秒，保存当前秒计数
                    now_sec = int(time.time())
                    if now_sec != current_second:
                        per_second_counts.append(current_count)
                        current_count = 0
                        current_second = now_sec
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

    if not per_second_counts:
        print("[ERROR] 未收到任何 rawEeg 数据。请检查设备连接。")
        return None

    counts_list = list(per_second_counts)
    mean_rate = statistics.mean(counts_list)
    median_rate = statistics.median(counts_list)
    min_rate = min(counts_list)
    max_rate = max(counts_list)
    std_rate = statistics.stdev(counts_list) if len(counts_list) > 1 else 0.0

    # 分类报告
    warnings: List[str] = []
    for i, rate in enumerate(counts_list):
        if abs(rate - TARGET_SAMPLE_RATE) > RATE_TOLERANCE_HZ:
            warnings.append(
                f"第 {i+1} 秒: {classify_rate(rate)}"
            )

    passed = abs(mean_rate - TARGET_SAMPLE_RATE) <= RATE_TOLERANCE_HZ

    below_min = sum(
        1 for r in counts_list if r < TARGET_SAMPLE_RATE - RATE_TOLERANCE_HZ
    )

    report = SampleRateReport(
        duration_seconds=time.time() - start_time,
        total_raw_packets=total_raw,
        samples_per_second=counts_list,
        min_rate=min_rate,
        max_rate=max_rate,
        mean_rate=mean_rate,
        median_rate=median_rate,
        std_rate=std_rate,
        below_min_seconds=float(below_min),
        warnings=warnings,
        passed_threshold=passed,
    )

    print(str(report))

    # 若不通过，给出明确的只读建议
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
        help=f"诊断时长（秒，默认 {DEFAULT_DURATION}）",
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
