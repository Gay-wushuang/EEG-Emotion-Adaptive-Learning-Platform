"""EEG采集后台线程。

预留 mock / tcp / serial 三种连接模式接口。
- mock 模式：生成模拟EEG数据，不假装设备已连接，connector_status=device_status=offline
- tcp/serial 模式：预留接口，界面显示"尚未接入"

通过信号 ``data_ready`` 向主线程推送数据快照，绝不阻塞UI线程。
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal


@dataclass
class AcquisitionConfig:
    mode: str = "mock"          # mock / tcp / serial
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 13854
    com_port: str = "COM6"
    baud_rate: int = 57600
    device_target_sample_rate: int = 512  # 设备目标采样率（信息展示用）
    stale_timeout: float = 2.5


@dataclass
class EEGSnapshot:
    """单次EEG数据快照。"""
    raw: int = 0
    attention: int = 50
    meditation: int = 50
    delta: int = 0
    theta: int = 0
    alpha1: int = 0
    alpha2: int = 0
    beta1: int = 0
    beta2: int = 0
    gamma1: int = 0
    gamma2: int = 0
    poor_signal: int = 200
    timestamp: float = 0.0
    ui_refresh_hz: int = 10    # Mock模式实际UI刷新率


class EEGAcquisitionWorker(QThread):
    """EEG采集工作线程。

    Signals:
        data_ready(EEGSnapshot): 每次采集到新数据时发射。
        status_changed(dict):    连接状态变化时发射。
    """

    data_ready = Signal(object)
    status_changed = Signal(dict)

    def __init__(self, config: AcquisitionConfig, parent=None):
        super().__init__(parent)
        self.cfg = config
        self._running = False
        self._sim_state = _MockSimState()

    def run(self):
        self._running = True
        if self.cfg.mode == "mock":
            self._mock_loop()
        elif self.cfg.mode == "tcp":
            self._tcp_loop()
        elif self.cfg.mode == "serial":
            self._serial_loop()

    def stop(self):
        self._running = False
        self.wait(3000)

    # ── Mock模式 ──

    def _mock_loop(self):
        """生成模拟EEG数据。

        Mock模式下 connector_status=offline, device_status=offline，
        不假装设备已连接。sample_rate_hz=None（由调用方设置）。
        UI刷新率约10Hz，与设备目标512Hz分开标识。
        """
        interval = 0.1  # 10 Hz UI更新频率
        t = 0.0

        # Mock模式：明确报告未连接
        self.status_changed.emit({
            "connector_status": "offline",
            "device_status": "offline",
            "mode": "live",
            "data_source": "mock",
            "note": "Mock模式 - 未接入真实设备",
        })

        while self._running:
            t += interval
            self._sim_state.advance(t)

            poor = self._sim_state.poor_signal
            att = self._sim_state.attention
            med = self._sim_state.meditation
            powers = self._sim_state.powers

            raw_val = int(
                400 * math.sin(t * 8.0)
                + 150 * math.sin(t * 23.0)
                + random.gauss(0, 80)
            )

            snap = EEGSnapshot(
                raw=raw_val,
                attention=att,
                meditation=med,
                delta=powers[0],
                theta=powers[1],
                alpha1=powers[2],
                alpha2=powers[3],
                beta1=powers[4],
                beta2=powers[5],
                gamma1=powers[6],
                gamma2=powers[7],
                poor_signal=poor,
                timestamp=time.time(),
                ui_refresh_hz=MOCK_UI_REFRESH_HZ,
            )
            self.data_ready.emit(snap)
            self.msleep(int(interval * 1000))

    # ── TCP模式（预留接口，尚未接入）──

    def _tcp_loop(self):
        """预留：对接 ThinkGear Connector TCP JSON流。

        界面必须显示"尚未接入"。
        实际部署时在此连接 127.0.0.1:13854，发送
        {"enableRawOutput": true, "format": "Json"}，
        解析 poorSignalLevel / eSense / eegPower / rawEeg 字段。
        """
        self.status_changed.emit({
            "connector_status": "offline",
            "device_status": "offline",
            "mode": "live",
            "data_source": "tcp",
            "note": "TCP模式尚未接入 - 等待Codex集成",
        })
        # TODO: 接入后端 ThinkGearCollector._tcp_collect_loop 逻辑
        while self._running:
            self.msleep(500)

    # ── Serial模式（预留接口，尚未接入）──

    def _serial_loop(self):
        """预留：对接 ThinkGear 串口模式。界面显示"尚未接入"。"""
        self.status_changed.emit({
            "connector_status": "offline",
            "device_status": "offline",
            "mode": "live",
            "data_source": "serial",
            "note": "串口模式尚未接入 - 等待Codex集成",
        })
        # TODO: 接入后端 ThinkGearCollector._serial_collect_loop 逻辑
        while self._running:
            self.msleep(500)


# 常量引用（避免循环导入）
from services.dashboard_state import MOCK_UI_REFRESH_HZ


class _MockSimState:
    """Mock信号状态机，生成有节律变化的注意力/冥想/频带功率。"""

    def __init__(self):
        self.t = 0.0
        self.attention = 55
        self.meditation = 50
        self.poor_signal = 0
        self.powers = [0] * 8
        self._poor_timer = 0.0
        self._phase = 0  # 0=正常, 1=信号波动, 2=短暂掉线
        self._emotion_trend = 0  # 0=positive 1=neutral 2=negative

    def advance(self, t: float):
        self.t = t

        # 模拟情绪趋势周期：约120秒
        cycle = (t % 120.0) / 120.0
        if cycle < 0.35:
            self._emotion_trend = 0
        elif cycle < 0.70:
            self._emotion_trend = 1
        else:
            self._emotion_trend = 2

        base_att = [70, 60, 38][self._emotion_trend]
        base_med = [65, 52, 35][self._emotion_trend]
        self.attention = int(np.clip(
            base_att + 10 * math.sin(t / 12.0) + random.gauss(0, 3), 0, 100))
        self.meditation = int(np.clip(
            base_med + 12 * math.sin(t / 15.0 + 1.5) + random.gauss(0, 3), 0, 100))

        amp_mult = [1.1, 1.0, 0.7][self._emotion_trend]
        bases = [50000, 40000, 30000, 25000, 20000, 18000, 15000, 12000]
        amps = [20000, 15000, 12000, 10000, 8000, 7000, 6000, 5000]
        self.powers = [
            int(max(0, b + a * amp_mult * math.sin(t / (14.0 + i)) + random.gauss(0, 500)))
            for i, (b, a) in enumerate(zip(bases, amps))
        ]

        self._poor_timer += 0.1
        if self._poor_timer > 25 + random.uniform(0, 15):
            self._phase = random.choice([1, 1, 2])
            self._poor_timer = 0.0

        if self._phase == 0:
            self.poor_signal = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 25])
        elif self._phase == 1:
            self.poor_signal = random.randint(40, 80)
            if random.random() < 0.3:
                self._phase = 0
        elif self._phase == 2:
            self.poor_signal = 200
            if random.random() < 0.2:
                self._phase = 0

    @property
    def emotion_trend(self) -> int:
        return self._emotion_trend
