from __future__ import annotations

import json
import logging
import socket
import struct
import time
from collections import deque
from dataclasses import dataclass
from threading import Thread, Lock
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

# 导入训练数据采样器
try:
    from src.training_data_sampler import get_sampler
    HAS_SAMPLER = True
except ImportError:
    HAS_SAMPLER = False

logger = logging.getLogger(__name__)


@dataclass
class ThinkGearConfig:
    connection_mode: str = "mock"
    com_port: str = "COM3"
    baud_rate: int = 57600
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 13854
    sample_rate: int = 512
    buffer_size: int = 1024
    analysis_window_seconds: float = 30.0
    use_mock: bool = True
    stale_timeout_seconds: float = 2.5
    use_training_data: bool = True  # 使用训练数据作为mock
    features_dir: str = "../features"  # 训练数据目录


@dataclass
class EEGData:
    raw: List[int]
    attention: int
    meditation: int
    delta: int
    theta: int
    alpha1: int
    alpha2: int
    beta1: int
    beta2: int
    gamma1: int
    gamma2: int
    timestamp: float


class ThinkGearCollector:
    """ThinkGear设备数据采集器 - 支持TCP、串口和模拟模式"""

    def __init__(self, cfg: ThinkGearConfig):
        self.cfg = cfg
        self._running = False
        self._thread: Optional[Thread] = None
        self._lock = Lock()
        
        self._data_buffer: Deque[EEGData] = deque(maxlen=cfg.buffer_size)
        self._raw_buffer: Deque[int] = deque(
            maxlen=round(cfg.sample_rate * cfg.analysis_window_seconds)
        )
        
        self._last_attention = 50
        self._last_meditation = 50
        self._last_powers = [0] * 8
        self._last_packet_time = 0.0
        self._last_valid_data_time = 0.0
        self._last_esense_time = 0.0
        self._last_power_time = 0.0
        self._last_poor_signal = 200
        self._raw_packet_count = 0
        self._esense_packet_count = 0
        self._power_packet_count = 0
        
        # 训练数据采样器
        self._sampler = None
        if cfg.use_training_data and HAS_SAMPLER:
            try:
                hold_samples = getattr(cfg, 'training_data_hold_samples', 30)
                self._sampler = get_sampler(cfg.features_dir, hold_samples=hold_samples)
                logger.info(f"ThinkGearCollector initialized with TRAINING DATA sampler (hold_samples={hold_samples})")
            except Exception as e:
                logger.warning(f"Failed to load training data sampler: {e}, falling back to simple mock")
                self._sampler = None
        
        is_mock = cfg.use_mock or cfg.connection_mode == "mock"
        
        if is_mock:
            if self._sampler:
                logger.info("ThinkGearCollector initialized in MOCK mode (using training data)")
            else:
                logger.info("ThinkGearCollector initialized in MOCK mode (simple mock)")
        elif cfg.connection_mode == "tcp":
            logger.info(f"ThinkGearCollector initialized for TCP: {cfg.tcp_host}:{cfg.tcp_port}")
        elif cfg.connection_mode == "serial":
            logger.info(f"ThinkGearCollector initialized for {cfg.com_port} @ {cfg.baud_rate}bps")
        else:
            raise ValueError(f"Unknown connection_mode: {cfg.connection_mode}")

    def start(self):
        if self._running:
            logger.warning("Collector already running")
            return
        
        self._running = True
        self._thread = Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
        logger.info("ThinkGearCollector started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("ThinkGearCollector stopped")

    def _collect_loop(self):
        is_mock = self.cfg.use_mock or self.cfg.connection_mode == "mock"
        if is_mock:
            self._mock_collect_loop()
        elif self.cfg.connection_mode == "tcp":
            self._tcp_collect_loop()
        elif self.cfg.connection_mode == "serial":
            self._serial_collect_loop()

    def _mock_collect_loop(self):
        """模拟数据采集（用于开发测试）"""
        t = 0.0
        log_counter = 0
        
        while self._running:
            try:
                att = int(50 + 20 * np.sin(t / 30.0) + np.random.randn() * 3)
                med = int(50 + 25 * np.sin(t / 45.0 + 1.2) + np.random.randn() * 3)
                
                powers = [
                    int(50000 + 20000 * np.sin(t / 20.0) + np.random.randn() * 2000),
                    int(40000 + 15000 * np.sin(t / 22.0) + np.random.randn() * 1500),
                    int(30000 + 12000 * np.sin(t / 18.0) + np.random.randn() * 1200),
                    int(25000 + 10000 * np.sin(t / 19.0) + np.random.randn() * 1000),
                    int(20000 + 8000 * np.sin(t / 17.0) + np.random.randn() * 800),
                    int(18000 + 7000 * np.sin(t / 16.0) + np.random.randn() * 700),
                    int(15000 + 6000 * np.sin(t / 15.0) + np.random.randn() * 600),
                    int(12000 + 5000 * np.sin(t / 14.0) + np.random.randn() * 500),
                ]
                
                att = np.clip(att, 0, 100)
                med = np.clip(med, 0, 100)
                powers = [max(0, p) for p in powers]
                
                # 每10个时间步记录一次mock数据生成日志
                if log_counter % 10 == 0:
                    logger.debug(f"[MOCK] Generated data - t={t:.1f}, att={att}, med={med}, "
                               f"powers_mean={np.mean(powers):.0f}, powers_std={np.std(powers):.0f}")
                
                with self._lock:
                    self._last_attention = int(att)
                    self._last_meditation = int(med)
                    self._last_powers = powers
                    self._last_packet_time = time.time()
                    self._last_valid_data_time = self._last_packet_time
                    self._last_esense_time = self._last_packet_time
                    self._last_power_time = self._last_packet_time
                    self._last_poor_signal = 0
                    self._raw_packet_count += 1
                    self._esense_packet_count += 1
                    self._power_packet_count += 1
                    raw_val = int(1000 * np.sin(t / 5.0))
                    self._raw_buffer.append(raw_val)
                
                eeg_data = EEGData(
                    raw=list(self._raw_buffer)[-128:],
                    attention=self._last_attention,
                    meditation=self._last_meditation,
                    delta=powers[0],
                    theta=powers[1],
                    alpha1=powers[2],
                    alpha2=powers[3],
                    beta1=powers[4],
                    beta2=powers[5],
                    gamma1=powers[6],
                    gamma2=powers[7],
                    timestamp=time.time(),
                )
                
                with self._lock:
                    self._data_buffer.append(eeg_data)
                
                t += 1.0
                log_counter += 1
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Mock collection error: {e}")
                time.sleep(0.1)

    def _tcp_collect_loop(self):
        """TCP模式采集 - 连接ThinkGear Connector"""
        sock = None
        buffer = ""
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            logger.info(f"Connecting to ThinkGear Connector at {self.cfg.tcp_host}:{self.cfg.tcp_port}...")
            sock.connect((self.cfg.tcp_host, self.cfg.tcp_port))
            logger.info("Connected to ThinkGear Connector!")
            
            config = {"enableRawOutput": True, "format": "Json"}
            sock.send(json.dumps(config).encode())
            logger.info("Sent configuration to ThinkGear Connector")
            
            sock.settimeout(0.1)
            
            while self._running:
                try:
                    data = sock.recv(4096)
                    if data:
                        buffer += data.decode('utf-8', errors='ignore')
                        
                        while '\r' in buffer:
                            packet, buffer = buffer.split('\r', 1)
                            if packet.strip():
                                self._parse_json(packet.strip())
                
                except socket.timeout:
                    pass
                except Exception as e:
                    logger.warning(f"TCP receive error: {e}")
                    time.sleep(0.1)
            
        except ConnectionRefusedError:
            logger.error(f"Could not connect to ThinkGear Connector at {self.cfg.tcp_host}:{self.cfg.tcp_port}")
            logger.error("Please start ThinkGear Connector first!")
        except Exception as e:
            logger.error(f"TCP communication error: {e}")
        finally:
            if sock:
                sock.close()
                logger.info("TCP connection closed")

    def _serial_collect_loop(self):
        """串口模式采集 - 直接连接ThinkGear设备"""
        logger.warning("Serial mode not fully implemented yet")
        while self._running:
            time.sleep(1.0)

    def _parse_json(self, json_str: str):
        """解析ThinkGear Connector的JSON数据（和之前程序保持一致）"""
        try:
            data = json.loads(json_str)
            now = time.time()
            
            with self._lock:
                self._last_packet_time = now
                if 'poorSignalLevel' in data:
                    self._last_poor_signal = int(data.get('poorSignalLevel') or 0)

                if 'rawEeg' in data:
                    self._raw_packet_count += 1
                    self._raw_buffer.append(int(data.get('rawEeg') or 0))
                    return

                attention = data.get('attention', self._last_attention)
                meditation = data.get('meditation', self._last_meditation)
                has_esense = False
                has_power = 'eegPower' in data
                
                if 'eSense' in data:
                    esense = data['eSense']
                    attention = esense.get('attention', attention)
                    meditation = esense.get('meditation', meditation)
                    has_esense = True
                
                self._last_attention = int(np.clip(int(attention), 0, 100))
                self._last_meditation = int(np.clip(int(meditation), 0, 100))
                if has_esense:
                    self._last_esense_time = now
                    self._esense_packet_count += 1
                
                if 'eegPower' in data:
                    eeg_power = data['eegPower']
                    self._last_powers = [
                        eeg_power.get('delta', self._last_powers[0]),
                        eeg_power.get('theta', self._last_powers[1]),
                        eeg_power.get('lowAlpha', self._last_powers[2]),
                        eeg_power.get('highAlpha', self._last_powers[3]),
                        eeg_power.get('lowBeta', self._last_powers[4]),
                        eeg_power.get('highBeta', self._last_powers[5]),
                        eeg_power.get('lowGamma', self._last_powers[6]),
                        eeg_power.get('highGamma', self._last_powers[7]),
                    ]
                    self._last_power_time = now
                    self._power_packet_count += 1
                
                if (has_esense or has_power) and self._last_poor_signal < 200:
                    self._last_valid_data_time = now
                
                raw_value = int(sum(self._last_powers) / 800) if sum(self._last_powers) > 0 else 0
                self._raw_buffer.append(raw_value)
                
                if len(self._raw_buffer) >= 128:
                    eeg_data = EEGData(
                        raw=list(self._raw_buffer)[-128:],
                        attention=self._last_attention,
                        meditation=self._last_meditation,
                        delta=self._last_powers[0],
                        theta=self._last_powers[1],
                        alpha1=self._last_powers[2],
                        alpha2=self._last_powers[3],
                        beta1=self._last_powers[4],
                        beta2=self._last_powers[5],
                        gamma1=self._last_powers[6],
                        gamma2=self._last_powers[7],
                        timestamp=time.time(),
                    )
                    self._data_buffer.append(eeg_data)
            
        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.debug(f"JSON parse error: {e}")

    def get_latest_data(self) -> Optional[EEGData]:
        """获取最新的EEG数据"""
        with self._lock:
            if self._data_buffer:
                return self._data_buffer[-1]
        return None

    def get_status(self) -> Dict[str, object]:
        """Return stream/device freshness for inference gating and Unity UI."""
        with self._lock:
            now = time.time()
            packet_age = now - self._last_packet_time if self._last_packet_time > 0 else 999999.0
            data_age = now - self._last_valid_data_time if self._last_valid_data_time > 0 else 999999.0
            esense_age = now - self._last_esense_time if self._last_esense_time > 0 else 999999.0
            power_age = now - self._last_power_time if self._last_power_time > 0 else 999999.0
            timeout = self.cfg.stale_timeout_seconds
            stream_connected = packet_age <= timeout
            has_live_esense = esense_age <= timeout
            has_live_power = power_age <= timeout
            device_connected = has_live_esense and self._last_poor_signal < 200
            if self.cfg.use_mock or self.cfg.connection_mode == "mock":
                source = "mock"
            elif not stream_connected:
                source = f"{self.cfg.connection_mode}/stale"
            elif has_live_esense:
                source = f"{self.cfg.connection_mode}/esense"
            elif has_live_power:
                source = f"{self.cfg.connection_mode}/power-only"
            else:
                source = f"{self.cfg.connection_mode}/raw-only"

            return {
                "stream_connected": stream_connected,
                "device_connected": device_connected,
                "packet_age_seconds": packet_age,
                "data_age_seconds": data_age,
                "esense_age_seconds": esense_age,
                "power_age_seconds": power_age,
                "poor_signal": int(self._last_poor_signal),
                "attention": int(self._last_attention),
                "meditation": int(self._last_meditation),
                "source": source,
                "raw_packet_count": int(self._raw_packet_count),
                "esense_packet_count": int(self._esense_packet_count),
                "power_packet_count": int(self._power_packet_count),
                "buffer_ready": len(self._raw_buffer) >= self._raw_buffer.maxlen,
                "buffer_fill_seconds": len(self._raw_buffer) / float(self.cfg.sample_rate),
                "analysis_window_seconds": float(self.cfg.analysis_window_seconds),
            }

    def get_multimodal_features(self, time_steps: int = 10, feat_dim: int = 4) -> Dict[str, np.ndarray]:
        """获取多模态特征数据（用于模型推理）"""
        with self._lock:
            # 如果使用训练数据采样器
            if self._sampler is not None:
                result, label = self._sampler.get_sample()
                
                # 记录训练数据的统计信息
                for modality, arr in result.items():
                    logger.debug(f"[TRAIN_DATA] {modality} - shape={arr.shape}, mean={arr.mean():.4f}, "
                               f"std={arr.std():.4f}, min={arr.min():.4f}, max={arr.max():.4f}")
                
                if label is not None:
                    class_names = ["happy", "sad", "normal"]
                    logger.debug(f"[TRAIN_DATA] Sample label: {label} ({class_names[label] if label < len(class_names) else 'unknown'})")
                
                return result
            
            # 否则使用简单mock数据
            result = {}
            
            # 记录原始输入数据
            logger.debug(f"[DATA] Raw values - att={self._last_attention}, med={self._last_meditation}, "
                       f"last_powers={self._last_powers}")
            
            cutoff = time.time() - self.cfg.analysis_window_seconds
            recent = [sample for sample in self._data_buffer if sample.timestamp >= cutoff]

            for modality in ['filtered', 'powerspec', 'att', 'med']:
                arr = np.zeros((time_steps, feat_dim), dtype=np.float32)
                
                if modality == 'filtered':
                    raw_values = list(self._raw_buffer)
                    if raw_values:
                        chunks = np.array_split(np.array(raw_values, dtype=np.float32), time_steps)
                        for i, chunk in enumerate(chunks[-time_steps:]):
                            arr[i, 0] = float(np.mean(chunk))
                            arr[i, 1] = float(np.std(chunk))
                            arr[i, 2] = float(np.max(chunk))
                            arr[i, 3] = float(np.min(chunk))
                
                elif modality == 'powerspec':
                    sample_groups = np.array_split(np.asarray(recent, dtype=object), time_steps)
                    for i, group in enumerate(sample_groups):
                        values = []
                        for sample in group:
                            values.extend([
                                sample.delta, sample.theta, sample.alpha1, sample.alpha2,
                                sample.beta1, sample.beta2, sample.gamma1, sample.gamma2,
                            ])
                        powers = np.asarray(values or self._last_powers, dtype=np.float32)
                        scaled = np.log1p(np.maximum(powers, 0.0))
                        arr[i] = [np.mean(scaled), np.std(scaled), np.max(scaled), np.min(scaled)]
                
                elif modality == 'att':
                    sample_groups = np.array_split(np.asarray(recent, dtype=object), time_steps)
                    for i, group in enumerate(sample_groups):
                        values = np.asarray(
                            [sample.attention for sample in group] or [self._last_attention],
                            dtype=np.float32,
                        )
                        arr[i] = [np.mean(values), np.std(values), np.max(values), np.min(values)]
                
                elif modality == 'med':
                    sample_groups = np.array_split(np.asarray(recent, dtype=object), time_steps)
                    for i, group in enumerate(sample_groups):
                        values = np.asarray(
                            [sample.meditation for sample in group] or [self._last_meditation],
                            dtype=np.float32,
                        )
                        arr[i] = [np.mean(values), np.std(values), np.max(values), np.min(values)]
                
                # 记录每个模态的统计信息
                logger.debug(f"[FEATURE] {modality} - shape={arr.shape}, mean={arr.mean():.4f}, "
                           f"std={arr.std():.4f}, min={arr.min():.4f}, max={arr.max():.4f}")
                
                result[modality] = arr
            
            return result
