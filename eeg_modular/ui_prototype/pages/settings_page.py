"""Formal runtime contract and system diagnostics page."""

from __future__ import annotations

import os
import platform
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QGridLayout, QLabel, QHBoxLayout, QVBoxLayout, QWidget, QPushButton

from pages.base_page import BasePage
from services.dashboard_state import DEVICE_TARGET_SAMPLE_HZ, INFERENCE_INTERVAL, WARMUP_SECONDS
from widgets.card import Card


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


class SettingsPage(BasePage):
    def __init__(self, state, service):
        self.state = state
        self.service = service
        self._diag_labels = {}
        self._diag_keys = {}
        super().__init__(
            "设置与系统诊断",
            "查看已冻结的生产配置、设备连接状态与本地数据位置。",
        )
        self._build_ui()
        self.set_role(self._role)

    @staticmethod
    def _wrap(layout) -> QWidget:
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    @staticmethod
    def _add_rows(layout: QGridLayout, rows):
        for index, (name, value) in enumerate(rows):
            key = QLabel(name + "：")
            key.setStyleSheet("color: #8FA0B8; font-size: 13px;")
            val = QLabel(str(value))
            val.setWordWrap(True)
            val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            val.setStyleSheet("color: #E8EDF3; font-size: 13px;")
            layout.addWidget(key, index, 0)
            layout.addWidget(val, index, 1)

    def _build_ui(self):
        columns = QHBoxLayout()
        columns.setSpacing(14)

        left = QVBoxLayout()
        left.setSpacing(12)

        connection = Card("设备与数据连接（只读）")
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(10)
        session_dir = self._sessions_dir()
        self._add_rows(grid, [
            ("连接方式", "ThinkGear Connector TCP（实时）"),
            ("服务地址", "127.0.0.1:13854"),
            ("目标采样率", f"{DEVICE_TARGET_SAMPLE_HZ} Hz"),
            ("会话CSV目录", str(Path(session_dir).resolve())),
            ("隐私策略", "原始EEG仅在本机处理和保存"),
        ])
        connection.add_widget(self._wrap(grid))
        open_folder = QPushButton("打开会话CSV文件夹")
        open_folder.clicked.connect(self._open_sessions_folder)
        connection.add_widget(open_folder)
        left.addWidget(connection)

        contract = Card("冻结分析契约（只读）")
        self._contract_card = contract
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(10)
        self._add_rows(grid, [
            ("观察窗口", f"{int(WARMUP_SECONDS)} 秒 / 15,360 Raw样点"),
            ("结果更新", f"每 {INFERENCE_INTERVAL:.0f} 秒"),
            ("时域特征", "filtered · 10×4"),
            ("频域特征", "bandpower · 10×4"),
            ("辅助指标", "ATT / MED（不进入情绪分类张量）"),
            ("主导状态", "最近90秒有效预测的众数；拒识窗口不计票"),
        ])
        contract.add_widget(self._wrap(grid))
        left.addWidget(contract)
        left.addStretch()
        columns.addLayout(left, 1)

        right = QVBoxLayout()
        right.setSpacing(12)

        model = Card("生产模型")
        self._model_card = model
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(9)
        self._add_rows(grid, [
            ("版本", "Production Baseline v1"),
            ("结构", "filtered + bandpower 双分支CNN（无CVAE）"),
            ("类别映射", "happy / normal / sad → positive / neutral / negative"),
            ("Dropout", "0.3"),
            ("全覆盖评估", "Accuracy 63.88% · Macro-F1 62.69%（受试者隔离）"),
            ("选择性识别", "90.20%（仅高置信度接受窗口；覆盖率18.41%）"),
        ])
        model.add_widget(self._wrap(grid))
        package_dir = PACKAGE_ROOT / "production_baseline_v1"
        required = (
            "model.pt", "baseline_contract.json", "class_mapping.json",
            "confidence_policy.json", "scaler_filtered.joblib",
            "scaler_bandpower.joblib", "checksums.sha256",
        )
        missing = [name for name in required if not (package_dir / name).is_file()]
        appdata_root = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
        app_dir = appdata_root / "EEGLearningAssistant"
        legacy_encoder = PACKAGE_ROOT / "features" / "label_encoder.joblib"
        resource_lines = [
            f"模型文件：{'正常' if (package_dir / 'model.pt').is_file() else '缺失'} · {package_dir / 'model.pt'}",
            f"配置文件：{'正常' if not missing else '缺失'} · {package_dir}",
            f"History目录：{'正常' if _safe_exists(self._sessions_dir()) else '待创建'} · {self._sessions_dir()}",
            f"Baseline结果：{'正常' if _safe_exists(app_dir / 'baseline_results.json') else '待创建'} · {app_dir / 'baseline_results.json'}",
            f"SQLite协调库：{'正常' if _safe_exists(app_dir / 'coordination.sqlite3') else '待创建'} · {app_dir / 'coordination.sqlite3'}",
        ]
        assets = QLabel(
            "生产必需资产\n状态：" + ("生产运行资产完整" if not missing else "生产资产缺失 " + "、".join(missing))
            + "\n" + "\n".join(resource_lines)
            + "\n\n兼容资产\nLegacy Label Encoder："
            + ("已安装" if legacy_encoder.is_file() else "未安装")
            + f" · {legacy_encoder}"
            + "\n说明：仅旧兼容链路需要，当前 Production Baseline v1 不依赖，不影响正式推理。"
        )
        assets.setWordWrap(True)
        assets.setTextInteractionFlags(Qt.TextSelectableByMouse)
        assets.setStyleSheet("color: #8FA0B8; font-size: 12px;")
        self._asset_diagnostics = assets
        model.add_widget(assets)
        right.addWidget(model)

        system = Card("运行环境")
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(9)
        self._add_rows(grid, [
            ("操作系统", f"{platform.system()} {platform.release()}"),
            ("Python", platform.python_version()),
            ("CPU架构", platform.machine()),
            ("CPU核心数", os.cpu_count() or "N/A"),
        ])
        system.add_widget(self._wrap(grid))
        right.addWidget(system)

        diagnostics = Card("设备、模型、数据质量与实验记录")
        self._diagnostics_card = diagnostics
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(9)
        items = [
            ("connector", "ThinkGear Connector"), ("device", "MindWave设备"),
            ("source", "数据模式"), ("sample_rate", "采样率"),
            ("quality", "质量等级"), ("reason", "质量说明"),
            ("warmup", "预热进度"), ("inference", "推理状态"),
            ("model", "模型状态"), ("experiment", "实验运行ID"),
            ("model_detail", "模型诊断详情"),
        ]
        for index, (key, name) in enumerate(items):
            row, pair = divmod(index, 2)
            label = QLabel(name + "：")
            label.setStyleSheet("color: #8FA0B8; font-size: 13px;")
            value = QLabel("--")
            value.setWordWrap(True)
            value.setStyleSheet("color: #E8EDF3; font-size: 13px;")
            grid.addWidget(label, row, pair * 2)
            grid.addWidget(value, row, pair * 2 + 1)
            self._diag_keys[key] = label
            self._diag_labels[key] = value
        diagnostics.add_widget(self._wrap(grid))
        right.addWidget(diagnostics)
        right.addStretch()
        columns.addLayout(right, 2)

        self.content_layout.addLayout(columns)

    def _sessions_dir(self) -> Path:
        folder = Path(getattr(self.service, "sessions_dir", Path("data/sessions")))
        if not folder.is_absolute():
            folder = PACKAGE_ROOT / folder
        return folder.resolve()

    def _open_sessions_folder(self):
        # 与 History/Replay 统一：优先 SessionStore 当前真实根目录。
        from services.session_store import resolve_sessions_root
        folder = resolve_sessions_root(self.state, self.service)
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def set_role(self, role: str):
        super().set_role(role)
        if not hasattr(self, "_diag_labels") or "model_detail" not in self._diag_labels:
            return
        research = self._role == "research"
        self._diag_keys["model_detail"].setVisible(research)
        self._diag_labels["model_detail"].setVisible(research)

    def update_state(self, state):
        connector = {"offline": "离线", "connecting": "连接中", "online": "在线"}
        device = {"offline": "离线", "waiting_raw": "等待首个Raw", "online": "在线"}
        quality = {"trusted": "可信", "warning": "警告", "rejected": "不合格"}
        self._diag_labels["connector"].setText(connector.get(state.connector_status, state.connector_status))
        self._diag_labels["device"].setText(device.get(state.device_status, state.device_status))
        source = {"live": "实时采集", "mock": "教学演示数据", "replay": "离线数据回放"}
        self._diag_labels["source"].setText(source.get(state.mode, state.mode))
        self._diag_labels["sample_rate"].setText(
            "尚无Raw数据" if state.sample_rate_hz is None else f"{state.sample_rate_hz:.0f} Hz"
        )
        self._diag_labels["quality"].setText(quality.get(state.quality_level, state.quality_level))
        self._diag_labels["reason"].setText("；".join(state.quality_reasons) or "--")
        self._diag_labels["warmup"].setText(f"{state.warmup_progress * 100:.0f}%")
        pipeline = str(getattr(state, "pipeline_state", "") or "")
        pipeline_display = {
            "waiting_data": "等待设备数据",
            "warming_up": "正在预热",
            "ready": "就绪",
            "rejected": "当前窗口已拒识",
            "error": "故障",
        }
        self._diag_labels["inference"].setText(
            pipeline_display.get(pipeline, "就绪" if state.inference_eligible else "未就绪")
        )
        model_status = str(getattr(state, "model_status", "") or "").upper()
        model_user = str(getattr(state, "model_error_user", "") or "")
        model_detail = str(getattr(state, "model_error_detail", "") or "")
        self._diag_labels["model"].setText(
            model_user or {
                "LOADING": "Production Baseline v1 加载中",
                "READY": "Production Baseline v1 可用",
                "FAILED": "Production Baseline v1 故障",
            }.get(model_status, "故障" if pipeline == "error" else "Production Baseline v1 可用")
        )
        self._diag_labels["model_detail"].setText(model_detail or "--")
        self._diag_labels["experiment"].setText(str(getattr(state, "run_id", "--")))
