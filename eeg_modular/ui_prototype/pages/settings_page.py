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


class SettingsPage(BasePage):
    def __init__(self, state, service):
        self.state = state
        self.service = service
        self._diag_labels = {}
        super().__init__(
            "设置与系统诊断",
            "查看已冻结的生产配置、设备连接状态与本地数据位置。",
        )
        self._build_ui()

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
        session_dir = getattr(self.service, "sessions_dir", Path("data/sessions"))
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

        diagnostics = Card("实时诊断")
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(9)
        items = [
            ("connector", "ThinkGear Connector"), ("device", "MindWave设备"),
            ("source", "数据模式"), ("sample_rate", "采样率"),
            ("quality", "质量等级"), ("reason", "质量说明"),
            ("warmup", "预热进度"), ("inference", "推理状态"),
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
            self._diag_labels[key] = value
        diagnostics.add_widget(self._wrap(grid))
        right.addWidget(diagnostics)
        right.addStretch()
        columns.addLayout(right, 2)

        self.content_layout.addLayout(columns)

    def _open_sessions_folder(self):
        folder = Path(getattr(self.service, "sessions_dir", Path("data/sessions"))).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def update_state(self, state):
        connector = {"offline": "离线", "connecting": "连接中", "online": "在线"}
        device = {"offline": "离线", "waiting_raw": "等待首个Raw", "online": "在线"}
        quality = {"trusted": "可信", "warning": "警告", "rejected": "不合格"}
        self._diag_labels["connector"].setText(connector.get(state.connector_status, state.connector_status))
        self._diag_labels["device"].setText(device.get(state.device_status, state.device_status))
        self._diag_labels["source"].setText("实时" if state.mode == "live" else "回放")
        self._diag_labels["sample_rate"].setText(
            "尚无Raw数据" if state.sample_rate_hz is None else f"{state.sample_rate_hz:.0f} Hz"
        )
        self._diag_labels["quality"].setText(quality.get(state.quality_level, state.quality_level))
        self._diag_labels["reason"].setText("；".join(state.quality_reasons) or "--")
        self._diag_labels["warmup"].setText(f"{state.warmup_progress * 100:.0f}%")
        self._diag_labels["inference"].setText("就绪" if state.inference_eligible else "未就绪")
