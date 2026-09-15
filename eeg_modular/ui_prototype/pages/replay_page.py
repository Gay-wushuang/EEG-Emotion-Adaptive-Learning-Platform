"""页面7：离线数据回放模式（管理员 / 开发 / 实验诊断工具）。

定位：加载历史会话或外部 CSV 数据，复现 EEG 信号并进行离线模型分析。

回放页面使用自身内部状态（self._data）驱动播放，不依赖 DashboardState 的
实时字段。DashboardState 的 prob_* 等字段在 Mock 模式下可能为 None，
因此概率面板通过直接 set_value 方式更新，绕过 update_state(state)。

隔离约束：本页只读取数据，绝不调用 begin_session / finalize_session /
add_event 等会话生命周期接口，不写入学生 History、runtime snapshot、
baseline 或教师观察事件。示例数据仅用于演示，不写入任何学生记录。
"""

from __future__ import annotations

import csv
import io
import math
import random
from collections import Counter
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
    QPushButton, QFrame, QSlider, QFileDialog, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QSizePolicy,
)

from pages.base_page import BasePage
from services.session_store import SessionStore
from widgets.card import Card
from widgets.eeg_plot import EEGPlotWidget
from widgets.trend_plot import TrendPlotWidget
from widgets.probability_bar import ProbabilityPanel
from widgets.gauge import ArcGauge
from services.dashboard_state import CLASS_DISPLAY


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _generate_sample_csv() -> str:
    """生成模拟CSV回放数据（演示数据）。"""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "timestamp", "raw", "attention", "meditation", "poor_signal",
        "delta", "theta", "alpha1", "alpha2", "beta1", "beta2", "gamma1", "gamma2",
        "prob_positive", "prob_neutral", "prob_negative", "predicted_class",
    ])
    t = 0.0
    for i in range(600):  # 60秒 @ 10Hz
        t += 0.1
        cycle = (t % 120.0) / 120.0
        if cycle < 0.35:
            trend, probs = 0, [0.55, 0.30, 0.15]
        elif cycle < 0.70:
            trend, probs = 1, [0.22, 0.55, 0.23]
        else:
            trend, probs = 2, [0.15, 0.28, 0.57]

        att = int(max(0, min(100, [70, 60, 38][trend] + 10 * math.sin(t / 12) + random.gauss(0, 3))))
        med = int(max(0, min(100, [65, 52, 35][trend] + 12 * math.sin(t / 15 + 1.5) + random.gauss(0, 3))))
        raw = int(400 * math.sin(t * 8) + 150 * math.sin(t * 23) + random.gauss(0, 80))
        poor = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 25])

        writer.writerow([
            f"{t:.1f}", raw, att, med, poor,
            *[int(50000 + 20000 * math.sin(t / (14 + j))) for j in range(8)],
            f"{probs[0]:.4f}", f"{probs[1]:.4f}", f"{probs[2]:.4f}",
            ["positive", "neutral", "negative"][probs.index(max(probs))],
        ])
    return output.getvalue()


class ReplayPage(BasePage):
    def __init__(self, state, service):
        self.state = state
        self.service = service
        self._data = []
        self._index = 0
        self._playing = False
        self._speed = 1.0
        self._is_sample = False  # 标记当前是否为示例（演示）数据
        self._timer = QTimer()
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)
        super().__init__(
            "离线数据回放",
            "加载历史会话或外部 CSV 数据，复现 EEG 信号并进行离线模型分析。"
        )
        self._build_ui()

    def _build_ui(self):
        root = self.content_layout
        root.setSpacing(10)

        # 页面定位横幅（管理员 / 开发 / 实验诊断工具，非学生学习与教师教学流程）
        self._role_banner = QLabel("定位：管理员 / 开发 / 实验诊断工具（离线分析与展示，不参与学生学习与教师教学流程）")
        self._role_banner.setObjectName("ReplayRoleBanner")
        self._role_banner.setStyleSheet(
            "background:#172235;border:1px solid #27364D;border-radius:7px;"
            "color:#9FB4D2;font-size:12px;font-weight:500;padding:7px 12px;"
        )
        self._role_banner.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        root.addWidget(self._role_banner)

        # 回放模式 - 演示数据 标记（加载示例数据后显示）
        self._demo_label = QLabel("示例数据（演示用途）— 不会写入学生 History，不冒充真实学生 Session")
        self._demo_label.setStyleSheet(
            "background-color: rgba(200,150,40,0.15); "
            "color: #FBBF24; font-size: 12px; font-weight: 600; "
            "padding: 6px 10px; border-radius: 4px;"
        )
        self._demo_label.setAlignment(Qt.AlignCenter)
        self._demo_label.setVisible(False)

        # ── 数据来源：近期会话 / 外部 CSV / 示例数据 ──
        source_card = Card("数据来源")
        self._source_card = source_card
        source_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        source_layout = QVBoxLayout()
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.setSpacing(6)

        recent_row = QHBoxLayout()
        recent_row.setSpacing(10)
        recent_row.addWidget(QLabel("近期会话:"))
        self._combo_recent = QComboBox()
        self._combo_recent.setMinimumWidth(420)
        self._combo_recent.setToolTip("仅列出包含可回放 EEG 数据的已保存会话")
        recent_row.addWidget(self._combo_recent, 1)

        self._btn_load_recent = QPushButton("加载选中会话")
        self._btn_load_recent.setObjectName("PrimaryButton")
        self._btn_load_recent.setEnabled(False)
        self._btn_load_recent.setToolTip("加载近期会话列表中选中的回放 CSV")
        self._btn_load_recent.clicked.connect(self._load_selected_session)
        recent_row.addWidget(self._btn_load_recent)

        self._btn_refresh_recent = QPushButton("刷新列表")
        self._btn_refresh_recent.clicked.connect(self._refresh_recent_sessions)
        recent_row.addWidget(self._btn_refresh_recent)
        source_layout.addLayout(recent_row)

        self._recent_hint = QLabel("")
        self._recent_hint.setStyleSheet("color: #6B7689; font-size: 12px;")
        self._recent_hint.setWordWrap(True)

        import_row = QHBoxLayout()
        import_row.setSpacing(10)
        self._btn_load = QPushButton("导入外部 CSV")
        self._btn_load.setToolTip("导入实验 / 开发调试 / 演示用的外部 CSV 文件；每次加载都会替换当前回放数据")
        self._btn_load.clicked.connect(self._load_file)
        import_row.addWidget(self._btn_load)

        self._btn_sample = QPushButton("加载示例数据")
        self._btn_sample.setToolTip("加载内置示例数据（演示用途，不写入任何学生记录）")
        self._btn_sample.clicked.connect(self._load_sample)
        import_row.addWidget(self._btn_sample)
        import_row.addWidget(self._demo_label, 1)
        import_row.addWidget(self._recent_hint, 1)
        source_layout.addLayout(import_row)

        source_card.add_widget(self._wrap(source_layout))
        root.addWidget(source_card)

        # 回放控制
        control_card = Card("回放控制")
        self._control_card = control_card
        control_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(10)

        self._btn_play = QPushButton("播放")
        self._btn_play.setObjectName("PrimaryButton")
        self._btn_play.setEnabled(False)
        self._btn_play.setToolTip("请先加载包含 Raw EEG 的会话 CSV")
        self._btn_play.clicked.connect(self._play)
        control_layout.addWidget(self._btn_play)

        self._btn_pause = QPushButton("暂停")
        self._btn_pause.setEnabled(False)
        self._btn_pause.clicked.connect(self._pause)
        control_layout.addWidget(self._btn_pause)

        self._btn_stop = QPushButton("停止")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop)
        control_layout.addWidget(self._btn_stop)

        self._btn_clear = QPushButton("清除当前数据")
        self._btn_clear.setObjectName("DangerButton")
        self._btn_clear.setEnabled(False)
        self._btn_clear.setToolTip("卸载当前回放数据并返回初始空状态")
        self._btn_clear.clicked.connect(self._clear_data)
        control_layout.addWidget(self._btn_clear)

        control_layout.addSpacing(20)

        control_layout.addWidget(QLabel("速度:"))
        self._combo_speed = QComboBox()
        self._combo_speed.addItems(["0.5x", "1x", "2x", "4x"])
        self._combo_speed.setCurrentIndex(1)
        self._combo_speed.currentIndexChanged.connect(self._change_speed)
        control_layout.addWidget(self._combo_speed)

        control_layout.addStretch()

        self._label_file = QLabel("未加载文件")
        self._label_file.setStyleSheet("color: #6B7689; font-size: 13px;")
        control_layout.addWidget(self._label_file)

        control_card.add_widget(self._wrap(control_layout))
        root.addWidget(control_card)

        progress_card = Card("回放进度")
        self._progress_card = progress_card
        progress_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        progress_layout = QVBoxLayout()
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(4)
        progress_header = QHBoxLayout()
        self._empty_label = QLabel("尚未加载回放数据 · 请选择历史会话或导入 CSV。")
        self._empty_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._empty_label.setStyleSheet(
            "color:#8491A5;font-size:12px;"
        )
        progress_header.addWidget(self._empty_label, 1)
        self._label_progress = QLabel("0 / 0")
        self._label_progress.setStyleSheet("color: #6B7689; font-size: 12px;")
        self._label_progress.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        progress_header.addWidget(self._label_progress)
        progress_layout.addLayout(progress_header)

        # 进度条
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_seek)
        progress_layout.addWidget(self._slider)
        progress_card.add_widget(self._wrap(progress_layout))
        root.addWidget(progress_card)

        # ── 主分析区：左侧信号图，右侧模型概览、仪表和数据预览 ──
        analysis = QWidget()
        self._analysis_area = analysis
        analysis.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        analysis_layout = QHBoxLayout(analysis)
        analysis_layout.setContentsMargins(0, 0, 0, 0)
        analysis_layout.setSpacing(10)

        charts = QWidget()
        self._charts_column = charts
        charts_layout = QVBoxLayout(charts)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(10)

        eeg_card = Card("原始脑电回放")
        self._eeg_card = eeg_card
        self._eeg_plot = EEGPlotWidget()
        self._eeg_plot.setMinimumHeight(90)
        self._eeg_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        eeg_card.add_widget(self._eeg_plot)
        charts_layout.addWidget(eeg_card, 3)

        trend_card = Card("专注度 / 放松度趋势")
        self._trend_card = trend_card
        self._trend_plot = TrendPlotWidget()
        self._trend_plot.setMinimumHeight(70)
        self._trend_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        trend_card.add_widget(self._trend_plot)
        charts_layout.addWidget(trend_card, 2)
        analysis_layout.addWidget(charts, 60)

        details = QWidget()
        self._details_column = details
        details_layout = QVBoxLayout(details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(10)

        # 概率面板
        prob_card = Card("模型概览")
        self._model_card = prob_card
        self._prob_panel = ProbabilityPanel()
        self._prob_panel.setVisible(False)
        prob_card.add_widget(self._prob_panel)

        self._model_empty = QLabel("尚未加载回放数据\n请选择历史会话或导入 CSV。")
        self._model_empty.setAlignment(Qt.AlignCenter)
        self._model_empty.setStyleSheet("color:#8491A5;font-size:12px;padding:6px;")
        prob_card.add_widget(self._model_empty)

        self._replay_pred = QLabel("预测状态：暂无数据")
        self._replay_pred.setObjectName("AccentLabel")
        self._replay_pred.setStyleSheet("font-size:12px;")

        self._replay_dominant = QLabel("主导状态：暂无数据")
        self._replay_dominant.setStyleSheet("font-size:12px;color:#AAB6C8;")
        state_row = QHBoxLayout()
        state_row.setContentsMargins(0, 0, 0, 0)
        state_row.setSpacing(12)
        state_row.addWidget(self._replay_pred, 1)
        state_row.addWidget(self._replay_dominant, 1)
        prob_card.add_widget(self._wrap(state_row))

        # 仪表
        gauge_row = QHBoxLayout()
        gauge_row.setSpacing(8)
        att_card = QFrame()
        att_card.setObjectName("ReplayGaugeCard")
        self._att_card = att_card
        att_card.setMinimumHeight(92)
        att_layout = QVBoxLayout(att_card)
        att_layout.setContentsMargins(8, 5, 8, 5)
        self._att_gauge = ArcGauge("专注度", "#4FC3F7")
        self._att_gauge.setFixedSize(82, 82)
        att_layout.addWidget(self._att_gauge, 0, Qt.AlignCenter)
        gauge_row.addWidget(att_card, 1)

        med_card = QFrame()
        med_card.setObjectName("ReplayGaugeCard")
        self._med_card = med_card
        med_card.setMinimumHeight(92)
        med_layout = QVBoxLayout(med_card)
        med_layout.setContentsMargins(8, 5, 8, 5)
        self._med_gauge = ArcGauge("放松度", "#4ADE80")
        self._med_gauge.setFixedSize(82, 82)
        med_layout.addWidget(self._med_gauge, 0, Qt.AlignCenter)
        gauge_row.addWidget(med_card, 1)
        att_card.setVisible(False)
        med_card.setVisible(False)
        prob_card.add_widget(self._wrap(gauge_row))
        details_layout.addWidget(prob_card, 2)

        # 数据表
        table_card = Card("回放数据预览")
        self._table_card = table_card
        table_layout = QVBoxLayout()
        table_layout.setContentsMargins(0, 0, 0, 0)
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["时间", "原始脑电", "专注度", "放松度", "预测"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setMinimumHeight(68)
        self._table.verticalHeader().setDefaultSectionSize(24)
        table_layout.addWidget(self._table)
        table_card.add_widget(self._wrap(table_layout))
        details_layout.addWidget(table_card, 1)
        analysis_layout.addWidget(details, 40)
        root.addWidget(analysis, 1)

        self.setStyleSheet(self.styleSheet() + """
            QFrame#ReplayGaugeCard {
                background:#172235;border:1px solid #27364D;border-radius:8px;
            }
            QFrame#ReplayGaugeCard QLabel { color:#AAB6C8;font-size:11px; }
        """)

    def _wrap(self, layout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        return w

    def _wrap_centered(self, widget) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        layout.addWidget(widget)
        layout.addStretch()
        return w

    def _load_file(self):
        # 外部 CSV 导入的默认目录是合理的数据目录（data 根目录），
        # 不再默认进入随机 session_id 目录。
        default_dir = PACKAGE_ROOT / "data"
        default_dir.mkdir(parents=True, exist_ok=True)
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "导入外部 CSV（替换当前数据）",
            str(default_dir),
            "CSV 文件 (*.csv);;所有文件 (*)",
        )
        if not paths:
            return
        self.load_paths(paths)

    def load_paths(self, paths):
        """加载一个或多个兼容 CSV；每次调用都替换而不是追加。"""
        try:
            combined = []
            missing_predictions = False
            for path in paths:
                with open(path, "r", encoding="utf-8-sig") as f:
                    rows = list(csv.DictReader(f))
                if not rows:
                    raise ValueError(
                        f"{Path(path).name} 没有数据行（可能只有表头，或文件为空）"
                    )
                if not any("raw" in row or "raw_eeg" in row for row in rows):
                    raise ValueError(
                        f"{Path(path).name} 不是兼容的回放 CSV："
                        f"缺少 Raw EEG 列（需要 raw 或 raw_eeg 列）。"
                        f"当前表头：{', '.join(rows[0].keys())}"
                    )
                normalized = [self._normalize_row(row, Path(path).name) for row in rows]
                missing_predictions |= not any(row["predicted_class"] for row in normalized)
                combined.extend(normalized)
            self._data = combined
            self._is_sample = False
            self._demo_label.setVisible(False)
            self._recent_hint.setVisible(True)
            self._label_file.setText(f"已替换为 {len(paths)} 个文件，共 {len(self._data)} 行")
            self._init_playback()
            if missing_predictions:
                QMessageBox.information(
                    self, "原始数据回放",
                    "部分文件没有模型概率，因此只能回放 Raw/ATT/MED，不能补造状态结果。\n"
                    "应用新保存的 session.csv 是包含采集、质量与推理结果的综合CSV。"
                )
        except Exception as e:
            QMessageBox.warning(self, "加载失败", f"无法加载文件：{e}")

    def _sessions_dir(self) -> Path:
        folder = Path(getattr(self.service, "sessions_dir", Path("data/sessions")))
        if not folder.is_absolute():
            folder = PACKAGE_ROOT / folder
        return folder.resolve()

    # ── 近期可回放会话（复用现有 SessionStore，不建第二套数据库）──

    @staticmethod
    def _csv_is_replayable(path: Path) -> bool:
        """快速判断 CSV 是否可被 ReplayPage 读取。

        只读文件头部，避免扫描数百 MB 的原始 EEG 文件：
        1. 文件存在且非空；
        2. 表头包含 raw 或 raw_eeg 列；
        3. 表头之后至少有一行数据（排除仅表头的占位 CSV）。
        """
        try:
            with path.open("rb") as handle:
                head = handle.read(65536)
        except OSError:
            return False
        if not head:
            return False
        text = head.decode("utf-8-sig", errors="replace")
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return False
        columns = {col.strip().lower() for col in lines[0].split(",")}
        return bool(columns & {"raw", "raw_eeg"})

    def _recent_replayable_sessions(self) -> list[tuple[dict, Path]]:
        """从现有 Session 元数据中筛选真正可回放的近期会话。

        返回 [(session_dict, csv_path)]，按开始时间倒序。
        无 CSV / 仅表头 / 空 CSV 的会话会被排除，不会假装可加载。
        """
        store = SessionStore(self._sessions_dir())
        results: list[tuple[dict, Path]] = []
        for record in store.load(include_demo=True):
            session_id = str(record.get("session_id", ""))
            if not session_id:
                continue
            raw_csv = str(record.get("data_files", {}).get("raw_csv", "session.csv"))
            if Path(raw_csv).name != raw_csv:
                continue
            demo = bool(record.get("demo", False))
            base = store.demo_root if demo else store.root
            csv_path = base / session_id / raw_csv
            if self._csv_is_replayable(csv_path):
                results.append((record, csv_path))
        return results

    @staticmethod
    def _format_start_time(value: str) -> str:
        """把 ISO 开始时间格式化为 'YYYY-MM-DD HH:MM'。"""
        text = str(value or "").strip()
        if "T" in text:
            text = text.replace("T", " ")
        return text[:16]

    @staticmethod
    def _session_display_name(record: dict) -> str:
        """生成会话显示名：学生名称/ID · 任务 · 时间（不把 session_id 作为主名称）。"""
        user_name = str(record.get("user_name") or "").strip()
        user_id = str(record.get("user_id") or "").strip()
        if user_name and user_name != user_id:
            student = f"{user_name} ({user_id})"
        elif user_id:
            student = user_id
        else:
            student = "未知学生"
        tasks = record.get("tasks") or []
        if isinstance(tasks, list) and tasks:
            first = tasks[0] if isinstance(tasks[0], dict) else {}
            task_name = str(first.get("name") or record.get("notes") or "自由学习")
        else:
            task_name = str(record.get("notes") or "自由学习")
        when = ReplayPage._format_start_time(record.get("start_time", ""))
        duration = int(float(record.get("duration_seconds") or 0.0))
        minutes = duration // 60
        duration_text = f" · {minutes}分" if minutes > 0 else ""
        return f"{student} · {task_name} · {when}{duration_text}"

    def _refresh_recent_sessions(self):
        """刷新近期可回放会话下拉框。"""
        self._combo_recent.clear()
        items = self._recent_replayable_sessions()
        for record, csv_path in items:
            self._combo_recent.addItem(
                self._session_display_name(record),
                userData=str(csv_path),
            )
        self._btn_load_recent.setEnabled(len(items) > 0)
        if items:
            self._recent_hint.setText(
                f"共 {len(items)} 个可回放会话；"
                f"无回放数据的会话已自动排除，不会出现在列表中。"
            )
        else:
            self._recent_hint.setText(
                "当前没有找到包含可回放 EEG 数据的会话。"
                "可通过下方「导入外部 CSV」或「加载示例数据」开始。"
            )

    def _load_selected_session(self):
        """加载近期会话列表中选中的回放 CSV。"""
        index = self._combo_recent.currentIndex()
        if index < 0:
            return
        csv_path = self._combo_recent.itemData(index)
        if not csv_path:
            return
        self.load_paths([csv_path])

    def on_show(self):
        """页面被切到前台时刷新近期会话列表。"""
        self._refresh_recent_sessions()

    def _open_sessions_folder(self):
        # 与 History/Settings 统一：优先 SessionStore 当前真实根目录。
        # （_sessions_dir() 仍按服务配置解析，专用于回放数据列表。）
        from services.session_store import resolve_sessions_root
        folder = resolve_sessions_root(self.state, self.service)
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    @staticmethod
    def _normalize_row(row: dict, source: str = "") -> dict:
        pred = (row.get("predicted_class") or row.get("prediction") or "").strip()
        pred = {"happy": "positive", "normal": "neutral", "sad": "negative"}.get(pred, pred)
        return {
            **row,
            "source_file": source,
            "timestamp": row.get("signal_time_seconds") or row.get("timestamp") or row.get("timestamp_unix") or "",
            "raw": row.get("raw") if row.get("raw") not in (None, "") else row.get("raw_eeg", ""),
            "attention": row.get("attention") if row.get("attention") not in (None, "") else row.get("att", ""),
            "meditation": row.get("meditation") if row.get("meditation") not in (None, "") else row.get("med", ""),
            "predicted_class": pred,
        }

    def _load_sample(self):
        """加载内置示例数据（演示数据）。"""
        csv_text = _generate_sample_csv()
        reader = csv.DictReader(io.StringIO(csv_text))
        self._data = list(reader)
        self._is_sample = True
        self._recent_hint.setVisible(False)
        self._demo_label.setVisible(True)
        self._label_file.setText(f"已加载示例数据（演示用途，不写入学生 History），共 {len(self._data)} 行")
        self._init_playback()

    def _init_playback(self):
        if not self._data:
            self._clear_data()
            return
        self._index = 0
        self._empty_label.setVisible(False)
        self._slider.setEnabled(True)
        self._slider.setRange(0, len(self._data) - 1)
        self._slider.setValue(0)
        self._btn_play.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_clear.setEnabled(True)
        self._trend_plot.reset()
        self._model_empty.setVisible(False)
        self._update_frame(0)
        self._populate_table()
        self._update_dominant_state()

    def _clear_data(self):
        """卸载回放数据并恢复默认空状态。"""
        self._playing = False
        self._timer.stop()
        self._data = []
        self._index = 0
        self._is_sample = False
        self._demo_label.setVisible(False)
        self._recent_hint.setVisible(True)
        self._empty_label.setVisible(True)
        self._label_file.setText("未加载文件")
        self._label_progress.setText("0 / 0")
        self._slider.blockSignals(True)
        self._slider.setRange(0, 0)
        self._slider.setValue(0)
        self._slider.setEnabled(False)
        self._slider.blockSignals(False)
        self._btn_play.setEnabled(False)
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_clear.setEnabled(False)
        self._table.setRowCount(0)
        self._trend_plot.reset()
        if hasattr(self._eeg_plot, "_data"):
            self._eeg_plot._data.fill(0)
            self._eeg_plot._curve.setData(self._eeg_plot._x, self._eeg_plot._data)
        self._att_gauge.set_value(0)
        self._med_gauge.set_value(0)
        self._att_card.setVisible(False)
        self._med_card.setVisible(False)
        self._prob_panel.setVisible(False)
        self._model_empty.setText("尚未加载回放数据\n请选择历史会话或导入 CSV。")
        self._model_empty.setVisible(True)
        for bar in self._prob_panel._bars.values():
            bar.set_value(0.0)
            bar.set_dimmed(True)
        self._prob_panel._confidence_label.setText("信号质量：--")
        self._prob_panel._warning_label.setVisible(False)
        self._replay_pred.setText("预测状态：暂无数据")
        self._replay_dominant.setText("主导状态：暂无数据")

    def _update_dominant_state(self):
        votes = []
        seen_inferences = set()
        for row in self._data:
            pred = row.get("predicted_class", "")
            if pred not in ("positive", "neutral", "negative"):
                continue
            inference_id = row.get("inference_index", "")
            if inference_id:
                key = (row.get("source_file", ""), inference_id)
                if key in seen_inferences:
                    continue
                seen_inferences.add(key)
            votes.append(pred)
        if not votes:
            self._replay_dominant.setText("主导状态：暂无数据（原文件无模型预测）")
            return
        counts = Counter(votes)
        highest = max(counts.values())
        tied = {name for name, count in counts.items() if count == highest}
        dominant = next(name for name in reversed(votes) if name in tied)
        self._replay_dominant.setText(
            f"主导状态：{CLASS_DISPLAY.get(dominant, dominant)} · {highest}/{len(votes)}"
        )

    def _populate_table(self):
        self._table.setRowCount(min(50, len(self._data)))
        for i in range(min(50, len(self._data))):
            row = self._data[i]
            self._table.setItem(i, 0, QTableWidgetItem(row.get("timestamp", "")))
            self._table.setItem(i, 1, QTableWidgetItem(row.get("raw", "")))
            self._table.setItem(i, 2, QTableWidgetItem(row.get("attention", "")))
            self._table.setItem(i, 3, QTableWidgetItem(row.get("meditation", "")))
            self._table.setItem(i, 4, QTableWidgetItem(row.get("predicted_class", "")))

    def _play(self):
        if not self._data:
            return
        self._playing = True
        self._btn_play.setEnabled(False)
        self._btn_pause.setEnabled(True)
        self._btn_stop.setEnabled(True)
        self._timer.start()

    def _pause(self):
        self._playing = False
        self._timer.stop()
        self._btn_play.setEnabled(True)
        self._btn_pause.setEnabled(False)

    def _stop(self):
        self._playing = False
        self._timer.stop()
        self._index = 0
        self._slider.setValue(0)
        self._btn_play.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)

    def _change_speed(self, idx: int):
        speeds = [0.5, 1.0, 2.0, 4.0]
        self._speed = speeds[idx]
        self._timer.setInterval(int(100 / self._speed))

    def _on_seek(self, value: int):
        self._index = value
        self._update_frame(value)

    def _tick(self):
        if self._index >= len(self._data) - 1:
            self._pause()
            return
        self._index += 1
        self._slider.setValue(self._index)
        self._update_frame(self._index)

    def _safe_float(self, row: dict, key: str, default: float = 0.0) -> float:
        """安全解析 CSV 行中的浮点数，避免 None 或非法值导致崩溃。"""
        val = row.get(key)
        if val is None or val == "":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _safe_int(self, row: dict, key: str, default: int = 0) -> int:
        """安全解析 CSV 行中的整数。"""
        val = row.get(key)
        if val is None or val == "":
            return default
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return default

    def _update_frame(self, idx: int):
        if not self._data or idx >= len(self._data):
            return
        row = self._data[idx]

        raw = self._safe_int(row, "raw")
        has_attention = row.get("attention") not in (None, "")
        has_meditation = row.get("meditation") not in (None, "")
        att = self._safe_int(row, "attention")
        med = self._safe_int(row, "meditation")
        pp = self._safe_float(row, "prob_positive")
        pn = self._safe_float(row, "prob_neutral")
        ng = self._safe_float(row, "prob_negative")
        pred = row.get("predicted_class", "") or ""

        self._eeg_plot.push_value(raw)
        self._trend_plot.push_values(att, med)
        self._att_gauge.set_value(att)
        self._med_gauge.set_value(med)
        self._att_card.setVisible(has_attention)
        self._med_card.setVisible(has_meditation)

        # 直接操作概率面板的条形组件，绕过 DashboardState
        # （回放模式下 DashboardState 的 prob_* 字段为 None，不能通过
        #   _prob_panel.update_state(state) 更新）
        has_probabilities = all(
            row.get(key) not in (None, "")
            for key in ("prob_positive", "prob_neutral", "prob_negative")
        )
        self._prob_panel.setVisible(has_probabilities)
        self._model_empty.setVisible(not has_probabilities)
        if has_probabilities:
            self._prob_panel._bars["positive"].set_value(pp)
            self._prob_panel._bars["neutral"].set_value(pn)
            self._prob_panel._bars["negative"].set_value(ng)
            for bar in self._prob_panel._bars.values():
                bar.set_dimmed(False)
            self._prob_panel._confidence_label.setText("回放模式 · 信号可信度：原文件未提供")
            self._prob_panel._confidence_label.setStyleSheet(
                "font-size: 12px; color: #6B7689; padding-top: 4px;"
            )
            self._prob_panel._warning_label.setVisible(False)
        else:
            self._model_empty.setText("当前回放数据不含模型概率")

        display = CLASS_DISPLAY.get(pred, pred) if pred else "暂无数据"
        self._replay_pred.setText(f"预测状态：{display}")

        self._label_progress.setText(
            f"{idx + 1} / {len(self._data)}  ({row.get('timestamp', '')}s)"
        )

    def update_state(self, state):
        """回放页面不消费 DashboardState 实时字段，空实现避免父类调用。"""
        pass
