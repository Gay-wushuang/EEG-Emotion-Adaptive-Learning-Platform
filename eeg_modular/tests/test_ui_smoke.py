"""UI冒烟测试 - 智学脑机助手原型。

测试内容：
  1. 七个页面创建无异常
  2. 页面导航切换正常
  3. DashboardState字段更新正常
  4. 后台线程正常退出

运行方式（在 eeg_modular 目录下）：
    E:\\anaconda3\\envs\\eegcnn\\python.exe -m pytest tests/test_ui_smoke.py -v
    或
    E:\\anaconda3\\envs\\eegcnn\\python.exe -m unittest tests.test_ui_smoke -v

注意：需要 PySide6 和 pyqtgraph 已安装。
使用 QT_QPA_PLATFORM=offscreen 支持无显示器环境。
"""

import os
import sys
import time
import unittest
from unittest.mock import patch

# 确保ui_prototype目录在sys.path中
_UI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui_prototype")
if _UI_DIR not in sys.path:
    sys.path.insert(0, _UI_DIR)

# 无显示器环境支持
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _import_qt():
    """延迟导入PySide6，给出清晰错误信息。"""
    try:
        from PySide6.QtCore import Qt, QTimer, QEventLoop
        from PySide6.QtWidgets import QApplication
        return Qt, QTimer, QEventLoop, QApplication
    except ImportError as e:
        raise unittest.SkipTest(f"PySide6未安装，跳过UI测试: {e}")


class UISmokeTest(unittest.TestCase):
    """UI冒烟测试。"""

    @classmethod
    def setUpClass(cls):
        _, QTimer, QEventLoop, QApplication = _import_qt()
        cls.QTimer = QTimer
        cls.QEventLoop = QEventLoop
        cls.app = QApplication.instance() or QApplication(sys.argv)

        # 加载QSS主题
        qss_path = os.path.join(_UI_DIR, "resources", "theme.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r", encoding="utf-8") as f:
                cls.app.setStyleSheet(f.read())

    def test_01_dashboard_state_fields(self):
        """测试DashboardState包含AGENTS.md第6节定义的全部正式字段。"""
        from services.dashboard_state import DashboardState

        state = DashboardState()

        required_fields = [
            "run_id", "mode", "connector_status", "device_status",
            "sample_rate_hz", "poor_signal", "quality_level", "quality_reasons",
            "warmup_progress", "prob_positive", "prob_neutral", "prob_negative",
            "predicted_state", "confidence", "stable_state",
            "attention", "meditation", "feedback_text", "session_seconds",
        ]

        for field_name in required_fields:
            self.assertTrue(
                hasattr(state, field_name),
                f"DashboardState缺少正式字段: {field_name}"
            )

        # 验证初始值
        self.assertEqual(state.mode, "live")
        self.assertEqual(state.connector_status, "offline")
        self.assertEqual(state.device_status, "offline")
        self.assertIsNone(state.sample_rate_hz)  # Mock模式为None
        self.assertEqual(state.quality_level, "rejected")
        self.assertIsInstance(state.quality_reasons, list)
        self.assertEqual(state.warmup_progress, 0.0)
        self.assertIsNone(state.prob_positive)
        self.assertIsNone(state.predicted_state)
        self.assertEqual(state.feedback_text, "等待信号稳定后将生成学习建议。")
        self.assertEqual(state.session_seconds, 0.0)

    def test_02_main_window_creation(self):
        """测试主窗口创建和7个页面初始化。"""
        from main_window import MainWindow

        self.window = MainWindow()
        self.assertEqual(self.window.stack.count(), 7)
        self.assertIn(self.app.font().family(), ("Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "SimSun"))

        expected_pages = [
            "welcome", "baseline", "dashboard",
            "task", "history", "settings", "replay",
        ]
        for key in expected_pages:
            self.assertIn(key, self.window._pages,
                          f"缺少页面: {key}")
            self.assertIsNotNone(self.window._pages[key],
                                 f"页面对象为None: {key}")

    def test_03_page_navigation(self):
        """测试7个页面导航切换。"""
        from main_window import MainWindow

        if not hasattr(self, "window"):
            self.window = MainWindow()

        pages = ["welcome", "baseline", "dashboard", "task",
                 "history", "settings", "replay"]

        for page_key in pages:
            self.window._navigate_to(page_key)
            current = self.window.stack.currentWidget()
            self.assertIs(
                current, self.window._pages[page_key],
                f"导航到{page_key}失败"
            )

        # 回到dashboard
        self.window._navigate_to("dashboard")

    def test_03b_visible_step_buttons_navigate(self):
        """欢迎页和基线页的主要操作按钮必须有真实响应。"""
        from main_window import MainWindow

        if not hasattr(self, "window"):
            self.window = MainWindow()
        self.window._navigate_to("welcome")
        self.window._pages["welcome"]._btn_start.click()
        self.assertIs(self.window.stack.currentWidget(), self.window._pages["baseline"])

        self.window._pages["baseline"]._btn_next.setEnabled(True)
        self.window._pages["baseline"]._btn_next.click()
        self.assertIs(self.window.stack.currentWidget(), self.window._pages["dashboard"])

    def test_04_state_update_signal(self):
        """测试DashboardState状态更新和信号发射。"""
        from services.dashboard_state import DashboardState

        state = DashboardState()
        received = []
        state.state_updated.connect(lambda s: received.append(s))

        state.attention = 75.0
        state.meditation = 60.0
        state.poor_signal = 0
        state.quality_level = "trusted"
        state.warmup_progress = 1.0
        state.feedback_text = "测试反馈"
        state.emit_update()

        self.assertEqual(len(received), 1)
        self.assertIs(received[0], state)
        self.assertEqual(received[0].attention, 75.0)
        self.assertEqual(received[0].quality_level, "trusted")
        self.assertTrue(received[0].warmup_complete)
        self.assertTrue(received[0].inference_eligible)

    def test_05_demo_history_marked(self):
        """测试12条预填充历史记录全部标记demo=True。"""
        from services.dashboard_state import DashboardState

        state = DashboardState()
        self.assertEqual(len(state._history_sessions), 12)

        for record in state._history_sessions:
            self.assertTrue(record.demo,
                            f"会话{record.session_id}未标记demo=True")

    def test_06_mock_mode_not_connected(self):
        """测试Mock模式下不假装设备已连接。"""
        from services.dashboard_state import DashboardState

        state = DashboardState()
        # Mock模式初始状态
        self.assertEqual(state.connector_status, "offline")
        self.assertEqual(state.device_status, "offline")
        self.assertIsNone(state.sample_rate_hz)
        self.assertEqual(state.mode, "live")

    def test_07_thread_exit(self):
        """测试后台线程正常退出。"""
        from main_window import MainWindow

        if not hasattr(self, "window"):
            self.window = MainWindow()

        service = self.window.service
        self.assertTrue(service.acq_worker.isRunning(),
                        "采集线程应正在运行")
        self.assertTrue(service.inf_worker.isRunning(),
                        "推理线程应正在运行")

        # 停止
        service.stop_streaming()

        self.assertFalse(service.acq_worker.isRunning(),
                         "采集线程未正常退出")
        self.assertFalse(service.inf_worker.isRunning(),
                         "推理线程未正常退出")

    def test_08_resolution_scaling(self):
        """测试1920x1080和1280x700分辨率适配。"""
        from main_window import MainWindow

        if not hasattr(self, "window"):
            self.window = MainWindow()

        # 1920x1080
        self.window.resize(1920, 1080)
        self.assertEqual(self.window.width(), 1920)
        self.assertEqual(self.window.height(), 1080)

        # 导航所有页面不崩溃
        for p in ["welcome", "baseline", "dashboard", "task",
                  "history", "settings", "replay"]:
            self.window._navigate_to(p)

        # 1280x700（最小尺寸验证）
        self.window.resize(1280, 700)
        self.assertEqual(self.window.width(), 1280)
        self.assertEqual(self.window.height(), 700)

        # 再次导航所有页面不崩溃
        for p in ["welcome", "baseline", "dashboard", "task",
                  "history", "settings", "replay"]:
            self.window._navigate_to(p)

    def test_10_offline_poor_signal(self):
        """测试设备离线时Poor Signal显示为--，不显示合格。"""
        from main_window import MainWindow

        if not hasattr(self, "window"):
            self.window = MainWindow()

        # 等待信号传播：采集线程的 status_changed 需要被主线程处理
        self.app.processEvents()

        self.window._navigate_to("dashboard")
        dashboard = self.window._pages["dashboard"]
        state = self.window.state

        # 再处理一次事件，确保 _on_acq_status 已执行
        self.app.processEvents()

        # Mock模式初始状态：connector_status=offline, device_status=offline
        self.assertEqual(state.connector_status, "offline")
        self.assertEqual(state.device_status, "offline")

        # 离线时 poor_signal 必须为 None
        self.assertIsNone(
            state.poor_signal,
            "设备离线时 poor_signal 应为 None，实际为: "
            + str(state.poor_signal),
        )
        self.assertIsNone(state.attention)
        self.assertIsNone(state.meditation)
        self.assertEqual(len(state._eeg_raw_buffer), 0)
        self.assertEqual(state.warmup_progress, 0.0)

        # 离线时 quality_level 应为 rejected
        self.assertEqual(state.quality_level, "rejected")

        # 刷新UI后检查 Poor Signal 卡片显示
        dashboard.update_state(state)
        poor_value_text = dashboard._card_poor["value"].text()
        self.assertIn(
            "--", poor_value_text,
            f"离线时Poor Signal卡片应显示'--'，实际显示: '{poor_value_text}'"
        )

        # 检查状态栏显示
        sb_text = self.window._sb_signal.text()
        self.assertIn(
            "--", sb_text,
            f"离线时状态栏应显示'Poor Signal: --'，实际显示: '{sb_text}'"
        )

        # 检查信号质量卡片显示"不可评估"
        conf_value_text = dashboard._card_conf["value"].text()
        self.assertIn(
            "不可评估", conf_value_text,
            f"离线时信号质量卡片应显示'不可评估'，实际显示: '{conf_value_text}'"
        )

    def test_11_dashboard_geometry_visible(self):
        """测试1366x768下仪表盘关键控件几何可见。"""
        from main_window import MainWindow

        if not hasattr(self, "window"):
            self.window = MainWindow()

        self.window._navigate_to("dashboard")
        self.app.processEvents()
        dashboard = self.window._pages["dashboard"]

        # 1366x768 窗口
        self.window.resize(1366, 768)
        self.app.processEvents()

        # 强制布局更新
        dashboard.updateGeometry()
        self.window.updateGeometry()
        self.app.processEvents()

        # 关键控件列表及其几何位置检查（不依赖 isVisible，
        # 因为控件可能在 QScrollArea 内尚未被视作可见）
        key_widgets = [
            ("顶部状态卡", dashboard._card_connector["frame"]),
            ("Poor Signal卡", dashboard._card_poor["frame"]),
            ("信号质量卡", dashboard._card_conf["frame"]),
            ("预热进度卡", dashboard._warmup_bar),
            ("EEG曲线图", dashboard._eeg_plot),
            ("ATT/MED趋势图", dashboard._trend_plot),
            ("三分类概率面板", dashboard._prob_panel),
            ("稳定状态卡", dashboard._prob_trend),
            ("AI建议卡", dashboard._ai_label),
            ("开始按钮", dashboard._btn_start),
        ]

        # 控件几何在窗口内（通过 size() 检查是否已分配非零尺寸）
        zero_size_widgets = []
        for name, widget in key_widgets:
            size = widget.size()
            if size.width() <= 0 or size.height() <= 0:
                zero_size_widgets.append(
                    f"{name}({size.width()}x{size.height()})"
                )

        self.assertEqual(
            len(zero_size_widgets), 0,
            f"1366x768下以下控件尺寸为0: {zero_size_widgets}"
        )

        # 控件位置在窗口内（未完全移出屏幕）
        window_rect = self.window.rect()
        out_of_bounds = []
        for name, widget in key_widgets:
            widget_rect = widget.geometry()
            if not window_rect.intersects(widget_rect):
                out_of_bounds.append(name)

        self.assertEqual(
            len(out_of_bounds), 0,
            f"1366x768下以下控件位置超出窗口: {out_of_bounds}"
        )

    def test_09_replay_sample_data(self):
        """测试CSV回放模式加载示例数据。"""
        if not hasattr(self, "window"):
            from main_window import MainWindow
            self.window = MainWindow()

        self.window._navigate_to("replay")
        rp = self.window._pages["replay"]

        # 加载示例数据
        rp._load_sample()
        self.assertGreater(len(rp._data), 0,
                           "示例数据加载失败")

        # 播放一帧
        rp._update_frame(0)
        self.assertGreater(rp._index, -1)

    def tearDown(self):
        """每个测试独立释放窗口和线程，避免Qt在进程退出时原生崩溃。"""
        window = getattr(self, "window", None)
        if window is not None:
            window.service.stop_streaming()
            window.close()
            window.deleteLater()
            del self.window
        self.app.processEvents()

    @classmethod
    def tearDownClass(cls):
        # 处理 deleteLater 队列；测试进程应以真实退出码0结束。
        cls.app.processEvents()


if __name__ == "__main__":
    unittest.main(verbosity=2)
