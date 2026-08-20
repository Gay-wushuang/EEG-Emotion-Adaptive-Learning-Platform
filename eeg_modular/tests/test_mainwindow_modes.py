"""MainWindow Mock/Live 双模式测试。

验证：
1. MainWindow(mode="mock") 正常创建，使用 MockDataService
2. MainWindow(mode="live") 在 patch 掉 socket/thread 后正常创建，使用 LiveDataService
3. production_baseline_v1 路径解析正确
4. 模式切换不影响 DashboardState 字段完整性

Live 测试中 patch 掉真正 socket/thread，不得要求测试环境连接真实 MindWave。

运行方式（在 eeg_modular 目录下）：
    E:\\anaconda3\\envs\\eegcnn\\python.exe -m pytest tests/test_mainwindow_modes.py -v
"""

import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

# 确保路径正确
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "ui_prototype"))
sys.path.insert(0, str(_ROOT))

# 无显示器环境支持
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _import_qt():
    try:
        from PySide6.QtWidgets import QApplication, QWidget, QLabel
        return QApplication, QWidget, QLabel
    except ImportError as e:
        raise unittest.SkipTest(f"PySide6未安装，跳过测试: {e}")


class TestMainWindowModes(unittest.TestCase):
    """MainWindow Mock/Live 双模式测试。"""

    @classmethod
    def setUpClass(cls):
        QApplication, QWidget, QLabel = _import_qt()
        cls.QWidget = QWidget
        cls.QLabel = QLabel
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_A1_mock_mode_creates_mock_service(self):
        """Mock 模式：MainWindow(mode='mock') 使用 MockDataService。"""
        from main_window import MainWindow
        from services.mock_data_service import MockDataService

        window = MainWindow(mode="mock")
        try:
            self.assertIsInstance(window.service, MockDataService,
                                  "Mock 模式应使用 MockDataService")
            self.assertEqual(window._mode, "mock")
            self.assertEqual(window.stack.count(), 7)

            # 侧边栏版本标签：搜索所有子控件
            sidebar = self._find_sidebar(window, self.QWidget)
            sidebar_label_text = self._find_version_label(sidebar, self.QLabel)
            self.assertIn("Mock", sidebar_label_text,
                          f"Mock 模式侧边栏应标注 Mock模式，实际: '{sidebar_label_text}'")
        finally:
            window.service.stop_streaming()
            window.close()
            window.deleteLater()

    @staticmethod
    def _find_sidebar(window, qwidget_cls):
        """获取已构建的侧边栏 QWidget。"""
        for child in window.findChildren(qwidget_cls):
            if child.objectName() == "SideBar":
                return child
        return None

    @staticmethod
    def _find_version_label(sidebar, qlabel_cls):
        """从侧边栏获取版本标签文本。"""
        if sidebar is None:
            return ""
        for child in sidebar.findChildren(qlabel_cls):
            text = child.text()
            if "v1.0.0" in text:
                return text
        return ""

    def test_A2_mock_mode_default(self):
        """不指定 mode 时默认使用 Mock 模式。"""
        from main_window import MainWindow
        from services.mock_data_service import MockDataService

        window = MainWindow()  # 默认 mode="mock"
        try:
            self.assertIsInstance(window.service, MockDataService)
            self.assertEqual(window._mode, "mock")
        finally:
            window.service.stop_streaming()
            window.close()
            window.deleteLater()

    def test_A3_live_mode_uses_live_service(self):
        """Live 模式：MainWindow(mode='live') 使用 LiveDataService。

        Patch 掉 ThinkGearLiveWorker 和 ProductionInferenceWorker，
        不得要求真实 MindWave 连接。
        """
        from main_window import MainWindow

        # Patch 掉 LiveDataService 内部的线程启动和 socket 连接
        with patch(
            'smart_learning_app.live_service.ThinkGearLiveWorker',
        ) as mock_acq_class, patch(
            'smart_learning_app.live_service.ProductionInferenceWorker',
        ) as mock_inf_class:

            # 设置 mock worker 行为
            mock_acq = mock_acq_class.return_value
            mock_acq.isRunning.return_value = False
            mock_inf = mock_inf_class.return_value
            mock_inf.isRunning.return_value = False

            window = MainWindow(mode="live")
            try:
                from smart_learning_app.live_service import LiveDataService
                self.assertIsInstance(window.service, LiveDataService,
                                      "Live 模式应使用 LiveDataService")
                self.assertEqual(window._mode, "live")
                self.assertEqual(window.stack.count(), 7)

                # production 包路径应已设置
                self.assertTrue(
                    hasattr(window.state, '_production_package_dir'),
                    "Live 模式下 state 应记录 production 包路径"
                )
            finally:
                window.service.stop_streaming()
                window.close()
                window.deleteLater()

    def test_A4_production_package_dir_resolution(self):
        """production_baseline_v1 路径解析正确。"""
        from main_window import _PRODUCTION_PACKAGE_DIR

        self.assertTrue(
            _PRODUCTION_PACKAGE_DIR.is_absolute(),
            "production 路径应为绝对路径"
        )
        # 检查目录名
        self.assertEqual(
            _PRODUCTION_PACKAGE_DIR.name,
            "production_baseline_v1",
            "目录名应为 production_baseline_v1"
        )
        # 检查是否包含关键文件（只读检查，不修改）
        if _PRODUCTION_PACKAGE_DIR.exists():
            self.assertTrue(
                (_PRODUCTION_PACKAGE_DIR / "model.pt").exists(),
                "production 包应包含 model.pt"
            )
            self.assertTrue(
                (_PRODUCTION_PACKAGE_DIR / "baseline_contract.json").exists(),
                "production 包应包含 baseline_contract.json"
            )

    def test_A5_live_fallback_when_package_missing(self):
        """Live 模式在 production 包不存在时回退到 Mock。"""
        from main_window import MainWindow

        with patch(
            'smart_learning_app.live_service.ThinkGearLiveWorker',
        ) as mock_acq, patch(
            'smart_learning_app.live_service.ProductionInferenceWorker',
        ) as mock_inf:

            # 传一个不存在的路径
            nonexistent_dir = Path(__file__).parent / "_nonexistent_package"
            window = MainWindow(mode="live", package_dir=nonexistent_dir)
            try:
                from services.mock_data_service import MockDataService
                self.assertIsInstance(
                    window.service, MockDataService,
                    "包不存在时应回退到 MockDataService"
                )
                # state 应记录回退原因
                self.assertTrue(
                    hasattr(window.state, '_live_fallback_reason'),
                    "state 应记录回退原因"
                )
            finally:
                window.service.stop_streaming()
                window.close()
                window.deleteLater()

    def test_A6_live_mode_service_has_adaptive_engine(self):
        """Live 模式的 LiveDataService 已配置 AdaptiveFeedbackEngine。"""
        from main_window import MainWindow

        with patch(
            'smart_learning_app.live_service.ThinkGearLiveWorker',
        ), patch(
            'smart_learning_app.live_service.ProductionInferenceWorker',
        ):
            window = MainWindow(mode="live")
            try:
                self.assertTrue(
                    hasattr(window.service, 'engine'),
                    "LiveDataService 应有 engine 属性"
                )
                self.assertEqual(
                    window.service.engine.negative_threshold, 0.60
                )
                self.assertEqual(
                    window.service.engine.sustain_seconds, 20.0
                )
                self.assertEqual(
                    window.service.engine.cooldown_seconds, 90.0
                )
            finally:
                window.service.stop_streaming()
                window.close()
                window.deleteLater()

    def test_A7_mock_mode_service_has_adaptive_engine(self):
        """Mock 模式的 MockDataService 已配置 AdaptiveFeedbackEngine。"""
        from main_window import MainWindow

        window = MainWindow(mode="mock")
        try:
            self.assertTrue(
                hasattr(window.service, 'engine'),
                "MockDataService 应有 engine 属性"
            )
            self.assertEqual(
                window.service.engine.negative_threshold, 0.60
            )
            self.assertEqual(
                window.service.engine.sustain_seconds, 20.0
            )
            self.assertEqual(
                window.service.engine.cooldown_seconds, 90.0
            )
        finally:
            window.service.stop_streaming()
            window.close()
            window.deleteLater()

    def test_A8_both_modes_share_dashboard_state(self):
        """Mock 和 Live 模式共用同一 DashboardState 接口。"""
        from main_window import MainWindow
        from services.dashboard_state import DashboardState

        # Mock 模式
        mock_window = MainWindow(mode="mock")
        try:
            self.assertIsInstance(mock_window.state, DashboardState)
            self.assertEqual(mock_window.state.mode, "live")
            self.assertEqual(mock_window.state.connector_status, "offline")
            self.assertEqual(mock_window.state.device_status, "offline")
        finally:
            mock_window.service.stop_streaming()
            mock_window.close()
            mock_window.deleteLater()

        # Live 模式（patch 掉 worker）
        with patch(
            'smart_learning_app.live_service.ThinkGearLiveWorker',
        ), patch(
            'smart_learning_app.live_service.ProductionInferenceWorker',
        ):
            live_window = MainWindow(mode="live")
            try:
                self.assertIsInstance(live_window.state, DashboardState)
                self.assertEqual(live_window.state.mode, "live")
                # Live 模式初始也是 offline（未连接设备）
                self.assertEqual(live_window.state.connector_status, "offline")
                self.assertEqual(live_window.state.device_status, "offline")
            finally:
                live_window.service.stop_streaming()
                live_window.close()
                live_window.deleteLater()

    def test_A9_live_integration_with_synthetic_result(self):
        """Live 模式 synthetic 集成：Fake InferenceResult → LiveDataService._on_result。

        完整链路：InferenceResult(accepted=True, negative=0.70)
        → LiveDataService._on_result() → AdaptiveFeedbackEngine
        → DashboardState 改变。
        """
        from main_window import MainWindow
        from smart_learning_app.inference_engine import InferenceResult
        from services.dashboard_state import (
            DIFFICULTY_HARD, DIFFICULTY_MEDIUM, AdaptiveAction,
        )

        with patch(
            'smart_learning_app.live_service.ThinkGearLiveWorker',
        ), patch(
            'smart_learning_app.live_service.ProductionInferenceWorker',
        ):
            window = MainWindow(mode="live")
            try:
                s = window.state
                s.connector_status = "online"
                s.device_status = "online"
                s.warmup_progress = 1.0
                s.quality_level = "trusted"
                s.poor_signal = 5
                s.task_difficulty = DIFFICULTY_HARD
                s.attention = 45.0

                # 调整引擎时间参数使测试在合理时间内完成
                svc = window.service
                svc.engine._policy.sustain_seconds = 0.1
                svc.engine._policy.cooldown_seconds = 0.5

                # 预填充 Temporal Policy 状态：模拟持续负性已建立
                import numpy as np
                svc.engine._policy.ewma = np.array([0.10, 0.20, 0.70])
                svc.engine._policy.above_since = time.time() - 1.0

                # 单次调用 _on_result
                fake = InferenceResult(
                    probabilities=(0.10, 0.20, 0.70),
                    internal_class="sad",
                    display_class="negative",
                    confidence=0.70,
                    accepted=True,
                    latency_ms=5.0,
                )
                svc._on_result(fake)

                self.assertTrue(s._intervention_triggered)
                self.assertEqual(s.task_difficulty, DIFFICULTY_MEDIUM,
                                 "困难应降低为中等")
                self.assertEqual(s.adaptive_action, AdaptiveAction.REDUCE_DIFFICULTY)
                self.assertIsNotNone(s.adaptive_action_time)

                interventions = [e for e in s._events if e.category == "intervention"]
                self.assertEqual(len(interventions), 1,
                                 "应恰好产生 1 次 intervention event")
            finally:
                window.service.stop_streaming()
                window.close()
                window.deleteLater()

    def test_A10_live_rejected_result_blocks(self):
        """Live 模式：accepted=False 连续高负性 → 0 intervention。"""
        from main_window import MainWindow
        from smart_learning_app.inference_engine import InferenceResult
        from services.dashboard_state import (
            DIFFICULTY_HARD, AdaptiveAction,
        )

        with patch(
            'smart_learning_app.live_service.ThinkGearLiveWorker',
        ), patch(
            'smart_learning_app.live_service.ProductionInferenceWorker',
        ):
            window = MainWindow(mode="live")
            try:
                s = window.state
                s.connector_status = "online"
                s.device_status = "online"
                s.warmup_progress = 1.0
                s.quality_level = "trusted"
                s.poor_signal = 5
                s.task_difficulty = DIFFICULTY_HARD
                s.attention = 45.0

                svc = window.service
                svc.engine._policy.sustain_seconds = 0.1
                svc.engine._policy.cooldown_seconds = 0.5

                before = len([e for e in s._events if e.category == "intervention"])

                # 连续 20 次 accepted=False
                for i in range(20):
                    fake = InferenceResult(
                        probabilities=(0.0, 0.01, 0.99),
                        internal_class="sad",
                        display_class="negative",
                        confidence=0.05,
                        accepted=False,
                        latency_ms=1.0,
                    )
                    svc._on_result(fake)
                    self.assertFalse(s._intervention_triggered,
                                     f"第 {i} 次不应触发")

                after = len([e for e in s._events if e.category == "intervention"])
                self.assertEqual(after - before, 0)
                self.assertEqual(s.task_difficulty, DIFFICULTY_HARD)
                self.assertEqual(s.adaptive_action, AdaptiveAction.NONE)
            finally:
                window.service.stop_streaming()
                window.close()
                window.deleteLater()


if __name__ == "__main__":
    unittest.main(verbosity=2)
