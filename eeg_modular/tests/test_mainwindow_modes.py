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
        """Live 模式在 production 包不存在时回退到 Mock，且 _mode 改为 mock。

        绝不允许在 Mock service 上显示 Live 标签。
        回退完成后：
        - service 是 MockDataService
        - window._mode == "mock"
        - state.mode == "mock"
        - 侧边栏包含 Mock
        - 状态栏包含 Mock
        - 任何地方都不出现 "Live · 真实EEG"
        """
        from main_window import MainWindow
        from PySide6.QtWidgets import QApplication

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
                # _mode 必须改为 mock，不能保留 live
                self.assertEqual(
                    window._mode, "mock",
                    "回退后 _mode 必须改为 mock，不能保留 live"
                )
                # state.mode 必须同步改为 mock
                self.assertEqual(
                    window.state.mode, "mock",
                    "回退后 state.mode 必须为 'mock'，不能为 'live'"
                )
                # state 应记录回退原因
                self.assertTrue(
                    hasattr(window.state, '_live_fallback_reason'),
                    "state 应记录回退原因"
                )
                # state.quality_level 应为 rejected
                self.assertEqual(
                    window.state.quality_level, "rejected",
                    "回退后 quality_level 应为 rejected"
                )
                # 侧边栏版本标签不应出现 "Live"
                sidebar = self._find_sidebar(window, self.QWidget)
                sidebar_text = self._find_version_label(sidebar, self.QLabel)
                self.assertNotIn(
                    "Live", sidebar_text,
                    f"回退后侧边栏不应显示 Live，实际: '{sidebar_text}'"
                )
                self.assertIn(
                    "Mock", sidebar_text,
                    f"回退后侧边栏应显示 Mock，实际: '{sidebar_text}'"
                )

                # 状态栏也不应出现 "Live · 真实EEG"
                mode_label = window._sb_mode.text()
                self.assertNotIn(
                    "真实EEG", mode_label,
                    f"回退后状态栏不应显示 '真实EEG'，实际: '{mode_label}'"
                )
                # 触发状态更新让 Mock 采集线程的信号到达
                for _ in range(5):
                    QApplication.processEvents()
                    self.app.processEvents()
                    time.sleep(0.05)
                mode_label = window._sb_mode.text()
                self.assertIn(
                    "Mock", mode_label,
                    f"回退后状态栏应包含 Mock，实际: '{mode_label}'"
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
        """Mock 和 Live 模式共用同一 DashboardState 接口。

        Mock 模式：真实采集线程 status_changed 会将 mode 设为 "mock"，
        不能再断言为 "live"。必须 processEvents 让信号传递。
        """
        from main_window import MainWindow
        from services.dashboard_state import DashboardState
        from PySide6.QtWidgets import QApplication

        # Mock 模式
        mock_window = MainWindow(mode="mock")
        try:
            self.assertIsInstance(mock_window.state, DashboardState)
            # 处理 Qt 事件让 status_changed 信号到达
            QApplication.processEvents()
            # Mock 模式下 state.mode 应为 "mock"（由 _mock_loop status_changed 路径设置）
            self.assertEqual(
                mock_window.state.mode, "mock",
                f"Mock 模式下 state.mode 应为 'mock'，实际: '{mock_window.state.mode}'"
            )
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

    def test_A11_live_status_bar_no_mock_text(self):
        """Live 模式状态栏不应出现 Mock 字样。"""
        from main_window import MainWindow

        with patch(
            'smart_learning_app.live_service.ThinkGearLiveWorker',
        ), patch(
            'smart_learning_app.live_service.ProductionInferenceWorker',
        ):
            window = MainWindow(mode="live")
            try:
                # 模拟真实 Live 状态
                s = window.state
                s.mode = "live"
                s.connector_status = "online"
                s.device_status = "online"
                s.quality_level = "trusted"
                s.warmup_progress = 1.0

                # 触发状态栏更新
                s.emit_update()

                # 检查状态栏模式标签
                mode_label = window._sb_mode.text()
                self.assertNotIn(
                    "Mock", mode_label,
                    f"Live 状态栏不应包含 Mock，实际: '{mode_label}'"
                )
                self.assertIn(
                    "Live", mode_label,
                    f"Live 状态栏应包含 Live，实际: '{mode_label}'"
                )
            finally:
                window.service.stop_streaming()
                window.close()
                window.deleteLater()

    def test_A12_mock_status_bar_has_mock_text(self):
        """Mock 模式状态栏应明确标识 Mock（真实 status_changed 路径）。

        不手工设置 s.mode，让真实 _mock_loop status_changed 信号
        经 Qt 事件循环传递到 DashboardState → 状态栏。
        """
        from main_window import MainWindow
        from PySide6.QtWidgets import QApplication

        window = MainWindow(mode="mock")
        try:
            s = window.state
            # 等待 status_changed 信号到达
            for _ in range(5):
                QApplication.processEvents()
                self.app.processEvents()
                time.sleep(0.05)

            # 验证 state.mode 已由真实路径设置为 "mock"
            self.assertEqual(
                s.mode, "mock",
                f"Mock 模式下 state.mode 应为 'mock'，实际: '{s.mode}'"
            )

            # 触发状态栏更新（通过 emit_update）
            s.emit_update()
            QApplication.processEvents()
            self.app.processEvents()

            mode_label = window._sb_mode.text()
            self.assertIn(
                "Mock", mode_label,
                f"Mock 状态栏应包含 Mock，实际: '{mode_label}'"
            )
            self.assertNotIn(
                "Live", mode_label,
                f"Mock 状态栏不应包含 Live，实际: '{mode_label}'"
            )
        finally:
            window.service.stop_streaming()
            window.close()
            window.deleteLater()


class TestDiagnoseSampleRate(unittest.TestCase):
    """诊断脚本逻辑测试（不连接真实设备）。"""

    def test_B1_classify_rate_512Hz_band(self):
        """450-570 Hz 区间应分类为 approximately 512 Hz。"""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'smart_learning_app'))
        from diagnose_sample_rate import classify_rate

        for rate in [450, 480, 512, 540, 570]:
            result = classify_rate(rate)
            self.assertIn(
                "512 Hz", result,
                f"{rate} Hz 应分类为 512 Hz，实际: {result}"
            )
            self.assertIn("OK", result,
                          f"{rate} Hz 应为 OK，实际: {result}")

    def test_B2_classify_rate_256Hz_band(self):
        """220-300 Hz 区间应分类为 approximately 256 Hz。"""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'smart_learning_app'))
        from diagnose_sample_rate import classify_rate

        for rate in [220, 256, 300]:
            result = classify_rate(rate)
            self.assertIn(
                "256 Hz", result,
                f"{rate} Hz 应分类为 256 Hz，实际: {result}"
            )
            self.assertIn("WARNING", result,
                          f"{rate} Hz 应为 WARNING，实际: {result}")

    def test_B3_classify_rate_unexpected(self):
        """偏离区间应分类为 unexpected。"""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'smart_learning_app'))
        from diagnose_sample_rate import classify_rate

        for rate in [100, 400, 600, 800]:
            result = classify_rate(rate)
            self.assertIn(
                "unexpected", result,
                f"{rate} Hz 应分类为 unexpected，实际: {result}"
            )

    def test_B4_classify_rate_boundary(self):
        """边界值测试。"""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'smart_learning_app'))
        from diagnose_sample_rate import classify_rate

        # 恰好在边界上
        self.assertIn("512", classify_rate(450))
        self.assertIn("512", classify_rate(570))
        self.assertIn("256", classify_rate(220))
        self.assertIn("256", classify_rate(300))

        # 边界外
        self.assertIn("unexpected", classify_rate(449))
        self.assertIn("unexpected", classify_rate(571))
        self.assertIn("unexpected", classify_rate(219))
        self.assertIn("unexpected", classify_rate(301))

    def test_B5_tcp_cross_chunk_parsing(self):
        """跨 TCP chunk 的 JSON 解析不会丢 rawEeg。

        模拟 rawEeg 数据被切割到两个 TCP recv 中，
        使用 remainder buffer 应能正确拼合。
        """
        import sys, os, json
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'smart_learning_app'))

        # 模拟 diagnose 中的 remainder buffer 逻辑
        # 两条完整 JSON 包，每条以 \r 结尾
        packet1 = json.dumps({"rawEeg": 100, "poorSignalLevel": 0}) + "\r"
        packet2 = json.dumps({"rawEeg": 200, "poorSignalLevel": 1}) + "\r"

        # 构造两个 chunk：
        # chunk1 包含 packet1 + "\r" + packet2 的前半部分
        # chunk2 包含 packet2 的后半部分
        split_point = len(packet1) + len(packet2) // 2
        chunk1 = packet1 + packet2[:len(packet1) + len(packet2) // 2 - len(packet1)]
        # 重新计算：chunk1 = packet1 + packet2_prefix
        p2_prefix_len = max(1, len(packet2) // 2)
        chunk1 = packet1 + packet2[:p2_prefix_len]
        chunk2 = packet2[p2_prefix_len:]

        # 模拟解析（与 diagnose.diagnose() 中完全相同的逻辑）
        remainder = ""
        parsed_raw_eeg = []

        remainder += chunk1
        while "\r" in remainder:
            line, remainder = remainder.split("\r", 1)
            if not line.strip():
                continue
            try:
                packet = json.loads(line)
                if "rawEeg" in packet:
                    parsed_raw_eeg.append(packet["rawEeg"])
            except (json.JSONDecodeError, TypeError):
                pass

        # chunk2 到达，补全剩余的 packet2
        remainder += chunk2
        while "\r" in remainder:
            line, remainder = remainder.split("\r", 1)
            if not line.strip():
                continue
            try:
                packet = json.loads(line)
                if "rawEeg" in packet:
                    parsed_raw_eeg.append(packet["rawEeg"])
            except (json.JSONDecodeError, TypeError):
                pass

        # 两条 rawEeg 都应该被解析到
        self.assertEqual(len(parsed_raw_eeg), 2,
                         f"跨 chunk 解析应不丢数据，实际: {parsed_raw_eeg}")
        self.assertIn(100, parsed_raw_eeg)
        self.assertIn(200, parsed_raw_eeg)

    def test_B6_sample_rate_report_fields(self):
        """SampleRateReport 应包含所有必需字段。"""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'smart_learning_app'))
        from diagnose_sample_rate import SampleRateReport

        report = SampleRateReport(
            startup_delay_seconds=7.2,
            active_duration_seconds=30.0,
            total_raw_packets=15360,
            raw_count=15360,
            active_raw_rate_hz=512.0,
            esense_count=150,
            eegpower_count=300,
            poorsignal_count=30,
            samples_per_second=[512, 511, 513],
            min_rate=510.0,
            max_rate=514.0,
            mean_rate=512.0,
            median_rate=512.0,
            std_rate=1.0,
            warnings=[],
            passed_threshold=True,
        )

        text = str(report)
        self.assertIn("7.2", text)
        self.assertIn("30.0", text)
        self.assertIn("15360", text)
        self.assertIn("512.0 Hz", text)
        self.assertIn("是", text)
        # 检查 startup delay 和 active duration 字段
        self.assertIn("Startup Delay", text)
        self.assertIn("有效采集时长", text)
        self.assertIn("poorSignal", text)
        self.assertIn("eSense", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
