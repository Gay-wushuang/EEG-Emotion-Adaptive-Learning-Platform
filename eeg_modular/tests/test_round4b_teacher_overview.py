"""Round 4B：教师端"学生近期学习概览"（纯读取统计）测试。

覆盖：最近 N 次选择、多任务摘要、缺失 Attention 不进平均、事件计数、
Demo/Live 隔离、时间规律门槛、单次异常不生成长期结论、零副作用。
"""

from __future__ import annotations

import copy
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui_prototype"))

from PySide6.QtWidgets import QApplication

from services.dashboard_state import (
    DashboardState,
    EventMarker,
    SessionRecord,
    TaskRecord,
)
from services.student_history_summary import (
    MISSING,
    TIMING_INSUFFICIENT,
    build_recent_summary,
)


def _epoch(text: str) -> float:
    from datetime import datetime
    return datetime.fromisoformat(text.replace(" ", "T")).timestamp()


def _task(name="英语阅读", task_id="T1", start="2026-09-14 10:00:00",
          duration=600.0):
    return TaskRecord(
        name=name, difficulty="medium", status="completed",
        start_time=start, end_time=start, duration_seconds=duration,
        task_id=task_id,
    )


def _event(content, *, type_="marker", source="system", category="",
           timestamp=0.0, task_id="T1", note=""):
    return EventMarker(
        label=content, content=content, note=note,
        source=source, type=type_, category=category,
        timestamp=timestamp, task_id=task_id,
    )


def _session(idx, *, start, attention=50.0, duration=600.0, tasks=None,
             events=None, source="live", demo=False):
    return SessionRecord(
        session_id=f"s_{idx}", source=source, demo=demo,
        user_id="st_001", user_name="学生一",
        start_time=start, end_time=start, duration_seconds=duration,
        avg_attention=attention, avg_meditation=40.0,
        tasks=tasks if tasks is not None else [_task()],
        events=events if events is not None else [],
    )


class _StartService:
    def __init__(self, state=None):
        self.state = state

    def start_session(self):
        pass

    def end_session(self):
        pass

    def pause_session(self):
        pass

    def resume_session(self):
        pass


class Round4BRecentSelectionTest(unittest.TestCase):
    """1～3：最近 N 次选择与多任务摘要。"""

    def test_01_selects_most_recent_five(self):
        records = [
            _session(i, start=f"2026-09-{i:02d} 10:00:00") for i in range(1, 8)
        ]
        summary = build_recent_summary(records, source="live")
        self.assertEqual(summary["stats"]["count"], 5)
        dates = [row["date"] for row in summary["sessions"]]
        # newest first：09-07 … 09-03
        self.assertEqual(dates[0], "2026-09-07 10:00:00")
        self.assertEqual(dates[-1], "2026-09-03 10:00:00")
        self.assertNotIn("2026-09-01 10:00:00", dates)
        self.assertNotIn("2026-09-02 10:00:00", dates)

    def test_02_more_than_five_counts_only_recent_five(self):
        records = [
            _session(i, start=f"2026-09-{i:02d} 10:00:00",
                     events=[_event("AI建议：降低后续学习负荷",
                                    type_="intervention")])
            for i in range(1, 11)
        ]
        summary = build_recent_summary(records, source="live")
        self.assertEqual(summary["stats"]["count"], 5)
        # 只统计最近 5 次（09-06…09-10 各 1 条）= 5，而非全部 10
        self.assertEqual(summary["stats"]["ai_advice_count"], 5)

    def test_03_multi_task_summary(self):
        two = _session(1, start="2026-09-14 10:00:00",
                       tasks=[_task("英语阅读"), _task("物理复习")])
        three = _session(2, start="2026-09-13 10:00:00",
                         tasks=[_task("英语阅读"), _task("物理复习"),
                                _task("生词复习")])
        summary = build_recent_summary([two, three], source="live")
        self.assertEqual(summary["sessions"][0]["task_summary"],
                         "英语阅读 + 物理复习")
        self.assertEqual(summary["sessions"][1]["task_summary"], "3项任务")


class Round4BAverageTest(unittest.TestCase):
    """4～5：缺失 Attention 不参与平均。"""

    def test_04_missing_attention_excluded_from_average(self):
        records = [
            _session(1, start="2026-09-14 10:00:00", attention=60.0),
            _session(2, start="2026-09-13 10:00:00", attention=0.0),  # 无效样本
        ]
        summary = build_recent_summary(records, source="live")
        att = summary["stats"]["avg_attention"]
        # 只有 60.0 参与平均，缺失值绝不当 0
        self.assertAlmostEqual(att["value"], 60.0)
        self.assertEqual(att["based_on"], 1)
        self.assertIn("基于1次有效记录", att["text"])
        self.assertEqual(summary["sessions"][1]["attention"]["text"], MISSING)

    def test_05_partial_attention_shows_based_on(self):
        records = [
            _session(1, start="2026-09-14 10:00:00", attention=52.0),
            _session(2, start="2026-09-13 10:00:00", attention=0.0),
            _session(3, start="2026-09-12 10:00:00", attention=56.0),
            _session(4, start="2026-09-11 10:00:00", attention=54.0),
            _session(5, start="2026-09-10 10:00:00", attention=0.0),
        ]
        summary = build_recent_summary(records, source="live")
        att = summary["stats"]["avg_attention"]
        self.assertAlmostEqual(att["value"], (52.0 + 56.0 + 54.0) / 3)
        self.assertEqual(att["based_on"], 3)
        self.assertIn("基于3次有效记录", att["text"])
        self.assertNotIn("基于5次", att["text"])


class Round4BEventCountTest(unittest.TestCase):
    """6～8：AI建议 / 教师观察 / 学生反馈计数。"""

    def test_06_ai_advice_count(self):
        records = [
            _session(1, start="2026-09-14 10:00:00", events=[
                _event("AI建议：降低后续学习负荷", type_="intervention"),
                _event("AI建议：适当休息", category="intervention"),
                _event("会话开始", type_="session_start"),
            ]),
            _session(2, start="2026-09-13 10:00:00", events=[
                _event("AI建议：调整节奏", type_="intervention",
                       category="intervention"),
            ]),
        ]
        summary = build_recent_summary(records, source="live")
        self.assertEqual(summary["stats"]["ai_advice_count"], 3)
        self.assertEqual(summary["sessions"][0]["ai_advice_count"], 2)

    def test_07_teacher_observation_count(self):
        records = [
            _session(1, start="2026-09-14 10:00:00", events=[
                _event("教师观察：专注下降", type_="teacher_observation",
                       source="teacher"),
                _event("教师观察：走神", source="teacher"),
                _event("会话开始", type_="session_start"),
            ]),
        ]
        summary = build_recent_summary(records, source="live")
        self.assertEqual(summary["stats"]["teacher_observation_count"], 2)

    def test_08_self_report_count(self):
        records = [
            _session(1, start="2026-09-14 10:00:00", events=[
                _event("感觉有点累", type_="self_report",
                       source="self_report"),
                _event("状态不错", source="self_report"),
                _event("会话开始", type_="session_start"),
            ]),
        ]
        summary = build_recent_summary(records, source="live")
        self.assertEqual(summary["stats"]["self_report_count"], 2)


class Round4BDemoIsolationTest(unittest.TestCase):
    """9～10：Demo 不进入默认 Live 趋势，且不混合统计。"""

    def test_09_demo_excluded_from_live_summary(self):
        live = _session(1, start="2026-09-14 10:00:00", attention=60.0)
        demo = _session(2, start="2026-09-13 10:00:00", attention=30.0,
                        source="mock", demo=True,
                        events=[_event("AI建议：降低后续学习负荷",
                                       type_="intervention")])
        summary = build_recent_summary([live, demo], source="live")
        self.assertEqual(summary["stats"]["count"], 1)
        self.assertAlmostEqual(summary["stats"]["avg_attention"]["value"], 60.0)
        self.assertEqual(summary["stats"]["ai_advice_count"], 0)
        self.assertFalse(summary["is_demo"])

    def test_10_live_and_demo_never_mix(self):
        live = _session(1, start="2026-09-14 10:00:00", attention=60.0)
        demo = _session(2, start="2026-09-13 10:00:00", attention=30.0,
                        source="mock", demo=True)
        # Demo 视图只统计演示记录
        demo_summary = build_recent_summary([live, demo], source="mock")
        self.assertEqual(demo_summary["stats"]["count"], 1)
        self.assertAlmostEqual(demo_summary["stats"]["avg_attention"]["value"], 30.0)
        self.assertTrue(demo_summary["is_demo"])
        # 不提供混合模式
        with self.assertRaises(ValueError):
            build_recent_summary([live, demo], source="all")


class Round4BTimingRuleTest(unittest.TestCase):
    """11～13：时间规律门槛与单次异常限制。"""

    def _timing_session(self, idx, day, offsets):
        start = f"2026-09-{day:02d} 10:00:00"
        base = _epoch(start)
        events = [
            _event("AI建议：降低后续学习负荷", type_="intervention",
                   timestamp=base + minutes * 60)
            for minutes in offsets
        ]
        return _session(idx, start=start, events=events,
                        tasks=[_task(start=start)])

    def test_11_insufficient_history_no_pattern(self):
        records = [self._timing_session(1, 14, [16])]
        summary = build_recent_summary(records, source="live")
        self.assertFalse(summary["event_timing"]["available"])
        self.assertEqual(summary["event_timing"]["text"], TIMING_INSUFFICIENT)
        self.assertIn("不足", summary["performance_summary"][0])

    def test_12_timing_computed_when_data_sufficient(self):
        records = [
            self._timing_session(1, 14, [16]),
            self._timing_session(2, 13, [18]),
            self._timing_session(3, 12, [15, 19]),
        ]
        summary = build_recent_summary(records, source="live")
        timing = summary["event_timing"]
        self.assertTrue(timing["available"])
        self.assertEqual(timing["sample_count"], 4)
        self.assertEqual(timing["distinct_sessions"], 3)
        self.assertIn("约15～19分钟后", timing["text"])

    def test_13_single_session_anomaly_no_long_term_conclusion(self):
        # 单次会话内 3 个事件，仍不足以生成长期规律（须跨 ≥2 个会话）
        records = [self._timing_session(1, 14, [10, 15, 20])]
        summary = build_recent_summary(records, source="live")
        self.assertFalse(summary["event_timing"]["available"])
        # 单次负荷建议也不触发"多次负荷"表现总结
        self.assertNotIn("负荷", "".join(summary["performance_summary"]))


class Round4BSummaryRuleTest(unittest.TestCase):
    """表现总结 / 教学参考规则。"""

    def test_stable_case(self):
        records = [
            _session(i, start=f"2026-09-{i:02d} 10:00:00") for i in (14, 13, 12)
        ]
        summary = build_recent_summary(records, source="live")
        self.assertIn("整体较稳定", "".join(summary["performance_summary"]))
        self.assertTrue(summary["teaching_suggestions"])
        self.assertIn("保持", summary["teaching_suggestions"][0])

    def test_load_case_and_suggestion_wording(self):
        records = [
            _session(i, start=f"2026-09-{i:02d} 10:00:00", duration=1800.0,
                     events=[_event("AI建议：降低后续学习负荷",
                                    type_="intervention")])
            for i in (14, 13, 12)
        ]
        summary = build_recent_summary(records, source="live")
        performance = "".join(summary["performance_summary"])
        self.assertIn("负荷", performance)
        joined = "\n".join(summary["teaching_suggestions"])
        # 措辞必须为建议式
        self.assertTrue(
            any(k in joined for k in ("建议", "参考", "可考虑"))
        )
        for forbidden in ("必须", "诊断为", "疾病"):
            self.assertNotIn(forbidden, joined)
        self.assertNotIn(forbidden, performance)

    def test_distracted_case(self):
        records = [
            _session(i, start=f"2026-09-{i:02d} 10:00:00", events=[
                _event("教师观察：学生走神", type_="teacher_observation",
                       source="teacher"),
            ])
            for i in (14, 13, 12)
        ]
        summary = build_recent_summary(records, source="live")
        self.assertIn("走神", "".join(summary["performance_summary"]))


class Round4BReadOnlyTest(unittest.TestCase):
    """14～15：零副作用 + UI 集成。"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_14_summary_does_not_modify_history(self):
        records = [
            _session(1, start="2026-09-14 10:00:00", attention=50.0,
                     events=[_event("AI建议：降低后续学习负荷",
                                    type_="intervention")]),
            _session(2, start="2026-09-13 10:00:00"),
        ]
        before = [r.to_dict() for r in records]
        summary = build_recent_summary(records, source="live")
        after = [r.to_dict() for r in records]
        self.assertEqual(before, after)
        self.assertEqual(summary["stats"]["count"], 2)

    def test_15_dashboard_teacher_overview_integration(self):
        from pages.dashboard_page import DashboardPage

        state = DashboardState(seed_demo_history=False)
        state.mode = "live"
        page = DashboardPage(state, _StartService(state))
        page.set_role("teacher")

        live = _session(1, start="2026-09-14 10:00:00", attention=52.0)
        demo = _session(2, start="2026-09-13 10:00:00", source="mock",
                        demo=True, attention=30.0)
        fake_store = MagicMock()
        fake_store.load.return_value = [live.to_dict(), demo.to_dict()]
        with patch("pages.dashboard_page.SessionStore",
                   return_value=fake_store):
            # 选择学生 st_001 → 自动加载概览
            page._teacher_student_combo.addItem("学生一", "st_001")
            page._teacher_student_combo.setCurrentIndex(
                page._teacher_student_combo.count() - 1
            )
            text_live = page._recent_overview_label.text()
        self.assertIn("最近 1 次", text_live)
        self.assertIn("实时采集", text_live)
        self.assertNotIn("【教学演示数据】", text_live)
        self.assertIn("52.0", text_live)
        self.assertNotIn("30.0", text_live)  # Demo 未混入 Live 统计

        # 切换到教学演示来源 → 标注【教学演示数据】且只统计 Demo
        with patch("pages.dashboard_page.SessionStore",
                   return_value=fake_store):
            page._recent_source_combo.setCurrentIndex(1)
            text_demo = page._recent_overview_label.text()
        self.assertIn("【教学演示数据】", text_demo)
        self.assertIn("30.0", text_demo)
        self.assertNotIn("52.0", text_demo)

        # 零副作用：源记录未被修改
        self.assertEqual(live.to_dict()["session_id"], "s_1")

    def test_16_real_store_path_reads_existing_history(self):
        """回归：真实 SessionStore(root) 路径解析（不 mock）。

        Round 4B 首版误用无参 SessionStore()（root 必填），
        导致教师端显示"近期学习概览暂时不可用"。本测试用真实
        临时目录走完整链路，确保已有 session.json 可直接读取。
        """
        import tempfile

        from pages.dashboard_page import DashboardPage
        from services.session_store import SessionStore

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root)
            # 2 条已有历史（少于 5 条不是错误，正常显示现有条数）
            for idx, (day, attention) in enumerate(
                [(14, 52.0), (13, 48.0)], start=1
            ):
                record = _session(
                    idx, start=f"2026-09-{day:02d} 10:00:00",
                    attention=attention,
                )
                store.save(record)

            state = DashboardState(seed_demo_history=False)
            state.mode = "live"
            service = SimpleNamespace(sessions_dir=str(root))
            page = DashboardPage(state, service)
            page.set_role("teacher")
            page._teacher_student_combo.addItem("学生一", "st_001")
            page._teacher_student_combo.setCurrentIndex(
                page._teacher_student_combo.count() - 1
            )
            text = page._recent_overview_label.text()
            # 绝不能出现加载失败占位文案
            self.assertNotIn("暂时不可用", text)
            self.assertIn("最近 2 次", text)
            self.assertIn("英语阅读", text)
            self.assertIn("50.0", text)  # (52+48)/2
            # 少于 3 次历史：仅长期趋势提示不足，不影响显示
            self.assertIn("不足", text)

    def test_17_absolute_and_relative_sessions_dir(self):
        """sessions_dir 相对路径基于包根目录解析（与 ReplayPage 一致）。"""
        from pages.dashboard_page import DashboardPage
        from services.teaching_store import TeacherSelectionContext

        state = DashboardState(seed_demo_history=False)
        state.mode = "live"
        service = SimpleNamespace(sessions_dir="data/sessions")
        # Round 4C：用独立选择上下文，保证"未选择学生"的前置条件
        # 不受共享单例中残留选择的影响。
        page = DashboardPage(state, service,
                             selection_context=TeacherSelectionContext())
        page.set_role("teacher")
        # 相对路径不应抛错；未选择学生时显示引导文案而非失败
        page._load_recent_overview()
        self.assertEqual(page._recent_overview_label.text(),
                         "请选择学生后查看近期学习概览")


if __name__ == "__main__":
    unittest.main()
