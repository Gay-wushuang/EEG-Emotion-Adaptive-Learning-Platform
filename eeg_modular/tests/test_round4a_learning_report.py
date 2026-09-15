"""Round 4A：学生单次学习报告（只读）测试。

覆盖：
- 任务名称 / 难度 / duration_seconds（绝不重算 start/end）；
- 多任务 Session 全部显示；
- Demo / Live 来源标识；
- avg_attention / probability_summary 缺失时“暂无数据”且不用 0% 兜底；
- 教师观察 / self_report / intervention 事件展示；
- AI 建议仅转述事件 label/note，绝不写成已执行难度调整；
- Baseline 对比与缺失兜底；
- 打开报告零副作用（不改 SessionRecord / Event 数量）；
- History Demo/Live 来源筛选继续可用。
"""

from __future__ import annotations

import copy
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

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
from services.session_report_builder import (
    MISSING,
    MISSING_AI,
    MISSING_BASELINE,
    MISSING_FEEDBACK,
    build_learning_report,
)
from pages.history_page import HistoryPage


def _task(name="英语阅读", difficulty="medium", duration=120.0, status="completed"):
    # start/end 跨度 20 分钟，但 duration_seconds 只有 2 分钟：
    # 用于验证报告绝不通过 end - start 重算任务时长（暂停已排除）。
    return TaskRecord(
        name=name, difficulty=difficulty, status=status,
        start_time="2026-09-13 10:00:00", end_time="2026-09-13 10:20:00",
        duration_seconds=duration,
    )


def _event(label, category="user", content="", note="", source="live",
           type_="marker", ts=1000.0, time_=""):
    return EventMarker(
        label=label, category=category, content=content, note=note,
        source=source, type=type_, timestamp=ts,
        time=time_ or "2026-09-13T10:05:00+08:00",
    )


def _record(**overrides):
    base = dict(
        session_id="s_r4a", user_id="st_001", user_name="学生一",
        start_time="2026-09-13T10:00:00+08:00",
        end_time="2026-09-13T10:40:00+08:00",
        duration_seconds=300.0,
        avg_attention=52.3, avg_meditation=46.1,
        positive_ratio=0.5, neutral_ratio=0.3, negative_ratio=0.2,
        event_count=3, notes="英语阅读",
        source="live", demo=False,
        tasks=[_task()],
        events=[
            _event("会话开始", category="system", source="system",
                   type_="session_start", content="会话开始", ts=1000.0),
            _event("专注下降", category="teacher", source="teacher",
                   type_="teacher_observation", content="教师观察：专注有所下降",
                   note="已提醒学生调整坐姿", ts=1001.0),
            _event("学习感受", category="user", source="self_report",
                   type_="self_report", content="感觉有点累", ts=1002.0),
            _event("AI建议：降低后续学习负荷", category="intervention",
                   source="system", type_="intervention",
                   content="AI建议：降低后续学习负荷",
                   note="建议调整后续难度或学习节奏（未自动修改当前任务）；原因：负荷偏高",
                   ts=1003.0),
        ],
        quality_summary={"sample_count": 100, "trusted_count": 90,
                         "warning_count": 8, "rejected_count": 2,
                         "trusted_ratio": 0.9, "usable_ratio": 0.98},
        probability_summary={"sample_count": 80, "mean_positive": 0.5,
                             "mean_neutral": 0.3, "mean_negative": 0.2,
                             "dominant_state": "positive",
                             "final_state": "positive"},
    )
    base.update(overrides)
    return SessionRecord(**base)


BASELINE_OK = {
    "status": "COMPLETED", "completed_at": "2026-09-01 09:00:00",
    "duration_seconds": 120.0, "avg_attention": 48.0, "avg_meditation": 44.0,
    "quality_rate": 0.9, "sample_count": 512,
}


class Round4ALearningReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    # ── 1～3：任务名称 / 难度 / duration_seconds ──
    def test_01_single_task_name(self):
        report = build_learning_report(_record())
        self.assertEqual(report["tasks"][0]["name"], "英语阅读")

    def test_02_task_difficulty_display(self):
        report = build_learning_report(
            _record(tasks=[_task(difficulty="hard")])
        )
        self.assertEqual(report["tasks"][0]["difficulty_display"], "困难")

    def test_03_task_duration_uses_record_value(self):
        report = build_learning_report(_record(tasks=[_task(duration=120.0)]))
        self.assertEqual(report["tasks"][0]["duration_seconds"], 120.0)
        self.assertEqual(report["tasks"][0]["duration_text"], "2分0秒")

    def test_04_multi_task_session_all_shown(self):
        tasks = [
            _task(name="英语阅读", difficulty="easy", duration=60.0),
            _task(name="数学练习", difficulty="medium", duration=90.0),
            _task(name="生词复习", difficulty="hard", duration=30.0),
        ]
        report = build_learning_report(_record(tasks=tasks))
        self.assertEqual(len(report["tasks"]), 3)
        self.assertEqual(
            [t["name"] for t in report["tasks"]],
            ["英语阅读", "数学练习", "生词复习"],
        )
        self.assertEqual([t["duration_text"] for t in report["tasks"]],
                         ["1分0秒", "1分30秒", "0分30秒"])

    # ── 5：暂停时间不会通过 start/end 重算 ──
    def test_05_pause_time_not_recomputed_from_start_end(self):
        # 任务 start/end 跨度 20 分钟、会话 start/end 跨度 40 分钟，
        # 但持久化 duration 分别为 120s / 300s（暂停已被排除）。
        record = _record(duration_seconds=300.0, tasks=[_task(duration=120.0)])
        report = build_learning_report(record)
        self.assertEqual(report["tasks"][0]["duration_seconds"], 120.0)
        self.assertEqual(report["duration_seconds"], 300.0)
        self.assertEqual(report["duration_text"], "5分0秒")
        self.assertNotIn("20分", report["tasks"][0]["duration_text"])
        self.assertNotIn("40分", report["duration_text"])

    # ── 6～7：Demo / Live 标识 ──
    def test_06_demo_report_has_demo_badge(self):
        record = _record(source="mock", demo=True)
        report = build_learning_report(record)
        self.assertTrue(report["is_demo"])
        self.assertEqual(report["source_display"], "教学演示")

    def test_07_live_report_has_no_demo_mark(self):
        record = _record(source="live", demo=False)
        report = build_learning_report(record)
        self.assertFalse(report["is_demo"])
        self.assertEqual(report["source_display"], "实时采集")
        self.assertNotIn("教学演示", report["source_display"])

    # ── 8～9：缺失数据显示暂无数据，不用 0% 兜底 ──
    def test_08_missing_attention_shows_missing(self):
        record = _record(avg_attention=0.0, avg_meditation=0.0)
        report = build_learning_report(record)
        self.assertFalse(report["attention"]["available"])
        self.assertEqual(report["attention"]["text"], MISSING)
        self.assertEqual(report["meditation"]["text"], MISSING)

    def test_09_missing_probability_never_shows_zero_pct(self):
        record = _record(probability_summary={})
        report = build_learning_report(record)
        state = report["state_summary"]
        self.assertFalse(state["available"])
        for key in ("mean_positive_text", "mean_neutral_text",
                    "mean_negative_text"):
            self.assertEqual(state[key], MISSING)
        dumped = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("0.0%", dumped)
        self.assertNotIn("0%", dumped)
        # 基线缺失时也不能拿 0% 顶替
        self.assertFalse(report["baseline"]["available"])

    # ── 10：教师观察事件中文展示 ──
    def test_10_teacher_observation_displayed(self):
        report = build_learning_report(_record())
        teacher_items = [i for i in report["timeline"]
                         if i["source_display"] == "教师观察"]
        self.assertEqual(len(teacher_items), 1)
        self.assertIn("教师观察：专注有所下降", teacher_items[0]["content"])
        self.assertEqual(teacher_items[0]["note"], "已提醒学生调整坐姿")

    # ── 11～12：学生主观反馈 ──
    def test_11_self_report_displayed(self):
        report = build_learning_report(_record())
        feedback = report["self_feedback"]
        self.assertTrue(feedback["available"])
        self.assertEqual(feedback["items"][0]["text"], "感觉有点累")

    def test_12_no_self_report_shows_placeholder(self):
        record = _record(events=[
            _event("会话开始", category="system", source="system",
                   type_="session_start", content="会话开始"),
        ])
        report = build_learning_report(record)
        self.assertFalse(report["self_feedback"]["available"])
        self.assertEqual(report["self_feedback"]["empty_text"],
                         MISSING_FEEDBACK)

    # ── 13～14：AI 学习建议 ──
    def test_13_ai_intervention_events_displayed(self):
        report = build_learning_report(_record())
        advice = report["ai_advice"]
        self.assertTrue(advice["available"])
        self.assertEqual(advice["items"][0]["label"],
                         "AI建议：降低后续学习负荷")
        self.assertIn("未自动修改当前任务", advice["items"][0]["note"])

    def test_14_ai_advice_never_claims_executed_adjustment(self):
        report = build_learning_report(_record())
        dumped = json.dumps(report, ensure_ascii=False)
        for forbidden in ("已自动降低", "已降低任务难度", "已自动调整",
                          "已执行", "已修改当前任务"):
            self.assertNotIn(forbidden, dumped)
        # AI 建议必须是建议口吻
        self.assertIn("AI建议", dumped)
        self.assertEqual(report["summary_text"],
                         "本次学习过程中出现阶段性专注下降或学习负荷偏高，"
                         "后续可适当调整连续学习时间或学习节奏。")

    # ── 15～16：基线参考 ──
    def test_15_baseline_comparison_displayed(self):
        report = build_learning_report(_record(), baseline=BASELINE_OK)
        baseline = report["baseline"]
        self.assertTrue(baseline["available"])
        joined = "\n".join(baseline["lines"])
        self.assertIn("本次平均专注度：52.3", joined)
        self.assertIn("个人基线平均专注度：48.0", joined)
        self.assertIn("本次平均放松度：46.1", joined)
        self.assertIn("个人基线平均放松度：44.0", joined)
        # 不计算百分比提升
        self.assertNotIn("%", joined)

    def test_16_baseline_missing_shows_placeholder(self):
        report = build_learning_report(_record(), baseline=None)
        self.assertFalse(report["baseline"]["available"])
        self.assertEqual(report["baseline"]["empty_text"], MISSING_BASELINE)
        # 会话侧缺失也无法对比
        report2 = build_learning_report(
            _record(avg_attention=0.0, avg_meditation=0.0), baseline=BASELINE_OK
        )
        self.assertFalse(report2["baseline"]["available"])

    # ── 17～18：打开报告零副作用 ──
    def test_17_opening_report_does_not_change_record(self):
        state = DashboardState(seed_demo_history=False)
        page = HistoryPage(state, SimpleNamespace(sessions_dir="data/sessions"))
        page._baseline_store = SimpleNamespace(
            get=lambda student_id: BASELINE_OK
        )
        record = _record()
        before = record.to_dict()
        page._show_report(record)
        self.assertIsNotNone(page._report_dialog)
        self.assertEqual(record.to_dict(), before)
        # 关闭报告同样零副作用
        page._report_dialog.close()
        self.assertEqual(record.to_dict(), before)

    def test_18_opening_report_does_not_change_events(self):
        state = DashboardState(seed_demo_history=False)
        page = HistoryPage(state, SimpleNamespace(sessions_dir="data/sessions"))
        page._baseline_store = SimpleNamespace(get=lambda student_id: None)
        record = _record()
        count_before = len(record.events)
        history_before = list(state._history_sessions)
        page._show_report(record)
        page._report_dialog.close()
        self.assertEqual(len(record.events), count_before)
        self.assertEqual(state._events, [])
        self.assertEqual(list(state._history_sessions), history_before)

    # ── 19：History Demo/Live 筛选继续可用 ──
    def test_19_history_source_filter_still_works(self):
        state = DashboardState(seed_demo_history=False)
        state._history_sessions = [
            _record(session_id="live_1", source="live", demo=False),
            _record(session_id="mock_1", source="mock", demo=True),
        ]
        page = HistoryPage(state, SimpleNamespace(sessions_dir="data/sessions"))

        page._combo_source.setCurrentIndex(0)  # 全部（默认索引不触发信号）
        page._refresh_table()
        self.assertEqual(page._table.rowCount(), 2)

        page._combo_source.setCurrentIndex(1)  # 实时采集
        ids = [s.session_id for s in page._get_sorted_sessions()]
        self.assertEqual(ids, ["live_1"])

        page._combo_source.setCurrentIndex(2)  # 教学演示
        ids = [s.session_id for s in page._get_sorted_sessions()]
        self.assertEqual(ids, ["mock_1"])

    # ── 附加：报告视图演示标识（UI 层）──
    def test_20_dialog_demo_badge_visibility(self):
        from pages.history_page import SessionReportDialog

        demo_report = build_learning_report(_record(source="mock", demo=True))
        dialog = SessionReportDialog(demo_report)
        self.assertFalse(dialog._demo_badge.isHidden())

        live_report = build_learning_report(_record(source="live"))
        dialog2 = SessionReportDialog(live_report)
        self.assertTrue(dialog2._demo_badge.isHidden())

    # ── 附加：无干预 + 数据齐全 → 稳定总结；无数据 → 信息不足总结 ──
    def test_21_summary_rules(self):
        stable = build_learning_report(_record(events=[
            _event("会话开始", category="system", source="system",
                   type_="session_start", content="会话开始"),
        ]))
        self.assertIn("整体较稳定", stable["summary_text"])

        empty = build_learning_report(
            _record(
                avg_attention=0.0, avg_meditation=0.0,
                probability_summary={}, quality_summary={},
                events=[_event("会话开始", category="system", source="system",
                               type_="session_start", content="会话开始")],
            )
        )
        self.assertIn("暂无足够数据", empty["summary_text"])


class Round4AFixesTest(unittest.TestCase):
    """Round 4A 人工验收修复：时长规则 / 默认筛选 / 多任务展示。"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def _make_page(self, records, mode="live"):
        state = DashboardState(seed_demo_history=False)
        state.mode = mode
        state._history_sessions = list(records)
        # on_show 会 reload_history；测试中打桩避免读取磁盘真实数据。
        state.reload_history = lambda **kw: list(records)
        page = HistoryPage(state, SimpleNamespace(sessions_dir="data/sessions"))
        return state, page

    # ── 1～2：97.9 秒统一显示为 1分37秒 ──
    def test_22_duration_979_shows_1min37s_everywhere(self):
        # 学习报告
        report = build_learning_report(
            _record(tasks=[_task(name="英语阅读", duration=97.9)],
                    duration_seconds=97.9)
        )
        self.assertEqual(report["tasks"][0]["duration_text"], "1分37秒")
        self.assertEqual(report["duration_text"], "1分37秒")
        # History 列表
        self.assertEqual(HistoryPage._fmt_duration(97.9), "1分37秒")
        # 任务页（int 截断规则：97.9 → 97 → 01:37）
        elapsed = 97.9
        mins, secs = int(elapsed) // 60, int(elapsed) % 60
        self.assertEqual(f"{mins:02d}:{secs:02d}", "01:37")
        # 三处秒数一致，绝不允许四舍五入成 1分38秒 / 01:38
        self.assertNotIn("1分38秒", json.dumps(report, ensure_ascii=False))

    def test_23_all_duration_formats_share_truncation_rule(self):
        from services.session_report_builder import _duration_text

        for seconds in (97.9, 59.9, 60.5, 125.7, 3600.99, 0.4, 0.0):
            truncated = int(max(0, seconds))
            expect = f"{truncated // 60}分{truncated % 60}秒"
            self.assertEqual(_duration_text(seconds), expect)
            self.assertEqual(HistoryPage._fmt_duration(seconds), expect)

    # ── 3～5：进入 History 时默认筛选跟随当前模式 ──
    def test_24_live_mode_defaults_to_live_filter(self):
        record = _record(session_id="live_1", source="live")
        state, page = self._make_page([record], mode="live")
        page.on_show()
        self.assertEqual(page._combo_source.currentIndex(), 1)
        self.assertEqual(page._combo_source.currentData(), "live")
        self.assertEqual(
            page._demo_banner.text(), "当前显示实时采集学习记录。"
        )
        self.assertEqual(page._table.rowCount(), 1)

    def test_25_mock_mode_defaults_to_mock_filter(self):
        record = _record(session_id="mock_1", source="mock", demo=True)
        state, page = self._make_page([record], mode="mock")
        page.on_show()
        self.assertEqual(page._combo_source.currentIndex(), 2)
        self.assertEqual(page._combo_source.currentData(), "mock")
        self.assertEqual(
            page._demo_banner.text(),
            "当前显示教学演示记录，与正式学习记录分开保存。"
        )
        self.assertEqual(page._table.rowCount(), 1)

    def test_26_user_can_still_switch_to_all(self):
        live = _record(session_id="live_1", source="live")
        mock = _record(session_id="mock_1", source="mock", demo=True)
        state, page = self._make_page([live, mock], mode="live")
        page.on_show()
        # 手动切换到"全部"
        page._combo_source.setCurrentIndex(0)
        self.assertEqual(page._combo_source.currentData(), "all")
        self.assertEqual(page._table.rowCount(), 2)
        self.assertEqual(
            page._demo_banner.text(),
            "当前显示全部学习记录，教学演示与正式学习记录分别保存。"
        )

    def test_27_switching_filter_never_modifies_history(self):
        live = _record(session_id="live_1", source="live")
        mock = _record(session_id="mock_1", source="mock", demo=True)
        state, page = self._make_page([live, mock], mode="live")
        before = [r.to_dict() for r in state._history_sessions]
        for index in (0, 1, 2, 0):
            page._combo_source.setCurrentIndex(index)
        self.assertEqual(page._table.rowCount(), 2)  # 全部：记录完整可见
        after = [r.to_dict() for r in state._history_sessions]
        self.assertEqual(before, after)

    # ── 7～9：多任务 Session 在 History 中完整展示 ──
    def _multi_task_record(self):
        tasks = [
            _task(name="英语阅读", difficulty="medium", duration=97.9),
            _task(name="物理复习", difficulty="hard", duration=125.7),
        ]
        return _record(session_id="multi_1", tasks=tasks, duration_seconds=223.6)

    def test_28_multi_task_list_shows_both_names(self):
        state, page = self._make_page([self._multi_task_record()])
        page._refresh_table()
        cell = page._table.item(0, 1).text()  # 任务列（新列序 index 1）
        self.assertEqual(cell, "英语阅读 + 物理复习")
        self.assertNotEqual(cell, "物理复习")  # 不只显示最后一个任务

    def test_29_multi_task_details_show_both_tasks(self):
        state, page = self._make_page([self._multi_task_record()])
        page._refresh_table()
        page._table.selectRow(0)
        self.assertEqual(page._detail_labels["task"].text(),
                         "英语阅读 + 物理复习")
        segments = page._tasks_label.text()
        self.assertIn("任务段：2项", segments)
        self.assertIn("英语阅读", segments)
        self.assertIn("物理复习", segments)

    def test_30_multi_task_details_show_each_duration(self):
        state, page = self._make_page([self._multi_task_record()])
        page._refresh_table()
        page._table.selectRow(0)
        segments = page._tasks_label.text()
        self.assertIn("有效学习：1分37秒", segments)  # 97.9 → 97
        self.assertIn("有效学习：2分5秒", segments)   # 125.7 → 125
        self.assertIn("难度：中等", segments)
        self.assertIn("难度：困难", segments)
        self.assertIn("状态：已完成", segments)

    def test_30b_three_tasks_collapse_to_count(self):
        tasks = [
            _task(name="英语阅读", duration=60.0),
            _task(name="物理复习", duration=60.0),
            _task(name="生词复习", duration=60.0),
        ]
        state, page = self._make_page([_record(tasks=tasks)])
        page._refresh_table()
        self.assertEqual(page._table.item(0, 1).text(), "3项任务")  # 任务列（新列序 index 1）
        page._table.selectRow(0)
        self.assertIn("任务段：3项", page._tasks_label.text())

    # ── 10：学习报告同时显示全部 Task ──
    def test_31_report_lists_all_tasks_with_own_durations(self):
        report = build_learning_report(self._multi_task_record())
        self.assertEqual(len(report["tasks"]), 2)
        self.assertEqual([t["name"] for t in report["tasks"]],
                         ["英语阅读", "物理复习"])
        self.assertEqual([t["duration_seconds"] for t in report["tasks"]],
                         [97.9, 125.7])
        self.assertEqual([t["duration_text"] for t in report["tasks"]],
                         ["1分37秒", "2分5秒"])
        self.assertEqual([t["difficulty_display"] for t in report["tasks"]],
                         ["中等", "困难"])


if __name__ == "__main__":
    unittest.main()
