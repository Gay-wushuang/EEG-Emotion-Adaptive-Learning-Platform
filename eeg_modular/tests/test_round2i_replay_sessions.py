"""Round 2I：离线数据回放 - 近期会话选择与可回放性校验。

覆盖：
1. 近期可回放会话发现（复用 SessionStore，不建第二套数据库）；
2. 显示学生/任务/时间，而不是 session_id；
3. 选择近期会话正确加载；
4. 无回放数据（无 CSV / 仅表头 / demo 无 CSV）的会话被排除；
5. 外部 CSV 导入；
6. 错误 schema CSV 给出明确错误；
7. 示例数据仍可用且标记演示；
8. 播放/暂停/停止/速度/帧更新；
9. 回放不写入学生 History、不改 runtime、不创建新会话文件。
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui_prototype"))

from PySide6.QtWidgets import QApplication
from services.dashboard_state import DashboardState
from services.session_store import SessionStore
from pages.replay_page import ReplayPage

CSV_HEADER = (
    "timestamp_unix,signal_time_seconds,raw_sample_index,raw,attention,meditation,"
    "poor_signal,prob_positive,prob_neutral,prob_negative,predicted_class,"
    "confidence,quality_level,inference_index"
)


def _write_session_json(sessions_root: Path, session_id: str, *, demo=False, user_id="st_001",
                        user_name="学生一", task="英语阅读", start_time="2026-09-13T22:57:18+08:00",
                        duration=300.0, data_files=None, status="completed"):
    base = sessions_root / "_demo" if demo else sessions_root
    directory = base / session_id
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "session_id": session_id,
        "source": "mock" if demo else "live",
        "demo": demo,
        "status": status,
        "user_id": user_id,
        "user_name": user_name,
        "start_time": start_time,
        "end_time": "2026-09-13T23:05:18+08:00" if status != "running" else "",
        "duration_seconds": duration,
        "tasks": [{
            "task_id": f"T{session_id}01",
            "session_id": session_id,
            "name": task,
            "difficulty": "medium",
            "start_time": start_time,
            "end_time": start_time,
            "status": "completed" if status != "running" else "running",
            "notes": "",
            "duration_seconds": duration,
            "assignment_id": "A001",
        }],
        "events": [],
        "quality_summary": {},
        "probability_summary": {},
        "data_files": dict(data_files or {"raw_csv": "session.csv"}),
        "notes": task,
    }
    (directory / "session.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[tuple], *, header=CSV_HEADER):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header.split(","))
        for row in rows:
            writer.writerow(row)


def _data_row(raw=26, att=37.0, med=41.0, prob=(0.5, 0.3, 0.2), pred="positive", index=0):
    return (
        1789314736.0, 0.000000, index, raw, att, med, 0,
        prob[0], prob[1], prob[2], pred, 0.8, "trusted", index,
    )


class Round2IReplaySessionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.sessions_root = Path(self.temp.name)
        # s1：真实可回放会话
        _write_session_json(self.sessions_root, "s1")
        _write_csv(self.sessions_root / "s1" / "session.csv", [
            _data_row(index=0), _data_row(raw=27, index=1), _data_row(raw=28, index=2),
        ])
        # s2：仅表头占位 CSV（不可回放）
        _write_session_json(self.sessions_root, "s2", start_time="2026-09-13T23:10:18+08:00")
        _write_csv(self.sessions_root / "s2" / "session.csv", [])
        # s3：demo 会话，无 CSV（不可回放）
        _write_session_json(self.sessions_root, "s3", demo=True, start_time="2026-09-13T23:20:18+08:00")
        # s4：可回放，且 data_files 自定义 CSV 名
        _write_session_json(
            self.sessions_root, "s4", user_id="st_002", user_name="学生二",
            task="数学练习", start_time="2026-09-13T23:30:18+08:00",
            data_files={"raw_csv": "raw.csv"},
        )
        _write_csv(self.sessions_root / "s4" / "raw.csv", [
            _data_row(index=0), _data_row(raw=30, index=1),
        ])
        # s5：running 会话（不应出现在列表中）
        _write_session_json(self.sessions_root, "s5", status="running",
                            start_time="2026-09-13T23:40:18+08:00")
        _write_csv(self.sessions_root / "s5" / "session.csv", [
            _data_row(index=0),
        ])

        self.state = DashboardState(seed_demo_history=False)
        self.service = SimpleNamespace(sessions_dir=str(self.sessions_root))
        self.page = ReplayPage(self.state, self.service)

    def tearDown(self):
        self.page._clear_data()
        self.temp.cleanup()

    # ── 1. 近期可回放会话发现 ──
    def test_01_recent_replayable_sessions(self):
        items = self.page._recent_replayable_sessions()
        ids = [record["session_id"] for record, _ in items]
        # s1、s4 可回放；s2（仅表头）、s3（demo 无 CSV）、s5（running）被排除
        self.assertEqual(ids, ["s4", "s1"])
        # 返回的路径真实存在
        for _, csv_path in items:
            self.assertTrue(csv_path.is_file())

    def test_02_display_shows_student_task_time(self):
        self.page._refresh_recent_sessions()
        count = self.page._combo_recent.count()
        self.assertEqual(count, 2)
        names = [self.page._combo_recent.itemText(i) for i in range(count)]
        joined = " | ".join(names)
        # 显示学生名/user_id、任务名、时间，而不是 session_id 作为主名称
        self.assertIn("学生一", joined)
        self.assertIn("学生二 (st_002)", joined)
        self.assertIn("英语阅读", joined)
        self.assertIn("数学练习", joined)
        self.assertIn("2026-09-13 23:30", joined)
        self.assertNotIn("s4", joined.split("(")[0].strip().split()[0] if "s4" in joined else joined)

    def test_03_load_selected_session(self):
        self.page._refresh_recent_sessions()
        with patch("pages.replay_page.QMessageBox.information"):
            self.page._load_selected_session()
        self.assertGreater(len(self.page._data), 0)
        self.assertFalse(self.page._is_sample)
        # 行被 normalize 成 ReplayPage 可用 schema
        self.assertEqual(self.page._data[0]["raw"], "26")
        self.assertEqual(self.page._data[0]["predicted_class"], "positive")

    def test_04_non_replayable_excluded(self):
        self.page._refresh_recent_sessions()
        texts = [self.page._combo_recent.itemText(i) for i in range(self.page._combo_recent.count())]
        joined = " | ".join(texts)
        self.assertNotIn("s2", joined)
        self.assertNotIn("s3", joined)
        self.assertNotIn("s5", joined)
        self.assertTrue(self.page._btn_load_recent.isEnabled())

    def test_05_external_csv_import(self):
        external = self.sessions_root / "external.csv"
        _write_csv(external, [_data_row(index=0), _data_row(index=1)])
        with patch("pages.replay_page.QMessageBox.information"):
            self.page.load_paths([str(external)])
        self.assertEqual(len(self.page._data), 2)
        self.assertFalse(self.page._is_sample)

    def test_06_wrong_schema_gives_clear_error(self):
        bad = self.sessions_root / "bad.csv"
        bad.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        with patch("pages.replay_page.QMessageBox.warning") as warning:
            self.page.load_paths([str(bad)])
        self.assertTrue(warning.called)
        message = warning.call_args[0][2]
        self.assertIn("缺少 Raw EEG 列", message)
        # 数据未被污染
        self.assertEqual(self.page._data, [])

    def test_06b_empty_csv_gives_clear_error(self):
        empty = self.sessions_root / "empty.csv"
        empty.write_text(CSV_HEADER + "\n", encoding="utf-8")
        with patch("pages.replay_page.QMessageBox.warning") as warning:
            self.page.load_paths([str(empty)])
        self.assertTrue(warning.called)
        self.assertIn("没有数据行", warning.call_args[0][2])

    # ── 7. 示例数据仍可用且标记演示 ──
    def test_07_sample_data_still_works(self):
        self.page._load_sample()
        self.assertGreater(len(self.page._data), 0)
        self.assertTrue(self.page._is_sample)
        # 页面未放入可见窗口时 isVisible() 恒为 False，改用 isHidden() 判断显隐状态
        self.assertFalse(self.page._demo_label.isHidden())
        self.assertIn("示例数据", self.page._label_file.text())

    # ── 8. 播放 / 暂停 / 停止 / 速度 / 帧更新 ──
    def test_08_play_pause_stop_speed_and_frame(self):
        self.page._load_sample()
        self.page._play()
        self.assertTrue(self.page._playing)
        self.assertFalse(self.page._btn_play.isEnabled())
        self.page._pause()
        self.assertFalse(self.page._playing)
        self.assertTrue(self.page._btn_play.isEnabled())

        self.page._change_speed(0)  # 0.5x
        self.assertEqual(self.page._speed, 0.5)
        self.assertEqual(self.page._timer.interval(), 200)
        self.page._change_speed(2)  # 2x
        self.assertEqual(self.page._speed, 2.0)
        self.assertEqual(self.page._timer.interval(), 50)

        self.page._update_frame(0)
        self.assertIn("预测状态", self.page._replay_pred.text())
        # 进度条可用
        self.assertTrue(self.page._slider.isEnabled())

        self.page._stop()
        self.assertEqual(self.page._index, 0)
        self.assertFalse(self.page._playing)

    # ── 9. 回放隔离：不写 History / 不创建会话文件 / 不改 runtime ──
    def test_09_replay_does_not_pollute_history_or_runtime(self):
        state = self.state
        history_before = list(state._history_sessions)
        run_id_before = state.run_id
        session_status_before = state.session_status
        files_before = {
            str(p.relative_to(self.sessions_root))
            for p in self.sessions_root.rglob("*") if p.is_file()
        }

        self.page._load_sample()
        self.page._update_frame(5)
        self.page._play()
        self.page._pause()
        self.page._clear_data()
        self.page._refresh_recent_sessions()

        self.assertEqual(list(state._history_sessions), history_before)
        self.assertEqual(state.run_id, run_id_before)
        self.assertEqual(state.session_status, session_status_before)
        self.assertFalse(getattr(state, "_session_active", False))
        # 未创建任何新文件
        files_after = {
            str(p.relative_to(self.sessions_root))
            for p in self.sessions_root.rglob("*") if p.is_file()
        }
        self.assertEqual(files_after, files_before)

    def test_09b_replay_does_not_touch_live_service(self):
        # ReplayPage 不应拥有 begin_session / finalize_session / add_event 调用路径
        source = io.StringIO(Path(__file__).resolve().parent.parent.joinpath(
            "ui_prototype", "pages", "replay_page.py"
        ).read_text(encoding="utf-8"))
        text = source.getvalue()
        # 用带括号的调用形式检查，避免命中模块文档字符串中的说明文字
        for forbidden in ("begin_session(", "finalize_session(", "add_event(",
                          "runtime_registry", "publish(", "set_baseline"):
            self.assertNotIn(forbidden, text,
                             f"ReplayPage 不应调用 {forbidden}")


if __name__ == "__main__":
    unittest.main()
