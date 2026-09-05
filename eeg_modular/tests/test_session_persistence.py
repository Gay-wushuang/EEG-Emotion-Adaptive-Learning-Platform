import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui_prototype"))

from services.dashboard_state import DashboardState, EventMarker
from services.session_store import SessionStore


class SessionPersistenceTest(unittest.TestCase):
    def test_event_contract_keeps_legacy_fields(self):
        event = EventMarker(
            timestamp=123.0,
            label="开始任务: 数学练习",
            category="system",
            source="live",
            type="task_start",
            session_id="session-1",
            task_id="task-1",
        )
        payload = event.to_dict()
        required = {"source", "type", "session_id", "task_id", "time", "content"}
        self.assertTrue(required.issubset(payload))
        self.assertEqual(payload["content"], payload["label"])
        self.assertEqual(payload["timestamp"], 123.0)

    def test_session_task_event_summary_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = DashboardState(
                sessions_dir=temp_dir,
                seed_demo_history=False,
            )
            state.configure_session_store(temp_dir)
            state.run_id = "live-session-1"
            state._user_id = "student-001"
            state._user_name = "匿名学生"
            state.session_seconds = 12.5

            started = state.begin_session(
                source="live", data_files={"raw_csv": "session.csv"}
            )
            self.assertIsNotNone(started)
            self.assertTrue(Path(started).is_file())
            task_id = state.begin_task("数学练习", "hard", "章节测试")
            state.add_event("题目作答", "user", "第3题", event_type="answer")
            state.end_task()

            state._attention_history.extend([60.0, 80.0])
            state._meditation_history.extend([40.0, 60.0])
            state._prob_history.extend([
                (1.0, 0.6, 0.3, 0.1),
                (2.0, 0.4, 0.4, 0.2),
            ])
            state.quality_level = "trusted"
            state.poor_signal = 10
            state.capture_session_snapshot()
            state.quality_level = "warning"
            state.poor_signal = 60
            state.capture_session_snapshot()

            saved = state.finalize_session()
            self.assertEqual(Path(saved), Path(started))
            with Path(saved).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            self.assertEqual(payload["source"], "live")
            self.assertFalse(payload["demo"])
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["tasks"][0]["task_id"], task_id)
            self.assertEqual(payload["tasks"][0]["status"], "completed")
            self.assertGreaterEqual(len(payload["events"]), 5)
            for event in payload["events"]:
                self.assertTrue({
                    "source", "type", "session_id", "task_id", "time", "content",
                }.issubset(event))
            self.assertEqual(payload["quality_summary"]["sample_count"], 2)
            self.assertAlmostEqual(payload["quality_summary"]["usable_ratio"], 1.0)
            self.assertAlmostEqual(
                payload["probability_summary"]["mean_positive"], 0.5
            )

            records = state.reload_history()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].primary_task, "数学练习")
            self.assertEqual(records[0].events[1].task_id, task_id)

    def test_demo_records_are_physically_and_logically_isolated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = DashboardState(
                sessions_dir=temp_dir,
                seed_demo_history=False,
            )
            state.configure_session_store(temp_dir)
            state.run_id = "mock-session-1"
            state.begin_session(source="mock", demo=True)
            saved = state.finalize_session()

            self.assertIn("_demo", Path(saved).parts)
            store = SessionStore(temp_dir)
            self.assertEqual(store.load(), [])
            self.assertEqual(len(store.load(include_demo=True)), 1)
            self.assertEqual(state.reload_history(include_demo=False), [])
            demos = state.reload_history(include_demo=True)
            self.assertEqual(len(demos), 1)
            self.assertTrue(demos[0].demo)
            self.assertEqual(demos[0].source, "mock")

    def test_pipeline_error_removes_stale_interpretation(self):
        state = DashboardState(seed_demo_history=False)
        state.prob_positive = 0.8
        state.prob_neutral = 0.1
        state.prob_negative = 0.1
        state.predicted_state = "positive"
        state.stable_state = "positive"
        state.confidence = 0.8

        state.set_pipeline_error("模型暂不可用", "checksum mismatch")

        self.assertEqual(state.pipeline_state, "error")
        self.assertEqual(state.quality_level, "rejected")
        self.assertEqual(state.model_error_user, "模型暂不可用")
        self.assertEqual(state.model_error_detail, "checksum mismatch")
        self.assertIsNone(state.prob_positive)
        self.assertIsNone(state.predicted_state)
        self.assertIsNone(state.stable_state)


if __name__ == "__main__":
    unittest.main()
