"""Small local stores for teacher/student relationships and assignments."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from contextlib import contextmanager
import time
import uuid
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


def _app_dir() -> Path:
    appdata = os.environ.get("APPDATA")
    root = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    return root / "EEGLearningAssistant"


class _JsonStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _read(self, fallback):
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value
        except (OSError, UnicodeError, json.JSONDecodeError):
            return fallback

    def _write(self, value) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)


class TeacherStudentStore(_JsonStore):
    def __init__(self, path: Optional[Path] = None, identity_store=None):
        super().__init__(path or (_app_dir() / "teacher_students.json"))
        if identity_store is None:
            from services.identity_store import IdentityStore
            identity_store = IdentityStore()
        self.identity_store = identity_store

    def students_for(self, teacher_id: str) -> list[str]:
        data = self._read({"version": 1, "bindings": {}})
        values = data.get("bindings", {}).get(teacher_id, []) if isinstance(data, dict) else []
        return sorted({str(value) for value in values if str(value).startswith("st_")})

    def add(self, teacher_id: str, student_id: str) -> bool:
        if not str(teacher_id).lower().startswith(("teacher", "tc_")):
            raise ValueError("无效教师账号。")
        profile = self.identity_store.get_profile(student_id)
        if profile is None:
            raise ValueError("未找到该学生账号")
        if profile.get("last_role") != "student":
            raise ValueError("该账号不是学生身份")
        data = self._read({"version": 1, "bindings": {}})
        bindings = data.setdefault("bindings", {})
        students = bindings.setdefault(teacher_id, [])
        if student_id in students:
            return False
        students.append(student_id)
        self._write(data)
        return True

    def remove(self, teacher_id: str, student_id: str) -> bool:
        data = self._read({"version": 1, "bindings": {}})
        students = data.setdefault("bindings", {}).setdefault(teacher_id, [])
        if student_id not in students:
            return False
        students.remove(student_id)
        self._write(data)
        return True


@dataclass
class AssignmentRecord:
    assignment_id: str
    teacher_id: str
    student_id: str
    task_type: str
    difficulty: str
    note: str
    created_at: float
    status: str = "pending"


class AssignmentStore(_JsonStore):
    VALID_STATUS = {"pending", "running", "paused", "completed", "cancelled"}

    def __init__(self, path: Optional[Path] = None):
        super().__init__(path or (_app_dir() / "assignments.json"))

    def list_all(self) -> list[AssignmentRecord]:
        raw = self._read({"version": 1, "assignments": []})
        result = []
        for item in raw.get("assignments", []) if isinstance(raw, dict) else []:
            try:
                result.append(AssignmentRecord(**item))
            except (TypeError, ValueError):
                continue
        return result

    def publish(self, teacher_id: str, student_id: str, task_type: str,
                difficulty: str, note: str = "") -> AssignmentRecord:
        if not student_id:
            raise ValueError("发布任务前必须选择学生。")
        record = AssignmentRecord(
            assignment_id=f"A{uuid.uuid4().hex[:12]}", teacher_id=teacher_id,
            student_id=student_id, task_type=task_type, difficulty=difficulty,
            note=note, created_at=time.time(), status="pending",
        )
        records = self.list_all()
        records.append(record)
        self._write({"version": 1, "assignments": [asdict(item) for item in records]})
        return record

    def for_student(self, student_id: str) -> list[AssignmentRecord]:
        return [item for item in self.list_all() if item.student_id == student_id]

    def for_teacher(self, teacher_id: str, student_id: str = "") -> list[AssignmentRecord]:
        return [item for item in self.list_all()
                if item.teacher_id == teacher_id and (not student_id or item.student_id == student_id)]

    def update_status(self, assignment_id: str, status: str) -> bool:
        if status not in self.VALID_STATUS:
            raise ValueError("未知任务安排状态。")
        records = self.list_all()
        changed = False
        for item in records:
            if item.assignment_id == assignment_id:
                item.status = status
                changed = True
                break
        if changed:
            self._write({"version": 1, "assignments": [asdict(item) for item in records]})
        return changed


class BaselineResultStore(_JsonStore):
    """Persisted, per-student baseline summaries for read-only teacher views."""

    def __init__(self, path: Optional[Path] = None):
        super().__init__(path or (_app_dir() / "baseline_results.json"))

    def get(self, student_id: str) -> Optional[dict]:
        data = self._read({"version": 1, "results": {}})
        value = data.get("results", {}).get(str(student_id)) if isinstance(data, dict) else None
        return dict(value) if isinstance(value, dict) else None

    def save(self, student_id: str, result: dict) -> dict:
        data = self._read({"version": 1, "results": {}})
        if not isinstance(data, dict):
            data = {"version": 1, "results": {}}
        results = data.setdefault("results", {})
        results[str(student_id)] = dict(result)
        self._write(data)
        return dict(results[str(student_id)])


class StudentRuntimeRegistry:
    """Low-frequency SQLite snapshots shared by local application processes."""
    _shared = None

    def __init__(self, path: Optional[Path] = None, *, stale_seconds: float = 5.0,
                 publish_interval: float = 0.25):
        self.path = Path(path or (_app_dir() / "coordination.sqlite3"))
        self.stale_seconds = float(stale_seconds)
        self.publish_interval = float(publish_interval)
        self._states = {}
        self._last_publish = {}
        try:
            self._ensure_schema()
        except (OSError, sqlite3.Error):
            self.path = Path(tempfile.gettempdir()) / "EEGLearningAssistant" / "coordination.sqlite3"
            self._ensure_schema()

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=2.0)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _ensure_schema(self):
        with self._connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS student_runtime ("
                "student_id TEXT PRIMARY KEY, snapshot_json TEXT NOT NULL, "
                "updated_at REAL NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS teacher_event_inbox ("
                "event_id TEXT PRIMARY KEY, student_id TEXT NOT NULL, teacher_id TEXT NOT NULL, "
                "session_id TEXT NOT NULL, task_id TEXT NOT NULL, label TEXT NOT NULL, "
                "note TEXT NOT NULL, created_at REAL NOT NULL, delivered INTEGER NOT NULL DEFAULT 0)"
            )

    @classmethod
    def shared(cls):
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    def publish(self, state, *, force: bool = False) -> None:
        student_id = str(getattr(state, "_user_id", ""))
        if not student_id.startswith("st_"):
            return
        now = time.time()
        if not force and now - self._last_publish.get(student_id, 0.0) < self.publish_interval:
            return
        # Deliver queued teacher observations before composing the read-only
        # snapshot, so the same canonical student Event appears in the timeline.
        self._drain_teacher_events(student_id, state)
        task = state._task_by_id(getattr(state, "current_task_id", ""))
        last_completed_task = next(
            (item for item in reversed(getattr(state, "_tasks", []))
             if getattr(item, "status", "") != "running"),
            None,
        )
        active_task_id = getattr(state, "current_task_id", "")
        last_completed_task_id = getattr(last_completed_task, "task_id", "")
        displayed_task_id = active_task_id or last_completed_task_id
        displayed_task = task or last_completed_task
        raw = list(getattr(state, "_eeg_raw_buffer", []))[-1024:]
        display_step = max(1, len(raw) // 128)
        snapshot = {
            "student_id": student_id,
            "student_name": getattr(state, "_user_name", ""),
            "data_mode": "demo" if getattr(state, "mode", "live") == "mock" else "live",
            "baseline_status": getattr(state, "baseline_status", "IDLE"),
            "baseline_elapsed": getattr(state, "_baseline_elapsed", 0.0),
            "baseline_samples": getattr(state, "_baseline_samples", 0),
            "baseline_avg_attention": getattr(state, "_baseline_avg_attention", None),
            "baseline_progress": getattr(state, "_baseline_progress", 0.0),
            "baseline_avg_meditation": getattr(state, "_baseline_avg_meditation", None),
            "baseline_quality_rate": getattr(state, "_baseline_quality_rate", None),
            "baseline_completed_at": getattr(state, "_baseline_completed_at", ""),
            "baseline_failure_reason": getattr(state, "_baseline_failure_reason", ""),
            # Heartbeat freshness represents the student application; device
            # availability is reported separately by device_status.
            "online": True,
            "connector_status": getattr(state, "connector_status", "offline"),
            "device_status": getattr(state, "device_status", "offline"),
            "poor_signal": getattr(state, "poor_signal", None),
            "sample_rate": getattr(state, "sample_rate_hz", None),
            "session_status": getattr(state, "session_status", "IDLE"),
            "session_id": getattr(state, "run_id", "") if getattr(state, "session_active", False) else "",
            "task_id": getattr(state, "current_task_id", ""),
            "active_task_id": active_task_id,
            "last_completed_task_id": last_completed_task_id,
            "displayed_task_id": displayed_task_id,
            "assignment_id": getattr(displayed_task, "assignment_id", "") if displayed_task else "",
            "task_name": getattr(displayed_task, "name", "") if displayed_task else "",
            "task_status": getattr(displayed_task, "status", "") if displayed_task else "",
            "elapsed_seconds": getattr(state, "current_task_elapsed_seconds", 0.0),
            "quality_level": getattr(state, "quality_level", "rejected"),
            "stable_state": getattr(state, "stable_state", None),
            "attention": getattr(state, "attention", None),
            "meditation": getattr(state, "meditation", None),
            "advice": getattr(state, "adaptive_feedback_text", ""),
            "probabilities": {
                "positive": getattr(state, "prob_positive", None),
                "neutral": getattr(state, "prob_neutral", None),
                "negative": getattr(state, "prob_negative", None),
            },
            "raw_eeg": raw[::display_step][-128:],
            "attention_history": list(getattr(state, "_attention_history", []))[-300:],
            "meditation_history": list(getattr(state, "_meditation_history", []))[-300:],
            "recent_events": [
                event.to_dict() for event in getattr(state, "_events", [])
                if getattr(event, "session_id", "") == getattr(state, "run_id", "")
                and getattr(event, "task_id", "") in {
                    "", displayed_task_id
                }
            ][-100:],
            "updated_at": now,
        }
        payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO student_runtime(student_id,snapshot_json,updated_at) "
                "VALUES(?,?,?) ON CONFLICT(student_id) DO UPDATE SET "
                "snapshot_json=excluded.snapshot_json, updated_at=excluded.updated_at",
                (student_id, payload, now),
            )
        self._last_publish[student_id] = now
        self._states[student_id] = weakref.ref(state)

    def get(self, student_id: str):
        try:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT snapshot_json, updated_at FROM student_runtime WHERE student_id=?",
                    (student_id,),
                ).fetchone()
        except sqlite3.OperationalError as exc:
            # UI timers may overlap test/local cleanup of the coordination DB.
            # Recreate only a missing schema; do not hide unrelated SQLite errors.
            if "no such table" not in str(exc).lower():
                raise
            self._ensure_schema()
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT snapshot_json, updated_at FROM student_runtime WHERE student_id=?",
                    (student_id,),
                ).fetchone()
        if not row:
            return None
        try:
            value = json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            return None
        stale = time.time() - float(row[1]) > self.stale_seconds
        value["stale"] = stale
        if stale:
            value["online"] = False
        return value

    def mark_offline(self, student_id: str) -> None:
        snapshot = self.get(student_id)
        if not snapshot:
            return
        snapshot["online"] = False
        snapshot["session_status"] = "IDLE"
        snapshot["updated_at"] = time.time()
        snapshot.pop("stale", None)
        with self._connection() as connection:
            connection.execute(
                "UPDATE student_runtime SET snapshot_json=?, updated_at=? WHERE student_id=?",
                (json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")),
                 snapshot["updated_at"], student_id),
            )

    def add_teacher_event(self, student_id: str, teacher_id: str,
                          label: str, note: str = ""):
        snapshot = self.get(student_id)
        reference = self._states.get(student_id)
        state = reference() if reference else None
        if not snapshot or snapshot.get("stale"):
            return None
        if not snapshot["session_id"] or not snapshot["task_id"]:
            return None
        if not state or getattr(state, "_user_id", "") != student_id:
            event_id = f"TE{uuid.uuid4().hex[:12]}"
            with self._connection() as connection:
                connection.execute(
                    "INSERT INTO teacher_event_inbox VALUES(?,?,?,?,?,?,?,?,0)",
                    (event_id, student_id, teacher_id, snapshot["session_id"],
                     snapshot["task_id"], label, note, time.time()),
                )
            return event_id
        return state.add_event(
            label, "teacher", note, source="teacher", event_type="teacher_observation",
            session_id=snapshot["session_id"], task_id=snapshot["task_id"],
            observer_id=teacher_id, student_id=student_id,
        )

    def _drain_teacher_events(self, student_id: str, state) -> None:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT event_id,teacher_id,session_id,task_id,label,note "
                "FROM teacher_event_inbox WHERE student_id=? AND delivered=0 ORDER BY created_at",
                (student_id,),
            ).fetchall()
            for event_id, teacher_id, session_id, task_id, label, note in rows:
                state.add_event(
                    label, "teacher", note, source="teacher",
                    event_type="teacher_observation", session_id=session_id,
                    task_id=task_id, observer_id=teacher_id, student_id=student_id,
                )
                connection.execute(
                    "UPDATE teacher_event_inbox SET delivered=1 WHERE event_id=?",
                    (event_id,),
                )


class TeacherObserverService:
    """No-op service: teacher windows consume snapshots and never acquire EEG."""
    def start_streaming(self): pass
    def stop_streaming(self): pass
    def start_session(self): return None
    def pause_session(self): pass
    def resume_session(self): pass
    def end_session(self, status="completed"): return None


class TeacherSelectionContext:
    _shared = None

    def __init__(self):
        self.teacher_id = ""
        self.selected_student_id = ""

    @classmethod
    def shared(cls):
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    def set_teacher(self, teacher_id: str):
        teacher_id = str(teacher_id or "")
        if teacher_id != self.teacher_id:
            self.teacher_id = teacher_id
            self.selected_student_id = ""

    def select(self, student_id: str):
        self.selected_student_id = str(student_id or "")
