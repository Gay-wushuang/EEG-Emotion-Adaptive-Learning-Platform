"""Teacher-to-student realtime notifications, isolated from History events."""

from __future__ import annotations

import math
import sqlite3
import struct
import tempfile
import time
import uuid
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QUrl
from PySide6.QtMultimedia import QSoundEffect


@dataclass(frozen=True)
class StudentNotification:
    notification_id: str
    student_id: str
    teacher_id: str
    label: str
    created_at: float


class NotificationService:
    """Small cross-process inbox keyed strictly by the real student ID."""

    def __init__(self, database_path: Path | str, *, binding_store=None):
        self.database_path = Path(database_path)
        self.binding_store = binding_store
        self._ensure_schema()

    def _connect(self):
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.database_path), timeout=2.0)
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
                "CREATE TABLE IF NOT EXISTS student_notification ("
                "notification_id TEXT PRIMARY KEY, student_id TEXT NOT NULL, "
                "teacher_id TEXT NOT NULL, label TEXT NOT NULL, "
                "created_at REAL NOT NULL, delivered_at REAL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_student_notification_pending "
                "ON student_notification(student_id, delivered_at, created_at)"
            )

    def publish(self, *, teacher_id: str, student_id: str, label: str) -> StudentNotification:
        teacher_id = str(teacher_id or "").strip()
        student_id = str(student_id or "").strip()
        label = str(label or "").strip()
        if not student_id.startswith("st_") or not teacher_id or not label:
            raise ValueError("教师通知缺少有效的教师、学生或标签。")
        if self.binding_store is not None and student_id not in self.binding_store.students_for(teacher_id):
            raise PermissionError("只能向当前教师已绑定的学生发送提醒。")
        item = StudentNotification(
            notification_id=f"N{uuid.uuid4().hex[:16]}", student_id=student_id,
            teacher_id=teacher_id, label=label, created_at=time.time(),
        )
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO student_notification VALUES(?,?,?,?,?,NULL)",
                (item.notification_id, item.student_id, item.teacher_id,
                 item.label, item.created_at),
            )
        return item

    def take_pending(self, student_id: str, *, limit: int = 10) -> list[StudentNotification]:
        student_id = str(student_id or "").strip()
        if not student_id.startswith("st_"):
            return []
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT notification_id,student_id,teacher_id,label,created_at "
                "FROM student_notification WHERE student_id=? AND delivered_at IS NULL "
                "ORDER BY created_at LIMIT ?", (student_id, max(1, int(limit))),
            ).fetchall()
            if rows:
                now = time.time()
                connection.executemany(
                    "UPDATE student_notification SET delivered_at=? WHERE notification_id=?",
                    [(now, row[0]) for row in rows],
                )
        return [StudentNotification(*row) for row in rows]


class NotificationSound(QObject):
    """Reusable Qt sound effect with a generated local notification tone."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.effect = QSoundEffect(self)
        self.effect.setVolume(0.55)
        self.effect.setSource(QUrl.fromLocalFile(str(self._tone_path())))

    @staticmethod
    def _tone_path() -> Path:
        path = Path(tempfile.gettempdir()) / "eeg_learning_teacher_notification.wav"
        if path.exists():
            return path
        sample_rate, duration, frequency = 22050, 0.18, 880.0
        frames = bytearray()
        for index in range(int(sample_rate * duration)):
            envelope = min(1.0, index / 300) * max(0.0, 1.0 - index / (sample_rate * duration))
            value = int(10000 * envelope * math.sin(2 * math.pi * frequency * index / sample_rate))
            frames.extend(struct.pack("<h", value))
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1); output.setsampwidth(2); output.setframerate(sample_rate)
            output.writeframes(frames)
        return path

    def play(self):
        self.effect.stop()
        self.effect.play()
