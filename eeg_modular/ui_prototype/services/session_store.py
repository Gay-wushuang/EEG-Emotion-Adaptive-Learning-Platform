"""Local, auditable persistence for learning-session metadata.

The raw/replay CSV remains owned by the acquisition service.  This store writes
the small ``session.json`` companion used by the history page.  Formal records
live directly below ``data/sessions``; synthetic/demo records are kept below
``data/sessions/_demo`` so a normal history scan can never mix the two.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Iterable, Mapping


SESSION_METADATA_FILENAME = "session.json"
DEMO_DIRECTORY_NAME = "_demo"


def resolve_sessions_root(state, service=None) -> Path:
    """UI"打开会话文件夹"统一入口：返回 SessionStore 当前真实根目录。

    解析顺序（与持久化完全一致，不在 UI 层重新硬编码路径）：
    1. DashboardState 当前 SessionStore 实例的 root；
    2. DashboardState 配置的 _sessions_dir（恒为绝对路径）；
    3. 回退：service.sessions_dir，相对路径基于包根目录解析
       （历史遗留行为：曾基于进程 CWD 解析，导致打开了
       ui_prototype/data/sessions 空目录）。
    """
    store = getattr(state, "_session_store", None)
    root = getattr(store, "root", None) if store is not None else None
    if root is not None:
        return Path(root)
    configured = getattr(state, "_sessions_dir", None)
    if configured is not None:
        return Path(configured)
    folder = Path(getattr(service, "sessions_dir", "data/sessions"))
    if not folder.is_absolute():
        folder = Path(__file__).resolve().parents[2] / folder
    return folder.resolve()


class SessionStore:
    """Persist and index versioned session dictionaries.

    The class deliberately does not import ``DashboardState`` or its data
    classes.  Accepting mappings (or objects exposing ``to_dict``) keeps the
    storage layer independent and avoids a circular dependency.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.last_errors: list[str] = []

    @property
    def demo_root(self) -> Path:
        return self.root / DEMO_DIRECTORY_NAME

    def session_dir(self, session_id: str, *, demo: bool = False) -> Path:
        safe_id = self._safe_session_id(session_id)
        base = self.demo_root if demo else self.root
        return base / safe_id

    def metadata_path(self, session_id: str, *, demo: bool = False) -> Path:
        return self.session_dir(session_id, demo=demo) / SESSION_METADATA_FILENAME

    def save(self, record) -> Path:
        """Atomically write one record and return its metadata path."""
        payload = record.to_dict() if hasattr(record, "to_dict") else dict(record)
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            raise ValueError("session_id 不能为空")
        payload.setdefault("schema_version", 2)
        payload["session_id"] = session_id
        demo = bool(payload.get("demo", False))
        path = self.metadata_path(session_id, demo=demo)
        path.parent.mkdir(parents=True, exist_ok=True)

        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    def load(self, *, include_demo: bool = False) -> list[dict]:
        """Index completed session metadata, newest first.

        Corrupt/incomplete files are skipped and reported through
        ``last_errors`` so one bad session cannot make the history page fail.
        Running sessions are intentionally hidden until their final save is
        confirmed.
        """
        self.last_errors = []
        candidates = list(self._metadata_files(self.root, formal_only=True))
        if include_demo:
            candidates.extend(self._metadata_files(self.demo_root, formal_only=False))

        records: list[dict] = []
        for path in candidates:
            try:
                with path.open("r", encoding="utf-8-sig") as handle:
                    payload = json.load(handle)
                if not isinstance(payload, dict):
                    raise ValueError("顶层必须为 JSON 对象")
                if not payload.get("session_id"):
                    raise ValueError("缺少 session_id")
                if payload.get("status", "completed") == "running":
                    continue
                records.append(payload)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self.last_errors.append(f"{path}: {exc}")

        records.sort(
            key=lambda item: str(item.get("start_time", "")),
            reverse=True,
        )
        return records

    @staticmethod
    def _metadata_files(root: Path, *, formal_only: bool) -> Iterable[Path]:
        if not root.exists():
            return ()
        files = []
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if formal_only and child.name.startswith("_"):
                continue
            candidate = child / SESSION_METADATA_FILENAME
            if candidate.is_file():
                files.append(candidate)
        return files

    @staticmethod
    def _safe_session_id(value: str) -> str:
        session_id = str(value).strip()
        if not session_id or session_id in {".", ".."}:
            raise ValueError("无效的 session_id")
        if Path(session_id).name != session_id or any(
            separator in session_id for separator in ("/", "\\")
        ):
            raise ValueError("session_id 不能包含路径分隔符")
        return session_id
