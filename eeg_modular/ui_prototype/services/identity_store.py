"""Local identity persistence for the UI prototype.

Only a short local identifier, display name and last selected role are stored.
No EEG/session data or credentials are written by this module.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional


ROLE_STUDENT = "student"
ROLE_TEACHER = "teacher"
ROLE_RESEARCH = "research"
ROLE_ADMIN = ROLE_RESEARCH  # Compatibility alias; persisted value remains "research".
VALID_ROLES = (ROLE_STUDENT, ROLE_TEACHER, ROLE_RESEARCH)
ROLE_LABELS = {
    ROLE_STUDENT: "学生端",
    ROLE_TEACHER: "教师端",
    ROLE_RESEARCH: "管理端",
}

_USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{2,32}$")


def role_for_user_id(user_id: str) -> str:
    """Return the immutable role encoded by a local test-account prefix."""
    value = str(user_id or "").strip().lower()
    if value.startswith("st_"):
        return ROLE_STUDENT
    if value.startswith(("teacher", "tc_")):
        return ROLE_TEACHER
    if value.startswith("admin_"):
        return ROLE_RESEARCH
    raise ValueError(
        "账号 ID 必须以 st_、teacher、tc_ 或 admin_ 开头，角色由账号自动确定。"
    )


def default_identity_path() -> Path:
    """Return a per-user writable path without depending on a running Qt app."""
    appdata = os.environ.get("APPDATA")
    root = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    return root / "EEGLearningAssistant" / "identities.json"


class IdentityStore:
    """Small, resilient JSON store for local UI identities."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else default_identity_path()

    @staticmethod
    def validate(user_id: str, name: str) -> tuple[str, str]:
        user_id = user_id.strip()
        name = name.strip()
        if not _USER_ID_PATTERN.fullmatch(user_id):
            raise ValueError("ID 需为 2–32 位字母、数字、下划线、短横线或点号。")
        if not (1 <= len(name) <= 40):
            raise ValueError("姓名需为 1–40 个字符。")
        return user_id, name

    def _empty_data(self) -> Dict[str, object]:
        return {"version": 1, "last_user_id": None, "profiles": []}

    def _load(self) -> Dict[str, object]:
        if not self.path.exists():
            return self._empty_data()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return self._empty_data()

        profiles = raw.get("profiles", []) if isinstance(raw, dict) else []
        if not isinstance(profiles, list):
            profiles = []
        cleaned = []
        for profile in profiles:
            if not isinstance(profile, dict):
                continue
            try:
                user_id, name = self.validate(
                    str(profile.get("user_id", "")),
                    str(profile.get("name", "")),
                )
            except ValueError:
                continue
            try:
                role = role_for_user_id(user_id)
            except ValueError:
                continue
            cleaned.append({"user_id": user_id, "name": name, "last_role": role})

        last_user_id = raw.get("last_user_id") if isinstance(raw, dict) else None
        if last_user_id not in {item["user_id"] for item in cleaned}:
            last_user_id = None
        return {"version": 1, "last_user_id": last_user_id, "profiles": cleaned}

    def _save(self, data: Dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def list_profiles(self) -> List[Dict[str, str]]:
        return [dict(item) for item in self._load()["profiles"]]

    def get_profile(self, user_id: str) -> Optional[Dict[str, str]]:
        for profile in self.list_profiles():
            if profile["user_id"] == user_id:
                return profile
        return None

    def last_user_id(self) -> Optional[str]:
        value = self._load().get("last_user_id")
        return str(value) if value else None

    def save_profile(
        self,
        user_id: str,
        name: str,
        role: Optional[str] = None,
        *,
        make_current: bool = True,
    ) -> Dict[str, str]:
        user_id, name = self.validate(user_id, name)
        fixed_role = role_for_user_id(user_id)
        if role is not None and role != fixed_role:
            raise ValueError("账号角色由 ID 前缀固定，不能修改。")

        data = self._load()
        profile = {"user_id": user_id, "name": name, "last_role": fixed_role}
        profiles = data["profiles"]
        for index, existing in enumerate(profiles):
            if existing["user_id"] == user_id:
                profiles[index] = profile
                break
        else:
            profiles.append(profile)
        if make_current:
            data["last_user_id"] = user_id
        self._save(data)
        return dict(profile)

    def delete_profile(self, user_id: str) -> bool:
        data = self._load()
        profiles = data["profiles"]
        remaining = [item for item in profiles if item["user_id"] != user_id]
        if len(remaining) == len(profiles):
            return False
        data["profiles"] = remaining
        if data.get("last_user_id") == user_id:
            data["last_user_id"] = remaining[0]["user_id"] if remaining else None
        self._save(data)
        return True
