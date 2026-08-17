from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore


class ConfigError(RuntimeError):
    pass


def load_config(path: str) -> Dict[str, Any]:
    """Load a config from YAML or JSON.

    YAML support requires PyYAML. If not installed, JSON is still supported.
    """
    if not os.path.exists(path):
        raise ConfigError(f"Config not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    if ext in (".yaml", ".yml"):
        if yaml is None:
            raise ConfigError("PyYAML is not installed. Install with: pip install pyyaml")
        cfg = yaml.safe_load(raw)
    elif ext == ".json":
        cfg = json.loads(raw)
    else:
        raise ConfigError(f"Unsupported config extension: {ext}")

    if not isinstance(cfg, dict):
        raise ConfigError("Config root must be a mapping/dict.")
    return cfg


def require(cfg: Dict[str, Any], key: str, expected_type: type) -> Any:
    if key not in cfg:
        raise ConfigError(f"Missing required config key: {key}")
    val = cfg[key]
    if not isinstance(val, expected_type):
        raise ConfigError(f"Config key '{key}' must be {expected_type.__name__}, got {type(val).__name__}")
    return val


def get(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    return cfg.get(key, default)
