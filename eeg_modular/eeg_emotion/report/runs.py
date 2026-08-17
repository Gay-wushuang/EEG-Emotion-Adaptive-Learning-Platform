from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore


@dataclass(frozen=True)
class RunRecord:
    run_dir: str
    timestamp: str
    accuracy: float
    model_type: str
    best_params: Dict[str, Any]
    config_path: Optional[str]


def _load_yaml_or_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    if ext in (".yaml", ".yml"):
        if yaml is None:
            return None
        out = yaml.safe_load(raw)
    elif ext == ".json":
        out = json.loads(raw)
    else:
        return None
    return out if isinstance(out, dict) else None


def infer_model_type_from_config(config_path: Optional[str]) -> str:
    cfg = _load_yaml_or_json(config_path or "")
    if not cfg:
        return "unknown"
    model = cfg.get("model", {})
    if isinstance(model, dict):
        t = model.get("type", "unknown")
        return str(t).lower()
    return "unknown"


def read_metrics(metrics_path: str) -> Dict[str, Any]:
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def scan_runs(outputs_dir: str = "outputs") -> List[RunRecord]:
    if not os.path.isdir(outputs_dir):
        raise FileNotFoundError(f"outputs_dir not found: {outputs_dir}")

    records: List[RunRecord] = []
    for name in sorted(os.listdir(outputs_dir)):
        run_dir = os.path.join(outputs_dir, name)
        metrics_path = os.path.join(run_dir, "metrics.json")
        if not os.path.isfile(metrics_path):
            continue

        m = read_metrics(metrics_path)
        acc = float(m.get("accuracy", 0.0))
        best_params = m.get("best_params") or {}
        if not isinstance(best_params, dict):
            best_params = {"value": best_params}

        config_path = m.get("config_path")
        if config_path is not None:
            config_path = str(config_path)

        model_type = infer_model_type_from_config(config_path)
        records.append(
            RunRecord(
                run_dir=run_dir,
                timestamp=name,
                accuracy=acc,
                model_type=model_type,
                best_params=best_params,
                config_path=config_path,
            )
        )
    return records
