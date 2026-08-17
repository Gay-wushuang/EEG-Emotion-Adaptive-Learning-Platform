from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class RunPaths:
    run_dir: str
    figures_dir: str
    models_dir: str
    logs_dir: str
    artifacts_dir: str


def make_run_paths(base_dir: str = "outputs", run_name: Optional[str] = None) -> RunPaths:
    """Create a standard run directory structure.

    Layout:
      outputs/<run_name>/
        figures/
        models/
        logs/
        artifacts/
    """
    os.makedirs(base_dir, exist_ok=True)
    if not run_name:
        run_name = os.environ.get("RUN_NAME") or datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = os.path.join(base_dir, run_name)
    figures_dir = os.path.join(run_dir, "figures")
    models_dir = os.path.join(run_dir, "models")
    logs_dir = os.path.join(run_dir, "logs")
    artifacts_dir = os.path.join(run_dir, "artifacts")

    for d in (run_dir, figures_dir, models_dir, logs_dir, artifacts_dir):
        os.makedirs(d, exist_ok=True)

    return RunPaths(
        run_dir=run_dir,
        figures_dir=figures_dir,
        models_dir=models_dir,
        logs_dir=logs_dir,
        artifacts_dir=artifacts_dir,
    )
