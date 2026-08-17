from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def setup_logging(log_file_path: str, level: int = logging.INFO, name: str = "eeg_emotion") -> logging.Logger:
    """Configure a logger that logs to both file and stdout.

    This avoids calling logging.basicConfig() globally, which can be hard to control in large projects.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # prevent duplicate logs if root logger is configured elsewhere

    # Clear existing handlers if re-initialized (e.g., notebooks)
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_file_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(level)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
