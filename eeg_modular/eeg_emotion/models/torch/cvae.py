from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CVAEConfig:
    latent_dim: int = 64
    checkpoint: str = ""
    py_path: Optional[str] = None  # optional path to cvae_model.py
    strict: bool = False


def _import_cvae_module(py_path: Optional[str] = None):
    if py_path:
        py_path = os.path.abspath(py_path)
        spec = importlib.util.spec_from_file_location("cvae_model_dyn", py_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import CVAE module from {py_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return importlib.import_module("cvae_model")


def load_cvae(
    input_dim: int,
    num_classes: int,
    cfg: CVAEConfig,
    device: torch.device,
) -> nn.Module:
    """Load your CVAE checkpoint. Compatible with:
    - checkpoint is dict with 'model_state_dict'
    - checkpoint is state_dict directly
    """
    if not cfg.checkpoint:
        raise ValueError("CVAE checkpoint path is empty")
    mod = _import_cvae_module(cfg.py_path)
    if not hasattr(mod, "CVAE"):
        raise AttributeError("cvae_model module has no attribute 'CVAE'")
    CVAE = getattr(mod, "CVAE")

    model = CVAE(input_dim=input_dim, latent_dim=int(cfg.latent_dim), num_classes=int(num_classes))
    ckpt = torch.load(cfg.checkpoint, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=bool(cfg.strict))
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model
