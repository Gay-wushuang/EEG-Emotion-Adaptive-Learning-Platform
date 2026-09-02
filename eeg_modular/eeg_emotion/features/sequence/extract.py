from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceFeatureConfig:
    data_dir: str
    emotions: List[str]
    csv_files: List[str]
    time_steps: int = 128
    min_cols_per_file: int = 10


def _infer_n_cols_per_file(data_dir: str, emotions: List[str], csv_files: List[str], min_cols: int) -> List[int]:
    n_cols_per_file: List[int] = []
    for csv_name in csv_files:
        max_cols = 0
        for emo in emotions:
            emo_dir = os.path.join(data_dir, emo)
            if not os.path.exists(emo_dir):
                continue
            for sample_name in os.listdir(emo_dir):
                folder_path = os.path.join(emo_dir, sample_name)
                file_path = os.path.join(folder_path, csv_name)
                if not os.path.exists(file_path):
                    continue
                df = pd.read_csv(file_path).select_dtypes(include=[np.number])
                max_cols = max(max_cols, int(df.shape[1]))
        n_cols_per_file.append(max(max_cols, min_cols))
    return n_cols_per_file


def extract_features_from_folder(folder_path: str, csv_files: List[str], time_steps: int, n_cols_per_file: List[int]) -> np.ndarray:
    """Return one sample as (time_steps, total_features) by concatenating per-csv numeric columns."""
    feats = []
    for idx, csv_name in enumerate(csv_files):
        file_path = os.path.join(folder_path, csv_name)
        n_cols = int(n_cols_per_file[idx])

        if not os.path.exists(file_path):
            feats.append(np.zeros((time_steps, n_cols), dtype=np.float32))
            continue

        df = pd.read_csv(file_path).select_dtypes(include=[np.number])
        data = df.values.astype(np.float32)

        # col pad/crop
        if data.shape[1] < n_cols:
            data = np.pad(data, ((0, 0), (0, n_cols - data.shape[1])), mode="constant")
        elif data.shape[1] > n_cols:
            data = data[:, :n_cols]

        # time pad/crop
        if data.shape[0] < time_steps:
            data = np.pad(data, ((0, time_steps - data.shape[0]), (0, 0)), mode="constant")
        elif data.shape[0] > time_steps:
            data = data[:time_steps, :]

        feats.append(data)

    combined = np.concatenate(feats, axis=1)
    return combined.astype(np.float32)


def extract_all_features(cfg: SequenceFeatureConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Build dataset: X (N, time_steps, total_features), y (N,) label index."""
    n_cols_per_file = _infer_n_cols_per_file(cfg.data_dir, cfg.emotions, cfg.csv_files, cfg.min_cols_per_file)

    X_list, y_list = [], []
    for label_idx, emo in enumerate(cfg.emotions):
        emo_dir = os.path.join(cfg.data_dir, emo)
        if not os.path.exists(emo_dir):
            continue
        for sample_name in os.listdir(emo_dir):
            folder_path = os.path.join(emo_dir, sample_name)
            try:
                x = extract_features_from_folder(folder_path, cfg.csv_files, cfg.time_steps, n_cols_per_file)
                X_list.append(x)
                y_list.append(label_idx)
            except Exception:
                # caller logs; keep going
                continue

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y
