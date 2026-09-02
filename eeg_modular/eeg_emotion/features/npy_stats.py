"""NPY data loader for sklearn models.

This module provides functions to load and format NPY data from time_data_preprocess
for use with sklearn models. It handles multi-modal data and flattens 3D arrays
to 2D for sklearn compatibility.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np


@dataclass(frozen=True)
class MultiModalNPYConfig:
    """Configuration for loading multi-modal NPY data."""
    data_dir: str
    modalities: List[str]
    x_train_prefix: str = "X_train_"
    x_test_prefix: str = "X_test_"
    y_train_name: str = "y_train_filtered.npy"
    y_test_name: str = "y_test_filtered.npy"
    label_encoder_name: str = "label_encoder.joblib"
    onehot_encoder_name: str = "onehot_encoder.joblib"


def flatten_3d_to_2d(arr: np.ndarray) -> np.ndarray:
    """Flatten 3D array (N, T, F) to 2D (N, T*F).

    Args:
        arr: 3D array of shape (N, T, F)

    Returns:
        2D array of shape (N, T*F)
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape={arr.shape}")
    n, t, f = arr.shape
    return arr.reshape(n, t * f)


def load_modality_features(
    data_dir: str,
    modality: str,
    x_train_prefix: str = "X_train_",
    x_test_prefix: str = "X_test_",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and flatten features for a single modality.

    Args:
        data_dir: Directory containing NPY files
        modality: Modality name (e.g., "filtered", "powerspec", "att", "med")
        x_train_prefix: Prefix for training data files
        x_test_prefix: Prefix for test data files

    Returns:
        Tuple of (X_train, X_test) as 2D arrays
    """
    train_path = os.path.join(data_dir, f"{x_train_prefix}{modality}.npy")
    test_path = os.path.join(data_dir, f"{x_test_prefix}{modality}.npy")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing training file: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test file: {test_path}")

    X_train = np.load(train_path)
    X_test = np.load(test_path)

    # Flatten 3D to 2D for sklearn compatibility
    if X_train.ndim == 3:
        X_train = flatten_3d_to_2d(X_train)
    if X_test.ndim == 3:
        X_test = flatten_3d_to_2d(X_test)

    return X_train.astype(np.float32), X_test.astype(np.float32)


def load_multimodal_npy_for_sklearn(
    cfg: MultiModalNPYConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load multi-modal NPY data for sklearn models.

    This function loads multi-modal NPY data and concatenates features
    from all modalities. Labels are loaded and converted to integer format.

    Args:
        cfg: Configuration for loading NPY data

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, class_names)
        - X_train: Training features (2D array)
        - X_test: Test features (2D array)
        - y_train: Training labels (1D integer array)
        - y_test: Test labels (1D integer array)
        - class_names: List of class names

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If data dimensions don't match
    """
    # Load and concatenate features from all modalities
    X_train_list: List[np.ndarray] = []
    X_test_list: List[np.ndarray] = []

    for modality in cfg.modalities:
        X_train_mod, X_test_mod = load_modality_features(
            data_dir=cfg.data_dir,
            modality=modality,
            x_train_prefix=cfg.x_train_prefix,
            x_test_prefix=cfg.x_test_prefix,
        )
        X_train_list.append(X_train_mod)
        X_test_list.append(X_test_mod)

    # Concatenate features from all modalities
    X_train = np.concatenate(X_train_list, axis=1)
    X_test = np.concatenate(X_test_list, axis=1)

    # Load labels
    y_train_path = os.path.join(cfg.data_dir, cfg.y_train_name)
    y_test_path = os.path.join(cfg.data_dir, cfg.y_test_name)

    if not os.path.exists(y_train_path):
        raise FileNotFoundError(f"Missing training labels: {y_train_path}")
    if not os.path.exists(y_test_path):
        raise FileNotFoundError(f"Missing test labels: {y_test_path}")

    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)

    # Convert to integer labels
    if y_train.ndim == 2:  # One-hot encoded
        y_train = np.argmax(y_train, axis=1)
    y_train = y_train.astype(np.int64).ravel()

    if y_test.ndim == 2 and y_test.shape[1] > 1:  # One-hot encoded
        y_test = np.argmax(y_test, axis=1)
    elif y_test.ndim == 2 and y_test.shape[1] == 1:  # Shape (N, 1)
        y_test = y_test.ravel()
    y_test = y_test.astype(np.int64).ravel()

    # Load class names from label encoder (avoid sklearn import issues)
    le_path = os.path.join(cfg.data_dir, cfg.label_encoder_name)
    try:
        if os.path.exists(le_path):
            # Try to load without importing sklearn
            import pickle
            with open(le_path, 'rb') as f:
                le = pickle.load(f)
            class_names = list(getattr(le, "classes_", []))
        else:
            # Fallback: infer from labels
            num_classes = int(np.max(y_train)) + 1
            class_names = [str(i) for i in range(num_classes)]
    except Exception:
        # Fallback: infer from labels
        num_classes = int(np.max(y_train)) + 1
        class_names = [str(i) for i in range(num_classes)]

    # Validate dimensions
    n_train = len(y_train)
    n_test = len(y_test)

    if X_train.shape[0] != n_train:
        raise ValueError(
            f"X_train has {X_train.shape[0]} samples but y_train has {n_train}"
        )
    if X_test.shape[0] != n_test:
        raise ValueError(
            f"X_test has {X_test.shape[0]} samples but y_test has {n_test}"
        )

    return X_train, X_test, y_train, y_test, class_names


def get_npy_feature_info(cfg: MultiModalNPYConfig) -> Dict[str, Dict[str, int]]:
    """Get information about NPY feature dimensions.

    Args:
        cfg: Configuration for loading NPY data

    Returns:
        Dictionary mapping modality names to their shape information
    """
    info = {}

    for modality in cfg.modalities:
        train_path = os.path.join(cfg.data_dir, f"{cfg.x_train_prefix}{modality}.npy")
        test_path = os.path.join(cfg.data_dir, f"{cfg.x_test_prefix}{modality}.npy")

        if os.path.exists(train_path):
            arr = np.load(train_path)
            info[modality] = {
                "train_samples": arr.shape[0],
                "train_shape": arr.shape,
            }

        if os.path.exists(test_path):
            arr = np.load(test_path)
            if modality in info:
                info[modality]["test_samples"] = arr.shape[0]
                info[modality]["test_shape"] = arr.shape
            else:
                info[modality] = {
                    "test_samples": arr.shape[0],
                    "test_shape": arr.shape,
                }

    return info