from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np


def add_gaussian_noise(data: np.ndarray, mean: float = 0.0, std: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to a single sample (T,F) or a batch (N,T,F)."""
    noise = np.random.normal(mean, std, data.shape).astype(np.float32)
    return (data + noise).astype(np.float32)


def apply_gaussian_noise_batch(X: np.ndarray, mean: float = 0.0, std: float = 0.01) -> np.ndarray:
    """Apply Gaussian noise to a batch X: (N,T,F)."""
    if std <= 0:
        return X.astype(np.float32)
    return add_gaussian_noise(X, mean=mean, std=std)


def time_warp(data: np.ndarray, warp_factor: float = 0.1) -> np.ndarray:
    timesteps, features = data.shape
    new_timesteps = int(timesteps * (1 + np.random.uniform(-warp_factor, warp_factor)))
    new_timesteps = max(10, new_timesteps)
    warped = np.zeros((new_timesteps, features), dtype=np.float32)
    for f in range(features):
        warped[:, f] = np.interp(
            np.linspace(0, timesteps - 1, new_timesteps),
            np.arange(timesteps),
            data[:, f],
        )
    return warped


def amplitude_scaling(data: np.ndarray, scale_range=(0.9, 1.1)) -> np.ndarray:
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return data * scale


def random_crop_or_pad(data: np.ndarray, target_length: int) -> np.ndarray:
    timesteps, features = data.shape
    if timesteps > target_length:
        start = np.random.randint(0, timesteps - target_length)
        return data[start:start + target_length, :]
    if timesteps < target_length:
        pad_width = target_length - timesteps
        pad_left = np.random.randint(0, pad_width)
        pad_right = pad_width - pad_left
        return np.pad(data, ((pad_left, pad_right), (0, 0)), mode="constant")
    return data


def random_mask(data: np.ndarray, mask_prob: float = 0.05) -> np.ndarray:
    mask = (np.random.rand(*data.shape) > mask_prob).astype(np.float32)
    return data * mask


def augment_sample(sample: np.ndarray, target_length: int = 128, noise_mean: float = 0.0, noise_std: float = 0.01) -> np.ndarray:
    if random.random() < 0.5:
        sample = add_gaussian_noise(sample, mean=noise_mean, std=noise_std)
    if random.random() < 0.3:
        sample = time_warp(sample)
    if random.random() < 0.3:
        sample = amplitude_scaling(sample)
    if random.random() < 0.2:
        sample = random_mask(sample)
    sample = random_crop_or_pad(sample, target_length)
    return sample.astype(np.float32)


def augment_class_samples(
    X: np.ndarray,
    y: np.ndarray,
    target_labels: List[int],
    augment_times: int = 1,
    target_length: Optional[int] = None,
    noise_mean: float = 0.0,
    noise_std: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    if target_length is None:
        target_length = int(X.shape[1])

    X_aug, y_aug = [], []
    for label in target_labels:
        idxs = np.where(y == label)[0]
        for i in idxs:
            sample = X[i]
            for _ in range(int(augment_times)):
                aug_sample = augment_sample(sample, target_length=target_length, noise_mean=noise_mean, noise_std=noise_std)
                X_aug.append(aug_sample)
                y_aug.append(label)

    if X_aug:
        X_combined = np.concatenate([X, np.asarray(X_aug, dtype=np.float32)], axis=0)
        y_combined = np.concatenate([y, np.asarray(y_aug, dtype=np.int64)], axis=0)
        return X_combined, y_combined
    return X, y


def compute_sample_stats(X: np.ndarray) -> np.ndarray:
    stats = []
    for s in range(X.shape[0]):
        arr = X[s]
        stats.append([np.mean(arr), np.std(arr), np.min(arr), np.max(arr), np.median(arr)])
    return np.asarray(stats, dtype=np.float32)


def mixup_augment(X: np.ndarray, y_onehot: np.ndarray, alpha: float = 0.2, augment_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Mixup works for both vector features (N,D) and sequence tensors (N,T,F)."""
    n = X.shape[0]
    n_aug = int(n * augment_ratio)
    if n_aug <= 0:
        return X, y_onehot

    X_aug, y_aug = [], []
    for _ in range(n_aug):
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        lam = np.random.beta(alpha, alpha)
        X_new = lam * X[i] + (1.0 - lam) * X[j]
        y_new = lam * y_onehot[i] + (1.0 - lam) * y_onehot[j]
        X_aug.append(X_new)
        y_aug.append(y_new)

    X_aug = np.asarray(X_aug, dtype=np.float32)
    y_aug = np.asarray(y_aug, dtype=np.float32)
    return np.concatenate([X, X_aug], axis=0), np.concatenate([y_onehot, y_aug], axis=0)
