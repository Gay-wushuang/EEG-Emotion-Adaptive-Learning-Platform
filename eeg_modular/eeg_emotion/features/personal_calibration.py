"""Unsupervised short-baseline calibration in production-scaled feature space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PersonalCalibrationConfig:
    minimum_windows: int = 16
    maximum_offset: float = 3.0
    required_trusted_fraction: float = 0.8


class PersonalBaselineCalibrator:
    """Remove a clipped personal median offset without using labels."""

    def __init__(self, config: PersonalCalibrationConfig | None = None):
        self.config = config or PersonalCalibrationConfig()
        self.offsets: dict[str, np.ndarray] = {}

    def fit(
        self,
        scaled_baseline: dict[str, np.ndarray],
        quality_levels: list[str] | np.ndarray,
    ) -> "PersonalBaselineCalibrator":
        levels = np.asarray(quality_levels)
        if not scaled_baseline:
            raise ValueError("No baseline modalities supplied")
        sample_count = len(next(iter(scaled_baseline.values())))
        if sample_count < self.config.minimum_windows:
            raise ValueError(
                f"Need at least {self.config.minimum_windows} baseline windows"
            )
        if len(levels) != sample_count:
            raise ValueError("Quality levels and baseline windows differ in length")
        trusted = levels == "trusted"
        trusted_fraction = float(trusted.mean())
        if trusted_fraction < self.config.required_trusted_fraction:
            raise ValueError(
                f"Trusted baseline fraction {trusted_fraction:.3f} is below "
                f"{self.config.required_trusted_fraction:.3f}"
            )
        for modality, values in scaled_baseline.items():
            if values.ndim != 3 or values.shape[1:] != (10, 4):
                raise ValueError(f"{modality} must have shape (N, 10, 4)")
            offset = np.median(values[trusted], axis=(0, 1))
            self.offsets[modality] = np.clip(
                offset, -self.config.maximum_offset, self.config.maximum_offset
            ).astype(np.float32)
        return self

    def transform(self, scaled: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not self.offsets:
            raise RuntimeError("Calibrator has not been fitted")
        if set(scaled) != set(self.offsets):
            raise ValueError("Modalities differ from fitted calibration contract")
        return {
            modality: (values - self.offsets[modality][None, None, :]).astype(
                np.float32
            )
            for modality, values in scaled.items()
        }

    def contract(self) -> dict:
        return {
            "method": "trusted_window_median_offset_in_scaled_space",
            "label_usage": "none",
            "minimum_windows": self.config.minimum_windows,
            "maximum_offset": self.config.maximum_offset,
            "required_trusted_fraction": self.config.required_trusted_fraction,
            "modalities": sorted(self.offsets),
            "offsets": {name: value.tolist() for name, value in self.offsets.items()},
        }
