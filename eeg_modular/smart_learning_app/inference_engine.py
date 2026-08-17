"""Strict Raw-to-result inference facade for the desktop product."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np

from eeg_emotion.features.canonical import (
    CanonicalFeatureConfig,
    extract_canonical_features,
)
from eeg_emotion.models.production_baseline import (
    CLASS_NAMES,
    DISPLAY_NAME_BY_INTERNAL,
    MODALITIES,
    load_production_package,
    predict_probabilities,
)


@dataclass(frozen=True)
class InferenceResult:
    probabilities: tuple[float, float, float]
    internal_class: str
    display_class: str
    confidence: float
    accepted: bool
    latency_ms: float


class ProductionInferenceEngine:
    """Load and execute the frozen Production Baseline v1 package.

    ATT/MED are accepted only because the canonical feature extractor shares one
    Raw-to-feature interface. They are deliberately discarded before model
    inference and therefore cannot enter the emotion-classification tensor.
    """

    def __init__(
        self,
        package_dir: Path,
        *,
        device: str = "cpu",
        rejection_threshold: float | None = None,
    ) -> None:
        self.package_dir = Path(package_dir).resolve()
        self.device = device
        self.model, self.scalers, self.contract = load_production_package(
            self.package_dir, device=device
        )
        self.feature_config = CanonicalFeatureConfig()
        contract_threshold = float(self.contract["rejection_threshold"])
        if rejection_threshold is not None and rejection_threshold != contract_threshold:
            raise ValueError(
                "The product threshold is frozen by the production contract; "
                f"expected {contract_threshold}, got {rejection_threshold}"
            )
        self.rejection_threshold = contract_threshold

    @property
    def required_samples(self) -> int:
        return self.feature_config.window_samples

    def infer_window(
        self,
        raw_eeg: np.ndarray,
        attention: np.ndarray,
        meditation: np.ndarray,
    ) -> InferenceResult:
        started = perf_counter()
        features = extract_canonical_features(
            raw_eeg, attention, meditation, self.feature_config
        )
        classifier_inputs = {name: features[name][None, ...] for name in MODALITIES}
        probabilities = predict_probabilities(
            self.model, self.scalers, classifier_inputs, device=self.device
        )[0]
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        internal = CLASS_NAMES[index]
        return InferenceResult(
            probabilities=tuple(float(value) for value in probabilities),
            internal_class=internal,
            display_class=DISPLAY_NAME_BY_INTERNAL[internal],
            confidence=confidence,
            accepted=confidence >= self.rejection_threshold,
            latency_ms=(perf_counter() - started) * 1000.0,
        )

    def infer_features(self, arrays: Mapping[str, np.ndarray]) -> InferenceResult:
        """Inference hook for replay/tests; rejects ATT/MED and extra modalities."""
        if tuple(arrays.keys()) != MODALITIES:
            raise ValueError(
                f"Expected exactly {MODALITIES}; ATT/MED must not enter classifier"
            )
        started = perf_counter()
        batched = {
            name: np.asarray(arrays[name], dtype=np.float32)[None, ...]
            for name in MODALITIES
        }
        probabilities = predict_probabilities(
            self.model, self.scalers, batched, device=self.device
        )[0]
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        internal = CLASS_NAMES[index]
        return InferenceResult(
            probabilities=tuple(float(value) for value in probabilities),
            internal_class=internal,
            display_class=DISPLAY_NAME_BY_INTERNAL[internal],
            confidence=confidence,
            accepted=confidence >= self.rejection_threshold,
            latency_ms=(perf_counter() - started) * 1000.0,
        )

