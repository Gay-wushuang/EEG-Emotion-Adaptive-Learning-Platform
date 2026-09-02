"""Frozen Quality Gate v1 evaluator."""

from __future__ import annotations


def evaluate_quality(metrics: dict[str, float], policy: dict) -> dict:
    warning_reasons, ood_reasons = [], []
    poor = policy["poor_signal_rules"]
    if metrics["poor_signal_bad_fraction"] > poor["warning_bad_fraction_greater_than"]:
        warning_reasons.append("poor_signal_nonzero")
    if (
        metrics["poor_signal_bad_fraction"] > poor["low_ood_bad_fraction_greater_than"]
        or metrics["poor_signal_mean"] >= poor["low_ood_mean_at_least"]
    ):
        ood_reasons.append("poor_signal")
    for name, limits in policy["limits"].items():
        value = metrics[name]
        if "ood_high" in limits and value > limits["ood_high"]:
            ood_reasons.append(f"{name}:high")
        elif "warning_high" in limits and value > limits["warning_high"]:
            warning_reasons.append(f"{name}:high")
        if "ood_low" in limits and value < limits["ood_low"]:
            ood_reasons.append(f"{name}:low")
        elif "warning_low" in limits and value < limits["warning_low"]:
            warning_reasons.append(f"{name}:low")
    level = "low_ood" if ood_reasons else ("warning" if warning_reasons else "trusted")
    return {
        "quality_level": level,
        "emotion_interpretation_allowed": level == "trusted",
        "warning_reasons": warning_reasons,
        "ood_reasons": ood_reasons,
    }

