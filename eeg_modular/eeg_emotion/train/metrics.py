from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, target_names=list(class_names), output_dict=True)
    return {"accuracy": acc, "report": report}
