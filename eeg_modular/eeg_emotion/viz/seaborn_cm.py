from __future__ import annotations

from typing import List, Optional

import numpy as np


def save_confusion_matrix_seaborn(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str,
    normalize: Optional[str] = None,  # None | 'true'
    title: str = "Confusion Matrix",
) -> None:
    """Seaborn-style confusion matrix heatmap.

    - If seaborn isn't installed, this will raise ImportError (caller should catch).
    - normalize='true' will normalize rows (true labels).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm = cm.astype(np.float32)

    if normalize == "true":
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    fmt = ".2f" if normalize == "true" else "d"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        linecolor="gray",
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
