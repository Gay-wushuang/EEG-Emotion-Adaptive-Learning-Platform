from __future__ import annotations

import os
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    save_path: str,
    normalize: Optional[str] = None,  # None, 'true', 'pred', 'all'
    title: str = "Confusion Matrix",
    dpi: int = 180,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names))

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap=None, values_format=".2f" if normalize else "d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
