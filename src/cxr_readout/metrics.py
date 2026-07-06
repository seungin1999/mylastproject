from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

LABEL_TO_ID = {"improving": 0, "stable": 1, "worsening": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def split_pipe(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split("|") if x.strip()]


def add_label_id(df: pd.DataFrame, *, label_col: str = "label") -> pd.DataFrame:
    out = df.copy()
    out = out[out[label_col].isin(LABEL_TO_ID)].copy()
    out["label_id"] = out[label_col].map(LABEL_TO_ID).astype(np.int64)
    return out


def metrics_from_probs(probs: np.ndarray, y: np.ndarray) -> dict:
    """Compute 3-way, changed-vs-stable, and direction-only metrics."""
    probs = np.asarray(probs, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    pred = probs.argmax(axis=1).astype(np.int64)

    changed = (y != LABEL_TO_ID["stable"]).astype(np.int64)
    changed_pred = (pred != LABEL_TO_ID["stable"]).astype(np.int64)
    changed_score = (
        probs[:, LABEL_TO_ID["improving"]] + probs[:, LABEL_TO_ID["worsening"]]
    ).astype(np.float32)

    changed_auc = None
    if len(np.unique(changed)) == 2:
        changed_auc = float(roc_auc_score(changed, changed_score))

    changed_mask = y != LABEL_TO_ID["stable"]
    direction_auc = None
    direction_balacc = None
    if changed_mask.any():
        y_dir = (y[changed_mask] == LABEL_TO_ID["worsening"]).astype(np.int64)
        dir_score = (
            probs[changed_mask, LABEL_TO_ID["worsening"]]
            - probs[changed_mask, LABEL_TO_ID["improving"]]
        ).astype(np.float32)
        if len(np.unique(y_dir)) == 2:
            direction_auc = float(roc_auc_score(y_dir, dir_score))
        direction_balacc = float(balanced_accuracy_score(y_dir, (dir_score >= 0).astype(np.int64)))

    recalls = {}
    for label_id, name in ID_TO_LABEL.items():
        mask = y == label_id
        recalls[f"recall_{name}"] = float((pred[mask] == y[mask]).mean()) if mask.any() else None
    macro_accuracy = float(np.mean([v for v in recalls.values() if v is not None]))

    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_accuracy": macro_accuracy,
        "changed_balacc": float(balanced_accuracy_score(changed, changed_pred)),
        "changed_auc": changed_auc,
        "direction_balacc": direction_balacc,
        "direction_auc": direction_auc,
        "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).astype(int).tolist(),
        **recalls,
    }
