"""TamperDetector: AUROC-based separation scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


class TamperDetector:
    def __init__(self, base_projections: np.ndarray, test_projections: np.ndarray):
        self.base = base_projections
        self.test = test_projections
        self.threshold: float | None = None

    def compute_auroc(self) -> float:
        # test=1 (tampered/anomalous), base=0 (clean); negate so test has higher scores
        y_true = np.array([0] * len(self.base) + [1] * len(self.test))
        scores = np.concatenate([-self.base, -self.test])
        return float(roc_auc_score(y_true, scores))

    def compute_separation(self) -> dict:
        pooled_std = np.sqrt(
            (self.base.std() ** 2 + self.test.std() ** 2) / 2
        )
        cohen_d = float((self.base.mean() - self.test.mean()) / (pooled_std + 1e-9))
        return {
            "base_mean": float(self.base.mean()),
            "test_mean": float(self.test.mean()),
            "cohen_d": cohen_d,
        }

    def fit_threshold(self, fpr_target: float = 0.05) -> float:
        # base=0 (clean/negative class), threshold at fpr_target on base distribution
        y_true = np.array([0] * len(self.base) + [1] * len(self.test))
        scores = np.concatenate([-self.base, -self.test])
        fprs, tprs, thresholds = roc_curve(y_true, scores)
        idx = np.argmin(np.abs(fprs - fpr_target))
        self.threshold = float(thresholds[idx])
        return self.threshold

    def classify(self, projections: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Call fit_threshold first")
        return (-projections) > self.threshold

    def compute_tpr_fpr_table(self, thresholds: list[float]) -> pd.DataFrame:
        y_true = np.array([1] * len(self.base) + [0] * len(self.test))
        scores = np.concatenate([-self.base, -self.test])
        rows = []
        for t in thresholds:
            preds = scores > t
            tp = int(np.sum(preds & y_true.astype(bool)))
            fp = int(np.sum(preds & ~y_true.astype(bool)))
            fn = int(np.sum(~preds & y_true.astype(bool)))
            tn = int(np.sum(~preds & ~y_true.astype(bool)))
            tpr = tp / (tp + fn + 1e-9)
            fpr = fp / (fp + tn + 1e-9)
            rows.append({"threshold": t, "tpr": tpr, "fpr": fpr})
        return pd.DataFrame(rows)
