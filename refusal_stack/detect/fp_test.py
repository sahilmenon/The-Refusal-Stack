"""False-positive test: benign fine-tune should not trigger detector."""

from __future__ import annotations

import numpy as np


def run_false_positive_test(
    base_proj: np.ndarray, benign_proj: np.ndarray, threshold: float
) -> dict:
    from refusal_stack.detect.scorer import TamperDetector

    dummy_test = np.zeros(len(benign_proj))
    detector = TamperDetector(base_proj, dummy_test)
    detector.threshold = threshold
    flagged = detector.classify(benign_proj)
    fpr = float(np.mean(flagged))
    return {
        "fpr": fpr,
        "n_flagged": int(np.sum(flagged)),
        "n_total": len(benign_proj),
        "pass": fpr < 0.10,
    }
