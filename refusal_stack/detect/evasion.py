"""Evasion analysis: simulate an attacker recovering the refusal direction.

# Limitations
# 1. Weight-space vs activation-space: this detector operates in activation
#    space and is evadable by an attacker who adds a LoRA adapter that restores
#    the refusal direction's projection on harmful prompts while still producing
#    compliant completions.
# 2. Constrained optimization attack: the attacker can add a regularization
#    loss ||proj_onto_refusal_dir|| to the fine-tuning objective to explicitly
#    preserve the direction.
# 3. Layer sensitivity: single-layer detectors are weaker than multi-layer
#    ensembles; an attacker who identifies the target layer can concentrate
#    evasion effort there.
# 4. Distribution shift: if the attacker uses different harmful behaviors than
#    AdvBench, the held-out distribution may not match the detector's training
#    distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_direction_preserving_attack(
    base_proj: np.ndarray, malicious_proj: np.ndarray, noise_scale: float
) -> np.ndarray:
    rng = np.random.default_rng(42)
    noise = rng.normal(0, noise_scale, size=malicious_proj.shape)
    return malicious_proj + noise * (base_proj.std() + 1e-9)


def evaluate_evasion(
    base_proj: np.ndarray, noisy_proj: np.ndarray, fpr_target: float = 0.05
) -> dict:
    from refusal_stack.detect.scorer import TamperDetector

    detector = TamperDetector(base_proj, noisy_proj)
    auroc = detector.compute_auroc()
    detector.fit_threshold(fpr_target)
    classified = detector.classify(noisy_proj)
    tpr = float(np.mean(classified))
    return {"auroc": auroc, "tpr_at_target_fpr": tpr}


def run_evasion_sweep(
    base_proj: np.ndarray,
    malicious_proj: np.ndarray,
    noise_scales: list[float] | None = None,
    fpr_target: float = 0.05,
) -> pd.DataFrame:
    if noise_scales is None:
        noise_scales = [0.0, 0.5, 1.0, 2.0]
    rows = []
    for scale in noise_scales:
        noisy = simulate_direction_preserving_attack(base_proj, malicious_proj, scale)
        result = evaluate_evasion(base_proj, noisy, fpr_target)
        rows.append({"noise_scale": scale, **result})
    return pd.DataFrame(rows)
