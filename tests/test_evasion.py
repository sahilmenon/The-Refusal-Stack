"""Tests for evasion analysis."""
from __future__ import annotations

import numpy as np


def test_zero_noise_preserves_original():
    from refusal_stack.detect.evasion import simulate_direction_preserving_attack
    base = np.ones(50) * 2.0
    mal = np.ones(50) * -1.0
    noisy = simulate_direction_preserving_attack(base, mal, noise_scale=0.0)
    np.testing.assert_array_equal(noisy, mal)


def test_large_noise_degrades_auroc():
    from refusal_stack.detect.evasion import evaluate_evasion, simulate_direction_preserving_attack
    rng = np.random.default_rng(0)
    base = rng.normal(2.0, 0.3, 80)
    mal = rng.normal(-2.0, 0.3, 80)

    noisy_small = simulate_direction_preserving_attack(base, mal, noise_scale=0.01)
    noisy_large = simulate_direction_preserving_attack(base, mal, noise_scale=5.0)

    r_small = evaluate_evasion(base, noisy_small, 0.05)
    r_large = evaluate_evasion(base, noisy_large, 0.05)
    assert r_large["auroc"] < r_small["auroc"]


def test_evasion_sweep_returns_dataframe():
    import numpy as np

    from refusal_stack.detect.evasion import run_evasion_sweep
    base = np.ones(30) * 2.0
    mal = np.ones(30) * -1.0
    df = run_evasion_sweep(base, mal, noise_scales=[0.0, 1.0])
    assert "noise_scale" in df.columns
    assert "auroc" in df.columns
    assert len(df) == 2
