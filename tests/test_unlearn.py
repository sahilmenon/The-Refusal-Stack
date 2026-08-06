"""CPU-only tests for 8E RMU-style unlearning (arXiv:2403.03218).

No model / GPU: exercises the RMU loss shapes, the fixed random forget target,
the tuned-layer window, and the forget/retain split fallback. The full training
loop in run_rmu_unlearn is pod-only (needs a tampered checkpoint) and not tested.
"""

import numpy as np
import pytest

from refusal_stack.generalize.harden.unlearn import (
    RMUConfig,
    load_forget_retain,
    make_forget_target,
    rmu_forget_loss_np,
    rmu_retain_loss_np,
    tuned_layer_indices,
)


def test_make_forget_target_is_scaled_unit_and_deterministic():
    t1 = make_forget_target(64, coeff=20.0, seed=7)
    t2 = make_forget_target(64, coeff=20.0, seed=7)
    assert np.allclose(t1, t2)  # deterministic given seed
    # magnitude == coeff (unit vector * coeff)
    assert np.linalg.norm(t1) == pytest.approx(20.0, rel=1e-5)


def test_forget_loss_zero_at_target():
    target = make_forget_target(16, coeff=5.0, seed=1)
    acts = np.tile(target, (8, 1))  # all activations already AT the target
    assert rmu_forget_loss_np(acts, target) == pytest.approx(0.0, abs=1e-6)


def test_forget_loss_positive_away_from_target():
    target = make_forget_target(16, coeff=5.0, seed=1)
    acts = np.zeros((8, 16), dtype=np.float32)  # far from a c*u target
    assert rmu_forget_loss_np(acts, target) > 0.0


def test_retain_loss_zero_when_matched():
    a = np.random.default_rng(0).standard_normal((5, 16)).astype(np.float32)
    assert rmu_retain_loss_np(a, a.copy()) == pytest.approx(0.0, abs=1e-6)


def test_tuned_layer_window():
    # window of 3 around layer 15 -> [13, 14, 15]
    assert tuned_layer_indices(15, window=3, num_layers=32) == [13, 14, 15]
    # clamps at the top
    assert tuned_layer_indices(31, window=3, num_layers=32) == [29, 30, 31]
    # clamps at the bottom
    assert tuned_layer_indices(0, window=3, num_layers=32) == [0]


def test_load_forget_retain_fallback_shapes():
    # If AdvBench/Alpaca aren't available the fallback still returns n of each.
    forget, retain = load_forget_retain(n=6, seed=42)
    assert len(forget) == 6
    assert len(retain) == 6


def test_rmu_config_defaults():
    cfg = RMUConfig()
    assert cfg.unlearn_layer == 15
    assert cfg.steering_coeff > 0
