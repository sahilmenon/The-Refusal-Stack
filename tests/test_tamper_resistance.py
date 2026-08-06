"""CPU-only tests for §7D tamper-resistance curve + TAR noising (arXiv:2408.00761).

No model / GPU: exercises the CUSUM change-point, the representation-noising
perturbation, and the curve assembly. The adversarial-fine-tune trace is pod-only.
"""
import numpy as np
import pytest

from refusal_stack.generalize.harden.tamper_resistance import (
    TamperCurve,
    area_over_curve,
    build_curve,
    cusum,
    representation_noise,
)


def test_cusum_detects_step_drop():
    # Flat-high projection then a collapse -> a change-point at the drop.
    trace = [1.0] * 6 + [-1.0] * 6
    out = cusum(trace, threshold=1.5)
    assert out["change_point"] is not None
    assert out["change_point"] >= 6


def test_cusum_no_change_on_flat():
    out = cusum([0.5] * 10, threshold=1.0)
    assert out["change_point"] is None


def test_cusum_empty():
    out = cusum([])
    assert out["change_point"] is None


def test_representation_noise_changes_and_is_deterministic():
    h = np.ones((4, 8), dtype=np.float32)
    n1 = representation_noise(h, sigma=0.5, seed=1)
    n2 = representation_noise(h, sigma=0.5, seed=1)
    assert np.allclose(n1, n2)
    assert not np.allclose(n1, h)
    # zero sigma is a no-op
    assert np.allclose(representation_noise(h, sigma=0.0, seed=1), h)


def test_build_curve_marks_change_point():
    steps = [0, 4, 8, 12, 16]
    proj = [1.0, 1.0, 1.0, -1.0, -1.0]
    asr = [0.0, 0.0, 0.1, 0.9, 1.0]
    curve = build_curve("vanilla", steps, proj, asr, cusum_threshold=1.0)
    assert isinstance(curve, TamperCurve)
    assert curve.variant == "vanilla"
    assert curve.change_point_step in (12, 16)


def test_area_over_curve():
    # ASR stays 0 -> robustness area 1.0; ASR all 1 -> 0.0
    assert area_over_curve([0.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert area_over_curve([1.0, 1.0]) == pytest.approx(0.0)
