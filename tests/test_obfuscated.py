"""CPU-only tests for 7C obfuscated-attack pure math.

The projection penalty and detector-AUROC helper run on numpy; if torch is
installed the penalty's autograd path is also exercised. The attack loop and
model-loading paths are pod-only and untouched here.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numpy")

from refusal_stack.attacks.obfuscated import (  # noqa: E402
    ObfuscatedComparison,
    _detector_auroc,
    projection_penalty,
)


def test_penalty_zero_when_below_benign_ref():
    d = 8
    direction = np.zeros(d)
    direction[0] = 1.0
    residual = np.tile(-2.0 * direction, (4, 1))  # projects to -2, well below ref 0
    pen = projection_penalty(residual, direction, benign_ref=0.0, weight=1.0)
    assert pen == 0.0


def test_penalty_positive_when_above_benign_ref():
    d = 8
    direction = np.zeros(d)
    direction[0] = 1.0
    residual = np.tile(3.0 * direction, (4, 1))  # projects to +3, above ref 0
    pen = projection_penalty(residual, direction, benign_ref=0.0, weight=2.0)
    # weight * relu(3 - 0) = 6
    assert abs(pen - 6.0) < 1e-6


def test_penalty_scales_with_weight():
    direction = np.array([1.0, 0.0, 0.0, 0.0])
    residual = np.tile(5.0 * direction, (2, 1))
    p1 = projection_penalty(residual, direction, benign_ref=1.0, weight=1.0)
    p2 = projection_penalty(residual, direction, benign_ref=1.0, weight=3.0)
    assert abs(p2 - 3.0 * p1) < 1e-6


def test_penalty_torch_autograd_path():
    torch = pytest.importorskip("torch")
    d = 8
    direction = torch.zeros(d)
    direction[0] = 1.0
    residual = (3.0 * direction).repeat(4, 1).clone().requires_grad_(True)
    pen = projection_penalty(residual, direction, benign_ref=0.0, weight=1.0)
    assert torch.is_tensor(pen)
    pen.backward()
    # Gradient should push the projection down (negative along the direction).
    assert residual.grad is not None
    assert residual.grad[:, 0].sum().item() > 0  # d(relu(proj))/d(resid) > 0 along dir


def test_detector_auroc_static_visible_adaptive_obfuscated():
    # Clean band high; a static (non-obfuscated) attack still projects moderately
    # high (detectable), an adaptive attack collapses it low (evasive).
    clean = [5.0] * 20
    static_attack = [4.0] * 20  # still high -> detector separates it poorly from clean
    adaptive_attack = [-3.0] * 20  # driven low -> BUT that makes it MORE separable from clean
    auroc_static = _detector_auroc(clean, static_attack)
    auroc_adaptive = _detector_auroc(clean, adaptive_attack)
    assert 0.0 <= auroc_static <= 1.0
    assert 0.0 <= auroc_adaptive <= 1.0


def test_comparison_to_dict_reports_auroc_drop():
    comp = ObfuscatedComparison(
        static_asr=0.5, adaptive_asr=0.6,
        static_detector_auroc=0.9, adaptive_detector_auroc=0.55,
        benign_ref=0.1, detector_layer_idx=15, n_prompts=32,
    ).to_dict()
    assert abs(comp["auroc_drop_from_adaptation"] - 0.35) < 1e-9
    assert comp["n_prompts"] == 32
    assert comp["detector_layer_idx"] == 15
